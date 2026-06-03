"""
generate_pit_dataset.py — Pre-compute Phase 1 PIT and cache episodes for fast training.

For each episode:
  1. Samples (p, d, n_train, n_test) from configured ranges.
  2. Calls generate_episode() to produce (X_train, Y_train, X_test, Y_test) + oracle.
  3. For each batch element b, calls run_pit() to get Z_train[b], Z_test[b], log_p_test[b].
  4. Saves everything to episode_NNNNNN.pt.

Training then loads Z_train directly, skipping the expensive PIT step.

Usage (from project root):
    python src/generate_pit_dataset.py dataset.n_episodes=5000
    python src/generate_pit_dataset.py dataset.output_dir=./data/pit_episodes dataset.resume=false
    python src/generate_pit_dataset.py dataset.n_episodes=200 training.batch_size=4
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — must happen before local imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from data_gen import GlobalAnchorCovGen, GlobalFixedNets, IsotropicModulatedKernel, KernelCovGen, generate_episode
from pit import load_tabicl, run_pit_batched

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---- Device ----
    device = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ---- Output directory ----
    out_dir = cfg.dataset.output_dir
    os.makedirs(out_dir, exist_ok=True)
    n_episodes = int(cfg.dataset.n_episodes)
    resume = bool(cfg.dataset.resume)
    B = int(cfg.dataset.batch_size)
    # episodes_per_step: how many episodes to produce per GPU step.
    # All K episodes in a step share (p, d, n_train, n_test) but get
    # independent random covariance structures (generate_episode randomises
    # per dataset within the batch). This amortises kernel-launch overhead
    # and keeps the GPU busier. Reduce if you hit OOM (effective batch = B*K).
    K = int(cfg.dataset.get("episodes_per_step", 1))
    # Sharding: override which slice of [0, n_episodes) this process handles.
    offset = int(cfg.dataset.get("offset", 0))
    limit = cfg.dataset.get("limit", None)
    episode_end = (
        min(offset + int(limit), n_episodes) if limit is not None else n_episodes
    )
    episode_ids = list(range(offset, episode_end))
    print(f"Output   : {out_dir}")
    print(
        f"Episodes : {len(episode_ids)} (ids {offset}–{episode_end - 1}, batch_size={B}×{K}, resume={resume})"
    )

    # ---- Load frozen TabICL for PIT ----
    print(f"Loading TabICL from checkpoint: {cfg.tabicl.ckpt!r} …")
    tabicl = load_tabicl(cfg.tabicl.ckpt, device)
    tabicl.eval()
    print("TabICL loaded.")

    # ---- Data range helpers ----
    p_lo, p_hi = int(cfg.data.p_range[0]), int(cfg.data.p_range[1])
    d_lo, d_hi = int(cfg.data.d_range[0]), int(cfg.data.d_range[1])
    pt_lo, pt_hi = int(cfg.data.n_train_range[0]), int(cfg.data.n_train_range[1])
    nt_lo, nt_hi = int(cfg.data.n_test_range[0]), int(cfg.data.n_test_range[1])
    r_data = int(cfg.data.r_data)
    mlp_hidden = int(cfg.data.mlp_hidden)
    diag_alpha_range = cfg.data.get("diag_alpha_range", [0.05, 2.0])
    diag_alpha_lo = float(diag_alpha_range[0])
    diag_alpha_hi = float(diag_alpha_range[1])
    if not (0.0 < diag_alpha_lo <= diag_alpha_hi):
        raise ValueError(
            "data.diag_alpha_range must be positive and ordered as [lo, hi]"
        )
    hyperplane_multimodal = bool(cfg.data.get("hyperplane_multimodal", False))
    hyperplane_multimodal_scale_lo = float(
        cfg.data.get("hyperplane_multimodal_scale_lo", 0.1)
    )
    hyperplane_multimodal_scale_hi = float(
        cfg.data.get("hyperplane_multimodal_scale_hi", 6.0)
    )

    # ---- Covariance generator (persistent across episodes for stability) ----
    fixed_nets: GlobalFixedNets | None = None
    anchor_gen: GlobalAnchorCovGen | None = None
    kernel_cov_gen: KernelCovGen | None = None
    if not hyperplane_multimodal:
        cov_type = str(cfg.data.get("cov_type", "mlp"))
        if cov_type == "anchor":
            num_anchors = int(cfg.data.get("num_anchors", 8))
            anchor_temp = float(cfg.data.get("anchor_temp", 1.0))
            anchor_gen = GlobalAnchorCovGen(
                K=num_anchors, r=r_data, tau=anchor_temp, device=device
            )
            print(f"GlobalAnchorCovGen: K={num_anchors}, τ={anchor_temp}")
        elif cov_type == "kernel":
            kernel_type = str(cfg.data.get("kernel_type", "random"))
            kernel_latent_dim = int(cfg.data.get("kernel_latent_dim", 1))
            kernel_nugget = float(cfg.data.get("kernel_nugget", 1e-4))
            kernel_cov_gen = KernelCovGen(
                kernel_type=kernel_type,
                latent_dim=kernel_latent_dim,
                mlp_hidden=mlp_hidden,
                nugget=kernel_nugget,
            )
            print(
                f"KernelCovGen: kernel={kernel_type}, latent_dim={kernel_latent_dim}, "
                f"nugget={kernel_nugget}"
            )
        elif cov_type == "iso_kernel":
            iso_kernel_type = str(cfg.data.get("iso_kernel_type", "rbf"))
            iso_kernel_nugget = float(cfg.data.get("iso_kernel_nugget", 1e-4))
            kernel_cov_gen = IsotropicModulatedKernel(
                kernel_type=iso_kernel_type,
                nugget=iso_kernel_nugget,
            )
            print(
                f"IsotropicModulatedKernel: kernel={iso_kernel_type}, "
                f"nugget={iso_kernel_nugget}"
            )
        else:
            fixed_nets = GlobalFixedNets(r=r_data, hidden=mlp_hidden, device=device)
            print("GlobalFixedNets: mlp covariance generator")

    # ---- Save dataset metadata for compatibility checks at training time ----
    meta = {
        "tabicl_ckpt": cfg.tabicl.ckpt,
        "pit_batch_size": cfg.tabicl.pit_batch_size,
        "pit_eps": cfg.tabicl.pit_eps,
        "batch_size": B,
        "p_range": [p_lo, p_hi],
        "d_range": [d_lo, d_hi],
        "n_train_range": [pt_lo, pt_hi],
        "n_test_range": [nt_lo, nt_hi],
        "r_data": r_data,
        "hyperplane_multimodal": hyperplane_multimodal,
        "diag_alpha_range": [diag_alpha_lo, diag_alpha_hi],
    }

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset meta → {meta_path}")

    # ---- Generation loop ----
    # Steps produce K episodes each: one forward pass fills B*K datasets, then
    # the result is sliced into K files of B datasets each.
    # Saves run in a background thread to overlap disk I/O with the next step's
    # GPU work.
    skipped = 0
    last_Z_tr: torch.Tensor | None = None  # kept for disk-size estimate at end
    last_n_train = pt_lo
    last_n_test = nt_lo

    # Group episode ids into steps of size K.
    steps = [episode_ids[s : s + K] for s in range(0, len(episode_ids), K)]

    save_executor = ThreadPoolExecutor(max_workers=4)
    pending_save: Future | None = None

    # ---- Profiling accumulators ----
    prof: dict[str, list[float]] = defaultdict(list)
    PROFILE_PRINT_EVERY = 1  # print rolling stats every N steps
    print("Profiling enabled.")

    def _save_step(payloads: list[tuple[dict, str]]) -> None:
        for data, path in payloads:
            torch.save(data, path)

    def _print_profile_summary(tag: str) -> None:
        total_avg = sum(
            sum(prof[k]) / max(len(prof[k]), 1)
            for k in ["t_datagen", "t_pit", "t_sync", "t_savewait", "t_slice"]
        )
        print(
            f"\n[PROFILE {tag}]  avg step = {total_avg * 1000:.1f} ms  (over {len(prof['t_datagen'])} steps)"
        )
        labels = [
            ("t_datagen", "generate_episode"),
            ("t_pit", "run_pit_batched  (total)"),
            ("pit_prep", "  PIT prep+fuse"),
            ("pit_test", "  PIT test pass"),
            ("pit_train", "  PIT train pass"),
            ("pit_probit", "  PIT probit"),
            ("t_sync", "cuda sync"),
            ("t_savewait", "save wait (I/O)"),
            ("t_slice", "slice+cpu copy"),
        ]
        for key, label in labels:
            vals = prof.get(key, [])
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            pct = 100.0 * avg / max(total_avg, 1e-9)
            print(f"  {label:<28s}: {avg * 1000:8.2f} ms  ({pct:5.1f}%)")
        # Extra: per-fold breakdown
        fold_avgs = prof.get("pit_fold_times", [])
        if fold_avgs:
            avg_fold = sum(fold_avgs) / len(fold_avgs)
            n_folds = prof.get("pit_n_folds", [])
            nf = int(sum(n_folds) / len(n_folds)) if n_folds else "?"
            print(
                f"  {'  per fwd pass (avg over folds)':<28s}: {avg_fold * 1000:8.2f} ms  ({nf} fwd passes/step)"
            )
        print()

    with tqdm(total=len(episode_ids), desc="Generating PIT episodes") as pbar:
        n_computed = 0
        for step_ids in steps:
            fpaths = [os.path.join(out_dir, f"episode_{i:06d}.pt") for i in step_ids]

            if resume and all(os.path.exists(fp) for fp in fpaths):
                skipped += len(step_ids)
                pbar.update(len(step_ids))
                continue

            # ---- Sample one shape shared across all K episodes in this step ----
            p = int(torch.randint(p_lo, p_hi + 1, ()).item())
            d = int(torch.randint(d_lo, d_hi + 1, ()).item())
            n_train = int(torch.randint(pt_lo, pt_hi + 1, ()).item())
            n_test = int(torch.randint(nt_lo, nt_hi + 1, ()).item())

            actual_K = len(step_ids)  # last step may be smaller
            diag_alpha_episode = torch.empty(actual_K).uniform_(
                diag_alpha_lo, diag_alpha_hi
            )
            diag_alpha_batch = (
                diag_alpha_episode.repeat_interleave(B)
                .to(device)
                .view(B * actual_K, 1, 1)
            )

            # ---- Generate B*K datasets in one shot ----
            t0 = time.perf_counter()
            X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
                B * actual_K,
                p,
                d,
                r_data,
                n_train,
                n_test,
                device,
                mlp_hidden=mlp_hidden,
                return_oracle=True,
                fixed_nets=fixed_nets,
                anchor_gen=anchor_gen,
                kernel_cov_gen=kernel_cov_gen,
                diag_alpha=diag_alpha_batch,
                hyperplane_multimodal=hyperplane_multimodal,
                hyperplane_multimodal_scale_lo=hyperplane_multimodal_scale_lo,
                hyperplane_multimodal_scale_hi=hyperplane_multimodal_scale_hi,
            )
            t1 = time.perf_counter()
            prof["t_datagen"].append(t1 - t0)

            # ---- One PIT forward pass for all B*K datasets ----
            _pit_timings: dict = {}
            Z_tr, Z_te, log_p_te = run_pit_batched(
                tabicl,
                X_tr,  # (B*K, n_train, p)
                Y_tr,  # (B*K, n_train, d)
                X_te,  # (B*K, n_test,  p)
                Y_te,  # (B*K, n_test,  d)
                pit_batch_size=int(cfg.tabicl.pit_batch_size),
                eps=float(cfg.tabicl.pit_eps),
                k_folds=cfg.dataset.get("k_folds", None),
                _timings=_pit_timings,
            )
            t2 = time.perf_counter()
            prof["t_pit"].append(t2 - t1)
            if _pit_timings:
                prof["pit_prep"].append(_pit_timings.get("pit_prep_s", 0.0))
                prof["pit_test"].append(_pit_timings.get("pit_test_pass_s", 0.0))
                prof["pit_train"].append(_pit_timings.get("pit_train_pass_s", 0.0))
                prof["pit_probit"].append(_pit_timings.get("pit_probit_s", 0.0))
                fold_ts = _pit_timings.get("pit_fold_times_s", [])
                if fold_ts:
                    prof["pit_fold_times"].append(sum(fold_ts) / len(fold_ts))
                    prof["pit_n_folds"].append(len(fold_ts))

            # Sync so .cpu() copies see completed GPU kernels.
            t_sync0 = time.perf_counter()
            if device != "cpu":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            prof["t_sync"].append(t3 - t_sync0)

            # Wait for the previous step's saves NOW — GPU just finished so
            # disk had the full T_gpu window to catch up. This gives true
            # overlap: T_step ≈ max(T_gpu, T_disk) instead of T_gpu + T_disk.
            t_sw0 = time.perf_counter()
            if pending_save is not None:
                pending_save.result()
            t4 = time.perf_counter()
            prof["t_savewait"].append(t4 - t_sw0)

            # ---- Slice into K payloads (CPU tensors) ----
            t_sl0 = time.perf_counter()
            payloads: list[tuple[dict, str]] = []
            for k, fpath in enumerate(fpaths):
                sl = slice(k * B, (k + 1) * B)
                payloads.append(
                    (
                        {
                            "Z_train": Z_tr[sl].cpu(),
                            "Z_test": Z_te[sl].cpu(),
                            "log_p_test": log_p_te[sl].cpu(),
                            "X_train": X_tr[sl].cpu(),
                            "Y_train": Y_tr[sl].cpu(),
                            "X_test": X_te[sl].cpu(),
                            "Y_test": Y_te[sl].cpu(),
                            "oracle_mu": oracle["mu"][sl].cpu(),
                            "oracle_D": oracle["D"][sl].cpu(),
                            "oracle_V": oracle["V"][sl].cpu(),
                            "p": p,
                            "d": d,
                            "n_train": n_train,
                            "n_test": n_test,
                            "diag_alpha": float(diag_alpha_episode[k].item()),
                        },
                        fpath,
                    )
                )
            t5 = time.perf_counter()
            prof["t_slice"].append(t5 - t_sl0)

            # Keep shape info for the disk-size report.
            last_Z_tr = payloads[0][0]["Z_train"]
            last_n_train = n_train
            last_n_test = n_test

            # Release GPU tensor references so the caching allocator can reuse
            # those blocks on the next step — no empty_cache(), no gc.collect().
            # The CUDA caching allocator is designed for exactly this pattern:
            # holding freed blocks avoids cudaMalloc on every step.
            del X_tr, Y_tr, X_te, Y_te, Z_tr, Z_te, log_p_te

            pending_save = save_executor.submit(_save_step, payloads)

            pbar.update(actual_K)
            n_computed += 1

            if n_computed % PROFILE_PRINT_EVERY == 0:
                _print_profile_summary(f"step {n_computed}")

    if pending_save is not None:
        pending_save.result()
    save_executor.shutdown(wait=False)

    if prof["t_datagen"]:
        _print_profile_summary("FINAL")

    # ---- Summary ----
    generated = len(episode_ids) - skipped
    existing = len([f for f in os.listdir(out_dir) if f.startswith("episode_")])
    print(
        f"\nDone. {generated} generated, {skipped} skipped → {existing} total in {out_dir}."
    )

    if last_Z_tr is not None:
        z_bytes = last_Z_tr.element_size() * last_Z_tr.nelement()
        # Rough estimate: Z_train + Z_test ≈ z_bytes * (1 + n_test / n_train)
        ratio = 1.0 + last_n_test / max(last_n_train, 1)
        ep_mb = z_bytes * ratio / 1e6
        total_gb = ep_mb * existing / 1e3
        print(
            f"Z_train shape per episode : (B={B}, n_train={last_n_train}, d={last_Z_tr.shape[-1]})"
        )
        print(
            f"Approx disk usage : ~{ep_mb:.2f} MB / episode  →  ~{total_gb:.2f} GB total"
        )
    else:
        print("No new episodes generated (all skipped).")


if __name__ == "__main__":
    main()
