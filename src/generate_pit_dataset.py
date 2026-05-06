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

from data_gen import GlobalAnchorCovGen, GlobalFixedNets, generate_episode
from pit import load_tabicl, run_pit


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
    print(f"Output   : {out_dir}")
    print(f"Episodes : {n_episodes}  (batch_size={B}, resume={resume})")

    # ---- Load frozen TabICL for PIT ----
    print(f"Loading TabICL from checkpoint: {cfg.tabicl.ckpt!r} …")
    tabicl = load_tabicl(cfg.tabicl.ckpt, device)
    tabicl.eval()
    print("TabICL loaded.")

    # ---- Data range helpers ----
    p_lo, p_hi = int(cfg.data.p_range[0]),       int(cfg.data.p_range[1])
    d_lo, d_hi = int(cfg.data.d_range[0]),       int(cfg.data.d_range[1])
    pt_lo, pt_hi = int(cfg.data.n_train_range[0]), int(cfg.data.n_train_range[1])
    nt_lo, nt_hi = int(cfg.data.n_test_range[0]),  int(cfg.data.n_test_range[1])
    r_data     = int(cfg.data.r_data)
    mlp_hidden = int(cfg.data.mlp_hidden)
    fixed_cov  = bool(cfg.data.get("fixed_cov", False))
    fixed_cov_rho = float(cfg.data.get("fixed_cov_rho", 0.8))
    diag_alpha = float(cfg.data.get("diag_alpha", 0.0))

    # ---- Covariance generator (persistent across episodes for stability) ----
    fixed_nets: GlobalFixedNets | None = None
    anchor_gen: GlobalAnchorCovGen | None = None
    if not fixed_cov:
        cov_type = str(cfg.data.get("cov_type", "mlp"))
        if cov_type == "anchor":
            num_anchors = int(cfg.data.get("num_anchors", 8))
            anchor_temp = float(cfg.data.get("anchor_temp", 1.0))
            anchor_gen = GlobalAnchorCovGen(
                K=num_anchors, r=r_data, tau=anchor_temp, device=device
            )
            print(f"GlobalAnchorCovGen: K={num_anchors}, τ={anchor_temp}")
        else:
            fixed_nets = GlobalFixedNets(r=r_data, hidden=mlp_hidden, device=device)
            print("GlobalFixedNets: mlp covariance generator")

    # ---- Save dataset metadata for compatibility checks at training time ----
    meta = {
        "tabicl_ckpt":    cfg.tabicl.ckpt,
        "pit_batch_size": cfg.tabicl.pit_batch_size,
        "pit_eps":        cfg.tabicl.pit_eps,
        "batch_size":     B,
        "p_range":        [p_lo, p_hi],
        "d_range":        [d_lo, d_hi],
        "n_train_range":  [pt_lo, pt_hi],
        "n_test_range":   [nt_lo, nt_hi],
        "r_data":         r_data,
        "fixed_cov":      fixed_cov,
        "diag_alpha":     diag_alpha,
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset meta → {meta_path}")

    # ---- Generation loop ----
    skipped = 0
    last_Z_tr: torch.Tensor | None = None  # kept for disk-size estimate at end
    last_n_train = pt_lo
    last_n_test  = nt_lo

    for i in tqdm(range(n_episodes), desc="Generating PIT episodes"):
        fpath = os.path.join(out_dir, f"episode_{i:06d}.pt")

        if resume and os.path.exists(fpath):
            skipped += 1
            continue

        # ---- Sample episode hyperparameters ----
        p       = int(torch.randint(p_lo,  p_hi  + 1, ()).item())
        d       = int(torch.randint(d_lo,  d_hi  + 1, ()).item())
        n_train = int(torch.randint(pt_lo, pt_hi + 1, ()).item())
        n_test  = int(torch.randint(nt_lo, nt_hi + 1, ()).item())

        # ---- Generate raw episode (with oracle parameters) ----
        X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
            B,
            p,
            d,
            r_data,
            n_train,
            n_test,
            device,
            mlp_hidden=mlp_hidden,
            return_oracle=True,
            fixed_cov=fixed_cov,
            fixed_cov_rho=fixed_cov_rho,
            fixed_nets=fixed_nets,
            anchor_gen=anchor_gen,
            diag_alpha=diag_alpha,
        )

        # ---- Phase 1: run PIT per batch element ----
        # run_pit operates on one dataset at a time (unbatched across the B dim).
        Z_tr_list:     list[torch.Tensor] = []
        Z_te_list:     list[torch.Tensor] = []
        log_p_te_list: list[torch.Tensor] = []

        for b in range(B):
            Z_tr_b, Z_te_b, log_p_te_b = run_pit(
                tabicl,
                X_tr[b],        # (n_train, p)
                Y_tr[b],        # (n_train, d)
                X_te[b],        # (n_test,  p)
                Y_te[b],        # (n_test,  d)
                pit_batch_size=int(cfg.tabicl.pit_batch_size),
                eps=float(cfg.tabicl.pit_eps),
            )
            Z_tr_list.append(Z_tr_b)
            Z_te_list.append(Z_te_b)
            log_p_te_list.append(log_p_te_b)

        # Stack to (B, n_*, d)
        Z_tr      = torch.stack(Z_tr_list,     dim=0)   # (B, n_train, d)
        Z_te      = torch.stack(Z_te_list,     dim=0)   # (B, n_test,  d)
        log_p_te  = torch.stack(log_p_te_list, dim=0)   # (B, n_test,  d)

        # ---- Save episode ----
        torch.save(
            {
                # PIT-transformed targets — Phase 2 input
                "Z_train":    Z_tr.cpu(),       # (B, n_train, d)
                "Z_test":     Z_te.cpu(),        # (B, n_test,  d)
                "log_p_test": log_p_te.cpu(),    # (B, n_test, d)
                # Raw tabular data
                "X_train":    X_tr.cpu(),        # (B, n_train, p)
                "Y_train":    Y_tr.cpu(),        # (B, n_train, d)
                "X_test":     X_te.cpu(),        # (B, n_test,  p)
                "Y_test":     Y_te.cpu(),        # (B, n_test,  d)
                # Oracle parameters (for optional auxiliary supervision)
                "oracle_mu":  oracle["mu"].cpu(),  # (B, n_test, d)
                "oracle_D":   oracle["D"].cpu(),   # (B, n_test, d)
                "oracle_V":   oracle["V"].cpu(),   # (B, n_test, d, r_data)
                # Metadata
                "p":       p,
                "d":       d,
                "n_train": n_train,
                "n_test":  n_test,
            },
            fpath,
        )

        # Keep the last tensors for the disk-size report
        last_Z_tr    = Z_tr
        last_n_train = n_train
        last_n_test  = n_test

    # ---- Summary ----
    existing = len([f for f in os.listdir(out_dir) if f.startswith("episode_")])
    print(
        f"\nDone. {existing} episodes in {out_dir}  ({skipped} skipped / already existed)."
    )

    if last_Z_tr is not None:
        z_bytes  = last_Z_tr.element_size() * last_Z_tr.nelement()
        # Rough estimate: Z_train + Z_test ≈ z_bytes * (1 + n_test / n_train)
        ratio    = 1.0 + last_n_test / max(last_n_train, 1)
        ep_mb    = z_bytes * ratio / 1e6
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
