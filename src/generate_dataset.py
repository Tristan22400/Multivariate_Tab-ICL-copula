"""
generate_dataset.py — Pre-compute and cache episode embeddings for fast training.

For each episode this script:
  1. Samples random (p, d, n_train, n_test) from the configured ranges.
  2. Calls generate_episode() to produce X_train, Y_train, X_test, Y_test.
  3. Runs the frozen TabICL base model via wrapper.get_embeddings() to produce
     E_all — the full-sequence embedding that is the input to JointReadoutLayer.
  4. Saves everything to a single .pt file per episode.

Training can then call wrapper.readout(E_all, P) directly, completely skipping
the expensive TabICL d-inference pass.

Usage (from project root):
    python src/generate_dataset.py dataset.n_episodes=5000
    python src/generate_dataset.py dataset.n_episodes=10000 dataset.output_dir=./data/big
    python src/generate_dataset.py dataset.resume=false     # overwrite existing files

Disk space estimate (current config: B=8, d=4, n_train=20, n_test=10):
    Each episode is dominated by E_all ~ (8, 30, 4*d_model_base) float32.
    With d_model_base≈768 this is roughly 5–10 MB per episode.
    5 000 episodes ≈ 25–50 GB.  Adjust n_episodes accordingly.
"""

from __future__ import annotations

import os
import sys

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tabicl_upstream", "src"))

from model import build_model

from data_gen import GlobalAnchorCovGen, GlobalFixedNets, generate_episode


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
    print(f"Output : {out_dir}")
    print(f"Episodes to generate: {n_episodes}  (resume={resume})")

    # ---- Model (frozen base only — readout is not used here) ----
    print("Loading TabICL base …")
    wrapper = build_model(cfg, device)
    wrapper.eval()

    # ---- Range helpers ----
    p_lo, p_hi = int(cfg.data.p_range[0]), int(cfg.data.p_range[1])
    d_lo, d_hi = int(cfg.data.d_range[0]), int(cfg.data.d_range[1])
    pt_lo, pt_hi = int(cfg.data.n_train_range[0]), int(cfg.data.n_train_range[1])
    nt_lo, nt_hi = int(cfg.data.n_test_range[0]), int(cfg.data.n_test_range[1])
    r_data = int(cfg.data.r_data)
    mlp_hidden = int(cfg.data.mlp_hidden)
    B = int(cfg.training.batch_size)
    fixed_cov = bool(cfg.data.get("fixed_cov", False))
    fixed_cov_rho = float(cfg.data.get("fixed_cov_rho", 0.8))
    diag_alpha = float(cfg.data.get("diag_alpha", 0.0))

    fixed_nets: GlobalFixedNets | None = None
    anchor_gen: GlobalAnchorCovGen | None = None
    if not fixed_cov:
        cov_type = cfg.data.get("cov_type", "mlp")
        if cov_type == "anchor":
            num_anchors = int(cfg.data.get("num_anchors", 8))
            anchor_temp = float(cfg.data.get("anchor_temp", 1.0))
            anchor_gen = GlobalAnchorCovGen(
                K=num_anchors, r=r_data, tau=anchor_temp, device=device
            )
            print(f"GlobalAnchorCovGen: K={num_anchors}, τ={anchor_temp}")
        else:
            fixed_nets = GlobalFixedNets(r=r_data, hidden=mlp_hidden, device=device)

    # ---- Save meta for config compatibility checks at training time ----
    # Training will load this and warn if the current config doesn't match.
    import json

    meta = {
        "model_ckpt": cfg.model.ckpt,
        "d_max": wrapper.d_max,
        "d_model_base": wrapper.d_model_base,
        "batch_size": B,
        "p_range": [p_lo, p_hi],
        "d_range": [d_lo, d_hi],
        "n_train_range": [pt_lo, pt_hi],
        "n_test_range": [nt_lo, nt_hi],
        "r_data": r_data,
        "fixed_cov": fixed_cov,
        "diag_alpha": diag_alpha,
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset meta → {meta_path}")

    # ---- Generation loop ----
    skipped = 0
    for i in tqdm(range(n_episodes), desc="Generating episodes"):
        fpath = os.path.join(out_dir, f"episode_{i:06d}.pt")

        if resume and os.path.exists(fpath):
            skipped += 1
            continue

        # Sample episode hyperparameters
        p = int(torch.randint(p_lo, p_hi + 1, ()).item())
        d = int(torch.randint(d_lo, d_hi + 1, ()).item())
        n_train = int(torch.randint(pt_lo, pt_hi + 1, ()).item())
        n_test = int(torch.randint(nt_lo, nt_hi + 1, ()).item())

        # Generate raw episode data + oracle parameters
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

        # Run TabICL to get JointReadoutLayer input — no gradients needed
        X_all = torch.cat([X_tr, X_te], dim=1)  # (B, P+N, p)
        E_all, P = wrapper.get_embeddings(X_all, Y_tr)

        torch.save(
            {
                # ---- JointReadoutLayer input (the expensive part to pre-compute) ----
                "E_all": E_all.cpu(),  # (B, T, d_max*d_model_base)
                "P": P,  # int: number of context rows
                "d": d,  # int: actual target dim (for output slicing)
                # ---- Raw data (for interpretability / architecture changes) ----
                "X_train": X_tr.cpu(),  # (B, P, p)
                "Y_train": Y_tr.cpu(),  # (B, P, d)
                "X_test": X_te.cpu(),  # (B, N, p)
                "Y_test": Y_te.cpu(),  # (B, N, d)
                # ---- Oracle parameters (for aux_lambda supervision) ----
                "oracle_mu": oracle["mu"].cpu(),  # (B, N, d)
                "oracle_D": oracle["D"].cpu(),  # (B, N, d)
                "oracle_V": oracle["V"].cpu(),  # (B, N, d, r_data)
                # ---- Metadata ----
                "p": p,
                "n_train": n_train,
                "n_test": n_test,
            },
            fpath,
        )

    existing = len([f for f in os.listdir(out_dir) if f.startswith("episode_")])
    print(
        f"\nDone. {existing} episodes in {out_dir}  ({skipped} skipped / already existed)."
    )
    print(
        f"E_all shape per episode: (B={B}, T={n_train + n_test}, d_max*d_model={wrapper.d_max * wrapper.d_model_base})"
    )
    size_mb = E_all.element_size() * E_all.nelement() / 1e6
    print(
        f"E_all size: ~{size_mb:.1f} MB per episode  →  ~{size_mb * existing / 1e3:.1f} GB total"
    )


if __name__ == "__main__":
    main()
