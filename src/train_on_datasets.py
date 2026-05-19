"""
train_on_datasets.py — Simple conditional Gaussian MLP vs CopulaTransformer
on one synthetic PIT episode.

Loads one pre-computed episode from dataset_dir (b=0 of the batch),
trains a simple MLP directly on (X_train, Y_train) in y-space, then compares
its test NLL against a pretrained CopulaTransformer evaluated in z-space
(converted to y-space via the PIT log-Jacobian).

Covariance predictions from both models are visualised with
plot_prediction_comparison:
  - MLP:  predicted vs empirical Y_test covariance
  - CT:   predicted vs oracle (z-space, ground truth from episode)

Usage:
    python src/train_on_datasets.py \\
        --config conf/config.yaml \\
        --ckpt   ./checkpoints/copula_transformer/step_0029999_final.pt \\
        [--episode_idx 0] [--steps 5000] [--wandb_project copula-mlp-vs-ct]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.covariance import OAS
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from loss import marginal_nll, woodbury_nll
from model import build_copula_tabicl_v2, build_copula_transformer
from train import _cov_to_woodbury_params
from viz import plot_prediction_comparison

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_ct_model(cfg) -> nn.Module:
    """Dispatch to the correct CopulaTransformer builder from saved config."""
    mcfg = cfg.model
    if hasattr(mcfg, "n_layers_s1"):
        return build_copula_tabicl_v2(cfg)
    return build_copula_transformer(cfg)


# ---------------------------------------------------------------------------
# Simple conditional Gaussian MLP
# ---------------------------------------------------------------------------


class ConditionalGaussianMLP(nn.Module):
    """Maps x_i -> (mu_y, D_y, V_y) — conditional Gaussian in y-space.

    Architecture: Linear(p, hidden) -> SiLU -> [Linear(h,h) -> SiLU] x (n_layers-1)
    then three heads: mu (d), log_d (d, softplus+eps), V_flat (d*r -> (d,r)).
    """

    def __init__(self, p: int, d: int, r: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(p, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden, d)
        self.fc_d = nn.Linear(hidden, d)
        self.fc_V = nn.Linear(hidden, d * r)
        self._d = d
        self._r = r

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (N, p) -> mu (N,d), d_out (N,d), V (N,d,r)"""
        h = self.backbone(x)
        mu = self.fc_mu(h)
        d_out = F.softplus(self.fc_d(h)) + 1e-4
        V = self.fc_V(h).view(x.shape[0], self._d, self._r)
        return mu, d_out, V


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train simple MLP baseline and compare with pretrained CopulaTransformer"
    )
    parser.add_argument(
        "--config",
        default="conf/config.yaml",
        help="Hydra config YAML (reads dataset_dir and model cfg).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to pretrained CopulaTransformer checkpoint (.pt).",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Index of the episode to use from dataset_dir.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--wandb_project", default="copula-mlp-vs-ct")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ---- Config & episode path ----
    cfg = OmegaConf.load(args.config)
    dataset_dir = cfg.training.dataset_dir
    ep_path = os.path.join(dataset_dir, f"episode_{args.episode_idx:06d}.pt")
    print(f"Loading episode: {ep_path}")

    # ---- Load episode, take batch element 0 ----
    ep = torch.load(ep_path, map_location=device, weights_only=True)
    X_train = ep["X_train"][0].to(device)  # (n_train, p)
    Y_train = ep["Y_train"][0].to(device)  # (n_train, d)
    X_test = ep["X_test"][0].to(device)  # (n_test, p)
    Y_test = ep["Y_test"][0].to(device)  # (n_test, d)
    Z_train = ep["Z_train"][0].to(device)  # (n_train, d)
    Z_test = ep["Z_test"][0].to(device)  # (n_test, d)
    log_p_test = ep["log_p_test"][0].to(device)  # (n_test, d)
    oracle_mu = ep["oracle_mu"][0].to(device)  # (n_test, d)
    oracle_D = ep["oracle_D"][0].to(device)  # (n_test, d)
    oracle_V = ep["oracle_V"][0].to(device)  # (n_test, d, r_oracle)

    n_train, p = X_train.shape
    n_test, d = Y_test.shape
    print(
        f"Episode {args.episode_idx}: n_train={n_train}, n_test={n_test}, p={p}, d={d}"
    )

    # ---- W&B ----
    wandb_run = None
    try:
        import wandb

        run_name = args.wandb_name or (
            f"ep{args.episode_idx}_s{args.steps}_lr{args.lr:.0e}_h{args.hidden}x{args.n_layers}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        print(f"W&B run: {wandb_run.url}")
    except Exception as e:
        print(f"W&B unavailable ({e}), logging to console only.")

    # ====================================================================
    # Phase 1 — Train simple MLP on (X_train, Y_train) in y-space
    # ====================================================================
    print(f"\n--- Training Simple MLP ({args.steps} steps) ---")
    mlp = ConditionalGaussianMLP(
        p=p, d=d, r=args.rank, hidden=args.hidden, n_layers=args.n_layers
    ).to(device)
    n_params = sum(param.numel() for param in mlp.parameters())
    print(f"MLP parameters: {n_params:,}")

    optimizer = Adam(mlp.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(args.steps, 1), eta_min=args.lr_min
    )

    mlp.train()
    for step in range(args.steps):
        idx = torch.randperm(n_train, device=device)[: args.batch_size]
        mu, d_out, V_out = mlp(X_train[idx])
        loss = woodbury_nll(
            Y_train[idx].unsqueeze(0),
            mu.unsqueeze(0),
            d_out.unsqueeze(0),
            V_out.unsqueeze(0),
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mlp.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                mnll_val = marginal_nll(
                    Y_train[idx].unsqueeze(0),
                    mu.unsqueeze(0),
                    d_out.unsqueeze(0),
                ).item()
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[mlp  step {step:>5d}]  woodbury_nll={loss.item():.4f}"
                f"  marginal_nll={mnll_val:.4f}  lr={lr_now:.2e}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "mlp/train/woodbury_nll": loss.item(),
                        "mlp/train/marginal_nll": mnll_val,
                        "mlp/train/lr": lr_now,
                    },
                    step=step,
                )

    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        mlp_path = os.path.join(args.ckpt_dir, f"mlp_ep{args.episode_idx}.pt")
        torch.save({"model_state": mlp.state_dict(), "args": vars(args)}, mlp_path)
        print(f"MLP checkpoint → {mlp_path}")

    # ====================================================================
    # Phase 2 — MLP evaluation on Y_test
    # ====================================================================
    print("\n--- Evaluating MLP on test set (y-space) ---")
    mlp.eval()
    with torch.no_grad():
        mu_mlp, d_mlp, V_mlp = mlp(X_test)  # (N_te, d), (N_te, d), (N_te, d, r)

    nll_mlp = woodbury_nll(
        Y_test.unsqueeze(0), mu_mlp.unsqueeze(0), d_mlp.unsqueeze(0), V_mlp.unsqueeze(0)
    ).item()
    mnll_mlp = marginal_nll(
        Y_test.unsqueeze(0), mu_mlp.unsqueeze(0), d_mlp.unsqueeze(0)
    ).item()
    print(f"MLP  — woodbury_nll (y): {nll_mlp:.4f}   marginal_nll (y): {mnll_mlp:.4f}")

    # ====================================================================
    # Phase 3 — CopulaTransformer evaluation (pretrained, z-space -> y-space)
    # ====================================================================
    print(f"\n--- Loading CopulaTransformer from {args.ckpt} ---")
    ct_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ct_cfg = ct_ckpt.get("cfg", cfg)
    if isinstance(ct_cfg, dict):
        ct_cfg = OmegaConf.create(ct_cfg)

    ct_model = _build_ct_model(ct_cfg).to(device)
    ct_model.load_state_dict(ct_ckpt["model_state"])
    ct_model.eval()
    n_ct_params = sum(param.numel() for param in ct_model.parameters())
    print(f"CopulaTransformer parameters: {n_ct_params:,}")

    print("Running CopulaTransformer inference...")
    with torch.no_grad():
        # Concatenate context + query; zeros mask the query Z values.
        X_all = torch.cat([X_train, X_test], dim=0).unsqueeze(0)  # (1, N_tr+N_te, p)
        Z_all = torch.cat([Z_train, torch.zeros_like(Z_test)], dim=0).unsqueeze(0)
        mu_ct, d_ct, V_ct = ct_model(X_all, Z_all, n_support=n_train)  # (1, N_te, ...)
        mu_ct = mu_ct.squeeze(0)  # (N_te, d)
        d_ct = d_ct.squeeze(0)  # (N_te, d)
        V_ct = V_ct.squeeze(0)  # (N_te, d, r_ct)

    nll_ct_z = woodbury_nll(
        Z_test.unsqueeze(0), mu_ct.unsqueeze(0), d_ct.unsqueeze(0), V_ct.unsqueeze(0)
    ).item()
    mnll_ct_z = marginal_nll(
        Z_test.unsqueeze(0), mu_ct.unsqueeze(0), d_ct.unsqueeze(0)
    ).item()
    # NLL_y = NLL_z - E[sum_j log|dz_j/dy_j|]  (change-of-variables)
    nll_ct_y = nll_ct_z - log_p_test.sum(dim=-1).mean().item()

    nll_oracle_z = woodbury_nll(
        Z_test.unsqueeze(0),
        oracle_mu.unsqueeze(0),
        oracle_D.unsqueeze(0),
        oracle_V.unsqueeze(0),
    ).item()

    print(
        f"CT   — woodbury_nll (z): {nll_ct_z:.4f}   marginal_nll (z): {mnll_ct_z:.4f}   nll (y): {nll_ct_y:.4f}"
    )
    print(f"Oracle NLL (z):           {nll_oracle_z:.4f}")

    # ====================================================================
    # Phase 4 — Log final metrics
    # ====================================================================
    final_step = args.steps
    eval_metrics = {
        "eval/mlp_nll_y": nll_mlp,
        "eval/mlp_mnll_y": mnll_mlp,
        "eval/ct_nll_z": nll_ct_z,
        "eval/ct_mnll_z": mnll_ct_z,
        "eval/ct_nll_y": nll_ct_y,
        "eval/oracle_nll_z": nll_oracle_z,
    }
    print("\n--- Summary ---")
    col_w = max(len(k) for k in eval_metrics) + 2
    for k, v in eval_metrics.items():
        print(f"  {k:<{col_w}}: {v:.4f}")
    if wandb_run is not None:
        wandb_run.log(eval_metrics, step=final_step)

    # ====================================================================
    # Phase 5 — Covariance comparison plots
    # ====================================================================
    import matplotlib.pyplot as plt

    print("\n--- Generating covariance plots ---")
    plot_figs = []

    # -- Plot 1: MLP (y-space) predicted vs empirical Y_test covariance --
    mu_emp = Y_test.mean(0)  # (d,)
    Sigma_emp = torch.cov(Y_test.T)  # (d, d)
    mu_ref, D_ref, V_ref = _cov_to_woodbury_params(Sigma_emp, mu_emp, n_test, device)
    oas_y = OAS().fit(Y_test.cpu().numpy())

    fig_mlp = plot_prediction_comparison(
        mu_pred=mu_mlp.unsqueeze(0),
        D_pred=d_mlp.unsqueeze(0),
        V_pred=V_mlp.unsqueeze(0),
        mu_true=mu_ref.unsqueeze(0),
        D_true=D_ref.unsqueeze(0),
        V_true=V_ref.unsqueeze(0),
        sigma_oas=oas_y.covariance_,
        dataset_label="Simple MLP — predicted vs empirical Y covariance",
    )
    plot_figs.append(fig_mlp)

    # -- Plot 2: CopulaTransformer (z-space) predicted vs oracle --
    oas_z = OAS().fit(Z_train.cpu().numpy())

    fig_ct = plot_prediction_comparison(
        mu_pred=mu_ct.unsqueeze(0),
        D_pred=d_ct.unsqueeze(0),
        V_pred=V_ct.unsqueeze(0),
        mu_true=oracle_mu.unsqueeze(0),
        D_true=oracle_D.unsqueeze(0),
        V_true=oracle_V.unsqueeze(0),
        sigma_oas=oas_z.covariance_,
        dataset_label="CopulaTransformer — predicted vs oracle (Z-space)",
    )
    plot_figs.append(fig_ct)

    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {"eval/cov_plot": [wandb.Image(f) for f in plot_figs]},
            step=final_step,
        )
        print("Covariance plots logged to W&B.")

    for fig in plot_figs:
        plt.close(fig)

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
