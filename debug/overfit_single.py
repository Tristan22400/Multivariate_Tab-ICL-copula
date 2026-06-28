"""
overfit_single.py — Overfit a single dataset to test architecture learning capacity.

Trains copula_tabicl_v2 and/or tabicl-archi on ONE fixed dataset (one batch element
of one episode) using pure NLL loss.

KEY DESIGN: Z_test is RESAMPLED from the oracle distribution every step.
  - X_train, Z_train, X_test stay fixed (same context/query features every step).
  - Z_test ~ N(oracle_mu, oracle_Sigma) is freshly drawn each step.
  - This prevents the degenerate rank-1 collapse (aligning V with a fixed z).
  - The model must learn the conditional covariance structure to reduce NLL.

Interpretation of copula_gain at convergence:
  gain ≈ oracle  → model learned the correct structure ✓
  0 < gain << oracle → partial learning, architecture alive but limited
  gain ≈ 0        → stuck at independence ✗

Usage (from project root):
    conda run -n multivariate-icl python debug/overfit_single.py
    conda run -n multivariate-icl python debug/overfit_single.py --arch tabicl-archi
    conda run -n multivariate-icl python debug/overfit_single.py --arch copula_tabicl_v2
    conda run -n multivariate-icl python debug/overfit_single.py --arch both --steps 5000
    conda run -n multivariate-icl python debug/overfit_single.py --episode 3 --batch_idx 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loss import indep_normal_nll, woodbury_nll
from model import build_copula_tabicl, build_copula_tabicl_v2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR  = ROOT / "data" / "pit_simple_agg_cosine_p20_d8_nt512_ntest64_r8_B16_K3"
LR        = 3e-4
LR_MIN    = 1e-6
WD        = 1e-4
GRAD_CLIP = 1.0
LOG_EVERY = 50
BATCH_IDX = 0   # which dataset within the episode (0 = first of B=16)

# Model configs (mirrors conf/model/*.yaml)
CFG_V2 = OmegaConf.create({
    "model": {
        "name": "copula_tabicl_v2",
        "d_model": 32, "n_heads": 4,
        "n_layers_s1": 2, "n_layers_s2": 2, "n_layers_s3": 4,
        "n_inducing": 32, "n_cls": 4,
        "p_max": 20, "d_max": 8, "rank": 8,
        "d_ff": None, "dropout": 0.0,
    }
})

CFG_TABICL = OmegaConf.create({
    "model": {
        "name": "tabicl-archi",
        "d": 8, "k": 4,
        "embed_dim": 32,
        "col_num_blocks": 3, "col_nhead": 8, "col_num_inds": 128,
        "row_num_blocks": 3, "row_nhead": 8, "row_num_cls": 4,
        "icl_num_blocks": 12, "icl_nhead": 8,
        "dropout": 0.0, "pre_icl_aux": False,
    }
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_episode(ep_path: Path, device: torch.device, batch_idx: int):
    """Load one episode, return oracle params for the chosen dataset.

    Returns:
        X_train   : (1, n_train, p)  — fixed context features
        Z_train   : (1, n_train, d)  — fixed oracle Z context
        X_test    : (1, n_test,  p)  — fixed query features
        oracle_mu : (1, n_test,  d)  — per-instance conditional mean
        oracle_D  : (1, n_test,  d)  — per-instance diagonal variance
        oracle_V  : (1, n_test,  d, r) — per-instance low-rank factor
    """
    ep = torch.load(ep_path, map_location=device, weights_only=True)
    s = slice(batch_idx, batch_idx + 1)

    X_train     = ep["X_train"][s].to(device)
    Y_train     = ep["Y_train"][s].to(device)
    X_test      = ep["X_test"][s].to(device)
    oracle_mu_tr = ep["oracle_mu_train"][s].to(device)
    oracle_D_tr  = ep["oracle_D_train"][s].to(device)
    oracle_V_tr  = ep["oracle_V_train"][s].to(device)
    oracle_mu_te = ep["oracle_mu"][s].to(device)
    oracle_D_te  = ep["oracle_D"][s].to(device)
    oracle_V_te  = ep["oracle_V"][s].to(device)

    # Oracle Z_train: z = (y - mu) / sigma,  sigma = sqrt(D + ||V_row||^2)
    sigma_tr = (oracle_D_tr + oracle_V_tr.pow(2).sum(-1)).clamp(min=1e-8).sqrt()
    Z_train  = (Y_train - oracle_mu_tr) / sigma_tr

    return X_train, Z_train, X_test, oracle_mu_te, oracle_D_te, oracle_V_te


def resample_z_test(
    oracle_mu: torch.Tensor,   # (B, n_test, d)
    oracle_D:  torch.Tensor,   # (B, n_test, d)
    oracle_V:  torch.Tensor,   # (B, n_test, d, r)
) -> torch.Tensor:
    """Draw one fresh Z_test sample per test instance from oracle N(mu, Sigma).

    Sigma_i = diag(D_i) + V_i V_i^T  (low-rank Gaussian)
    z_i     = (y_i - mu_i) / sigma_i  where sigma_i = sqrt(diag(Sigma_i))

    This is the same normalisation as train.py use_oracle_z=True.
    Returns Z_test : (B, n_test, d)
    """
    B, n_test, d = oracle_mu.shape
    r = oracle_V.shape[-1]

    # Sample Y from the oracle: y = mu + L eps
    eps_d = torch.randn_like(oracle_mu)              # (B, n_test, d)
    eps_r = torch.randn(B, n_test, r,
                        device=oracle_mu.device,
                        dtype=oracle_mu.dtype)        # (B, n_test, r)

    Y_sample = (oracle_mu
                + oracle_D.sqrt() * eps_d
                + (eps_r.unsqueeze(-2) @ oracle_V.transpose(-2, -1)).squeeze(-2))
    # Y_sample: (B, n_test, d)

    # Normalise to copula Z: z = (y - mu) / sigma
    sigma = (oracle_D + oracle_V.pow(2).sum(-1)).clamp(min=1e-8).sqrt()
    return (Y_sample - oracle_mu) / sigma


def oracle_copula_nll_estimate(
    oracle_mu: torch.Tensor,
    oracle_D:  torch.Tensor,
    oracle_V:  torch.Tensor,
    n_samples: int = 500,
) -> float:
    """Monte-Carlo estimate of oracle copula NLL (average over fresh Z draws)."""
    nlls = []
    with torch.no_grad():
        Sigma_Y = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
        std_Y   = Sigma_Y.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        rho     = Sigma_Y / (std_Y.unsqueeze(-1) * std_Y.unsqueeze(-2))
        lam, U_eig = torch.linalg.eigh(rho)
        delta   = 1e-4
        D_rho   = torch.full_like(oracle_D, delta)
        V_rho   = U_eig * (lam - delta).clamp(min=0).sqrt().unsqueeze(-2)
        mu_zero = torch.zeros_like(oracle_mu)
        for _ in range(n_samples):
            z = resample_z_test(oracle_mu, oracle_D, oracle_V)
            wnll   = woodbury_nll(z, mu_zero, D_rho, V_rho).item()
            indep_z = indep_normal_nll(z).item()
            nlls.append(wnll - indep_z)
    return float(sum(nlls) / len(nlls))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_overfit(
    arch:       str,
    X_train:    torch.Tensor,   # (1, n_train, p)  — fixed
    Z_train:    torch.Tensor,   # (1, n_train, d)  — fixed
    X_test:     torch.Tensor,   # (1, n_test,  p)  — fixed
    oracle_mu:  torch.Tensor,   # (1, n_test,  d)
    oracle_D:   torch.Tensor,   # (1, n_test,  d)
    oracle_V:   torch.Tensor,   # (1, n_test,  d, r)
    n_steps:    int,
    device:     torch.device,
) -> dict[str, list]:

    if arch == "copula_tabicl_v2":
        model = build_copula_tabicl_v2(CFG_V2).to(device)
    elif arch == "tabicl-archi":
        model = build_copula_tabicl(CFG_TABICL).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    B, n_train, _ = X_train.shape
    n_test = X_test.shape[1]

    print(f"\n{'='*65}")
    print(f"  Architecture  : {arch}  ({n_params:,} params)")
    print(f"  Steps         : {n_steps}   LR={LR:.0e}  WD={WD:.0e}")
    print(f"  X_train       : {tuple(X_train.shape)}   X_test : {tuple(X_test.shape)}")
    print(f"  Z_test        : resampled fresh from oracle every step")

    print(f"  Estimating oracle copula NLL (500 MC draws)...")
    oracle_nll = oracle_copula_nll_estimate(oracle_mu, oracle_D, oracle_V)
    print(f"  Oracle copula NLL ≈ {oracle_nll:.4f}  (target for the model)")
    print(f"{'='*65}\n")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=LR_MIN)

    # Fixed forward layout: full X_train as support, X_test as query.
    # Z positions for test are zeros — the model never sees Z_test in the forward.
    # Only the LOSS target (Z_test_fresh) changes each step.
    X_fwd  = torch.cat([X_train, X_test], dim=1)   # (1, n_train+n_test, p)
    Z_fwd_support = Z_train                          # reused; zeros appended per step

    history: dict[str, list] = {
        "step": [], "copula_gain": [], "copula_nll": [], "lr": []
    }

    model.train()
    t0 = time.perf_counter()

    for step in range(n_steps):
        # Fresh Z_test drawn from oracle every step — prevents rank-1 collapse
        Z_test_fresh = resample_z_test(oracle_mu, oracle_D, oracle_V)
        Z_fwd = torch.cat([Z_fwd_support, torch.zeros_like(Z_test_fresh)], dim=1)

        optimizer.zero_grad()
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=n_train)
        loss = woodbury_nll(Z_test_fresh, mu_Z, d_Z, V_Z)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                indep_z    = indep_normal_nll(Z_test_fresh).item()
                copula_nll = loss.item() - indep_z
                copula_gain = -copula_nll
                lr_now = scheduler.get_last_lr()[0]

            elapsed = time.perf_counter() - t0
            print(
                f"[{arch[:16]:<16s}  step {step:>5d}]  "
                f"copula_nll={copula_nll:+.4f}  "
                f"gain={copula_gain:+.4f}  "
                f"oracle≈{oracle_nll:+.4f}  "
                f"lr={lr_now:.1e}  t={elapsed:.1f}s"
            )
            history["step"].append(step)
            history["copula_gain"].append(copula_gain)
            history["copula_nll"].append(copula_nll)
            history["lr"].append(lr_now)

    # -----------------------------------------------------------------------
    # Final evaluation: average over 200 fresh Z draws for stable metrics
    # -----------------------------------------------------------------------
    model.eval()
    eval_gains, eval_nlls = [], []
    R_pred_accum = None
    with torch.no_grad():
        for _ in range(200):
            Z_eval = resample_z_test(oracle_mu, oracle_D, oracle_V)
            Z_fwd_eval = torch.cat([Z_fwd_support, torch.zeros_like(Z_eval)], dim=1)
            mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd_eval, n_support=n_train)
            wnll = woodbury_nll(Z_eval, mu_Z, d_Z, V_Z).item()
            iz   = indep_normal_nll(Z_eval).item()
            eval_gains.append(iz - wnll)
            eval_nlls.append(wnll - iz)

        # Predicted correlation matrix (last forward, deterministic given X_fwd)
        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        std_pred   = Sigma_pred.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_pred     = Sigma_pred / (std_pred.unsqueeze(-1) * std_pred.unsqueeze(-2))

        # Oracle correlation matrix
        Sigma_ora = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
        std_ora   = Sigma_ora.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_ora     = Sigma_ora / (std_ora.unsqueeze(-1) * std_ora.unsqueeze(-2))

    final_gain = sum(eval_gains) / len(eval_gains)
    final_nll  = sum(eval_nlls)  / len(eval_nlls)

    d_dim = R_pred.shape[-1]
    ri, ci = torch.triu_indices(d_dim, d_dim, offset=1, device=device)
    off_diag_var = R_pred[..., ri, ci].var(dim=-2).mean().item()

    elapsed_total = time.perf_counter() - t0
    print(f"\n{'─'*65}")
    print(f"  {arch}  —  final (200-draw average, {elapsed_total:.1f}s total)")
    print(f"  copula_nll   : {final_nll:+.4f}")
    print(f"  copula_gain  : {final_gain:+.4f}  "
          f"  oracle≈{oracle_nll:+.4f}")
    frac = final_gain / abs(oracle_nll) if oracle_nll != 0 else float("nan")
    print(f"  oracle_frac  : {frac:.2%}  "
          f"({'✓ learning' if frac > 0.1 else '✗ stuck'})")
    print(f"  off-diag var : {off_diag_var:.4e}  "
          f"({'varied' if off_diag_var > 1e-4 else 'near-zero → V≈0'})")

    def _print_matrix(label, M):
        """Print a d×d matrix with aligned columns."""
        d = M.shape[0]
        print(f"\n  {label}  (instance 0)")
        print("        " + "".join(f"   [{j}] " for j in range(d)))
        for i in range(d):
            print(f"  [{i}]   " + "".join(f"{M[i,j]:+.3f} " for j in range(d)))

    R_ora_np  = R_ora[0, 0].cpu().float().numpy()
    R_pred_np = R_pred[0, 0].cpu().float().numpy()
    _print_matrix("Oracle  R", R_ora_np)
    _print_matrix("Pred    R", R_pred_np)
    _print_matrix("Delta   R  (pred − oracle)", R_pred_np - R_ora_np)

    mse_off = float(((R_pred_np - R_ora_np)[ri.cpu().numpy(), ci.cpu().numpy()]**2).mean())
    print(f"\n  MSE off-diag (instance 0) : {mse_off:.4f}")
    print(f"{'─'*65}\n")

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overfit a single dataset to test architecture learning capacity"
    )
    parser.add_argument("--arch", default="both",
                        choices=["copula_tabicl_v2", "tabicl-archi", "both"])
    parser.add_argument("--episode",   type=int, default=0,
                        help="Episode file index (default: 0)")
    parser.add_argument("--batch_idx", type=int, default=BATCH_IDX,
                        help="Dataset index within the episode (default: 0)")
    parser.add_argument("--steps",     type=int, default=5_000,
                        help="Training steps per architecture (default: 5000)")
    parser.add_argument("--device",    default="auto")
    args = parser.parse_args()

    device_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    device = torch.device(device_str)
    print(f"Device: {device}")

    ep_path = DATA_DIR / f"episode_{args.episode:06d}.pt"
    if not ep_path.exists():
        raise FileNotFoundError(f"Episode not found: {ep_path}")

    print(f"Loading episode {args.episode}, dataset {args.batch_idx} ...")
    X_train, Z_train, X_test, oracle_mu, oracle_D, oracle_V = load_episode(
        ep_path, device, args.batch_idx
    )

    archs = ["copula_tabicl_v2", "tabicl-archi"] if args.arch == "both" else [args.arch]

    all_histories: dict[str, dict] = {}
    for arch in archs:
        hist = run_overfit(
            arch, X_train, Z_train, X_test,
            oracle_mu, oracle_D, oracle_V,
            args.steps, device,
        )
        all_histories[arch] = hist

    if len(archs) > 1:
        print("\n" + "=" * 65)
        print("  COMPARISON  (final copula_gain, 200-draw avg — higher is better)")
        print("=" * 65)
        for arch, hist in all_histories.items():
            g = hist["copula_gain"][-1] if hist["copula_gain"] else float("nan")
            print(f"  {arch:<28s}  gain={g:+.4f}")
        print("=" * 65)


if __name__ == "__main__":
    main()
