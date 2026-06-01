"""
tabicl_mse_only.py — CopulaTabICLv2 trained with MSE-ONLY (no NLL).

The previous run (overtrain_single.py) showed:
  • NLL keeps decreasing past oracle (overfit to observed Z values)
  • MSE plateaus at ~0.51  (model predicts same R for all queries)
  • Attention H_norm ≈ 1.000 throughout (Stage 3 fully collapsed)

The simple CrossAttnCopulaNet (overfit_correlation.py) reached:
  • MSE = 0.00001  after 500 steps  (r=1.000)

Two competing explanations for CopulaTabICLv2's failure:
  A. NLL gradient dominates → drives V_Z large → arbitrary off-diagonal structure
  B. Stage-3 oversmoothing is architectural → all queries get same context
     regardless of loss, so MSE gradients cancel (weak+strong cancel each other)

This script tests CopulaTabICLv2 with MSE-ONLY to distinguish A from B.
  • If MSE decreases and attention sharpens → A was the culprit (fix: drop NLL)
  • If attention stays at 1.000 and MSE plateaus → B (architectural, needs fixing)

Usage:
    conda run -n multivariate-icl python debug/tabicl_mse_only.py
"""

from __future__ import annotations

import math, os, sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import build_copula_tabicl_v2  # noqa: E402
from viz import plot_corr_grid            # noqa: E402

EPISODE_FILE = ROOT / "data" / "pit_hyperplane_debug" / "episode_000000.pt"
OUTPUT_DIR   = ROOT / "debug" / "overtrain_results" / "mse_only"
N_STEPS      = 10_000
LR           = 1e-3
GRAD_CLIP    = 1.0
LOG_EVERY    = 200
CHECK_EVERY  = 1_000


def _cov_to_corr(D, V):
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def _oracle_wby(oracle_D, oracle_V):
    S   = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    V_c = oracle_V / std.unsqueeze(-1)
    D_c = (1.0 - (V_c**2).sum(-1)).clamp(min=1e-6)
    return D_c, V_c


def get_s3_attn_entropy(model, X_fwd, Z_fwd, n_support):
    captured = {}
    orig_forwards = []
    for i, blk in enumerate(model.s3_blocks):
        orig_forwards.append(blk.forward)
        def _make_patched(idx, orig):
            def patched(x, ns, **kwargs):
                out, attn_w = orig(x, ns, return_attn_weights=True)
                captured[idx] = attn_w.detach().cpu()
                return out
            return patched
        blk.forward = _make_patched(i, orig_forwards[-1])

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(X_fwd, Z_fwd, n_support=n_support)
    finally:
        for i, blk in enumerate(model.s3_blocks):
            blk.forward = orig_forwards[i]
        if was_training:
            model.train()

    log_ns = math.log(max(n_support, 2))
    h_norms = {}
    for idx, attn_w in captured.items():
        w = attn_w[:, n_support:, :n_support].clamp(min=1e-10)
        H = -(w * w.log()).sum(dim=-1).mean().item()
        h_norms[idx] = H / log_ns
    return h_norms


def prediction_diversity(Sigma_pred):
    d = Sigma_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=Sigma_pred.device)
    off = Sigma_pred[0, :, ri, ci]
    return off.std(dim=0).mean().item()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ep       = torch.load(EPISODE_FILE, weights_only=True)
    b        = 0
    X_train  = ep["X_train"][[b]].float().to(device)
    Z_train  = ep["Z_train"][[b]].float().to(device)
    X_test   = ep["X_test"][[b]].float().to(device)
    Z_test   = ep["Z_test"][[b]].float().to(device)
    oracle_D = ep["oracle_D"][[b]].float().to(device)
    oracle_V = ep["oracle_V"][[b]].float().to(device)

    N, n_test, d = X_train.shape[1], X_test.shape[1], Z_train.shape[2]
    print(f"N={N}  n_test={n_test}  d={d}")

    R_ora  = _cov_to_corr(oracle_D, oracle_V)
    X_fwd  = torch.cat([X_train, X_test], dim=1)
    Z_fwd  = torch.cat([Z_train, Z_test], dim=1)
    ri, ci = torch.triu_indices(d, d, offset=1, device=device)

    v_norms    = oracle_V[0].norm(dim=(-2, -1))
    groups     = (v_norms > v_norms.median()).long().cpu().numpy()
    n_weak     = int((groups == 0).sum())
    n_strong   = int((groups == 1).sum())
    sort_order = np.argsort(groups)

    torch.manual_seed(42)
    cfg_model = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    model     = build_copula_tabicl_v2(SimpleNamespace(model=cfg_model)).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    n_params  = sum(p.numel() for p in model.parameters())
    n_s3      = len(model.s3_blocks)

    print(f"\nCopulaTabICLv2: {n_params:,} params  |  MSE-ONLY training (no NLL)")
    print(f"{'─'*65}")
    print(f"  {'step':>6}  {'MSE':>9}  {'diversity':>10}  " +
          "  ".join(f"blk{i}" for i in range(n_s3)))

    steps_log, mse_log, div_log = [], [], []
    attn_log = {i: [] for i in range(n_s3)}
    attn_steps = []

    model.train()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
        # Woodbury parametrization already ensures unit diagonal → Sigma IS correlation
        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        loss = F.mse_loss(Sigma_pred[:, :, ri, ci], R_ora[:, :, ri, ci])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse = F.mse_loss(Sigma_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item()
                div = prediction_diversity(Sigma_pred)
            steps_log.append(step); mse_log.append(mse); div_log.append(div)

            if step % CHECK_EVERY == 0:
                h_norms = get_s3_attn_entropy(model, X_fwd, Z_fwd, N)
                attn_steps.append(step)
                h_vals = [h_norms.get(i, float("nan")) for i in range(n_s3)]
                for i, h in enumerate(h_vals):
                    attn_log[i].append(h)
                h_str = "  ".join(f"{h:.3f}" for h in h_vals)
                model.train()
                print(f"  {step:>6d}  {mse:>9.5f}  {div:>10.6f}  {h_str}")
            else:
                print(f"  {step:>6d}  {mse:>9.5f}  {div:>10.6f}")

    # Final predictions
    model.eval()
    with torch.no_grad():
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
    Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
    final_mse  = F.mse_loss(Sigma_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item()
    final_div  = prediction_diversity(Sigma_pred)
    final_h    = get_s3_attn_entropy(model, X_fwd, Z_fwd, N)

    R_pred_np   = Sigma_pred[0].cpu().numpy()
    R_oracle_np = R_ora[0].cpu().numpy()
    ri_np, ci_np = np.triu_indices(d, k=1)
    x_all = R_oracle_np[:, ri_np, ci_np].ravel()
    y_all = R_pred_np[:,  ri_np, ci_np].ravel()
    r_val = float(np.corrcoef(x_all, y_all)[0, 1]) if x_all.std() > 1e-8 else float("nan")
    slope = float(np.polyfit(x_all, y_all, 1)[0]) if x_all.std() > 1e-8 else 1.0

    collapsed = [i for i, h in final_h.items() if h > 0.95]

    # Training curve plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps_log, mse_log, lw=1.5, color="steelblue", label="Off-diag MSE")
    axes[0].axhline(0.02, color="green", ls="--", lw=1, label="target 0.02")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("MSE")
    axes[0].set_title("CopulaTabICLv2 — MSE-only training")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, div_log, lw=1.5, color="purple")
    axes[1].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Prediction diversity (std)")
    axes[1].set_title("Cross-query diversity (0 = all same)")

    fig.suptitle(
        f"CopulaTabICLv2 MSE-only  —  final MSE={final_mse:.5f}  r={r_val:.3f}  "
        f"{'COLLAPSED' if collapsed else 'HEALTHY'} attn",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Attention entropy over time
    colors = plt.cm.viridis(np.linspace(0, 1, n_s3))
    fig_a, ax_a = plt.subplots(figsize=(10, 4))
    for i in range(n_s3):
        ax_a.plot(attn_steps, attn_log[i], color=colors[i],
                  lw=1.5, marker="o", ms=4, label=f"S3 blk {i}")
    ax_a.axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6, label="1.0 = collapsed")
    ax_a.axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6, label="0.0 = peaked")
    ax_a.set_ylim(-0.05, 1.10); ax_a.set_xlabel("Step")
    ax_a.set_ylabel("H_norm"); ax_a.set_title("Stage-3 attention entropy (MSE-only)")
    ax_a.legend(fontsize=8, ncol=2)
    fig_a.tight_layout()
    fig_a.savefig(OUTPUT_DIR / "attention_entropy.png", dpi=120, bbox_inches="tight")
    plt.close(fig_a)

    # Correlation grid
    R_pred_s   = Sigma_pred[0, sort_order].cpu()
    R_oracle_s = R_ora[0, sort_order].cpu()
    fig_g = plot_corr_grid(
        estimators={"CopulaTabICLv2": R_pred_s},
        oracle_R=R_oracle_s,
        n_instances=n_test,
        title=(f"CopulaTabICLv2 MSE-only — {n_weak} weak (top) / {n_strong} strong (bottom)\n"
               f"MSE={final_mse:.5f}  r={r_val:.3f}  div={final_div:.4f}"),
    )
    fig_g.savefig(OUTPUT_DIR / "all_predictions.png", dpi=100, bbox_inches="tight")
    plt.close(fig_g)

    print(f"\n{'='*65}")
    print(f"SUMMARY — CopulaTabICLv2 MSE-only")
    print(f"{'='*65}")
    print(f"  Final MSE:         {final_mse:.5f}")
    print(f"  Pearson r:         {r_val:.3f}")
    print(f"  OLS slope:         {slope:.3f}")
    print(f"  Prediction div:    {final_div:.4f}")
    print(f"  S3 H_norm (final):")
    for i, h in final_h.items():
        status = "COLLAPSED" if h > 0.95 else "OK"
        print(f"    blk {i}: {h:.4f}  [{status}]")
    if collapsed:
        print(f"\n→ DIAGNOSIS B: Oversmoothing is ARCHITECTURAL.")
        print(f"  Blocks {collapsed} are still collapsed even without NLL.")
        print(f"  Root cause: Q·K^T produces uniform scores regardless of loss.")
    else:
        print(f"\n→ DIAGNOSIS A: Oversmoothing was caused by NLL dominating.")
        print(f"  With MSE-only, attention sharpened → fix is to train without NLL.")
    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
