"""
overtrain_single.py — Overtraining diagnostic on a single hyperplane episode.

Tests whether CopulaTabICLv2 can memorize one dataset (model expressiveness test).
  - If loss decreases to near oracle level → architecture can learn the mapping.
  - If loss plateaus far above oracle → oversmoothing / representation collapse persists.

Attention health check: every CHECK_ATTN_EVERY steps we collect per-S3-block
normalised attention entropy H_norm = H / log(n_support):
  H_norm ≈ 1.0  → uniform attention (over-smoothed, model ignores features)
  H_norm ≪ 1.0  → peaked attention (model uses X to select relevant support rows)

The hyperplane episode has two groups of test instances with distinct correlation
structures (weak vs strong), separated by a random hyperplane in feature space.
A healthy model should predict different R matrices for the two groups.

Usage:
    conda run -n multivariate-icl python debug/overtrain_single.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loss import woodbury_nll  # noqa: E402
from model import build_copula_tabicl_v2  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from viz import plot_corr_grid  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPISODE_FILE   = ROOT / "data" / "pit_hyperplane_debug" / "episode_000000.pt"
OUTPUT_DIR     = ROOT / "debug" / "overtrain_results"
N_STEPS        = 50_000
LR             = 1e-3
GRAD_CLIP      = 1.0
LOG_EVERY      = 500     # print + record loss every N steps
CHECK_ATTN_EVERY = 5_000  # attention entropy snapshot every N steps
BATCH_ELEM     = 0       # which dataset in the episode batch to overtrain on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cov_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Woodbury (D, V) → correlation matrix R."""
    Sigma = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))


def _oracle_woodbury_corr(
    oracle_D: torch.Tensor, oracle_V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize oracle (D, V) into correlation-space Woodbury form (D_c, V_c).

    Computes the correlation matrix R = normalize(diag(D) + VV^T) then expresses
    it as diag(D_c) + V_c V_c^T where D_c = 1 - ||V_c||^2 (unit diagonal).
    """
    Sigma = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    V_c = oracle_V / std.unsqueeze(-1)
    D_c = (1.0 - (V_c ** 2).sum(-1)).clamp(min=1e-6)
    return D_c, V_c


def get_s3_attn_entropy(
    model: torch.nn.Module,
    X_fwd: torch.Tensor,
    Z_fwd: torch.Tensor,
    n_support: int,
) -> dict[int, float]:
    """Run one no-grad forward pass and return per-S3-block normalised entropy.

    Temporarily patches ICLBlock.forward to call with return_attn_weights=True
    so no architectural changes are needed.

    H_norm = H / log(n_support) where H is entropy of query→support weights:
      1.0 → fully uniform (over-smoothed)
      0.0 → one-hot (perfectly selective)
    """
    captured: dict[int, torch.Tensor] = {}
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

    n_sup = n_support
    log_ns = math.log(max(n_sup, 2))
    h_norms: dict[int, float] = {}

    for idx, attn_w in captured.items():
        # attn_w: (B, N, N) — query→support slice
        w = attn_w[:, n_support:, :n_support].clamp(min=1e-10)  # (B, n_q, n_sup)
        H = -(w * w.log()).sum(dim=-1).mean().item()
        h_norms[idx] = H / log_ns

    return h_norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1.  Load a single dataset from the episode
    # ------------------------------------------------------------------
    print(f"Loading episode: {EPISODE_FILE}")
    episode = torch.load(EPISODE_FILE, weights_only=True)

    b = BATCH_ELEM
    X_train  = episode["X_train"][[b]].float().to(device)    # (1, N, p)
    Z_train  = episode["Z_train"][[b]].float().to(device)    # (1, N, d)
    X_test   = episode["X_test"][[b]].float().to(device)     # (1, n_test, p)
    Z_test   = episode["Z_test"][[b]].float().to(device)     # (1, n_test, d)
    oracle_D = episode["oracle_D"][[b]].float().to(device)   # (1, n_test, d)
    oracle_V = episode["oracle_V"][[b]].float().to(device)   # (1, n_test, d, r_data)

    N      = X_train.shape[1]
    n_test = X_test.shape[1]
    d      = Z_train.shape[2]
    print(f"Dataset: N={N} support, n_test={n_test} test, d={d} dims, "
          f"r_data={oracle_V.shape[-1]}")

    # ------------------------------------------------------------------
    # 2.  Oracle and baseline NLL
    # ------------------------------------------------------------------
    with torch.no_grad():
        R_ora = _cov_to_corr(oracle_D, oracle_V)          # (1, n_test, d, d)
        D_ora_c, V_ora_c = _oracle_woodbury_corr(oracle_D, oracle_V)

        oracle_nll = woodbury_nll(
            Z_test, torch.zeros_like(Z_test), D_ora_c, V_ora_c
        ).item()

        V_zero = torch.zeros(1, n_test, d, 1, device=device)
        D_ones = torch.ones(1, n_test, d, device=device)
        indep_nll = woodbury_nll(
            Z_test, torch.zeros_like(Z_test), D_ones, V_zero
        ).item()

    print(f"Oracle NLL   (floor):    {oracle_nll:.4f}")
    print(f"Indep N(0,I) (baseline): {indep_nll:.4f}")

    # ------------------------------------------------------------------
    # 3.  Identify hyperplane groups (weak vs strong correlations)
    # ------------------------------------------------------------------
    v_norms  = oracle_V[0].norm(dim=(-2, -1))          # (n_test,)
    groups   = (v_norms > v_norms.median()).long().cpu().numpy()
    n_weak   = int((groups == 0).sum())
    n_strong = int((groups == 1).sum())
    print(f"Hyperplane groups: {n_weak} weak, {n_strong} strong")

    # ------------------------------------------------------------------
    # 4.  Build a fresh CopulaTabICLv2 (no checkpoint)
    # ------------------------------------------------------------------
    cfg_model = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    model     = build_copula_tabicl_v2(SimpleNamespace(model=cfg_model)).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # ------------------------------------------------------------------
    # 5.  Overtraining loop
    # ------------------------------------------------------------------
    X_fwd = torch.cat([X_train, X_test], dim=1)   # (1, N+n_test, p)
    Z_fwd = torch.cat([Z_train, Z_test], dim=1)   # (1, N+n_test, d)

    ri, ci = torch.triu_indices(d, d, offset=1, device=device)
    n_s3   = len(model.s3_blocks)

    steps_log: list[int]           = []
    nll_log:   list[float]         = []
    mse_log:   list[float]         = []
    attn_log:  dict[int, list[float]] = {i: [] for i in range(n_s3)}
    attn_steps: list[int]          = []

    print(f"\nOvertraining for {N_STEPS} steps  LR={LR}  grad_clip={GRAD_CLIP}")
    print(f"Loss logged every {LOG_EVERY} steps | "
          f"Attention checked every {CHECK_ATTN_EVERY} steps\n")

    model.train()
    for step in range(N_STEPS):
        optimizer.zero_grad()

        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)

        loss_nll = woodbury_nll(Z_test, mu_Z, d_Z, V_Z)

        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        loss_mse   = F.mse_loss(Sigma_pred[..., ri, ci], R_ora[..., ri, ci])

        loss = loss_nll + loss_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # ---- Loss logging ----
        if step % LOG_EVERY == 0:
            steps_log.append(step)
            nll_log.append(loss_nll.item())
            mse_log.append(loss_mse.item())
            print(
                f"  step {step:6d}  NLL={loss_nll.item():.4f}  "
                f"oracle={oracle_nll:.4f}  MSE={loss_mse.item():.5f}"
            )

        # ---- Attention entropy snapshot ----
        if step % CHECK_ATTN_EVERY == 0:
            h_norms = get_s3_attn_entropy(model, X_fwd, Z_fwd, N)
            attn_steps.append(step)
            flags = []
            for i in range(n_s3):
                h = h_norms.get(i, float("nan"))
                attn_log[i].append(h)
                flags.append(f"blk{i}={h:.3f}")
            model.train()  # restore training mode after eval inside helper
            print(f"  [attn]  H_norm per S3 block:  {' | '.join(flags)}"
                  f"  (1.0=uniform/collapsed, 0.0=peaked/healthy)")

    print(f"\nFinal: NLL={nll_log[-1]:.4f}  oracle={oracle_nll:.4f}  "
          f"indep={indep_nll:.4f}")

    # ------------------------------------------------------------------
    # 6.  Loss-curve plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(steps_log, nll_log, color="steelblue", lw=1.5, label="Model NLL")
    axes[0].axhline(oracle_nll, color="green",  ls="--", lw=1.2,
                    label=f"Oracle NLL ({oracle_nll:.3f})")
    axes[0].axhline(indep_nll,  color="gray",   ls=":",  lw=1.2,
                    label=f"Indep N(0,I) ({indep_nll:.3f})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Woodbury NLL")
    axes[0].set_title("NLL during overtraining")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, mse_log, color="tomato", lw=1.5, label="Off-diag MSE")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Off-diagonal MSE during overtraining")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Overtraining diagnostic — single hyperplane episode (batch elem {b})",
        fontsize=12,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "loss_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # 7.  Attention entropy plot (per S3 block over training)
    # ------------------------------------------------------------------
    colors_attn = plt.cm.viridis(np.linspace(0, 1, n_s3))
    fig_attn, ax_attn = plt.subplots(figsize=(10, 4))
    for i in range(n_s3):
        ax_attn.plot(attn_steps, attn_log[i], color=colors_attn[i],
                     lw=1.5, marker="o", ms=4, label=f"S3 blk {i}")
    ax_attn.axhline(1.0, color="red",  ls="--", lw=0.8, alpha=0.6,
                    label="H_norm=1.0 (fully uniform / collapsed)")
    ax_attn.axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6,
                    label="H_norm=0.0 (perfectly peaked)")
    ax_attn.set_xlabel("Step")
    ax_attn.set_ylabel("H_norm = H / log(n_support)")
    ax_attn.set_ylim(-0.05, 1.10)
    ax_attn.set_title(
        "Stage-3 attention entropy over training\n"
        "1.0 = uniform (over-smoothed) | 0.0 = peaked (healthy)"
    )
    ax_attn.legend(fontsize=8, ncol=2)
    fig_attn.tight_layout()
    path = OUTPUT_DIR / "attention_entropy.png"
    fig_attn.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig_attn)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # 8.  Final predictions on all test instances
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)

    Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
    R_pred     = Sigma_pred[0]   # (n_test, d, d)
    R_oracle   = R_ora[0]        # (n_test, d, d)

    # Sort: weak first, strong last so the two groups form visual blocks
    sort_order      = np.argsort(groups)
    R_pred_sorted   = R_pred[sort_order].cpu()
    R_oracle_sorted = R_oracle[sort_order].cpu()
    groups_sorted   = groups[sort_order]

    # ------------------------------------------------------------------
    # 9.  Correlation-matrix grid (all n_test instances)
    # ------------------------------------------------------------------
    fig_grid = plot_corr_grid(
        estimators={"Model": R_pred_sorted},
        oracle_R=R_oracle_sorted,
        n_instances=n_test,
        title=(
            f"All {n_test} test instances — "
            f"{n_weak} weak (top) / {n_strong} strong (bottom)\n"
            f"after {N_STEPS:,} overtraining steps on a single dataset"
        ),
    )
    path = OUTPUT_DIR / "all_predictions_final.png"
    fig_grid.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # 10.  Off-diagonal scatter colored by group
    # ------------------------------------------------------------------
    d_dim = R_pred.shape[-1]
    t_r, t_c = np.triu_indices(d_dim, k=1)

    R_pred_np   = R_pred.cpu().numpy()    # (n_test, d, d)
    R_oracle_np = R_oracle.cpu().numpy()  # (n_test, d, d)

    pred_off   = R_pred_np[:, t_r, t_c]    # (n_test, n_pairs)
    oracle_off = R_oracle_np[:, t_r, t_c]  # (n_test, n_pairs)

    fig_sc, ax = plt.subplots(figsize=(7, 7))
    clrs  = ["steelblue", "tomato"]
    lbls  = [f"Weak group (n={n_weak})", f"Strong group (n={n_strong})"]
    for g in [0, 1]:
        mask = groups == g
        ax.scatter(
            oracle_off[mask].ravel(), pred_off[mask].ravel(),
            alpha=0.45, s=14, color=clrs[g], label=lbls[g], linewidths=0,
        )

    x_all = oracle_off.ravel()
    y_all = pred_off.ravel()
    if x_all.std() > 1e-8:
        slope, intercept = np.polyfit(x_all, y_all, 1)
    else:
        slope, intercept = 1.0, 0.0

    lim = max(float(np.abs(x_all).max()), float(np.abs(y_all).max()), 1e-4) * 1.15
    x_line = np.array([-lim, lim])
    ax.plot(x_line, slope * x_line + intercept, "k-", lw=1.5,
            label=f"OLS slope={slope:.2f}  b={intercept:.3f}")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    ax.axvline(0, color="gray", lw=0.4, ls=":")

    r_val = (
        float(np.corrcoef(x_all, y_all)[0, 1])
        if x_all.std() > 1e-8 and y_all.std() > 1e-8
        else float("nan")
    )
    ax.set_xlabel("Oracle off-diagonal R*", fontsize=11)
    ax.set_ylabel("Predicted off-diagonal R̂", fontsize=11)
    ax.set_title(
        f"Off-diagonal scatter — all {n_test} test instances\n"
        f"Pearson r = {r_val:.3f}   OLS slope = {slope:.2f}   "
        f"intercept = {intercept:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    fig_sc.tight_layout()
    path = OUTPUT_DIR / "scatter_final.png"
    fig_sc.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig_sc)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # 11.  Summary
    # ------------------------------------------------------------------
    final_h = {i: attn_log[i][-1] for i in range(n_s3)}
    collapsed = [i for i, h in final_h.items() if h > 0.95]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Initial NLL:         {nll_log[0]:.4f}")
    print(f"  Final NLL:           {nll_log[-1]:.4f}")
    print(f"  Oracle NLL (floor):  {oracle_nll:.4f}")
    print(f"  Indep N(0,I) NLL:    {indep_nll:.4f}")
    print(f"  Off-diag Pearson r:  {r_val:.3f}")
    print(f"  OLS slope:           {slope:.3f}")
    print(f"  S3 attention H_norm (final):")
    for i, h in final_h.items():
        status = "COLLAPSED" if h > 0.95 else "OK"
        print(f"    blk {i}: {h:.4f}  [{status}]")
    if collapsed:
        print(f"\n[WARNING] Blocks {collapsed} have near-uniform attention "
              f"(H_norm > 0.95) → possible over-smoothing in Stage 3.")
    else:
        print("\n[OK] All S3 blocks show non-uniform attention.")
    if r_val < 0.3 or slope < 0.2:
        print("[WARNING] Low r / slope — model may not learn correlation structure.")
    else:
        print("[OK] Model appears to have learned the correlation structure.")
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
