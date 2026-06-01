"""
architecture_diagnostics.py — Three targeted architectural health checks.

  1. ReZero alpha magnitudes: do Stage-3 ICL blocks ever activate?
     (alpha_attn / alpha_ffn initialised at 0 — if they stay near 0 the blocks
     are identity maps and Stage 3 does nothing.)

  2. Feature ablation: does the model actually use X_test features?
     Compare predicted R when X_test is (a) original, (b) zeroed, (c) random.
     If predictions are unchanged, the model ignores query-side features.

  3. Support-size scaling: does adding more support context improve predictions?
     Oracle MSE should improve monotonically with n_support if ICL is working.

Workflow:
  • Load one hyperplane episode (same as overtrain_single.py).
  • Train a fresh model for TRAIN_STEPS steps on that episode to get
    non-trivial alpha values, then run the three diagnostics.

Usage:
    conda run -n multivariate-icl python debug/architecture_diagnostics.py
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPISODE_FILE = ROOT / "data" / "pit_hyperplane_debug" / "episode_000000.pt"
OUTPUT_DIR   = ROOT / "debug" / "arch_diagnostics"
BATCH_ELEM   = 0

TRAIN_STEPS  = 3_000   # short run — enough for alphas to leave 0
LR           = 1e-3
GRAD_CLIP    = 1.0
LOG_EVERY    = 100

SUPPORT_FRACS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]  # for scaling test


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cov_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    Sigma = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))


def _oracle_woodbury_corr(oracle_D, oracle_V):
    Sigma = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    V_c = oracle_V / std.unsqueeze(-1)
    D_c = (1.0 - (V_c ** 2).sum(-1)).clamp(min=1e-6)
    return D_c, V_c


def _off_diag_mse(R_pred: torch.Tensor, R_oracle: torch.Tensor) -> float:
    """Mean squared error on upper-triangle off-diagonal entries."""
    d = R_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=R_pred.device)
    return F.mse_loss(R_pred[..., ri, ci], R_oracle[..., ri, ci]).item()


def _predict(model, X_fwd, Z_fwd, N, device):
    model.eval()
    with torch.no_grad():
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
    model.train()
    Sigma = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
    return Sigma  # (1, n_test, d, d)


# ---------------------------------------------------------------------------
# 1.  Short training + ReZero alpha monitoring
# ---------------------------------------------------------------------------

def run_training_with_alpha_log(model, optimizer, X_fwd, Z_fwd, N, Z_test,
                                R_ora, device):
    """Train for TRAIN_STEPS steps; return per-step alpha log."""
    ri, ci = torch.triu_indices(Z_test.shape[-1], Z_test.shape[-1],
                                offset=1, device=device)
    n_s3 = len(model.s3_blocks)

    steps_log   = []
    nll_log     = []
    alpha_attn_log = {i: [] for i in range(n_s3)}
    alpha_ffn_log  = {i: [] for i in range(n_s3)}

    print(f"\n{'─'*60}")
    print(f"  Training {TRAIN_STEPS} steps to activate ReZero alphas")
    print(f"{'─'*60}")
    model.train()

    for step in range(TRAIN_STEPS):
        optimizer.zero_grad()
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
        loss_nll = woodbury_nll(Z_test, mu_Z, d_Z, V_Z)
        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        loss_mse   = F.mse_loss(Sigma_pred[..., ri, ci], R_ora[..., ri, ci])
        (loss_nll + loss_mse).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            steps_log.append(step)
            nll_log.append(loss_nll.item())
            for i, blk in enumerate(model.s3_blocks):
                alpha_attn_log[i].append(blk.alpha_attn.item())
                alpha_ffn_log[i].append(blk.alpha_ffn.item())

            alphas = " ".join(
                f"blk{i}=({model.s3_blocks[i].alpha_attn.item():.3f},"
                f"{model.s3_blocks[i].alpha_ffn.item():.3f})"
                for i in range(n_s3)
            )
            print(f"  step {step:5d}  NLL={loss_nll.item():.4f}  "
                  f"alpha(attn,ffn): {alphas}")

    return steps_log, nll_log, alpha_attn_log, alpha_ffn_log


def plot_rezero_alphas(steps_log, alpha_attn_log, alpha_ffn_log):
    n_s3   = len(alpha_attn_log)
    colors = plt.cm.viridis(np.linspace(0, 1, n_s3))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

    for i in range(n_s3):
        axes[0].plot(steps_log, alpha_attn_log[i], color=colors[i],
                     lw=1.5, label=f"S3 blk {i}")
        axes[1].plot(steps_log, alpha_ffn_log[i], color=colors[i],
                     lw=1.5, label=f"S3 blk {i}")

    for ax, title in zip(axes, ["alpha_attn (attention gate)",
                                 "alpha_ffn (FFN gate)"]):
        ax.axhline(0, color="red", ls="--", lw=0.8, alpha=0.5,
                   label="0 = identity (inactive)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("alpha value")
        ax.set_title(f"ReZero {title}")
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        "ReZero gate magnitudes — Stage 3 ICL blocks\n"
        "If all alphas stay near 0, Stage 3 is an identity and does no ICL.",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "rezero_alphas.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 2.  Feature ablation
# ---------------------------------------------------------------------------

def run_feature_ablation(model, X_train, Z_train, X_test, Z_test,
                         R_ora, device):
    """Compare off-diagonal MSE under four X configurations.

    Conditions
    ----------
    baseline       : X_train=orig, X_test=orig   (normal inference)
    zero_X_test    : X_train=orig, X_test=0       (test features nulled)
    random_X_test  : X_train=orig, X_test=random  (random test features)
    zero_X_all     : X_train=0,    X_test=0        (all features nulled)
    """
    N      = X_train.shape[1]
    n_test = X_test.shape[1]
    p      = X_test.shape[2]

    torch.manual_seed(0)
    X_test_random = torch.randn_like(X_test)
    X_train_zero  = torch.zeros_like(X_train)
    X_test_zero   = torch.zeros_like(X_test)

    conditions = {
        "baseline\n(orig X)":       (X_train,      X_test),
        "zero X_test\n(null query)": (X_train,      X_test_zero),
        "random X_test\n(noise)":    (X_train,      X_test_random),
        "zero X_all\n(no features)": (X_train_zero, X_test_zero),
    }

    mse_results = {}
    nll_results = {}
    D_ora_c, V_ora_c = _oracle_woodbury_corr(
        episode_oracle_D, episode_oracle_V
    )

    model.eval()
    for label, (xt, xq) in conditions.items():
        X_fwd = torch.cat([xt, xq], dim=1)
        Z_fwd = torch.cat([Z_train, Z_test], dim=1)
        with torch.no_grad():
            mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        mse_results[label] = _off_diag_mse(Sigma_pred, R_ora)
        nll_results[label] = woodbury_nll(
            Z_test, mu_Z, d_Z, V_Z
        ).item()
    model.train()

    # Plot
    labels = list(mse_results.keys())
    mse_vals = [mse_results[l] for l in labels]
    nll_vals = [nll_results[l] for l in labels]
    baseline_mse = mse_vals[0]
    baseline_nll = nll_vals[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_bar = ["steelblue", "tomato", "orange", "gray"]

    for ax, vals, base, ylabel, title in [
        (axes[0], mse_vals, baseline_mse,
         "Off-diag MSE vs oracle R", "Off-diagonal MSE (lower = better)"),
        (axes[1], nll_vals, baseline_nll,
         "Woodbury NLL", "Woodbury NLL (lower = better)"),
    ]:
        bars = ax.bar(labels, vals, color=colors_bar, edgecolor="k", lw=0.7)
        ax.axhline(base, color="steelblue", ls="--", lw=1.0, alpha=0.6,
                   label="baseline")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Feature ablation — effect of nulling / randomising X on predictions\n"
        "If MSE/NLL are unchanged when X_test=0: model ignores query features.",
        fontsize=10,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "feature_ablation.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Report
    print("\n  Feature ablation results:")
    print(f"  {'Condition':<28}  {'MSE':>8}  {'ΔMSE':>8}  {'NLL':>8}")
    print(f"  {'─'*58}")
    for label, mse, nll in zip(labels, mse_vals, nll_vals):
        delta = mse - baseline_mse
        clean = label.replace("\n", " ")
        print(f"  {clean:<28}  {mse:>8.5f}  {delta:>+8.5f}  {nll:>8.4f}")

    return mse_results


# ---------------------------------------------------------------------------
# 3.  Support-size scaling
# ---------------------------------------------------------------------------

def run_support_scaling(model, X_train, Z_train, X_test, Z_test,
                        R_ora, device):
    """Measure off-diagonal MSE and NLL as n_support increases.

    If the model uses context, both metrics should improve monotonically.
    """
    N_full = X_train.shape[1]
    D_ora_c, V_ora_c = _oracle_woodbury_corr(
        episode_oracle_D, episode_oracle_V
    )

    n_support_vals = [max(1, int(f * N_full)) for f in SUPPORT_FRACS]
    mse_vals = []
    nll_vals = []

    model.eval()
    print(f"\n  Support-size scaling  (N_full={N_full})")
    print(f"  {'n_sup':>6}  {'frac':>6}  {'MSE':>9}  {'NLL':>9}")
    print(f"  {'─'*38}")

    for n_sup in n_support_vals:
        Xt = X_train[:, :n_sup, :]
        Zt = Z_train[:, :n_sup, :]
        X_fwd = torch.cat([Xt, X_test], dim=1)
        Z_fwd = torch.cat([Zt, Z_test], dim=1)

        with torch.no_grad():
            mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=n_sup)

        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        mse = _off_diag_mse(Sigma_pred, R_ora)
        nll = woodbury_nll(Z_test, mu_Z, d_Z, V_Z).item()
        mse_vals.append(mse)
        nll_vals.append(nll)
        print(f"  {n_sup:>6d}  {n_sup/N_full:>6.2f}  {mse:>9.5f}  {nll:>9.4f}")

    model.train()

    # Oracle NLL for reference
    with torch.no_grad():
        oracle_nll = woodbury_nll(
            Z_test, torch.zeros_like(Z_test), D_ora_c, V_ora_c
        ).item()

    # Plot
    fracs = [n / N_full for n in n_support_vals]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(fracs, mse_vals, "o-", color="steelblue", lw=1.8, ms=6)
    axes[0].set_xlabel("Support fraction (n_support / N_full)")
    axes[0].set_ylabel("Off-diagonal MSE vs oracle R")
    axes[0].set_title("Support scaling — off-diagonal MSE\n"
                       "Should decrease as more support is given")
    axes[0].set_xticks(fracs)
    axes[0].set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=8)

    axes[1].plot(fracs, nll_vals, "o-", color="tomato", lw=1.8, ms=6,
                 label="Model NLL")
    axes[1].axhline(oracle_nll, color="green", ls="--", lw=1.2,
                    label=f"Oracle NLL ({oracle_nll:.3f})")
    axes[1].set_xlabel("Support fraction (n_support / N_full)")
    axes[1].set_ylabel("Woodbury NLL")
    axes[1].set_title("Support scaling — NLL\n"
                       "Should decrease toward oracle as support grows")
    axes[1].set_xticks(fracs)
    axes[1].set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=8)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Support-size scaling — does more context help?\n"
        "Flat curves → model is not using support context (ICL failure).",
        fontsize=10,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "support_scaling.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    return n_support_vals, mse_vals, nll_vals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Module-level episode tensors shared across diagnostic functions
episode_oracle_D: torch.Tensor
episode_oracle_V: torch.Tensor


def main() -> None:
    global episode_oracle_D, episode_oracle_V

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load episode
    # ------------------------------------------------------------------
    print(f"Loading: {EPISODE_FILE}")
    episode = torch.load(EPISODE_FILE, weights_only=True)
    b = BATCH_ELEM

    X_train  = episode["X_train"][[b]].float().to(device)
    Z_train  = episode["Z_train"][[b]].float().to(device)
    X_test   = episode["X_test"][[b]].float().to(device)
    Z_test   = episode["Z_test"][[b]].float().to(device)
    oracle_D = episode["oracle_D"][[b]].float().to(device)
    oracle_V = episode["oracle_V"][[b]].float().to(device)

    episode_oracle_D = oracle_D
    episode_oracle_V = oracle_V

    N = X_train.shape[1]
    d = Z_train.shape[2]
    print(f"N={N}, n_test={X_test.shape[1]}, d={d}, r={oracle_V.shape[-1]}")

    R_ora  = _cov_to_corr(oracle_D, oracle_V)   # (1, n_test, d, d)
    X_fwd  = torch.cat([X_train, X_test], dim=1)
    Z_fwd  = torch.cat([Z_train, Z_test], dim=1)

    # ------------------------------------------------------------------
    # Build fresh model
    # ------------------------------------------------------------------
    cfg_model = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    model     = build_copula_tabicl_v2(SimpleNamespace(model=cfg_model)).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # ------------------------------------------------------------------
    # Diagnostic 1 — ReZero alpha monitoring (during short training)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 1: ReZero alpha magnitudes")
    print("=" * 60)

    steps_log, nll_log, alpha_attn_log, alpha_ffn_log = run_training_with_alpha_log(
        model, optimizer, X_fwd, Z_fwd, N, Z_test, R_ora, device
    )
    plot_rezero_alphas(steps_log, alpha_attn_log, alpha_ffn_log)

    # Final alpha summary
    print("\n  Final alpha values after training:")
    for i, blk in enumerate(model.s3_blocks):
        aa = blk.alpha_attn.item()
        af = blk.alpha_ffn.item()
        status = "INACTIVE" if abs(aa) < 1e-3 and abs(af) < 1e-3 else "active"
        print(f"    S3 blk {i}: alpha_attn={aa:.5f}  alpha_ffn={af:.5f}  [{status}]")

    # ------------------------------------------------------------------
    # Diagnostic 2 — Feature ablation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 2: Feature ablation")
    print("=" * 60)

    run_feature_ablation(model, X_train, Z_train, X_test, Z_test, R_ora, device)

    # ------------------------------------------------------------------
    # Diagnostic 3 — Support-size scaling
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 3: Support-size scaling")
    print("=" * 60)

    n_sup_vals, mse_vals, nll_vals = run_support_scaling(
        model, X_train, Z_train, X_test, Z_test, R_ora, device
    )

    # Is scaling monotone?
    mse_decreasing = all(mse_vals[i] >= mse_vals[i+1]
                         for i in range(len(mse_vals)-1))
    nll_decreasing = all(nll_vals[i] >= nll_vals[i+1]
                         for i in range(len(nll_vals)-1))

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    final_alphas = [(b.alpha_attn.item(), b.alpha_ffn.item())
                    for b in model.s3_blocks]
    any_active = any(abs(aa) > 1e-3 or abs(af) > 1e-3
                     for aa, af in final_alphas)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  ReZero alphas activated:   {'YES' if any_active else 'NO — Stage 3 is identity!'}")
    print(f"  MSE decreases with support: {'YES (ICL working)' if mse_decreasing else 'NO — model ignores context!'}")
    print(f"  NLL decreases with support: {'YES' if nll_decreasing else 'NO'}")
    print(f"\n  All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
