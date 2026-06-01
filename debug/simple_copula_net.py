"""
simple_copula_net.py — Minimal cross-attention baseline for copula prediction.

Architecture (SimpleCopulaNet):
  1. Support encoder  : MLP( cat(X_i, Z_i, vech(Z_i Z_i^T)) ) → h_i  ∈ R^d_h
  2. Query encoder    : MLP( X_q )                              → q_j  ∈ R^d_h
  3. Cross-attention  : q_j attends over {h_i} → context c_j   ∈ R^d_h
  4. Readout MLP      : cat(c_j, q_j)           → U_j  ∈ R^{d×r}
  5. Woodbury reparameterisation (same as CopulaTabICLv2):
       C_diag = 1/(1+‖U‖²)    W = U/√(1+‖U‖²)   →  Σ = diag(C) + WW^T

Design intent:
  • No ReZero (standard residual from the start → gradients flow immediately).
  • Cross-attention is structurally forced to compare each query's X to support
    (X_i, Z_i) pairs → can learn "which side of the hyperplane am I on?"
  • vech(Z_i Z_i^T) in the support input gives the covariance signal directly.
  • Attention entropy should stay non-uniform if the model uses features.

Comparison targets (CopulaTabICLv2, same episode, same steps):
  5k steps  → NLL≈-17   MSE≈0.594   r≈0.245   H_norm≈1.000 (collapsed)
  50k steps → NLL≈-22.8 MSE≈0.514   r≈0.252   H_norm≈1.000 (collapsed)

Usage:
    conda run -n multivariate-icl python debug/simple_copula_net.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loss import woodbury_nll  # noqa: E402
from viz import plot_corr_grid  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPISODE_FILE  = ROOT / "data" / "pit_hyperplane_debug" / "episode_000000.pt"
OUTPUT_DIR    = ROOT / "debug" / "simple_copula_results"
BATCH_ELEM    = 0

N_STEPS       = 5_000
LR            = 1e-3
GRAD_CLIP     = 1.0
LOG_EVERY     = 100
D_HIDDEN      = 128
N_HEADS       = 4

SUPPORT_FRACS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

# CopulaTabICLv2 reference numbers for comparison (from overtrain runs)
REF = {
    "tabicl_5k":  dict(nll=-16.99, mse=0.594, r=0.245, h_norm=1.000),
    "tabicl_50k": dict(nll=-22.76, mse=0.514, r=0.252, h_norm=1.000),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleCopulaNet(nn.Module):
    """Cross-attention copula predictor.

    Args:
        p_max    : max number of input features (pad/clip applied).
        d_max    : max number of target dimensions.
        rank_max : rank of the low-rank factor W.
        d_hidden : hidden dimension for encoders and attention.
        n_heads  : number of attention heads in the cross-attention layer.
    """

    def __init__(
        self,
        p_max:    int = 20,
        d_max:    int = 8,
        rank_max: int = 8,
        d_hidden: int = 128,
        n_heads:  int = 4,
    ) -> None:
        super().__init__()
        self.p_max    = p_max
        self.d_max    = d_max
        self.rank_max = rank_max
        self.d_hidden = d_hidden

        d_vech   = d_max * (d_max + 1) // 2
        d_sup_in = p_max + d_max + d_vech   # X_i ‖ Z_i ‖ vech(Z_i Z_i^T)

        # --- Support encoder ---
        self.enc_sup = nn.Sequential(
            nn.Linear(d_sup_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
        )

        # --- Query encoder ---
        self.enc_qry = nn.Sequential(
            nn.Linear(p_max, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
        )

        # --- Cross-attention: q_j attends over support {h_i} ---
        self.cross_attn = nn.MultiheadAttention(
            d_hidden, n_heads, dropout=0.0, batch_first=True
        )
        self.post_attn_norm = nn.LayerNorm(d_hidden)

        # --- Per-dim embedding: breaks output symmetry across dimensions ---
        self.dim_emb = nn.Parameter(torch.randn(d_max, d_hidden // 4))

        # --- Readout: (ctx ‖ qry ‖ dim_emb) → U_j ∈ R^rank_max per dim ---
        d_head_in = d_hidden + d_hidden + d_hidden // 4
        self.readout = nn.Sequential(
            nn.Linear(d_head_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, rank_max),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        X_all:     torch.Tensor,   # (B, N, p)
        Z_all:     torch.Tensor,   # (B, N, d)
        n_support: int,
        return_attn: bool = False,
    ) -> tuple:
        B, N, p = X_all.shape
        d = Z_all.shape[-1]
        n_qry = N - n_support

        # --- Pad/clip features and targets to (p_max, d_max) ---
        if p < self.p_max:
            X_all = F.pad(X_all, (0, self.p_max - p))
        else:
            X_all = X_all[..., : self.p_max]

        if d < self.d_max:
            Z_sup_pad = F.pad(Z_all[:, :n_support], (0, self.d_max - d))
        else:
            Z_sup_pad = Z_all[:, :n_support, : self.d_max]

        X_sup = X_all[:, :n_support]   # (B, n_sup, p_max)
        X_qry = X_all[:, n_support:]   # (B, n_qry, p_max)

        # --- vech(Z_i Z_i^T) for support instances ---
        outer    = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)
        tril_i, tril_j = torch.tril_indices(self.d_max, self.d_max, device=X_all.device)
        vech     = outer[:, :, tril_i, tril_j]   # (B, n_sup, d_vech)

        # --- Encode support ---
        sup_in  = torch.cat([X_sup, Z_sup_pad, vech], dim=-1)
        sup_emb = self.enc_sup(sup_in)            # (B, n_sup, d_h)

        # --- Encode queries ---
        qry_emb = self.enc_qry(X_qry)            # (B, n_qry, d_h)

        # --- Cross-attention: each query attends to all support instances ---
        ctx, attn_w = self.cross_attn(
            qry_emb, sup_emb, sup_emb,
            need_weights=True,
            average_attn_weights=True,
        )   # ctx: (B, n_qry, d_h),  attn_w: (B, n_qry, n_sup)

        # Residual + norm
        ctx = self.post_attn_norm(ctx + qry_emb)   # (B, n_qry, d_h)

        # --- Readout: tile over d dimensions + per-dim embedding ---
        ctx_exp  = ctx.unsqueeze(2).expand(B, n_qry, self.d_max, -1)
        qry_exp  = qry_emb.unsqueeze(2).expand(B, n_qry, self.d_max, -1)
        dim_exp  = self.dim_emb.unsqueeze(0).unsqueeze(0).expand(B, n_qry, -1, -1)
        head_in  = torch.cat([ctx_exp, qry_exp, dim_exp], dim=-1)
        # (B, n_qry, d_max, d_h + d_h + d_h//4)

        U = self.readout(head_in)                  # (B, n_qry, d_max, rank_max)
        U = U[:, :, :d, :]                         # slice to actual d

        # --- Woodbury reparameterisation (unit diagonal) ---
        U_sq_norm = (U ** 2).sum(-1)               # (B, n_qry, d)
        C_diag    = 1.0 / (1.0 + U_sq_norm)        # (B, n_qry, d)
        W         = U / (1.0 + U_sq_norm.unsqueeze(-1)).sqrt()  # (B, n_qry, d, r)

        mu_Z = torch.zeros(B, n_qry, d, device=X_all.device, dtype=X_all.dtype)

        if return_attn:
            return mu_Z, C_diag, W, attn_w
        return mu_Z, C_diag, W


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cov_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    Sigma = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std   = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))


def _oracle_woodbury_corr(oracle_D, oracle_V):
    Sigma = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
    std   = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    V_c   = oracle_V / std.unsqueeze(-1)
    D_c   = (1.0 - (V_c ** 2).sum(-1)).clamp(min=1e-6)
    return D_c, V_c


def _off_diag_mse(Sigma_pred: torch.Tensor, R_ora: torch.Tensor) -> float:
    d = Sigma_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=Sigma_pred.device)
    return F.mse_loss(Sigma_pred[..., ri, ci], R_ora[..., ri, ci]).item()


def _attn_h_norm(attn_w: torch.Tensor, n_support: int) -> float:
    """Normalised entropy of cross-attention weights (B, n_qry, n_sup)."""
    w = attn_w.clamp(min=1e-10)
    H = -(w * w.log()).sum(dim=-1).mean().item()
    return H / math.log(max(n_support, 2))


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------

def run_ablation(model, X_train, Z_train, X_test, Z_test, R_ora, device):
    N = X_train.shape[1]
    conditions = {
        "orig X":        (X_train,              X_test),
        "zero X_test":   (X_train,              torch.zeros_like(X_test)),
        "random X_test": (X_train,              torch.randn_like(X_test)),
        "zero X_all":    (torch.zeros_like(X_train), torch.zeros_like(X_test)),
    }
    model.eval()
    results = {}
    for label, (xt, xq) in conditions.items():
        X_fwd = torch.cat([xt, xq], dim=1)
        Z_fwd = torch.cat([Z_train, Z_test], dim=1)
        with torch.no_grad():
            mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
        Sigma = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        results[label] = dict(
            mse=_off_diag_mse(Sigma, R_ora),
            nll=woodbury_nll(Z_test, mu_Z, d_Z, V_Z).item(),
        )
    model.train()
    return results


# ---------------------------------------------------------------------------
# Support scaling
# ---------------------------------------------------------------------------

def run_scaling(model, X_train, Z_train, X_test, Z_test, R_ora, device):
    N_full = X_train.shape[1]
    D_ora_c, V_ora_c = _oracle_woodbury_corr(
        oracle_D.to(device), oracle_V.to(device)
    )
    with torch.no_grad():
        oracle_nll = woodbury_nll(
            Z_test, torch.zeros_like(Z_test), D_ora_c, V_ora_c
        ).item()

    n_sups, mse_vals, nll_vals = [], [], []
    model.eval()
    for frac in SUPPORT_FRACS:
        n_sup = max(1, int(frac * N_full))
        X_fwd = torch.cat([X_train[:, :n_sup], X_test], dim=1)
        Z_fwd = torch.cat([Z_train[:, :n_sup], Z_test], dim=1)
        with torch.no_grad():
            mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=n_sup)
        Sigma = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        n_sups.append(n_sup)
        mse_vals.append(_off_diag_mse(Sigma, R_ora))
        nll_vals.append(woodbury_nll(Z_test, mu_Z, d_Z, V_Z).item())
    model.train()
    return n_sups, mse_vals, nll_vals, N_full, oracle_nll


# ---------------------------------------------------------------------------
# Module-level oracle tensors (set in main, used by run_scaling)
# ---------------------------------------------------------------------------
oracle_D: torch.Tensor
oracle_V: torch.Tensor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global oracle_D, oracle_V
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load episode
    # ------------------------------------------------------------------
    episode  = torch.load(EPISODE_FILE, weights_only=True)
    b        = BATCH_ELEM
    X_train  = episode["X_train"][[b]].float().to(device)
    Z_train  = episode["Z_train"][[b]].float().to(device)
    X_test   = episode["X_test"][[b]].float().to(device)
    Z_test   = episode["Z_test"][[b]].float().to(device)
    oracle_D = episode["oracle_D"][[b]].float()
    oracle_V = episode["oracle_V"][[b]].float()

    oracle_D_dev = oracle_D.to(device)
    oracle_V_dev = oracle_V.to(device)

    N, n_test, d = X_train.shape[1], X_test.shape[1], Z_train.shape[2]
    print(f"N={N}  n_test={n_test}  d={d}  r={oracle_V.shape[-1]}")

    R_ora  = _cov_to_corr(oracle_D_dev, oracle_V_dev)   # (1, n_test, d, d)
    D_ora_c, V_ora_c = _oracle_woodbury_corr(oracle_D_dev, oracle_V_dev)
    with torch.no_grad():
        oracle_nll = woodbury_nll(
            Z_test, torch.zeros_like(Z_test), D_ora_c, V_ora_c
        ).item()
        V_z   = torch.zeros(1, n_test, d, 1, device=device)
        D_one = torch.ones(1, n_test, d, device=device)
        indep_nll = woodbury_nll(
            Z_test, torch.zeros_like(Z_test), D_one, V_z
        ).item()

    print(f"Oracle NLL: {oracle_nll:.4f}   Indep NLL: {indep_nll:.4f}")

    # Hyperplane groups
    v_norms = oracle_V_dev[0].norm(dim=(-2, -1))
    groups  = (v_norms > v_norms.median()).long().cpu().numpy()
    n_weak, n_strong = int((groups==0).sum()), int((groups==1).sum())
    print(f"Groups: {n_weak} weak, {n_strong} strong")

    # ------------------------------------------------------------------
    # Build SimpleCopulaNet
    # ------------------------------------------------------------------
    model = SimpleCopulaNet(
        p_max=X_train.shape[2], d_max=d,
        rank_max=oracle_V.shape[-1], d_hidden=D_HIDDEN, n_heads=N_HEADS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleCopulaNet params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    ri, ci    = torch.triu_indices(d, d, offset=1, device=device)
    X_fwd     = torch.cat([X_train, X_test], dim=1)
    Z_fwd     = torch.cat([Z_train, Z_test], dim=1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    steps_log, nll_log, mse_log, h_log = [], [], [], []

    print(f"\nTraining SimpleCopulaNet for {N_STEPS} steps  LR={LR}\n")
    model.train()

    for step in range(N_STEPS):
        optimizer.zero_grad()

        mu_Z, d_Z, V_Z, attn_w = model(X_fwd, Z_fwd, n_support=N, return_attn=True)

        loss_nll   = woodbury_nll(Z_test, mu_Z, d_Z, V_Z)
        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        loss_mse   = F.mse_loss(Sigma_pred[..., ri, ci], R_ora[..., ri, ci])
        loss       = loss_nll + loss_mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            h = _attn_h_norm(attn_w.detach(), N)
            steps_log.append(step)
            nll_log.append(loss_nll.item())
            mse_log.append(loss_mse.item())
            h_log.append(h)
            print(
                f"  step {step:5d}  NLL={loss_nll.item():.4f}  "
                f"oracle={oracle_nll:.4f}  MSE={loss_mse.item():.5f}  "
                f"H_norm={h:.3f}"
            )

    # ------------------------------------------------------------------
    # Final inference
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        mu_Z, d_Z, V_Z, attn_w_final = model(
            X_fwd, Z_fwd, n_support=N, return_attn=True
        )
    Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
    R_pred     = Sigma_pred[0]    # (n_test, d, d)
    R_oracle   = R_ora[0]         # (n_test, d, d)

    # off-diagonal scatter stats
    t_r, t_c   = np.triu_indices(d, k=1)
    pred_off   = R_pred.cpu().numpy()[:, t_r, t_c]
    oracle_off = R_oracle.cpu().numpy()[:, t_r, t_c]
    x_all, y_all = oracle_off.ravel(), pred_off.ravel()
    slope, intercept = np.polyfit(x_all, y_all, 1) if x_all.std() > 1e-8 else (1., 0.)
    r_val = float(np.corrcoef(x_all, y_all)[0, 1]) if x_all.std() > 1e-8 else float("nan")
    final_h = _attn_h_norm(attn_w_final, N)
    final_mse = _off_diag_mse(Sigma_pred, R_ora)

    # ------------------------------------------------------------------
    # Plot 1: Training curves
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(steps_log, nll_log, color="steelblue", lw=1.5)
    axes[0].axhline(oracle_nll, color="green", ls="--", lw=1.2,
                    label=f"Oracle ({oracle_nll:.3f})")
    axes[0].axhline(indep_nll,  color="gray",  ls=":",  lw=1.2,
                    label=f"Indep ({indep_nll:.3f})")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Woodbury NLL")
    axes[0].set_title("NLL"); axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, mse_log, color="tomato", lw=1.5)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Off-diag MSE")
    axes[1].set_title("Off-diagonal MSE vs oracle R")

    axes[2].plot(steps_log, h_log, color="purple", lw=1.5)
    axes[2].axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6,
                    label="1.0 = uniform (collapsed)")
    axes[2].axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6,
                    label="0.0 = peaked (healthy)")
    axes[2].set_ylim(-0.05, 1.10)
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("H_norm")
    axes[2].set_title("Cross-attention entropy\n(should stay < 1.0)")
    axes[2].legend(fontsize=8)

    fig.suptitle("SimpleCopulaNet — training curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'training_curves.png'}")

    # ------------------------------------------------------------------
    # Plot 2: All test instances — sorted by group
    # ------------------------------------------------------------------
    sort_order      = np.argsort(groups)
    R_pred_sorted   = R_pred[sort_order].cpu()
    R_oracle_sorted = R_oracle[sort_order].cpu()

    fig_grid = plot_corr_grid(
        estimators={"SimpleCopulaNet": R_pred_sorted},
        oracle_R=R_oracle_sorted,
        n_instances=n_test,
        title=(
            f"SimpleCopulaNet — all {n_test} test instances\n"
            f"{n_weak} weak (top) / {n_strong} strong (bottom) — {N_STEPS:,} steps"
        ),
    )
    fig_grid.savefig(OUTPUT_DIR / "all_predictions.png", dpi=100, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"Saved: {OUTPUT_DIR / 'all_predictions.png'}")

    # ------------------------------------------------------------------
    # Plot 3: Off-diagonal scatter
    # ------------------------------------------------------------------
    fig_sc, ax = plt.subplots(figsize=(7, 7))
    clrs = ["steelblue", "tomato"]
    for g in [0, 1]:
        mask = groups == g
        ax.scatter(oracle_off[mask].ravel(), pred_off[mask].ravel(),
                   alpha=0.45, s=14, color=clrs[g], linewidths=0,
                   label=f"{'Weak' if g==0 else 'Strong'} (n={int((groups==g).sum())})")
    lim = max(float(np.abs(x_all).max()), float(np.abs(y_all).max()), 1e-4) * 1.15
    x_line = np.array([-lim, lim])
    ax.plot(x_line, slope*x_line+intercept, "k-", lw=1.5,
            label=f"OLS slope={slope:.2f}  b={intercept:.3f}")
    ax.plot([-lim,lim], [-lim,lim], "k--", lw=0.8, alpha=0.4, label="y=x")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.4, ls=":"); ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set_xlabel("Oracle R*"); ax.set_ylabel("Predicted R̂")
    ax.set_title(f"Off-diagonal scatter\nr={r_val:.3f}  slope={slope:.2f}")
    ax.legend(fontsize=9)
    fig_sc.tight_layout()
    fig_sc.savefig(OUTPUT_DIR / "scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig_sc)
    print(f"Saved: {OUTPUT_DIR / 'scatter.png'}")

    # ------------------------------------------------------------------
    # Feature ablation
    # ------------------------------------------------------------------
    print("\n--- Feature ablation ---")
    abl = run_ablation(model, X_train, Z_train, X_test, Z_test, R_ora, device)
    print(f"  {'Condition':<20}  {'MSE':>8}  {'ΔMSE':>8}")
    baseline_mse = abl["orig X"]["mse"]
    for label, res in abl.items():
        print(f"  {label:<20}  {res['mse']:>8.5f}  {res['mse']-baseline_mse:>+8.5f}")

    # Plot ablation
    fig_abl, ax_abl = plt.subplots(figsize=(8, 4))
    lbls = list(abl.keys())
    mses = [abl[l]["mse"] for l in lbls]
    bars = ax_abl.bar(lbls, mses,
                      color=["steelblue","tomato","orange","gray"],
                      edgecolor="k", lw=0.7)
    ax_abl.axhline(baseline_mse, color="steelblue", ls="--", lw=1.0, alpha=0.6)
    for bar, v in zip(bars, mses):
        ax_abl.text(bar.get_x()+bar.get_width()/2, v+0.002, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
    ax_abl.set_ylabel("Off-diag MSE vs oracle R")
    ax_abl.set_title("Feature ablation — SimpleCopulaNet\n"
                      "MSE increases when X_test=0 → model uses query features")
    fig_abl.tight_layout()
    fig_abl.savefig(OUTPUT_DIR / "feature_ablation.png", dpi=120, bbox_inches="tight")
    plt.close(fig_abl)
    print(f"Saved: {OUTPUT_DIR / 'feature_ablation.png'}")

    # ------------------------------------------------------------------
    # Support scaling
    # ------------------------------------------------------------------
    print("\n--- Support scaling ---")
    n_sups, mse_sc, nll_sc, N_full, oracle_nll_sc = run_scaling(
        model, X_train, Z_train, X_test, Z_test, R_ora, device
    )
    fracs = [n/N_full for n in n_sups]
    print(f"  {'n_sup':>6}  {'frac':>6}  {'MSE':>9}  {'NLL':>9}")
    for ns, frac, mse, nll in zip(n_sups, fracs, mse_sc, nll_sc):
        print(f"  {ns:>6d}  {frac:>6.2f}  {mse:>9.5f}  {nll:>9.4f}")

    mse_decreasing = all(mse_sc[i] >= mse_sc[i+1] for i in range(len(mse_sc)-1))

    fig_sc2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    axes2[0].plot(fracs, mse_sc, "o-", color="steelblue", lw=1.8, ms=6)
    axes2[0].set_xlabel("Support fraction"); axes2[0].set_ylabel("Off-diag MSE")
    axes2[0].set_title("Support scaling — MSE\n(should decrease monotonically)")
    axes2[0].set_xticks(fracs)
    axes2[0].set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=8)

    axes2[1].plot(fracs, nll_sc, "o-", color="tomato", lw=1.8, ms=6, label="NLL")
    axes2[1].axhline(oracle_nll_sc, color="green", ls="--", lw=1.2,
                     label=f"Oracle ({oracle_nll_sc:.3f})")
    axes2[1].set_xlabel("Support fraction"); axes2[1].set_ylabel("NLL")
    axes2[1].set_title("Support scaling — NLL")
    axes2[1].set_xticks(fracs)
    axes2[1].set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=8)
    axes2[1].legend(fontsize=8)
    fig_sc2.suptitle("SimpleCopulaNet — support scaling", fontsize=11)
    fig_sc2.tight_layout()
    fig_sc2.savefig(OUTPUT_DIR / "support_scaling.png", dpi=120, bbox_inches="tight")
    plt.close(fig_sc2)
    print(f"Saved: {OUTPUT_DIR / 'support_scaling.png'}")

    # ------------------------------------------------------------------
    # Summary + comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SUMMARY — SimpleCopulaNet vs CopulaTabICLv2")
    print("=" * 65)
    print(f"  {'Model':<28}  {'NLL':>8}  {'MSE':>8}  {'r':>6}  {'H_norm':>8}")
    print(f"  {'─'*64}")
    print(f"  {'SimpleCopulaNet (5k)':<28}  {nll_log[-1]:>8.3f}  {final_mse:>8.4f}"
          f"  {r_val:>6.3f}  {final_h:>8.4f}")
    for tag, ref in REF.items():
        print(f"  {f'CopulaTabICLv2 ({tag})':<28}  {ref['nll']:>8.3f}"
              f"  {ref['mse']:>8.4f}  {ref['r']:>6.3f}  {ref['h_norm']:>8.4f}")
    print(f"\n  Feature ablation  (ΔMSE when X_test=0):")
    print(f"    SimpleCopulaNet:  {abl['zero X_test']['mse'] - baseline_mse:>+.5f}")
    print(f"    CopulaTabICLv2:   -0.25261  (X=0 was BETTER → model ignored features)")
    print(f"\n  Support scaling MSE monotone decreasing: "
          f"{'YES' if mse_decreasing else 'NO'}")
    print(f"\n  All plots → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
