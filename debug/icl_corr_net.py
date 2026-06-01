"""
icl_corr_net.py — Minimal ICL model that learns correlation matrices in-context.

Architecture
------------
  enc_sup : (X_i, vech(Y_i⊗Y_i)) → d_h     encodes support instance + local correlation
  enc_qry : X_q               → d_h           encodes query position
  cross_attn: Q=query, K=V=support            each query reads relevant support instances
  readout_L : (ctx, qry) → Cholesky L         per-query correlation matrix

Why this does real ICL
----------------------
  vech(Y_i⊗Y_i) is a rank-1 noisy estimate of Σ_i.  For a query X_q in group G,
  the cross-attention peaks on support instances with similar X (same group), and
  their averaged Y⊗Y ≈ Σ_G.  The model learns: "look at who is nearby in X space
  and inherit their correlation pattern".  Each episode has a different hyperplane
  and different (Σ₁, Σ₂), so the model cannot memorise any fixed X→R mapping —
  it must read the support Z values.

ICL test
--------
  After training, freeze weights and swap support between two episodes while
  keeping X_test fixed.  If predictions change substantially → true ICL.

Usage
-----
  conda run -n multivariate-icl python debug/icl_corr_net.py
"""

from __future__ import annotations

import math, os, sys
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
from data_gen import generate_episode  # noqa: E402
from viz import plot_corr_grid         # noqa: E402

OUTPUT_DIR = ROOT / "debug" / "icl_test_results"

# ── Hyperparameters ─────────────────────────────────────────────────────────
P, D, R       = 8, 6, 4          # feature dim, target dim, low-rank
N_TRAIN       = 128              # support size
N_TEST        = 16               # query size
D_HIDDEN      = 64
N_HEADS       = 4
N_STEPS       = 5_000
LR            = 3e-4
GRAD_CLIP     = 1.0
LOG_EVERY     = 500
VAL_EPISODES  = 50               # episodes for validation


# ── Helpers ──────────────────────────────────────────────────────────────────

def cov_to_corr(D, V):
    """Woodbury (D,V) → correlation matrix."""
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def sample_ep(device):
    """Generate one hyperplane episode; return (X_tr,Y_tr,X_te,Y_te,R_oracle)."""
    X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
        B=1, p=P, d=D, r=R,
        n_train=N_TRAIN, n_test=N_TEST,
        device=device,
        hyperplane_bimodal=True,
        return_oracle=True,
    )
    R_ora = cov_to_corr(oracle["D"], oracle["V"])   # (1, n_test, d, d)
    return X_tr, Y_tr, X_te, Y_te, R_ora


# ── Model ────────────────────────────────────────────────────────────────────

class ICLCorrNet(nn.Module):
    """
    Simple cross-attention model for in-context correlation prediction.

    Support token = (X_i, vech(Y_i⊗Y_i)):  encodes position + local covariance.
    Query token   = X_q:                    encodes position only.
    Cross-attention reads relevant support instances for each query.
    Cholesky readout produces a valid correlation matrix per query.
    """

    def __init__(self, p, d, d_h=64, n_heads=4):
        super().__init__()
        self.d = d
        d_vech = d * (d + 1) // 2

        self.enc_sup = nn.Sequential(
            nn.Linear(p + d_vech, d_h), nn.LayerNorm(d_h), nn.GELU(),
            nn.Linear(d_h, d_h),        nn.LayerNorm(d_h),
        )
        self.enc_qry = nn.Sequential(
            nn.Linear(p, d_h),   nn.LayerNorm(d_h), nn.GELU(),
            nn.Linear(d_h, d_h), nn.LayerNorm(d_h),
        )
        self.cross_attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.post_norm  = nn.LayerNorm(d_h)

        # Cholesky readout: takes (ctx ‖ qry_emb) → L lower-triangular entries
        d_L = d * (d + 1) // 2
        self.readout_L = nn.Sequential(
            nn.Linear(d_h * 2, d_h), nn.GELU(),
            nn.Linear(d_h, d_L),
        )

        ti, tj = torch.tril_indices(d, d)
        self.register_buffer("ti", ti)
        self.register_buffer("tj", tj)
        diag_idx = torch.arange(d)
        self.register_buffer("diag_idx", diag_idx)

    def forward(self, X_tr, Y_tr, X_te):
        """
        X_tr : (B, N, p)   support features
        Y_tr : (B, N, d)   support observations  (used as proxy for Z)
        X_te : (B, n_q, p) query features
        Returns R_pred : (B, n_q, d, d) predicted correlation matrices
                attn_w : (B, n_q, N)   attention weights (for ICL diagnostics)
        """
        B, N, p  = X_tr.shape
        n_q      = X_te.shape[1]
        d        = self.d

        # ── Support encoding: include vech(Y_i⊗Y_i) as correlation evidence ──
        outer = Y_tr.unsqueeze(-1) * Y_tr.unsqueeze(-2)    # (B, N, d, d)
        vech  = outer[:, :, self.ti, self.tj]              # (B, N, d*(d+1)//2)
        sup_in  = torch.cat([X_tr, vech], dim=-1)          # (B, N, p + d_vech)
        sup_emb = self.enc_sup(sup_in)                     # (B, N, d_h)

        # ── Query encoding ────────────────────────────────────────────────────
        qry_emb = self.enc_qry(X_te)                       # (B, n_q, d_h)

        # ── Cross-attention: each query reads relevant support instances ──────
        ctx, attn_w = self.cross_attn(
            qry_emb, sup_emb, sup_emb,
            need_weights=True, average_attn_weights=True,
        )                                                   # ctx: (B, n_q, d_h)
        ctx = self.post_norm(ctx + qry_emb)                # residual

        # ── Cholesky readout ──────────────────────────────────────────────────
        head = torch.cat([ctx, qry_emb], dim=-1)           # (B, n_q, 2*d_h)
        L_flat = self.readout_L(head)                      # (B, n_q, d*(d+1)//2)

        L = torch.zeros(B, n_q, d, d, device=X_tr.device, dtype=X_tr.dtype)
        L[:, :, self.ti, self.tj] = L_flat
        L[:, :, self.diag_idx, self.diag_idx] = (
            F.softplus(L[:, :, self.diag_idx, self.diag_idx]) + 1e-4
        )
        Sigma_raw = L @ L.transpose(-2, -1)
        std = Sigma_raw.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_pred = Sigma_raw / (std.unsqueeze(-1) * std.unsqueeze(-2))

        return R_pred, attn_w


# ── Training ─────────────────────────────────────────────────────────────────

def train(model, device):
    ri, ci    = torch.triu_indices(D, D, offset=1, device=device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    steps_log, mse_log, div_log, h_log = [], [], [], []

    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Training {N_STEPS} steps  |  fresh episode each step  |  MSE-only")
    print(f"{'─'*60}")
    print(f"  {'step':>5}  {'MSE':>8}  {'div':>8}  {'H_norm':>8}")

    model.train()
    for step in range(N_STEPS):
        X_tr, Y_tr, X_te, _, R_ora = sample_ep(device)

        optimizer.zero_grad()
        R_pred, attn_w = model(X_tr, Y_tr, X_te)
        loss = F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse = loss.item()
                div = R_pred[0, :, ri, ci].std(dim=0).mean().item()
                w   = attn_w.clamp(min=1e-10)
                H   = -(w * w.log()).sum(-1).mean().item()
                h   = H / math.log(max(N_TRAIN, 2))

            steps_log.append(step); mse_log.append(mse)
            div_log.append(div);    h_log.append(h)
            print(f"  {step:>5d}  {mse:>8.5f}  {div:>8.5f}  {h:>8.4f}")

    return steps_log, mse_log, div_log, h_log


# ── Validation: average MSE over many fresh episodes ─────────────────────────

@torch.no_grad()
def validate(model, device):
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()
    mses = []
    for _ in range(VAL_EPISODES):
        X_tr, Y_tr, X_te, _, R_ora = sample_ep(device)
        R_pred, _ = model(X_tr, Y_tr, X_te)
        mses.append(F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item())
    return float(np.mean(mses))


# ── ICL test: support swap ────────────────────────────────────────────────────

@torch.no_grad()
def icl_test(model, device):
    """
    Generate two episodes A and B.
    Baseline:  ep-A support + ep-A queries  → R_pred_AA
    Swap:      ep-B support + ep-A queries  → R_pred_BA
    Zero-supp: zero support                 → R_pred_0
    ZeroX_te:  ep-A support, X_test=0      → R_pred_A0  (diversity check)

    True ICL: |R_pred_AA - R_pred_BA| >> 0  (support swap changes predictions)
              div(R_pred_A0) ≈ 0              (X_test drives per-query diversity)
    """
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()

    Xa_tr, Ya_tr, Xa_te, _, Ra = sample_ep(device)
    Xb_tr, Yb_tr, _, _, _      = sample_ep(device)

    def mse(R_pred, R_ora):
        return F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item()
    def div(R_pred):
        return R_pred[0, :, ri, ci].std(dim=0).mean().item()

    Raa, attn_aa = model(Xa_tr, Ya_tr, Xa_te)           # baseline
    Rba, attn_ba = model(Xb_tr, Yb_tr, Xa_te)           # support swap
    Rza, _       = model(torch.zeros_like(Xa_tr),        # zero support
                         torch.zeros_like(Ya_tr), Xa_te)
    Ra0, _       = model(Xa_tr, Ya_tr,                   # zero X_test
                         torch.zeros_like(Xa_te))

    chg_swap = (Rba - Raa).abs().mean().item()
    chg_zero = (Rza - Raa).abs().mean().item()
    chg_Xte0 = (Ra0 - Raa).abs().mean().item()

    # Attention entropy
    def h_norm(w):
        w = w.clamp(min=1e-10)
        H = -(w * w.log()).sum(-1).mean().item()
        return H / math.log(max(N_TRAIN, 2))

    print(f"\n{'═'*60}")
    print(f"ICL TEST")
    print(f"{'═'*60}")
    print(f"  Baseline    MSE={mse(Raa,Ra):.5f}  div={div(Raa):.4f}  "
          f"H_norm={h_norm(attn_aa):.4f}")
    print(f"  Supp swap   MSE={mse(Rba,Ra):.5f}  div={div(Rba):.4f}  "
          f"H_norm={h_norm(attn_ba):.4f}  pred_chg={chg_swap:.5f}")
    print(f"  Zero supp   MSE={mse(Rza,Ra):.5f}  div={div(Rza):.4f}  "
          f"pred_chg={chg_zero:.5f}")
    print(f"  Zero X_te   MSE={mse(Ra0,Ra):.5f}  div={div(Ra0):.4f}  "
          f"pred_chg={chg_Xte0:.5f}")
    print()

    doing_icl = chg_swap > 0.02 and div(Ra0) < 0.01
    x_regress = chg_swap < 0.005 and div(Ra0) < 0.01
    both      = chg_swap > 0.02 and div(Ra0) > 0.01

    if doing_icl:
        verdict = "TRUE ICL: support swap changes predictions, zero-X_test collapses div"
    elif x_regress:
        verdict = "X REGRESSION: support ignored, predictions from X only"
    elif both:
        verdict = "HYBRID: both support context and X features contribute"
    else:
        verdict = f"UNCLEAR: chg_swap={chg_swap:.4f} div_Xte0={div(Ra0):.4f}"

    print(f"  VERDICT: {verdict}")
    return {
        "mse_base": mse(Raa, Ra), "mse_swap": mse(Rba, Ra),
        "chg_swap": chg_swap, "chg_zero": chg_zero,
        "div_base": div(Raa), "div_Xte0": div(Ra0),
        "h_base": h_norm(attn_aa), "h_swap": h_norm(attn_ba),
        "verdict": verdict,
        "Raa": Raa, "Rba": Rba, "Ra": Ra,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  p={P} d={D} r={R}  N={N_TRAIN} n_test={N_TEST}")

    torch.manual_seed(0)
    model = ICLCorrNet(p=P, d=D, d_h=D_HIDDEN, n_heads=N_HEADS).to(device)

    steps_log, mse_log, div_log, h_log = train(model, device)

    val_mse = validate(model, device)
    print(f"\nVal MSE ({VAL_EPISODES} fresh episodes): {val_mse:.5f}")

    result = icl_test(model, device)

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps_log, mse_log, lw=1.5, color="steelblue")
    axes[0].axhline(0.02, color="green", ls="--", lw=1, label="target 0.02")
    axes[0].set_title("Training MSE (MSE-only)"); axes[0].set_xlabel("Step")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, div_log, lw=1.5, color="purple", label="diversity")
    axes[1].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[1].set_title("Prediction diversity (std across queries)")
    axes[1].set_xlabel("Step")

    axes[2].plot(steps_log, h_log, lw=1.5, color="tomato")
    axes[2].axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6, label="1=uniform")
    axes[2].axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6, label="0=peaked")
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_title("Cross-attention H_norm")
    axes[2].set_xlabel("Step"); axes[2].legend(fontsize=8)

    fig.suptitle(
        f"ICLCorrNet — val_MSE={val_mse:.4f}  [{result['verdict'][:40]}]",
        fontsize=10
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "icl_corr_net_training.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Correlation grid: baseline vs support-swap
    v_norms  = result["Ra"][0].diagonal(dim1=-2, dim2=-1)  # proxy for group ordering
    n_test_  = result["Raa"].shape[1]
    idx      = torch.arange(n_test_)

    fig_g = plot_corr_grid(
        estimators={
            "Baseline (ep-A supp)": result["Raa"][0, idx].cpu(),
            "Swap (ep-B supp)":     result["Rba"][0, idx].cpu(),
        },
        oracle_R=result["Ra"][0, idx].cpu(),
        n_instances=n_test_,
        title=f"ICLCorrNet — support swap test\n{result['verdict']}",
    )
    path2 = OUTPUT_DIR / "icl_corr_net_swap.png"
    fig_g.savefig(path2, dpi=100, bbox_inches="tight")
    plt.close(fig_g)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
