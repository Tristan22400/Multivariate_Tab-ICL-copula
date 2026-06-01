"""
icl_corr_net3.py — ICL copula net v3: deeper, more capacity, longer training.

Changes from v2
---------------
  d_h = 256, 2 cross-attn layers (stacked), 10k steps, cosine LR decay.
  Also adds a "mean baseline" comparison: predict the global mean R across
  all episodes so we can see how much the model improves over that floor.

Usage
-----
  conda run -n multivariate-icl python debug/icl_corr_net3.py
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
from torch.optim.lr_scheduler import CosineAnnealingLR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_gen import generate_episode
from viz import plot_corr_grid

OUTPUT_DIR = ROOT / "debug" / "icl_test_results"

P, D, R      = 8, 6, 4
N_TRAIN      = 128
N_TEST       = 16
BATCH_SIZE   = 16
D_HIDDEN     = 256
N_HEADS      = 8
N_LAYERS     = 2          # stacked cross-attn layers
N_STEPS      = 10_000
LR           = 3e-4
GRAD_CLIP    = 1.0
LOG_EVERY    = 1_000
VAL_EPISODES = 100


def cov_to_corr(D, V):
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def sample_ep(device, B=1):
    X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
        B=B, p=P, d=D, r=R,
        n_train=N_TRAIN, n_test=N_TEST,
        device=device,
        hyperplane_bimodal=True,
        return_oracle=True,
    )
    return X_tr, Y_tr, X_te, cov_to_corr(oracle["D"], oracle["V"])


class CrossAttnLayer(nn.Module):
    """One cross-attention layer: Q from query, K from support-X, V from support-Y."""

    def __init__(self, d_h, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_h // n_heads
        self.scale   = self.d_head ** -0.5
        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h)
        self.norm1 = nn.LayerNorm(d_h)
        self.norm2 = nn.LayerNorm(d_h)
        self.ff = nn.Sequential(
            nn.Linear(d_h, d_h * 2), nn.GELU(), nn.Linear(d_h * 2, d_h)
        )

    def forward(self, Q_in, K_in, V_in):
        B, n_q, _ = Q_in.shape
        N = K_in.shape[1]
        H, Dh = self.n_heads, self.d_head

        Q = self.W_q(Q_in).view(B, n_q, H, Dh).transpose(1, 2)
        K = self.W_k(K_in).view(B, N,   H, Dh).transpose(1, 2)
        V = self.W_v(V_in).view(B, N,   H, Dh).transpose(1, 2)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_w  = F.softmax(scores, dim=-1)
        ctx     = torch.matmul(attn_w, V).transpose(1, 2).reshape(B, n_q, -1)
        ctx     = self.norm1(Q_in + self.W_o(ctx))
        ctx     = self.norm2(ctx + self.ff(ctx))
        return ctx, attn_w.mean(dim=1)   # (B, n_q, d_h), (B, n_q, N)


class ICLCorrNet3(nn.Module):
    def __init__(self, p, d, d_h=256, n_heads=8, n_layers=2):
        super().__init__()
        self.d = d
        d_vech = d * (d + 1) // 2

        def mlp(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.GELU(),
                nn.Linear(d_h, d_out), nn.LayerNorm(d_out),
            )

        self.enc_qry = mlp(p,      d_h)
        self.enc_key = mlp(p,      d_h)
        self.enc_val = mlp(d_vech, d_h)

        self.layers = nn.ModuleList(
            [CrossAttnLayer(d_h, n_heads) for _ in range(n_layers)]
        )

        d_L = d * (d + 1) // 2
        self.readout_L = nn.Sequential(
            nn.Linear(d_h * 2, d_h), nn.GELU(),
            nn.Linear(d_h,     d_L),
        )

        ti, tj = torch.tril_indices(d, d)
        self.register_buffer("ti", ti)
        self.register_buffer("tj", tj)
        self.register_buffer("diag_idx", torch.arange(d))

    def forward(self, X_tr, Y_tr, X_te):
        B, N, _ = X_tr.shape
        d = self.d

        outer = Y_tr.unsqueeze(-1) * Y_tr.unsqueeze(-2)
        vech  = outer[:, :, self.ti, self.tj]

        Q = self.enc_qry(X_te)
        K = self.enc_key(X_tr)
        V = self.enc_val(vech)

        attn_last = None
        ctx = Q
        for layer in self.layers:
            ctx, attn_last = layer(ctx, K, V)

        L_flat = self.readout_L(torch.cat([ctx, Q], dim=-1))
        L = torch.zeros(B, Q.shape[1], d, d, device=X_tr.device, dtype=X_tr.dtype)
        L[:, :, self.ti, self.tj] = L_flat
        L[:, :, self.diag_idx, self.diag_idx] = (
            F.softplus(L[:, :, self.diag_idx, self.diag_idx]) + 1e-4
        )
        Sigma_raw = L @ L.transpose(-2, -1)
        std = Sigma_raw.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_pred = Sigma_raw / (std.unsqueeze(-1) * std.unsqueeze(-2))
        return R_pred, attn_last


def compute_mean_baseline(device, n=200):
    """MSE of always predicting the identity matrix (independence baseline)."""
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    mses = []
    for _ in range(n):
        _, _, _, R_ora = sample_ep(device, B=1)
        I = torch.eye(D, device=device).unsqueeze(0).unsqueeze(0).expand_as(R_ora)
        mses.append(F.mse_loss(I[:, :, ri, ci], R_ora[:, :, ri, ci]).item())
    return float(np.mean(mses))


def train(model, device):
    ri, ci    = torch.triu_indices(D, D, offset=1, device=device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=LR * 0.1)
    steps_log, mse_log, div_log, h_log = [], [], [], []

    print(f"params={sum(p.numel() for p in model.parameters()):,}  "
          f"d_h={D_HIDDEN}  layers={N_LAYERS}  B={BATCH_SIZE}  {N_STEPS} steps")
    print(f"{'─'*65}")
    print(f"  {'step':>6}  {'MSE':>8}  {'div':>8}  {'H_norm':>8}  {'LR':>10}")

    model.train()
    for step in range(N_STEPS):
        X_tr, Y_tr, X_te, R_ora = sample_ep(device, B=BATCH_SIZE)
        optimizer.zero_grad()
        R_pred, attn_w = model(X_tr, Y_tr, X_te)
        loss = F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse = loss.item()
                div = R_pred[:, :, ri, ci].std(dim=1).mean().item()
                w   = attn_w.clamp(min=1e-10)
                h   = (-(w * w.log()).sum(-1).mean() / math.log(max(N_TRAIN,2))).item()
                lr  = optimizer.param_groups[0]["lr"]
            steps_log.append(step); mse_log.append(mse)
            div_log.append(div);    h_log.append(h)
            print(f"  {step:>6d}  {mse:>8.5f}  {div:>8.5f}  {h:>8.4f}  {lr:>10.2e}")

    return steps_log, mse_log, div_log, h_log


@torch.no_grad()
def validate(model, device):
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()
    mses = []
    for _ in range(VAL_EPISODES):
        X_tr, Y_tr, X_te, R_ora = sample_ep(device, B=1)
        R_pred, _ = model(X_tr, Y_tr, X_te)
        mses.append(F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item())
    return float(np.mean(mses))


@torch.no_grad()
def icl_test(model, device):
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()

    Xa_tr, Ya_tr, Xa_te, Ra = sample_ep(device, B=1)
    Xb_tr, Yb_tr, _, _      = sample_ep(device, B=1)

    def mse(Rp, Ro): return F.mse_loss(Rp[:, :, ri, ci], Ro[:, :, ri, ci]).item()
    def div(Rp):     return Rp[0, :, ri, ci].std(dim=0).mean().item()
    def hn(w):
        w = w.clamp(min=1e-10)
        return (-(w*w.log()).sum(-1).mean() / math.log(max(N_TRAIN,2))).item()

    Raa, waa = model(Xa_tr, Ya_tr, Xa_te)
    Rba, wba = model(Xb_tr, Yb_tr, Xa_te)
    Rza, _   = model(torch.zeros_like(Xa_tr), torch.zeros_like(Ya_tr), Xa_te)
    Ra0, _   = model(Xa_tr, Ya_tr, torch.zeros_like(Xa_te))

    chg_swap = (Rba - Raa).abs().mean().item()
    chg_zero = (Rza - Raa).abs().mean().item()

    print(f"\n{'═'*65}")
    print(f"ICL TEST (v3)")
    print(f"{'═'*65}")
    print(f"  Baseline   MSE={mse(Raa,Ra):.5f}  div={div(Raa):.4f}  H={hn(waa):.4f}")
    print(f"  Supp swap  MSE={mse(Rba,Ra):.5f}  div={div(Rba):.4f}  "
          f"H={hn(wba):.4f}  chg={chg_swap:.5f}")
    print(f"  Zero supp  MSE={mse(Rza,Ra):.5f}  div={div(Rza):.4f}  chg={chg_zero:.5f}")
    print(f"  Zero Xte   MSE={mse(Ra0,Ra):.5f}  div={div(Ra0):.4f}")

    doing_icl = chg_swap > 0.02
    print(f"\n  VERDICT: {'TRUE ICL' if doing_icl else 'X REGRESSION'}  "
          f"chg_swap={chg_swap:.4f}")
    return {"Raa": Raa, "Rba": Rba, "Ra": Ra,
            "chg_swap": chg_swap, "verdict": "TRUE ICL" if doing_icl else "X REGRESSION"}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  p={P} d={D} r={R}  N={N_TRAIN}")

    print("Computing independence baseline MSE...")
    baseline_mse = compute_mean_baseline(device)
    print(f"Independence baseline MSE: {baseline_mse:.5f}")

    torch.manual_seed(2)
    model = ICLCorrNet3(p=P, d=D, d_h=D_HIDDEN, n_heads=N_HEADS,
                        n_layers=N_LAYERS).to(device)

    steps_log, mse_log, div_log, h_log = train(model, device)

    val_mse = validate(model, device)
    print(f"\nVal MSE ({VAL_EPISODES} episodes): {val_mse:.5f}  "
          f"(baseline: {baseline_mse:.5f}  "
          f"improvement: {(1-val_mse/baseline_mse)*100:.1f}%)")

    result = icl_test(model, device)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps_log, mse_log, lw=1.5, color="steelblue", label="train MSE")
    axes[0].axhline(val_mse,      color="green",  ls="--", lw=1.2,
                    label=f"val MSE={val_mse:.3f}")
    axes[0].axhline(baseline_mse, color="gray",   ls=":",  lw=1.2,
                    label=f"indep baseline={baseline_mse:.3f}")
    axes[0].set_title("MSE"); axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, div_log, lw=1.5, color="purple")
    axes[1].set_title("Prediction diversity (std across queries)")

    axes[2].plot(steps_log, h_log, lw=1.5, color="tomato")
    axes[2].axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6)
    axes[2].axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6)
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_title("Cross-attn H_norm  (0=peaked, 1=uniform)")

    fig.suptitle(
        f"ICLCorrNet v3 — val={val_mse:.4f}  base={baseline_mse:.4f}  "
        f"[{result['verdict']}  chg={result['chg_swap']:.3f}]",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "icl_corr_net3_training.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    n_test_ = result["Raa"].shape[1]
    idx = torch.arange(n_test_)
    fig_g = plot_corr_grid(
        estimators={
            "Base (ep-A supp)": result["Raa"][0, idx].cpu(),
            "Swap (ep-B supp)": result["Rba"][0, idx].cpu(),
        },
        oracle_R=result["Ra"][0, idx].cpu(),
        n_instances=n_test_,
        title=f"ICLCorrNet v3 — support swap  [{result['verdict']}]",
    )
    fig_g.savefig(OUTPUT_DIR / "icl_corr_net3_swap.png", dpi=100, bbox_inches="tight")
    plt.close(fig_g)
    print(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
