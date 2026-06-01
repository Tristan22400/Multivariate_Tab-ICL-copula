"""
icl_corr_net2.py — ICL copula net v2: separated key/value encoders + more capacity.

Key change from v1
------------------
  v1: enc_sup(X_i, vech(Y_i⊗Y_i)) → single embedding used for both K and V.
      Attention scores mix X-position and Y-correlation signals, making it
      hard to simultaneously learn X-peaked attention and Y-content values.

  v2: enc_key(X_i)                → K   (X-position only, for peaked attention)
      enc_val(vech(Y_i⊗Y_i))      → V   (Y-correlation content only)
      enc_qry(X_q)                → Q   (X-position only, matches K space)

  This cleanly separates:
    - "who to attend to" (same-X-group → same-hyperplane side)   via Q·K^T
    - "what to read"    (correlation evidence from Y⊗Y values)   via V

Also: larger model (d_h=128), more steps (10k), lower LR (1e-4).

Usage
-----
  conda run -n multivariate-icl python debug/icl_corr_net2.py
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
from data_gen import generate_episode
from viz import plot_corr_grid

OUTPUT_DIR = ROOT / "debug" / "icl_test_results"

P, D, R      = 8, 6, 4
N_TRAIN      = 128
N_TEST       = 16
BATCH_SIZE   = 8          # episodes per step — fills GPU, amortises dispatch overhead
D_HIDDEN     = 128
N_HEADS      = 4
N_STEPS      = 5_000
LR           = 3e-4
GRAD_CLIP    = 1.0
LOG_EVERY    = 500
VAL_EPISODES = 50


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
    R_ora = cov_to_corr(oracle["D"], oracle["V"])
    return X_tr, Y_tr, X_te, Y_te, R_ora


class ICLCorrNet2(nn.Module):
    """
    Separated key/value encoders for clean ICL:
      Q = enc_qry(X_q)          → position-based query
      K = enc_key(X_i)          → position-based key  (X similarity for peaked attn)
      V = enc_val(vech(Y_i⊗Yi)) → correlation-content value
    """

    def __init__(self, p, d, d_h=128, n_heads=4):
        super().__init__()
        self.d = d
        d_vech = d * (d + 1) // 2

        def mlp(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.GELU(),
                nn.Linear(d_h, d_out), nn.LayerNorm(d_out),
            )

        self.enc_qry = mlp(p,      d_h)   # Q
        self.enc_key = mlp(p,      d_h)   # K  (X-based, matches Q space)
        self.enc_val = mlp(d_vech, d_h)   # V  (Y-correlation content)

        # Manual multi-head attention with separate K/V encoders
        self.n_heads = n_heads
        self.d_head  = d_h // n_heads
        self.scale   = self.d_head ** -0.5
        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h)
        self.post_norm = nn.LayerNorm(d_h)

        d_L = d * (d + 1) // 2
        self.readout_L = nn.Sequential(
            nn.Linear(d_h * 2, d_h), nn.GELU(),
            nn.Linear(d_h, d_L),
        )

        ti, tj = torch.tril_indices(d, d)
        self.register_buffer("ti", ti)
        self.register_buffer("tj", tj)
        self.register_buffer("diag_idx", torch.arange(d))

    def forward(self, X_tr, Y_tr, X_te):
        B, N, p = X_tr.shape
        n_q = X_te.shape[1]
        d   = self.d
        H, Dh = self.n_heads, self.d_head

        # ── Encode ────────────────────────────────────────────────────────────
        outer = Y_tr.unsqueeze(-1) * Y_tr.unsqueeze(-2)   # (B, N, d, d)
        vech  = outer[:, :, self.ti, self.tj]             # (B, N, d_vech)

        Q_in = self.enc_qry(X_te)                         # (B, n_q, d_h)
        K_in = self.enc_key(X_tr)                         # (B, N, d_h)  X only
        V_in = self.enc_val(vech)                         # (B, N, d_h)  Y⊗Y only

        # ── Multi-head attention (Q from query-X, K from support-X, V from Y⊗Y) ──
        Q = self.W_q(Q_in).view(B, n_q, H, Dh).transpose(1, 2)  # (B,H,n_q,Dh)
        K = self.W_k(K_in).view(B, N,   H, Dh).transpose(1, 2)  # (B,H,N,Dh)
        V = self.W_v(V_in).view(B, N,   H, Dh).transpose(1, 2)  # (B,H,N,Dh)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,H,n_q,N)
        attn_w  = F.softmax(scores, dim=-1)                           # (B,H,n_q,N)
        ctx_mh  = torch.matmul(attn_w, V)                            # (B,H,n_q,Dh)
        ctx     = self.W_o(ctx_mh.transpose(1, 2).reshape(B, n_q, -1))
        ctx     = self.post_norm(ctx + Q_in)

        attn_mean = attn_w.mean(dim=1)                               # (B,n_q,N)

        # ── Cholesky readout ──────────────────────────────────────────────────
        L_flat = self.readout_L(torch.cat([ctx, Q_in], dim=-1))     # (B,n_q,d_L)
        L = torch.zeros(B, n_q, d, d, device=X_tr.device, dtype=X_tr.dtype)
        L[:, :, self.ti, self.tj] = L_flat
        L[:, :, self.diag_idx, self.diag_idx] = (
            F.softplus(L[:, :, self.diag_idx, self.diag_idx]) + 1e-4
        )
        Sigma_raw = L @ L.transpose(-2, -1)
        std = Sigma_raw.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_pred = Sigma_raw / (std.unsqueeze(-1) * std.unsqueeze(-2))

        return R_pred, attn_mean


def train(model, device):
    ri, ci    = torch.triu_indices(D, D, offset=1, device=device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps_log, mse_log, div_log, h_log = [], [], [], []

    print(f"params={sum(p.numel() for p in model.parameters()):,}  "
          f"d_h={D_HIDDEN}  sep-KV  {N_STEPS} steps")
    print(f"{'─'*60}")
    print(f"  {'step':>6}  {'MSE':>8}  {'div':>8}  {'H_norm':>8}")

    model.train()
    for step in range(N_STEPS):
        X_tr, Y_tr, X_te, _, R_ora = sample_ep(device, B=BATCH_SIZE)
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
                h   = (-(w * w.log()).sum(-1).mean() / math.log(max(N_TRAIN,2))).item()
            steps_log.append(step); mse_log.append(mse)
            div_log.append(div);    h_log.append(h)
            print(f"  {step:>6d}  {mse:>8.5f}  {div:>8.5f}  {h:>8.4f}")

    return steps_log, mse_log, div_log, h_log


@torch.no_grad()
def validate(model, device):
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()
    mses = []
    for _ in range(VAL_EPISODES):
        X_tr, Y_tr, X_te, _, R_ora = sample_ep(device, B=1)
        R_pred, _ = model(X_tr, Y_tr, X_te)
        mses.append(F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item())
    return float(np.mean(mses))


@torch.no_grad()
def icl_test(model, device):
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()

    Xa_tr, Ya_tr, Xa_te, _, Ra = sample_ep(device)
    Xb_tr, Yb_tr, _,     _, _  = sample_ep(device)

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

    print(f"\n{'═'*60}")
    print(f"ICL TEST (v2 — separated K/V)")
    print(f"{'═'*60}")
    print(f"  Baseline   MSE={mse(Raa,Ra):.5f}  div={div(Raa):.4f}  H={hn(waa):.4f}")
    print(f"  Supp swap  MSE={mse(Rba,Ra):.5f}  div={div(Rba):.4f}  "
          f"H={hn(wba):.4f}  chg={chg_swap:.5f}")
    print(f"  Zero supp  MSE={mse(Rza,Ra):.5f}  div={div(Rza):.4f}  chg={chg_zero:.5f}")
    print(f"  Zero Xte   MSE={mse(Ra0,Ra):.5f}  div={div(Ra0):.4f}")

    doing_icl = chg_swap > 0.02
    verdict = ("TRUE ICL — support swap changes predictions"
               if doing_icl else "X REGRESSION — support ignored")
    print(f"\n  VERDICT: {verdict}")
    return {"Raa": Raa, "Rba": Rba, "Ra": Ra,
            "chg_swap": chg_swap, "val_mse": None, "verdict": verdict}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  p={P} d={D} r={R}  N={N_TRAIN}")

    torch.manual_seed(1)
    model = ICLCorrNet2(p=P, d=D, d_h=D_HIDDEN, n_heads=N_HEADS).to(device)

    steps_log, mse_log, div_log, h_log = train(model, device)

    val_mse = validate(model, device)
    print(f"\nVal MSE ({VAL_EPISODES} episodes): {val_mse:.5f}")

    result = icl_test(model, device)
    result["val_mse"] = val_mse

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps_log, mse_log, lw=1.5, color="steelblue")
    axes[0].axhline(0.05, color="green", ls="--", lw=1, label="target 0.05")
    axes[0].set_title("Training MSE"); axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, div_log, lw=1.5, color="purple")
    axes[1].set_title("Prediction diversity")

    axes[2].plot(steps_log, h_log, lw=1.5, color="tomato")
    axes[2].axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6)
    axes[2].axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6)
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_title("Cross-attn H_norm  (0=peaked=ICL)")

    fig.suptitle(f"ICLCorrNet v2 — val_MSE={val_mse:.4f}  [{result['verdict'][:45]}]",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "icl_corr_net2_training.png", dpi=120, bbox_inches="tight")
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
        title=f"ICLCorrNet v2 — swap test\n{result['verdict']}",
    )
    fig_g.savefig(OUTPUT_DIR / "icl_corr_net2_swap.png", dpi=100, bbox_inches="tight")
    plt.close(fig_g)
    print(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
