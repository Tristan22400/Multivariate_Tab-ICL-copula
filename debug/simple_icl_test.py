"""
simple_icl_test.py — Does CrossAttnCopulaNet do ICL or just X regression?

Same three-test protocol as icl_test.py but on the simple model:

  Test 1 — Support swap: ep1 support + ep0 test queries
  Test 2 — Zero support: X_train=0, Z_train=0
  Test 3 — Zero X_test: support intact, X_test=0

Key difference from CopulaTabICLv2: the simple model includes vech(Z⊗Z) in the
support encoder, so Z_train's correlation structure is explicitly encoded.

Usage:
    conda run -n multivariate-icl python debug/simple_icl_test.py
"""

from __future__ import annotations

import os, sys
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
from viz import plot_corr_grid  # noqa: E402

DATA_DIR   = ROOT / "data" / "pit_hyperplane_debug"
OUTPUT_DIR = ROOT / "debug" / "icl_test_results"
N_TRAIN    = 3_000
LR         = 1e-3
GRAD_CLIP  = 1.0


# Paste CrossAttnCopulaNet here (same as overfit_correlation.py)
class CrossAttnCopulaNet(nn.Module):
    def __init__(self, p_max, d_max, rank_max, d_hidden, n_heads=4, output="woodbury"):
        super().__init__()
        self.p_max = p_max; self.d_max = d_max; self.rank_max = rank_max
        self.d_hidden = d_hidden; self.output = output

        d_vech   = d_max * (d_max + 1) // 2
        d_sup_in = p_max + d_max + d_vech

        self.enc_sup = nn.Sequential(
            nn.Linear(d_sup_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
        )
        self.enc_qry = nn.Sequential(
            nn.Linear(p_max,    d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
        )
        self.cross_attn     = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.post_attn_norm = nn.LayerNorm(d_hidden)
        self.dim_emb = nn.Parameter(torch.randn(d_max, d_hidden // 4))

        d_head = d_hidden + d_hidden + d_hidden // 4
        self.readout = nn.Sequential(
            nn.Linear(d_head, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, rank_max),
        )

    def _encode(self, X_all, Z_all, n_support):
        B, N_total, p = X_all.shape
        d = Z_all.shape[-1]

        Xp = F.pad(X_all, (0, self.p_max - p)) if p < self.p_max else X_all[..., :self.p_max]
        Zs = Z_all[:, :n_support, :]
        Zsp = F.pad(Zs, (0, self.d_max - d)) if d < self.d_max else Zs[:, :, :self.d_max]
        outer = Zsp.unsqueeze(-1) * Zsp.unsqueeze(-2)
        ti, tj = torch.tril_indices(self.d_max, self.d_max, device=X_all.device)
        vech = outer[:, :, ti, tj]
        sup_in  = torch.cat([Xp[:, :n_support], Zsp, vech], -1)
        sup_emb = self.enc_sup(sup_in)
        qry_emb = self.enc_qry(Xp[:, n_support:])
        ctx, attn_w = self.cross_attn(qry_emb, sup_emb, sup_emb,
                                      need_weights=True, average_attn_weights=True)
        ctx = self.post_attn_norm(ctx + qry_emb)
        return ctx, qry_emb, attn_w, d

    def forward(self, X_all, Z_all, n_support):
        ctx, qry_emb, attn_w, d = self._encode(X_all, Z_all, n_support)
        B, n_q, _ = ctx.shape
        ctx_exp = ctx.unsqueeze(2).expand(B, n_q, self.d_max, -1)
        qry_exp = qry_emb.unsqueeze(2).expand(B, n_q, self.d_max, -1)
        dim_exp = self.dim_emb.unsqueeze(0).unsqueeze(0).expand(B, n_q, -1, -1)
        U = self.readout(torch.cat([ctx_exp, qry_exp, dim_exp], -1))
        U = U[:, :, :d, :]
        C = 1.0 / (1.0 + (U**2).sum(-1))
        W = U  / (1.0 + (U**2).sum(-1).unsqueeze(-1)).sqrt()
        return torch.diag_embed(C) + W @ W.transpose(-2, -1), attn_w


def _cov_to_corr(D, V):
    S = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def load_ep(idx, device):
    ep = torch.load(DATA_DIR / f"episode_{idx:06d}.pt", weights_only=True)
    b  = 0
    tensor_keys = [k for k, v in ep.items() if isinstance(v, torch.Tensor)]
    return {k: ep[k][[b]].float().to(device) for k in tensor_keys}


def train(model, X_fwd, Z_fwd, R_ora, N, d, device):
    ri, ci = torch.triu_indices(d, d, offset=1, device=device)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    model.train()
    for _ in range(N_TRAIN):
        opt.zero_grad()
        Sp, _ = model(X_fwd, Z_fwd, n_support=N)
        F.mse_loss(Sp[:, :, ri, ci], R_ora[:, :, ri, ci]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    model.eval()


def infer(model, X_fwd, Z_fwd, N):
    with torch.no_grad():
        Sp, _ = model(X_fwd, Z_fwd, n_support=N)
    return Sp


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ep0 = load_ep(0, device)
    ep1 = load_ep(1, device)

    X0t = ep0["X_train"]; Z0t = ep0["Z_train"]
    X0q = ep0["X_test"];  Z0q = ep0["Z_test"]
    R0  = _cov_to_corr(ep0["oracle_D"], ep0["oracle_V"])
    R1  = _cov_to_corr(ep1["oracle_D"], ep1["oracle_V"])

    N, n_test, d = X0t.shape[1], X0q.shape[1], Z0t.shape[2]
    p = X0t.shape[2]
    rank = ep0["oracle_V"].shape[-1]

    torch.manual_seed(42)
    model = CrossAttnCopulaNet(p_max=p, d_max=d, rank_max=rank,
                               d_hidden=128, n_heads=4).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Training {N_TRAIN} steps MSE-only on episode 0...")

    X0_fwd = torch.cat([X0t, X0q], dim=1)
    Z0_fwd = torch.cat([Z0t, Z0q], dim=1)
    train(model, X0_fwd, Z0_fwd, R0, N, d, device)

    ri, ci = torch.triu_indices(d, d, offset=1, device=device)
    def mse(Sp, R): return F.mse_loss(Sp[:, :, ri, ci], R[:, :, ri, ci]).item()
    def div(Sp): return Sp[0, :, ri, ci].std(dim=0).mean().item()

    # Baseline
    Sp_base = infer(model, X0_fwd, Z0_fwd, N)
    print(f"\n[Baseline] ep0 support + ep0 test → MSE={mse(Sp_base,R0):.5f}  div={div(Sp_base):.4f}")

    # Test 1: Support swap (ep1 support, ep0 queries)
    X1_fwd = torch.cat([ep1["X_train"], X0q], dim=1)
    Z1_fwd = torch.cat([ep1["Z_train"], Z0q], dim=1)
    Sp_swap = infer(model, X1_fwd, Z1_fwd, N)
    chg_swap = (Sp_swap - Sp_base).abs().mean().item()
    print(f"[Test 1 — support swap] ep1 support + ep0 test:")
    print(f"  MSE vs ep0 oracle={mse(Sp_swap,R0):.5f}  div={div(Sp_swap):.4f}  "
          f"pred_change={chg_swap:.6f}")

    # Test 2: Zero support
    Xz_fwd = torch.cat([torch.zeros_like(X0t), X0q], dim=1)
    Zz_fwd = torch.cat([torch.zeros_like(Z0t), Z0q], dim=1)
    Sp_zero = infer(model, Xz_fwd, Zz_fwd, N)
    chg_zero = (Sp_zero - Sp_base).abs().mean().item()
    print(f"[Test 2 — zero support] X_train=0, Z_train=0:")
    print(f"  MSE vs ep0 oracle={mse(Sp_zero,R0):.5f}  div={div(Sp_zero):.4f}  "
          f"pred_change={chg_zero:.6f}")

    # Test 3: Zero X_test (can support alone differentiate queries?)
    X0q_z_fwd = torch.cat([X0t, torch.zeros_like(X0q)], dim=1)
    Sp_noXq   = infer(model, X0q_z_fwd, Z0_fwd, N)
    chg_noXq  = (Sp_noXq - Sp_base).abs().mean().item()
    print(f"[Test 3 — zero X_test] support intact, X_test=0:")
    print(f"  MSE vs ep0 oracle={mse(Sp_noXq,R0):.5f}  div={div(Sp_noXq):.4f}  "
          f"pred_change={chg_noXq:.6f}")

    # Test 4 (extra): Correct Z_train for ep0, wrong Z_train (random)
    Z_rand = torch.randn_like(Z0t)
    Xr_fwd = torch.cat([X0t, X0q], dim=1)
    Zr_fwd = torch.cat([Z_rand, Z0q], dim=1)
    Sp_randZ = infer(model, Xr_fwd, Zr_fwd, N)
    chg_randZ = (Sp_randZ - Sp_base).abs().mean().item()
    print(f"[Test 4 — random Z_train] X_train correct, Z_train=random:")
    print(f"  MSE vs ep0 oracle={mse(Sp_randZ,R0):.5f}  div={div(Sp_randZ):.4f}  "
          f"pred_change={chg_randZ:.6f}")

    print(f"\n{'='*60}")
    print(f"SimpleCopulaNet vs CopulaTabICLv2 — ICL capability")
    print(f"{'='*60}")
    print(f"  {'Condition':<25}  {'SimpleCopula':>14}  {'TabICLv2 (ref)':>16}")
    print(f"  {'─'*57}")
    tabicl_ref = {
        "Baseline MSE":      0.00007,
        "Support swap chg":  0.068,
        "Zero supp chg":     0.119,
        "Zero X_test div":   0.000,
    }
    simple = {
        "Baseline MSE":      mse(Sp_base, R0),
        "Support swap chg":  chg_swap,
        "Zero supp chg":     chg_zero,
        "Zero X_test div":   div(Sp_noXq),
    }
    for k in tabicl_ref:
        print(f"  {k:<25}  {simple[k]:>14.5f}  {tabicl_ref[k]:>16.5f}")

    if div(Sp_noXq) > 0.01:
        print(f"\n→ Simple model: ICL works without X_test (div={div(Sp_noXq):.3f} > 0)")
    else:
        print(f"\n→ Simple model: also relies on X regression (same as TabICLv2)")

    # Plot
    v_norms  = ep0["oracle_V"][0].norm(dim=(-2, -1))
    groups   = (v_norms > v_norms.median()).long().cpu().numpy()
    sort_ord = np.argsort(groups)
    n_weak   = int((groups == 0).sum()); n_strong = int((groups == 1).sum())

    fig = plot_corr_grid(
        estimators={
            "Base (ep0 supp)":   Sp_base[0, sort_ord].cpu(),
            "Swap (ep1 supp)":   Sp_swap[0, sort_ord].cpu(),
            "ZeroX_test":        Sp_noXq[0, sort_ord].cpu(),
            "RandZ_train":       Sp_randZ[0, sort_ord].cpu(),
        },
        oracle_R=R0[0, sort_ord].cpu(),
        n_instances=n_test,
        title=(f"SimpleCopulaNet ICL test — {n_weak} weak / {n_strong} strong\n"
               f"Does swapping support/X_test change predictions?"),
    )
    path = OUTPUT_DIR / "simple_icl_comparison.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {path}")


if __name__ == "__main__":
    main()
