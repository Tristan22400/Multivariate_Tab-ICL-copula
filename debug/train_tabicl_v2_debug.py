"""
train_tabicl_v2_debug.py — CopulaTabICLv2 debug training loop.

MSE-only on pit_hyperplane_debug episodes, 10k steps.
Monitors per-block attention entropy, ReZero alphas, ICL gate, and
prediction diversity at every LOG_EVERY / ATTN_EVERY step.

Usage (from project root):
    conda run -n multivariate-icl python debug/train_tabicl_v2_debug.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import infinite_episode_iter, make_episode_loader, split_episode_files
from model import build_copula_tabicl_v2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = ROOT / "data" / "pit_hyperplane_debug"
OUTPUT_DIR = ROOT / "debug" / "tabicl_v2_train_debug"
N_STEPS    = 10_000
LR         = 3e-4
WD         = 1e-4
GRAD_CLIP  = 1.0
LOG_EVERY  = 100
ATTN_EVERY = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cov_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def get_s3_attn_entropy(model, X_fwd, Z_fwd, n_support):
    """Capture Stage-3 attention weights and return normalised entropy per block."""
    captured: dict[int, torch.Tensor] = {}
    orig_forwards = []
    for i, blk in enumerate(model.s3_blocks):
        orig_forwards.append(blk.forward)
        def _patch(idx, orig):
            def _f(x, ns, **kw):
                out, w = orig(x, ns, return_attn_weights=True)
                captured[idx] = w.detach().cpu()
                return out
            return _f
        blk.forward = _patch(i, blk.forward)

    was_train = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(X_fwd, Z_fwd, n_support=n_support)
    finally:
        for i, blk in enumerate(model.s3_blocks):
            blk.forward = orig_forwards[i]
        if was_train:
            model.train()

    log_ns = math.log(max(n_support, 2))
    h_norms = {}
    for idx, w in captured.items():
        # w: (B, N, N) — rows are queries, cols are support
        wq = w[:, n_support:, :n_support].clamp(min=1e-12)
        H  = -(wq * wq.log()).sum(dim=-1).mean().item()
        h_norms[idx] = H / log_ns
    return h_norms


def prediction_diversity(Sigma_pred: torch.Tensor) -> float:
    d  = Sigma_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=Sigma_pred.device)
    off = Sigma_pred[..., ri, ci]   # (B, n_query, n_pairs)
    return off.std(dim=-2).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Data: {DATA_DIR}")

    # ---- Model ----
    cfg_model = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    model = build_copula_tabicl_v2(SimpleNamespace(model=cfg_model)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}  |  d_model={cfg_model.d_model}  "
          f"d_icl={model.d_icl}  n_s3={len(model.s3_blocks)}")

    # ---- Optimizer ----
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=1e-6)

    # ---- Data ----
    train_files, val_files = split_episode_files(str(DATA_DIR), val_n_episodes=50)
    print(f"Episodes: {len(train_files)} train / {len(val_files)} val")
    loader = make_episode_loader(files=train_files, shuffle=True, num_workers=2)
    ep_iter = infinite_episode_iter(loader)

    # Pre-load one val episode for attention diagnostics (small, fixed)
    val_ep  = torch.load(val_files[0], weights_only=True)
    Xv_tr   = val_ep["X_train"][[0]].float().to(device)
    Zv_tr   = val_ep["Z_train"][[0]].float().to(device)
    Xv_te   = val_ep["X_test"][[0]].float().to(device)
    Zv_te   = val_ep["Z_test"][[0]].float().to(device)
    Nv      = Xv_tr.shape[1]
    Xv_fwd  = torch.cat([Xv_tr, Xv_te], dim=1)
    Zv_fwd  = torch.cat([Zv_tr, Zv_te], dim=1)

    n_s3 = len(model.s3_blocks)

    # ---- Logging buffers ----
    steps_log  = []
    mse_log    = []
    div_log    = []
    gate_log   = []
    attn_steps = []
    h_log      = {i: [] for i in range(n_s3)}
    alpha_a_log = {i: [] for i in range(n_s3)}
    alpha_f_log = {i: [] for i in range(n_s3)}

    t0 = time.perf_counter()
    model.train()

    for step in range(N_STEPS):
        ep      = next(ep_iter)
        X_train = ep["X_train"].to(device)
        Z_train = ep["Z_train"].to(device)
        X_test  = ep["X_test"].to(device)
        Z_test  = ep["Z_test"].to(device)
        oD      = ep["oracle_D"].to(device)
        oV      = ep["oracle_V"].to(device)

        B, N, d = Z_train.shape
        X_fwd   = torch.cat([X_train, X_test], dim=1)
        Z_fwd   = torch.cat([Z_train, Z_test], dim=1)

        optimizer.zero_grad()
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)

        # MSE on off-diagonal entries of predicted vs oracle correlation matrix
        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        R_ora      = _cov_to_corr(oD, oV)
        nt         = Sigma_pred.shape[-1]
        ri, ci     = torch.triu_indices(nt, nt, offset=1, device=device)
        loss       = F.mse_loss(Sigma_pred[..., ri, ci], R_ora[..., ri, ci])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse_val = loss.item()
                div_val = prediction_diversity(Sigma_pred)
                gate_val = torch.sigmoid(model.icl_gate).item()
            steps_log.append(step)
            mse_log.append(mse_val)
            div_log.append(div_val)
            gate_log.append(gate_val)

        if step % ATTN_EVERY == 0:
            h_norms = get_s3_attn_entropy(model, Xv_fwd, Zv_fwd, Nv)
            attn_steps.append(step)
            h_vals = [h_norms.get(i, float("nan")) for i in range(n_s3)]
            aa = [model.s3_blocks[i].alpha_attn.item() for i in range(n_s3)]
            af = [model.s3_blocks[i].alpha_ffn.item()  for i in range(n_s3)]
            for i in range(n_s3):
                h_log[i].append(h_vals[i])
                alpha_a_log[i].append(aa[i])
                alpha_f_log[i].append(af[i])

            gate_v = torch.sigmoid(model.icl_gate).item()
            elapsed = time.perf_counter() - t0
            h_str  = " ".join(f"b{i}={h_vals[i]:.3f}" for i in range(n_s3))
            aa_str = " ".join(f"{a:.4f}" for a in aa)
            af_str = " ".join(f"{a:.4f}" for a in af)
            print(
                f"[{step:>6d}]  mse={mse_log[-1] if mse_log else float('nan'):.5f}  "
                f"div={div_log[-1] if div_log else float('nan'):.4f}  "
                f"gate={gate_v:.3f}  "
                f"H_norm=[{h_str}]  "
                f"alpha_a=[{aa_str}]  "
                f"alpha_f=[{af_str}]  "
                f"t={elapsed:.0f}s"
            )
            model.train()

    # ---------------------------------------------------------------------------
    # Final eval on held-out val episode
    # ---------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        mu_Z, d_Z, V_Z = model(Xv_fwd, Zv_fwd, n_support=Nv)
    Sigma_pred_f = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
    oD_v   = val_ep["oracle_D"][[0]].float().to(device)
    oV_v   = val_ep["oracle_V"][[0]].float().to(device)
    R_ora_v = _cov_to_corr(oD_v, oV_v)
    final_mse = F.mse_loss(Sigma_pred_f[..., ri, ci], R_ora_v[..., ri, ci]).item()
    final_div = prediction_diversity(Sigma_pred_f)
    final_h   = get_s3_attn_entropy(model, Xv_fwd, Zv_fwd, Nv)

    R_pred_np  = Sigma_pred_f[0].cpu().numpy()
    R_ora_np   = R_ora_v[0].cpu().numpy()
    d_dim = R_pred_np.shape[-1]
    ri_np, ci_np = np.triu_indices(d_dim, k=1)
    x_all = R_ora_np[:, ri_np, ci_np].ravel()
    y_all = R_pred_np[:, ri_np, ci_np].ravel()
    r_val  = float(np.corrcoef(x_all, y_all)[0, 1]) if x_all.std() > 1e-8 else float("nan")
    slope  = float(np.polyfit(x_all, y_all, 1)[0]) if x_all.std() > 1e-8 else 1.0

    print(f"\n{'='*70}")
    print(f"FINAL  mse={final_mse:.5f}  r={r_val:.3f}  slope={slope:.3f}  div={final_div:.4f}")
    print(f"Stage-3 attention entropy (H_norm):")
    for i, h in final_h.items():
        status = "COLLAPSED" if h > 0.90 else ("ok" if h < 0.5 else "partial")
        print(f"  blk {i}: {h:.4f}  [{status}]")
    print(f"ReZero alphas (attn / ffn):")
    for i in range(n_s3):
        aa = model.s3_blocks[i].alpha_attn.item()
        af = model.s3_blocks[i].alpha_ffn.item()
        print(f"  blk {i}: alpha_attn={aa:.5f}  alpha_ffn={af:.5f}")
    print(f"ICL gate: {torch.sigmoid(model.icl_gate).item():.4f}")
    print(f"{'='*70}")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    colors = plt.cm.viridis(np.linspace(0, 1, n_s3))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # 1. MSE curve
    axes[0, 0].plot(steps_log, mse_log, lw=1.5, color="steelblue")
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("Off-diag MSE")
    axes[0, 0].set_title("MSE (off-diagonal, oracle corr)")
    axes[0, 0].axhline(0.02, color="green", ls="--", lw=1, label="target 0.02")
    axes[0, 0].legend(fontsize=8)

    # 2. Prediction diversity
    axes[0, 1].plot(steps_log, div_log, lw=1.5, color="purple")
    axes[0, 1].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[0, 1].set_xlabel("Step"); axes[0, 1].set_ylabel("Std of off-diag predictions")
    axes[0, 1].set_title("Prediction diversity (0 = all queries get same R)")

    # 3. ICL gate
    axes[0, 2].plot(steps_log, gate_log, lw=1.5, color="orange")
    axes[0, 2].axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    axes[0, 2].set_ylim(0, 1); axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("sigmoid(icl_gate)"); axes[0, 2].set_title("ICL gate value")

    # 4. Attention entropy
    for i in range(n_s3):
        axes[1, 0].plot(attn_steps, h_log[i], color=colors[i], lw=1.5,
                        marker="o", ms=3, label=f"blk {i}")
    axes[1, 0].axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6, label="collapsed")
    axes[1, 0].axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6, label="peaked")
    axes[1, 0].set_ylim(-0.05, 1.1); axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("H_norm"); axes[1, 0].set_title("Stage-3 attention entropy")
    axes[1, 0].legend(fontsize=7, ncol=2)

    # 5. ReZero alpha_attn
    for i in range(n_s3):
        axes[1, 1].plot(attn_steps, alpha_a_log[i], color=colors[i], lw=1.5,
                        marker="o", ms=3, label=f"blk {i}")
    axes[1, 1].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[1, 1].set_xlabel("Step"); axes[1, 1].set_ylabel("alpha_attn")
    axes[1, 1].set_title("ReZero alpha_attn per S3 block")
    axes[1, 1].legend(fontsize=7, ncol=2)

    # 6. ReZero alpha_ffn
    for i in range(n_s3):
        axes[1, 2].plot(attn_steps, alpha_f_log[i], color=colors[i], lw=1.5,
                        marker="o", ms=3, label=f"blk {i}")
    axes[1, 2].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[1, 2].set_xlabel("Step"); axes[1, 2].set_ylabel("alpha_ffn")
    axes[1, 2].set_title("ReZero alpha_ffn per S3 block")
    axes[1, 2].legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"CopulaTabICLv2 debug — 10k steps MSE-only  |  "
        f"final MSE={final_mse:.5f}  r={r_val:.3f}  div={final_div:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / "training_diagnostics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # Oracle vs predicted scatter
    fig_s, ax_s = plt.subplots(figsize=(5, 5))
    ax_s.scatter(x_all, y_all, s=4, alpha=0.4, rasterized=True)
    lo = min(float(x_all.min()), float(y_all.min()))
    hi = max(float(x_all.max()), float(y_all.max()))
    ax_s.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax_s.set_xlabel("Oracle off-diag corr"); ax_s.set_ylabel("Predicted off-diag corr")
    ax_s.set_title(f"Step {N_STEPS} — r={r_val:.3f}  slope={slope:.3f}  n={len(x_all)}")
    fig_s.tight_layout()
    out_sc = OUTPUT_DIR / "scatter.png"
    fig_s.savefig(out_sc, dpi=120, bbox_inches="tight")
    plt.close(fig_s)
    print(f"Scatter saved: {out_sc}")


if __name__ == "__main__":
    main()
