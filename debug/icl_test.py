"""
icl_test.py — Does CopulaTabICLv2 actually do ICL for correlation prediction?

Two paths through the model:
  Regression path: X_test → enc_qry → readout → R  (ignores support)
  ICL path:        (X_train, Z_train) → support ctx → cross-attn → R  (true ICL)

Test 1 — Support swap:
  Train a model with MSE-only on episode_000000.
  Then at inference time, swap in the support from episode_000001 (different
  correlation structure) while keeping X_test fixed.
  → If predictions change substantially → model uses ICL.
  → If predictions are unchanged → model ignores support (pure regression).

Test 2 — Zero-support ablation:
  Inference with zero X_train and random Z_train (pure noise support).
  → If predictions collapse → model was using support context.
  → If predictions stay the same → model ignores support.

Test 3 — Support-only ablation:
  Zero out X_test so query embeddings carry no X information.
  → If predictions are still meaningful → Stage 3 ICL is working.
  → If predictions collapse to mean → model can only use regression path.

Usage:
    conda run -n multivariate-icl python debug/icl_test.py
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

DATA_DIR   = ROOT / "data" / "pit_hyperplane_debug"
OUTPUT_DIR = ROOT / "debug" / "icl_test_results"
N_TRAIN_STEPS = 3_000
LR            = 1e-3
GRAD_CLIP     = 1.0


def _cov_to_corr(D, V):
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def off_diag_mse(S_pred, R_ora):
    d = S_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=S_pred.device)
    return F.mse_loss(S_pred[..., ri, ci], R_ora[..., ri, ci]).item()


def prediction_diversity(Sigma):
    d = Sigma.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=Sigma.device)
    return Sigma[0, :, ri, ci].std(dim=0).mean().item()


def load_ep(idx, device):
    path = DATA_DIR / f"episode_{idx:06d}.pt"
    ep   = torch.load(path, weights_only=True)
    b    = 0
    tensor_keys = [k for k, v in ep.items() if isinstance(v, torch.Tensor)]
    return {k: ep[k][[b]].float().to(device) for k in tensor_keys}


def train_model(model, X_fwd, Z_fwd, R_ora, N, d, device):
    ri, ci    = torch.triu_indices(d, d, offset=1, device=device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    model.train()
    for step in range(N_TRAIN_STEPS):
        optimizer.zero_grad()
        _, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
        Sp = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        F.mse_loss(Sp[:, :, ri, ci], R_ora[:, :, ri, ci]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    model.eval()


def infer(model, X_fwd, Z_fwd, N, d):
    with torch.no_grad():
        _, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
    return torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load two episodes
    ep0 = load_ep(0, device)
    ep1 = load_ep(1, device)

    X0_train = ep0["X_train"];  Z0_train = ep0["Z_train"]
    X0_test  = ep0["X_test"];   Z0_test  = ep0["Z_test"]
    oD0 = ep0["oracle_D"];      oV0 = ep0["oracle_V"]

    X1_train = ep1["X_train"];  Z1_train = ep1["Z_train"]
    oD1 = ep1["oracle_D"];      oV1 = ep1["oracle_V"]

    N, n_test, d = X0_train.shape[1], X0_test.shape[1], Z0_train.shape[2]
    p = X0_train.shape[2]

    R_ora0 = _cov_to_corr(oD0, oV0)    # oracle for episode 0 test instances
    R_ora1 = _cov_to_corr(oD1, oV1)    # oracle for episode 1 test instances

    print(f"Episode 0: N={N}  n_test={n_test}  d={d}  p={p}")
    print(f"Training for {N_TRAIN_STEPS} steps with MSE-only on episode 0...")

    # Build and train model on episode 0
    torch.manual_seed(42)
    cfg = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    model = build_copula_tabicl_v2(SimpleNamespace(model=cfg)).to(device)

    X0_fwd = torch.cat([X0_train, X0_test], dim=1)
    Z0_fwd = torch.cat([Z0_train, Z0_test], dim=1)
    train_model(model, X0_fwd, Z0_fwd, R_ora0, N, d, device)

    # Baseline: original episode 0 support → episode 0 test instances
    Sp_base = infer(model, X0_fwd, Z0_fwd, N, d)
    mse_base = off_diag_mse(Sp_base, R_ora0)
    div_base = prediction_diversity(Sp_base)
    print(f"\n[Baseline] ep0 support + ep0 test → MSE={mse_base:.5f}  div={div_base:.4f}")

    # Test 1: Swap support from episode 1 (but keep ep0 X_test)
    X1_fwd = torch.cat([X1_train, X0_test], dim=1)
    Z1_fwd = torch.cat([Z1_train, Z0_test], dim=1)
    Sp_swap = infer(model, X1_fwd, Z1_fwd, N, d)
    mse_swap = off_diag_mse(Sp_swap, R_ora0)
    div_swap = prediction_diversity(Sp_swap)
    change_swap = (Sp_swap - Sp_base).abs().mean().item()
    print(f"[Test 1 — support swap] ep1 support + ep0 test:")
    print(f"  MSE vs ep0 oracle={mse_swap:.5f}  div={div_swap:.4f}  "
          f"pred_change_vs_baseline={change_swap:.6f}")
    if change_swap < 1e-4:
        print(f"  → Model IGNORES support (pure regression from X)")
    else:
        print(f"  → Model uses support context (ICL working)")

    # Test 2: Zero support (noise X_train, zero Z_train)
    Xz_train = torch.zeros_like(X0_train)
    Zz_train = torch.zeros_like(Z0_train)
    Xz_fwd   = torch.cat([Xz_train, X0_test], dim=1)
    Zz_fwd   = torch.cat([Zz_train, Z0_test], dim=1)
    Sp_zero  = infer(model, Xz_fwd, Zz_fwd, N, d)
    mse_zero = off_diag_mse(Sp_zero, R_ora0)
    div_zero = prediction_diversity(Sp_zero)
    change_zero = (Sp_zero - Sp_base).abs().mean().item()
    print(f"\n[Test 2 — zero support] zero X_train, zero Z_train:")
    print(f"  MSE vs ep0 oracle={mse_zero:.5f}  div={div_zero:.4f}  "
          f"pred_change_vs_baseline={change_zero:.6f}")
    if change_zero < 1e-3:
        print(f"  → Predictions unchanged → support context has NO effect")
    else:
        print(f"  → Predictions changed → support contributes to predictions")

    # Test 3: Zero X_test (query features zeroed — can model still predict from support?)
    X0q_zero = torch.cat([X0_train, torch.zeros_like(X0_test)], dim=1)
    Sp_noXq  = infer(model, X0q_zero, Z0_fwd, N, d)
    mse_noXq = off_diag_mse(Sp_noXq, R_ora0)
    div_noXq = prediction_diversity(Sp_noXq)
    change_noXq = (Sp_noXq - Sp_base).abs().mean().item()
    print(f"\n[Test 3 — zero X_test] support intact, X_test = 0:")
    print(f"  MSE vs ep0 oracle={mse_noXq:.5f}  div={div_noXq:.4f}  "
          f"pred_change_vs_baseline={change_noXq:.6f}")
    if div_noXq < 0.01:
        print(f"  → Predictions collapsed (diversity≈0) → model relies on X regression")
    elif mse_noXq < mse_base * 2:
        print(f"  → ICL path maintains predictions even without X_test (partially)")
    else:
        print(f"  → MSE doubled → model mainly relies on X regression path")

    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")
    print(f"  Baseline MSE:          {mse_base:.5f}")
    print(f"  Support swap MSE:      {mse_swap:.5f}  (pred_change={change_swap:.4f})")
    print(f"  Zero support MSE:      {mse_zero:.5f}  (pred_change={change_zero:.4f})")
    print(f"  Zero X_test MSE:       {mse_noXq:.5f}  (pred_change={change_noXq:.4f})")
    print()

    if change_swap < 0.01 and change_zero < 0.01:
        print("CONCLUSION: Model uses REGRESSION FROM X ONLY.")
        print("  → Completely ignores the support set (no ICL).")
        print("  → Stage-3 cross-attention contributes nothing.")
        print("  → The architecture cannot do ICL for correlation prediction.")
    elif change_swap > 0.01 and div_noXq > 0.05:
        print("CONCLUSION: Model does PROPER ICL.")
        print("  → Support swap changes predictions (reads support).")
        print("  → Zero X_test still works (Stage-3 carries information).")
    else:
        print("CONCLUSION: PARTIAL ICL — X features dominate, support contributes.")

    # Correlation grid comparison
    v_norms  = oV0[0].norm(dim=(-2, -1))
    groups   = (v_norms > v_norms.median()).long().cpu().numpy()
    sort_ord = np.argsort(groups)
    n_weak   = int((groups == 0).sum())
    n_strong = int((groups == 1).sum())

    fig = plot_corr_grid(
        estimators={
            "Base (ep0 support)": Sp_base[0, sort_ord].cpu(),
            "Swap (ep1 support)": Sp_swap[0, sort_ord].cpu(),
            "ZeroX_test":         Sp_noXq[0, sort_ord].cpu(),
        },
        oracle_R=R_ora0[0, sort_ord].cpu(),
        n_instances=n_test,
        title=(f"ICL test — {n_weak} weak (top) / {n_strong} strong (bottom)\n"
               f"Trained MSE-only on ep0 | Does swapping support change predictions?"),
    )
    path = OUTPUT_DIR / "icl_comparison.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {path}")


if __name__ == "__main__":
    main()
