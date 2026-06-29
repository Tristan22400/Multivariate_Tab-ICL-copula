"""
nll_debug.py — NLL training diagnostics for CopulaTabICLv2

Runs a series of diagnostics to identify why NLL training stagnates:

  SECTION 0  Data sanity
             oracle NLL, sample-covariance baseline, independence baseline
             on the val set.  If oracle < indep, the signal is there.

  SECTION 1  Gradient norms
             Per-module grad norms after the very first backward pass on one
             batch.  Vanishing grads upstream of the ICL embedding would
             explain why the transformer never learns to read support Z.

  SECTION 2  Support-shuffle test
             Do model predictions change when we permute the support Z values?
             If not, the Stage-3 ICL path is completely ignored.

  SECTION 3  Single-episode overfit
             Train 1000 steps on ONE fixed episode.  If the model can
             memorize one episode under NLL, the ICL mechanism works but
             generalisation is the challenge.

  SECTION 4  Full NLL training — tiny model
             d_model=16, n_cls=2, rank=2, 1 layer per stage.  ~5k params.
             Fast enough to run 5 000 steps on CPU in a few minutes.
             Logs copula_nll = woodbury_nll - indep_nll so negative means
             better-than-independence.

Usage (from project root):
    conda run -n multivariate-icl python debug/nll_debug.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import make_episode_loader, split_episode_files
from loss import indep_normal_nll, woodbury_nll
from model import CopulaTabICLv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = ROOT / "data" / "pit_hyperplane_debug"
N_STEPS    = 5_000
LR         = 3e-4
GRAD_CLIP  = 1.0

TINY = dict(d_model=16, n_heads=2, n_layers_s1=1, n_layers_s2=1, n_layers_s3=2,
            n_inducing=8, n_cls=2, p_max=20, d_max=8, rank=2)
SMALL = dict(d_model=32, n_heads=2, n_layers_s1=2, n_layers_s2=2, n_layers_s3=3,
             n_inducing=16, n_cls=2, p_max=20, d_max=8, rank=4)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copula_nll(y: torch.Tensor, d_Z: torch.Tensor, V_Z: torch.Tensor) -> float:
    """woodbury_nll - indep_normal_nll: negative = better than independence."""
    mu = torch.zeros_like(d_Z)
    return (woodbury_nll(y, mu, d_Z, V_Z) - indep_normal_nll(y)).item()


def sample_cov_baseline(Z_support: torch.Tensor, Z_query: torch.Tensor) -> float:
    """Compute NLL using the empirical correlation of Z_support as predictor.
    This is the trivial ICL baseline: no model, just sample statistics."""
    B, N_sup, d = Z_support.shape
    total_nll = 0.0
    count = 0
    for b in range(B):
        Z = Z_support[b]  # (N_sup, d)
        Z_c = Z - Z.mean(0, keepdim=True)
        S = (Z_c.T @ Z_c) / (N_sup - 1)   # (d, d) — sample covariance
        # Normalize to correlation
        std = S.diagonal().clamp(min=1e-6).sqrt()
        R = S / (std.unsqueeze(1) * std.unsqueeze(0))
        R = R.clamp(-0.999, 0.999)
        R.fill_diagonal_(1.0)
        # Add jitter for numerical stability
        R = R + 1e-4 * torch.eye(d, device=R.device)
        # Compute NLL via Cholesky on query Z
        try:
            L = torch.linalg.cholesky(R)
            z_q = Z_query[b]  # (N_q, d)
            v = torch.linalg.solve_triangular(L.unsqueeze(0).expand(z_q.shape[0],-1,-1),
                                              z_q.unsqueeze(-1), upper=False).squeeze(-1)
            logdet = 2.0 * L.diagonal().log().sum()
            quad = (v**2).sum(-1)
            nll = 0.5 * (d * math.log(2*math.pi) + logdet + quad).mean().item()
            total_nll += nll
            count += 1
        except Exception:
            pass
    if count == 0:
        return float('nan')
    return total_nll / count


def oracle_nll(D: torch.Tensor, V: torch.Tensor, Z_query: torch.Tensor) -> float:
    """NLL using ground-truth oracle covariance parameters."""
    # D, V are raw covariance (not correlation); normalise to correlation
    Sigma = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    R = Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))
    R.clamp_(-0.999, 0.999)
    R.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    # Convert R back to Woodbury (diag + low-rank) for woodbury_nll
    # Use eigendecomposition of (R - I):
    off = R - torch.eye(R.shape[-1], device=R.device)
    eigvals, eigvecs = torch.linalg.eigh(off)
    eigvals = eigvals.clamp(min=0.0)
    V_ora = eigvecs * eigvals.sqrt().unsqueeze(-2)
    D_ora = torch.ones_like(D[..., :R.shape[-1]])  # diag = 1 - row_norm² is complex; use identity fallback
    # Simpler: use full cholesky NLL directly
    try:
        L = torch.linalg.cholesky(R + 1e-5 * torch.eye(R.shape[-1], device=R.device))
        z_q = Z_query  # (B, N_q, d)
        B, Nq, d = z_q.shape
        L_exp = L.unsqueeze(2).expand(B, -1, Nq, -1, -1).reshape(B * Nq, d, d)
        z_flat = z_q.reshape(B * Nq, d).unsqueeze(-1)
        v = torch.linalg.solve_triangular(L_exp, z_flat, upper=False).squeeze(-1)
        logdet = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (B, N_sup-like)
        # average over queries
        quad = (v**2).reshape(B, Nq, d).sum(-1)  # (B, Nq)
        logdet_exp = logdet.unsqueeze(1).expand(B, Nq)
        nll = 0.5 * (d * math.log(2*math.pi) + logdet_exp + quad).mean().item()
        return nll
    except Exception as e:
        return float('nan')


def grad_norms(model: nn.Module) -> dict[str, float]:
    """Return L2 grad norm per named module group."""
    groups: dict[str, list[float]] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        top = name.split('.')[0]
        groups.setdefault(top, []).append(p.grad.norm().item())
    return {k: sum(v)/len(v) for k, v in groups.items()}


# ---------------------------------------------------------------------------
# SECTION 0: Data sanity
# ---------------------------------------------------------------------------
def section0_data_sanity(val_episodes: list, device: str):
    print("\n" + "="*60)
    print("SECTION 0 — Data sanity (val episodes)")
    print("="*60)

    total_indep = 0.0
    total_oracle = 0.0
    total_sampcov = 0.0
    n = 0

    for ep in val_episodes:
        Z_tr = ep["Z_train"].float().to(device)
        Z_te = ep["Z_test"].float().to(device)
        D_o  = ep["oracle_D"].float().to(device)
        V_o  = ep["oracle_V"].float().to(device)

        B, N_te, d = Z_te.shape

        # Independence baseline
        total_indep += indep_normal_nll(Z_te).item()

        # Oracle NLL (using ground-truth covariance from FIRST support instance as prior)
        # oracle_D/V are (B, N_train, d) — use mean over train instances
        D_mean = D_o[:, :, :d].mean(1)  # (B, d)
        V_mean = V_o[:, :, :d, :].mean(1)  # (B, d, r)
        # Expand for all query instances
        D_exp = D_mean.unsqueeze(1).expand(-1, N_te, -1)
        V_exp = V_mean.unsqueeze(1).expand(-1, N_te, -1, -1)
        ora_nll = woodbury_nll(Z_te, torch.zeros_like(Z_te), D_exp, V_exp)
        total_oracle += ora_nll.item()

        # Sample covariance baseline
        total_sampcov += sample_cov_baseline(Z_tr, Z_te)
        n += 1

    print(f"  Independence NLL:     {total_indep/n:.4f}")
    print(f"  Oracle NLL:           {total_oracle/n:.4f}  (target — should be < indep)")
    print(f"  Sample-cov baseline:  {total_sampcov/n:.4f}  (trivial ICL — sample Z corr)")
    print(f"  Copula NLL gap (oracle vs indep): {(total_oracle - total_indep)/n:.4f}")
    print("  (negative gap = oracle uses correlations beneficially)")


# ---------------------------------------------------------------------------
# SECTION 1: Gradient norms
# ---------------------------------------------------------------------------
def section1_grad_norms(model: nn.Module, ep, device: str):
    print("\n" + "="*60)
    print("SECTION 1 — Gradient norms after one backward pass")
    print("="*60)

    model.train()
    Z_tr = ep["Z_train"].float().to(device)
    Z_te = ep["Z_test"].float().to(device)
    X_tr = ep["X_train"].float().to(device)
    X_te = ep["X_test"].float().to(device)
    B, N, d = Z_tr.shape

    Xf = torch.cat([X_tr, X_te], 1)
    Zf = torch.cat([Z_tr, Z_te], 1)

    model.zero_grad()
    mu_Z, d_Z, V_Z = model(Xf, Zf, n_support=N)
    nll = woodbury_nll(Z_te, mu_Z, d_Z, V_Z)
    nll.backward()

    norms = grad_norms(model)
    for name, g in sorted(norms.items(), key=lambda x: -x[1]):
        print(f"  {name:<30s}  grad_norm={g:.2e}")

    # Also check: does embed_tae get gradient?
    if hasattr(model, 'embed_tae') and model.embed_tae.weight.grad is not None:
        print(f"\n  embed_tae.weight grad norm = {model.embed_tae.weight.grad.norm():.2e}")
    if hasattr(model, 'embed_icl') and model.embed_icl.weight.grad is not None:
        print(f"  embed_icl.weight grad norm = {model.embed_icl.weight.grad.norm():.2e}")
    if hasattr(model, 'icl_gate_sup') and model.icl_gate_sup.grad is not None:
        print(f"  icl_gate grad              = {model.icl_gate_sup.grad.mean().item():.4f}  (current val={torch.sigmoid(model.icl_gate_sup).mean().item():.3f})")


# ---------------------------------------------------------------------------
# SECTION 2: Support shuffle test
# ---------------------------------------------------------------------------
def section2_shuffle_test(model: nn.Module, ep, device: str):
    print("\n" + "="*60)
    print("SECTION 2 — Z-sensitivity test (does model use Z_support values?)")
    print("NOTE: permuting instance ORDER is wrong (MHA is permutation-invariant).")
    print("      Correct test: REPLACE Z_support with random noise.")
    print("="*60)

    model.eval()
    Z_tr = ep["Z_train"].float().to(device)
    Z_te = ep["Z_test"].float().to(device)
    X_tr = ep["X_train"].float().to(device)
    X_te = ep["X_test"].float().to(device)
    B, N, d = Z_tr.shape

    with torch.no_grad():
        Xf = torch.cat([X_tr, X_te], 1)
        Zf = torch.cat([Z_tr, Z_te], 1)
        _, d_Z_orig, V_Z_orig = model(Xf, Zf, n_support=N)
        R_orig = torch.diag_embed(d_Z_orig) + V_Z_orig @ V_Z_orig.transpose(-2,-1)

        # Replace support Z with strong random noise — if the model reads Z at all,
        # predictions should differ substantially.
        Z_tr_rand = torch.randn_like(Z_tr) * 3.0
        Zf_rand = torch.cat([Z_tr_rand, Z_te], 1)
        _, d_Z_rand, V_Z_rand = model(Xf, Zf_rand, n_support=N)
        R_rand = torch.diag_embed(d_Z_rand) + V_Z_rand @ V_Z_rand.transpose(-2,-1)

        # Replace support Z with zeros — the null case
        Zf_zero = torch.cat([torch.zeros_like(Z_tr), Z_te], 1)
        _, d_Z_zero, V_Z_zero = model(Xf, Zf_zero, n_support=N)
        R_zero = torch.diag_embed(d_Z_zero) + V_Z_zero @ V_Z_zero.transpose(-2,-1)

        ri, ci = torch.triu_indices(d, d, offset=1, device=device)
        diff_rand = (R_orig[..., ri, ci] - R_rand[..., ri, ci]).abs().mean().item()
        diff_zero = (R_orig[..., ri, ci] - R_zero[..., ri, ci]).abs().mean().item()
        print(f"  |R_orig - R_random|: {diff_rand:.4f}  (>0.01 = model reads Z values)")
        print(f"  |R_orig - R_zero|:   {diff_zero:.4f}")
        print(f"  R_orig off-diag: mean={R_orig[...,ri,ci].mean():.3f}  std={R_orig[...,ri,ci].std():.3f}")
        print(f"  R_rand off-diag: mean={R_rand[...,ri,ci].mean():.3f}  std={R_rand[...,ri,ci].std():.3f}")
        print(f"  R_zero off-diag: mean={R_zero[...,ri,ci].mean():.3f}  std={R_zero[...,ri,ci].std():.3f}")


# ---------------------------------------------------------------------------
# SECTION 3: Single-episode overfit
# ---------------------------------------------------------------------------
def section3_overfit(model: nn.Module, ep, device: str, n_steps: int = 1000):
    print("\n" + "="*60)
    print(f"SECTION 3 — Overfit test ({n_steps} steps on ONE episode)")
    print("="*60)

    # Fresh tiny model for this test
    m = CopulaTabICLv2(**TINY).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    Z_tr = ep["Z_train"].float().to(device)
    Z_te = ep["Z_test"].float().to(device)
    X_tr = ep["X_train"].float().to(device)
    X_te = ep["X_test"].float().to(device)
    B, N, d = Z_tr.shape
    Xf = torch.cat([X_tr, X_te], 1)
    Zf = torch.cat([Z_tr, Z_te], 1)

    # Oracle R for reference
    D_o = ep["oracle_D"].float().to(device)
    V_o = ep["oracle_V"].float().to(device)
    D_exp = D_o[:, :, :d].mean(1, keepdim=True).expand(-1, Z_te.shape[1], -1)
    V_exp = V_o[:, :, :d, :].mean(1, keepdim=True).expand(-1, Z_te.shape[1], -1, -1)
    ora = woodbury_nll(Z_te, torch.zeros_like(D_exp), D_exp, V_exp).item()
    indep = indep_normal_nll(Z_te).item()
    print(f"  Oracle NLL={ora:.4f}  Independence NLL={indep:.4f}")

    m.train()
    prev = float('inf')
    for step in range(n_steps + 1):
        opt.zero_grad()
        mu_Z, d_Z, V_Z = m(Xf, Zf, n_support=N)
        nll = woodbury_nll(Z_te, mu_Z, d_Z, V_Z)
        if torch.isnan(nll): print(f"  NaN at step {step}"); break
        nll.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            cnll = nll.item() - indep
            ri, ci = torch.triu_indices(d, d, offset=1)
            R = (torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2,-1)).detach()
            print(f"  step={step:4d}  nll={nll.item():.4f}  copula_nll={cnll:.4f}  "
                  f"off_mean={R[...,ri,ci].mean():.3f}  off_std={R[...,ri,ci].std():.3f}")
    print(f"  Final copula NLL: {nll.item()-indep:.4f}  (oracle gap: {ora-indep:.4f})")


# ---------------------------------------------------------------------------
# SECTION 4: Full NLL training — tiny + small models
# ---------------------------------------------------------------------------
def section4_full_training(train_files, val_files, device: str, n_steps: int = N_STEPS):
    print("\n" + "="*60)
    print(f"SECTION 4 — Full NLL training: tiny + small model ({n_steps} steps)")
    print("="*60)

    val_episodes = [torch.load(f, weights_only=True) for f in val_files]

    def train_model(name: str, kwargs: dict):
        print(f"\n  -- {name} --")
        torch.manual_seed(0)
        model = CopulaTabICLv2(**kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        loader = make_episode_loader(files=train_files, shuffle=True, num_workers=0)
        ep_iter = iter(loader)
        def next_ep():
            nonlocal ep_iter
            try: return next(ep_iter)
            except StopIteration:
                nonlocal loader
                ep_iter = iter(loader); return next(ep_iter)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)

        t0 = time.perf_counter()
        model.train()
        for step in range(n_steps + 1):
            ep = next_ep()
            Xt = ep["X_train"].float().to(device)
            Zt = ep["Z_train"].float().to(device)
            Xq = ep["X_test"].float().to(device)
            Zq = ep["Z_test"].float().to(device)
            B, N, d = Zt.shape
            Xf = torch.cat([Xt, Xq], 1); Zf = torch.cat([Zt, Zq], 1)

            opt.zero_grad()
            mu_Z, d_Z, V_Z = model(Xf, Zf, n_support=N)
            nll = woodbury_nll(Zq, mu_Z, d_Z, V_Z)
            if torch.isnan(nll): print(f"  NaN at step {step}"); break
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step()

            if step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    ri, ci = torch.triu_indices(d, d, offset=1, device=device)
                    R = (torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2,-1)).detach()
                    off_mean = R[..., ri, ci].mean().item()
                    off_std  = R[..., ri, ci].std().item()
                    c_diag   = d_Z.mean().item()
                    gate     = torch.sigmoid(model.icl_gate_sup).mean().item()
                    # Quick val pass
                    val_cnll = 0.0; nv = 0
                    for vep in val_episodes[:8]:
                        Xv = vep["X_train"].float().to(device); Zv = vep["Z_train"].float().to(device)
                        Xq2 = vep["X_test"].float().to(device); Zq2 = vep["Z_test"].float().to(device)
                        Nv_ = Zv.shape[1]
                        mu2, d2, V2 = model(torch.cat([Xv,Xq2],1), torch.cat([Zv,Zq2],1), Nv_)
                        val_cnll += (woodbury_nll(Zq2,mu2,d2,V2) - indep_normal_nll(Zq2)).item()
                        nv += 1
                    val_cnll /= nv
                print(f"  step={step:5d}  train_nll={nll.item():.4f}  "
                      f"val_copula={val_cnll:.4f}  C_diag={c_diag:.3f}  "
                      f"off_mean={off_mean:.3f}  off_std={off_std:.3f}  gate={gate:.3f}  "
                      f"t={time.perf_counter()-t0:.0f}s")
                model.train()
        return model

    m_tiny  = train_model("TINY  (d_model=16, n_cls=2, rank=2, 1-1-2 layers)", TINY)
    m_small = train_model("SMALL (d_model=32, n_cls=2, rank=4, 2-2-3 layers)", SMALL)
    return m_tiny, m_small


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Data: {DATA_DIR}")

    train_files, val_files = split_episode_files(str(DATA_DIR), val_n_episodes=50)
    print(f"Episodes: {len(train_files)} train / {len(val_files)} val")

    # Pre-load a few val episodes for fast diagnostics
    val_episodes = [torch.load(f, weights_only=True) for f in val_files[:10]]
    ep0 = val_episodes[0]

    # ---- Section 0: data sanity ----
    section0_data_sanity(val_episodes, device)

    # ---- Section 1 + 2 on a tiny fresh model ----
    torch.manual_seed(0)
    tiny_model = CopulaTabICLv2(**TINY).to(device)
    n_params = sum(p.numel() for p in tiny_model.parameters())
    print(f"\nTiny model params: {n_params:,}  config={TINY}")

    section1_grad_norms(tiny_model, ep0, device)
    section2_shuffle_test(tiny_model, ep0, device)

    # ---- Section 3: overfit test ----
    section3_overfit(tiny_model, ep0, device, n_steps=1000)

    # ---- Section 4: full NLL training ----
    section4_full_training(train_files, val_files, device, n_steps=N_STEPS)
