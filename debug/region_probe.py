"""region_probe.py — localize per-query *region collapse* in CopulaTabICLv2.

Symptom: on the hyperplane data every query in a dataset gets ~the same
predicted correlation matrix R, instead of the region-specific R (low-cov side
vs high-cov side of the hyperplane).

This probe trains the *current* architecture (src/model.py) briefly with the
same loss as train.py (NLL + per-query oracle-correlation MSE), then attributes
the collapse to one of:

  (a) region NOT encoded   — Fisher separability of region labels drops through
                             Stage 1/2 -> Stage 3 embeddings.
  (b) attention collapse   — Stage-3 query->support attention is uniform
                             (H_norm ~ 1) and identical across regions
                             (attn_div ~ 0).
  (c) readout collapse     — embeddings separable + attention localizes, but the
                             per-region predicted off-diag |r| gap is ~0.

Region labels are recovered post-hoc by clustering the saved oracle_V Frobenius
norm per test instance (each region has a distinct V_k), so no extra data is
needed.

Usage (from project root):
    conda run -n multivariate-icl python debug/region_probe.py
    conda run -n multivariate-icl python debug/region_probe.py --train-steps 6000 --d-model 64
    conda run -n multivariate-icl python debug/region_probe.py --ckpt path/to.pt --train-steps 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import infinite_episode_iter, make_episode_loader, split_episode_files
from loss import indep_normal_nll, woodbury_nll
from model import build_copula_tabicl_v2


# ---------------------------------------------------------------------------
# Region labels from oracle_V norm
# ---------------------------------------------------------------------------

def region_labels(oracle_V_b: torch.Tensor, decimals: int = 3) -> tuple[torch.Tensor, int]:
    """Map each test instance to a region id, ordered by covariance strength.

    Args:
        oracle_V_b : (n_test, d, r) low-rank factor for one dataset.
    Returns:
        labels : (n_test,) long, region id in {0..K-1} sorted by V-norm (0 = weakest cov).
        K      : number of distinct regions.
    """
    vn = oracle_V_b.norm(dim=(-2, -1))                  # (n_test,)
    key = vn.round(decimals=decimals)
    uniq = torch.unique(key)
    uniq, _ = torch.sort(uniq)                          # ascending = weak -> strong cov
    labels = torch.bucketize(key, uniq, right=False).clamp(max=len(uniq) - 1)
    # bucketize can misplace exact matches; remap by nearest unique instead
    labels = (key.unsqueeze(-1) - uniq.unsqueeze(0)).abs().argmin(dim=-1)
    return labels.long(), int(len(uniq))


def corr_offdiag_absmean(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Mean |off-diagonal correlation| per instance.  D:(...,d) V:(...,d,r) -> (...,)."""
    S = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    R = S / (std.unsqueeze(-1) * std.unsqueeze(-2))
    d = R.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=R.device)
    return R[..., ri, ci].abs().mean(dim=-1)


def fisher_ratio(emb: torch.Tensor, labels: torch.Tensor) -> float:
    """Between-class / within-class scatter (summed over dims) for 2+ regions.

    High => regions are linearly separable in this embedding.
    emb : (n, h)  labels : (n,)
    """
    classes = labels.unique()
    if len(classes) < 2:
        return float("nan")
    grand = emb.mean(0)
    sb = 0.0
    sw = 0.0
    for c in classes:
        m = emb[labels == c]
        if m.shape[0] < 1:
            continue
        mu = m.mean(0)
        sb += m.shape[0] * ((mu - grand) ** 2).sum().item()
        sw += ((m - mu) ** 2).sum().item()
    return sb / max(sw, 1e-8)


# ---------------------------------------------------------------------------
# Stage-3 attention capture (patch s3_blocks like train_tabicl_v2_debug.py)
# ---------------------------------------------------------------------------

def forward_with_captures(model, X_fwd, Z_fwd, n_support):
    """Run model.eval() forward, capturing per-block S3 attention, the Stage-2
    row embedding (query rows), and the final query embedding."""
    caps: dict = {"attn": {}, "s2": None, "s3": None}

    # capture s2 row embedding (after s2_norm) and s3 final norm output
    h2 = model.s2_norm.register_forward_hook(
        lambda m, i, o: caps.__setitem__("s2_raw", o.detach())
    )
    h3 = model.s3_norm.register_forward_hook(
        lambda m, i, o: caps.__setitem__("s3", o.detach())
    )

    orig = []
    for idx, blk in enumerate(model.s3_blocks):
        orig.append(blk.forward)
        def _patch(i, o):
            def _f(x, ns, **kw):
                # Pass all kwargs (including x_sim_bias) to the real forward so
                # captured attention weights reflect the bias.
                out, w = o(x, ns, return_attn_weights=True, **kw)
                caps["attn"][i] = w.detach()
                return out
            return _f
        blk.forward = _patch(idx, blk.forward)

    was_train = model.training
    model.eval()
    try:
        with torch.no_grad():
            mu, dZ, VZ = model(X_fwd, Z_fwd, n_support=n_support)
    finally:
        for idx, blk in enumerate(model.s3_blocks):
            blk.forward = orig[idx]
        h2.remove(); h3.remove()
        if was_train:
            model.train()
    return mu, dZ, VZ, caps


# ---------------------------------------------------------------------------
# Quick trainer (mirrors train.py loss: NLL + per-query oracle-corr MSE)
# ---------------------------------------------------------------------------

def quick_train(model, train_files, device, steps, lr=3e-4, wd=1e-4, n_think=0,
                aux_w=1.0, log_every=200):
    if steps <= 0:
        return
    loader = make_episode_loader(files=train_files, shuffle=True, num_workers=2)
    it = infinite_episode_iter(loader)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    model.train()
    for step in range(steps):
        ep = next(it)
        X_tr = ep["X_train"].to(device).float()
        Z_tr = ep["Z_train"].to(device).float()
        X_te = ep["X_test"].to(device).float()
        Z_te = ep["Z_test"].to(device).float()
        oD = ep["oracle_D"].to(device).float()
        oV = ep["oracle_V"].to(device).float()
        B, N, d = Z_tr.shape
        if n_think > 0:
            tX = torch.zeros(B, n_think, X_tr.shape[-1], device=device)
            tZ = torch.zeros(B, n_think, d, device=device)
            X_tr = torch.cat([tX, X_tr], 1); Z_tr = torch.cat([tZ, Z_tr], 1); N += n_think
        X_fwd = torch.cat([X_tr, X_te], 1)
        Z_fwd = torch.cat([Z_tr, Z_te], 1)
        opt.zero_grad()
        mu, dZ, VZ = model(X_fwd, Z_fwd, n_support=N)
        loss_nll = woodbury_nll(Z_te, mu, dZ, VZ)
        # per-query oracle correlation MSE (off-diag)
        S_o = torch.diag_embed(oD) + oV @ oV.transpose(-1, -2)
        std_o = S_o.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_o = S_o / (std_o.unsqueeze(-1) * std_o.unsqueeze(-2))
        S_p = torch.diag_embed(dZ) + VZ @ VZ.transpose(-1, -2)
        ri, ci = torch.triu_indices(d, d, offset=1, device=device)
        loss_mse = F.mse_loss(S_p[..., ri, ci], R_o[..., ri, ci])
        loss = loss_nll + aux_w * loss_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % log_every == 0:
            cnll = loss_nll.item() - indep_normal_nll(Z_te).item()
            print(f"  [train {step:>5d}] copula_nll={cnll:.4f}  mse={loss_mse.item():.5f}")


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe(model, episodes, device, n_think=0):
    n_s3 = len(model.s3_blocks)
    agg = {
        "collapse_ratio": [], "pred_gap": [], "oracle_gap": [],
        "fisher_X": [], "fisher_s2": [], "fisher_s3": [],
        "Hnorm": {i: [] for i in range(n_s3)},
        "attn_div": [],
        "pred_lo": [], "pred_hi": [], "ora_lo": [], "ora_hi": [],
    }

    for ep in episodes:
        X_tr = ep["X_train"].to(device).float()
        Z_tr = ep["Z_train"].to(device).float()
        X_te = ep["X_test"].to(device).float()
        Z_te = ep["Z_test"].to(device).float()
        oD = ep["oracle_D"].to(device).float()
        oV = ep["oracle_V"].to(device).float()
        B, N, d = Z_tr.shape
        if n_think > 0:
            tX = torch.zeros(B, n_think, X_tr.shape[-1], device=device)
            tZ = torch.zeros(B, n_think, d, device=device)
            X_tr = torch.cat([tX, X_tr], 1); Z_tr = torch.cat([tZ, Z_tr], 1); N += n_think
        n_sup = N
        X_fwd = torch.cat([X_tr, X_te], 1)
        Z_fwd = torch.cat([Z_tr, Z_te], 1)

        mu, dZ, VZ, caps = forward_with_captures(model, X_fwd, Z_fwd, n_sup)
        mu, dZ, VZ = mu.float(), dZ.float(), VZ.float()

        # Stage-2 / Stage-3 query embeddings
        s2 = caps.get("s2_raw")  # (B, N_full, n_cls, d_model)
        s2_q = s2.reshape(B, s2.shape[1], -1)[:, n_sup:].float() if s2 is not None else None
        s3 = caps.get("s3")      # (B, N, d_icl)
        s3_q = s3[:, n_sup:].float() if s3 is not None else None

        pred_r = corr_offdiag_absmean(dZ, VZ)          # (B, n_test)
        ora_r = corr_offdiag_absmean(oD, oV)           # (B, n_test)

        for b in range(B):
            lab, K = region_labels(oV[b])
            lab = lab.to(device)
            if K < 2:
                continue
            # per-region mean predicted / oracle off-diag |r|
            pr = torch.stack([pred_r[b][lab == g].mean() for g in range(K)])
            orc = torch.stack([ora_r[b][lab == g].mean() for g in range(K)])
            agg["pred_gap"].append((pr.max() - pr.min()).item())
            agg["oracle_gap"].append((orc.max() - orc.min()).item())
            denom = (orc.max() - orc.min()).item()
            if denom > 1e-6:
                agg["collapse_ratio"].append((pr.max() - pr.min()).item() / denom)
            # weakest vs strongest region (region 0 = weakest by construction)
            agg["pred_lo"].append(pr[0].item()); agg["pred_hi"].append(pr[-1].item())
            agg["ora_lo"].append(orc[0].item()); agg["ora_hi"].append(orc[-1].item())

            # Fisher separability of region labels in X / stage2 / stage3 emb
            agg["fisher_X"].append(fisher_ratio(X_te[b], lab))
            if s2_q is not None:
                agg["fisher_s2"].append(fisher_ratio(s2_q[b], lab))
            if s3_q is not None:
                agg["fisher_s3"].append(fisher_ratio(s3_q[b], lab))

            # attention diversity across regions (use weakest vs strongest)
            for i in range(n_s3):
                w = caps["attn"][i][b]                  # (N, N)
                wq = w[n_sup:, :n_sup].clamp(min=1e-12) # (n_test, n_support)
                H = -(wq * wq.log()).sum(-1).mean().item() / math.log(max(n_sup, 2))
                agg["Hnorm"][i].append(H)
            # last-block attention: mean attn of region0 queries vs regionK-1 queries
            wlast = caps["attn"][n_s3 - 1][b][n_sup:, :n_sup]  # (n_test, n_support)
            a_lo = wlast[lab == 0].mean(0)
            a_hi = wlast[lab == (K - 1)].mean(0)
            agg["attn_div"].append((a_lo - a_hi).abs().sum().item())

    def m(x):
        x = [v for v in x if v == v]  # drop nan
        return float(np.mean(x)) if x else float("nan")

    print("\n" + "=" * 70)
    print("REGION-COLLAPSE PROBE  (averaged over datasets)")
    print("=" * 70)
    print(f"  per-region predicted |r| gap : {m(agg['pred_gap']):.4f}")
    print(f"  per-region oracle    |r| gap : {m(agg['oracle_gap']):.4f}")
    print(f"  COLLAPSE RATIO (pred/oracle) : {m(agg['collapse_ratio']):.3f}"
          "   (~0 = full collapse, ~1 = perfect)")
    print(f"    weak-region  |r|  pred={m(agg['pred_lo']):.3f}  oracle={m(agg['ora_lo']):.3f}")
    print(f"    strong-region|r|  pred={m(agg['pred_hi']):.3f}  oracle={m(agg['ora_hi']):.3f}")
    print(f"\n  Fisher separability of region (higher = more separable):")
    print(f"    X_test input   : {m(agg['fisher_X']):.3f}")
    print(f"    Stage-2 row emb: {m(agg['fisher_s2']):.3f}")
    print(f"    Stage-3 qry emb: {m(agg['fisher_s3']):.3f}")
    print(f"\n  Stage-3 attention entropy H_norm per block (~1 = uniform/collapsed):")
    print("    " + "  ".join(f"b{i}={m(agg['Hnorm'][i]):.3f}" for i in range(n_s3)))
    print(f"  Attn diversity weak-vs-strong queries (L1, ~0 = same routing): "
          f"{m(agg['attn_div']):.4f}")

    # ---- attribution ----
    cr = m(agg["collapse_ratio"])
    fX, fS2, fS3 = m(agg["fisher_X"]), m(agg["fisher_s2"]), m(agg["fisher_s3"])
    Hlast = m(agg["Hnorm"][n_s3 - 1])
    adiv = m(agg["attn_div"])
    print("\n  ATTRIBUTION:")
    if cr > 0.5:
        print("    -> No collapse: model differentiates regions (ratio > 0.5).")
    else:
        if fS3 < 0.25 * fX or (fS3 < fS2 * 0.5):
            print("    -> (a) REGION NOT ENCODED: separability collapses by Stage-3.")
        elif Hlast > 0.9 and adiv < 0.05:
            print("    -> (b) ATTENTION COLLAPSE: uniform + identical routing across regions.")
        else:
            print("    -> (c) READOUT COLLAPSE: regions encoded & attention routes, but R is flat.")
    print("=" * 70)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "pit_hyperplane_debug"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--train-steps", type=int, default=None,
                    help="default: 0 if --ckpt given, else 4000")
    ap.add_argument("--n-episodes", type=int, default=16)
    ap.add_argument("--n-think", type=int, default=0)
    ap.add_argument("--device", default="auto")
    # model overrides (for Phase-2 capacity experiments)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-inducing", type=int, default=None)
    ap.add_argument("--n-cls", type=int, default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}  data: {args.data_dir}")

    # When a checkpoint is given, build from its saved model cfg so the
    # architecture always matches the weights; else use the repo yaml.
    ck = None
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg_model = OmegaConf.create(ck["cfg"]["model"])
        print(f"Building model from checkpoint cfg: d_model={cfg_model.d_model} "
              f"s3={cfg_model.n_layers_s3} rank={cfg_model.rank}")
    else:
        cfg_model = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    if args.d_model is not None: cfg_model.d_model = args.d_model
    if args.n_inducing is not None: cfg_model.n_inducing = args.n_inducing
    if args.n_cls is not None: cfg_model.n_cls = args.n_cls
    model = build_copula_tabicl_v2(SimpleNamespace(model=cfg_model)).to(device)

    if ck is not None:
        sd = ck.get("model_state", ck)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"Loaded {args.ckpt}  (missing={len(miss)} unexpected={len(unexp)})")
        if miss or unexp:
            print(f"  missing={list(miss)[:6]}  unexpected={list(unexp)[:6]}")

    train_steps = args.train_steps
    if train_steps is None:
        train_steps = 0 if args.ckpt else 4000

    train_files, val_files = split_episode_files(args.data_dir, val_n_episodes=50)
    quick_train(model, train_files, device, train_steps, n_think=args.n_think)

    val_eps = [torch.load(f, weights_only=True) for f in val_files[: args.n_episodes]]
    probe(model, val_eps, device, n_think=args.n_think)


if __name__ == "__main__":
    main()
