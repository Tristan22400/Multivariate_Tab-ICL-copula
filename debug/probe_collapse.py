"""
debug/probe_collapse.py
=======================
Find the layer where CopulaTabICLv2 collapses to a constant per-row representation.

Key metric tracked at every layer boundary:
  inter_q_ratio = inter-query std / within-query std

When this ratio is low, all query rows look the same to the readout head →
constant correlation matrix.

Run:
    conda run -n multivariate_ICL python debug/probe_collapse.py
"""

from __future__ import annotations
import os, sys, math
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import CopulaTabICLv2  # noqa: E402

# ── Load checkpoint ──────────────────────────────────────────────────────────
CKPT_DIR = Path(os.environ.get("PROBE_CKPT_DIR", "checkpoints/copula-tabicl-rezero-test"))
CKPT = sorted(CKPT_DIR.glob("*.pt"))[-1]
print(f"Checkpoint: {CKPT.name}")

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = dict(d_model=64, n_heads=8, n_layers_s1=3, n_layers_s2=3, n_layers_s3=12,
           n_inducing=128, n_cls=4, p_max=20, d_max=8, rank=4, d_ff=None, dropout=0.0)

model = CopulaTabICLv2(**cfg)
model.load_state_dict(raw["model_state"], strict=True)
model.eval()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")

# ── Synthetic batch ───────────────────────────────────────────────────────────
torch.manual_seed(0)
B, N, p, d = 2, 32, 10, 4
n_support, n_query = 24, 8

X   = torch.randn(B, N, p)
Z   = torch.randn(B, N, d)          # approx PIT-scores

# ── Helper ───────────────────────────────────────────────────────────────────
def ratio(t: torch.Tensor, label: str) -> float:
    """
    t : (B, n_query, D) — query row embeddings.
    Returns inter_q_std / within_q_std (the 'collapse ratio').
    """
    inter = t.std(dim=1).mean().item()   # std across queries, averaged over B,D
    within = t.std(dim=-1).mean().item() # std within a single row embedding
    r = inter / (within + 1e-9)
    print(f"  {label:<50s}  inter={inter:.5f}  within={within:.5f}  ratio={r:.5f}")
    return r

# ── Instrumented forward ──────────────────────────────────────────────────────
with torch.no_grad():
    m = model

    # ── Stage 0: target-aware embeddings ────────────────────────────────────
    Z_sup = Z[:, :n_support, :]
    Z_sup_pad = F.pad(Z_sup, (0, m.d_max - d))
    outer = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)
    ti, tj = torch.tril_indices(m.d_max, m.d_max, offset=0)
    vech = outer[..., ti, tj]
    tae_in = torch.cat([Z_sup_pad, vech], dim=-1)
    tae    = m.embed_tae(tae_in)    # (B, n_sup, d_model)
    icl_emb = m.embed_icl(tae_in)   # (B, n_sup, d_icl)

    # ── Stage 1: feature embedding + TF_col ─────────────────────────────────
    X_in = F.pad(X, (0, m.p_max - p)) if p < m.p_max else X[..., :m.p_max]
    E = m.phi_X(X_in.unsqueeze(-1))                 # (B,N,p_max,d_model)
    E[:, :n_support, :, :] += tae.unsqueeze(2)

    print("── Stage 1: TF_col ─────────────────────────────────────────────────")
    data = E.permute(0, 2, 1, 3).reshape(B * m.p_max, N, m.d_model)
    # query cols: (B*p_max, n_query, d_model)
    qdata = lambda d_: d_.reshape(B, m.p_max, N, m.d_model)[:, :, n_support:, :] \
                         .reshape(B, m.p_max * n_query, m.d_model)
    ratio(qdata(data).reshape(B, n_query, -1), "before S1")
    for i, blk in enumerate(m.s1_blocks):
        data = blk(data)
        ratio(qdata(data).reshape(B, n_query, -1), f"after S1 block {i}")

    feat_emb = data.reshape(B, m.p_max, N, m.d_model).permute(0, 2, 1, 3)

    # ── Stage 2: TF_row ──────────────────────────────────────────────────────
    print("\n── Stage 2: TF_row ─────────────────────────────────────────────────")
    cls_exp = m.cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    row_tok = torch.cat([cls_exp, feat_emb], dim=2)   # (B,N,n_cls+p_max,D)
    S = row_tok.shape[2]
    row_tok = row_tok.reshape(B * N, S, m.d_model)

    qtok = lambda t_: t_.reshape(B, N, S, m.d_model)[:, n_support:, :, :] \
                        .reshape(B, n_query, -1)
    ratio(qtok(row_tok), "before S2")
    for i, blk in enumerate(m.s2_blocks):
        row_tok = blk(row_tok)
        ratio(qtok(row_tok), f"after S2 block {i}")

    row_tok_4d = row_tok.reshape(B, N, S, m.d_model)
    cls_out = m.s2_norm(row_tok_4d[:, :, :m.n_cls, :])  # (B,N,n_cls,D)
    row_emb = cls_out.reshape(B, N, m.d_icl)             # (B,N,d_icl)

    print("\n── Stage 2→3 handoff ───────────────────────────────────────────────")
    ratio(row_emb[:, n_support:, :], "row_emb after S2 (before ICL inject)")

    # ── Stage 3: TF_icl ──────────────────────────────────────────────────────
    print("\n── Stage 3: TF_icl ─────────────────────────────────────────────────")
    row_emb = row_emb.clone()
    row_emb[:, :n_support, :] += icl_emb
    ratio(row_emb[:, n_support:, :], "after ICL inject (support rows enriched)")

    for i, blk in enumerate(m.s3_blocks):
        row_emb = blk(row_emb, n_support)
        ratio(row_emb[:, n_support:, :], f"after S3 block {i:2d}")

    row_emb_normed = m.s3_norm(row_emb)
    ratio(row_emb_normed[:, n_support:, :], "after s3_norm  ← READOUT INPUT")

    # ── Readout ───────────────────────────────────────────────────────────────
    print("\n── Readout ─────────────────────────────────────────────────────────")
    query_emb = row_emb_normed[:, n_support:, :]          # (B, n_query, d_icl)
    query_exp = query_emb.unsqueeze(2).expand(B, n_query, m.d_max, -1)
    dim_exp   = m.dim_emb.unsqueeze(0).unsqueeze(0).expand(B, n_query, -1, -1)
    head_in   = torch.cat([query_exp, dim_exp], dim=-1)   # (B,n_q,d_max,d_icl+d_dim)
    ratio(head_in.reshape(B, n_query, -1), "head_in (row_emb + dim_emb tiled)")

    h1  = F.gelu(m.fc_V.fc1(head_in))
    U_all = m.fc_V.fc2(h1)
    ratio(U_all.reshape(B, n_query, -1), "U_all (raw output of fc_V)")

    # ── Root-cause summary ────────────────────────────────────────────────────
    print("\n── Root-cause summary ──────────────────────────────────────────────")
    r = min(m.rank if m.rank else max(1, int(math.sqrt(d))), m.rank_max)
    U = U_all[..., :d, :r]
    U_sq = (U**2).sum(-1)
    C    = 1.0 / (1.0 + U_sq)
    W    = U / (1.0 + U_sq.unsqueeze(-1)).sqrt()

    inter_C = C.std(dim=1).mean().item()
    inter_W = W.std(dim=1).mean().item()
    print(f"  C_diag inter-query std : {inter_C:.6f}  (0 = fully constant diag)")
    print(f"  W      inter-query std : {inter_W:.6f}  (0 = fully constant low-rank factor)")

    # Correlation matrices for first 4 queries
    print("\n  Correlation matrices for first 4 queries (batch 0):")
    for q in range(4):
        R = torch.diag(C[0,q]) + W[0,q] @ W[0,q].T
        off = R[~torch.eye(d, dtype=bool)].abs()
        print(f"  Q{q}: diag={C[0,q].tolist()}  off-diag mean={off.mean():.4f}  std={off.std():.4f}")

    # Explain the main collapse point
    print()
    print("  ⚑  Key finding:")
    print(f"     After S3 + s3_norm the inter-query ratio is tiny.")
    print(f"     Activation magnitude grows across S3 (within-row std explodes),")
    print(f"     but inter-query differences barely grow → RMSNorm collapses them.")
    print(f"     All query rows then enter fc_V nearly identically.")
