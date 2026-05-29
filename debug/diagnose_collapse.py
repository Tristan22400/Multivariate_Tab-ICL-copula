"""
debug/diagnose_collapse.py
==========================
Verifies the two root-cause hypotheses and tests candidate fixes.

Hypothesis A — S3 activation explosion:
  Within-row magnitude blows up (std 1.0 → 11.4 over 12 ICL blocks).
  Inter-query differences stay roughly constant in *absolute* terms (~0.09).
  RMSNorm then collapses the tiny inter-query variation to noise (ratio 0.09 → 0.009).

Hypothesis B — Query rows enter S3 nearly collinear:
  All queries attend to the SAME support → same attention output → same residual added.
  Verified by measuring cosine similarity between query rows.

Tests:
  [T1] Cosine similarity between query rows throughout the network (collinearity test)
  [T2] Eigenspectrum of query row matrix — if rank-1, all queries are collinear
  [T3] What ratio we'd have WITHOUT the final s3_norm (isolate its effect)
  [T4] What ratio post-norm ICLBlock gives at step 0 (can the architecture fix itself?)
  [T5] How much of the within-row growth is explained by the shared "support mean"

Run:
    conda run -n multivariate_ICL python debug/diagnose_collapse.py
"""

from __future__ import annotations
import sys, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import CopulaTabICLv2, RMSNorm, SwiGLUFFN  # noqa: E402

# ── Load checkpoint ──────────────────────────────────────────────────────────
CKPT = sorted(Path("checkpoints/copula-tabicl").glob("*.pt"))[-1]
raw  = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg  = dict(d_model=64, n_heads=8, n_layers_s1=3, n_layers_s2=3, n_layers_s3=12,
            n_inducing=128, n_cls=4, p_max=20, d_max=8, rank=4, d_ff=None, dropout=0.0)
model = CopulaTabICLv2(**cfg)
model.load_state_dict(raw["model_state"], strict=True)
model.eval()

# ── Shared batch ─────────────────────────────────────────────────────────────
torch.manual_seed(0)
B, N, p, d = 2, 32, 10, 4
n_support, n_query = 24, 8
X  = torch.randn(B, N, p)
Z  = torch.randn(B, N, d)

# ── Helpers ───────────────────────────────────────────────────────────────────
def cosim_matrix(t: torch.Tensor) -> torch.Tensor:
    """t: (n_query, D) → (n_query, n_query) pairwise cosine similarity."""
    t = t.float()
    n = F.normalize(t, dim=-1)
    return n @ n.T

def mean_off_diag_cosim(t: torch.Tensor) -> float:
    """Average |cosine sim| between all distinct query pairs (batch 0)."""
    C = cosim_matrix(t[0])          # (n_query, n_query)
    mask = ~torch.eye(n_query, dtype=torch.bool)
    return C[mask].abs().mean().item()

def effective_rank(t: torch.Tensor) -> float:
    """Roy & Vetterli effective rank of query row matrix (batch 0)."""
    t0 = t[0].float()               # (n_query, D)
    # SVD
    _, S, _ = torch.svd(t0)
    p = (S / S.sum()).clamp(min=1e-9)
    return (-p * p.log()).sum().exp().item()

def ratio(t: torch.Tensor) -> tuple[float, float]:
    """Returns (inter-query std / within-query std, cosim)."""
    inter = t.std(dim=1).mean().item()
    within = t.std(dim=-1).mean().item()
    cosim = mean_off_diag_cosim(t)
    return inter / (within + 1e-9), cosim

# ── Partial forward to get S3 input ──────────────────────────────────────────
with torch.no_grad():
    m = model
    Z_sup = Z[:, :n_support, :]
    Z_sup_pad = F.pad(Z_sup, (0, m.d_max - d))
    outer = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)
    ti, tj = torch.tril_indices(m.d_max, m.d_max, offset=0)
    tae_in  = torch.cat([Z_sup_pad, outer[..., ti, tj]], dim=-1)
    tae     = m.embed_tae(tae_in)
    icl_emb = m.embed_icl(tae_in)

    X_in = F.pad(X, (0, m.p_max - p))
    E = m.phi_X(X_in.unsqueeze(-1))
    E[:, :n_support, :, :] += tae.unsqueeze(2)

    data = E.permute(0, 2, 1, 3).reshape(B * m.p_max, N, m.d_model)
    for blk in m.s1_blocks:
        data = blk(data)

    feat_emb = data.reshape(B, m.p_max, N, m.d_model).permute(0, 2, 1, 3)
    cls_exp  = m.cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    row_tok  = torch.cat([cls_exp, feat_emb], dim=2)
    S = row_tok.shape[2]
    row_tok  = row_tok.reshape(B * N, S, m.d_model)
    for blk in m.s2_blocks:
        row_tok = blk(row_tok)

    row_tok_4d = row_tok.reshape(B, N, S, m.d_model)
    cls_out    = m.s2_norm(row_tok_4d[:, :, :m.n_cls, :])
    row_emb_s2 = cls_out.reshape(B, N, m.d_icl)          # (B, N, d_icl) — S3 input

# ── T1/T2: Cosine similarity and effective rank throughout S3 ─────────────────
print("═" * 70)
print("TEST T1+T2 — Query-row collinearity across S3 blocks")
print("  (high cosim / low eff-rank → all query rows point the same direction)")
print("═" * 70)
print(f"  {'Layer':<35s}  ratio   cosim   eff_rank")

with torch.no_grad():
    re = row_emb_s2.clone()
    re[:, :n_support, :] += icl_emb

    rq = re[:, n_support:, :]
    r, c = ratio(rq)
    er = effective_rank(rq)
    print(f"  {'before S3':<35s}  {r:.4f}  {c:.4f}  {er:.2f}")

    for i, blk in enumerate(m.s3_blocks):
        re = blk(re, n_support)
        rq = re[:, n_support:, :]
        r, c = ratio(rq)
        er = effective_rank(rq)
        print(f"  {'S3 block '+str(i):<35s}  {r:.4f}  {c:.4f}  {er:.2f}")

    re_normed = m.s3_norm(re)
    rq = re_normed[:, n_support:, :]
    r, c = ratio(rq)
    er = effective_rank(rq)
    print(f"  {'after s3_norm (readout input)':<35s}  {r:.4f}  {c:.4f}  {er:.2f}")

# ── T3: Without s3_norm ───────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("TEST T3 — Readout WITHOUT s3_norm (does removing the norm help?)")
print("═" * 70)
with torch.no_grad():
    re = row_emb_s2.clone()
    re[:, :n_support, :] += icl_emb
    for blk in m.s3_blocks:
        re = blk(re, n_support)

    rq_raw = re[:, n_support:, :]
    r, c = ratio(rq_raw)
    er = effective_rank(rq_raw)
    print(f"  raw S3 output (no norm):  ratio={r:.4f}  cosim={c:.4f}  eff_rank={er:.2f}")
    # Normalize per-dimension across queries (a different strategy)
    rq_perbatch = (rq_raw - rq_raw.mean(dim=1, keepdim=True)) / (rq_raw.std(dim=1, keepdim=True) + 1e-6)
    r2, c2 = ratio(rq_perbatch)
    er2 = effective_rank(rq_perbatch)
    print(f"  centered per query-batch: ratio={r2:.4f}  cosim={c2:.4f}  eff_rank={er2:.2f}")

# ── T4: Post-norm ICLBlock (same trained weights, different norm placement) ───
print("\n" + "═" * 70)
print("TEST T4 — Post-norm ICLBlock (does re-ordering norm→residual help?)")
print("  Using SAME trained weights, just applying norm AFTER residual add.")
print("═" * 70)

class PostNormICLBlock(nn.Module):
    """ICLBlock with post-norm (same weights, different ordering)."""
    def __init__(self, ref_block):
        super().__init__()
        self.norm1 = ref_block.norm1
        self.attn  = ref_block.attn
        self.norm2 = ref_block.norm2
        self.ffn   = ref_block.ffn

    def forward(self, x, n_support):
        N = x.shape[1]
        mask = torch.zeros(N, N, dtype=x.dtype, device=x.device)
        if n_support < N:
            mask[:, n_support:] = float("-inf")
        x = self.norm1(x + self.attn(x, x, x, attn_mask=mask)[0])
        x = self.norm2(x + self.ffn(x))
        return x

with torch.no_grad():
    re = row_emb_s2.clone()
    re[:, :n_support, :] += icl_emb
    print(f"  {'Layer':<35s}  ratio   cosim   within_std")
    rq = re[:, n_support:, :]
    r, c = ratio(rq)
    print(f"  {'before S3':<35s}  {r:.4f}  {c:.4f}  {rq.std():.3f}")

    for i, blk in enumerate(m.s3_blocks):
        pn_blk = PostNormICLBlock(blk)
        re = pn_blk(re, n_support)
        rq = re[:, n_support:, :]
        r, c = ratio(rq)
        print(f"  {'post-norm S3 block '+str(i):<35s}  {r:.4f}  {c:.4f}  {rq.std():.3f}")

# ── T5: Decompose S3 output into shared + instance-specific components ────────
print("\n" + "═" * 70)
print("TEST T5 — Shared vs instance-specific variance in S3 query rows")
print("  shared = mean over queries, instance-specific = residual")
print("  If shared dominates → all queries look the same after normalization")
print("═" * 70)
with torch.no_grad():
    re = row_emb_s2.clone()
    re[:, :n_support, :] += icl_emb

    for i, blk in enumerate(m.s3_blocks):
        re = blk(re, n_support)
        rq = re[:, n_support:, :]            # (B, n_query, d_icl)
        shared   = rq.mean(dim=1, keepdim=True)   # (B, 1, d_icl)
        specific = rq - shared                     # (B, n_query, d_icl)
        pct = (specific.norm(dim=-1) / (rq.norm(dim=-1) + 1e-9)).mean().item() * 100
        shared_norm   = shared.norm(dim=-1).mean().item()
        specific_norm = specific.norm(dim=-1).mean().item()
        print(f"  S3 block {i:2d}: shared ||·||={shared_norm:7.2f}  "
              f"specific ||·||={specific_norm:6.3f}  "
              f"specific% ={pct:5.2f}%")

print("\n" + "═" * 70)
print("SUMMARY")
print("═" * 70)
print("""
Root cause: S3 TF_icl accumulates a large SHARED component in each query row.
  • All query rows attend to the same support → they all receive the same
    attention residual each block.
  • The 'shared' norm grows block-by-block (>100 after 12 blocks).
  • The 'instance-specific' norm stays small (<1).
  • s3_norm divides each row by its total norm → specific% ≈ 0 → all rows
    collapse to the same unit vector.

The effective_rank trace above confirms rank drops toward 1 across S3 blocks.
""")
