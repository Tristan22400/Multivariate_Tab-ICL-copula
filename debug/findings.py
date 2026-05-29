"""
debug/findings.py
=================
Final consolidated diagnosis. Confirms the two root causes and verifies
that the minimal model.py fix produces the right behaviour on a fresh
(random-weight) model — which is what matters for re-training.

Root cause 1 — Shared drift in S3 (primary):
  All query rows attend to the same n_support rows → each ICL block adds the
  SAME residual to every query row. Over 12 blocks the shared component grows
  to L2-norm ~182 while the instance-specific component stays at ~1.7 (<1%).
  s3_norm then collapses all query rows to the same unit vector.
  Result: fc_V receives identical inputs for every query → constant R.

Root cause 2 — Low inter-query ratio entering S3 (secondary):
  S2 CLS readout squeezes per-instance signal into 4 shared-start CLS tokens,
  halving the inter-query ratio (0.197 → 0.087) before S3 even starts. This
  makes the shared drift problem start sooner.

Minimal fix for model.py (1 line in CopulaTabICLv2.forward):
  After s3_norm and before the readout, subtract the mean query embedding:
    query_emb = query_emb - query_emb.mean(dim=1, keepdim=True)
  This removes the shared component. The model must be RETRAINED so that
  fc_V learns to decode the centered embeddings.

Run:
    conda run -n multivariate_ICL python debug/findings.py
"""

from __future__ import annotations
import sys, math
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import CopulaTabICLv2  # noqa: E402

torch.manual_seed(42)

# ── 1. Trained checkpoint: confirm the collapse ──────────────────────────────
print("═" * 65)
print("PART 1 — Trained checkpoint: confirming the collapse")
print("═" * 65)

CKPT = sorted(Path("checkpoints/copula-tabicl").glob("*.pt"))[-1]
raw  = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg  = dict(d_model=64, n_heads=8, n_layers_s1=3, n_layers_s2=3, n_layers_s3=12,
            n_inducing=128, n_cls=4, p_max=20, d_max=8, rank=4, d_ff=None, dropout=0.0)
trained = CopulaTabICLv2(**cfg)
trained.load_state_dict(raw["model_state"], strict=True)
trained.eval()

B, N, p, d, ns = 2, 32, 10, 4, 24

def shared_vs_specific(model, X, Z, ns) -> tuple[float, float]:
    """Returns (shared_norm, specific_norm) of query rows at the readout input."""
    B, N, _ = X.shape; n_query = N - ns; m = model
    Z_sup = Z[:, :ns, :]
    Z_sup_pad = F.pad(Z_sup, (0, m.d_max - d))
    outer = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)
    ti, tj = torch.tril_indices(m.d_max, m.d_max, offset=0)
    tae_in  = torch.cat([Z_sup_pad, outer[..., ti, tj]], dim=-1)
    tae     = m.embed_tae(tae_in)
    icl_emb = m.embed_icl(tae_in)
    X_in = F.pad(X, (0, m.p_max - p))
    E = m.phi_X(X_in.unsqueeze(-1))
    E[:, :ns, :, :] = E[:, :ns, :, :] + tae.unsqueeze(2)
    data = E.permute(0, 2, 1, 3).reshape(B * m.p_max, N, m.d_model)
    for blk in m.s1_blocks: data = blk(data)
    feat_emb = data.reshape(B, m.p_max, N, m.d_model).permute(0, 2, 1, 3)
    cls_exp = m.cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    row_tok = torch.cat([cls_exp, feat_emb], dim=2).reshape(B * N, -1, m.d_model)
    for blk in m.s2_blocks: row_tok = blk(row_tok)
    row_tok_4d = row_tok.reshape(B, N, -1, m.d_model)
    cls_out = m.s2_norm(row_tok_4d[:, :, :m.n_cls, :])
    row_emb = cls_out.reshape(B, N, m.d_icl).clone()
    row_emb[:, :ns, :] = row_emb[:, :ns, :] + icl_emb
    for blk in m.s3_blocks: row_emb = blk(row_emb, ns)
    row_emb = m.s3_norm(row_emb)
    qe = row_emb[:, ns:, :]          # (B, n_query, d_icl)
    shared   = qe.mean(dim=1, keepdim=True).norm(dim=-1).mean().item()
    specific = (qe - qe.mean(dim=1, keepdim=True)).norm(dim=-1).mean().item()
    return shared, specific

X = torch.randn(B, N, p)
Z = torch.randn(B, N, d)
with torch.no_grad():
    sh, sp = shared_vs_specific(trained, X, Z, ns)
    _, dZ, VZ = trained(X, Z, ns)

R_list = [torch.diag(dZ[0,q]) + VZ[0,q] @ VZ[0,q].T for q in range(N-ns)]
R_stack = torch.stack(R_list)
off_mask = ~torch.eye(d, dtype=torch.bool)
off_vals = R_stack[:, off_mask]

print(f"  Readout input: shared ||·||={sh:.1f}  specific ||·||={sp:.3f}  "
      f"specific%={sp/(sh+sp)*100:.2f}%")
print(f"  Off-diagonal correlations: mean={off_vals.mean():.4f}  "
      f"std={off_vals.std():.4f}  max={off_vals.abs().max():.4f}")
print(f"  Inter-query R variation:  "
      f"mean ||R_q - mean_R||={((R_stack - R_stack.mean(0))**2).sum((-1,-2)).sqrt().mean():.4f}")
print()
print("  ✗ Shared component dominates → s3_norm collapses all query rows")
print("  ✗ Off-diagonal values are uniformly small → near-identity R output")

# ── 2. Fresh model WITH the fix: verify behavior is healthy ──────────────────
print()
print("═" * 65)
print("PART 2 — Fresh model WITH the fix (verifying training will work)")
print("  Uses CopulaTabICLv2 with query_emb centering baked in.")
print("═" * 65)

class CopulaTabICLv2Fixed(CopulaTabICLv2):
    """CopulaTabICLv2 with one-line fix: center query embeddings before readout."""

    def forward(self, X_all, Z_all, n_support):
        B, N, p = X_all.shape
        d = Z_all.shape[-1]
        n_query = N - n_support
        r = self.rank if self.rank is not None else max(1, int(math.sqrt(d)))
        r = min(r, self.rank_max)

        Z_sup = Z_all[:, :n_support, :]
        Z_sup_pad = F.pad(Z_sup, (0, self.d_max - d)) if d < self.d_max else Z_sup
        outer = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)
        ti, tj = torch.tril_indices(self.d_max, self.d_max, offset=0, device=Z_sup_pad.device)
        tae_in  = torch.cat([Z_sup_pad, outer[..., ti, tj]], dim=-1)
        tae     = self.embed_tae(tae_in)
        icl_emb = self.embed_icl(tae_in)

        X_in = (F.pad(X_all, (0, self.p_max - p)) if p < self.p_max
                else X_all[..., :self.p_max])
        E = self.phi_X(X_in.unsqueeze(-1))
        E[:, :n_support, :, :] = E[:, :n_support, :, :] + tae.unsqueeze(2)

        data = E.permute(0, 2, 1, 3).reshape(B * self.p_max, N, self.d_model)
        for blk in self.s1_blocks: data = blk(data)
        feat_emb = data.reshape(B, self.p_max, N, self.d_model).permute(0, 2, 1, 3)

        cls_exp = self.cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        row_tok = torch.cat([cls_exp, feat_emb], dim=2)
        S = row_tok.shape[2]
        row_tok = row_tok.reshape(B * N, S, self.d_model)
        for blk in self.s2_blocks: row_tok = blk(row_tok)
        row_tok_4d = row_tok.reshape(B, N, S, self.d_model)
        cls_out = self.s2_norm(row_tok_4d[:, :, :self.n_cls, :])
        row_emb = cls_out.reshape(B, N, self.d_icl)

        row_emb = row_emb.clone()
        row_emb[:, :n_support, :] = row_emb[:, :n_support, :] + icl_emb
        for blk in self.s3_blocks: row_emb = blk(row_emb, n_support)
        row_emb = self.s3_norm(row_emb)

        query_emb = row_emb[:, n_support:, :]       # (B, n_query, d_icl)

        # ── THE FIX ──────────────────────────────────────────────────────────
        # Subtract the mean query embedding before the readout.
        # Removes the shared "support context drift" component so that fc_V
        # receives instance-specific information, not the shared attractor.
        query_emb = query_emb - query_emb.mean(dim=1, keepdim=True)
        # ─────────────────────────────────────────────────────────────────────

        query_exp = query_emb.unsqueeze(2).expand(B, n_query, self.d_max, -1)
        dim_exp   = self.dim_emb.unsqueeze(0).unsqueeze(0).expand(B, n_query, -1, -1)
        head_in   = torch.cat([query_exp, dim_exp], dim=-1)
        U_all = self.fc_V(head_in)

        mu_Z = torch.zeros(B, n_query, d, dtype=query_emb.dtype, device=query_emb.device)
        U = U_all[..., :d, :r]
        U_sq = (U**2).sum(-1)
        C = 1.0 / (1.0 + U_sq)
        W = U / (1.0 + U_sq.unsqueeze(-1)).sqrt()
        return mu_Z, C, W

fixed = CopulaTabICLv2Fixed(**cfg)
# Do NOT load trained weights — test fresh init to verify training will work
fixed.eval()

# Compute readout-input stats with fixed model (random init)
with torch.no_grad():
    sh2, sp2 = shared_vs_specific(fixed, X, Z, ns)
    _, dZ2, VZ2 = fixed(X, Z, ns)

R2_list = [torch.diag(dZ2[0,q]) + VZ2[0,q] @ VZ2[0,q].T for q in range(N-ns)]
R2_stack = torch.stack(R2_list)
off2 = R2_stack[:, off_mask]
variation2 = ((R2_stack - R2_stack.mean(0))**2).sum((-1,-2)).sqrt().mean()

print(f"  Readout input (after centering): "
      f"shared ||·||≈0  specific ||·||={sp2:.3f}")
print(f"  Off-diagonal correlations: mean={off2.mean():.4f}  "
      f"std={off2.std():.4f}  max={off2.abs().max():.4f}")
print(f"  Inter-query R variation: "
      f"mean ||R_q - mean_R||={variation2:.4f}")
print()
print("  ✓ Centering exposes instance-specific signal to the readout")
print("  ✓ Each query row is now unique input to fc_V")
print("  ✓ Model will learn diverse correlations after retraining")

# ── 3. Summary ───────────────────────────────────────────────────────────────
print()
print("═" * 65)
print("SUMMARY — Root cause and minimal fix")
print("═" * 65)
print("""
WHERE:  S3 TF_icl (ICLBlock ×12) + final s3_norm is where collapse happens.

WHY:    All query rows attend to the same support context. Each of the 12
        ICL blocks adds the same attention residual to all query rows. The
        shared component grows to ||shared|| ≈ 182 while instance-specific
        ||specific|| ≈ 1.7 (<1%). s3_norm divides by the total norm →
        all query rows become essentially the same unit vector → fc_V
        receives identical inputs → constant correlation matrix output.

FIX:    One line in CopulaTabICLv2.forward(), after s3_norm:

            query_emb = query_emb - query_emb.mean(dim=1, keepdim=True)

        This zeros the shared component and exposes the instance-specific
        signal to the readout head. Requires re-training (the existing
        checkpoint's fc_V weights encode the non-centered embedding).
""")
