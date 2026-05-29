"""
debug/check_attention_weights.py
=================================
Diagnose attention over-smoothing in Stage 3 (TF_icl) of CopulaTabICLv2.

For each ICLBlock we extract the query→support attention weight slice
  w : (B, n_query, n_support)   -- averaged over heads

and report three statistics per block:
  mean_w   -- should equal 1/n_support if perfectly uniform
  weight_std -- std across support positions (0 = over-smoothed)
  H_norm   -- normalised entropy  H / log(n_support)
              1.0 = fully uniform (maximum over-smoothing)
              0.0 = one-hot (perfectly sharp)

We run the analysis twice:
  [A] loaded checkpoint, phi_x_scale = 1.0  (current model)
  [B] same checkpoint weights,  phi_x_scale = 5.0  (boosted X signal)

Run:
    conda run -n multivariate_ICL python debug/check_attention_weights.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import CopulaTabICLv2  # noqa: E402

# ── Checkpoint ────────────────────────────────────────────────────────────────
import os
CKPT_DIR = Path(os.environ.get("PROBE_CKPT_DIR", "checkpoints/copula-tabicl-rezero-test"))
CKPT = sorted(CKPT_DIR.glob("*.pt"))[-1]
print(f"Checkpoint : {CKPT.name}")

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
BASE_CFG = dict(
    d_model=64, n_heads=8, n_layers_s1=3, n_layers_s2=3, n_layers_s3=12,
    n_inducing=128, n_cls=4, p_max=20, d_max=8, rank=4, d_ff=None, dropout=0.0,
)

# ── Synthetic batch ───────────────────────────────────────────────────────────
torch.manual_seed(0)
B, N, p, d = 2, 32, 10, 4
n_support, n_query = 24, 8

X = torch.randn(B, N, p)
Z = torch.randn(B, N, d)


# ── Helper: instrumented Stage-3 forward ─────────────────────────────────────
def run_stage3_attn(model: CopulaTabICLv2) -> list[torch.Tensor]:
    """Returns list of attn_weight tensors (one per S3 block)."""
    m = model

    with torch.no_grad():
        # -- Stages 0-2: replicate CopulaTabICLv2.forward up to row_emb --------
        Z_sup = Z[:, :n_support, :]
        Z_sup_pad = F.pad(Z_sup, (0, m.d_max - d))
        outer = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)
        ti, tj = torch.tril_indices(m.d_max, m.d_max, offset=0)
        vech = outer[..., ti, tj]
        tae_in = torch.cat([Z_sup_pad, vech], dim=-1)

        tae = m.embed_tae(tae_in)
        icl_emb = m.embed_icl(tae_in)

        X_in = F.pad(X, (0, m.p_max - p)) if p < m.p_max else X[..., :m.p_max]
        E1 = m.phi_X(X_in.unsqueeze(-1)) * m.phi_x_scale
        E2 = E1.clone()
        E2[:, :n_support, :, :] += tae.unsqueeze(2)

        data = E2.permute(0, 2, 1, 3).reshape(B * m.p_max, N, m.d_model)
        for blk in m.s1_blocks:
            data = blk(data)
        feat_emb = data.reshape(B, m.p_max, N, m.d_model).permute(0, 2, 1, 3)

        cls_exp = m.cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        row_tok = torch.cat([cls_exp, feat_emb], dim=2)
        S = row_tok.shape[2]
        row_tok = row_tok.reshape(B * N, S, m.d_model)
        for blk in m.s2_blocks:
            row_tok = blk(row_tok)
        row_tok_4d = row_tok.reshape(B, N, S, m.d_model)
        cls_out = m.s2_norm(row_tok_4d[:, :, :m.n_cls, :])
        row_emb = cls_out.reshape(B, N, m.d_icl)

        # ICL injection
        row_emb = row_emb.clone()
        row_emb[:, :n_support, :] += icl_emb

        # -- Stage 3: collect attention weights --------------------------------
        attn_list: list[torch.Tensor] = []
        for blk in m.s3_blocks:
            row_emb, attn_w = blk(row_emb, n_support, return_attn_weights=True)
            # attn_w: (B, N, N) — query→key weights
            attn_list.append(attn_w)

    return attn_list


def entropy_stats(
    attn_list: list[torch.Tensor],
    label: str,
) -> list[float]:
    """Print per-block stats; return list of normalised entropies."""
    uniform_baseline = 1.0 / n_support
    log_ns = math.log(n_support)
    h_norms: list[float] = []

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  uniform baseline = {uniform_baseline:.4f}  (= 1/{n_support})")
    print(f"{'─'*70}")
    print(f"  {'blk':>3}  {'mean_w':>8}  {'w_std':>8}  {'H_norm':>8}")

    for i, attn_w in enumerate(attn_list):
        # slice: query rows attending to support keys
        w = attn_w[:, n_support:, :n_support]   # (B, n_query, n_support)
        w = w.clamp(min=1e-10)                   # numerical safety for log

        mean_w = w.mean().item()
        w_std  = w.std(dim=-1).mean().item()     # std over support positions
        # per-query entropy, averaged
        H = -(w * w.log()).sum(dim=-1).mean().item()
        H_norm = H / log_ns
        h_norms.append(H_norm)

        flag = "  ← UNIFORM (over-smoothed)" if H_norm > 0.98 else ""
        print(f"  {i:>3}  {mean_w:>8.5f}  {w_std:>8.5f}  {H_norm:>8.4f}{flag}")

    return h_norms


# ── Run A: loaded checkpoint, scale=1 ────────────────────────────────────────
model_a = CopulaTabICLv2(**BASE_CFG, phi_x_scale=1.0)
model_a.load_state_dict(raw["model_state"], strict=True)
model_a.eval()
print(f"\nParams: {sum(p.numel() for p in model_a.parameters()):,}")

attn_a = run_stage3_attn(model_a)
h_a = entropy_stats(attn_a, "A — checkpoint  phi_x_scale=1.0")

# ── Run B: same weights, scale=5 ─────────────────────────────────────────────
model_b = CopulaTabICLv2(**BASE_CFG, phi_x_scale=5.0)
model_b.load_state_dict(raw["model_state"], strict=True)
model_b.eval()

attn_b = run_stage3_attn(model_b)
h_b = entropy_stats(attn_b, "B — checkpoint  phi_x_scale=5.0")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'═'*50}")
print("  Summary: normalised entropy H/log(n_support)")
print(f"  {'blk':>3}  {'scale=1':>10}  {'scale=5':>10}  {'delta':>8}")
print(f"{'─'*50}")
for i, (ha, hb) in enumerate(zip(h_a, h_b)):
    delta = hb - ha
    print(f"  {i:>3}  {ha:>10.4f}  {hb:>10.4f}  {delta:>+8.4f}")
print(f"{'═'*50}")
print()
print("Interpretation:")
print("  H_norm ≈ 1.0  → attention is near-uniform (over-smoothed)")
print("  H_norm ≪ 1.0  → attention is peaked (model uses X to select support)")
print("  Negative delta → phi_x_scale=5 sharpens attention (desired direction)")
