"""
inspect_models.py — Architecture overview for TabICL (Phase 1) and CopulaTransformer (Phase 2).

Prints:
  - High-level architecture summary
  - Per-component and per-layer parameter counts
  - Breakdown by attention type (feature / instance / ICL) and FFN

Usage:
  cd /home/mtristan/Documents/Research/Multivariate_Tab-ICL/copula
  python scratch/inspect_models.py
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TABICL_SRC = os.path.join(_ROOT, "tabicl_upstream", "src")
_SRC = os.path.join(_ROOT, "src")
if _TABICL_SRC not in sys.path:
    sys.path.insert(0, _TABICL_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 80
SEP2 = "-" * 80

def _params(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))

def _fmt(n: int) -> str:
    """Format integer with thousands separator and M/K suffix."""
    if n >= 1_000_000:
        return f"{n:>12,}  ({n / 1e6:.3f} M)"
    elif n >= 1_000:
        return f"{n:>12,}  ({n / 1e3:.1f} K)"
    return f"{n:>12,}"

def _row(label: str, n: int, indent: int = 0) -> str:
    prefix = "  " * indent
    return f"  {prefix}{label:<50s}  {_fmt(n)}"

def _header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def _subheader(title: str) -> None:
    print(f"\n  {SEP2}")
    print(f"  {title}")
    print(f"  {SEP2}")


# ---------------------------------------------------------------------------
# 1. TabICL (frozen Phase-1 model)
# ---------------------------------------------------------------------------

def inspect_tabicl() -> None:
    """Load and inspect the upstream TabICL regressor."""
    _header("TabICL  —  Phase 1 (Frozen Regressor)")

    try:
        from tabicl._model.tabicl import TabICL  # type: ignore
    except ImportError as exc:
        print(f"  [SKIP] Cannot import TabICL: {exc}")
        return

    # ------------------------------------------------------------------ #
    # Build model with default hyper-parameters (matches the checkpoint)  #
    # ------------------------------------------------------------------ #
    model = TabICL(
        max_classes=0,           # regression mode
        num_quantiles=999,
        embed_dim=128,
        col_num_blocks=3,
        col_nhead=8,
        col_num_inds=128,
        col_affine=False,
        col_feature_group="same",
        col_feature_group_size=3,
        col_target_aware=True,
        col_ssmax="qassmax-mlp-elementwise",
        row_num_blocks=3,
        row_nhead=8,
        row_num_cls=4,
        row_rope_base=100_000,
        row_rope_interleaved=False,
        icl_num_blocks=12,
        icl_nhead=8,
        icl_ssmax="qassmax-mlp-elementwise",
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        bias_free_ln=False,
    )

    embed_dim  = model.embed_dim
    icl_dim    = embed_dim * model.row_num_cls   # = 128 * 4 = 512
    ff_factor  = model.ff_factor

    total = _params(model, trainable_only=False)

    print(f"\n  Architecture class   : TabICL")
    print(f"  Task                 : Regression (quantile prediction)")
    print(f"  embed_dim            : {embed_dim}")
    print(f"  ICL dim (row_num_cls): {icl_dim}  (= embed_dim × {model.row_num_cls} CLS tokens)")
    print(f"  ff_factor            : {ff_factor}")
    print(f"  FFN hidden (col/row) : {embed_dim * ff_factor}")
    print(f"  FFN hidden (ICL)     : {icl_dim * ff_factor}")
    print(f"  Total parameters     : {_fmt(total)}")

    # ---- Stage 1: Column Embedding ----------------------------------------
    _subheader("Stage 1 — Column Embedding  (col_embedder)")
    col = model.col_embedder
    col_total = _params(col, trainable_only=False)
    print(_row("col_embedder (total)", col_total))

    # Induced Self-Attention blocks
    if hasattr(col, 'blocks') and col.blocks:
        for i, blk in enumerate(col.blocks):
            blk_p = _params(blk, trainable_only=False)
            print(_row(f"  block[{i}] (InducedSelfAttn)", blk_p, indent=1))
            # attn1 = induced → input cross-attn
            if hasattr(blk, 'multihead_attn1'):
                attn1_p = _params(blk.multihead_attn1, trainable_only=False)
                print(_row(f"    multihead_attn1 (induced→input)", attn1_p, indent=2))
            # attn2 = input → induced cross-attn
            if hasattr(blk, 'multihead_attn2'):
                attn2_p = _params(blk.multihead_attn2, trainable_only=False)
                print(_row(f"    multihead_attn2 (input→induced)", attn2_p, indent=2))
    else:
        # Try iterating named children
        for name, child in col.named_children():
            print(_row(f"  {name}", _params(child, trainable_only=False), indent=1))

    # ---- Stage 2: Row Interaction -----------------------------------------
    _subheader("Stage 2 — Row Interaction  (row_interactor)")
    row = model.row_interactor
    row_total = _params(row, trainable_only=False)
    print(_row("row_interactor (total)", row_total))

    if hasattr(row, 'blocks') and row.blocks:
        for i, blk in enumerate(row.blocks):
            blk_p = _params(blk, trainable_only=False)
            attn_p = _params(blk.attn, trainable_only=False) if hasattr(blk, 'attn') else 0
            ff_p   = (_params(blk.linear1, trainable_only=False) +
                      _params(blk.linear2, trainable_only=False)) if hasattr(blk, 'linear1') else 0
            norm_p = blk_p - attn_p - ff_p
            print(_row(f"  block[{i}]  (row self-attention)", blk_p, indent=1))
            print(_row(f"    attention", attn_p, indent=2))
            print(_row(f"    FFN", ff_p, indent=2))
            print(_row(f"    norms / other", norm_p, indent=2))
    else:
        for name, child in row.named_children():
            print(_row(f"  {name}", _params(child, trainable_only=False), indent=1))

    # ---- Stage 3: ICL (in-context learning) Transformer -------------------
    _subheader("Stage 3 — In-Context Learning Transformer  (icl_predictor)")
    icl = model.icl_predictor
    icl_total = _params(icl, trainable_only=False)
    print(_row("icl_predictor (total)", icl_total))
    print(f"\n    d_model  = {icl_dim}   (embed_dim × row_num_cls)")
    print(f"    n_heads  = {model.icl_nhead}")
    print(f"    n_blocks = {model.icl_num_blocks}")
    print(f"    d_ff     = {icl_dim * ff_factor}")
    print()

    # Iterate blocks
    if hasattr(icl, 'blocks') and icl.blocks:
        blocks = icl.blocks
        # Show first block in detail; all blocks are identical
        blk0   = blocks[0]
        attn_p = _params(blk0.attn, trainable_only=False) if hasattr(blk0, 'attn') else 0
        ff_p   = 0
        if hasattr(blk0, 'linear1'):
            ff_p = (_params(blk0.linear1, trainable_only=False) +
                    _params(blk0.linear2, trainable_only=False))
        norm_p = _params(blk0, trainable_only=False) - attn_p - ff_p

        print(_row(f"  Per block (×{len(blocks)}):", _params(blk0, trainable_only=False), indent=1))
        print(_row(f"    attention (ICL self-attn)", attn_p, indent=2))
        print(_row(f"    FFN", ff_p, indent=2))
        print(_row(f"    norms / other", norm_p, indent=2))
        all_blocks_p = sum(_params(b, trainable_only=False) for b in blocks)
        print(_row(f"  All {len(blocks)} blocks (subtotal)", all_blocks_p, indent=1))
    else:
        for name, child in icl.named_children():
            print(_row(f"  {name}", _params(child, trainable_only=False), indent=1))

    # ---- Summary -----------------------------------------------------------
    _subheader("Summary by Stage")
    print(_row("  col_embedder  (Stage 1)", col_total, indent=1))
    print(_row("  row_interactor (Stage 2)", row_total, indent=1))
    print(_row("  icl_predictor  (Stage 3)", icl_total, indent=1))
    other = total - col_total - row_total - icl_total
    if other:
        print(_row("  other (quantile_dist, etc.)", other, indent=1))
    print(_row("TOTAL TabICL", total))


# ---------------------------------------------------------------------------
# 2. CopulaTransformer (trainable Phase-2 model)
# ---------------------------------------------------------------------------

def inspect_copula() -> None:
    """Instantiate and inspect the CopulaTransformer with the project config values."""
    _header("CopulaTransformer  —  Phase 2 (Trainable)")

    # Read config values (mirrors conf/model/copula_transformer.yaml)
    d_model  = 128
    n_heads  = 8
    n_layers = 6
    p_max    = 20
    d_max    = 8
    rank     = 3
    # d_ff defaults to nearest multiple of 64 above 8/3 * d_model
    d_ff     = max(round(8 / 3 * d_model / 64) * 64, 64)
    dropout  = 0.0

    head_dim = d_model // n_heads
    rank_max = rank  # rank is fixed (not None) in config

    from model import CopulaTransformer   # local src/model.py
    model = CopulaTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        p_max=p_max,
        d_max=d_max,
        rank=rank,
        d_ff=d_ff,
        dropout=dropout,
    )

    total = _params(model)
    S     = p_max + d_max   # max token sequence length per instance

    print(f"\n  Architecture class  : CopulaTransformer")
    print(f"  Task                : Multivariate Gaussian over Z-space (copula)")
    print(f"  d_model             : {d_model}")
    print(f"  n_heads             : {n_heads}  (head_dim = {head_dim})")
    print(f"  n_layers            : {n_layers}")
    print(f"  d_ff                : {d_ff}  (auto = 8/3 × d_model rounded to 64)")
    print(f"  p_max               : {p_max}  (max feature columns, zero-padded)")
    print(f"  d_max               : {d_max}  (max target dimensions)")
    print(f"  rank                : {rank}  (low-rank factor r for covariance V)")
    print(f"  S (tokens/instance) : {S}  (= p_max + d_max)")
    print(f"  Total parameters    : {_fmt(total)}")

    # ---- Input embeddings --------------------------------------------------
    _subheader("Input Embeddings")
    phi_X_p   = _params(model.phi_X)
    phi_Z_p   = _params(model.phi_Z)
    type_enc_p = model.type_enc.numel()
    feat_enc_p = _params(model.feat_enc)
    dim_enc_p  = _params(model.dim_enc)
    mask_tok_p = model.mask_tokens.numel()

    embed_total = phi_X_p + phi_Z_p + type_enc_p + feat_enc_p + dim_enc_p + mask_tok_p

    print(_row("  phi_X  (scalar x → d_model)", phi_X_p, indent=1))
    print(_row("  phi_Z  (scalar z → d_model)", phi_Z_p, indent=1))
    print(_row("  type_enc  (2 × d_model, feat/target type)", type_enc_p, indent=1))
    print(_row("  feat_enc  (p_max × d_model embeddings)", feat_enc_p, indent=1))
    print(_row("  dim_enc   (d_max × d_model embeddings)", dim_enc_p, indent=1))
    print(_row("  mask_tokens (d_max × d_model)", mask_tok_p, indent=1))
    print(_row("  Embeddings subtotal", embed_total))

    # ---- Transformer blocks ------------------------------------------------
    _subheader(f"Transformer Blocks  ({n_layers} × TransformerBlock)")

    # Inspect block[0] in detail (all blocks identical)
    blk0 = model.blocks[0]

    # RMSNorm params: d_model scalars each
    norm1_p = _params(blk0.norm1)
    norm2_p = _params(blk0.norm2)
    norm3_p = _params(blk0.norm3)

    # Feature attention (PyTorch MHA): in_proj_weight + in_proj_bias + out_proj
    feat_attn  = blk0.feat_attn
    feat_attn_p = _params(feat_attn)

    # Feature attention breakdown
    # in_proj_weight: (3*d_model, d_model); in_proj_bias: (3*d_model,)
    # out_proj.weight: (d_model, d_model); out_proj.bias: (d_model,)
    fa_in_proj_p  = feat_attn.in_proj_weight.numel()
    fa_in_bias_p  = feat_attn.in_proj_bias.numel() if feat_attn.in_proj_bias is not None else 0
    fa_out_proj_p = feat_attn.out_proj.weight.numel() + (
        feat_attn.out_proj.bias.numel() if feat_attn.out_proj.bias is not None else 0
    )

    # Instance attention (same structure)
    inst_attn  = blk0.inst_attn
    inst_attn_p = _params(inst_attn)
    ia_in_proj_p  = inst_attn.in_proj_weight.numel()
    ia_in_bias_p  = inst_attn.in_proj_bias.numel() if inst_attn.in_proj_bias is not None else 0
    ia_out_proj_p = inst_attn.out_proj.weight.numel() + (
        inst_attn.out_proj.bias.numel() if inst_attn.out_proj.bias is not None else 0
    )

    # FFN (SwiGLU: w1, w2, w3)
    ffn   = blk0.ffn
    ffn_p = _params(ffn)
    w1_p  = ffn.w1.weight.numel()
    w2_p  = ffn.w2.weight.numel()
    w3_p  = ffn.w3.weight.numel()

    blk0_total = _params(blk0)

    print(f"\n  Block architecture (pre-norm residual, no bias on linear layers):")
    print(f"    1. RMSNorm  →  FeatureAttn  →  residual")
    print(f"    2. RMSNorm  →  InstanceAttn →  residual  (masked for target slots)")
    print(f"    3. RMSNorm  →  SwiGLUFFN   →  residual")
    print()
    print(_row(f"  block[0] total  (all blocks identical)", blk0_total, indent=1))

    print(_row(f"    norm1 (RMSNorm, d_model={d_model})", norm1_p, indent=2))

    print(_row(f"    feat_attn  (FeatureAttn, within-instance)", feat_attn_p, indent=2))
    print(_row(f"      in_proj_weight  (3d_model × d_model = {3*d_model}×{d_model})", fa_in_proj_p, indent=3))
    print(_row(f"      in_proj_bias    (3d_model = {3*d_model})", fa_in_bias_p, indent=3))
    print(_row(f"      out_proj        (d_model × d_model + bias)", fa_out_proj_p, indent=3))

    print(_row(f"    norm2 (RMSNorm, d_model={d_model})", norm2_p, indent=2))

    print(_row(f"    inst_attn  (InstanceAttn, across instances)", inst_attn_p, indent=2))
    print(_row(f"      in_proj_weight  (3d_model × d_model = {3*d_model}×{d_model})", ia_in_proj_p, indent=3))
    print(_row(f"      in_proj_bias    (3d_model = {3*d_model})", ia_in_bias_p, indent=3))
    print(_row(f"      out_proj        (d_model × d_model + bias)", ia_out_proj_p, indent=3))

    print(_row(f"    norm3 (RMSNorm, d_model={d_model})", norm3_p, indent=2))

    print(_row(f"    SwiGLUFFN  (d_model={d_model} → d_ff={d_ff} → d_model)", ffn_p, indent=2))
    print(_row(f"      w1  (gate proj, d_model→d_ff = {d_model}×{d_ff})", w1_p, indent=3))
    print(_row(f"      w2  (value proj, d_model→d_ff = {d_model}×{d_ff})", w2_p, indent=3))
    print(_row(f"      w3  (output proj, d_ff→d_model = {d_ff}×{d_model})", w3_p, indent=3))

    all_blocks_p = sum(_params(b) for b in model.blocks)
    print()
    print(_row(f"  All {n_layers} blocks subtotal", all_blocks_p))

    # ---- Readout heads -----------------------------------------------------
    _subheader("Readout Heads  (query target tokens only)")
    fc_mu_p = _params(model.fc_mu)
    fc_d_p  = _params(model.fc_d)
    fc_V_p  = _params(model.fc_V)
    heads_p = fc_mu_p + fc_d_p + fc_V_p

    print(_row(f"  fc_mu  (d_model→1, mean)",          fc_mu_p, indent=1))
    print(_row(f"  fc_d   (d_model→1, log-variance)",  fc_d_p,  indent=1))
    print(_row(f"  fc_V   (d_model→rank_max={rank_max}, low-rank factor)", fc_V_p,  indent=1))
    print(_row("  Readout heads subtotal", heads_p))

    # ---- Summary -----------------------------------------------------------
    _subheader("Summary by Component")
    print(_row("  Input embeddings", embed_total, indent=1))
    # Break down blocks by sub-type
    feat_attn_all   = feat_attn_p  * n_layers
    inst_attn_all   = inst_attn_p  * n_layers
    ffn_all         = ffn_p        * n_layers
    norms_all       = (norm1_p + norm2_p + norm3_p) * n_layers
    print(_row(f"  Transformer blocks ({n_layers} layers):", all_blocks_p, indent=1))
    print(_row(f"    Feature Attention  ({n_layers}×)", feat_attn_all, indent=2))
    print(_row(f"    Instance Attention ({n_layers}×)", inst_attn_all, indent=2))
    print(_row(f"    SwiGLU FFN         ({n_layers}×)", ffn_all, indent=2))
    print(_row(f"    RMSNorms           ({n_layers}×3)", norms_all, indent=2))
    print(_row("  Readout heads", heads_p, indent=1))
    print(_row("TOTAL CopulaTransformer", total))

    # Percentage breakdown
    print()
    print(f"  {'Component':<45s}  {'% of total':>10s}")
    print(f"  {'-'*57}")
    comps = [
        ("Input embeddings",          embed_total),
        ("Feature Attention (all)",   feat_attn_all),
        ("Instance Attention (all)",  inst_attn_all),
        ("SwiGLU FFN (all)",          ffn_all),
        ("RMSNorms (all)",            norms_all),
        ("Readout heads",             heads_p),
    ]
    for label, cnt in comps:
        pct = 100.0 * cnt / total
        print(f"  {label:<45s}  {pct:>9.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(SEP)
    print("  Model Architecture Inspector")
    print("  Project: Multivariate_Tab-ICL / copula")
    print(SEP)

    inspect_tabicl()
    inspect_copula()

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)
