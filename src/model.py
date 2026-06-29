"""
model.py — CopulaTransformer: Phase 2 model that learns multivariate Gaussian
dependency structure in Z-space.

Architecture overview
---------------------
The CopulaTransformer operates purely in Z-space at runtime (no TabICL
dependency).  Given a context of (X, Z) support pairs it predicts, for each
query instance, a low-rank multivariate Gaussian over the d-dimensional target
vector Z:

    p(Z_query | X_all, Z_support) = N(mu_Z, diag(D) + V V^T)

Token layout (per instance, S = p + d tokens):

    [ φ_X(x_{i,1}), ..., φ_X(x_{i,p}),  φ_Z(z_{i,1}), ..., φ_Z(z_{i,d}) ]
      feat_tok[0]         feat_tok[p-1]    tgt_tok[0]          tgt_tok[d-1]

Each scalar feature x_{i,k} and each scalar Z dimension z_{i,j} get their own
token, enriched by learnable per-slot index embeddings.  This is more
expressive than pooling all features into a single token.

For query instances:
  • Feature tokens φ_X(x_{i,k}) are embedded normally (X is always observed).
  • Target tokens are replaced by learnable mask tokens θ_mask[j].

Processing alternates between:
  • FeatureAttn  — MultiheadAttention within each instance (across S tokens)
  • InstanceAttn — MultiheadAttention across N instances (per token position)
  • SwiGLU FFN   — position-wise feed-forward

Instance attention masking:
  • Feature token slots (0..p-1): query instances ARE visible keys — their
    features are real observations, not masked quantities.
  • Target token slots (p..p+d-1): query instances are NOT valid keys — their
    target tokens carry mask tokens, not observed Z values.  The corresponding
    columns are blocked with -inf.

Readout (query target tokens only) → (mu_Z=0, d_Z, V_Z) via a single linear head fc_V.
mu_Z is fixed at zero and Sigma_ii = 1 (correlation matrix) to satisfy copula constraints.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not re-centre (no mean subtraction), which
    reduces computation while retaining most of the stabilising benefit.

    Args:
        d_model : feature dimension to normalise over.
        eps     : small constant added to the RMS for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension of *x*.

        Args:
            x : (..., d_model)

        Returns:
            Tensor of the same shape as *x*.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block (Noam Shazeer, 2020).

    Computes:  dropout( w3( silu(w1(x)) * w2(x) ) )

    w1, w2 are the gating projections (d_model → d_ff) and w3 is the output
    projection (d_ff → d_model).  The element-wise product of a SiLU-gated
    branch with an ungated branch provides a smooth, expressive non-linearity
    without a bias-free bottleneck.

    Args:
        d_model : input/output feature dimension.
        d_ff    : hidden dimension for the two up-projections.
        dropout : dropout probability applied before the output projection.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU FFN to *x*.

        Args:
            x : (..., d_model)

        Returns:
            Tensor of the same shape as *x*.
        """
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))




class MaskedSelfAttnBlock(nn.Module):
    """Pre-norm self-attention block with an optional additive attention mask.

    All N instance tokens are processed jointly.  Passing a mask with
    ``mask[:, n_support:] = -inf`` reproduces the TF_icl causal structure
    from TabICLv2: support tokens attend only to other support tokens, while
    query tokens attend to all support tokens but not to each other.
    """

    def __init__(self, d_h: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_h)
        self.attn  = nn.MultiheadAttention(d_h, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d_h)
        d_ff       = max(round(8 / 3 * d_h / 64) * 64, 64)
        self.ffn   = SwiGLUFFN(d_h, d_ff, dropout=dropout)

    def forward(
        self,
        x:         torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x : (B, N, d_h);  attn_mask : (N, N) additive mask or None."""
        x_n = self.norm1(x)
        x   = x + self.attn(x_n, x_n, x_n, attn_mask=attn_mask, need_weights=False)[0]
        x   = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# CopulaTabICLv2 — 3-stage architecture faithful to TabICLv2 (Qu et al., 2026)
# ---------------------------------------------------------------------------


class InducingPointBlock(nn.Module):
    """Stage 1 (TF_col) block: two-phase cross-attention through M inducing points.

    Each block owns its own ``inducing_points`` parameter of shape (M, d_model).
    In ``forward`` these are expanded to (B_cols, M, d_model) where
    B_cols = B × n_cols, so each (batch, column) pair gets an independent copy
    that evolves within the block without cross-block state.

    Phase 1 aggregates N row representations into M inducing vectors.
    Phase 2 broadcasts the updated inducing state back to data.
    A SwiGLU FFN is applied to the updated data.

    Complexity per column: O(N·M) instead of O(N²).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(n_inducing, d_model))
        # Phase 1: inducing ← data
        self.norm_d1 = RMSNorm(d_model)
        self.norm_i1 = RMSNorm(d_model)
        self.ca_i_from_d = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Phase 2: data ← inducing
        self.norm_d2 = RMSNorm(d_model)
        self.norm_i2 = RMSNorm(d_model)
        self.ca_d_from_i = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # FFN on data
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply one inducing-point block.

        Args:
            data : (B_cols, N, d_model) — B_cols = B × n_cols.

        Returns:
            Updated data of the same shape.
        """
        B_cols = data.shape[0]
        # Expand own inducing points to cover every (batch, column) independently
        ind = self.inducing_points.unsqueeze(0).expand(B_cols, -1, -1)

        # Phase 1: inducing reads from data
        ind_out, _ = self.ca_i_from_d(
            self.norm_i1(ind), self.norm_d1(data), self.norm_d1(data)
        )
        ind = ind + ind_out

        # Phase 2: data reads from updated inducing
        data_out, _ = self.ca_d_from_i(
            self.norm_d2(data), self.norm_i2(ind), self.norm_i2(ind)
        )
        data = data + data_out

        # FFN on data
        data = data + self.ffn(self.norm_ffn(data))
        return data


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE) utilities
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim [a, b] → [-b, a]."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x : (..., S, d_head);  cos/sin : (1, 1, S, d_head)."""
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE cos/sin for the TF_row sequence (S = n_cls + p_max).

    Args:
        d_head : head dimension (must be even).
        base   : RoPE base frequency (default 10000).
    """

    def __init__(self, d_head: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin of shape (1, 1, seq_len, d_head) for broadcasting."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)           # (S, d_head//2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (S, d_head)
        return emb.cos()[None, None], emb.sin()[None, None]


class RowAggregatorBlock(nn.Module):
    """Stage 2 (TF_row) block: pre-norm transformer with RoPE on Q and K.

    Uses manual Q/K/V projections so RoPE can be applied before the dot
    product.  ``forward`` handles full self-attention; ``forward_cls_cross_attn``
    handles the last-block CLS-only readout (q=CLS, k=v=all).
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    def _qkv(
        self, x_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to (B_N, h, S, dh)."""
        B_N, S, _ = x_n.shape
        h, dh = self.n_heads, self.d_head
        Q = self.q_proj(x_n).reshape(B_N, S, h, dh).transpose(1, 2)
        K = self.k_proj(x_n).reshape(B_N, S, h, dh).transpose(1, 2)
        V = self.v_proj(x_n).reshape(B_N, S, h, dh).transpose(1, 2)
        return Q, K, V

    def _attn(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Scaled dot-product attention → (B_N, h, Sq, dh)."""
        scale = 1.0 / math.sqrt(self.d_head)
        attn_w = (Q @ K.transpose(-2, -1)) * scale
        attn_w = attn_w.softmax(dim=-1)
        if self.training and self.dropout_p > 0.0:
            attn_w = F.dropout(attn_w, p=self.dropout_p)
        return attn_w @ V

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Full self-attention with RoPE.
        x   : (B_N, S, d_model)
        cos : (1, 1, S, d_head)
        sin : (1, 1, S, d_head)
        """
        B_N, S, d = x.shape
        x_n = self.norm1(x)
        Q, K, V = self._qkv(x_n)
        Q = _apply_rope(Q, cos, sin)
        K = _apply_rope(K, cos, sin)
        attn_out = self._attn(Q, K, V).transpose(1, 2).reshape(B_N, S, d)
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_cls_cross_attn(
        self,
        x: torch.Tensor,
        n_cls: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Last-block CLS cross-attention: q=CLS, k=v=all.
        x   : (B_N, S, d_model)
        cos : (1, 1, S, d_head)
        sin : (1, 1, S, d_head)
        Returns: (B_N, n_cls, d_model)
        """
        B_N, S, d = x.shape
        x_n = self.norm1(x)
        Q, K, V = self._qkv(x_n)
        Q_cls = _apply_rope(Q[:, :, :n_cls, :], cos[:, :, :n_cls, :], sin[:, :, :n_cls, :])
        K = _apply_rope(K, cos, sin)
        attn_out = self._attn(Q_cls, K, V).transpose(1, 2).reshape(B_N, n_cls, d)
        cls_out = x[:, :n_cls, :] + self.out_proj(attn_out)
        cls_out = cls_out + self.ffn(self.norm2(cls_out))
        return cls_out


class ICLBlock(nn.Module):
    """Stage 3 (TF_icl) block: row-level ICL with instance-awareness masking.

    Operates on row embeddings (B, N, d_icl) where d_icl = n_cls * d_model.
    Test rows (n_support..N-1) are blocked as keys so they are never attended to.

    Uses SSMax (Scalable Softmax, TabICLv2): the attention temperature is
    scaled by a learnable per-head factor s times log(N), so the softmax
    entropy does not collapse when the sequence length grows.
    Effective scale = softplus(ssmax_log_s) * log(N) / sqrt(d_head).
    """

    def __init__(
        self, d_icl: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_icl // n_heads
        self.dropout = dropout

        self.norm1   = RMSNorm(d_icl)
        # Manual Q/K/V projections so we can intercept before softmax.
        self.q_proj  = nn.Linear(d_icl, d_icl, bias=False)
        self.k_proj  = nn.Linear(d_icl, d_icl, bias=False)
        self.v_proj  = nn.Linear(d_icl, d_icl, bias=False)
        self.out_proj = nn.Linear(d_icl, d_icl, bias=False)
        # SSMax: one learnable log-scale per head.  softplus(0) ≈ 0.693, so the
        # initial effective scale is ~0.693 * log(N) / sqrt(d_head).
        self.ssmax_log_s = nn.Parameter(torch.zeros(n_heads))

        self.norm2 = RMSNorm(d_icl)
        self.ffn   = SwiGLUFFN(d_icl, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        n_support: int,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Args:
        x                  : (B, N, d_icl).
        n_support          : number of support instances.
        return_attn_weights: if True, return (x, attn_weights) where
                             attn_weights is (B, N, N) averaged over heads.
        """
        B, N, d = x.shape
        h, dh = self.n_heads, self.d_head

        # Build causal mask: query instances cannot be keys.
        mask = torch.zeros(N, N, dtype=x.dtype, device=x.device)
        if n_support < N:
            mask[:, n_support:] = float("-inf")

        x_n = self.norm1(x)

        # Project and reshape to (B, h, N, dh)
        def _proj(lin):
            return lin(x_n).reshape(B, N, h, dh).transpose(1, 2)

        Q = _proj(self.q_proj)   # (B, h, N, dh)
        K = _proj(self.k_proj)
        V = _proj(self.v_proj)

        # SSMax scale: softplus(s) * log(N) replaces the standard 1/sqrt(dh).
        # This increases the attention temperature with sequence length, preventing
        # softmax collapse on long contexts.
        s     = F.softplus(self.ssmax_log_s).view(1, h, 1, 1)  # (1, h, 1, 1)
        scale = s * math.log(max(N, 1)) / math.sqrt(dh)        # (1, h, 1, 1)

        # Attention logits with SSMax scale
        attn_logits = (Q @ K.transpose(-2, -1)) * scale        # (B, h, N, N)

        attn_logits = attn_logits + mask                        # broadcast (N,N)
        attn_w_full = attn_logits.softmax(dim=-1)              # (B, h, N, N)

        if self.training and self.dropout > 0.0:
            attn_w_full = F.dropout(attn_w_full, p=self.dropout)

        attn_out = attn_w_full @ V                              # (B, h, N, dh)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, d)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        if return_attn_weights:
            # Average over heads for compatibility
            return x, attn_w_full.mean(dim=1)
        return x


class _MLP2(nn.Module):
    """Two-layer MLP with GELU activation: Linear → GELU → Linear."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class CopulaTabICLv2(nn.Module):
    """CopulaTabICLv2 — 3-stage architecture adapted from TabICLv2 for copula modelling.

    Adapts the TabICLv2 architecture (Qu et al., 2026) to the Phase-2 copula learning
    setting: given support (X, Z) pairs and query X vectors, predict per-query low-rank
    Gaussian parameters in Z-space.

        p(Z_q | X_all, Z_support) = N(mu_Z, diag(d_Z) + V_Z V_Z^T)

    **Stage 1 — TF_col** (column-wise, N rows per column):
      The Z target of each support row is embedded first (Embed_TAE(Z_i) → d_model)
      and added to all p_max feature token positions before Stage 1.  Then each of
      the p_max feature columns is processed independently via inducing-point
      cross-attention (Perceiver/Set-Transformer style, O(N·M) per column).
      Each InducingPointBlock owns its own inducing points expanded to (B*p_max, M, D).

    **Stage 2 — TF_row** (row-wise, n_cls + p_max tokens):
      Four learnable [CLS] tokens are prepended to each row's p_max column embeddings
      and processed by a non-causal transformer.  Concatenated CLS outputs form a
      fixed d_icl = n_cls * d_model = 512-dimensional row embedding.

    **Stage 3 — TF_icl** (instance-level ICL, d_icl-dim row embeddings):
      Support row embeddings receive Embed_ICL(Z_i) injection.  An ICL transformer
      with instance-awareness masking processes all N row embeddings; test instances
      attend only to support instances.

    Readout: a 2-layer MLP (d_icl → 1024 → d_max) for each of mu/d/V.

    Args:
        d_model    : TF_col / TF_row model dimension (default 128).
        n_heads    : attention heads for all stages (default 8).
        n_layers_s1: Stage 1 InducingPointBlock layers (default 3).
        n_layers_s2: Stage 2 RowAggregatorBlock layers (default 3).
        n_layers_s3: Stage 3 ICLBlock layers (default 6).
        n_inducing : inducing vectors per InducingPointBlock (default 128).
        n_cls      : [CLS] tokens; d_icl = n_cls * d_model (default 4).
        p_max      : maximum number of input feature columns (default 20).
        d_max      : maximum number of target dimensions (default 8).
        rank       : low-rank factor size r; None → max(1, floor(sqrt(d))).
        d_ff       : SwiGLU hidden size for Stage 1/2; None → nearest 64 above 8/3*d.
        dropout    : dropout probability (default 0.0).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers_s1: int = 3,
        n_layers_s2: int = 3,
        n_layers_s3: int = 6,
        n_inducing: int = 128,
        n_cls: int = 4,
        p_max: int = 20,
        d_max: int = 8,
        rank: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        phi_x_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if d_ff is None:
            d_ff = max(round(8 / 3 * d_model / 64) * 64, 64)

        d_icl: int = n_cls * d_model  # Stage 3 dim (e.g. 512)
        d_ff_icl: int = max(round(8 / 3 * d_icl / 64) * 64, 64)
        d_ff_head: int = 1024
        rank_max: int = max(1, int(math.sqrt(d_max))) if rank is None else rank

        # ---- Target-aware embeddings (computed before feature embedding) ------
        # Inputs include vech(z z^T) so the model can see pairwise covariance signal
        # per support instance: e.g. "z_2 * z_5 is large" → dims 2,5 are correlated.
        # Without outer products the model only sees marginal Z values and cannot
        # learn which pairs of target dimensions covary.
        d_vech = d_max * (d_max + 1) // 2
        self.d_vech = d_vech
        # Embed_TAE: [Z_sup, vech(Z_sup ⊗ Z_sup)] → d_model
        self.embed_tae = nn.Linear(d_max + d_vech, d_model)
        # Embed_ICL: same input → d_icl
        self.embed_icl = nn.Linear(d_max + d_vech, d_icl)
        # Per-dimension gates (d_icl-vector) instead of a single scalar.
        # A scalar gate aggregates its gradient over B × N_sup × d_icl terms —
        # signs cancel across dimensions, leaving near-zero grad → gate never
        # opens, embed_icl never trains.  A vector gate avoids this: each
        # element's gradient sums over B × N_sup only, no cross-dim cancellation.
        # Init at 0 → sigmoid(0)=0.5: gentler ramp than ones (sigmoid≈0.73) so
        # feature representations mature before Z injection dominates.
        self.icl_gate_sup = nn.Parameter(torch.zeros(d_icl))  # support injection, sigmoid(0)=0.5

        # ---- Feature embedding -----------------------------------------------
        # Group features in circular triplets (k, k+1, k+2) mod p_max — adapted from
        # TabICLv2's col_feature_group="same" with group_size=3.  Each token already
        # sees two neighbours before TF_col, providing local inductive bias.
        self.phi_X = nn.Linear(3, d_model, bias=False)

        # ---- Stage 1: TF_col — per-column inducing-point blocks ---------------
        # Each block owns its own inducing_points; no shared state across blocks
        self.s1_blocks = nn.ModuleList(
            [
                InducingPointBlock(d_model, n_heads, n_inducing, d_ff, dropout)
                for _ in range(n_layers_s1)
            ]
        )

        # ---- Stage 2: TF_row — row-wise aggregation via CLS tokens ------------
        self.cls_tokens = nn.Parameter(torch.empty(n_cls, d_model))
        # RoPE provides positional information for Stage 2 tokens (CLS at 0..n_cls-1,
        # features at n_cls..n_cls+p_max-1), replacing the fixed row_pos_emb.
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.s2_blocks = nn.ModuleList(
            [
                RowAggregatorBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers_s2)
            ]
        )
        self.s2_norm = RMSNorm(d_model)

        # ---- Stage 3: TF_icl — ICL over d_icl-dim row embeddings -------------
        self.s3_blocks = nn.ModuleList(
            [ICLBlock(d_icl, n_heads, d_ff_icl, dropout) for _ in range(n_layers_s3)]
        )
        self.s3_norm = RMSNorm(d_icl)

        # ---- Per-dimension conditioning for readout --------------------------
        # Learnable embedding for each target dimension index, concatenated with
        # the row embedding before fc_V. This breaks the symmetry that forces all
        # d rows of U to be identical projections of the same row embedding.
        d_dim_emb = d_icl // 4  # 128 for default d_icl=512
        self.dim_emb = nn.Parameter(torch.empty(d_max, d_dim_emb))

        # ---- Readout head (2-layer MLP) --------------------------------------
        # Input: row embedding (d_icl) + dimension embedding (d_dim_emb)
        self.fc_V = _MLP2(d_icl + d_dim_emb, d_ff_head, rank_max)

        # ---- Config ----------------------------------------------------------
        self.phi_x_scale = phi_x_scale
        self.p_max = p_max
        self.d_max = d_max
        self.n_cls = n_cls
        self.d_icl = d_icl
        self.d_dim_emb = d_icl // 4
        self.rank = rank
        self.rank_max = rank_max
        self.d_model = d_model

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        # Orthogonal init for dim_emb: each dimension gets a distinct direction.
        # Scale to match query_emb contribution through fc_V.fc1 so that the
        # per-dimension signal is competitive with the shared query signal.
        # query_emb magnitude at fc1 ≈ 0.1 * sqrt(d_icl); dim_emb should match:
        # norm_target = sqrt(d_icl) so scale = sqrt(d_icl) / (orthogonal_row_norm).
        nn.init.orthogonal_(self.dim_emb)
        ortho_row_norm = math.sqrt(self.d_dim_emb / self.d_max)  # orthogonal row norm
        scale = math.sqrt(self.d_icl) / ortho_row_norm
        self.dim_emb.data *= scale * 0.25  # 0.25 keeps dim contribution at ~25% of query
        for block in self.s1_blocks:
            nn.init.trunc_normal_(block.inducing_points, std=0.02)
        # embed_tae: stronger init (0.1 vs default 0.02) so the Z-target signal
        # is competitive with X features through Stages 1+2.
        nn.init.normal_(self.embed_tae.weight, std=0.1)
        nn.init.zeros_(self.embed_tae.bias)
        nn.init.normal_(self.embed_icl.weight, std=0.1)
        nn.init.zeros_(self.embed_icl.bias)
        # fc_V readout: calibrate output variance so ||U_i||² ≈ 1 at init,
        # independent of rank_max.  Without rank-aware scaling, ||U_i||² grows
        # linearly with rank_max (each extra component adds variance), causing
        # C_diag → 0 and R → 11^T for large ranks.
        # Fix: fc2 std = 1/sqrt(d_ff_head * rank_max) keeps ||U_i||² ≈ const.
        nn.init.normal_(self.fc_V.fc1.weight, std=0.1)
        nn.init.zeros_(self.fc_V.fc1.bias)
        nn.init.normal_(
            self.fc_V.fc2.weight,
            std=1.0 / math.sqrt(self.fc_V.fc2.in_features * self.rank_max),
        )
        nn.init.zeros_(self.fc_V.fc2.bias)

    def forward(
        self,
        X_all: torch.Tensor,
        Z_all: torch.Tensor,
        n_support: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict conditional low-rank Gaussian parameters for query instances.

        Args:
            X_all     : (B, N, p) — feature vectors for all N instances.
            Z_all     : (B, N, d) — Z values; only first n_support rows observed.
            n_support : number of support instances.

        Returns:
            mu_Z : (B, n_query, d)    — conditional mean.
            d_Z  : (B, n_query, d)    — diagonal variance.
            V_Z  : (B, n_query, d, r) — low-rank factor.
        """
        B, N, p = X_all.shape
        d = Z_all.shape[-1]
        n_query = N - n_support

        if self.rank is None:
            r = max(1, int(math.sqrt(d)))
        else:
            r = self.rank
        r = min(r, self.rank_max)

        # ------------------------------------------------------------------
        # 0. Target-aware embedding: computed BEFORE feature embedding
        # ------------------------------------------------------------------
        Z_sup = Z_all[:, :n_support, :]
        if d < self.d_max:
            Z_sup_pad = F.pad(Z_sup, (0, self.d_max - d))  # (B, n_sup, d_max)
        else:
            Z_sup_pad = Z_sup

        # Include pairwise outer product vech(z z^T) so the model sees which
        # target dimensions covary in each support instance.
        outer = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(
            -2
        )  # (B, n_sup, d_max, d_max)
        tril_i, tril_j = torch.tril_indices(
            self.d_max, self.d_max, offset=0, device=Z_sup_pad.device
        )
        vech = outer[..., tril_i, tril_j]  # (B, n_sup, d_vech)
        tae_in = torch.cat([Z_sup_pad, vech], dim=-1)  # (B, n_sup, d_max + d_vech)

        tae = self.embed_tae(tae_in)  # (B, n_sup, d_model)
        icl_emb = self.embed_icl(tae_in)  # (B, n_sup, d_icl)

        # ------------------------------------------------------------------
        # 1. Feature embedding
        # ------------------------------------------------------------------
        if p > self.p_max:
            X_in = X_all[..., : self.p_max]
        elif p < self.p_max:
            X_in = F.pad(X_all, (0, self.p_max - p))
        else:
            X_in = X_all

        # Circular triplet grouping: token k sees (x_k, x_{k+1}, x_{k+2}) mod p_max
        _idx = torch.arange(self.p_max, device=X_in.device)
        X_grouped = torch.stack([
            X_in,
            X_in[..., (_idx + 1) % self.p_max],
            X_in[..., (_idx + 2) % self.p_max],
        ], dim=-1)                                                   # (B, N, p_max, 3)
        E1 = self.phi_X(X_grouped) * self.phi_x_scale               # (B, N, p_max, d_model)

        # Add target-aware embedding to all feature tokens of support rows
        E2 = E1.clone()
        E2[:, :n_support, :, :] = E2[:, :n_support, :, :] + tae.unsqueeze(2)

        # ------------------------------------------------------------------
        # 2. Stage 1: TF_col — column-wise inducing-point attention
        # ------------------------------------------------------------------
        # Each column processed independently: (B, N, p_max, D) → (B*p_max, N, D)
        B_cols = B * self.p_max
        data = E2.permute(0, 2, 1, 3).reshape(B_cols, N, self.d_model)

        for block in self.s1_blocks:
            data = block(data)  # inducing points are internal to each block

        # (B*p_max, N, D) → (B, N, p_max, D)
        feat_emb = data.reshape(B, self.p_max, N, self.d_model).permute(0, 2, 1, 3)

        # ------------------------------------------------------------------
        # 3. Stage 2: TF_row — row-wise aggregation via CLS tokens
        # ------------------------------------------------------------------
        cls_exp = self.cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        row_tok = torch.cat([cls_exp, feat_emb], dim=2)  # (B, N, n_cls+p_max, D)
        S = row_tok.shape[2]
        row_tok = row_tok.reshape(B * N, S, self.d_model)

        # RoPE cos/sin for sequence length S = n_cls + p_max.
        # CLS tokens at positions 0..n_cls-1, feature tokens at n_cls..S-1.
        cos, sin = self.rope(S, row_tok.device)                 # (1,1,S,d_head)

        # All blocks except the last: full self-attention with RoPE.
        # Last block: q = CLS tokens only, k = v = full sequence — adapted from
        # TabICLv2's RowInteraction.  RoPE is applied with matching position indices.
        for block in self.s2_blocks[:-1]:
            row_tok = block(row_tok, cos, sin)

        cls_out_raw = self.s2_blocks[-1].forward_cls_cross_attn(
            row_tok, self.n_cls, cos, sin
        )                                                       # (B*N, n_cls, D)

        cls_out_raw = cls_out_raw.reshape(B, N, self.n_cls, self.d_model)
        cls_out = self.s2_norm(cls_out_raw)                     # (B, N, n_cls, D)
        row_emb = cls_out.reshape(B, N, self.d_icl)            # (B, N, d_icl)

        # ------------------------------------------------------------------
        # 4. Stage 3: TF_icl — ICL over row embeddings
        # ------------------------------------------------------------------
        row_emb = row_emb.clone()

        # Support rows: inject per-instance Z embedding (vector gate, no cancellation)
        gate_sup = torch.sigmoid(self.icl_gate_sup)  # (d_icl,)
        row_emb[:, :n_support, :] = row_emb[:, :n_support, :] + gate_sup * icl_emb

        for block in self.s3_blocks:
            row_emb = block(row_emb, n_support)

        row_emb = self.s3_norm(row_emb)

        query_emb = row_emb[:, n_support:, :]  # (B, n_query, d_icl)

        # ------------------------------------------------------------------
        # 5. Readout: per-dimension MLP on query row embeddings
        # ------------------------------------------------------------------

        # Tile row embedding over d_max dimensions, then concat per-dim embedding.
        # This breaks the symmetry: each dimension gets a distinct input to fc_V,
        # allowing it to produce different U rows for each target dimension.
        query_exp = query_emb.unsqueeze(2).expand(
            B, n_query, self.d_max, -1
        )  # (B, n_query, d_max, d_icl)
        dim_exp = (
            self.dim_emb.unsqueeze(0).unsqueeze(0).expand(B, n_query, -1, -1)
        )  # (B, n_query, d_max, d_dim_emb)
        head_in = torch.cat(
            [query_exp, dim_exp], dim=-1
        )  # (B, n_query, d_max, d_icl+d_dim_emb)

        # fc_V maps each (d_icl+d_dim_emb)-dim vector to rank_max scalars
        U_all = self.fc_V(head_in)  # (B, n_query, d_max, rank_max)

        # Slice to actual (d, r) and apply Woodbury reparameterisation.
        #
        # Copula constraints: mu_Z = 0 and Sigma_ii = 1 (correlation matrix).
        # After the probit-PIT, z_{i,j} = Phi^{-1}(F_j(y_{i,j}|x_i)) has
        # standard-normal marginals: E[z_j|x_i] = 0 and Var[z_j|x_i] = 1.
        # The Gaussian copula density is
        #
        #   -log c(u) = 1/2 log|R| + 1/2 z^T (R^{-1} - I) z
        #
        # which requires R to be a correlation matrix (R_{ii} = 1, mu = 0).
        # Setting s_Z = 1 enforces this:
        #   Sigma_ii = s_Z^2 * (C_diag_i + ||W_i||^2) = 1 * 1 = 1.

        mu_Z = torch.zeros(
            B, n_query, d, dtype=query_emb.dtype, device=query_emb.device
        )

        # s_Z = 1 (correlation matrix constraint, kept explicit for clarity)
        s_Z = torch.ones(B, n_query, d, dtype=query_emb.dtype, device=query_emb.device)
        U = U_all[..., :d, :r]

        # Woodbury decomposition: R = diag(C_diag) + W W^T
        # C_diag_i = 1/(1+||U_i||^2),  W_i = U_i/sqrt(1+||U_i||^2)
        # => R_{ii} = C_diag_i + ||W_i||^2 = 1  (verified by construction)
        U_sq_norm = (U**2).sum(dim=-1)  # (B, n_query, d)
        C_diag = 1.0 / (1.0 + U_sq_norm)  # (B, n_query, d)
        W = U / torch.sqrt(1.0 + U_sq_norm.unsqueeze(-1))  # (B, n_query, d, r)

        d_Z = (s_Z**2) * C_diag  # = C_diag since s_Z = 1
        V_Z = s_Z.unsqueeze(-1) * W  # = W since s_Z = 1

        return mu_Z, d_Z, V_Z


def build_copula_tabicl_v2(cfg) -> CopulaTabICLv2:
    """Instantiate a CopulaTabICLv2 from a Hydra DictConfig.

    Expected config keys under ``cfg.model``:

    ==================  ======================================================
    Key                 Description
    ==================  ======================================================
    d_model             TF_col / TF_row embedding dimension.
    n_heads             Attention heads for all stages.
    n_layers_s1         Stage 1 InducingPointBlock layers.
    n_layers_s2         Stage 2 RowAggregatorBlock layers.
    n_layers_s3         Stage 3 ICLBlock layers.
    n_inducing          Number of inducing vectors per InducingPointBlock.
    n_cls               Number of [CLS] tokens; d_icl = n_cls * d_model.
    p_max               Maximum number of input features.
    d_max               Maximum number of target dimensions.
    rank                Low-rank factor size r. Pass *null* for sqrt(d) auto.
    d_ff                (optional) SwiGLU hidden size for Stage 1/2; None=auto.
    dropout             (optional, default 0.0) Dropout probability.
    ==================  ======================================================
    """
    mcfg = cfg.model

    rank: Optional[int] = None if mcfg.rank is None else int(mcfg.rank)
    d_ff: Optional[int] = (
        int(mcfg.d_ff) if getattr(mcfg, "d_ff", None) is not None else None
    )
    dropout: float = float(getattr(mcfg, "dropout", 0.0))

    model = CopulaTabICLv2(
        d_model=int(mcfg.d_model),
        n_heads=int(mcfg.n_heads),
        n_layers_s1=int(mcfg.n_layers_s1),
        n_layers_s2=int(mcfg.n_layers_s2),
        n_layers_s3=int(mcfg.n_layers_s3),
        n_inducing=int(mcfg.n_inducing),
        n_cls=int(mcfg.n_cls),
        p_max=int(mcfg.p_max),
        d_max=int(mcfg.d_max),
        rank=rank,
        d_ff=d_ff,
        dropout=dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _mname = getattr(mcfg, "name", "copula_tabicl_v2")
    print(
        f"[{_mname}] d_model={mcfg.d_model}  d_icl={model.d_icl}  "
        f"n_heads={mcfg.n_heads}  "
        f"s1={mcfg.n_layers_s1}/s2={mcfg.n_layers_s2}/s3={mcfg.n_layers_s3}  "
        f"n_inducing={mcfg.n_inducing}  n_cls={mcfg.n_cls}  "
        f"p_max={mcfg.p_max}  d_max={mcfg.d_max}  rank={rank}  "
        f"|  params={n_params:,}"
    )

    return model


# ---------------------------------------------------------------------------
# CopulaTabICL adapter — wraps CopulaTabICL to match (X_all, Z_all, n_support) interface
# ---------------------------------------------------------------------------


class _CopulaTabICLWrapper(nn.Module):
    """Wraps CopulaTabICL so it matches the (mu_Z, d_Z, V_Z) interface used by train.py.

    CopulaTabICL.forward(X, Z_train) → (W_tilde, D_tilde)
    This wrapper adapts the call to (X_all, Z_all, n_support) → (mu_Z, d_Z, V_Z).
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        X_all: torch.Tensor,
        Z_all: torch.Tensor,
        n_support: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = X_all.shape
        d = Z_all.shape[-1]
        n_query = N - n_support
        Z_train = Z_all[:, :n_support]
        W_tilde, D_tilde = self.inner(X_all, Z_train)
        mu_Z = torch.zeros(B, n_query, d, device=X_all.device, dtype=X_all.dtype)
        return mu_Z, D_tilde, W_tilde


def build_copula_tabicl(cfg) -> _CopulaTabICLWrapper:
    """Instantiate a CopulaTabICL from a Hydra DictConfig, wrapped for train.py.

    Expected config keys under ``cfg.model``:
      d, k, embed_dim, col_num_blocks, col_nhead, col_num_inds,
      row_num_blocks, row_nhead, row_num_cls, icl_num_blocks, icl_nhead,
      dropout (optional), pre_icl_aux (optional).
    """
    from tabicl_archi import CopulaTabICL  # imported lazily to avoid TabICL overhead

    mcfg = cfg.model
    tabicl_kwargs = dict(
        embed_dim=int(mcfg.embed_dim),
        col_num_blocks=int(mcfg.col_num_blocks),
        col_nhead=int(mcfg.col_nhead),
        col_num_inds=int(mcfg.col_num_inds),
        row_num_blocks=int(mcfg.row_num_blocks),
        row_nhead=int(mcfg.row_nhead),
        row_num_cls=int(mcfg.row_num_cls),
        icl_num_blocks=int(mcfg.icl_num_blocks),
        icl_nhead=int(mcfg.icl_nhead),
        dropout=float(getattr(mcfg, "dropout", 0.0)),
    )
    pre_icl_aux = bool(getattr(mcfg, "pre_icl_aux", False))

    inner = CopulaTabICL(
        d=int(mcfg.d),
        k=int(mcfg.k),
        pre_icl_aux=pre_icl_aux,
        **tabicl_kwargs,
    )

    model = _CopulaTabICLWrapper(inner)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    d_icl = int(mcfg.embed_dim) * int(mcfg.row_num_cls)
    print(
        f"[copula_tabicl]  d={mcfg.d}  k={mcfg.k}  embed_dim={mcfg.embed_dim}  "
        f"d_icl={d_icl}  col_blocks={mcfg.col_num_blocks}  "
        f"row_blocks={mcfg.row_num_blocks}  icl_blocks={mcfg.icl_num_blocks}  "
        f"|  params={n_params:,}"
    )
    return model


_BUILD_DISPATCH = {
    "copula_tabicl_v2":   build_copula_tabicl_v2,
    "tabicl-archi":       build_copula_tabicl,
}


def build_copula_transformer(cfg):
    """Dispatch to the correct builder based on cfg.model.name."""
    name = getattr(cfg.model, "name", "copula_tabicl_v2")
    builder = _BUILD_DISPATCH.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown model name {name!r}. Known: {list(_BUILD_DISPATCH)}"
        )
    return builder(cfg)
