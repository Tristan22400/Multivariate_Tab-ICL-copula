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


# ---------------------------------------------------------------------------
# ICLCorrNet — cross-attention ICL model for Gaussian copula prediction
# ---------------------------------------------------------------------------


class CrossAttnLayer(nn.Module):
    """One cross-attention layer: Q from query-X, K from support-X, V from support-Z⊗Z.

    MHA with pre-norm on QKV, residual + LayerNorm after attention, then FFN.
    """

    def __init__(self, d_h: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_h // n_heads
        self.scale   = self.d_head ** -0.5

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h)

        self.norm1   = nn.LayerNorm(d_h)
        self.norm2   = nn.LayerNorm(d_h)
        self.dropout = nn.Dropout(dropout)
        self.ff      = nn.Sequential(
            nn.Linear(d_h, d_h * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_h * 2, d_h),
        )

    def forward(
        self,
        Q_in: torch.Tensor,
        K_in: torch.Tensor,
        V_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, n_q, _ = Q_in.shape
        N          = K_in.shape[1]
        H, Dh      = self.n_heads, self.d_head

        Q = self.W_q(Q_in).view(B, n_q, H, Dh).transpose(1, 2)
        K = self.W_k(K_in).view(B, N,   H, Dh).transpose(1, 2)
        V = self.W_v(V_in).view(B, N,   H, Dh).transpose(1, 2)

        attn_w = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        ctx    = torch.matmul(attn_w, V).transpose(1, 2).reshape(B, n_q, -1)
        ctx    = self.norm1(Q_in + self.dropout(self.W_o(ctx)))
        ctx    = self.norm2(ctx  + self.ff(ctx))
        return ctx, attn_w.mean(dim=1)   # (B, n_q, d_h), (B, n_q, N)


class ICLCorrNet(nn.Module):
    """Cross-attention ICL model for Gaussian copula parameter prediction.

    Given support (X_sup, Z_sup) and query X_qry, predicts Woodbury params:
        p(Z_q | X_all, Z_sup) = N(0, diag(d_Z) + V_Z V_Z^T)
    with the copula constraint R_ii = 1 (d_Z_i + ||V_Z_i||² = 1).

    Encoder (separated K/V):
      Q = enc_qry(X_qry)                   — query X position
      K = enc_key(X_sup)                   — support X position (Q·K^T similarity)
      V = enc_val(vech(Z_sup ⊗ Z_sup))    — support correlation content

    n_layers stacked CrossAttnLayer, then readout MLP → U → Woodbury.

    Interface: mu_Z, d_Z, V_Z = model(X_all, Z_all, n_support=N)

    Args:
        p_max    : maximum number of input feature columns.
        d_max    : maximum number of target dimensions.
        d_hidden : hidden dimension for encoders and cross-attention.
        n_heads  : number of attention heads (must divide d_hidden).
        n_layers : number of stacked CrossAttnLayer blocks.
        rank     : Woodbury low-rank factor size r.
        dropout  : dropout probability (default 0.0).
    """

    def __init__(
        self,
        p_max:    int,
        d_max:    int,
        d_hidden: int   = 256,
        n_heads:  int   = 8,
        n_layers: int   = 2,
        rank:     int   = 8,
        dropout:  float = 0.0,
    ) -> None:
        super().__init__()
        self.p_max    = p_max
        self.d_max    = d_max
        self.d_hidden = d_hidden
        self.rank_max = rank

        d_vech = d_max * (d_max + 1) // 2

        def mlp(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_out), nn.LayerNorm(d_out),
            )

        self.enc_qry = mlp(p_max,  d_hidden)
        self.enc_key = mlp(p_max,  d_hidden)
        self.enc_val = mlp(d_vech, d_hidden)

        self.layers = nn.ModuleList(
            [CrossAttnLayer(d_hidden, n_heads, dropout) for _ in range(n_layers)]
        )

        # Readout: cat([ctx, Q]) → d_max × rank scalars (the U matrix flat)
        self.readout_U = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_max * rank),
        )

        ti, tj = torch.tril_indices(d_max, d_max)
        self.register_buffer("ti", ti)
        self.register_buffer("tj", tj)

        self._init_weights()

    def _init_weights(self) -> None:
        # Non-zero init to escape the near-diagonal NLL saddle (grad≈0 at U≈0).
        nn.init.normal_(self.readout_U[-1].weight, std=0.1)
        nn.init.zeros_(self.readout_U[-1].bias)

    def forward(
        self,
        X_all:     torch.Tensor,
        Z_all:     torch.Tensor,
        n_support: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X_all     : (B, N, p) — all instances (support + query).
            Z_all     : (B, N, d) — only rows 0..n_support-1 are used.
            n_support : number of support instances.

        Returns:
            mu_Z : (B, n_query, d)     — zero (copula mean fixed at 0).
            d_Z  : (B, n_query, d)     — C_diag = 1/(1+||U_i||²).
            V_Z  : (B, n_query, d, r)  — W = U/sqrt(1+||U||²).  R_ii = 1.
        """
        B, N, p = X_all.shape
        d       = Z_all.shape[-1]
        n_query = N - n_support

        X_sup = X_all[:, :n_support]
        X_qry = X_all[:, n_support:]
        Z_sup = Z_all[:, :n_support]

        # Pad / truncate X to p_max
        if p < self.p_max:
            X_sup = F.pad(X_sup, (0, self.p_max - p))
            X_qry = F.pad(X_qry, (0, self.p_max - p))
        elif p > self.p_max:
            X_sup = X_sup[..., : self.p_max]
            X_qry = X_qry[..., : self.p_max]

        # Pad Z_sup to d_max for vech computation
        Z_sup_pad = F.pad(Z_sup, (0, max(0, self.d_max - d))) if d < self.d_max \
                    else Z_sup[..., : self.d_max]

        outer = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)  # (B,n_sup,d_max,d_max)
        vech  = outer[:, :, self.ti, self.tj]                       # (B,n_sup,d_vech)

        Q = self.enc_qry(X_qry)
        K = self.enc_key(X_sup)
        V = self.enc_val(vech)

        ctx = Q
        for layer in self.layers:
            ctx, _ = layer(ctx, K, V)

        U_flat = self.readout_U(torch.cat([ctx, Q], dim=-1))          # (B,n_qry,d_max*r)
        U = U_flat.reshape(B, n_query, self.d_max, self.rank_max)     # (B,n_qry,d_max,r)
        U = U[..., :d, :]                                              # (B,n_qry,d,r)

        # Woodbury: R_ii = C_diag_i + ||W_i||² = 1 by construction
        U_sq_norm = (U ** 2).sum(dim=-1)
        C_diag    = 1.0 / (1.0 + U_sq_norm)
        W         = U / (1.0 + U_sq_norm.unsqueeze(-1)).sqrt()

        mu_Z = torch.zeros(B, n_query, d, dtype=X_all.dtype, device=X_all.device)
        return mu_Z, C_diag, W


def build_icl_corr_net(cfg) -> ICLCorrNet:
    """Instantiate an ICLCorrNet from a Hydra DictConfig.

    Expected config keys under ``cfg.model``:
      d_hidden, n_heads, n_layers, p_max, d_max, rank, dropout (optional).
    """
    mcfg    = cfg.model
    dropout = float(getattr(mcfg, "dropout", 0.0))

    model = ICLCorrNet(
        p_max    = int(mcfg.p_max),
        d_max    = int(mcfg.d_max),
        d_hidden = int(mcfg.d_hidden),
        n_heads  = int(mcfg.n_heads),
        n_layers = int(mcfg.n_layers),
        rank     = int(mcfg.rank),
        dropout  = dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[ICLCorrNet]  d_hidden={mcfg.d_hidden}  n_heads={mcfg.n_heads}  "
        f"n_layers={mcfg.n_layers}  p_max={mcfg.p_max}  d_max={mcfg.d_max}  "
        f"rank={mcfg.rank}  dropout={dropout}  |  params={n_params:,}"
    )
    return model


# ---------------------------------------------------------------------------
# ICLCorrNetV2 — joint pre-norm transformer with masked instance attention
# ---------------------------------------------------------------------------


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


class ICLCorrNetV2(nn.Module):
    """ICLCorrNetV2 — joint masked transformer with pre-norm support encoding.

    All N instances (support + query) are embedded into a single sequence and
    processed by a stack of pre-norm masked self-attention blocks.  The mask
    blocks query positions from acting as keys, so:

      • Support tokens attend to all other support tokens (dataset-level context).
      • Query tokens attend to all support tokens (but not to other query tokens).

    This collapses the separate "support self-attn → cross-attn" pipeline into
    one unified transformer, avoiding the artificial split between stages.

    **Embedding**:
      • All tokens start from ``enc_x(X_all_padded)`` (X-only, shared encoder).
      • Support tokens additionally receive ``enc_z(vech(Z_sup ⊗ Z_sup))``, which
        injects per-instance pairwise covariance signal before the first block.
        Query tokens get no Z injection (their Z is unobserved at inference time).

    **Readout**: query token positions after the final block → Woodbury MLP.

    Interface: mu_Z, d_Z, V_Z = model(X_all, Z_all, n_support=N)

    Args:
        p_max    : maximum number of input feature columns.
        d_max    : maximum number of target dimensions.
        d_hidden : hidden dimension for all layers.
        n_heads  : attention heads (must divide d_hidden).
        n_layers : number of stacked MaskedSelfAttnBlock blocks.
        rank     : Woodbury low-rank factor size r.
        dropout  : dropout probability (default 0.0).
    """

    def __init__(
        self,
        p_max:    int,
        d_max:    int,
        d_hidden: int   = 256,
        n_heads:  int   = 8,
        n_layers: int   = 6,
        rank:     int   = 8,
        dropout:  float = 0.0,
    ) -> None:
        super().__init__()
        self.p_max    = p_max
        self.d_max    = d_max
        self.d_hidden = d_hidden
        self.rank_max = rank

        d_vech = d_max * (d_max + 1) // 2

        # X encoder shared across all instances
        self.enc_x = nn.Sequential(
            nn.Linear(p_max, d_hidden), RMSNorm(d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Z-covariance encoder for support tokens only.
        # Input is [Z_sup, vech(Z⊗Z)] so the model sees both raw Z values
        # (preserving sign) and pairwise products.  bias=True because the
        # diagonal of vech has non-zero mean (~1 for standard-normal Z).
        self.enc_z        = nn.Linear(d_max + d_vech, d_hidden, bias=True)
        self.z_gate       = nn.Parameter(torch.tensor(-1.0))   # sigmoid≈0.27 at init
        self.sup_emb_norm  = RMSNorm(d_hidden)
        self.query_emb_norm = RMSNorm(d_hidden)

        # Joint masked transformer
        self.blocks = nn.ModuleList([
            MaskedSelfAttnBlock(d_hidden, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_hidden)

        # Readout: query_tok → d_max × rank
        self.readout_U = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_max * rank),
        )

        ti, tj = torch.tril_indices(d_max, d_max)
        self.register_buffer("ti", ti)
        self.register_buffer("tj", tj)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.readout_U[-1].weight, std=0.1)
        nn.init.zeros_(self.readout_U[-1].bias)

    def forward(
        self,
        X_all:     torch.Tensor,
        Z_all:     torch.Tensor,
        n_support: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X_all     : (B, N, p) — all instances (support + query).
            Z_all     : (B, N, d) — only rows 0..n_support-1 are used.
            n_support : number of support instances.

        Returns:
            mu_Z : (B, n_query, d)    — zero (copula mean fixed at 0).
            d_Z  : (B, n_query, d)    — diagonal of the correlation matrix.
            V_Z  : (B, n_query, d, r) — low-rank factor W.  R_ii = 1.
        """
        B, N, p = X_all.shape
        d       = Z_all.shape[-1]
        n_query = N - n_support

        # Pad / truncate X to p_max
        if p < self.p_max:
            X_in = F.pad(X_all, (0, self.p_max - p))
        elif p > self.p_max:
            X_in = X_all[..., : self.p_max]
        else:
            X_in = X_all

        # [Z_sup, vech(Z_sup ⊗ Z_sup)] — raw Z (preserves sign) + pairwise products
        Z_sup = Z_all[:, :n_support]
        if d < self.d_max:
            Z_sup_pad = F.pad(Z_sup, (0, self.d_max - d))
        else:
            Z_sup_pad = Z_sup[..., : self.d_max]
        outer  = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)  # (B,n_sup,d_max,d_max)
        vech   = outer[:, :, self.ti, self.tj]                       # (B,n_sup,d_vech)
        z_in   = torch.cat([Z_sup_pad, vech], dim=-1)                # (B,n_sup,d_max+d_vech)

        # Embed all instances from X; inject Z signal into support tokens.
        # Use cat instead of in-place writes — two consecutive in-place ops on tok
        # cause autograd version-counter errors during backward.
        raw = self.enc_x(X_in)                                        # (B, N, d_hidden)
        z_emb = self.enc_z(z_in)                                      # (B, n_sup, d_hidden)
        sup_tok = self.sup_emb_norm(
            raw[:, :n_support] + torch.sigmoid(self.z_gate) * z_emb
        )                                                              # (B, n_sup, d_hidden)
        qry_tok = self.query_emb_norm(raw[:, n_support:])             # (B, n_qry, d_hidden)
        tok = torch.cat([sup_tok, qry_tok], dim=1)                    # (B, N, d_hidden)

        # Causal mask: block query positions from acting as keys.
        # Shape (N, N); broadcast over B by nn.MultiheadAttention automatically.
        mask = torch.zeros(N, N, dtype=tok.dtype, device=tok.device)
        if n_query > 0:
            mask[:, n_support:] = float("-inf")

        # Joint masked transformer over all N instance tokens
        for block in self.blocks:
            tok = block(tok, attn_mask=mask)

        tok = self.out_norm(tok)

        # Readout from query positions
        query_tok = tok[:, n_support:]                               # (B, n_query, d_hidden)
        U_flat = self.readout_U(query_tok)
        U      = U_flat.reshape(B, n_query, self.d_max, self.rank_max)
        U      = U[..., :d, :]

        # Woodbury reparameterisation: R_ii = C_diag_i + ||W_i||² = 1
        U_sq_norm = (U ** 2).sum(dim=-1)
        C_diag    = 1.0 / (1.0 + U_sq_norm)
        W         = U / (1.0 + U_sq_norm.unsqueeze(-1)).sqrt()

        mu_Z = torch.zeros(B, n_query, d, dtype=X_all.dtype, device=X_all.device)
        return mu_Z, C_diag, W


def build_icl_corr_net_v2(cfg) -> ICLCorrNetV2:
    """Instantiate an ICLCorrNetV2 from a Hydra DictConfig.

    Expected config keys under ``cfg.model``:
      d_hidden, n_heads, n_layers, p_max, d_max, rank, dropout (optional).
    """
    mcfg    = cfg.model
    dropout = float(getattr(mcfg, "dropout", 0.0))

    model = ICLCorrNetV2(
        p_max    = int(mcfg.p_max),
        d_max    = int(mcfg.d_max),
        d_hidden = int(mcfg.d_hidden),
        n_heads  = int(mcfg.n_heads),
        n_layers = int(mcfg.n_layers),
        rank     = int(mcfg.rank),
        dropout  = dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[ICLCorrNetV2]  d_hidden={mcfg.d_hidden}  n_heads={mcfg.n_heads}  "
        f"n_layers={mcfg.n_layers}  p_max={mcfg.p_max}  d_max={mcfg.d_max}  "
        f"rank={mcfg.rank}  dropout={dropout}  |  params={n_params:,}"
    )
    return model


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


class RowAggregatorBlock(nn.Module):
    """Stage 2 (TF_row) block: standard non-causal pre-norm transformer.

    Processes per-row token sequences of shape (B*N, S, d_model) where
    S = n_cls + p_max. All tokens within a row attend freely (no masking).
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x : (B_N, S, d_model) — flattened batch×instance token sequences.
        """
        x_n = self.norm1(x)
        x = x + self.attn(x_n, x_n, x_n)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ICLBlock(nn.Module):
    """Stage 3 (TF_icl) block: row-level ICL with instance-awareness masking.

    Operates on row embeddings (B, N, d_icl) where d_icl = n_cls * d_model.
    Test rows (n_support..N-1) are blocked as keys so they are never attended to.
    The (N, N) attn_mask broadcasts over B automatically with batch_first=True.
    """

    def __init__(
        self, d_icl: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_icl)
        self.attn = nn.MultiheadAttention(
            d_icl, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_icl)
        self.ffn = SwiGLUFFN(d_icl, d_ff, dropout=dropout)

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
        N = x.shape[1]
        mask = torch.zeros(N, N, dtype=x.dtype, device=x.device)
        if n_support < N:
            mask[:, n_support:] = float("-inf")

        x_n = self.norm1(x)
        attn_out, attn_w = self.attn(
            x_n, x_n, x_n,
            attn_mask=mask,
            need_weights=return_attn_weights,
            average_attn_weights=True,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if return_attn_weights:
            return x, attn_w
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
        self.icl_gate_sup = nn.Parameter(torch.ones(d_icl))   # support injection, sigmoid(1)≈0.73

        # ---- Feature embedding -----------------------------------------------
        self.phi_X = nn.Linear(1, d_model, bias=False)

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
        # Per-slot position embeddings: break the shared-CLS symmetry so each
        # of the n_cls+p_max token positions gets a distinct starting point.
        # Without this, all row embeddings start identical after Stage 2 and
        # Stage 3's Q·K dot products are uniformly zero.
        self.row_pos_emb = nn.Parameter(torch.empty(n_cls + p_max, d_model))
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
        nn.init.trunc_normal_(self.row_pos_emb, std=0.02)
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

        E1 = self.phi_X(X_in.unsqueeze(-1)) * self.phi_x_scale  # (B, N, p_max, d_model)

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
        row_tok = row_tok + self.row_pos_emb  # break shared-CLS symmetry
        S = row_tok.shape[2]
        row_tok = row_tok.reshape(B * N, S, self.d_model)

        for block in self.s2_blocks:
            row_tok = block(row_tok)

        row_tok = row_tok.reshape(B, N, S, self.d_model)
        cls_out = self.s2_norm(row_tok[:, :, : self.n_cls, :])  # (B, N, n_cls, D)
        row_emb = cls_out.reshape(B, N, self.d_icl)  # (B, N, d_icl)

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

        # ------------------------------------------------------------------
        # 5. Readout: per-dimension MLP on query row embeddings
        # ------------------------------------------------------------------
        query_emb = row_emb[:, n_support:, :]  # (B, n_query, d_icl)

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
    print(
        f"[CopulaTabICLv2] d_model={mcfg.d_model}  d_icl={model.d_icl}  "
        f"n_heads={mcfg.n_heads}  "
        f"s1={mcfg.n_layers_s1}/s2={mcfg.n_layers_s2}/s3={mcfg.n_layers_s3}  "
        f"n_inducing={mcfg.n_inducing}  n_cls={mcfg.n_cls}  "
        f"p_max={mcfg.p_max}  d_max={mcfg.d_max}  rank={rank}  "
        f"|  params={n_params:,}"
    )

    return model
