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
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer block with alternating feature- and instance-attention.

    Processing order (pre-norm residual connections throughout):

    1. **Feature attention** (FeatureAttn):  MultiheadAttention within each
       instance across the S = p + d token slots.  All p feature tokens and
       all d target tokens interact with each other.

    2. **Instance attention** (InstanceAttn):  MultiheadAttention across the N
       instances for each token position independently.

       Masking is slot-dependent:
         • Feature token slots (0..p-1): no masking — query X values are
           real observations and can serve as keys.
         • Target token slots (p..p+d-1): query columns (n_support..N-1) are
           blocked with -inf — they hold mask tokens, not observed Z values.

    3. **FFN** (SwiGLUFFN):  Position-wise feed-forward applied per token.

    Args:
        d_model  : embedding dimension.
        n_heads  : number of attention heads (must divide d_model).
        d_ff     : hidden dimension of the SwiGLU FFN.
        dropout  : dropout probability (attention + FFN).
        p_max    : maximum number of feature tokens (needed for masking split).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        p_max: int = 1,
        num_cls: int = 0,
    ) -> None:
        super().__init__()
        self.p_max = p_max
        self.num_cls = num_cls
        self.norm1 = RMSNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_model)
        self.inst_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        n_support: int,
        p: int,
    ) -> torch.Tensor:
        """Apply one transformer block.

        Args:
            tokens    : (B, N, S, d_model) — all instance tokens.
                        S = num_cls + p_max + d  (CLS + feature + target tokens).
            n_support : number of support instances (leading rows of dim-1).
            p         : actual number of feature columns in this batch
                        (may be less than p_max if input was padded).

        Returns:
            Updated tokens tensor of the same shape.
        """
        B, N, S, D = tokens.shape

        # ------------------------------------------------------------------
        # 1. Feature attention: within each instance across all S tokens
        # ------------------------------------------------------------------
        t = tokens.reshape(B * N, S, D)
        t_norm = self.norm1(t)
        t = t + self.feat_attn(t_norm, t_norm, t_norm)[0]
        tokens = t.reshape(B, N, S, D)

        # ------------------------------------------------------------------
        # 2. Instance attention: across N instances, per token slot
        # ------------------------------------------------------------------
        # CLS slots (0..num_cls-1) are NOT updated here — they are summary
        # tokens updated only by feature attention.  Feature and target slots
        # both use a (N, N) additive mask that blocks test key columns to
        # prevent train→test and test→test information flow.
        #
        # Slot layout in tokens dim-2 (after CLS prepend in CopulaTransformer):
        #   0 .. num_cls-1               : CLS tokens        → skip (no update)
        #   num_cls .. num_cls+p_max-1   : feature tokens    → block test keys
        #   num_cls+p_max .. S-1         : target tokens     → block test keys

        # Reshape to (B, S, N, D) for per-slot access
        tokens_s = tokens.permute(0, 2, 1, 3)  # (B, S, N, D)
        tokens_s = tokens_s.clone()

        # Shared instance-attention mask: block query instances as keys
        inst_mask = torch.zeros(N, N, dtype=tokens.dtype, device=tokens.device)
        if n_support < N:
            inst_mask[:, n_support:] = float("-inf")

        # Feature slots
        feat_s = tokens_s[:, self.num_cls : self.num_cls + self.p_max, :, :]
        feat_flat = feat_s.reshape(B * self.p_max, N, D)
        feat_norm = self.norm2(feat_flat)
        feat_flat = (
            feat_flat
            + self.inst_attn(feat_norm, feat_norm, feat_norm, attn_mask=inst_mask)[0]
        )
        tokens_s[:, self.num_cls : self.num_cls + self.p_max, :, :] = feat_flat.reshape(
            B, self.p_max, N, D
        )

        # Target slots
        d_slots = S - self.p_max - self.num_cls
        if d_slots > 0:
            tgt_s = tokens_s[:, self.num_cls + self.p_max :, :, :]
            tgt_flat = tgt_s.reshape(B * d_slots, N, D)
            tgt_norm = self.norm2(tgt_flat)
            tgt_flat = (
                tgt_flat
                + self.inst_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=inst_mask)[0]
            )
            tokens_s[:, self.num_cls + self.p_max :, :, :] = tgt_flat.reshape(
                B, d_slots, N, D
            )

        tokens = tokens_s.permute(0, 2, 1, 3)  # back to (B, N, S, D)

        # ------------------------------------------------------------------
        # 3. FFN: applied independently per (instance, token) position
        # ------------------------------------------------------------------
        t = tokens.reshape(B * N, S, D)
        t = t + self.ffn(self.norm3(t))
        tokens = t.reshape(B, N, S, D)

        return tokens


# ---------------------------------------------------------------------------
# CopulaTransformer
# ---------------------------------------------------------------------------


class CopulaTransformer(nn.Module):
    """CopulaTransformer — Phase 2 model for multivariate dependency in Z-space.

    Given a labelled support set and unlabelled queries (represented as
    feature vectors X and copula scores Z), the model predicts a low-rank
    multivariate Gaussian over the query Z vectors:

        p(Z_q | X_all, Z_support) = N( mu_Z,  diag(d_Z) + V_Z V_Z^T )

    The covariance is parameterised by:
      • mu_Z : (B, n_query, d)     — conditional mean
      • d_Z  : (B, n_query, d)     — diagonal variance  (strictly positive)
      • V_Z  : (B, n_query, d, r)  — low-rank factor

    These outputs feed directly into ``loss.woodbury_nll`` for training.

    Token layout per instance: S = p + d tokens
      Slots 0 .. p-1       : feature tokens  φ_X(x_{i,k})  k=0..p-1
      Slots p .. p+d-1     : target tokens   φ_Z(z_{i,j})  j=0..d-1

    Each slot carries a learnable type embedding and a per-slot index
    embedding, making the representation sensitive to both the semantic role
    (feature vs. target) and the position within that role.

    Args:
        d_model  : embedding dimension for all transformer tokens.
        n_heads  : number of attention heads (must divide d_model).
        n_layers : number of TransformerBlock layers.
        p_max    : maximum number of input features (X columns).
                   Inputs with fewer features are zero-padded to p_max;
                   S = p_max + d at runtime.
        d_max    : maximum number of target dimensions (Z columns).
                   Sets the size of the learnable mask-token table.
        rank     : low-rank factor size r.  If *None*, r is computed
                   dynamically per-forward as max(1, floor(sqrt(d))).
        d_ff     : hidden size of SwiGLU FFN.  Defaults to the nearest
                   multiple of 64 above 8/3 * d_model.
        dropout  : dropout probability for attention and FFN layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        p_max: int,
        d_max: int,
        rank: Optional[int],
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Default d_ff: round(8/3 * d_model) to nearest 64 (minimum 64)
        if d_ff is None:
            d_ff = max(round(8 / 3 * d_model / 64) * 64, 64)

        # Maximum rank used to size the output head; actual rank at forward
        # time may be smaller when rank=None and d < d_max.
        rank_max: int = max(1, int(math.sqrt(d_max))) if rank is None else rank

        # ---- Input embeddings ------------------------------------------------
        # Each scalar feature x_{i,k} is embedded independently
        self.phi_X = nn.Linear(1, d_model, bias=False)
        # Each scalar Z dimension z_{i,j} is embedded independently
        self.phi_Z = nn.Linear(1, d_model, bias=False)

        # Type encoding: index 0 → feature token, 1 → target token
        self.type_enc = nn.Parameter(torch.zeros(2, d_model))

        # Learnable mask tokens θ_mask[j] for query target positions
        self.mask_tokens = nn.Parameter(torch.zeros(d_max, d_model))

        # ---- CLS tokens: per-instance summary tokens prepended to S dimension -
        self.num_cls = 4
        # (num_cls, d_model) — same initialization broadcast to all instances/batch
        self.cls_tokens = nn.Parameter(torch.empty(self.num_cls, d_model))

        # ---- Transformer body -----------------------------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout, p_max=p_max, num_cls=self.num_cls
                )
                for _ in range(n_layers)
            ]
        )

        # ---- Readout head ---------------------------------------------------
        # Only fc_V is needed: mu_Z = 0 and Sigma_ii = 1 (copula constraints).
        # head_in = (1 + num_cls) * d_model
        head_in = (1 + self.num_cls) * d_model
        self.fc_V = nn.Linear(head_in, rank_max)

        # ---- Store configuration -------------------------------------------
        self.p_max = p_max
        self.d_max = d_max
        self.rank = rank
        self.rank_max = rank_max
        self.d_model = d_model

        # ---- Initialisation -------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply custom weight initialisations.

        • mask_tokens : small random noise — avoids symmetric saddle point.
        • type_enc    : zero-init — learned from gradient signal.
        • fc_V        : non-zero init (std=0.02) — V=0 is a saddle of the copula
                        NLL; gradients vanish at V=0 for both the log-det and
                        quadratic terms. Non-zero init escapes this saddle.
        """
        nn.init.normal_(self.mask_tokens, std=0.01)
        nn.init.zeros_(self.type_enc)

        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        nn.init.normal_(self.fc_V.weight, std=0.02)
        nn.init.zeros_(self.fc_V.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        X_all: torch.Tensor,
        Z_all: torch.Tensor,
        n_support: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict conditional low-rank Gaussian parameters for query instances.

        Args:
            X_all     : (B, N, p) — feature vectors for all N instances.
                        p may be less than p_max; inputs are zero-padded.
            Z_all     : (B, N, d) — Z values; only the first *n_support*
                        rows are treated as observed (the rest are masked).
            n_support : number of support instances (leading rows of dim-1).

        Returns:
            mu_Z : (B, n_query, d)     — zero (copula mean is fixed at 0).
            d_Z  : (B, n_query, d)     — diagonal of correlation matrix; equals
                                         1/(1 + ||U_i||^2), ensuring Sigma_ii = 1.
            V_Z  : (B, n_query, d, r)  — low-rank factor; W_i = U_i/sqrt(1+||U_i||^2).
        """
        B, N, p = X_all.shape
        d = Z_all.shape[-1]
        n_query = N - n_support

        # Effective rank for this forward pass
        if self.rank is None:
            r = max(1, int(math.sqrt(d)))
        else:
            r = self.rank
        r = min(r, self.rank_max)  # cannot exceed what fc_V was sized for

        # ------------------------------------------------------------------
        # 1. Feature tokens:  (B, N, p_max, d_model)
        # ------------------------------------------------------------------
        # Pad or truncate X to exactly p_max features; embed each scalar
        # independently via phi_X then add type and per-slot index embeddings.
        if p > self.p_max:
            X_in = X_all[..., : self.p_max]
            p_eff = self.p_max
        elif p < self.p_max:
            X_in = F.pad(X_all, (0, self.p_max - p))  # (B, N, p_max)
            p_eff = p  # real columns; padded slots get zero before embedding
        else:
            X_in = X_all
            p_eff = p

        # phi_X expects (..., 1) — add feature dimension, embed, then squeeze
        feat_tok = self.phi_X(X_in.unsqueeze(-1))  # (B, N, p_max, d_model)
        type_feat = self.type_enc[0]  # (d_model,)

        feat_tok = feat_tok + type_feat  # (B, N, p_max, d_model)

        # ------------------------------------------------------------------
        # 2. Target tokens:  (B, N, d, d_model)
        # ------------------------------------------------------------------
        type_tgt = self.type_enc[1]  # (d_model,)

        # Support instances: embed their observed Z values
        Z_sup = Z_all[:, :n_support, :]  # (B, n_support, d)
        tgt_sup = self.phi_Z(Z_sup.unsqueeze(-1))  # (B, n_support, d, d_model)
        tgt_sup = tgt_sup + type_tgt  # broadcast over B, n_support

        # Query instances: replace Z values with learnable mask tokens
        mask_tok = self.mask_tokens[:d]  # (d, d_model)
        tgt_qry = mask_tok + type_tgt  # (d, d_model)
        tgt_qry = (
            tgt_qry.unsqueeze(0).unsqueeze(0).expand(B, n_query, d, -1)
        )  # (B, n_query, d, d_model)

        tgt_tok = torch.cat([tgt_sup, tgt_qry], dim=1)  # (B, N, d, d_model)

        # ------------------------------------------------------------------
        # 3. Assemble full token sequence:  (B, N, p_max+d, d_model)
        # ------------------------------------------------------------------
        tokens = torch.cat([feat_tok, tgt_tok], dim=2)  # (B, N, S, d_model)

        # ------------------------------------------------------------------
        # 4. Prepend CLS tokens and run transformer blocks
        # ------------------------------------------------------------------
        # CLS tokens: (num_cls, D) → (B, N, num_cls, D) via expand.
        # expand (not repeat/clone) ensures gradients sum over B×N back to the
        # shared (num_cls, D) parameter automatically.
        cls_exp = self.cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        tokens = torch.cat([cls_exp, tokens], dim=2)  # (B, N, num_cls+S, D)

        for block in self.blocks:
            tokens = block(tokens, n_support, p=p_eff)

        # ------------------------------------------------------------------
        # 5. Readout from query target tokens + per-instance CLS tokens
        # ------------------------------------------------------------------
        # Token layout after prepend: [cls_0..cls_{k-1}, feat_0..feat_{p-1}, tgt_0..tgt_{d-1}]
        # Query instances are at rows n_support .. N-1
        query_tgt = tokens[
            :, n_support:, self.num_cls + self.p_max :, :
        ]  # (B, n_query, d, D)
        query_cls = tokens[:, n_support:, : self.num_cls, :]  # (B, n_query, num_cls, D)

        # Tile CLS over d target dimensions for the shared readout input
        cls_concat = query_cls.reshape(B, n_query, self.num_cls * self.d_model)
        cls_tiled = cls_concat.unsqueeze(2).expand(
            B, n_query, d, -1
        )  # (B, n_query, d, num_cls*D)
        head_in = torch.cat(
            [query_tgt, cls_tiled], dim=-1
        )  # (B, n_query, d, (1+num_cls)*D)

        # ---- Copula constraint: mu_Z = 0, Sigma_ii = 1 ----------------------
        # After the probit-PIT, z_{i,j} = Phi^{-1}(F_j(y_{i,j}|x_i)) has
        # standard-normal marginals by construction: E[z_j|x_i] = 0 and
        # Var[z_j|x_i] = 1.  The Gaussian copula density
        #
        #   -log c(u) = 1/2 log|R| + 1/2 z^T (R^{-1} - I) z
        #
        # requires R to be a correlation matrix (R_{ii} = 1, mu = 0).
        # Fixing s_Z = 1 enforces this: Sigma_ii = s_Z^2 * (C_diag + ||W_i||^2)
        #                                         = 1 * 1 = 1.
        # mu_Z = 0 because the marginals are already perfectly standardised.

        mu_Z = torch.zeros(B, n_query, d, dtype=X_all.dtype, device=X_all.device)

        # s_Z = 1 (correlation matrix constraint)
        s_Z = torch.ones(B, n_query, d, dtype=head_in.dtype, device=head_in.device)
        U = self.fc_V(head_in)[..., :r]  # (B, n_query, d, r)

        # Woodbury decomposition of the correlation matrix R = diag(C_diag) + W W^T
        # C_diag_i = 1/(1+||U_i||^2),  W_i = U_i/sqrt(1+||U_i||^2)
        # => R_{ii} = C_diag_i + ||W_i||^2 = 1  (verified by construction)
        U_sq_norm = (U**2).sum(dim=-1)  # (B, n_query, d)
        C_diag = 1.0 / (1.0 + U_sq_norm)  # (B, n_query, d)
        W = U / torch.sqrt(1.0 + U_sq_norm.unsqueeze(-1))  # (B, n_query, d, r)

        d_Z = (s_Z**2) * C_diag  # (B, n_query, d)  = C_diag since s_Z = 1
        V_Z = s_Z.unsqueeze(-1) * W  # (B, n_query, d, r)  = W since s_Z = 1

        return mu_Z, d_Z, V_Z


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_copula_transformer(cfg) -> CopulaTransformer:
    """Instantiate a CopulaTransformer from a Hydra DictConfig.

    Expected config keys under ``cfg.model``:

    ==================  ======================================================
    Key                 Description
    ==================  ======================================================
    d_model             Embedding dimension.
    n_heads             Number of attention heads.
    n_layers            Number of transformer blocks.
    p_max               Maximum number of input features.
    d_max               Maximum number of target dimensions.
    rank                Low-rank factor size r.  Pass *null* in YAML for the
                        dynamic sqrt(d) heuristic.
    d_ff                (optional) Hidden size of SwiGLU FFN.  Defaults to the
                        nearest multiple of 64 above 8/3 * d_model.
    dropout             (optional, default 0.0) Dropout probability.
    ==================  ======================================================

    Args:
        cfg : Hydra DictConfig with a ``model`` sub-config as described above.

    Returns:
        Initialised :class:`CopulaTransformer`.
    """
    mcfg = cfg.model

    rank: Optional[int] = None if mcfg.rank is None else int(mcfg.rank)
    d_ff: Optional[int] = (
        int(mcfg.d_ff) if getattr(mcfg, "d_ff", None) is not None else None
    )
    dropout: float = float(getattr(mcfg, "dropout", 0.0))

    model = CopulaTransformer(
        d_model=int(mcfg.d_model),
        n_heads=int(mcfg.n_heads),
        n_layers=int(mcfg.n_layers),
        p_max=int(mcfg.p_max),
        d_max=int(mcfg.d_max),
        rank=rank,
        d_ff=d_ff,
        dropout=dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    d_ff_actual = model.blocks[0].ffn.w1.out_features if model.blocks else d_ff
    print(
        f"[CopulaTransformer] d_model={mcfg.d_model}  n_heads={mcfg.n_heads}  "
        f"n_layers={mcfg.n_layers}  p_max={mcfg.p_max}  d_max={mcfg.d_max}  "
        f"rank={rank}  d_ff={d_ff_actual}  dropout={dropout}  "
        f"|  params={n_params:,}"
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
        # ReZero: learned scalars, init=0 → block starts as identity.
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ffn  = nn.Parameter(torch.zeros(1))

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
        x = x + self.alpha_attn * attn_out
        x = x + self.alpha_ffn  * self.ffn(self.norm2(x))
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
        # Gate on ICL injection: sigmoid(-1) ≈ 0.27 at init.
        # Provides 4× stronger gradient than the original -3.0 init (sigmoid≈0.05)
        # which caused the gate to remain permanently closed across all runs.
        self.icl_gate = nn.Parameter(torch.tensor(-1.0))

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
        # Normalise the mean-subtracted query embeddings before the readout head.
        # After mean-subtraction the residual has RMS ~0.06 while dim_emb has
        # magnitude ~1, so fc_V would be blind to instance-specific signal.
        # Initialised as identity (scale=ones) so checkpoint loading with
        # strict=False preserves the current behaviour until further training.
        self.query_emb_norm = RMSNorm(d_icl)

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
        nn.init.trunc_normal_(self.dim_emb, std=0.02)
        for block in self.s1_blocks:
            nn.init.trunc_normal_(block.inducing_points, std=0.02)
        nn.init.normal_(self.embed_tae.weight, std=0.02)
        nn.init.zeros_(self.embed_tae.bias)
        nn.init.normal_(self.embed_icl.weight, std=0.02)
        nn.init.zeros_(self.embed_icl.bias)
        # fc_V: use std=0.1 (not 0.02) so U starts far enough from zero to
        # escape the near-diagonal saddle. At U≈0, grad(log|M|)/dV ≈ 2V ≈ 0
        # and grad(quadratic)/dV ≈ 0, so the model gets no signal to grow V.
        nn.init.normal_(self.fc_V.fc1.weight, std=0.1)
        nn.init.zeros_(self.fc_V.fc1.bias)
        nn.init.normal_(self.fc_V.fc2.weight, std=0.1)
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
        row_emb[:, :n_support, :] += torch.sigmoid(self.icl_gate) * icl_emb

        for block in self.s3_blocks:
            row_emb = block(row_emb, n_support)

        row_emb = self.s3_norm(row_emb)

        # ------------------------------------------------------------------
        # 5. Readout: per-dimension MLP on query row embeddings
        # ------------------------------------------------------------------
        query_emb = row_emb[:, n_support:, :]  # (B, n_query, d_icl)

        # Remove the shared "support-context drift" component.
        # All query rows attend to the same support in Stage 3, so each ICL
        # block adds the same large residual to all of them.  After 12 blocks
        # the shared component dominates (||shared|| >> ||specific||) and
        # s3_norm collapses every query row to the same unit vector.
        # Subtracting the mean zeros the shared component and exposes the
        # instance-specific signal that varies across query rows.
        # query_emb_norm then rescales to unit RMS so fc_V sees a consistent
        # magnitude regardless of how much S3 differentiated the query rows.
        query_emb = query_emb - query_emb.mean(dim=1, keepdim=True)
        query_emb = self.query_emb_norm(query_emb)

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
