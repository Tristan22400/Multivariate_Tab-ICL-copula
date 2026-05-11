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

Readout (query target tokens only) → (mu_Z, d_Z, V_Z) via three linear heads.
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
    ) -> None:
        super().__init__()
        self.p_max = p_max
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
                        S = p + d  (p feature tokens + d target tokens).
            n_support : number of support instances (leading rows of dim-1).
            p         : actual number of feature columns in this batch
                        (may be less than p_max if input was padded; used
                        to determine the feature/target split in the mask).

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
        # We process feature slots and target slots separately because they
        # require different (N, N) key-side masks.  PyTorch MHA with
        # batch_first=True accepts a 2D (N, N) additive float mask that is
        # broadcast over both the batch and head dimensions — this is the
        # only safe way to specify a per-slot policy without fighting the
        # (B*n_heads, N, N) 3D interpretation.
        #
        # Slot layout in tokens dim-2:
        #   0 .. p_max-1           : feature tokens  → no masking
        #   p_max .. p_max+d-1     : target tokens   → block query key columns

        # Reshape to (B, S, N, D) for per-slot access
        tokens_s = tokens.permute(0, 2, 1, 3)  # (B, S, N, D)

        # Feature slots: no masking (query X values are real observations)
        feat_s = tokens_s[:, : self.p_max, :, :]  # (B, p_max, N, D)
        feat_flat = feat_s.reshape(B * self.p_max, N, D)
        feat_norm = self.norm2(feat_flat)
        feat_flat = feat_flat + self.inst_attn(feat_norm, feat_norm, feat_norm)[0]
        tokens_s = tokens_s.clone()
        tokens_s[:, : self.p_max, :, :] = feat_flat.reshape(B, self.p_max, N, D)

        # Target slots: block query key columns (n_support..N-1 are unknown)
        d_slots = S - self.p_max  # number of target token slots
        if d_slots > 0:
            tgt_s = tokens_s[:, self.p_max :, :, :]  # (B, d_slots, N, D)
            tgt_flat = tgt_s.reshape(B * d_slots, N, D)
            tgt_norm = self.norm2(tgt_flat)

            # 2D (N, N) additive mask — broadcast over B*d_slots and n_heads
            tgt_mask = torch.zeros(N, N, dtype=tokens.dtype, device=tokens.device)
            if n_support < N:
                tgt_mask[:, n_support:] = float("-inf")

            tgt_flat = (
                tgt_flat
                + self.inst_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask)[0]
            )
            tokens_s[:, self.p_max :, :, :] = tgt_flat.reshape(B, d_slots, N, D)

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

        # Per-slot index embeddings:
        #   feat_enc[k]  adds positional identity to the k-th feature token
        #   dim_enc[j]   adds dimensional identity to the j-th target token
        self.feat_enc = nn.Embedding(p_max, d_model)
        self.dim_enc = nn.Embedding(d_max, d_model)

        # Learnable mask tokens θ_mask[j] for query target positions
        self.mask_tokens = nn.Parameter(torch.zeros(d_max, d_model))

        # ---- Transformer body -----------------------------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, p_max=p_max)
                for _ in range(n_layers)
            ]
        )

        # ---- Readout heads --------------------------------------------------
        self.fc_mu = nn.Linear(d_model, 1)  # → mu scalar per (query, dim)
        self.fc_d = nn.Linear(d_model, 1)  # → log-variance scalar
        self.fc_V = nn.Linear(d_model, rank_max)  # → V factor row per (query, dim)

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

        • mask_tokens  : small random noise — avoids a symmetric saddle
                         point while keeping initial queries nearly neutral.
        • type_enc     : zero-init — will be learned from gradient signal.
        • fc_mu, fc_d  : small weight init (std=0.01) — keeps initial
                         predictions near zero / unit variance.
        • fc_V         : moderate init (std=0.1) — prevents V≈0 saddle
                         where the low-rank component has zero gradient.
        """
        nn.init.normal_(self.mask_tokens, std=0.01)
        nn.init.zeros_(self.type_enc)

        nn.init.normal_(self.fc_mu.weight, std=0.01)
        nn.init.zeros_(self.fc_mu.bias)

        nn.init.normal_(self.fc_d.weight, std=0.01)
        nn.init.zeros_(self.fc_d.bias)

        nn.init.zeros_(self.fc_V.weight)
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
            mu_Z : (B, n_query, d)     — conditional mean.
            d_Z  : (B, n_query, d)     — diagonal variance (strictly positive,
                                         parameterised via softplus + eps).
            V_Z  : (B, n_query, d, r)  — low-rank factor rows.
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

        feat_idx = torch.arange(self.p_max, device=X_all.device)  # (p_max,)
        feat_emb = self.feat_enc(feat_idx)  # (p_max, d_model)
        type_feat = self.type_enc[0]  # (d_model,)

        feat_tok = feat_tok + type_feat + feat_emb  # (B, N, p_max, d_model)

        # ------------------------------------------------------------------
        # 2. Target tokens:  (B, N, d, d_model)
        # ------------------------------------------------------------------
        dim_idx = torch.arange(d, device=X_all.device)  # (d,)
        dim_emb = self.dim_enc(dim_idx)  # (d, d_model)
        type_tgt = self.type_enc[1]  # (d_model,)

        # Support instances: embed their observed Z values
        Z_sup = Z_all[:, :n_support, :]  # (B, n_support, d)
        tgt_sup = self.phi_Z(Z_sup.unsqueeze(-1))  # (B, n_support, d, d_model)
        tgt_sup = tgt_sup + type_tgt + dim_emb  # broadcast over B, n_support

        # Query instances: replace Z values with learnable mask tokens
        mask_tok = self.mask_tokens[:d]  # (d, d_model)
        tgt_qry = mask_tok + type_tgt + dim_emb  # (d, d_model)
        tgt_qry = (
            tgt_qry.unsqueeze(0).unsqueeze(0).expand(B, n_query, d, -1)
        )  # (B, n_query, d, d_model)

        tgt_tok = torch.cat([tgt_sup, tgt_qry], dim=1)  # (B, N, d, d_model)

        # ------------------------------------------------------------------
        # 3. Assemble full token sequence:  (B, N, p_max+d, d_model)
        # ------------------------------------------------------------------
        tokens = torch.cat([feat_tok, tgt_tok], dim=2)  # (B, N, S, d_model)

        # ------------------------------------------------------------------
        # 4. Transformer blocks
        # ------------------------------------------------------------------
        for block in self.blocks:
            tokens = block(tokens, n_support, p=p_eff)

        # ------------------------------------------------------------------
        # 5. Readout from query target tokens
        # ------------------------------------------------------------------
        # Target tokens occupy slots p_max .. p_max+d-1
        # Query instances are at rows n_support .. N-1
        query_tgt = tokens[:, n_support:, self.p_max :, :]  # (B, n_query, d, d_model)

        mu_Z = self.fc_mu(query_tgt).squeeze(-1)  # (B, n_query, d)
        s_Z = F.softplus(self.fc_d(query_tgt).squeeze(-1)) + 1e-4  # (B, n_query, d)
        U = self.fc_V(query_tgt)[..., :r]  # (B, n_query, d, r)

        U_sq_norm = (U**2).sum(dim=-1)  # (B, n_query, d)
        C_diag = 1.0 / (1.0 + U_sq_norm)  # (B, n_query, d)
        W = U / torch.sqrt(1.0 + U_sq_norm.unsqueeze(-1))  # (B, n_query, d, r)

        d_Z = (s_Z**2) * C_diag  # (B, n_query, d)
        V_Z = s_Z.unsqueeze(-1) * W  # (B, n_query, d, r)

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
