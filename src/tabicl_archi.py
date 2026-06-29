"""CopulaTabICL: TabICL backbone with a low-rank Gaussian copula head.

Z is injected at two stages, mirroring how TabICL injects scalar y:
  1. col_embedder (embedding.py:_compute_embeddings) — target-aware column embedding
  2. icl_predictor (learning.py:267-268)             — ICL transformer context

The head outputs (W_tilde, D_tilde) parametrising a unit-diagonal correlation matrix:
    R = diag(D_tilde) + W_tilde @ W_tilde.T
"""

from __future__ import annotations

import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tabicl.model.tabicl import TabICL


class _ZEncoder(nn.Module):
    """Linear(d, out_dim) that absorbs the unsqueeze(-1) added by TabICL's regression path.

    Both col_embedder._compute_embeddings and icl_predictor._icl_predictions call
        y_encoder(y_train.unsqueeze(-1))
    turning Z_train (B, T, d) into (B, T, d, 1). We squeeze that trailing 1 away
    before the linear so the rest of the pipeline is untouched.
    """

    def __init__(self, d: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.squeeze(-1))  # (..., d, 1) → (..., d) → (..., out_dim)


def _patch_col_embedder_expand(col_embedder) -> None:
    """Patch the two expand calls in ColEmbedding that assume scalar (2-D) y_train.

    _train_forward_without_feature_group (embedding.py:490) and
    _train_forward_with_feature_group    (embedding.py:472) both do:
        y_train.unsqueeze(1).expand(-1, n_features, -1)
    which fails when y_train is 3-D (B, train_size, d).
    We rebind both methods on the instance with a dim-agnostic expand.
    """

    def _train_forward_without_feature_group(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor], embed_with_test: bool
    ) -> Tensor:
        train_size = y_train.shape[1]

        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

        if d is None:
            features = X.transpose(1, 2).unsqueeze(-1)  # (B, p, T, 1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                extra = (-1,) * (y_train.dim() - 1)
                y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], *extra)
            embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test)
        else:
            if self.reserve_cls_tokens > 0:
                d = d + self.reserve_cls_tokens
            B, T, HC = X.shape
            X = X.transpose(1, 2)
            indices = torch.arange(HC, device=X.device).unsqueeze(0).expand(B, HC)
            mask = indices < d.unsqueeze(1)
            features = X[mask].unsqueeze(-1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                extra = (-1,) * (y_train.dim() - 1)
                y_train = y_train.unsqueeze(1).expand(-1, HC, *extra)
                y_train = y_train[mask]

            effective_embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test)

            embeddings = torch.zeros(B, HC, T, self.embed_dim, device=X.device, dtype=effective_embeddings.dtype)
            embeddings[mask] = effective_embeddings
        return embeddings.transpose(1, 2)

    def _train_forward_with_feature_group(
        self, X: Tensor, y_train: Tensor, embed_with_test: bool
    ) -> Tensor:
        train_size = y_train.shape[1]
        X = self.feature_grouping(X)  # (B, T, G, group_size)
        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)

        features = X.transpose(1, 2)  # (B, G+C, T, group_size)

        if self.target_aware:
            assert y_train is not None, "y_train must be provided when target_aware=True."
            extra = (-1,) * (y_train.dim() - 1)
            y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], *extra)
        embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test)
        return embeddings.transpose(1, 2)  # (B, T, G+C, E)

    def _inference_with_feature_group(
        self, X: Tensor, y_train: Tensor, train_size: int, embed_with_test: bool
    ) -> Tensor:
        """Inference path when feature grouping is enabled."""
        from collections import OrderedDict

        X = self.feature_grouping(X)  # (B, T, G, group_size)
        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)

        features = X.transpose(1, 2)  # (B, G+C, T, group_size)
        if self.target_aware:
            assert y_train is not None, "y_train must be provided when target_aware=True."
            extra = (-1,) * (y_train.dim() - 1)
            y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], *extra)
        else:
            y_train = None

        return self.inference_mgr(
            self._compute_embeddings,
            inputs=OrderedDict(
                [
                    ("features", features),
                    ("train_size", train_size),
                    ("y_train", y_train),
                    ("embed_with_test", embed_with_test),
                ]
            ),
        )

    def _inference_without_feature_group(
        self,
        X: Tensor,
        y_train: Tensor,
        train_size: int,
        embed_with_test: bool,
        feature_shuffles,
    ) -> Tensor:
        """Inference path when feature grouping is disabled."""
        from collections import OrderedDict

        if feature_shuffles is None:
            if self.reserve_cls_tokens > 0:
                X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

            features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                extra = (-1,) * (y_train.dim() - 1)
                y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], *extra)
            else:
                y_train = None

            embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict(
                    [
                        ("features", features),
                        ("train_size", train_size),
                        ("y_train", y_train),
                        ("embed_with_test", embed_with_test),
                    ]
                ),
            )
        else:
            # Shuffle optimisation: compute once, reorder for each table
            B = X.shape[0]
            first_table = X[0]
            if self.reserve_cls_tokens > 0:
                first_table = F.pad(first_table, (self.reserve_cls_tokens, 0), value=-100.0)

            features = first_table.transpose(0, 1).unsqueeze(-1)  # (H+C, T, 1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_first = y_train[0].unsqueeze(0).expand(features.shape[0], *((-1,) * y_train[0].dim()))
            else:
                y_first = None

            first_embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict(
                    [
                        ("features", features),
                        ("train_size", train_size),
                        ("y_train", y_first),
                        ("embed_with_test", embed_with_test),
                    ]
                ),
                output_repeat=B,
            )

            # Apply shuffles for tables after the first one
            embeddings = first_embeddings.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H+C, T, E)
            first_pattern = feature_shuffles[0]
            for i in range(1, B):
                mapping = self.map_feature_shuffle(first_pattern, feature_shuffles[i])
                if self.reserve_cls_tokens > 0:
                    mapping = [m + self.reserve_cls_tokens for m in mapping]
                    mapping = list(range(self.reserve_cls_tokens)) + mapping
                embeddings[i] = first_embeddings[mapping]

        return embeddings

    col_embedder._train_forward_without_feature_group = types.MethodType(
        _train_forward_without_feature_group, col_embedder
    )
    col_embedder._train_forward_with_feature_group = types.MethodType(
        _train_forward_with_feature_group, col_embedder
    )
    col_embedder._inference_with_feature_group = types.MethodType(
        _inference_with_feature_group, col_embedder
    )
    col_embedder._inference_without_feature_group = types.MethodType(
        _inference_without_feature_group, col_embedder
    )


class CopulaTabICL(TabICL):
    """TabICL with a low-rank Gaussian copula head.

    Three components are replaced to handle multivariate Z instead of scalar y:
      - col_embedder.y_encoder  : Linear(1, embed_dim) → _ZEncoder(d, embed_dim)
      - icl_predictor.y_encoder : Linear(1, icl_dim)   → _ZEncoder(d, icl_dim)
      - icl_predictor.decoder   : MLP(icl_dim, 1)      → MLP(icl_dim, d*k + d)

    Parameters
    ----------
    d : int   Number of target features (dimension of Z).
    k : int   Rank of the low-rank factor W.
    """

    def __init__(self, d: int, k: int, pre_icl_aux: bool = False, **tabicl_kwargs):
        tabicl_kwargs.setdefault("col_target_aware", True)
        super().__init__(max_classes=0, num_quantiles=1, **tabicl_kwargs)

        self.d = d
        self.k = k
        icl_dim = self.embed_dim * self.row_num_cls

        # col_embedder: replace y_encoder and patch the expand calls
        self.col_embedder.y_encoder = _ZEncoder(d, self.embed_dim)
        _patch_col_embedder_expand(self.col_embedder)

        # icl_predictor: replace y_encoder and decoder
        self.icl_predictor.y_encoder = _ZEncoder(d, icl_dim)
        self.icl_predictor.decoder = nn.Sequential(
            nn.Linear(icl_dim, 2 * icl_dim),
            nn.GELU(),
            nn.Linear(2 * icl_dim, d * k + d),
        )

        self._pre_icl_aux = pre_icl_aux
        self._pre_icl_test_pred: Optional[Tensor] = None
        if pre_icl_aux:
            n_pairs = d * (d - 1) // 2
            self.pre_icl_aux_head = nn.Sequential(
                nn.Linear(icl_dim, icl_dim // 2),
                nn.GELU(),
                nn.Linear(icl_dim // 2, n_pairs),
            )

    @staticmethod
    def _normalize(raw: Tensor, d: int, k: int) -> Tuple[Tensor, Tensor]:
        """Map head output → unit-diagonal correlation factors (W_tilde, D_tilde)."""
        W = raw[..., : d * k].reshape(*raw.shape[:-1], d, k)
        D = F.softplus(raw[..., d * k :])              # (..., d)  positive
        sigma = (W ** 2).sum(-1) + D                   # (..., d)
        return W / sigma.sqrt().unsqueeze(-1), D / sigma

    def forward(
        self,
        X: Tensor,
        Z_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        X       : (B, T, p)           all rows, train then test
        Z_train : (B, train_size, d)  PIT-transformed targets for train rows

        Returns
        -------
        W_tilde : (B, T_test, d, k)
        D_tilde : (B, T_test, d)
        """
        B, T, _ = X.shape
        train_size = Z_train.shape[1]

        # Z_train is passed as y_train: col_embedder adds it only to train tokens
        # (src[..., :train_size, :] += y_emb), test tokens are unaffected.
        R = self.row_interactor(
            self.col_embedder(X, y_train=Z_train, d=d, embed_with_test=embed_with_test),
            d=d,
        )

        if self._pre_icl_aux and hasattr(self, "pre_icl_aux_head"):
            self._pre_icl_test_pred = self.pre_icl_aux_head(R[:, train_size:])
        else:
            self._pre_icl_test_pred = None

        raw = self.icl_predictor._icl_predictions(R, Z_train)
        W_tilde, D_tilde = self._normalize(raw[:, train_size:], self.d, self.k)
        return W_tilde, D_tilde
