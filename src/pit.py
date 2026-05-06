"""
pit.py — Phase 1: Probability Integral Transform via TabICL's quantile CDF.

For each target dimension j, the frozen TabICL model estimates the conditional
CDF  F̂_j(y | x, context).  Evaluating it at the true target values maps
continuous observations to the unit interval U ~ Uniform(0, 1):

    u_{i,j} = F̂_j(y_{i,j} | x_i, context)

A probit transform then maps to standard-normal Z-space:

    z_{i,j} = Φ⁻¹(u_{i,j})

For training instances a batched leave-one-out (LOO) scheme avoids N_train
separate forward passes: all N_train LOO evaluations for a single dimension j
are packed into one batched TabICL call, chunked by `pit_batch_size` to stay
within GPU memory.

For test instances the full X_train is used as context (standard ICL).
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — find tabicl_upstream regardless of working directory
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_TABICL_SRC = os.path.join(_REPO_ROOT, "tabicl_upstream", "src")
if _TABICL_SRC not in sys.path:
    sys.path.insert(0, _TABICL_SRC)


# ---------------------------------------------------------------------------
# TabICL loader
# ---------------------------------------------------------------------------


def load_tabicl(ckpt_name: str, device: str) -> nn.Module:
    """Download (if needed) and load a frozen TabICL regressor.

    Args:
        ckpt_name : HuggingFace filename, e.g. "tabicl-regressor-v2-20260212.ckpt"
        device    : torch device string ("cpu", "cuda", etc.)

    Returns:
        base : TabICL module in eval() mode with all parameters frozen.
    """
    from huggingface_hub import hf_hub_download
    from tabicl._model.tabicl import TabICL  # type: ignore[import]

    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    base = TabICL(**checkpoint["config"])
    base.load_state_dict(checkpoint["state_dict"])

    for p in base.parameters():
        p.requires_grad_(False)

    base.eval()
    base.to(device)
    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probit(u: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp u to (eps, 1-eps) then apply Φ⁻¹ via erfinv."""
    u = u.clamp(eps, 1.0 - eps)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)


def _build_loo_chunk(
    X_train: torch.Tensor,
    y_train_j: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the LOO batch tensors for indices [chunk_start, chunk_end).

    For batch element `k` (local index for global index `chunk_start + k`):
      - context X : X_train with row `chunk_start + k` removed  (N-1 rows)
      - query   X : X_train[chunk_start + k]                    (1 row, appended last)
      - context y : y_train_j with element `chunk_start + k` removed

    Args:
        X_train    : (N, p)   — all training features
        y_train_j  : (N,)     — training labels for dimension j
        chunk_start: first global index in this chunk
        chunk_end  : one past the last global index

    Returns:
        X_loo : (chunk, N, p)   — context rows first, query at position N-1
        y_loo : (chunk, N-1)    — context labels
    """
    N, p = X_train.shape
    chunk = chunk_end - chunk_start
    device = X_train.device

    X_loo = torch.empty(chunk, N, p, device=device, dtype=X_train.dtype)
    y_loo = torch.empty(chunk, N - 1, device=device, dtype=y_train_j.dtype)

    all_idx = torch.arange(N, device=device)

    for local_k, global_i in enumerate(range(chunk_start, chunk_end)):
        mask = all_idx != global_i
        ctx_X = X_train[mask]         # (N-1, p)
        qry_X = X_train[global_i]     # (p,)

        X_loo[local_k, : N - 1] = ctx_X
        X_loo[local_k, N - 1]   = qry_X
        y_loo[local_k]           = y_train_j[mask]

    return X_loo, y_loo


# ---------------------------------------------------------------------------
# Main PIT function
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_pit(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    pit_batch_size: int = 64,
    eps: float = 1e-6,
    dequantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map raw targets Y → standard-normal latents Z via TabICL's quantile CDF.

    Performs the Probability Integral Transform:

        u_{i,j} = F̂_j(y_{i,j} | x_i, context_i)
        z_{i,j} = Φ⁻¹(u_{i,j})

    For training instances a batched LOO scheme ensures each u_{i,j} is
    computed without instance i in its own context.  Test instances use the
    full training set as context (standard ICL).

    Args:
        tabicl        : frozen TabICL model (from `load_tabicl`)
        X_train       : (N_train, p)  training features
        Y_train       : (N_train, d)  training targets
        X_test        : (N_test,  p)  test features
        Y_test        : (N_test,  d)  test targets
        pit_batch_size: max chunk size for the batched LOO forward pass
        eps           : clamping epsilon before probit (guards against ±∞)
        dequantize    : if True, add Uniform(0,1) noise to Y before CDF
                        evaluation — corrects for discrete/ordinal targets
                        (e.g. integer grades) where the tabular CDF has
                        probability mass rather than density

    Returns:
        Z_train    : (N_train, d)  standard-normal latents for training set
        Z_test     : (N_test,  d)  standard-normal latents for test set
        log_p_test : (N_test,  d)  per-instance per-dimension marginal
                        log-densities from TabICL (used for Y-space NLL)
    """
    N_train, d = Y_train.shape
    N_test      = X_test.shape[0]
    device      = X_train.device

    Z_train_cols:    list[torch.Tensor] = []
    Z_test_cols:     list[torch.Tensor] = []
    log_p_test_cols: list[torch.Tensor] = []

    for j in range(d):
        y_train_j = Y_train[:, j]  # (N_train,)
        y_test_j  = Y_test[:, j]   # (N_test,)

        # ---- A) Test instances: full context forward pass -----------------
        # X layout: [X_train | X_test] — first N_train rows are context
        X_concat    = torch.cat([X_train, X_test], dim=0).unsqueeze(0)  # (1, N_train+N_test, p)
        y_context_j = y_train_j.unsqueeze(0)                            # (1, N_train)

        logits_test = tabicl(X_concat, y_context_j)            # (1, N_test, Q)
        dist_test   = tabicl.quantile_dist(logits_test[0])     # batch_shape = (N_test,)

        y_test_j_eval = y_test_j + (torch.rand_like(y_test_j) if dequantize else 0.0)
        u_test_j      = dist_test.cdf(y_test_j_eval)           # (N_test,)
        lp_test_j     = dist_test.log_prob(y_test_j_eval)      # (N_test,)

        # ---- B) Training instances: batched LOO ---------------------------
        u_train_j = torch.empty(N_train, device=device, dtype=y_train_j.dtype)

        for chunk_start in range(0, N_train, pit_batch_size):
            chunk_end = min(chunk_start + pit_batch_size, N_train)

            X_loo, y_loo = _build_loo_chunk(X_train, y_train_j, chunk_start, chunk_end)
            # X_loo : (chunk, N_train, p) — context N_train-1 rows + query at end
            # y_loo : (chunk, N_train-1)  — context labels

            logits_loo = tabicl(X_loo, y_loo)               # (chunk, 1, Q)
            dist_loo   = tabicl.quantile_dist(logits_loo[:, 0, :])  # batch_shape = (chunk,)

            y_chunk_j = y_train_j[chunk_start:chunk_end]
            if dequantize:
                y_chunk_j = y_chunk_j + torch.rand_like(y_chunk_j)

            u_train_j[chunk_start:chunk_end] = dist_loo.cdf(y_chunk_j)  # (chunk,)

        # ---- C) Clamp + probit --------------------------------------------
        z_train_j = _probit(u_train_j, eps)
        z_test_j  = _probit(u_test_j,  eps)

        Z_train_cols.append(z_train_j)
        Z_test_cols.append(z_test_j)
        log_p_test_cols.append(lp_test_j)

    Z_train    = torch.stack(Z_train_cols,    dim=1)  # (N_train, d)
    Z_test     = torch.stack(Z_test_cols,     dim=1)  # (N_test,  d)
    log_p_test = torch.stack(log_p_test_cols, dim=1)  # (N_test,  d)

    return Z_train, Z_test, log_p_test
