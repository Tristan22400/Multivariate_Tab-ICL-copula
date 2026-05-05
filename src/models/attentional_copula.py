"""
attentional_copula.py — In-Context Attentional Copula Model.

Implements Sklar's theorem end-to-end:
  • Marginals:    handled externally (TabICL quantile function at inference)
  • Copula:       learned here via Transformer encoder + autoregressive decoder

Architecture:
  1. Context Preparation
       U_train = empirical_pit(Y_train)               — self-ranks, no leakage
       U_test  = smooth_context_pit(Y_test, Y_train)  — smoothed context CDF

  2. AttentionalCopulaEncoder  (alternating feature-attn / instance-attn)
       Input:  X_train, X_test, U_train
       Output: H_enc ∈ (B, T, p+d, d_model)

  3. AutoregressiveCopulaDecoder  (SDPA broadcasting cross-attn + discrete head)
       Training:  teacher-forced AR over random permutation π,
                  outputs logits (B, n_test, d, n_bins)
       Inference: autoregressive sampling

    Memory layout avoids OOM: K/V stays at (B, n_heads, n_train*S + n_test*k, d_head)
    (no expand/reshape).  A boolean mask enforces instance independence by
    blocking each query from attending to other instances' decoded targets.
    Total K/V size = n_train*(p+d) + n_test*k — independent of expand factor.

  4. Post-processing (external, inference only)
       y_test = TabICL.quantile(u_test | X_train, Y_train, X_test)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared building blocks (mirrors multivariate_tfm.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1   = nn.Linear(d_model, d_ff, bias=False)
        self.w2   = nn.Linear(d_model, d_ff, bias=False)
        self.w3   = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    """Alternating feature-attention / instance-attention block (pre-RMSNorm).

    H: (B, T, S, d_model)  where S = p + d tokens per instance.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        d_ff:      int,
        dropout:   float = 0.0,
        layer_idx: int   = 0,
        n_layers:  int   = 1,
    ):
        super().__init__()
        self.norm1     = RMSNorm(d_model)
        self.norm2     = RMSNorm(d_model)
        self.norm3     = RMSNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.inst_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn       = SwiGLUFFN(d_model, d_ff, dropout)

        std = 0.02 / math.sqrt(2 * max(n_layers, 1))
        nn.init.normal_(self.feat_attn.out_proj.weight, std=std)
        nn.init.zeros_(self.feat_attn.out_proj.bias)
        nn.init.normal_(self.inst_attn.out_proj.weight, std=std)
        nn.init.zeros_(self.inst_attn.out_proj.bias)
        nn.init.normal_(self.ffn.w3.weight, std=std)

    def forward(self, H: Tensor, inst_mask: Tensor | None = None) -> Tensor:
        """
        inst_mask: (T, T) bool, nn.MultiheadAttention convention — True = BLOCKED.
        Pass a mask that blocks test-to-test cross-instance attention so each
        test query is independent of other test instances.
        """
        B, T, S, D = H.shape

        H_flat = H.reshape(B * T, S, D)
        H_n    = self.norm1(H_flat)
        out, _ = self.feat_attn(H_n, H_n, H_n, need_weights=False)
        H = H + out.reshape(B, T, S, D)

        H_inst = H.permute(0, 2, 1, 3).reshape(B * S, T, D)
        H_n    = self.norm2(H_inst)
        out, _ = self.inst_attn(H_n, H_n, H_n, attn_mask=inst_mask, need_weights=False)
        H = H + out.reshape(B, S, T, D).permute(0, 2, 1, 3)

        H = H + self.ffn(self.norm3(H))
        return H


# ---------------------------------------------------------------------------
# Uniform embedding: scalar u ∈ [0,1] → d_model
# ---------------------------------------------------------------------------

class UniformEmbedding(nn.Module):
    """Small MLP for bounded scalar u ∈ [0,1] → d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = max(d_model // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, u: Tensor) -> Tensor:
        return self.net(u.unsqueeze(-1))


# ---------------------------------------------------------------------------
# Broadcasting cross-attention (SDPA — no OOM expansion)
# ---------------------------------------------------------------------------

class BroadcastingCrossAttention(nn.Module):
    """Multi-head cross-attention via F.scaled_dot_product_attention.

    Unlike nn.MultiheadAttention, this module accepts a 2-D boolean attention
    mask (n_q, n_kv) that is broadcast over the batch and head dimensions by
    SDPA.  This lets every query attend to the *same* K/V sequence but see
    only a query-specific subset of it — without ever expanding K/V.

    Memory cost:  O(B × n_heads × n_q × n_kv) for the attention matrix only.
    No extra allocation proportional to n_q × n_kv × n_kv.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        Q:         Tensor,             # (B, n_q,  D)
        KV:        Tensor,             # (B, n_kv, D)
        attn_mask: Tensor | None = None,  # (n_q, n_kv) bool — True = allowed
    ) -> Tensor:                       # (B, n_q, D)
        B, n_q,  D   = Q.shape
        _,  n_kv, _  = KV.shape
        H, Dh        = self.n_heads, self.d_head

        def _split(x: Tensor, n: int) -> Tensor:
            return x.reshape(B, n, H, Dh).transpose(1, 2)  # (B, H, n, Dh)

        q = _split(self.q_proj(Q),  n_q)
        k = _split(self.k_proj(KV), n_kv)
        v = _split(self.v_proj(KV), n_kv)

        # attn_mask (n_q, n_kv) broadcasts over (B, H, n_q, n_kv)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # out: (B, H, n_q, Dh)

        out = out.transpose(1, 2).reshape(B, n_q, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class AttentionalCopulaEncoder(nn.Module):
    """Joint context encoder in copula space.

    Input:  X_train (B,n_train,p), X_test (B,n_test,p), U_train (B,n_train,d)
    Output: H_enc (B, T, p+d, d_model)   T = n_train + n_test

    Instance attention uses a test-isolation mask so that each test instance
    can attend to all training instances but NOT to other test instances.
    Train instances retain full attention over all T rows.
    This makes each test instance's representation strictly independent of
    other test queries — predictions do not change with batch composition.

    nn.MultiheadAttention mask convention: True = BLOCKED (opposite of SDPA).
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.phi_X   = nn.Linear(1, d_model)
        self.phi_U   = UniformEmbedding(d_model)
        self.t_X     = nn.Parameter(torch.zeros(d_model))
        self.t_U     = nn.Parameter(torch.zeros(d_model))
        self.theta_mask = nn.Parameter(torch.zeros(d_model))
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, l, n_layers)
            for l in range(n_layers)
        ])
        nn.init.normal_(self.phi_X.weight, std=0.02)
        nn.init.zeros_(self.phi_X.bias)

    @staticmethod
    def _inst_mask(T: int, n_train: int, device: torch.device) -> Tensor:
        """Build (T, T) bool mask for instance attention.

        True = BLOCKED  (nn.MultiheadAttention convention).

        Full asymmetric mask for strict instance independence:
          train → train : OPEN   — train representations built from training data only
          test  → train : OPEN   — test queries read training context for ICL
          train → test  : BLOCKED — train representations must not depend on which
                                    test instances are present (ensures that test
                                    instance 0's context is the same in a batch of
                                    10 or a batch of 1)
          test  → test  : BLOCKED except self-attention

        With this mask, train representations are fully determined by the training
        set alone, so each test instance's prediction is identical whether it is
        processed in isolation or alongside other test queries.
        """
        n_test = T - n_train
        mask   = torch.zeros(T, T, dtype=torch.bool, device=device)
        # Train → test: blocked
        mask[:n_train, n_train:] = True
        # Test → test: blocked except diagonal (self-attention)
        if n_test > 1:
            mask[n_train:, n_train:] = True
            test_idx = torch.arange(n_test, device=device)
            mask[n_train + test_idx, n_train + test_idx] = False
        return mask   # (T, T)

    def forward(self, X_train: Tensor, X_test: Tensor, U_train: Tensor) -> Tensor:
        B, n_train, p = X_train.shape
        _, n_test,  _ = X_test.shape
        _, _,       d = U_train.shape

        X_all = torch.cat([X_train, X_test], dim=1)          # (B, T, p)
        E_X   = self.phi_X(X_all.unsqueeze(-1)) + self.t_X   # (B, T, p, D)

        E_U_tr   = self.phi_U(U_train) + self.t_U            # (B, n_train, d, D)
        E_U_te   = (self.theta_mask + self.t_U).expand(B, n_test, d, self.d_model)
        E_U      = torch.cat([E_U_tr, E_U_te], dim=1)        # (B, T, d, D)

        H = torch.cat([E_X, E_U], dim=2)                     # (B, T, p+d, D)
        T = n_train + n_test
        mask = self._inst_mask(T, n_train, X_train.device)
        for block in self.blocks:
            H = block(H, inst_mask=mask)
        return H


# ---------------------------------------------------------------------------
# Autoregressive decoder (memory-efficient, strictly instance-independent)
# ---------------------------------------------------------------------------

class AutoregressiveCopulaDecoder(nn.Module):
    """Autoregressive copula decoder.

    Memory layout at AR step k:
      KV = cat([
          train_tokens:  (B, n_train*(p+d), D)   — shared context
          all_decoded:   (B, n_test*k,      D)   — all instances' past targets
                                                    interleaved as (n_test, k)
      ], dim=1)

    Attention mask (n_test, n_train*(p+d) + n_test*k):
      • All queries can attend to all training tokens.
      • Query i can attend only to its own decoded block
        [n_train*(p+d) + i*k,  n_train*(p+d) + (i+1)*k).

    K/V is never expanded — SDPA broadcasts (B, n_heads, n_kv, d_head)
    over n_test queries.  Total KV memory: O(B * (n_train*S + n_test*k) * D).
    """

    def __init__(self, d_model: int, n_heads: int, n_bins: int):
        super().__init__()
        self.d_model = d_model
        self.n_bins  = n_bins

        # Dedicated embedding for AR conditioning (decoded test targets).
        # Separate from encoder's phi_U — different semantic role.
        self.phi_U_dec = UniformEmbedding(d_model)

        self.norm_q  = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.cross_attn = BroadcastingCrossAttention(d_model, n_heads)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_bins),
        )
        nn.init.normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_instance_mask(
        n_test:    int,
        n_train_S: int,
        k:         int,
        device:    torch.device,
    ) -> Tensor:
        """Boolean mask (n_test, n_train_S + n_test*k).

        True = query i is allowed to attend to that K/V position.
        Training tokens: all queries may attend (first n_train_S columns = True).
        Decoded targets: query i may attend only to its own k-token block.
        """
        n_kv = n_train_S + n_test * k
        mask = torch.zeros(n_test, n_kv, dtype=torch.bool, device=device)
        # Training tokens — always visible
        mask[:, :n_train_S] = True
        if k > 0:
            # Decoded block for instance i starts at n_train_S + i*k
            rows = torch.arange(n_test, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            cols = n_train_S + torch.arange(n_test * k, device=device)
            mask[rows, cols] = True
        return mask   # (n_test, n_kv)

    def _assemble_kv(
        self,
        H_enc:        Tensor,           # (B, T, S, D)
        n_train:      int,
        decoded_embs: list[Tensor],     # k tensors of (B, n_test, D)
    ) -> Tensor:
        """Assemble K/V sequence: [train_tokens | all_decoded_interleaved]."""
        B, T, S, D = H_enc.shape
        n_test = T - n_train
        k      = len(decoded_embs)

        train_kv = H_enc[:, :n_train, :, :].reshape(B, n_train * S, D)

        if k == 0:
            return train_kv   # (B, n_train*S, D)

        # Stack decoded_embs: k × (B, n_test, D)
        # → (B, k, n_test, D) → permute (B, n_test, k, D) → reshape (B, n_test*k, D)
        dec_kv = (
            torch.stack(decoded_embs, dim=1)     # (B, k, n_test, D)
            .permute(0, 2, 1, 3)                 # (B, n_test, k, D)
            .reshape(B, n_test * k, D)           # (B, n_test*k, D)
        )
        return torch.cat([train_kv, dec_kv], dim=1)   # (B, n_train*S + n_test*k, D)

    def _step(
        self,
        Q:            Tensor,   # (B, n_test, D)
        KV:           Tensor,   # (B, n_kv, D)
        attn_mask:    Tensor,   # (n_test, n_kv)
    ) -> Tensor:                # (B, n_test, n_bins)
        Q_n   = self.norm_q(Q)
        KV_n  = self.norm_kv(KV)
        ctx   = self.cross_attn(Q_n, KV_n, attn_mask=attn_mask)  # (B, n_test, D)
        return self.head(ctx)   # (B, n_test, n_bins)

    # ------------------------------------------------------------------
    # Forward (teacher forcing)
    # ------------------------------------------------------------------

    def forward_train(
        self,
        H_enc:       Tensor,   # (B, T, p+d, D)
        U_test_true: Tensor,   # (B, n_test, d)  detached teacher targets
        perm:        Tensor,   # (d,)
        n_train:     int,
        p:           int,
    ) -> Tensor:               # (B, n_test, d, n_bins)
        B, T, S, D = H_enc.shape
        n_test = T - n_train
        d      = S - p
        device = H_enc.device

        all_logits   = H_enc.new_zeros(B, n_test, d, self.n_bins)
        decoded_embs: list[Tensor] = []

        for k in range(d):
            dim_k = perm[k].item()

            Q        = H_enc[:, n_train:, p + dim_k, :]         # (B, n_test, D)
            KV       = self._assemble_kv(H_enc, n_train, decoded_embs)
            n_kv     = KV.shape[1]
            mask     = self._build_instance_mask(
                n_test, n_train * S, k, device
            )                                                    # (n_test, n_kv)

            logits_k = self._step(Q, KV, mask)                  # (B, n_test, n_bins)
            all_logits[:, :, dim_k, :] = logits_k

            emb_k = self.phi_U_dec(U_test_true[:, :, dim_k])   # (B, n_test, D)
            decoded_embs.append(emb_k)

        return all_logits   # (B, n_test, d, n_bins)

    # ------------------------------------------------------------------
    # Inference (autoregressive sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        H_enc:   Tensor,   # (B, T, p+d, D)
        perm:    Tensor,   # (d,)
        n_train: int,
        p:       int,
    ) -> Tensor:           # (B, n_test, d)
        B, T, S, D = H_enc.shape
        n_test = T - n_train
        d      = S - p
        device = H_enc.device

        U_out        = torch.zeros(B, n_test, d, device=device)
        decoded_embs: list[Tensor] = []
        bin_centers  = (
            torch.arange(self.n_bins, device=device).float() + 0.5
        ) / self.n_bins

        for k in range(d):
            dim_k = perm[k].item()

            Q    = H_enc[:, n_train:, p + dim_k, :]
            KV   = self._assemble_kv(H_enc, n_train, decoded_embs)
            mask = self._build_instance_mask(n_test, n_train * S, k, device)

            logits_k = self._step(Q, KV, mask)                  # (B, n_test, n_bins)
            probs    = logits_k.softmax(dim=-1).reshape(B * n_test, self.n_bins)
            bin_idx  = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B*n_test,)

            u_k = bin_centers[bin_idx].reshape(B, n_test)
            U_out[:, :, dim_k] = u_k

            decoded_embs.append(self.phi_U_dec(u_k))            # (B, n_test, D)

        return U_out   # (B, n_test, d)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class AttentionalCopulaModel(nn.Module):
    """In-Context Attentional Copula Model.

    Training:
        logits, U_test = model(X_train, X_test, Y_train, Y_test=Y_test)
        loss = copula_ce_loss(logits, U_test, n_bins)

    Inference:
        U_samples = model(X_train, X_test, Y_train)   # (B, n_test, d) in (0,1)
    """

    def __init__(
        self,
        d_model:  int = 128,
        n_heads:  int = 8,
        n_layers: int = 6,
        d_ff:     int | None = None,
        n_bins:   int = 100,
        dropout:  float = 0.0,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = ((8 * d_model // 3 + 63) // 64) * 64
        self.d_model = d_model
        self.n_bins  = n_bins
        self.encoder = AttentionalCopulaEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.decoder = AutoregressiveCopulaDecoder(d_model, n_heads, n_bins)

    def forward(
        self,
        X_train: Tensor,
        X_test:  Tensor,
        Y_train: Tensor,
        Y_test:  Tensor | None = None,
        perm:    Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """
        Training (Y_test provided):
            Returns (logits (B,n_test,d,n_bins), U_test (B,n_test,d))

        Inference (Y_test=None):
            Returns U_samples (B, n_test, d)
        """
        from copula_loss import empirical_pit, smooth_context_pit

        B, n_train, p = X_train.shape
        _, _,       d = Y_train.shape

        U_train = empirical_pit(Y_train)
        H_enc   = self.encoder(X_train, X_test, U_train)

        if perm is None:
            perm = torch.randperm(d, device=X_train.device)

        if Y_test is not None:
            U_test = smooth_context_pit(Y_test, Y_train).detach()
            logits = self.decoder.forward_train(H_enc, U_test, perm, n_train, p)
            return logits, U_test
        else:
            return self.decoder.sample(H_enc, perm, n_train, p)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_copula_model(cfg: DictConfig, device: str) -> AttentionalCopulaModel:
    d_model  = int(cfg.model.d_model)
    n_heads  = int(cfg.model.n_heads)
    n_layers = int(cfg.model.n_layers)
    n_bins   = int(cfg.model.n_bins)
    dropout  = float(cfg.model.get("dropout", 0.0))
    raw_d_ff = cfg.model.get("d_ff", None)
    d_ff     = int(raw_d_ff) if raw_d_ff is not None else None

    model = AttentionalCopulaModel(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, n_bins=n_bins, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  AttentionalCopulaModel: {n_params:,} trainable parameters")
    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from copula_loss import copula_ce_loss

    B, n_train, n_test, p, d = 4, 20, 5, 10, 4
    n_bins = 100

    model = AttentionalCopulaModel(d_model=64, n_heads=4, n_layers=2, n_bins=n_bins)
    model.train()

    X_train = torch.randn(B, n_train, p)
    X_test  = torch.randn(B, n_test,  p)
    Y_train = torch.randn(B, n_train, d)
    Y_test  = torch.randn(B, n_test,  d)

    # --- Shape and finiteness ---
    perm = torch.randperm(d)
    logits, U_test = model(X_train, X_test, Y_train, Y_test, perm=perm)

    assert logits.shape == (B, n_test, d, n_bins), f"logits shape: {logits.shape}"
    assert U_test.shape == (B, n_test, d),          f"U_test shape: {U_test.shape}"
    assert ((U_test > 0) & (U_test < 1)).all(),     "U_test outside (0,1)"
    assert torch.isfinite(logits).all(),            "non-finite logits"

    # --- Loss and gradient flow ---
    loss = copula_ce_loss(logits, U_test, n_bins)
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for: {name}"

    # --- End-to-end instance independence (logits, not samples) ---
    # With the test-isolation mask in the encoder AND the per-instance decoder memory,
    # logits for instance (0, 0) must be identical whether processed in a batch
    # or as a single query.  This verifies the mask is actually working.
    model.eval()
    with torch.no_grad():
        logits_batch, _  = model(X_train, X_test, Y_train, Y_test, perm=perm)
        logits_single, _ = model(
            X_train[:1], X_test[:1, :1], Y_train[:1], Y_test[:1, :1], perm=perm
        )
    assert torch.allclose(
        logits_batch[0, 0], logits_single[0, 0], atol=1e-4
    ), "Instance independence violated: logits[0,0] differ between batch and single-query run"

    # --- Inference sampling ---
    with torch.no_grad():
        U_samples = model(X_train, X_test, Y_train)
    assert U_samples.shape == (B, n_test, d),         f"U_samples shape: {U_samples.shape}"
    assert ((U_samples > 0) & (U_samples < 1)).all(), "sampled U outside (0,1)"

    n_params = sum(p.numel() for p in model.parameters())
    expected_ce = math.log(n_bins)
    print(f"  logits shape : {logits.shape}")
    print(f"  loss at init : {loss.item():.4f}  (expect ≈ log({n_bins}) = {expected_ce:.2f})")
    print(f"  U_test range : [{U_test.min():.4f}, {U_test.max():.4f}]")
    print(f"  Parameters   : {n_params:,}")
    print("All checks passed.")
