"""
data_gen.py — Episodic data generator for TFM-Multivariate training.

Implements the simulation protocol from §2 of the research specification:

    y_i | x_i ~ N( f(x_i),  Sigma(x_i) )

where the per-instance covariance uses a rank-r low-rank + diagonal structure:

    Sigma(x_i) = diag( d(x_i)^2 ) + V(x_i) @ V(x_i)^T

The three functions f, d, V are parameterised by independently and randomly
initialised MLPs that are frozen (and unique) for each training episode.
This gives every episode a distinct ground-truth conditional distribution for
the in-context learner to discover.

Sampling path (avoids instantiating a d×d matrix per point):

    eps_diag ~ N(0, I_d)      (diagonal noise)
    eps_low  ~ N(0, I_r)      (low-rank noise)
    y = f(x) + diag(x)^{1/2} * eps_diag + V(x) @ eps_low

Both X and Y are z-normalised feature-by-feature after sampling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Batched random MLP — fixed weights, one independent MLP per batch element
# ---------------------------------------------------------------------------


class BatchedRandomMLP(nn.Module):
    """Fixed (non-trainable) random MLPs applied in parallel across a batch.

    Each batch element receives an independently sampled set of weight matrices,
    drawn once at construction time and never updated.  Weights are stored as
    ``register_buffer`` so they move with the module to the target device.

    Architecture: x → Linear(p_in, H) → ReLU → Linear(H, p_out)

    He/Kaiming initialisation is used throughout:
        std = sqrt(2 / fan_in)

    Args:
        batch_size : B — number of independent MLPs
        p_in       : input dimension
        p_out      : output dimension
        hidden     : hidden layer width H
        device     : torch device

    Usage::

        mlp = BatchedRandomMLP(B, p_in=10, p_out=5, hidden=64, device=device)
        out = mlp(X)   # X: (B, T, p_in) → out: (B, T, p_out)
    """

    def __init__(
        self,
        batch_size: int,
        p_in: int,
        p_out: int,
        hidden: int,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        H = hidden
        # Layer 1: p_in → H   (He init: std = sqrt(2 / p_in))
        self.register_buffer(
            "W1", torch.randn(batch_size, H, p_in) * math.sqrt(2.0 / p_in)
        )
        self.register_buffer("b1", torch.zeros(batch_size, H))
        # Layer 2: H → p_out  (He init: std = sqrt(2 / H))
        self.register_buffer(
            "W2", torch.randn(batch_size, p_out, H) * math.sqrt(2.0 / H)
        )
        self.register_buffer("b2", torch.zeros(batch_size, p_out))
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            X : (B, T, p_in)

        Returns:
            out : (B, T, p_out)
        """
        # (B, T, p_in) × (B, p_in, H)^T → (B, T, H)
        h = F.relu(torch.bmm(X, self.W1.transpose(-2, -1)) + self.b1.unsqueeze(1))
        return torch.bmm(h, self.W2.transpose(-2, -1)) + self.b2.unsqueeze(1)


# ---------------------------------------------------------------------------
# Tabular feature prior sampler
# ---------------------------------------------------------------------------


def sample_tabular_x(
    B: int,
    T: int,
    p: int,
    device: torch.device | str,
    n_train: int | None = None,
) -> torch.Tensor:
    """Sample x from a tabular-like mixture prior.

    Each of the p feature dimensions is independently drawn from one of three
    marginal families, chosen uniformly at random for that (episode, feature):

        - Standard normal N(0, 1)
        - Uniform on [-3, 3]
        - Log-normal (exp of N(0, 0.5)) — positive-valued, skewed

    After sampling, every feature dimension is z-normalised using stats
    computed from the first n_train instances only (train split), then applied
    to all T instances.  This prevents test-set statistics from leaking into
    the normalisation.  When n_train is None the full T instances are used
    (backwards-compatible behaviour).

    Args:
        B       : batch size (number of independent datasets)
        T       : total number of instances (n_train + n_test)
        p       : feature dimension
        device  : torch device
        n_train : number of training instances used to compute norm stats;
                  if None, normalise over all T instances.

    Returns:
        X : (B, T, p) — z-normalised tabular features
    """
    # Choose family per (B, p) pair
    family = torch.randint(
        0, 3, (B, p), device=device
    )  # 0=normal, 1=uniform, 2=lognormal

    # Sample all three families up-front, select per feature
    x_normal = torch.randn(B, T, p, device=device)
    x_uniform = torch.rand(B, T, p, device=device) * 6.0 - 3.0
    x_lognorm = torch.exp(torch.randn(B, T, p, device=device) * 0.5)
    # family: (B, p) → (B, 1, p) for broadcasting
    f = family.unsqueeze(1)
    X = torch.where(f == 0, x_normal, torch.where(f == 1, x_uniform, x_lognorm))

    # Z-normalise using train-split stats to avoid test leakage
    ref = X[:, :n_train, :] if n_train is not None else X
    mu = ref.mean(dim=1, keepdim=True)
    std = ref.std(dim=1, keepdim=True) + 1e-8
    return (X - mu) / std


# ---------------------------------------------------------------------------
# Anchor-based covariance generator
# ---------------------------------------------------------------------------


class AnchorCovarianceGen:
    """Anchor keys + V prototypes for structured x-dependent covariance generation.

    Two modes:
      1) shared_across_batch=False (default): each batch element b has its own
         K anchor keys c_{b,k} ∈ R^p and anchor V matrices A_{b,k} ∈ R^{d×r}.
      2) shared_across_batch=True: a single set of anchors (c_k, A_k) is shared
         across all batch elements, reducing per-dataset variation.

    For instance x_i in batch b, the low-rank factor is:

        w_k  = softmax( x_i · c_{b,k} / τ )       k = 1 … K
        V(x_i) = Σ_k w_k * A_{b,k}                 ∈ R^{d×r}

    This creates smooth, x-dependent covariances as a convex combination of K
    prototypes per batch element, with a richer structure than a single random MLP.

    Provides the same call interface as BatchedRandomMLP:
        X : (B, T, p) → (B, T, d*r)
    """

    def __init__(
        self,
        K: int,
        B: int,
        p: int,
        d: int,
        r: int,
        tau: float,
        device: torch.device | str,
        use_mean: bool = False,
        shared_across_batch: bool = False,
    ) -> None:
        self.tau = tau
        self.shared_across_batch = shared_across_batch

        if shared_across_batch:
            # Shared anchors: (K, p), (K, d, r), (K, d)
            C = torch.randn(K, p, device=device)
            self.C = F.normalize(C, dim=-1)
            self.A = torch.randn(K, d, r, device=device) * math.sqrt(2.0 / r)
            self.M = torch.randn(K, d, device=device) if use_mean else None
        else:
            # Per-batch anchors: (B, K, p), (B, K, d, r), (B, K, d)
            C = torch.randn(B, K, p, device=device)
            self.C = F.normalize(C, dim=-1)
            self.A = torch.randn(B, K, d, r, device=device) * math.sqrt(2.0 / r)
            self.M = torch.randn(B, K, d, device=device) if use_mean else None

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X : (B, T, p)
        Returns:
            (B, T, d*r) — same shape as BatchedRandomMLP output (reshape externally)
        """
        if self.shared_across_batch:
            # C: (K, p), A: (K, d, r)
            S = torch.einsum("btp,kp->btk", X, self.C) / self.tau  # (B, T, K)
            W = F.softmax(S, dim=-1)  # (B, T, K)
            V_x = torch.einsum("btk,kdr->btdr", W, self.A)  # (B, T, d, r)
        else:
            # C: (B, K, p), A: (B, K, d, r)
            S = torch.einsum("btp,bkp->btk", X, self.C) / self.tau  # (B, T, K)
            W = F.softmax(S, dim=-1)  # (B, T, K)
            V_x = torch.einsum("btk,bkdr->btdr", W, self.A)  # (B, T, d, r)
        B, T, d, r = V_x.shape
        return V_x.reshape(B, T, d * r)

    def get_mean(self, X: torch.Tensor) -> torch.Tensor:
        """Piecewise constant mean via hard nearest-anchor assignment.

        Each x_i is assigned to the closest anchor by cosine similarity (argmax over
        dot products; C is unit-normalised so this equals nearest Voronoi region).
        All points in the same region share the same constant mean vector M_{b,k*}.

        Args:
            X : (B, T, p)
        Returns:
            mu : (B, T, d)
        """
        if self.M is None:
            raise RuntimeError("get_mean() called but use_mean=False at construction.")
        if self.shared_across_batch:
            # M: (K, d) shared across batch
            dots = torch.einsum("btp,kp->btk", X, self.C)  # (B, T, K)
            k_star = dots.argmax(dim=-1)  # (B, T)
            return self.M[k_star]  # (B, T, d)
        else:
            B_x = X.shape[0]
            dots = torch.einsum("btp,bkp->btk", X, self.C)  # (B, T, K)
            k_star = dots.argmax(dim=-1)  # (B, T)
            return self.M[
                torch.arange(B_x, device=X.device).unsqueeze(1), k_star, :
            ]  # (B, T, d)


class GlobalAnchorCovGen:
    """Cache of AnchorCovarianceGen instances keyed by (p, d) or (B, p, d).

    Drop-in replacement for GlobalFixedNets: created once at training start,
    returns the same AnchorCovarianceGen for a given (B, p, d) combination so
    the x→covariance mapping is fixed across all episodes.
    """

    def __init__(
        self,
        K: int,
        r: int,
        tau: float,
        device: torch.device | str,
        use_mean: bool = False,
        shared_across_batch: bool = False,
    ) -> None:
        self.K = K
        self.r = r
        self.tau = tau
        self.device = device
        self.use_mean = use_mean
        self.shared_across_batch = shared_across_batch
        self._cache: dict[tuple, AnchorCovarianceGen] = {}

    def get(self, B: int, p: int, d: int) -> AnchorCovarianceGen:
        key = (p, d) if self.shared_across_batch else (B, p, d)
        if key not in self._cache:
            self._cache[key] = AnchorCovarianceGen(
                K=self.K,
                B=B,
                p=p,
                d=d,
                r=self.r,
                tau=self.tau,
                device=self.device,
                use_mean=self.use_mean,
                shared_across_batch=self.shared_across_batch,
            )
        return self._cache[key]


# ---------------------------------------------------------------------------
# Global fixed nets cache
# ---------------------------------------------------------------------------


class GlobalFixedNets:
    """Cache of fixed random MLPs keyed by (B, p, d), created once and reused.

    When passed to generate_episode, the same x→covariance function is used
    for every episode throughout training instead of re-sampling new MLPs each
    call.  Different (p, d) combinations each get their own persistent MLP set.
    """

    def __init__(self, r: int, hidden: int, device: torch.device | str) -> None:
        self.r = r
        self.hidden = hidden
        self.device = device
        self._cache: dict[tuple, BatchedRandomMLP] = {}

    def get(self, B: int, p: int, d: int) -> BatchedRandomMLP:
        key = (B, p, d)
        if key not in self._cache:
            v_net = BatchedRandomMLP(
                B, p_in=p, p_out=d * self.r, hidden=self.hidden, device=self.device
            )
            self._cache[key] = v_net
        return self._cache[key]


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------


def generate_episode(
    B: int,
    p: int,
    d: int,
    r: int,
    n_train: int,
    n_test: int,
    device: torch.device | str,
    mlp_hidden: int = 64,
    return_oracle: bool = False,
    fixed_cov: bool = False,
    fixed_cov_rho: float = 0.8,
    fixed_cov_params: tuple | None = None,
    fixed_nets: GlobalFixedNets | None = None,
    anchor_gen: GlobalAnchorCovGen | None = None,
    diag_alpha: float = 0.0,
    return_norm_stats: bool = False,
) -> tuple:
    """Generate one training episode (one gradient step worth of data).

    Creates B independent datasets.  Each dataset has its own randomly
    initialised (and frozen) V_net network that defines the ground-truth
    conditional distribution for that dataset.

    The target covariance is purely low-rank (diagonal set to zero):

        Sigma(x_i) = V_net(x_i) @ V_net(x_i)^T

    Y is sampled via the reparameterisation:

        eps_low ~ N(0, I_r)
        y_i = V_net(x_i) @ eps_low_i

    Both X (feature-wise) and Y (dimension-wise) are z-normalised across
    all T = n_train + n_test instances within each batch element.

    Args:
        B          : number of independent datasets per episode (batch size)
        p          : feature dimension  (x_i ∈ R^p)
        d          : target dimension   (y_i ∈ R^d)
        r          : low-rank factor    (V_i ∈ R^{d×r})
        n_train    : context (training) set size
        n_test     : query (test) set size
        device     : torch device
        mlp_hidden : hidden layer width for the three random MLPs

    Returns:
        X_train : (B, n_train, p) — z-normalised context features
        Y_train : (B, n_train, d) — z-normalised context targets
        X_test  : (B, n_test, p)  — z-normalised query features
        Y_test  : (B, n_test, d)  — z-normalised query targets

        If return_oracle=True, also returns a dict with keys:
            "mu" : (B, n_test, d) — ground-truth conditional mean (normalised space)
            "D"  : (B, n_test, d) — ground-truth diagonal variance (normalised space)
            "V"  : (B, n_test, d, r) — ground-truth low-rank factor (normalised space)

    fixed_cov : bool
        When True, use a fixed zero mean and identity covariance (D=1, V=0) for
        all datasets in all episodes.  Y is still z-normalised so the empirical
        mean is exactly zero.  This is a sanity-check mode: the model only needs
        to output mu=0, D≈1, V≈0 regardless of X.
    """
    T = n_train + n_test

    # 1. Sample x from the tabular prior (z-normalised using train-split stats)
    X = sample_tabular_x(B, T, p, device, n_train=n_train)  # (B, T, p)

    with torch.no_grad():
        if fixed_cov:
            # Fixed zero mean + random covariance (same for all instances, batches, and episodes).
            # fixed_cov_params must be pre-generated once and passed in to guarantee consistency
            # across episodes: (D_fixed, V_fixed) with shapes (1, 1, d) and (1, 1, d, r).
            mu_x = torch.zeros(B, T, d, device=device)
            if fixed_cov_params is not None:
                D_fixed, V_fixed = fixed_cov_params
            else:
                D_fixed = (
                    torch.nn.functional.softplus(torch.randn(1, 1, d, device=device))
                    + 1e-6
                )
                V_fixed = torch.randn(1, 1, d, r, device=device) / math.sqrt(r)
            diag_x = D_fixed.expand(B, T, d)
            V_x = V_fixed.expand(B, T, d, r)
        else:
            # 2. Frozen covariance function — anchor-based, globally-fixed MLP, or fresh MLP.
            #    Priority: anchor_gen > fixed_nets > fresh BatchedRandomMLP per episode.
            if anchor_gen is not None:
                v_net = anchor_gen.get(B, p, d)
            elif fixed_nets is not None:
                v_net = fixed_nets.get(B, p, d)
            else:
                v_net = BatchedRandomMLP(
                    B, p_in=p, p_out=d * r, hidden=mlp_hidden, device=device
                )

            # Piecewise constant mean: hard nearest-anchor assignment. Falls back to zero
            # when anchor_gen does not carry mean vectors.
            if anchor_gen is not None and anchor_gen.use_mean:
                mu_x = v_net.get_mean(X)  # (B, T, d)
            else:
                mu_x = torch.zeros(B, T, d, device=device)  # (B, T, d)
            diag_x = torch.full((B, T, d), diag_alpha, device=device)  # (B, T, d)
            V_x = v_net(X).reshape(B, T, d, r)  # (B, T, d, r)

        # 3. Sample Y via reparameterisation
        eps_diag = torch.randn(B, T, d, device=device)
        eps_low = torch.randn(B, T, r, device=device)

        Y = (
            mu_x
            + diag_x.sqrt() * eps_diag
            + torch.einsum("btdr,btr->btd", V_x, eps_low)
        )  # (B, T, d)

    # 4. Z-normalise Y dimension-wise across context only — prevents leakage
    Y_train_raw = Y[:, :n_train, :]
    mu_y = Y_train_raw.mean(dim=1, keepdim=True)  # (B, 1, d)
    std_y = Y_train_raw.std(dim=1, keepdim=True) + 1e-8
    Y = (Y - mu_y) / std_y

    # 5. Split into train / test
    X_train, X_test = X[:, :n_train], X[:, n_train:]
    Y_train, Y_test = Y[:, :n_train], Y[:, n_train:]

    if return_oracle:
        # Ground-truth parameters in the normalised Y space.
        # y_norm = (y - mu_y) / std_y  transforms the conditional:
        #   mu*  →  (mu_x  - mu_y) / std_y
        #   D*   →  diag_x / std_y**2
        #   V*   →  V_x   / std_y.unsqueeze(-1)
        mu_oracle = (mu_x - mu_y) / std_y
        D_oracle = (diag_x / (std_y**2)).clamp(min=1e-6)  # Woodbury requires D > 0
        V_oracle = V_x / std_y.unsqueeze(-1)
        oracle = {
            "mu": mu_oracle[:, n_train:].detach(),
            "D": D_oracle[:, n_train:].detach(),
            "V": V_oracle[:, n_train:].detach(),
        }
        if return_norm_stats:
            oracle["mu_y"] = mu_y.detach()  # (B, 1, d) in raw Y space (pre-norm)
            oracle["std_y"] = std_y.detach()  # (B, 1, d) in raw Y space (pre-norm)
        return X_train, Y_train, X_test, Y_test, oracle

    if return_norm_stats:
        norm_stats = {"mu_y": mu_y.detach(), "std_y": std_y.detach()}
        return X_train, Y_train, X_test, Y_test, norm_stats

    return X_train, Y_train, X_test, Y_test


# ---------------------------------------------------------------------------
# Validation suite builder
# ---------------------------------------------------------------------------


def build_val_suite(
    cfg,
    device: torch.device | str,
    fixed_nets: GlobalFixedNets | None = None,
    anchor_gen: GlobalAnchorCovGen | None = None,
    fixed_cov_params: tuple | None = None,
) -> dict[str, dict]:
    """Pre-generate a fixed validation suite.

    Creates one episode per (d, n_train) grid point.  Each entry stores the
    episode tensors together with the episode metadata.

    Args:
        cfg    : Hydra DictConfig with cfg.data and cfg.training fields
        device : torch device

    Returns:
        suite : dict keyed by "d{d}_p{n_train}" whose values are dicts with
                keys X_train, Y_train, X_test, Y_test, d, p, n_train, n_test.
    """
    suite: dict[str, dict] = {}
    B = int(cfg.data.val_batch)
    n_test = int(cfg.data.val_n_test)
    r = int(cfg.data.r_data)
    hidden = int(cfg.data.mlp_hidden)
    fixed_cov = bool(cfg.data.get("fixed_cov", False))
    fixed_cov_rho = float(cfg.data.get("fixed_cov_rho", 0.8))
    diag_alpha = float(cfg.data.get("diag_alpha", 0.0))

    for d in cfg.data.val_d_list:
        for n_train in cfg.data.val_n_train_list:
            # Feature dimension p: sample randomly in p_range for each grid point
            p_lo, p_hi = cfg.data.p_range
            p = int(torch.randint(int(p_lo), int(p_hi) + 1, ()).item())

            key = f"d{d}_p{n_train}"
            X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
                B,
                p,
                d,
                r,
                n_train,
                n_test,
                device,
                mlp_hidden=hidden,
                return_oracle=True,
                fixed_cov=fixed_cov,
                fixed_cov_rho=fixed_cov_rho,
                fixed_cov_params=fixed_cov_params,
                fixed_nets=fixed_nets,
                anchor_gen=anchor_gen,
                diag_alpha=diag_alpha,
            )
            suite[key] = {
                "X_train": X_tr,
                "Y_train": Y_tr,
                "X_test": X_te,
                "Y_test": Y_te,
                "oracle_mu": oracle["mu"],
                "oracle_D": oracle["D"],
                "oracle_V": oracle["V"],
                "d": d,
                "p": p,
                "n_train": n_train,
                "n_test": n_test,
            }
    return suite
