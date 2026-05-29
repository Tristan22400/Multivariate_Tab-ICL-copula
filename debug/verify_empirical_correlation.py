"""
verify_empirical_correlation.py — Rigorous whitened-residual test for PIT copula preservation.

Under H₀ (PIT is correct):
    Z_test[b, t, :] ~ N(0, R_ora[b, t])   independently for each (b, t)

Whitening by the per-instance oracle:
    e_t = R_ora[b,t]^{-1/2} @ Z_test[b,t,:]  ~  N(0, I_d)  under H₀

Stacking all e_t yields i.i.d. N(0, I_d).  Empirical covariance should be the
identity regardless of oracle heteroscedasticity.  Any off-diagonal structure
in cov(E) means PIT is not capturing the joint distribution.

Output: debug/verify_correlation.png + printed stats table.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/pit_episodes")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--max_b", type=int, default=None,
                   help="Cap batch elements per episode (default: all)")
    p.add_argument("--out", default="debug/verify_correlation.png")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core: build R_ora and whiten
# ---------------------------------------------------------------------------

def build_R_ora(oracle_D: torch.Tensor, oracle_V: torch.Tensor) -> torch.Tensor:
    """
    oracle_D : (n_test, d)
    oracle_V : (n_test, d, r)
    Returns R_ora : (n_test, d, d)  — per-test-point correlation matrix
    """
    Sigma = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-1, -2)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()   # (n_test, d)
    R = Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))              # (n_test, d, d)
    return R


def whiten_observations(Z: torch.Tensor, R_ora: torch.Tensor) -> torch.Tensor:
    """
    Z     : (n_test, d)   — test observations in Z-space
    R_ora : (n_test, d, d) — per-test oracle correlation matrices

    Returns e : (n_test, d)  where e_t = L_t^{-1} z_t  (L_t = cholesky(R_ora[t]))
    Under H₀: e_t ~ N(0, I_d) i.i.d.
    """
    n_test, d = Z.shape
    jitter = 1e-6 * torch.eye(d, dtype=R_ora.dtype, device=R_ora.device)
    L = torch.linalg.cholesky(R_ora + jitter)                         # (n_test, d, d)
    e = torch.linalg.solve_triangular(
        L, Z.unsqueeze(-1), upper=False
    ).squeeze(-1)                                                      # (n_test, d)
    return e


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir = os.path.join(ROOT, args.data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))
    if not files:
        raise FileNotFoundError(f"No episode_*.pt in {data_dir!r}")
    files = files[: args.n_episodes]
    print(f"Loading {len(files)} episodes from {data_dir}")

    all_e = []   # accumulated whitened residuals

    for fpath in files:
        ep = torch.load(fpath, weights_only=True)
        Z_test   = ep["Z_test"].float()    # (B, n_test, d)
        oracle_D = ep["oracle_D"].float()  # (B, n_test, d)
        oracle_V = ep["oracle_V"].float()  # (B, n_test, d, r)

        B = Z_test.shape[0]
        max_b = B if args.max_b is None else min(B, args.max_b)

        for b in range(max_b):
            R_ora = build_R_ora(oracle_D[b], oracle_V[b])   # (n_test, d, d)
            e = whiten_observations(Z_test[b], R_ora)        # (n_test, d)
            all_e.append(e)

    E = torch.cat(all_e, dim=0)   # (N_total, d)
    N, d = E.shape
    print(f"\nTotal whitened residuals: N={N}, d={d}")

    # -----------------------------------------------------------------------
    # 1. Empirical covariance (should be I_d)
    # -----------------------------------------------------------------------
    cov_white = torch.cov(E.T)   # (d, d)

    diag_vals = cov_white.diagonal()
    ri, ci = torch.triu_indices(d, d, offset=1)
    off_vals = cov_white[ri, ci]

    print("\n--- Whitened residual covariance (should be I_d) ---")
    np.set_printoptions(precision=4, suppress=True)
    print(np.array2string(cov_white.numpy(), precision=4, suppress_small=True))

    print(f"\nDiagonal  : {diag_vals.numpy()}")
    print(f"  mean    : {diag_vals.mean().item():.4f}   (target 1.0)")
    print(f"  std     : {diag_vals.std().item():.4f}")

    print(f"\nOff-diag  : mean |r_ij| = {off_vals.abs().mean().item():.4f}   (target 0.0)")
    print(f"  max |r_ij|           = {off_vals.abs().max().item():.4f}")

    frob_off = off_vals.pow(2).sum().sqrt().item()
    print(f"  ||off-diag||_F       = {frob_off:.4f}   (target 0.0)")

    frob_from_I = (cov_white - torch.eye(d)).pow(2).sum().sqrt().item()
    print(f"\n||cov_white - I||_F   = {frob_from_I:.4f}   (target 0.0)")

    # -----------------------------------------------------------------------
    # 2. Marginal KS test (should be N(0,1))
    # -----------------------------------------------------------------------
    print("\n--- KS test per dimension vs N(0,1) ---")
    for j in range(d):
        ks_stat, ks_p = stats.kstest(E[:, j].numpy(), "norm")
        flag = "  OK" if ks_p > 0.05 else "  FAIL"
        print(f"  dim {j}: KS stat={ks_stat:.4f}, p={ks_p:.4f}{flag}")

    # -----------------------------------------------------------------------
    # 3. Plots
    # -----------------------------------------------------------------------
    n_pairs = min(d * (d - 1) // 2, 6)
    n_rows = 2
    n_cols = max(2, (n_pairs + 1 + 1) // 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    ax_idx = 0

    # Heatmap of cov_white
    ax = axes[ax_idx]; ax_idx += 1
    im = ax.imshow(cov_white.numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("cov(whitened residuals)\n(target: identity)")
    fig.colorbar(im, ax=ax)
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{cov_white[i, j]:.2f}", ha="center", va="center",
                    fontsize=8 if d <= 8 else 5)

    # Histogram of all whitened residuals
    ax = axes[ax_idx]; ax_idx += 1
    e_flat = E.numpy().flatten()
    ax.hist(e_flat, bins=80, density=True, alpha=0.6, label="whitened residuals")
    xs = np.linspace(-4, 4, 300)
    ax.plot(xs, stats.norm.pdf(xs), "r-", lw=2, label="N(0,1)")
    ax.set_title("Marginal distribution of e_t\n(target: N(0,1))")
    ax.legend(fontsize=8)

    # Pairwise scatter plots
    pair_count = 0
    for i in range(d):
        for j in range(i + 1, d):
            if ax_idx >= len(axes) or pair_count >= n_pairs:
                break
            ax = axes[ax_idx]; ax_idx += 1
            idx = np.random.choice(N, size=min(2000, N), replace=False)
            ax.scatter(E[idx, i].numpy(), E[idx, j].numpy(), alpha=0.2, s=5)
            ax.set_xlabel(f"e_dim{i}"); ax.set_ylabel(f"e_dim{j}")
            r = float(cov_white[i, j])
            ax.set_title(f"dims ({i},{j}), r={r:.3f}\n(target: circular cloud)")
            pair_count += 1

    for k in range(ax_idx, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")

    # -----------------------------------------------------------------------
    # 4. Summary verdict
    # -----------------------------------------------------------------------
    print("\n=== VERDICT ===")
    diag_ok = (diag_vals - 1).abs().max().item() < 0.15
    offdiag_ok = off_vals.abs().mean().item() < 0.05
    if diag_ok and offdiag_ok:
        print("PASS: PIT appears to preserve marginals and copula structure.")
    elif diag_ok and not offdiag_ok:
        print("PARTIAL: Marginals OK but off-diagonal correlations are non-zero.")
        print("         PIT is washing out covariance signal — model cannot learn copula.")
    elif not diag_ok and offdiag_ok:
        print("PARTIAL: Off-diagonals OK but marginal calibration is off.")
        print("         TabICL CDFs are miscalibrated.")
    else:
        print("FAIL: Both marginals and correlation structure are wrong.")
        print("      Fundamental PIT failure.")


if __name__ == "__main__":
    main()
