"""
oracle_pit_check.py — Per-episode check: does oracle always beat attn?

For each episode computes three oracle copula NLL values:
  1. oracle_tabicl_z  — oracle NLL evaluated on TabICL PIT Z (current metric)
  2. oracle_true_z    — oracle NLL evaluated on true Gaussian PIT Z (bypasses TabICL)
  3. attn_tabicl_z    — attention model NLL on TabICL PIT Z (if ckpt provided)

If the true copula is Gaussian (which it is by construction), oracle_true_z should
always be ≤ 0 and ≤ attn_tabicl_z.  If oracle_tabicl_z > attn_tabicl_z, that means
TabICL PIT distortion is responsible for oracle losing to attn.

Usage:
    conda run -n multivariate-icl python debug/oracle_pit_check.py \
        --data_dir data/pit_episodes \
        --n_episodes 50 \
        [--ckpt hyperplane_tabicl_v2_debug/step_0014999_final.pt]
"""

import argparse
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

from loss import indep_normal_nll, woodbury_nll


# ---------------------------------------------------------------------------
# Oracle copula NLL helpers
# ---------------------------------------------------------------------------


def oracle_rho_params(oracle_D: torch.Tensor, oracle_V: torch.Tensor):
    """Convert (D, V) oracle params to low-rank Cholesky of correlation matrix rho.

    Args:
        oracle_D : (B, N, d)    diagonal variances
        oracle_V : (B, N, d, r) low-rank factor

    Returns:
        D_rho : (B, N, d)    nugget diagonal for rho's Woodbury representation
        V_rho : (B, N, d, d) eigenvector * sqrt(eigenvalue) factor
    """
    Sigma_Y = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
    std_Y = Sigma_Y.diagonal(dim1=-2, dim2=-1).sqrt().clamp(min=1e-8)
    rho = Sigma_Y / (std_Y.unsqueeze(-1) * std_Y.unsqueeze(-2))
    lam, U_eig = torch.linalg.eigh(rho)
    _delta = 1e-4
    D_rho = torch.full_like(oracle_D, _delta)
    V_rho = U_eig * (lam - _delta).clamp(min=0).sqrt().unsqueeze(-2)
    return D_rho, V_rho


def oracle_copula_nll(Z: torch.Tensor, oracle_D: torch.Tensor, oracle_V: torch.Tensor) -> float:
    """Compute oracle copula NLL = woodbury_nll(Z; 0, rho) - indep_normal_nll(Z)."""
    D_rho, V_rho = oracle_rho_params(oracle_D, oracle_V)
    mu_zero = torch.zeros_like(Z)
    return woodbury_nll(Z, mu_zero, D_rho, V_rho).item() - indep_normal_nll(Z).item()


def true_pit_z(Y: torch.Tensor, oracle_mu: torch.Tensor,
               oracle_D: torch.Tensor, oracle_V: torch.Tensor) -> torch.Tensor:
    """Compute oracle PIT Z = (Y - mu) / sigma, bypassing TabICL.

    Args:
        Y        : (B, N, d) raw targets
        oracle_mu: (B, N, d) true conditional mean
        oracle_D : (B, N, d) true diagonal variance
        oracle_V : (B, N, d, r) true low-rank factor

    Returns:
        Z_oracle : (B, N, d) — should follow Gaussian copula with correlation rho
    """
    sigma2 = oracle_D + (oracle_V ** 2).sum(-1)   # (B, N, d)
    sigma = sigma2.clamp(min=1e-8).sqrt()
    return (Y - oracle_mu) / sigma


# ---------------------------------------------------------------------------
# Model loading (optional)
# ---------------------------------------------------------------------------


def load_model(ckpt_path: str, device: str):
    from model import build_copula_tabicl_v2, build_copula_tabicl_v3, build_icl_corr_net_v2
    from omegaconf import OmegaConf
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    # cfg can be a plain dict or OmegaConf DictConfig
    model_cfg_raw = cfg["model"] if isinstance(cfg, dict) else cfg.model
    model_cfg = OmegaConf.create(model_cfg_raw) if isinstance(model_cfg_raw, dict) else model_cfg_raw

    # build_* functions expect cfg.model, so wrap the model sub-config
    cfg_for_build = OmegaConf.create({"model": OmegaConf.to_container(model_cfg, resolve=True)})

    model_type = str(model_cfg.get("name", model_cfg.get("type", "copula_tabicl_v2")))
    if "v3" in model_type:
        model = build_copula_tabicl_v3(cfg_for_build)
    elif "icl_corr_net" in model_type:
        model = build_icl_corr_net_v2(cfg_for_build)
    else:
        model = build_copula_tabicl_v2(cfg_for_build)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/pit_episodes")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--ckpt", default=None, help="Optional model checkpoint for attn comparison")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(_HERE), data_dir)

    eps_files = sorted(f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".pt"))
    eps_files = eps_files[: args.n_episodes]
    print(f"Dataset : {data_dir}")
    print(f"Episodes: {len(eps_files)}  |  device: {device}")

    model = None
    if args.ckpt:
        ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(os.path.dirname(_HERE), args.ckpt)
        print(f"Loading model from {ckpt_path} …")
        model = load_model(ckpt_path, device)
        print("Model loaded.")

    rows = []  # (oracle_tabicl, oracle_true, attn_tabicl or None)

    for ep_file in eps_files:
        ep = torch.load(os.path.join(data_dir, ep_file), map_location=device)

        Z_tr     = ep["Z_train"].to(device)      # (B, n_train, d)
        Z_te     = ep["Z_test"].to(device)       # (B, n_test, d)
        X_tr     = ep["X_train"].to(device)      # (B, n_train, p)
        X_te     = ep["X_test"].to(device)       # (B, n_test,  p)
        oracle_mu = ep["oracle_mu"].to(device)   # (B, n_test, d)
        oracle_D  = ep["oracle_D"].to(device)    # (B, n_test, d)
        oracle_V  = ep["oracle_V"].to(device)    # (B, n_test, d, r)
        Y_te      = ep["Y_test"].to(device)      # (B, n_test, d)

        with torch.no_grad():
            # 1. Oracle NLL on TabICL Z (current pipeline)
            oc_tabicl = oracle_copula_nll(Z_te, oracle_D, oracle_V)

            # 2. Oracle NLL on true-PIT Z (bypasses TabICL)
            Z_oracle = true_pit_z(Y_te, oracle_mu, oracle_D, oracle_V)
            oc_true = oracle_copula_nll(Z_oracle, oracle_D, oracle_V)

            # 3. Attn NLL on TabICL Z (requires model)
            attn_nll = None
            if model is not None:
                B, n_train, _ = Z_tr.shape
                n_sup = max(1, int(0.7 * n_train))
                perm = torch.randperm(n_train, device=device)
                X_sup = X_tr[:, perm[:n_sup]]
                Z_sup = Z_tr[:, perm[:n_sup]]
                X_fwd = torch.cat([X_sup, X_te], dim=1)
                Z_fwd = torch.cat([Z_sup, torch.zeros_like(Z_te)], dim=1)
                mu_all, d_all, V_all = model(X_fwd, Z_fwd, n_support=n_sup)
                mu_all, d_all, V_all = mu_all.float(), d_all.float(), V_all.float()
                # model returns only query-position outputs; all n_test slots are here
                mu_Z = mu_all
                d_Z  = d_all
                V_Z  = V_all
                attn_nll = (
                    woodbury_nll(Z_te, mu_Z, d_Z, V_Z).item()
                    - indep_normal_nll(Z_te).item()
                )

        rows.append((oc_tabicl, oc_true, attn_nll))

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print()
    print(f"{'ep':>4s}  {'oracle(tabicl_z)':>17s}  {'oracle(true_z)':>15s}", end="")
    if model is not None:
        print(f"  {'attn(tabicl_z)':>15s}  {'oracle beats attn?':>18s}", end="")
    print()

    n_oracle_wins = 0
    n_oracle_true_wins = 0
    for i, (oc_tab, oc_true, attn) in enumerate(rows):
        line = f"{i:>4d}  {oc_tab:>17.4f}  {oc_true:>15.4f}"
        if attn is not None:
            oracle_wins = oc_tab <= attn
            oracle_true_wins = oc_true <= attn
            if oracle_wins:
                n_oracle_wins += 1
            if oracle_true_wins:
                n_oracle_true_wins += 1
            line += f"  {attn:>15.4f}  {'YES' if oracle_wins else 'NO':>18s}"
        print(line)

    print()
    oc_tab_mean = sum(r[0] for r in rows) / len(rows)
    oc_true_mean = sum(r[1] for r in rows) / len(rows)
    print(f"Mean oracle(tabicl_z) : {oc_tab_mean:.4f}")
    print(f"Mean oracle(true_z)   : {oc_true_mean:.4f}")
    print(f"PIT distortion cost   : {oc_tab_mean - oc_true_mean:.4f}  (oracle_tabicl - oracle_true, +ve = TabICL hurts oracle)")

    if model is not None:
        attn_mean = sum(r[2] for r in rows) / len(rows)
        print(f"Mean attn(tabicl_z)   : {attn_mean:.4f}")
        print()
        print(f"Oracle(tabicl_z) beats attn: {n_oracle_wins}/{len(rows)} episodes")
        print(f"Oracle(true_z)   beats attn: {n_oracle_true_wins}/{len(rows)} episodes")


if __name__ == "__main__":
    main()
