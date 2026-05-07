"""
Sanity checks for data generation, PIT, and uniform/gaussian transforms.

Run from project root:
    conda run -n multivariate_ICL pytest tests/test_dataset_generation.py -v -s
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from data_gen import generate_episode
from pit import _probit, load_tabicl, run_pit_batched

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TABICL_CKPT = "tabicl-regressor-v2-20260212.ckpt"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tabicl():
    print(f"\n[fixture] Loading frozen TabICL from {TABICL_CKPT!r} on {DEVICE} …")
    model = load_tabicl(TABICL_CKPT, DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fixture] TabICL loaded ({n_params:,} params).")
    return model


@pytest.fixture(scope="module")
def tiny_episode():
    """B=2 tiny episode used by PIT tests."""
    B, p, d, r = 2, 4, 2, 2
    n_train, n_test = 30, 10
    torch.manual_seed(42)
    X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
        B, p, d, r, n_train, n_test, device=DEVICE, return_oracle=True
    )
    return dict(
        X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te,
        oracle=oracle, B=B, p=p, d=d, r=r, n_train=n_train, n_test=n_test,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. generate_episode — shapes & normalisation
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateEpisode:
    B, p, d, r = 3, 5, 2, 2
    n_train, n_test = 20, 8

    @pytest.fixture(autouse=True)
    def episode(self):
        torch.manual_seed(0)
        X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
            self.B, self.p, self.d, self.r,
            self.n_train, self.n_test,
            device=DEVICE, return_oracle=True,
        )
        self.X_tr, self.Y_tr = X_tr, Y_tr
        self.X_te, self.Y_te = X_te, Y_te
        self.oracle = oracle

    def test_output_shapes(self):
        print(f"\n  X_train : {tuple(self.X_tr.shape)}")
        print(f"  Y_train : {tuple(self.Y_tr.shape)}")
        print(f"  X_test  : {tuple(self.X_te.shape)}")
        print(f"  Y_test  : {tuple(self.Y_te.shape)}")
        assert self.X_tr.shape == (self.B, self.n_train, self.p)
        assert self.Y_tr.shape == (self.B, self.n_train, self.d)
        assert self.X_te.shape == (self.B, self.n_test,  self.p)
        assert self.Y_te.shape == (self.B, self.n_test,  self.d)

    def test_oracle_shapes(self):
        print(f"\n  oracle['mu'] : {tuple(self.oracle['mu'].shape)}")
        print(f"  oracle['D']  : {tuple(self.oracle['D'].shape)}")
        print(f"  oracle['V']  : {tuple(self.oracle['V'].shape)}")
        assert self.oracle["mu"].shape == (self.B, self.n_test, self.d)
        assert self.oracle["D"].shape  == (self.B, self.n_test, self.d)
        assert self.oracle["V"].shape  == (self.B, self.n_test, self.d, self.r)

    def test_y_train_zero_mean(self):
        mu = self.Y_tr.mean(dim=1)   # (B, d)
        max_abs = mu.abs().max().item()
        print(f"\n  Y_train max |mean| along n_train = {max_abs:.2e}  (expected < 1e-4)")
        assert max_abs < 1e-4

    def test_y_train_unit_std(self):
        std = self.Y_tr.std(dim=1)   # (B, d)
        max_dev = (std - 1.0).abs().max().item()
        print(f"\n  Y_train max |std - 1| along n_train = {max_dev:.4f}  (expected < 0.05)")
        assert max_dev < 0.05

    def test_x_train_zero_mean(self):
        # X is normalised using train-split stats, so X_train itself has mean 0
        mu = self.X_tr.mean(dim=1)
        max_abs = mu.abs().max().item()
        print(f"\n  X_train max |mean| along n_train = {max_abs:.2e}  (expected < 1e-4)")
        assert max_abs < 1e-4

    def test_x_train_unit_std(self):
        std = self.X_tr.std(dim=1)
        max_dev = (std - 1.0).abs().max().item()
        print(f"\n  X_train max |std - 1| along n_train = {max_dev:.4f}  (expected < 0.05)")
        assert max_dev < 0.05

    def test_oracle_d_non_negative(self):
        min_d = self.oracle["D"].min().item()
        print(f"\n  oracle D min = {min_d:.6f}  (must be >= 0)")
        assert min_d >= 0

    def test_all_tensors_finite(self):
        for name, t in [
            ("X_train", self.X_tr), ("Y_train", self.Y_tr),
            ("X_test",  self.X_te), ("Y_test",  self.Y_te),
            ("oracle_mu", self.oracle["mu"]),
            ("oracle_D",  self.oracle["D"]),
            ("oracle_V",  self.oracle["V"]),
        ]:
            assert t.isfinite().all().item(), f"{name} contains non-finite values"


# ──────────────────────────────────────────────────────────────────────────────
# 2. _probit — uniform → gaussian transform
# ──────────────────────────────────────────────────────────────────────────────

class TestProbitTransform:
    eps = 1e-6

    def test_known_values(self):
        z_mid  = _probit(torch.tensor([0.5]),   self.eps).item()
        z_high = _probit(torch.tensor([0.975]), self.eps).item()
        print(f"\n  _probit(0.5)   = {z_mid:.6f}   expected 0.0")
        print(f"  _probit(0.975) = {z_high:.6f}  expected ≈ 1.96")
        assert abs(z_mid) < 1e-5
        assert abs(z_high - 1.96) < 0.01

    def test_large_uniform_becomes_gaussian(self):
        N = 50_000
        torch.manual_seed(1)
        u = torch.rand(N)
        z = _probit(u, self.eps)
        z_mean = z.mean().item()
        z_std  = z.std().item()
        print(f"\n  probit(U[0,1]) N={N}: mean={z_mean:.4f}, std={z_std:.4f}  (expected ≈ 0, 1)")
        assert abs(z_mean) < 0.05
        assert abs(z_std - 1.0) < 0.05

    def test_monotone_increasing(self):
        u = torch.linspace(0.01, 0.99, 100)
        z = _probit(u, self.eps)
        diffs = z.diff()
        print(f"\n  min diff between consecutive _probit values = {diffs.min().item():.4f}  (must be > 0)")
        assert (diffs > 0).all().item()

    def test_clamping_prevents_inf(self):
        u = torch.tensor([0.0, 1.0])
        z = _probit(u, eps=1e-6)
        print(f"\n  _probit([0, 1]) = {z.tolist()}  (must be finite)")
        assert z.isfinite().all().item()


# ──────────────────────────────────────────────────────────────────────────────
# 3. run_pit_batched — TabICL inference + PIT
# ──────────────────────────────────────────────────────────────────────────────

class TestRunPitBatched:

    def test_output_shapes(self, tabicl, tiny_episode):
        ep = tiny_episode
        Z_tr, Z_te, lp_te = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=16, eps=1e-6,
        )
        print(f"\n  Z_train    : {tuple(Z_tr.shape)}  expected ({ep['B']}, {ep['n_train']}, {ep['d']})")
        print(f"  Z_test     : {tuple(Z_te.shape)}  expected ({ep['B']}, {ep['n_test']},  {ep['d']})")
        print(f"  log_p_test : {tuple(lp_te.shape)}  expected ({ep['B']}, {ep['n_test']},  {ep['d']})")
        assert Z_tr.shape  == (ep["B"], ep["n_train"], ep["d"])
        assert Z_te.shape  == (ep["B"], ep["n_test"],  ep["d"])
        assert lp_te.shape == (ep["B"], ep["n_test"],  ep["d"])

    def test_outputs_finite(self, tabicl, tiny_episode):
        ep = tiny_episode
        Z_tr, Z_te, lp_te = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=16, eps=1e-6,
        )
        assert Z_tr.isfinite().all().item(),  "Z_train has non-finite values"
        assert Z_te.isfinite().all().item(),  "Z_test has non-finite values"
        assert lp_te.isfinite().all().item(), "log_p_test has non-finite values"
        print("\n  All PIT outputs are finite.")

    def test_z_in_reasonable_range(self, tabicl, tiny_episode):
        ep = tiny_episode
        Z_tr, Z_te, _ = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=16, eps=1e-6,
        )
        z_tr_min, z_tr_max = Z_tr.min().item(), Z_tr.max().item()
        z_te_min, z_te_max = Z_te.min().item(), Z_te.max().item()
        print(f"\n  Z_train range : [{z_tr_min:.3f}, {z_tr_max:.3f}]  (expected mostly in [-4, 4])")
        print(f"  Z_test  range : [{z_te_min:.3f}, {z_te_max:.3f}]  (expected mostly in [-4, 4])")
        assert z_tr_min > -8 and z_tr_max < 8
        assert z_te_min > -8 and z_te_max < 8

    def test_log_p_test_is_negative(self, tabicl, tiny_episode):
        ep = tiny_episode
        _, _, lp_te = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=16, eps=1e-6,
        )
        lp_mean = lp_te.mean().item()
        print(f"\n  log_p_test mean = {lp_mean:.4f}  (density → expected < 0)")
        assert lp_mean < 0

    def test_z_train_rough_gaussian(self, tabicl, tiny_episode):
        ep = tiny_episode
        Z_tr, _, _ = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=16, eps=1e-6,
        )
        z_mean = Z_tr.mean().item()
        z_std  = Z_tr.std().item()
        print(f"\n  Z_train empirical: mean={z_mean:.4f}, std={z_std:.4f}")
        print(f"    (n_train={ep['n_train']} — won't be perfect, but mean in (-1.5, 1.5))")
        assert abs(z_mean) < 1.5
        assert 0.2 < z_std < 3.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. K-fold vs LOO PIT consistency
# ──────────────────────────────────────────────────────────────────────────────

class TestKFoldPit:

    @pytest.fixture
    def medium_episode(self):
        B, p, d, r = 2, 4, 2, 2
        n_train, n_test = 40, 8
        torch.manual_seed(42)
        X_tr, Y_tr, X_te, Y_te, _ = generate_episode(
            B, p, d, r, n_train, n_test, device=DEVICE, return_oracle=True
        )
        return dict(X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te,
                    B=B, d=d, n_train=n_train, n_test=n_test)

    def test_test_set_identical(self, tabicl, medium_episode):
        """Test-set predictions are identical for LOO and k-fold (same forward pass)."""
        ep = medium_episode
        _, Z_te_loo, lp_loo = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=20, eps=1e-6,
        )
        _, Z_te_kf, lp_kf = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=20, eps=1e-6, k_folds=4,
        )
        z_diff  = (Z_te_loo - Z_te_kf).abs().max().item()
        lp_diff = (lp_loo   - lp_kf).abs().max().item()
        print(f"\n  Z_test  max abs diff (LOO vs 4-fold) = {z_diff:.2e}  (should be 0)")
        print(f"  log_p   max abs diff (LOO vs 4-fold) = {lp_diff:.2e}  (should be 0)")
        assert z_diff < 1e-5
        assert lp_diff < 1e-5

    def test_kfold_train_finite(self, tabicl, medium_episode):
        ep = medium_episode
        Z_tr_kf, _, _ = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=20, eps=1e-6, k_folds=4,
        )
        assert Z_tr_kf.isfinite().all().item()
        print(f"\n  K-fold Z_train finite and shape {tuple(Z_tr_kf.shape)} — OK.")

    def test_kfold_train_shape(self, tabicl, medium_episode):
        ep = medium_episode
        Z_tr_kf, _, _ = run_pit_batched(
            tabicl, ep["X_tr"], ep["Y_tr"], ep["X_te"], ep["Y_te"],
            pit_batch_size=20, eps=1e-6, k_folds=4,
        )
        assert Z_tr_kf.shape == (ep["B"], ep["n_train"], ep["d"])
