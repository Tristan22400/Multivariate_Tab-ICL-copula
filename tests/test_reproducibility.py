"""
tests/test_reproducibility.py — Strict determinism tests for the copula-intra pipeline.

Run from project root:
    conda run -n multivariate-icl pytest tests/test_reproducibility.py -v

Three test classes:
  1. TestDataGenDeterminism    — generate_episode() and KernelCovGen produce identical
                                 outputs for the same seed and different outputs for
                                 different seeds.
  2. TestDataLoaderDeterminism — make_episode_loader(seed=s) yields the same shuffle
                                 order across runs for equal seeds.
  3. TestModelForwardDeterminism — model.forward() is deterministic for fixed weights
                                   and fixed inputs.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Shared seed helper
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 1. Data generation determinism
# ===========================================================================

class TestDataGenDeterminism:
    """generate_episode() must produce identical output for the same seed."""

    DEVICE = "cpu"

    def _run(self, seed: int) -> tuple:
        from data_gen import generate_episode
        _set_seed(seed)
        return generate_episode(
            B=2, p=5, d=3, r=2,
            n_train=20, n_test=8,
            device=self.DEVICE,
        )

    def test_same_seed_identical_output(self):
        out1 = self._run(seed=0)
        out2 = self._run(seed=0)
        for name, t1, t2 in zip(
            ("X_train", "Y_train", "X_test", "Y_test"), out1, out2
        ):
            assert torch.equal(t1, t2), (
                f"{name}: outputs differ — max abs diff = "
                f"{(t1 - t2).abs().max().item():.6e}"
            )

    def test_different_seeds_different_output(self):
        out1 = self._run(seed=0)
        out2 = self._run(seed=1)
        # X_train must differ between seeds
        assert not torch.equal(out1[0], out2[0]), (
            "X_train is identical for seed=0 and seed=1 — seed is not controlling variance"
        )


class TestKernelCovGenDeterminism:
    """KernelCovGen with all kernel types (including device-RNG-fixed ones)."""

    DEVICE = "cpu"

    def _run_kernel(self, seed: int, kernel: str) -> torch.Tensor:
        from data_gen import KernelCovGen, generate_episode
        _set_seed(seed)
        X_train, _, X_test, _ = generate_episode(
            B=2, p=4, d=3, r=2, n_train=10, n_test=5, device=self.DEVICE
        )
        X_full = torch.cat([X_train, X_test], dim=1)  # (2, 15, 4)
        gen = KernelCovGen(
            kernel_type=kernel, latent_dim=2, mlp_hidden=16, nugget=1e-4
        )
        return gen(X_full, d=3)

    @pytest.mark.parametrize("kernel", [
        "rbf", "exponential", "matern32", "rational_quadratic", "periodic",
    ])
    def test_same_seed_identical(self, kernel: str):
        L1 = self._run_kernel(seed=7, kernel=kernel)
        L2 = self._run_kernel(seed=7, kernel=kernel)
        assert torch.equal(L1, L2), (
            f"KernelCovGen({kernel}): outputs differ for identical seeds — "
            f"max diff = {(L1 - L2).abs().max().item():.6e}"
        )

    @pytest.mark.parametrize("kernel", [
        "rbf", "exponential", "matern32", "rational_quadratic", "periodic",
    ])
    def test_different_seeds_differ(self, kernel: str):
        L1 = self._run_kernel(seed=7, kernel=kernel)
        L2 = self._run_kernel(seed=8, kernel=kernel)
        assert not torch.equal(L1, L2), (
            f"KernelCovGen({kernel}): outputs are identical for seed=7 and seed=8 — "
            "seed is not controlling variance"
        )


class TestIsotropicKernelDeterminism:
    """IsotropicModulatedKernel must also be deterministic per seed."""

    DEVICE = "cpu"

    def _run(self, seed: int, kernel: str) -> torch.Tensor:
        from data_gen import IsotropicModulatedKernel, sample_tabular_x
        _set_seed(seed)
        X = sample_tabular_x(B=2, T=12, p=4, device=self.DEVICE)
        gen = IsotropicModulatedKernel(kernel_type=kernel, nugget=1e-4)
        return gen(X, d=3)

    @pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
    def test_same_seed_identical(self, kernel: str):
        L1 = self._run(seed=3, kernel=kernel)
        L2 = self._run(seed=3, kernel=kernel)
        assert torch.equal(L1, L2), (
            f"IsotropicModulatedKernel({kernel}): max diff = "
            f"{(L1 - L2).abs().max().item():.6e}"
        )

    @pytest.mark.parametrize("kernel", ["rbf", "matern12", "matern32", "matern52"])
    def test_different_seeds_differ(self, kernel: str):
        L1 = self._run(seed=3, kernel=kernel)
        L2 = self._run(seed=4, kernel=kernel)
        assert not torch.equal(L1, L2), (
            f"IsotropicModulatedKernel({kernel}): identical output for different seeds"
        )


# ===========================================================================
# 2. DataLoader shuffle determinism
# ===========================================================================

class TestDataLoaderDeterminism:
    """make_episode_loader(seed=s) must yield the same shuffle order for equal seeds."""

    @staticmethod
    def _write_dummy_episodes(n: int, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for i in range(n):
            torch.save({"idx": i}, os.path.join(directory, f"episode_{i:06d}.pt"))

    @staticmethod
    def _collect_order(directory: str, seed: int) -> list[int]:
        from dataset import make_episode_loader
        loader = make_episode_loader(
            dataset_dir=directory, shuffle=True, num_workers=0, seed=seed
        )
        return [ep["idx"] for ep in loader]

    def test_same_seed_same_order(self, tmp_path):
        self._write_dummy_episodes(10, str(tmp_path))
        order1 = self._collect_order(str(tmp_path), seed=42)
        order2 = self._collect_order(str(tmp_path), seed=42)
        assert order1 == order2, (
            f"Shuffle order differs for seed=42:\n  run1: {order1}\n  run2: {order2}"
        )

    def test_different_seeds_different_order(self, tmp_path):
        self._write_dummy_episodes(10, str(tmp_path))
        order1 = self._collect_order(str(tmp_path), seed=42)
        order2 = self._collect_order(str(tmp_path), seed=99)
        assert order1 != order2, (
            "Different seeds produced identical shuffle order — generator is not wired"
        )

    def test_no_seed_still_loads(self, tmp_path):
        """Sanity: seed=None must not crash (falls back to unseeded behaviour)."""
        self._write_dummy_episodes(4, str(tmp_path))
        from dataset import make_episode_loader
        loader = make_episode_loader(
            dataset_dir=str(tmp_path), shuffle=False, num_workers=0, seed=None
        )
        orders = [ep["idx"] for ep in loader]
        assert len(orders) == 4


# ===========================================================================
# 3. Model forward-pass determinism
# ===========================================================================

class TestModelForwardDeterminism:
    """ICLCorrNetV2 forward pass must be deterministic on CPU with fixed init."""

    DEVICE = "cpu"

    def _build_and_run(self, init_seed: int, input_seed: int) -> tuple[torch.Tensor, ...]:
        from omegaconf import OmegaConf
        from model import build_icl_corr_net_v2

        _set_seed(init_seed)
        cfg = OmegaConf.create({
            "model": {
                "name": "icl_corr_net_v2",
                "d_hidden": 32,
                "n_heads": 2,
                "n_layers": 2,
                "p_max": 10,
                "d_max": 4,
                "rank": 2,
                "dropout": 0.0,
            }
        })
        model = build_icl_corr_net_v2(cfg).to(self.DEVICE)
        model.eval()

        _set_seed(input_seed)
        B, N, d, p = 1, 12, 3, 5
        X = torch.randn(B, N, p)
        Z = torch.randn(B, N, d)

        with torch.no_grad():
            return model(X, Z, n_support=8)

    def test_same_seeds_identical_output(self):
        out1 = self._build_and_run(init_seed=42, input_seed=0)
        out2 = self._build_and_run(init_seed=42, input_seed=0)
        for name, t1, t2 in zip(("mu", "D", "V"), out1, out2):
            assert torch.equal(t1, t2), (
                f"Model output '{name}' differs for identical seeds — "
                f"max diff = {(t1 - t2).abs().max().item():.6e}"
            )

    def test_different_init_seeds_different_output(self):
        out1 = self._build_and_run(init_seed=42, input_seed=0)
        out2 = self._build_and_run(init_seed=99, input_seed=0)
        any_differ = any(not torch.equal(t1, t2) for t1, t2 in zip(out1, out2))
        assert any_differ, (
            "Model outputs are identical for init_seed=42 and init_seed=99 — "
            "weight initialization is not seed-controlled"
        )

    def test_different_input_seeds_different_output(self):
        out1 = self._build_and_run(init_seed=42, input_seed=0)
        out2 = self._build_and_run(init_seed=42, input_seed=1)
        any_differ = any(not torch.equal(t1, t2) for t1, t2 in zip(out1, out2))
        assert any_differ, (
            "Model outputs are identical for different input seeds — "
            "input randomness is not propagating through the model"
        )
