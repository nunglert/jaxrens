"""Unit tests for InterREManager.

Coverage:
- fires() boundary cases
- apply() no-op for SingleRun
- apply() returns expected stats dict keys
- is_active property
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun, VmapRuns
from jaxrens.sampling.inter_re_manager import InterREManager
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.moves.replica_exchange import PressureRENSSwap
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel


_EXPECTED_STATS_KEYS = frozenset({
    "n_swap_pairs_attempted",
    "n_swap_pairs_accepted",
    "acceptance_rate",
    "n_energy_evals",
    "n_grad_evals",
})


def _make_vmap_ns_state(n_runs=2, n_walkers=10, pressures=None, seed=42):
    """Create a batched NSState suitable for VmapRuns tests."""
    backend = create_harmonic(k=1.0)
    descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    key = jax.random.key(seed)
    keys = jax.random.split(key, n_runs + 1)
    rng_keys = keys[:n_runs]
    pos_key = keys[-1]

    positions_all = jax.random.uniform(
        pos_key, (n_runs, n_walkers, 1, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies_all = jax.vmap(
        lambda pos: jax.vmap(lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0])(pos)
    )(positions_all)

    if pressures is None:
        ensemble_params_per_run = None
    else:
        ensemble_params_per_run = [{"pressure": p} for p in pressures]

    ns_state = init_ns_parallel(
        init_fn, positions_all, types, energies_all, None,
        rng_keys,
        step_sizes=jnp.array([0.2]),
        ensemble_params_per_run=ensemble_params_per_run,
    )
    return ns_state, step_fn, backend


# ---------------------------------------------------------------------------
# fires() boundary cases
# ---------------------------------------------------------------------------


class TestFires:
    def test_fires_at_every_1(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1
        )
        assert not mgr.fires(0)        # iteration 0 never fires
        assert mgr.fires(1)
        assert mgr.fires(2)
        assert mgr.fires(100)

    def test_fires_at_every_5(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=5
        )
        assert not mgr.fires(0)
        assert not mgr.fires(1)
        assert not mgr.fires(4)
        assert mgr.fires(5)
        assert mgr.fires(10)
        assert not mgr.fires(11)

    def test_fires_never_when_every_0(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=0
        )
        for i in range(10):
            assert not mgr.fires(i)

    def test_fires_never_when_every_large(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1000
        )
        for i in range(1, 999):
            assert not mgr.fires(i)
        assert mgr.fires(1000)

    def test_fires_single_run_irrelevant(self):
        # SingleRun: is_active=False, but fires() itself is descriptor-agnostic.
        mgr = InterREManager(
            PressureRENSSwap(), SingleRun(), None, every=1
        )
        # fires() returns True but is_active=False so _run_loop won't call apply.
        assert mgr.fires(1)
        assert not mgr.is_active


# ---------------------------------------------------------------------------
# is_active property
# ---------------------------------------------------------------------------


class TestIsActive:
    def test_single_run_not_active(self):
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, every=1)
        assert not mgr.is_active

    def test_vmap_active(self):
        mgr = InterREManager(PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1)
        assert mgr.is_active

    def test_vmap_not_active_when_every_0(self):
        mgr = InterREManager(PressureRENSSwap(), VmapRuns(n_runs=2), None, every=0)
        assert not mgr.is_active

    def test_pmap_vmap_active(self):
        mgr = InterREManager(PressureRENSSwap(), PmapVmapRuns(1, 2), None, every=1)
        assert mgr.is_active


# ---------------------------------------------------------------------------
# apply() no-op for SingleRun
# ---------------------------------------------------------------------------


class TestApplyNoOpSingleRun:
    def _make_single_ns_state(self):
        backend = create_harmonic(k=1.0)
        descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)
        key = jax.random.key(0)
        pos_key, run_key = jax.random.split(key)
        positions = jax.random.uniform(pos_key, (10, 1, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)
        return init_ns(init_fn, positions, types, energies, None, run_key)

    def test_single_run_state_unchanged(self):
        ns_state = self._make_single_ns_state()
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, every=1)
        key = jax.random.key(99)

        new_state, stats, new_key = mgr.apply(ns_state, key)

        # Populations must be identical (no mutation)
        assert jnp.allclose(
            new_state.population.energy, ns_state.population.energy
        )
        assert jnp.allclose(
            new_state.population.positions, ns_state.population.positions
        )

    def test_single_run_stats_all_zero(self):
        ns_state = self._make_single_ns_state()
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, every=1)
        key = jax.random.key(0)
        _, stats, _ = mgr.apply(ns_state, key)

        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())
        assert stats["n_swap_pairs_attempted"] == 0
        assert stats["n_swap_pairs_accepted"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["n_energy_evals"] == 0
        assert stats["n_grad_evals"] == 0

    def test_single_run_key_advances(self):
        ns_state = self._make_single_ns_state()
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, every=1)
        key = jax.random.key(42)
        _, _, new_key = mgr.apply(ns_state, key)
        # Key should advance (not be identical)
        assert not jnp.array_equal(key, new_key)


# ---------------------------------------------------------------------------
# apply() for VmapRuns returns expected stats keys
# ---------------------------------------------------------------------------


class TestApplyVmapRuns:
    def test_stats_keys_present(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1
        )
        key = jax.random.key(7)
        _, stats, _ = mgr.apply(ns_state, key)

        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())

    def test_stats_non_negative(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1
        )
        key = jax.random.key(7)
        _, stats, _ = mgr.apply(ns_state, key)

        assert stats["n_swap_pairs_attempted"] >= 0
        assert stats["n_swap_pairs_accepted"] >= 0
        assert 0.0 <= stats["acceptance_rate"] <= 1.0

    def test_stats_acceptance_rate_consistent(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1
        )
        key = jax.random.key(7)
        _, stats, _ = mgr.apply(ns_state, key)

        n_att = stats["n_swap_pairs_attempted"]
        n_acc = stats["n_swap_pairs_accepted"]
        expected_rate = n_acc / max(n_att, 1)
        assert abs(stats["acceptance_rate"] - expected_rate) < 1e-6

    def test_key_advances(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1
        )
        key = jax.random.key(42)
        _, _, new_key = mgr.apply(ns_state, key)
        assert not jnp.array_equal(key, new_key)

    def test_state_returned_is_valid_pytree(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1
        )
        key = jax.random.key(7)
        new_state, _, _ = mgr.apply(ns_state, key)

        # Shapes should be preserved
        assert new_state.population.positions.shape == ns_state.population.positions.shape
        assert new_state.population.energy.shape == ns_state.population.energy.shape

    def test_pressure_rens_with_pressures(self):
        """VmapRuns with different pressures: swap uses enthalpy check."""
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2, pressures=[0.01, 0.1])
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, every=1
        )
        key = jax.random.key(77)
        new_state, stats, _ = mgr.apply(ns_state, key)

        # Should complete without error; stats keys present
        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())


# ---------------------------------------------------------------------------
# PmapVmapRuns smoke test
# ---------------------------------------------------------------------------


class TestApplyPmapVmapRuns:
    def test_smoke_n_gpu_1(self):
        """n_gpu=1, n_per_gpu=2: apply runs without error."""
        from jaxrens.sampling.nested_sampling import init_ns_multi_gpu

        backend = create_harmonic(k=1.0)
        descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_gpu, n_per_gpu, n_walkers = 1, 2, 8
        key = jax.random.key(0)
        keys = jax.random.split(key, n_gpu * n_per_gpu + 1)
        rng_keys = keys[:-1]
        pos_key = keys[-1]

        positions = jax.random.uniform(
            pos_key, (n_gpu * n_per_gpu, n_walkers, 1, 3), minval=-2.0, maxval=2.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0])(pos)
        )(positions)

        ns_state = init_ns_multi_gpu(
            init_fn, positions, types, energies, None,
            rng_keys, n_gpu=n_gpu, n_per_gpu=n_per_gpu,
            step_sizes=jnp.array([0.2]),
        )

        descriptor = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)
        mgr = InterREManager(
            PressureRENSSwap(), descriptor, backend, every=1
        )

        swap_key = jax.random.key(99)
        new_state, stats, _ = mgr.apply(ns_state, swap_key)

        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())
        assert new_state.population.positions.shape == ns_state.population.positions.shape

    def test_pmap_vmap_is_active(self):
        descriptor = PmapVmapRuns(n_gpu=1, n_per_gpu=2)
        mgr = InterREManager(PressureRENSSwap(), descriptor, None, every=1)
        assert mgr.is_active


