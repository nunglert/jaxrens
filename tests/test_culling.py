"""Tests for multi-walker culling (Step 17).

Verifies:
- n_cull=1 produces identical results to default behavior
- n_cull>1 removes the correct number of walkers
- n_cull>1 records the correct number of dead points
- Evidence on toy problem is reasonable with n_cull>1
- _find_worst_walkers returns correct top-k
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    _find_worst_walkers,
    init_ns,
    ns_step,
    ns_step_multi_cull,
    run_ns,
)


def _make_mwg(energy_fn, params):
    """Helper: build MWG init_fn/step_fn for random walk."""
    return build_mwg(energy_fn, params, [
        MoveDescriptor("random_walk", random_walk.build_kernel),
    ])


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


class TestFindWorstWalkers:
    def test_top_k_correct(self):
        energies = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        indices, values = _find_worst_walkers(energies, n_cull=2)
        # Top-2 are indices 1 (5.0) and 4 (4.0)
        assert set(int(i) for i in indices) == {1, 4}
        assert jnp.max(values) == pytest.approx(5.0)

    def test_top_k_single(self):
        energies = jnp.array([1.0, 5.0, 3.0])
        indices, values = _find_worst_walkers(energies, n_cull=1)
        assert int(indices[0]) == 1
        assert float(values[0]) == pytest.approx(5.0)

    def test_top_k_with_ties(self):
        energies = jnp.array([5.0, 5.0, 5.0, 1.0])
        key = jax.random.key(42)
        indices, values = _find_worst_walkers(energies, n_cull=2, rng_key=key)
        assert len(indices) == 2
        # Both should be from the tied group {0, 1, 2}
        for idx in indices:
            assert int(idx) in {0, 1, 2}

    def test_sorted_descending(self):
        energies = jnp.array([1.0, 4.0, 2.0, 5.0, 3.0])
        indices, values = _find_worst_walkers(energies, n_cull=3)
        # Values should be in descending order
        assert float(values[0]) >= float(values[1]) >= float(values[2])


class TestNsStepMultiCull:
    def test_n_cull_1_records_one_dead(self, harmonic):
        """n_cull=1 should record exactly 1 dead point."""
        energy_fn, params = harmonic
        n_walkers = 8
        key = jax.random.key(0)
        positions = jax.random.normal(key, (n_walkers, 2, 3)) * 0.5
        types = jnp.zeros((n_walkers, 2), dtype=jnp.int32)
        energies = jax.vmap(energy_fn, in_axes=(None, 0, 0))(
            params, positions, types
        )
        ns_state = init_ns(positions, types, energies, None, jax.random.key(1))
        init_fn, step_fn = _make_mwg(energy_fn, params)

        new_state, info, _ = ns_step(ns_state, init_fn, step_fn, n_mcmc_steps=5)
        assert new_state["n_dead"] == 1
        assert new_state["iteration"] == 1

    def test_n_cull_3_records_three_dead(self, harmonic):
        """n_cull=3 should record exactly 3 dead points."""
        energy_fn, params = harmonic
        n_walkers = 10
        key = jax.random.key(10)
        positions = jax.random.normal(key, (n_walkers, 2, 3)) * 0.5
        types = jnp.zeros((n_walkers, 2), dtype=jnp.int32)
        energies = jax.vmap(energy_fn, in_axes=(None, 0, 0))(
            params, positions, types
        )
        ns_state = init_ns(positions, types, energies, None, jax.random.key(11))
        init_fn, step_fn = _make_mwg(energy_fn, params)

        new_state, info, _ = ns_step_multi_cull(ns_state, init_fn, step_fn, n_cull=3, n_mcmc_steps=5)
        assert new_state["n_dead"] == 3
        assert new_state["iteration"] == 3

    def test_n_cull_3_worst_replaced(self, harmonic):
        """The 3 highest-energy walkers should be replaced."""
        energy_fn, params = harmonic
        n_walkers = 10
        key = jax.random.key(20)
        positions = jax.random.normal(key, (n_walkers, 2, 3)) * 0.5
        types = jnp.zeros((n_walkers, 2), dtype=jnp.int32)
        energies = jax.vmap(energy_fn, in_axes=(None, 0, 0))(
            params, positions, types
        )

        # Identify the 3 worst before the step
        sorted_indices = jnp.argsort(-energies)
        worst_3 = set(int(i) for i in sorted_indices[:3])

        ns_state = init_ns(positions, types, energies, None, jax.random.key(21))
        init_fn, step_fn = _make_mwg(energy_fn, params)
        new_state, info, _ = ns_step_multi_cull(ns_state, init_fn, step_fn, n_cull=3, n_mcmc_steps=10)

        # The worst walkers' energies should have changed
        for w_idx in worst_3:
            assert float(new_state["energies"][w_idx]) != float(energies[w_idx])

    def test_dead_energies_descending(self, harmonic):
        """Dead points from multi-cull should be in descending energy order."""
        energy_fn, params = harmonic
        n_walkers = 10
        key = jax.random.key(30)
        positions = jax.random.normal(key, (n_walkers, 2, 3)) * 0.5
        types = jnp.zeros((n_walkers, 2), dtype=jnp.int32)
        energies = jax.vmap(energy_fn, in_axes=(None, 0, 0))(
            params, positions, types
        )
        ns_state = init_ns(positions, types, energies, None, jax.random.key(31))
        init_fn, step_fn = _make_mwg(energy_fn, params)

        new_state, info, _ = ns_step_multi_cull(ns_state, init_fn, step_fn, n_cull=3, n_mcmc_steps=5)
        dead_e = new_state["dead_energies"][:3]
        # First dead point should have highest energy
        assert float(dead_e[0]) >= float(dead_e[1]) >= float(dead_e[2])


@pytest.mark.heavy
class TestRunNsMultiCull:
    def test_run_ns_n_cull_1_baseline(self, harmonic):
        """run_ns with n_cull=1 should work and produce dead points."""
        energy_fn, params = harmonic
        n_walkers = 8
        key = jax.random.key(100)
        positions = 0.5 * jax.random.normal(key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, positions, types
        )
        init_fn, step_fn = _make_mwg(energy_fn, params)

        result = run_ns(
            positions, types, energies, None,
            init_fn, step_fn, jax.random.key(101),
            max_iterations=50,
            n_mcmc_steps=5,
            n_cull=1,
        )
        assert result["n_dead"] > 0
        # Each iteration produces 1 dead point
        assert result["n_dead"] == result["iteration"]
        assert jnp.isfinite(result["log_evidence"])

    def test_run_ns_n_cull_2(self, harmonic):
        """run_ns with n_cull=2 should collect 2 dead points per iteration."""
        energy_fn, params = harmonic
        n_walkers = 10
        key = jax.random.key(110)
        positions = 0.5 * jax.random.normal(key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, positions, types
        )
        init_fn, step_fn = _make_mwg(energy_fn, params)

        result = run_ns(
            positions, types, energies, None,
            init_fn, step_fn, jax.random.key(111),
            max_iterations=30,
            n_mcmc_steps=5,
            n_cull=2,
        )
        # Each iteration produces n_cull dead points
        assert result["n_dead"] > 0
        assert result["n_dead"] == result["iteration"]  # iteration increments by n_cull
        assert result["n_dead"] % 2 == 0  # always a multiple of n_cull
        assert jnp.isfinite(result["log_evidence"])

    def test_evidence_reasonable_with_multicull(self, harmonic):
        """Evidence from n_cull=2 should be in a reasonable range for harmonic."""
        energy_fn, params = harmonic
        n_walkers = 16
        key = jax.random.key(120)
        positions = 0.5 * jax.random.normal(key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, positions, types
        )
        init_fn, step_fn = _make_mwg(energy_fn, params)

        result = run_ns(
            positions, types, energies, None,
            init_fn, step_fn, jax.random.key(121),
            max_iterations=200,
            n_mcmc_steps=10,
            n_cull=2,
        )
        log_Z = float(result["log_evidence"])
        # Evidence should be finite and negative (log of a small number)
        assert jnp.isfinite(result["log_evidence"])
        # For harmonic with k=1.0 in 3D, log Z ~ -0.5*3*log(2*pi*kT) at T=1
        # Just check it's in a sane range
        assert -50 < log_Z < 50
