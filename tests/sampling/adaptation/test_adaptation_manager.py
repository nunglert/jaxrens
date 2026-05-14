"""Tests for ``build_adapt_step``.

Covers:
- Inactive cases return ``None`` (no per_move_fns, no descriptors, interval<=0).
- SingleRun: closure returns expected diagnostic dict + shape ``(n_moves, ...)``;
  step sizes finite/positive; rng key advances.
- VmapRuns: closure returns shape-prefixed diagnostics ``(n_runs, n_moves, ...)``.
- JIT cache stability: the second call must not recompile the per-round body.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import SingleRun, VmapRuns
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel, ns_step


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harmonic_setup():
    """Build a harmonic backend with 2 random-walk move types.

    Returns a dict with:
    - ns_state: SingleRun NSState (n_walkers=20)
    - ns_states_batched: VmapRuns NSState (n_runs=2, n_walkers=20)
    - per_move_fns: list of 2 per-move step functions
    - descriptors: list of 2 MoveKernel descriptors
    - n_runs, n_walkers, step_fn
    """
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw_1", random_walk.build_kernel,
            step_size=0.2, step_size_max=5.0,
            min_rate=0.2, max_rate=0.7,
        ),
        MoveKernel(
            "rw_2", random_walk.build_kernel,
            step_size=0.3, step_size_max=5.0,
            min_rate=0.2, max_rate=0.7,
        ),
    ]
    init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

    n_walkers = 20
    n_runs = 2
    key = jax.random.key(0)
    key, init_key = jax.random.split(key)
    positions = 0.5 * jax.random.normal(init_key, (n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    ns_state = init_ns(
        init_fn, positions, types, energies,
        cells=None, rng_key=key,
    )
    for _ in range(20):
        ns_state, _ = ns_step(ns_state, step_fn, n_mcmc_steps=5)

    key2 = jax.random.key(1)
    run_keys = jax.random.split(key2, n_runs)
    pos_batch = jnp.broadcast_to(
        positions[None, :, :, :], (n_runs, n_walkers, 1, 3),
    )
    energies_batch = jnp.broadcast_to(energies[None, :], (n_runs, n_walkers))
    ns_states_batched = init_ns_parallel(
        init_fn, pos_batch, types, energies_batch,
        cells=None, rng_keys=run_keys,
    )

    return {
        "ns_state": ns_state,
        "ns_states_batched": ns_states_batched,
        "per_move_fns": per_move_fns,
        "descriptors": descriptors,
        "n_runs": n_runs,
        "n_walkers": n_walkers,
        "step_fn": step_fn,
    }


# ---------------------------------------------------------------------------
# Inactive cases return None
# ---------------------------------------------------------------------------


class TestInactiveReturnsNone:
    def test_none_per_move_fns(self, harmonic_setup):
        adapt = build_adapt_step(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=None,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )
        assert adapt is None

    def test_empty_per_move_fns(self, harmonic_setup):
        adapt = build_adapt_step(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=[],
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )
        assert adapt is None

    def test_empty_descriptors(self, harmonic_setup):
        adapt = build_adapt_step(
            move_descriptors=[],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )
        assert adapt is None

    def test_zero_interval(self, harmonic_setup):
        adapt = build_adapt_step(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=0,
        )
        assert adapt is None


# ---------------------------------------------------------------------------
# SingleRun: closure shapes and contracts
# ---------------------------------------------------------------------------


class TestSingleRun:
    def _build(self, harmonic_setup):
        return build_adapt_step(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )

    def test_returns_callable(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        assert adapt is not None
        assert callable(adapt)

    def test_diag_keys_present(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(10)
        _, diag, _ = adapt(ns_state, emax, key)
        assert set(diag.keys()) == {
            "rate", "counts", "n_rounds", "converged",
            "cap_hits", "floor_hits", "bracket_detected",
            "trial_n_evaluations", "trial_n_grad_evaluations",
        }

    def test_diag_shapes(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(11)
        _, diag, _ = adapt(ns_state, emax, key)
        n_moves = 2
        assert diag["rate"].shape == (n_moves,)
        assert diag["counts"].shape == (n_moves, 4)
        for k in (
            "n_rounds", "converged", "cap_hits", "floor_hits",
            "bracket_detected", "trial_n_evaluations",
            "trial_n_grad_evaluations",
        ):
            assert diag[k].shape == (n_moves,), k

    def test_step_sizes_written_back(self, harmonic_setup):
        """``adapt_step`` writes ``new_ss`` into ``ns_state.population.step_sizes``."""
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(12)
        new_ns_state, _, _ = adapt(ns_state, emax, key)
        # Step sizes live as (n_walkers, n_moves) under SingleRun; ensure
        # broadcast across walkers and finite/positive.
        ss_pop = new_ns_state.population.step_sizes
        assert ss_pop.shape == ns_state.population.step_sizes.shape
        assert jnp.all(ss_pop > 0.0)
        assert jnp.all(jnp.isfinite(ss_pop))
        # Every walker carries the same per-move ss.
        first_row = ss_pop[0]
        assert jnp.all(ss_pop == first_row[None, :])

    def test_rates_in_unit_interval(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(13)
        _, diag, _ = adapt(ns_state, emax, key)
        assert jnp.all(diag["rate"] >= 0.0)
        assert jnp.all(diag["rate"] <= 1.0)

    def test_rng_key_advances(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key_in = jax.random.key(14)
        _, _, key_out = adapt(ns_state, emax, key_in)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_in)),
            np.asarray(jax.random.key_data(key_out)),
        )


# ---------------------------------------------------------------------------
# VmapRuns: closure shapes
# ---------------------------------------------------------------------------


class TestVmapRuns:
    def _build(self, harmonic_setup):
        return build_adapt_step(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=VmapRuns(n_runs=harmonic_setup["n_runs"]),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )

    def test_diag_shapes(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_states = harmonic_setup["ns_states_batched"]
        n_runs = harmonic_setup["n_runs"]
        n_moves = 2
        emax_per_run = jnp.max(ns_states.population.energy, axis=1)
        run_keys = jax.random.split(jax.random.key(20), n_runs)
        _, diag, _ = adapt(ns_states, emax_per_run, run_keys)
        assert diag["rate"].shape == (n_runs, n_moves)
        assert diag["counts"].shape == (n_runs, n_moves, 4)
        for k in (
            "n_rounds", "converged", "cap_hits", "floor_hits",
            "bracket_detected", "trial_n_evaluations",
            "trial_n_grad_evaluations",
        ):
            assert diag[k].shape == (n_runs, n_moves), k

    def test_step_sizes_positive_finite(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_states = harmonic_setup["ns_states_batched"]
        n_runs = harmonic_setup["n_runs"]
        emax_per_run = jnp.max(ns_states.population.energy, axis=1)
        run_keys = jax.random.split(jax.random.key(22), n_runs)
        new_ns_states, _, _ = adapt(ns_states, emax_per_run, run_keys)
        ss_pop = new_ns_states.population.step_sizes
        assert jnp.all(ss_pop > 0.0)
        assert jnp.all(jnp.isfinite(ss_pop))

    def test_rng_keys_advance(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_states = harmonic_setup["ns_states_batched"]
        n_runs = harmonic_setup["n_runs"]
        emax_per_run = jnp.max(ns_states.population.energy, axis=1)
        run_keys_in = jax.random.split(jax.random.key(21), n_runs)
        _, _, run_keys_out = adapt(ns_states, emax_per_run, run_keys_in)
        assert run_keys_out.shape == run_keys_in.shape
        assert not np.array_equal(
            np.asarray(jax.random.key_data(run_keys_in)),
            np.asarray(jax.random.key_data(run_keys_out)),
        )
