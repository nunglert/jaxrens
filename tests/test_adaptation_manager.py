"""Tests for AdaptationManager.

Covers:
- fires() logic (interval, iteration=0, off-interval)
- is_active flag (active vs inactive manager)
- SingleRun apply(): output keys, shapes (n_moves,) and (n_moves, 4) for counts
- VmapRuns apply(): output keys, shapes (n_runs, n_moves) for diagnostics
- JIT cache stability: no retracing after the first call
- is_active=False manager raises RuntimeError on apply()
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import AdaptationManager
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
    - pop: (n_walkers, ...) MCState population (SingleRun shape)
    - pop_batched: (n_runs, n_walkers, ...) MCState population (VmapRuns shape)
    - per_move_fns: list of 2 per-move step functions
    - descriptors: list of 2 MoveKernel descriptors
    - n_runs: 2
    - n_walkers: 20
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
    # Run a few NS steps to tighten the population
    for _ in range(20):
        ns_state, _ = ns_step(ns_state, step_fn, n_mcmc_steps=5)

    pop = ns_state.population

    # Build parallel state: n_runs independent runs
    key2 = jax.random.key(1)
    run_keys = jax.random.split(key2, n_runs)
    pos_batch = jnp.broadcast_to(positions[None, :, :, :], (n_runs, n_walkers, 1, 3))
    energies_batch = jnp.broadcast_to(energies[None, :], (n_runs, n_walkers))
    ns_states_batch = init_ns_parallel(
        init_fn, pos_batch, types, energies_batch,
        cells=None, rng_keys=run_keys,
    )
    pop_batched = ns_states_batch.population

    return {
        "pop": pop,
        "pop_batched": pop_batched,
        "per_move_fns": per_move_fns,
        "descriptors": descriptors,
        "n_runs": n_runs,
        "n_walkers": n_walkers,
        "step_fn": step_fn,
    }


# ---------------------------------------------------------------------------
# fires() tests
# ---------------------------------------------------------------------------


class TestFires:
    def test_fires_zero_always_false(self):
        """fires(0) must always be False regardless of interval."""
        mgr = AdaptationManager(
            move_descriptors=[],
            per_move_fns=None,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=100,
        )
        assert mgr.fires(0) is False

    def test_fires_at_interval(self):
        """fires(adjust_interval) must be True."""
        mgr = AdaptationManager(
            move_descriptors=[],
            per_move_fns=None,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=100,
        )
        assert mgr.fires(100) is True
        assert mgr.fires(200) is True

    def test_fires_off_interval(self):
        """fires(interval+1) must be False."""
        mgr = AdaptationManager(
            move_descriptors=[],
            per_move_fns=None,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=100,
        )
        assert mgr.fires(101) is False
        assert mgr.fires(99) is False

    def test_fires_interval_zero_always_false(self):
        """fires(*) must always be False when adjust_interval=0."""
        mgr = AdaptationManager(
            move_descriptors=[],
            per_move_fns=None,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=0,
        )
        for i in range(500):
            assert mgr.fires(i) is False


# ---------------------------------------------------------------------------
# is_active tests
# ---------------------------------------------------------------------------


class TestIsActive:
    def test_is_active_with_fns(self, harmonic_setup):
        """is_active must be True when per_move_fns + descriptors + interval>0."""
        setup = harmonic_setup
        mgr = AdaptationManager(
            move_descriptors=setup["descriptors"],
            per_move_fns=setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )
        assert mgr.is_active is True

    def test_is_active_false_no_fns(self, harmonic_setup):
        """is_active must be False when per_move_fns=None."""
        setup = harmonic_setup
        mgr = AdaptationManager(
            move_descriptors=setup["descriptors"],
            per_move_fns=None,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )
        assert mgr.is_active is False

    def test_is_active_false_interval_zero(self, harmonic_setup):
        """is_active must be False when adjust_interval=0."""
        setup = harmonic_setup
        mgr = AdaptationManager(
            move_descriptors=setup["descriptors"],
            per_move_fns=setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=0,
        )
        assert mgr.is_active is False

    def test_is_active_false_empty_fns(self, harmonic_setup):
        """is_active must be False when per_move_fns=[]."""
        setup = harmonic_setup
        mgr = AdaptationManager(
            move_descriptors=setup["descriptors"],
            per_move_fns=[],
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )
        assert mgr.is_active is False


# ---------------------------------------------------------------------------
# apply() raises when inactive
# ---------------------------------------------------------------------------


class TestApplyInactive:
    def test_apply_inactive_raises(self, harmonic_setup):
        """apply() must raise RuntimeError when is_active=False."""
        setup = harmonic_setup
        mgr = AdaptationManager(
            move_descriptors=setup["descriptors"],
            per_move_fns=None,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )
        pop = setup["pop"]
        emax = jnp.max(pop.energy)
        ss = pop.step_sizes[0]
        key = jax.random.key(0)
        with pytest.raises(RuntimeError, match="is_active=False"):
            mgr.apply(pop, emax, key, ss)


# ---------------------------------------------------------------------------
# SingleRun apply() tests
# ---------------------------------------------------------------------------


class TestSingleRunApply:
    def _build_mgr(self, harmonic_setup):
        setup = harmonic_setup
        return AdaptationManager(
            move_descriptors=setup["descriptors"],
            per_move_fns=setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )

    def test_output_keys_present(self, harmonic_setup):
        """apply() must return a dict with all 9 diagnostic keys."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop"]
        emax = jnp.max(pop.energy)
        ss = pop.step_sizes[0]  # (n_moves,)
        key = jax.random.key(10)
        new_ss, diags, _ = mgr.apply(pop, emax, key, ss)
        expected_keys = {
            "rate", "counts", "n_rounds", "converged",
            "cap_hits", "floor_hits", "bracket_detected",
            "trial_n_evaluations", "trial_n_grad_evaluations",
        }
        assert set(diags.keys()) == expected_keys

    def test_output_shapes_single_run(self, harmonic_setup):
        """Scalar diagnostics must be shape (n_moves,); counts must be (n_moves, 4)."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop"]
        emax = jnp.max(pop.energy)
        n_moves = 2
        ss = pop.step_sizes[0]  # (n_moves,)
        key = jax.random.key(11)
        new_ss, diags, _ = mgr.apply(pop, emax, key, ss)

        assert new_ss.shape == (n_moves,)
        assert diags["rate"].shape == (n_moves,)
        assert diags["counts"].shape == (n_moves, 4)
        assert diags["n_rounds"].shape == (n_moves,)
        assert diags["converged"].shape == (n_moves,)
        assert diags["cap_hits"].shape == (n_moves,)
        assert diags["floor_hits"].shape == (n_moves,)
        assert diags["bracket_detected"].shape == (n_moves,)
        assert diags["trial_n_evaluations"].shape == (n_moves,)
        assert diags["trial_n_grad_evaluations"].shape == (n_moves,)

    def test_new_step_sizes_positive(self, harmonic_setup):
        """Returned step sizes must all be positive and finite."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop"]
        emax = jnp.max(pop.energy)
        ss = pop.step_sizes[0]
        key = jax.random.key(12)
        new_ss, _, _ = mgr.apply(pop, emax, key, ss)
        assert jnp.all(new_ss > 0.0)
        assert jnp.all(jnp.isfinite(new_ss))

    def test_rates_in_unit_interval(self, harmonic_setup):
        """Acceptance rates must lie in [0, 1]."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop"]
        emax = jnp.max(pop.energy)
        ss = pop.step_sizes[0]
        key = jax.random.key(13)
        _, diags, _ = mgr.apply(pop, emax, key, ss)
        assert jnp.all(diags["rate"] >= 0.0)
        assert jnp.all(diags["rate"] <= 1.0)

    def test_rng_key_is_updated(self, harmonic_setup):
        """The returned rng_key must differ from the input (key was consumed)."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop"]
        emax = jnp.max(pop.energy)
        ss = pop.step_sizes[0]
        key_in = jax.random.key(14)
        _, _, key_out = mgr.apply(pop, emax, key_in, ss)
        # Keys are different after splitting
        key_in_data = np.asarray(jax.random.key_data(key_in))
        key_out_data = np.asarray(jax.random.key_data(key_out))
        assert not np.array_equal(key_in_data, key_out_data)


# ---------------------------------------------------------------------------
# VmapRuns apply() tests
# ---------------------------------------------------------------------------


class TestVmapRunsApply:
    def _build_mgr(self, harmonic_setup):
        n_runs = harmonic_setup["n_runs"]
        setup = harmonic_setup
        return AdaptationManager(
            move_descriptors=setup["descriptors"],
            per_move_fns=setup["per_move_fns"],
            batcher=VmapRuns(n_runs=n_runs),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )

    def test_output_shapes_vmap(self, harmonic_setup):
        """Diagnostics must have shape (n_runs, n_moves); counts: (n_runs, n_moves, 4)."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop_batched"]
        n_runs = harmonic_setup["n_runs"]
        n_moves = 2
        emax_per_run = jnp.max(pop.energy, axis=1)  # (n_runs,)
        # (n_runs, n_moves) step sizes
        ss = pop.step_sizes[:, 0, :]  # (n_runs, n_moves)
        run_keys = jax.random.split(jax.random.key(20), n_runs)
        new_ss, diags, _ = mgr.apply(pop, emax_per_run, run_keys, ss)

        assert new_ss.shape == (n_runs, n_moves)
        assert diags["rate"].shape == (n_runs, n_moves)
        assert diags["counts"].shape == (n_runs, n_moves, 4)
        assert diags["n_rounds"].shape == (n_runs, n_moves)
        assert diags["converged"].shape == (n_runs, n_moves)
        assert diags["cap_hits"].shape == (n_runs, n_moves)
        assert diags["floor_hits"].shape == (n_runs, n_moves)
        assert diags["bracket_detected"].shape == (n_runs, n_moves)
        assert diags["trial_n_evaluations"].shape == (n_runs, n_moves)
        assert diags["trial_n_grad_evaluations"].shape == (n_runs, n_moves)

    def test_vmap_rng_key_updated(self, harmonic_setup):
        """Returned (n_runs,) key must differ from input."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop_batched"]
        n_runs = harmonic_setup["n_runs"]
        emax_per_run = jnp.max(pop.energy, axis=1)
        ss = pop.step_sizes[:, 0, :]
        run_keys_in = jax.random.split(jax.random.key(21), n_runs)
        _, _, run_keys_out = mgr.apply(pop, emax_per_run, run_keys_in, ss)
        assert run_keys_out.shape == run_keys_in.shape
        # At least one key must have changed
        assert not np.array_equal(
            np.asarray(jax.random.key_data(run_keys_in)),
            np.asarray(jax.random.key_data(run_keys_out)),
        )

    def test_vmap_step_sizes_positive_finite(self, harmonic_setup):
        """All returned step sizes must be positive and finite (VmapRuns)."""
        mgr = self._build_mgr(harmonic_setup)
        pop = harmonic_setup["pop_batched"]
        n_runs = harmonic_setup["n_runs"]
        emax_per_run = jnp.max(pop.energy, axis=1)
        ss = pop.step_sizes[:, 0, :]
        run_keys = jax.random.split(jax.random.key(22), n_runs)
        new_ss, _, _ = mgr.apply(pop, emax_per_run, run_keys, ss)
        assert jnp.all(new_ss > 0.0)
        assert jnp.all(jnp.isfinite(new_ss))


# ---------------------------------------------------------------------------
# JIT cache stability test
# ---------------------------------------------------------------------------


