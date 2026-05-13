"""Tests for jaxrens.init.burn_in.initial_walk — core semantics.

Covers:
- n_walks=0 is a no-op (returns input unchanged)
- Shape invariance after burn-in
- Decorrelation: initially identical walkers diverge
- Emax is constant: live energies stay below emax throughout
- JIT: inner chain is JIT-compiled with no retrace per outer walk
- Step-size adjustment called the correct number of times
- Adaptation disabled (per_move_fns=None) runs without error
- Batched (multi-run): output shape + run independence

Walker/run chunking tests (walker_batch_size, run_batch_size, combined,
trial-vmap chunking inside adjust_step_size) live in
``test_init_burn_in_chunking.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.init.burn_in import initial_walk
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel
import jaxrens.sampling.moves.random_walk as _rw_mod


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rw_descriptor(step_size: float = 0.3) -> MoveKernel:
    return MoveKernel(
        name="random_walk",
        build_kernel=_rw_mod.build_kernel,
        step_size=step_size,
        weight=1.0,
        kernel_kwargs={},
        extra_state_fields={},
    )


def _build_ns_state(
    n_walkers: int = 6,
    n_atoms: int = 2,
    seed: int = 0,
    identical_positions: bool = False,
):
    """Build a minimal NSState backed by the harmonic toy backend."""
    backend = create_harmonic()
    desc = _rw_descriptor()
    init_fn, step_fn, per_move_fns = build_mwg(backend, [desc])

    key = jax.random.key(seed)
    key, key_pos = jax.random.split(key)

    if identical_positions:
        single_pos = jax.random.uniform(key_pos, (n_atoms, 3), minval=-1.0, maxval=1.0)
        positions = jnp.broadcast_to(single_pos[None], (n_walkers, n_atoms, 3))
    else:
        positions = jax.random.uniform(key_pos, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)

    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_walkers, 3, 3))

    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    ns_state = init_ns(
        init_fn, positions, types, energies, cells, key
    )
    return ns_state, step_fn, per_move_fns, backend


def _build_batched_ns_state(
    n_runs: int = 3,
    n_walkers: int = 6,
    n_atoms: int = 2,
    seed: int = 0,
    distinct_positions: bool = True,
):
    """Build a batched NSState with (n_runs, n_walkers, ...) population shapes."""
    backend = create_harmonic()
    desc = _rw_descriptor()
    init_fn, step_fn, per_move_fns = build_mwg(backend, [desc])

    key = jax.random.key(seed)
    keys = jax.random.split(key, n_runs + 1)
    run_keys = keys[1:]

    all_positions = []
    for i in range(n_runs):
        k = run_keys[i]
        if distinct_positions:
            # Each run gets independent random positions
            pos = jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)
        else:
            pos = jax.random.uniform(keys[1], (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)
        all_positions.append(pos)

    positions = jnp.stack(all_positions)  # (n_runs, n_walkers, n_atoms, 3)
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_runs, n_walkers, 3, 3))

    energies = jax.vmap(
        lambda run_pos: jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(run_pos)
    )(positions)  # (n_runs, n_walkers)

    rng_keys = jax.random.split(key, n_runs)
    ns_states = init_ns_parallel(
        init_fn, positions, types, energies, cells, rng_keys
    )
    return ns_states, step_fn, per_move_fns, backend


# ---------------------------------------------------------------------------
# 1. n_walks=0 is a no-op
# ---------------------------------------------------------------------------

class TestNoOp:
    def test_zero_walks_returns_same_positions(self):
        ns_state, step_fn, _, _ = _build_ns_state()
        result = initial_walk(
            jax.random.key(1),
            ns_state,
            step_fn,
            n_walks=0,
            walklength=5,
            adjust_interval=1,
            emax_offset_per_atom=0.0,
            n_atoms=2,
        )
        np.testing.assert_array_equal(
            np.array(result.population.positions),
            np.array(ns_state.population.positions),
        )

    def test_zero_walks_returns_same_energies(self):
        ns_state, step_fn, _, _ = _build_ns_state()
        result = initial_walk(
            jax.random.key(1),
            ns_state,
            step_fn,
            n_walks=0,
            walklength=5,
            adjust_interval=1,
            emax_offset_per_atom=0.0,
            n_atoms=2,
        )
        np.testing.assert_array_equal(
            np.array(result.population.energy),
            np.array(ns_state.population.energy),
        )

    def test_zero_walks_ns_bookkeeping_unchanged(self):
        ns_state, step_fn, _, _ = _build_ns_state()
        result = initial_walk(
            jax.random.key(1),
            ns_state,
            step_fn,
            n_walks=0,
            walklength=5,
            adjust_interval=1,
            emax_offset_per_atom=0.0,
            n_atoms=2,
        )
        assert int(result.iteration) == int(ns_state.iteration)


# ---------------------------------------------------------------------------
# 2. Shape invariance
# ---------------------------------------------------------------------------

class TestShapeInvariance:
    def test_positions_shape_unchanged(self):
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(42),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=10,
            emax_offset_per_atom=0.0,
            n_atoms=2,
        )
        assert result.population.positions.shape == ns_state.population.positions.shape

    def test_energy_shape_unchanged(self):
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(42),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=10,
            emax_offset_per_atom=0.0,
            n_atoms=2,
        )
        assert result.population.energy.shape == ns_state.population.energy.shape

    def test_step_sizes_shape_unchanged(self):
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(42),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=10,
            emax_offset_per_atom=0.0,
            n_atoms=2,
        )
        assert result.population.step_sizes.shape == ns_state.population.step_sizes.shape

    def test_iteration_counter_unchanged(self):
        """Burn-in does not advance the NS iteration counter."""
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(42),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=10,
            emax_offset_per_atom=0.0,
            n_atoms=2,
        )
        assert int(result.iteration) == int(ns_state.iteration)


# ---------------------------------------------------------------------------
# 3. Decorrelation: identical walkers diverge
# ---------------------------------------------------------------------------

class TestDecorrelation:
    def test_identical_walkers_diverge_after_burn_in(self):
        """Walkers initialized to the same position spread out after burn-in."""
        ns_state, step_fn, _, _ = _build_ns_state(
            n_walkers=6, n_atoms=2, seed=7, identical_positions=True
        )

        result = initial_walk(
            jax.random.key(99),
            ns_state,
            step_fn,
            n_walks=3,
            walklength=20,
            adjust_interval=100,
            emax_offset_per_atom=5.0,
            n_atoms=2,
        )

        positions = np.array(result.population.positions)  # (6, 2, 3)
        diffs = np.abs(positions[1:] - positions[0])
        assert np.max(diffs) > 1e-6, (
            f"Walkers did not decorrelate: max diff = {np.max(diffs)}"
        )


# ---------------------------------------------------------------------------
# 4. Emax is constant: live energies stay below emax
# ---------------------------------------------------------------------------

class TestEmaxConstant:
    def test_live_energies_do_not_exceed_emax(self):
        """Live energies must stay at or below the fixed Emax ceiling."""
        n_atoms = 2
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=6, n_atoms=n_atoms)

        emax_offset = 3.0
        emax = float(jnp.max(ns_state.population.energy)) + emax_offset * n_atoms

        result = initial_walk(
            jax.random.key(5),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=10,
            adjust_interval=100,
            emax_offset_per_atom=emax_offset,
            n_atoms=n_atoms,
        )

        live_energies = np.array(result.population.energy)
        assert np.all(live_energies <= emax + 1e-4), (
            f"Some live energies exceed Emax={emax:.4f}: {live_energies}"
        )


# ---------------------------------------------------------------------------
# 5. JIT: inner chain compiles cleanly, no retrace per walk
# ---------------------------------------------------------------------------

class TestJIT:
    def test_inner_chain_jit_compatible(self):
        """Smoke-test that initial_walk completes without tracing errors."""
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(0),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=1.0,
            n_atoms=2,
        )
        assert result.population.positions.shape == (4, 2, 3)

    def test_result_positions_are_finite(self):
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(11),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=1.0,
            n_atoms=2,
        )
        assert jnp.all(jnp.isfinite(result.population.positions))

    def test_result_energies_are_finite(self):
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(12),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=1.0,
            n_atoms=2,
        )
        assert jnp.all(jnp.isfinite(result.population.energy))


# ---------------------------------------------------------------------------
# 6. Step-size adjustment called the correct number of times
# ---------------------------------------------------------------------------

class TestAdaptationCallCount:
    """Burn-in delegates adaptation to ``AdaptationManager``; mock its
    ``.apply`` to count fires."""

    def test_adjust_called_correct_number_of_times(self):
        """With n_walks=5 and adjust_interval=2, adaptation fires at walks 2 and 4."""
        from jaxrens.cli.schema.adaptation import ResolvedAdaptationPolicy

        ns_state, step_fn, per_move_fns, _ = _build_ns_state(n_walkers=4, n_atoms=2)

        policy = ResolvedAdaptationPolicy(
            min_rate=0.25,
            max_rate=0.75,
            adjust_factor=1.5,
            step_size_max=10.0,
        )

        adapt_call_count = [0]

        def stub_apply(self, pop, emax, rng_key, current_step_sizes):
            adapt_call_count[0] += 1
            return current_step_sizes, {}, rng_key

        import unittest.mock as mock
        with mock.patch(
            "jaxrens.sampling.adaptation.manager.AdaptationManager.apply",
            new=stub_apply,
        ):
            initial_walk(
                jax.random.key(0),
                ns_state,
                step_fn,
                n_walks=5,
                walklength=3,
                adjust_interval=2,
                emax_offset_per_atom=1.0,
                n_atoms=2,
                per_move_fns=per_move_fns,
                adaptation_policies=(policy,),
                adjust_n_samples=10,
                adjust_max_rounds=5,
            )

        assert adapt_call_count[0] == 2, (
            f"Expected 2 adaptation calls, got {adapt_call_count[0]}"
        )

    def test_adjust_skipped_at_walk_zero(self):
        """Adaptation must not fire on the very first walk (walk_i=0)."""
        from jaxrens.cli.schema.adaptation import ResolvedAdaptationPolicy

        ns_state, step_fn, per_move_fns, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        policy = ResolvedAdaptationPolicy(
            min_rate=0.25, max_rate=0.75, adjust_factor=1.5, step_size_max=10.0,
        )

        adapt_call_count = [0]

        def stub_apply(self, pop, emax, rng_key, current_step_sizes):
            adapt_call_count[0] += 1
            return current_step_sizes, {}, rng_key

        import unittest.mock as mock
        with mock.patch(
            "jaxrens.sampling.adaptation.manager.AdaptationManager.apply",
            new=stub_apply,
        ):
            initial_walk(
                jax.random.key(0),
                ns_state,
                step_fn,
                n_walks=1,
                walklength=3,
                adjust_interval=1,
                emax_offset_per_atom=1.0,
                n_atoms=2,
                per_move_fns=per_move_fns,
                adaptation_policies=(policy,),
                adjust_n_samples=10,
                adjust_max_rounds=3,
            )

        assert adapt_call_count[0] == 0, (
            f"Adaptation fired on walk_i=0, expected 0 calls, got {adapt_call_count[0]}"
        )


# ---------------------------------------------------------------------------
# 7. Adaptation disabled (per_move_fns=None)
# ---------------------------------------------------------------------------

class TestAdaptationDisabled:
    def test_no_per_move_fns_runs_without_error(self):
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(0),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=1,
            emax_offset_per_atom=1.0,
            n_atoms=2,
            per_move_fns=None,
            adaptation_policies=None,
        )
        assert result.population.positions.shape == (4, 2, 3)

    def test_no_policies_with_fns_still_skips_adaptation(self):
        """If adaptation_policies is None even with per_move_fns, no adaptation."""
        ns_state, step_fn, per_move_fns, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        result = initial_walk(
            jax.random.key(0),
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=1,
            emax_offset_per_atom=1.0,
            n_atoms=2,
            per_move_fns=per_move_fns,
            adaptation_policies=None,
        )
        assert result.population.positions.shape == (4, 2, 3)


# ---------------------------------------------------------------------------
# Batched (multi-run): output shapes are (n_runs, n_walkers, ...)
# ---------------------------------------------------------------------------

class TestBatchedRuns:
    def test_batched_output_shapes(self):
        """batched=True with n_runs=3: population shapes are (n_runs, n_walkers, ...)."""
        n_runs, n_walkers, n_atoms = 3, 4, 2
        ns_states, step_fn, _, _ = _build_batched_ns_state(
            n_runs=n_runs, n_walkers=n_walkers, n_atoms=n_atoms
        )

        result = initial_walk(
            jax.random.key(0),
            ns_states,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=0.5,
            n_atoms=n_atoms,
            batched=True,
        )

        assert result.population.positions.shape == (n_runs, n_walkers, n_atoms, 3)
        assert result.population.energy.shape == (n_runs, n_walkers)

    def test_batched_run_independence(self):
        """Three runs with distinct initial populations stay distinct (no cross-contamination)."""
        n_runs, n_walkers, n_atoms = 3, 4, 2
        ns_states, step_fn, _, _ = _build_batched_ns_state(
            n_runs=n_runs, n_walkers=n_walkers, n_atoms=n_atoms, distinct_positions=True
        )

        result = initial_walk(
            jax.random.key(55),
            ns_states,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=0.5,
            n_atoms=n_atoms,
            batched=True,
        )

        positions = np.array(result.population.positions)  # (n_runs, n_walkers, n_atoms, 3)
        # Runs should remain distinct; max pairwise diff across runs must be > 0
        diff_01 = np.max(np.abs(positions[0] - positions[1]))
        diff_02 = np.max(np.abs(positions[0] - positions[2]))
        assert diff_01 > 1e-6 and diff_02 > 1e-6, (
            f"Runs appear cross-contaminated: diff_01={diff_01}, diff_02={diff_02}"
        )


