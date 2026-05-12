"""Tests for the batching/chunking knobs on ``initial_walk``.

Split out from ``test_init_burn_in.py``: the semantic tests (no-op, shape
invariance, decorrelation, Emax constraint, JIT, adaptation invocation)
live there. This file covers only the chunking axes:

- walker_batch_size (chunked vmap inside one_walk)
- run_batch_size (chunked vmap over n_runs in batched=True)
- combined walker- and run-chunking
- trial-vmap chunking inside adjust_step_size
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
# Helpers (duplicated from test_init_burn_in.py to keep the chunking split
# self-contained; if these drift, the semantic tests will catch it).
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


def _build_ns_state(n_walkers: int = 6, n_atoms: int = 2, seed: int = 0):
    backend = create_harmonic()
    init_fn, step_fn, per_move_fns = build_mwg(backend, [_rw_descriptor()])

    key = jax.random.key(seed)
    key, key_pos = jax.random.split(key)

    positions = jax.random.uniform(
        key_pos, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_walkers, 3, 3))

    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    ns_state = init_ns(init_fn, positions, types, energies, cells, key)
    return ns_state, step_fn, per_move_fns, backend


def _build_batched_ns_state(
    n_runs: int = 3,
    n_walkers: int = 6,
    n_atoms: int = 2,
    seed: int = 0,
    distinct_positions: bool = True,
):
    backend = create_harmonic()
    init_fn, step_fn, per_move_fns = build_mwg(backend, [_rw_descriptor()])

    key = jax.random.key(seed)
    keys = jax.random.split(key, n_runs + 1)
    run_keys = keys[1:]

    all_positions = []
    for i in range(n_runs):
        k = run_keys[i] if distinct_positions else keys[1]
        pos = jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)
        all_positions.append(pos)

    positions = jnp.stack(all_positions)
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_runs, n_walkers, 3, 3))

    energies = jax.vmap(
        lambda run_pos: jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(run_pos)
    )(positions)

    rng_keys = jax.random.split(key, n_runs)
    ns_states = init_ns_parallel(
        init_fn, positions, types, energies, cells, rng_keys
    )
    return ns_states, step_fn, per_move_fns, backend


# ---------------------------------------------------------------------------
# walker_batch_size: chunked vmap inside one_walk
# ---------------------------------------------------------------------------

class TestWalkerChunking:
    def test_walker_batch_size_divides_evenly_same_result(self):
        n_walkers, n_atoms = 4, 2
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=n_walkers, n_atoms=n_atoms)
        key = jax.random.key(7)

        result_full = initial_walk(
            key, ns_state, step_fn,
            n_walks=2, walklength=5, adjust_interval=100,
            emax_offset_per_atom=0.5, n_atoms=n_atoms,
            walker_batch_size=None,
        )
        result_chunked = initial_walk(
            key, ns_state, step_fn,
            n_walks=2, walklength=5, adjust_interval=100,
            emax_offset_per_atom=0.5, n_atoms=n_atoms,
            walker_batch_size=2,
        )

        np.testing.assert_allclose(
            np.array(result_full.population.positions),
            np.array(result_chunked.population.positions),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(result_full.population.energy),
            np.array(result_chunked.population.energy),
            atol=1e-5,
        )

    def test_walker_batch_size_not_dividing_pads_and_matches_full_vmap(self):
        n_walkers, n_atoms = 4, 2
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=n_walkers, n_atoms=n_atoms)
        key = jax.random.key(11)

        result_full = initial_walk(
            key, ns_state, step_fn,
            n_walks=2, walklength=5, adjust_interval=100,
            emax_offset_per_atom=0.5, n_atoms=n_atoms,
            walker_batch_size=None,
        )
        result_padded = initial_walk(
            key, ns_state, step_fn,
            n_walks=2, walklength=5, adjust_interval=100,
            emax_offset_per_atom=0.5, n_atoms=n_atoms,
            walker_batch_size=3,
        )

        assert result_padded.population.positions.shape == (
            n_walkers, *result_full.population.positions.shape[1:]
        )
        np.testing.assert_allclose(
            np.array(result_full.population.positions),
            np.array(result_padded.population.positions),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(result_full.population.energy),
            np.array(result_padded.population.energy),
            atol=1e-5,
        )

    def test_walker_batch_size_non_divisor_large(self):
        """walker_batch_size=8 on n_walkers=10 (user-reported pattern)."""
        n_walkers, n_atoms = 10, 2
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=n_walkers, n_atoms=n_atoms)
        result = initial_walk(
            jax.random.key(0), ns_state, step_fn,
            n_walks=1, walklength=2, adjust_interval=100,
            emax_offset_per_atom=0.0, n_atoms=n_atoms,
            walker_batch_size=8,
        )
        assert result.population.energy.shape[0] == n_walkers

    def test_walker_batch_size_equals_n_walkers_same_as_full_vmap(self):
        n_walkers, n_atoms = 4, 2
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=n_walkers, n_atoms=n_atoms)
        key = jax.random.key(13)

        result_full = initial_walk(
            key, ns_state, step_fn,
            n_walks=2, walklength=4, adjust_interval=100,
            emax_offset_per_atom=0.0, n_atoms=n_atoms,
            walker_batch_size=None,
        )
        result_equal = initial_walk(
            key, ns_state, step_fn,
            n_walks=2, walklength=4, adjust_interval=100,
            emax_offset_per_atom=0.0, n_atoms=n_atoms,
            walker_batch_size=n_walkers,
        )

        np.testing.assert_allclose(
            np.array(result_full.population.positions),
            np.array(result_equal.population.positions),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# run_batch_size: chunked vmap over n_runs
# ---------------------------------------------------------------------------

class TestRunChunking:
    def test_run_batch_size_divides_evenly_same_result(self):
        n_runs, n_walkers, n_atoms = 3, 4, 2
        ns_states, step_fn, _, _ = _build_batched_ns_state(
            n_runs=n_runs, n_walkers=n_walkers, n_atoms=n_atoms
        )
        key = jax.random.key(42)

        result_full = initial_walk(
            key, ns_states, step_fn,
            n_walks=2, walklength=4, adjust_interval=100,
            emax_offset_per_atom=0.0, n_atoms=n_atoms,
            batched=True, run_batch_size=None,
        )
        result_chunked = initial_walk(
            key, ns_states, step_fn,
            n_walks=2, walklength=4, adjust_interval=100,
            emax_offset_per_atom=0.0, n_atoms=n_atoms,
            batched=True, run_batch_size=1,
        )

        np.testing.assert_allclose(
            np.array(result_full.population.positions),
            np.array(result_chunked.population.positions),
            atol=1e-5,
        )

    def test_run_batch_size_not_dividing_raises(self):
        n_runs, n_walkers, n_atoms = 3, 4, 2
        ns_states, step_fn, _, _ = _build_batched_ns_state(
            n_runs=n_runs, n_walkers=n_walkers, n_atoms=n_atoms
        )
        with pytest.raises(ValueError, match="run_batch_size"):
            initial_walk(
                jax.random.key(0), ns_states, step_fn,
                n_walks=1, walklength=2, adjust_interval=100,
                emax_offset_per_atom=0.0, n_atoms=n_atoms,
                batched=True, run_batch_size=2,
            )


# ---------------------------------------------------------------------------
# walker + run chunking combined
# ---------------------------------------------------------------------------

class TestCombinedChunking:
    def test_combined_walker_and_run_batch_size(self):
        n_runs, n_walkers, n_atoms = 3, 4, 2
        ns_states, step_fn, _, _ = _build_batched_ns_state(
            n_runs=n_runs, n_walkers=n_walkers, n_atoms=n_atoms
        )

        result = initial_walk(
            jax.random.key(0),
            ns_states,
            step_fn,
            n_walks=2,
            walklength=3,
            adjust_interval=100,
            emax_offset_per_atom=0.0,
            n_atoms=n_atoms,
            batched=True,
            walker_batch_size=2,
            run_batch_size=1,
        )

        assert result.population.positions.shape == (n_runs, n_walkers, n_atoms, 3)
        assert result.population.energy.shape == (n_runs, n_walkers)
        assert jnp.all(jnp.isfinite(result.population.positions))


# ---------------------------------------------------------------------------
# Trial-vmap chunking inside adjust_step_size
# ---------------------------------------------------------------------------

class TestAdaptationChunking:
    def _run_with(
        self,
        *,
        walker_batch_size: int | None,
        run_batch_size: int | None,
    ):
        from jaxrens.cli.schema.adaptation import ResolvedAdaptationPolicy

        n_runs, n_walkers, n_atoms = 4, 4, 2
        ns_states, step_fn, per_move_fns, _ = _build_batched_ns_state(
            n_runs=n_runs, n_walkers=n_walkers, n_atoms=n_atoms,
        )
        policy = ResolvedAdaptationPolicy(
            min_rate=0.25, max_rate=0.75, adjust_factor=1.5, step_size_max=10.0,
        )
        result = initial_walk(
            jax.random.key(2026),
            ns_states,
            step_fn,
            n_walks=3,
            walklength=4,
            adjust_interval=1,
            emax_offset_per_atom=2.0,
            n_atoms=n_atoms,
            batched=True,
            walker_batch_size=walker_batch_size,
            run_batch_size=run_batch_size,
            per_move_fns=per_move_fns,
            adaptation_policies=(policy,),
            adjust_n_samples=8,
            adjust_max_rounds=4,
        )
        return result

    def test_walker_batch_size_matches_full_vmap(self):
        baseline = self._run_with(walker_batch_size=None, run_batch_size=None)
        chunked = self._run_with(walker_batch_size=2, run_batch_size=None)
        np.testing.assert_array_equal(
            np.array(baseline.population.step_sizes),
            np.array(chunked.population.step_sizes),
        )

    def test_run_batch_size_matches_full_vmap(self):
        baseline = self._run_with(walker_batch_size=None, run_batch_size=None)
        chunked = self._run_with(walker_batch_size=None, run_batch_size=2)
        np.testing.assert_array_equal(
            np.array(baseline.population.step_sizes),
            np.array(chunked.population.step_sizes),
        )

    def test_combined_chunking_matches_full_vmap(self):
        baseline = self._run_with(walker_batch_size=None, run_batch_size=None)
        chunked = self._run_with(walker_batch_size=2, run_batch_size=2)
        np.testing.assert_array_equal(
            np.array(baseline.population.step_sizes),
            np.array(chunked.population.step_sizes),
        )

    def test_walker_batch_size_non_divisor_in_trial_vmap(self):
        baseline = self._run_with(walker_batch_size=None, run_batch_size=None)
        chunked = self._run_with(walker_batch_size=3, run_batch_size=None)
        np.testing.assert_array_equal(
            np.array(baseline.population.step_sizes),
            np.array(chunked.population.step_sizes),
        )
        np.testing.assert_allclose(
            np.array(baseline.population.positions),
            np.array(chunked.population.positions),
            atol=1e-5,
        )
