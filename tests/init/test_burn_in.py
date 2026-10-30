"""Tests for initial_walk (jaxrens.init.burn_in).

initial_walk runs MCMC at a fixed Emax to decorrelate the live walkers before
nested sampling starts.  The tests cover the n_walks=0 no-op, shapes,
decorrelation, the Emax ceiling, step-size adaptation (disabled, call count,
chunking), multi-run batches, walker chunking, and the neighbor-bucket
overflow retry.
"""

from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxrens.sampling.moves.random_walk as _rw_mod
from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.adaptation import ResolvedAdaptationPolicy
from jaxrens.init.burn_in import initial_walk
from jaxrens.sampling.batch_descriptor import VmapRuns
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel


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
        single_pos = jax.random.uniform(
            key_pos, (n_atoms, 3), minval=-1.0, maxval=1.0
        )
        positions = jnp.broadcast_to(single_pos[None], (n_walkers, n_atoms, 3))
    else:
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
    """Build a batched NSState, population shapes (n_runs, n_walkers, ...)."""
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
            pos = jax.random.uniform(
                k, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
            )
        else:
            pos = jax.random.uniform(
                keys[1], (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
            )
        all_positions.append(pos)

    positions = jnp.stack(all_positions)  # (n_runs, n_walkers, n_atoms, 3)
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_runs, n_walkers, 3, 3))

    energies = jax.vmap(
        lambda run_pos: jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(run_pos)
    )(
        positions
    )  # (n_runs, n_walkers)

    rng_keys = jax.random.split(key, n_runs)
    ns_states = init_ns_parallel(
        init_fn, positions, types, energies, cells, rng_keys
    )
    return ns_states, step_fn, per_move_fns, backend


def test_zero_walks_returns_same_positions():
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


class TestShapeInvariance:
    """Check that the population keeps its shapes and the iteration counter."""

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
        assert (
            result.population.positions.shape
            == ns_state.population.positions.shape
        )

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
        assert (
            result.population.energy.shape == ns_state.population.energy.shape
        )

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
        assert (
            result.population.step_sizes.shape
            == ns_state.population.step_sizes.shape
        )

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


class TestDecorrelation:
    """Check that walkers that start identical spread out during burn-in."""

    def test_identical_walkers_diverge_after_burn_in(self):
        """Walkers that start at the same position spread out in burn-in."""
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
        assert np.max(diffs) > 1e-6


class TestEmaxConstant:
    """Check that live energies stay under the fixed Emax ceiling."""

    def test_live_energies_do_not_exceed_emax(self):
        """Live energies must stay at or below the fixed Emax ceiling."""
        n_atoms = 2
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=6, n_atoms=n_atoms)

        emax_offset = 3.0
        emax = (
            float(jnp.max(ns_state.population.energy)) + emax_offset * n_atoms
        )

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
        assert np.all(live_energies <= emax + 1e-4)


class TestJIT:
    """Check that initial_walk completes and returns finite values."""

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


class TestAdaptationCallCount:
    """Check how often initial_walk runs the adaptation step.

    initial_walk delegates adaptation to ``build_adapt_step``.  The tests
    patch that builder with a stub that counts its calls.
    """

    @staticmethod
    def _stub_builder(call_count):
        """Return a ``build_adapt_step`` patch that yields a counting closure.

        The closure is a no-op on the population state: it only increments
        ``call_count[0]`` and returns its inputs.  Drop-in for the real
        builder's contract: ``(ns_state, emax, key) -> (ns_state, {}, key)``.
        """

        def patched_builder(*_args, **_kwargs):
            def stub_adapt_step(ns_state, emax, rng_key):
                call_count[0] += 1
                return ns_state, {}, rng_key

            return stub_adapt_step

        return patched_builder

    def test_adjust_called_correct_number_of_times(self):
        """With n_walks=5 and adjust_interval=2, adaptation fires at walks
        0, 2 and 4."""
        ns_state, step_fn, per_move_fns, _ = _build_ns_state(
            n_walkers=4, n_atoms=2
        )

        policy = ResolvedAdaptationPolicy(
            min_rate=0.25,
            max_rate=0.75,
            adjust_factor=1.5,
            step_size_max=10.0,
        )

        adapt_call_count = [0]

        with mock.patch(
            "jaxrens.init.burn_in.build_adapt_step",
            new=self._stub_builder(adapt_call_count),
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

        assert adapt_call_count[0] == 3


class TestAdaptationDisabled:
    """Check that initial_walk skips adaptation when it is not configured."""

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
        """With per_move_fns but adaptation_policies=None, nothing adapts."""
        ns_state, step_fn, per_move_fns, _ = _build_ns_state(
            n_walkers=4, n_atoms=2
        )
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


class TestBatchedRuns:
    """Check initial_walk on a batch of independent runs (VmapRuns).

    The tests cover the output shapes and that runs do not mix.
    """

    def test_batched_output_shapes(self):
        """With n_runs=3, population shapes are (n_runs, n_walkers, ...)."""
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
            batcher=VmapRuns(n_runs=n_runs),
        )

        assert result.population.positions.shape == (
            n_runs,
            n_walkers,
            n_atoms,
            3,
        )
        assert result.population.energy.shape == (n_runs, n_walkers)

    def test_batched_run_independence(self):
        """Changing the population of one run leaves the other runs as is."""
        n_runs, n_walkers, n_atoms = 3, 4, 2
        ns_a, step_fn, _, backend = _build_batched_ns_state(
            n_runs=n_runs, n_walkers=n_walkers, n_atoms=n_atoms
        )
        # Batch B is batch A with run 1 started from other positions.
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        positions_b = ns_a.population.positions.at[1].multiply(0.5)
        energy_b = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions_b[1])
        ns_b = ns_a.set(
            population=ns_a.population.set(
                positions=positions_b,
                energy=ns_a.population.energy.at[1].set(energy_b),
            )
        )

        def walk(ns_states):
            return initial_walk(
                jax.random.key(55),
                ns_states,
                step_fn,
                n_walks=2,
                walklength=5,
                adjust_interval=100,
                emax_offset_per_atom=0.5,
                n_atoms=n_atoms,
                batcher=VmapRuns(n_runs=n_runs),
            )

        pos_a = np.array(walk(ns_a).population.positions)
        pos_b = np.array(walk(ns_b).population.positions)
        np.testing.assert_array_equal(pos_a[0], pos_b[0])
        np.testing.assert_array_equal(pos_a[2], pos_b[2])
        assert not np.array_equal(pos_a[1], pos_b[1])


class TestWalkerChunking:
    """Check that walker_batch_size does not change the result.

    Chunked runs are compared with the full vmap, for batch sizes that divide
    n_walkers, that do not (the population is padded), and that equal it.
    """

    def test_walker_batch_size_divides_evenly_same_result(self):
        n_walkers, n_atoms = 4, 2
        ns_state, step_fn, _, _ = _build_ns_state(
            n_walkers=n_walkers, n_atoms=n_atoms
        )
        key = jax.random.key(7)

        result_full = initial_walk(
            key,
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=0.5,
            n_atoms=n_atoms,
            walker_batch_size=None,
        )
        result_chunked = initial_walk(
            key,
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=0.5,
            n_atoms=n_atoms,
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
        ns_state, step_fn, _, _ = _build_ns_state(
            n_walkers=n_walkers, n_atoms=n_atoms
        )
        key = jax.random.key(11)

        result_full = initial_walk(
            key,
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=0.5,
            n_atoms=n_atoms,
            walker_batch_size=None,
        )
        result_padded = initial_walk(
            key,
            ns_state,
            step_fn,
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=0.5,
            n_atoms=n_atoms,
            walker_batch_size=3,
        )

        assert result_padded.population.positions.shape == (
            n_walkers,
            *result_full.population.positions.shape[1:],
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
        """walker_batch_size=8 on n_walkers=10 (a non-divisor)."""
        n_walkers, n_atoms = 10, 2
        ns_state, step_fn, _, _ = _build_ns_state(
            n_walkers=n_walkers, n_atoms=n_atoms
        )
        result = initial_walk(
            jax.random.key(0),
            ns_state,
            step_fn,
            n_walks=1,
            walklength=2,
            adjust_interval=100,
            emax_offset_per_atom=0.0,
            n_atoms=n_atoms,
            walker_batch_size=8,
        )
        assert result.population.energy.shape[0] == n_walkers

    def test_walker_batch_size_equals_n_walkers_same_as_full_vmap(self):
        n_walkers, n_atoms = 4, 2
        ns_state, step_fn, _, _ = _build_ns_state(
            n_walkers=n_walkers, n_atoms=n_atoms
        )
        key = jax.random.key(13)

        result_full = initial_walk(
            key,
            ns_state,
            step_fn,
            n_walks=2,
            walklength=4,
            adjust_interval=100,
            emax_offset_per_atom=0.0,
            n_atoms=n_atoms,
            walker_batch_size=None,
        )
        result_equal = initial_walk(
            key,
            ns_state,
            step_fn,
            n_walks=2,
            walklength=4,
            adjust_interval=100,
            emax_offset_per_atom=0.0,
            n_atoms=n_atoms,
            walker_batch_size=n_walkers,
        )

        np.testing.assert_allclose(
            np.array(result_full.population.positions),
            np.array(result_equal.population.positions),
            atol=1e-5,
        )


class TestAdaptationChunking:
    """Check that walker_batch_size also reaches the adaptation step.

    Chunked runs are compared with the full vmap on a batch of runs with
    adaptation enabled.
    """

    def _run_with(self, *, walker_batch_size: int | None):
        n_runs, n_walkers, n_atoms = 4, 4, 2
        ns_states, step_fn, per_move_fns, _ = _build_batched_ns_state(
            n_runs=n_runs,
            n_walkers=n_walkers,
            n_atoms=n_atoms,
        )
        policy = ResolvedAdaptationPolicy(
            min_rate=0.25,
            max_rate=0.75,
            adjust_factor=1.5,
            step_size_max=10.0,
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
            batcher=VmapRuns(n_runs=n_runs),
            walker_batch_size=walker_batch_size,
            per_move_fns=per_move_fns,
            adaptation_policies=(policy,),
            adjust_n_samples=8,
            adjust_max_rounds=4,
        )
        return result

    def test_walker_batch_size_matches_full_vmap(self):
        baseline = self._run_with(walker_batch_size=None)
        chunked = self._run_with(walker_batch_size=2)
        np.testing.assert_array_equal(
            np.array(baseline.population.step_sizes),
            np.array(chunked.population.step_sizes),
        )

    def test_walker_batch_size_non_divisor_in_trial_vmap(self):
        baseline = self._run_with(walker_batch_size=None)
        chunked = self._run_with(walker_batch_size=3)
        np.testing.assert_array_equal(
            np.array(baseline.population.step_sizes),
            np.array(chunked.population.step_sizes),
        )
        np.testing.assert_allclose(
            np.array(baseline.population.positions),
            np.array(chunked.population.positions),
            atol=1e-5,
        )


def _wrap_step_fn_with_overflow(
    real_step_fn, *, threshold: int, count_above: int
):
    """Wrap a real step_fn so it sets ``overflow=True`` whenever the current
    bucket ``state.max_neighbors`` is below ``threshold``.

    When overflow fires, ``max_neighbor_count`` is also set to
    ``count_above`` so the burn-in's call to ``_pick_next_bucket`` sees a
    concrete observed peak.  When ``state.max_neighbors >= threshold`` the
    real step_fn's output is passed through unchanged.

    The condition reads ``state.max_neighbors`` (a static int on the
    MCState pytree), so each JIT re-trace following a bucket bump observes
    the new static value and produces the appropriate overflow flag.
    """

    def wrapped(key, state, emax, move_key):
        new_state, info = real_step_fn(key, state, emax, move_key)
        if state.max_neighbors < threshold:
            new_state = new_state.set(
                overflow=jnp.asarray(True),
                max_neighbor_count=jnp.asarray(
                    count_above,
                    dtype=new_state.max_neighbor_count.dtype,
                ),
            )
        return new_state, info

    return wrapped


class TestBurnInOverflowRetry:
    """Check the neighbor-bucket retry in initial_walk.

    When the moves set the population.overflow flag, burn-in bumps the
    neighbor-count bucket and retries the same walk.
    """

    def test_overflow_triggers_bucket_bump_and_retry(self):
        """First walk overflows at bucket=10, retry succeeds at bucket=20."""
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        # Force the initial bucket below the overflow threshold.
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=10),
        )

        overflow_step = _wrap_step_fn_with_overflow(
            step_fn,
            threshold=15,
            count_above=17,
        )

        result = initial_walk(
            jax.random.key(7),
            ns_state,
            overflow_step,
            n_walks=2,
            walklength=3,
            adjust_interval=100,  # no adaptation; keep the test focused
            emax_offset_per_atom=1.0,
            n_atoms=2,
            max_neighbors_list=(10, 20, 30),
            max_neighbors_offset=0,
        )

        # Bucket was bumped 10 -> 20 via _pick_next_bucket(17, 10, ladder, 0).
        assert int(result.population.max_neighbors) == 20
        # The retry succeeded: the final overflow flag is False because
        # bucket=20 satisfies the threshold=15 condition.
        assert bool(jnp.any(result.population.overflow)) is False

    def test_overflow_aborts_when_ladder_exhausted(self):
        """No headroom in ladder → _pick_next_bucket raises RuntimeError."""
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=10),
        )

        # threshold=15, count_above=17, ladder only contains 10 → no entry
        # satisfies "> 10 AND >= 17" → RuntimeError.
        overflow_step = _wrap_step_fn_with_overflow(
            step_fn,
            threshold=15,
            count_above=17,
        )

        with pytest.raises(RuntimeError):
            initial_walk(
                jax.random.key(8),
                ns_state,
                overflow_step,
                n_walks=1,
                walklength=3,
                adjust_interval=100,
                emax_offset_per_atom=1.0,
                n_atoms=2,
                max_neighbors_list=(10,),
                max_neighbors_offset=0,
            )

    def test_no_overflow_path_is_semantics_preserving(self):
        """Without overflow the bucket stays put and the flag is clear."""
        ns_state, step_fn, _, _ = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=30),
        )

        result = initial_walk(
            jax.random.key(9),
            ns_state,
            step_fn,
            n_walks=3,
            walklength=4,
            adjust_interval=100,
            emax_offset_per_atom=1.0,
            n_atoms=2,
            max_neighbors_list=(10, 20, 30),
            max_neighbors_offset=0,
        )

        assert int(result.population.max_neighbors) == 30
        assert bool(jnp.any(result.population.overflow)) is False
