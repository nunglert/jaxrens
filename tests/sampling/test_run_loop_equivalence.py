"""Tests for run_ns, run_ns_parallel and the overflow retry of _run_loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import SingleRun
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns,
    ns_step,
    run_ns,
    run_ns_parallel,
)
from jaxrens.sampling.run_loop import _run_loop
from jaxrens.sampling.termination import IterationTermination


def _build_harmonic_setup(
    seed: int = 42,
    n_walkers: int = 20,
    n_atoms: int = 1,
):
    """Return a minimal harmonic NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("random_walk", random_walk.build_kernel),
        ],
    )
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "backend": backend,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
        "n_walkers": n_walkers,
    }


class TestShortRuns:
    """Check short, fixed-seed runs of run_ns and run_ns_parallel.

    ``convergence_threshold=1e6`` makes the prior-mass criterion fire as soon
    as it may, which is after ``n_walkers`` iterations.  The runs here use at
    most 20 iterations with 20 walkers, so they end at the iteration limit.
    """

    def _run_short_ns(self, seed: int = 42, n_iter: int = 10):
        s = _build_harmonic_setup(seed=seed)
        result = run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=n_iter,
            n_mcmc_steps=3,
            initial_step_size=0.3,
            convergence_threshold=1e6,
        )
        return result

    def test_determinism_single_run(self):
        """Two identical calls to run_ns produce bit-identical log_evidence."""
        r1 = self._run_short_ns(seed=42, n_iter=10)
        r2 = self._run_short_ns(seed=42, n_iter=10)
        assert float(r1["log_evidence"]) == float(r2["log_evidence"])
        assert r1["n_dead"] == r2["n_dead"]

    def test_log_evidence_finite(self):
        r = self._run_short_ns(seed=42, n_iter=10)
        assert jnp.isfinite(jnp.asarray(r["log_evidence"]))

    def test_n_dead_equals_iterations(self):
        r = self._run_short_ns(seed=42, n_iter=10)
        assert r["n_dead"] == 10

    def _run_parallel_n1(self, seed: int = 77, n_iter: int = 20):
        """Run run_ns_parallel with a single run."""
        s = _build_harmonic_setup(seed=seed, n_walkers=20)
        rng_keys = jax.random.split(s["key"], 1)
        result = run_ns_parallel(
            s["positions"][None],
            s["types"],
            s["energies"][None],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=rng_keys,
            max_iterations=n_iter,
            n_mcmc_steps=3,
            initial_step_size=0.3,
            convergence_threshold=1e6,
        )
        return result

    def test_parallel_n_runs_1_output_shapes(self):
        """A single parallel run gives (1, ...) arrays and finite evidence."""
        r = self._run_parallel_n1(seed=77, n_iter=20)
        assert jnp.all(jnp.isfinite(r["log_evidence"]))
        assert r["log_evidence"].shape == (1,)
        assert r["n_dead"].shape == (1,)
        assert r["n_runs"] == 1


class TestBatchDescriptorInInfo:
    """Check the BatchDescriptor that run_ns attaches to info."""

    def test_single_run_info_batch_is_single(self):
        """info["_batcher"] is a SingleRun, which is not batched."""
        s = _build_harmonic_setup(seed=10, n_walkers=15)

        captured_batches = []

        class _Capture:
            def on_iteration(self, iteration, ns_state, info):
                if "_batcher" in info:
                    captured_batches.append(info["_batcher"])

            def on_finish(self, ns_state):
                pass

        run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=5,
            n_mcmc_steps=3,
            callbacks=[_Capture()],
        )

        assert len(captured_batches) > 0
        assert all(isinstance(b, SingleRun) for b in captured_batches)
        assert all(not b.is_batched for b in captured_batches)


class _OverflowRunner:
    """Run _run_loop with an overflow injected on one chosen call.

    A real neighbor-list backend is not needed: a patched ``wrap_step`` calls
    the real ``ns_step`` and sets ``overflow=True`` on call ``overflow_call``.
    """

    @staticmethod
    def run_with_overflow(seed, n_walkers, n_iter, overflow_call, fake_count):
        """Return the final state, the call count and the buckets seen.

        ``max_neighbors_seen[i]`` is the bucket at the start of call i + 1.
        """
        s = _build_harmonic_setup(seed=seed, n_walkers=n_walkers)
        ns_state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
            step_sizes=jnp.full(1, 0.3),
        )

        call_count = [0]
        max_neighbors_seen = []
        real_jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))

        def patched_step_fn(state, sfn, n_mcmc, n_extra):
            call_count[0] += 1
            max_neighbors_seen.append(int(state.population.max_neighbors))
            new_state, info = real_jit_step(state, sfn, n_mcmc, n_extra)
            if call_count[0] == overflow_call:
                pop = new_state.population
                injected_pop = pop.set(
                    overflow=jnp.ones(pop.overflow.shape, dtype=jnp.bool_),
                    max_neighbor_count=jnp.full(
                        pop.max_neighbor_count.shape,
                        fake_count,
                        dtype=jnp.int32,
                    ),
                )
                return new_state.set(population=injected_pop), info
            return new_state, info

        class _PatchedSingleRun(SingleRun):
            def wrap_step(
                self, ns_step_fn, step_fn_inner, n_mcmc_steps, n_extra
            ):
                return lambda state, sfn, nm, ne: patched_step_fn(
                    state, sfn, nm, ne
                )

        batcher = _PatchedSingleRun()
        adapt_step = build_adapt_step(
            move_descriptors=[],
            per_move_fns=None,
            batcher=batcher,
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=0,
        )
        final_state, _, _ = _run_loop(
            batcher=batcher,
            adapt_step=adapt_step,
            adjust_interval=0,
            ns_state=ns_state,
            step_fn=s["step_fn"],
            n_mcmc_steps=3,
            n_extra=0,
            termination_criteria=[IterationTermination(n_iter)],
            callbacks=[],
            n_moves=1,
            move_descriptors=None,
            rng_key=s["key"],
            info_interval=10,
        )
        return final_state, call_count[0], max_neighbors_seen


class TestOverflowRetry(_OverflowRunner):
    """Check the overflow retry of _run_loop with an injected overflow."""

    def test_overflow_retry_does_not_advance_iteration(self):
        """An overflow on call 3 costs one extra call, not one iteration.

        The loop retries the same iteration with a larger bucket, so six
        iterations take seven calls and ``iteration`` still ends at 6.
        """
        n_iter = 6
        final_state, n_calls, _ = self.run_with_overflow(
            seed=55,
            n_walkers=15,
            n_iter=n_iter,
            overflow_call=3,
            fake_count=5,
        )
        assert n_calls == n_iter + 1
        assert int(final_state.iteration) == n_iter

    def test_overflow_retry_increases_max_neighbors(self):
        """The retry after an overflow uses a larger max_neighbors."""
        _, _, max_neighbors_seen = self.run_with_overflow(
            seed=66,
            n_walkers=10,
            n_iter=5,
            overflow_call=2,
            fake_count=10,
        )
        # Call 2 overflows; call 3 is its retry with a larger bucket.
        assert len(max_neighbors_seen) >= 3
        assert max_neighbors_seen[2] > max_neighbors_seen[1]
