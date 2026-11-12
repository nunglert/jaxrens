"""Integration tests for the multi-GPU NS path, run with n_gpu=1.

Covers the PmapVmapRuns branch of build_adapt_step, init_ns_multi_gpu and
run_ns_multi_gpu.  The PmapVmapRuns descriptor itself is tested in
test_batch_descriptor.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import PmapVmapRuns
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns_multi_gpu,
    init_ns_parallel,
    run_ns_multi_gpu,
    run_ns_parallel,
)


def _make_harmonic_problem(
    seed: int = 42,
    n_walkers: int = 20,
    n_atoms: int = 1,
):
    """Return a small harmonic NS problem (functions and arrays)."""
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw",
            random_walk.build_kernel,
            step_size=0.2,
            step_size_max=5.0,
            min_rate=0.2,
            max_rate=0.7,
        ),
    ]
    init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)
    key = jax.random.key(seed)
    key, pos_key = jax.random.split(key)
    positions = jax.random.uniform(
        pos_key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    return {
        "backend": backend,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "per_move_fns": per_move_fns,
        "descriptors": descriptors,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
        "n_walkers": n_walkers,
    }


@pytest.fixture(scope="module")
def pmap_harmonic_setup():
    """Harmonic setup with a (G, P, K, ...) population."""
    n_gpu = 1
    n_per_gpu = 2
    n_walkers = 20
    setup = _make_harmonic_problem(seed=99, n_walkers=n_walkers)
    positions = setup["positions"]
    energies = setup["energies"]
    types = setup["types"]
    init_fn = setup["init_fn"]
    step_fn = setup["step_fn"]
    per_move_fns = setup["per_move_fns"]
    descriptors = setup["descriptors"]

    n_total = n_gpu * n_per_gpu
    rng_keys_flat = jax.random.split(jax.random.key(42), n_total)
    pos_batch = jnp.broadcast_to(positions[None], (n_total, n_walkers, 1, 3))
    en_batch = jnp.broadcast_to(energies[None], (n_total, n_walkers))

    flat_states = init_ns_parallel(
        init_fn,
        pos_batch,
        types,
        en_batch,
        cells=None,
        rng_keys=rng_keys_flat,
    )

    def _reshape(x):
        arr = jnp.asarray(x)
        return arr.reshape((n_gpu, n_per_gpu) + arr.shape[1:])

    batched_states = jax.tree.map(_reshape, flat_states)
    pop = batched_states.population

    return {
        "pop": pop,
        "per_move_fns": per_move_fns,
        "descriptors": descriptors,
        "n_gpu": n_gpu,
        "n_per_gpu": n_per_gpu,
        "n_walkers": n_walkers,
        "step_fn": step_fn,
        "batched_states": batched_states,
    }


class TestBuildAdaptStepPmapVmap:
    """Check the adaptation closure of build_adapt_step with PmapVmapRuns."""

    def _build(self, setup):
        return build_adapt_step(
            move_descriptors=setup["descriptors"],
            per_move_fns=setup["per_move_fns"],
            batcher=PmapVmapRuns(
                n_gpu=setup["n_gpu"],
                n_per_gpu=setup["n_per_gpu"],
            ),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )

    def test_closure_outputs(self, pmap_harmonic_setup):
        """One call gives (G, P) keys, (G, P, n_moves) diagnostics and
        positive finite step sizes of the input shape."""
        setup = pmap_harmonic_setup
        adapt = self._build(setup)
        assert adapt is not None
        ns_states = setup["batched_states"]
        n_gpu, n_per_gpu = setup["n_gpu"], setup["n_per_gpu"]
        n_moves = 1

        emax = jnp.max(ns_states.population.energy, axis=2)
        run_keys = jax.random.split(
            jax.random.key(10),
            n_gpu * n_per_gpu,
        ).reshape(n_gpu, n_per_gpu)

        new_state, diag, new_key = adapt(ns_states, emax, run_keys)

        assert new_key.shape == (n_gpu, n_per_gpu)
        ss = new_state.population.step_sizes
        assert ss.shape == ns_states.population.step_sizes.shape
        assert jnp.all(ss > 0.0)
        assert jnp.all(jnp.isfinite(ss))
        assert diag["rate"].shape == (n_gpu, n_per_gpu, n_moves)
        assert diag["counts"].shape == (n_gpu, n_per_gpu, n_moves, 4)
        assert diag["n_rounds"].shape == (n_gpu, n_per_gpu, n_moves)
        assert diag["converged"].shape == (n_gpu, n_per_gpu, n_moves)


class TestInitNsMultiGpu:
    """Check the state shapes of init_ns_multi_gpu."""

    def test_state_shapes(self):
        """NSState fields have a leading (G, P) shape."""
        n_gpu, n_per_gpu = 1, 2
        n_walkers = 10
        setup = _make_harmonic_problem(seed=7, n_walkers=n_walkers)

        n_total = n_gpu * n_per_gpu
        rng_keys_flat = jax.random.split(jax.random.key(5), n_total)
        pos_flat = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_flat = jnp.broadcast_to(
            setup["energies"][None], (n_total, n_walkers)
        )

        state = init_ns_multi_gpu(
            setup["init_fn"],
            pos_flat,
            setup["types"],
            en_flat,
            cells=None,
            rng_keys=rng_keys_flat,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
        )

        assert state.log_evidence.shape == (n_gpu, n_per_gpu)
        assert state.iteration.shape == (n_gpu, n_per_gpu)
        assert state.population.energy.shape[:2] == (n_gpu, n_per_gpu)


class TestRunNsMultiGpu:
    """Check run_ns_multi_gpu end to end with n_gpu=1."""

    def _run_short(self, seed=42, n_per_gpu=2, n_iters=5, n_walkers=15):
        n_gpu = 1
        n_total = n_gpu * n_per_gpu
        setup = _make_harmonic_problem(seed=seed, n_walkers=n_walkers)
        rng_keys = jax.random.split(jax.random.key(seed + 1), n_total)

        pos_batch = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_batch = jnp.broadcast_to(
            setup["energies"][None], (n_total, n_walkers)
        )

        result = run_ns_multi_gpu(
            pos_batch,
            setup["types"],
            en_batch,
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_keys=rng_keys,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
            max_iterations=n_iters,
            n_mcmc_steps=3,
            initial_step_size=0.2,
            convergence_threshold=1e6,
            restart_states=None,
        )
        return result

    def test_short_run(self):
        """A fresh run gives finite evidence and (1, P) shaped outputs."""
        n_gpu, n_per_gpu, n_iters = 1, 2, 5
        result = self._run_short(seed=42, n_per_gpu=n_per_gpu, n_iters=n_iters)

        assert jnp.all(jnp.isfinite(result["log_evidence"]))
        assert result["log_evidence"].shape == (n_gpu, n_per_gpu)
        assert result["n_dead"].shape == (n_gpu, n_per_gpu)
        assert result["iteration"].shape == (n_gpu, n_per_gpu)
        assert result["n_gpu"] == n_gpu
        assert result["n_per_gpu"] == n_per_gpu
        assert jnp.all(result["n_dead"] == n_iters)


class TestRunNsMultiGpuValidation:
    """Check that run_ns_multi_gpu rejects invalid n_gpu and n_per_gpu."""

    def test_n_gpu_zero(self):
        with pytest.raises(ValueError, match="n_gpu must be >= 1"):
            run_ns_multi_gpu(
                jnp.zeros((2, 10, 1, 3)),
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.zeros((2, 10)),
                None,
                init_fn=lambda *a, **kw: None,
                step_fn=lambda *a, **kw: None,
                rng_keys=jax.random.split(jax.random.key(0), 2),
                n_gpu=0,
                n_per_gpu=2,
            )

    def test_n_per_gpu_zero(self):
        with pytest.raises(ValueError, match="n_per_gpu must be >= 1"):
            run_ns_multi_gpu(
                jnp.zeros((1, 10, 1, 3)),
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.zeros((1, 10)),
                None,
                init_fn=lambda *a, **kw: None,
                step_fn=lambda *a, **kw: None,
                rng_keys=jax.random.split(jax.random.key(0), 1),
                n_gpu=1,
                n_per_gpu=0,
            )

    def test_n_gpu_exceeds_devices(self):
        n_available = len(jax.devices())
        with pytest.raises(ValueError, match="exceeds available devices"):
            run_ns_multi_gpu(
                jnp.zeros((n_available + 1, 10, 1, 3)),
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.zeros((n_available + 1, 10)),
                None,
                init_fn=lambda *a, **kw: None,
                step_fn=lambda *a, **kw: None,
                rng_keys=jax.random.split(jax.random.key(0), n_available + 1),
                n_gpu=n_available + 1,
                n_per_gpu=1,
            )


class TestParityPmapVmapVsVmap:
    """Check PmapVmapRuns(n_gpu=1, P) against VmapRuns(n_runs=P)."""

    def test_parity_within_tolerance(self):
        """Both variants give the same evidence and dead-point counts."""
        n_per_gpu = 2
        n_walkers = 15
        seed = 88
        n_iters = 20
        n_gpu = 1

        setup = _make_harmonic_problem(seed=seed, n_walkers=n_walkers)
        n_total = n_gpu * n_per_gpu

        rng_keys_flat = jax.random.split(jax.random.key(seed), n_total)
        pos_batch = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_batch = jnp.broadcast_to(
            setup["energies"][None], (n_total, n_walkers)
        )

        result_pmap = run_ns_multi_gpu(
            pos_batch,
            setup["types"],
            en_batch,
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_keys=rng_keys_flat,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
            max_iterations=n_iters,
            n_mcmc_steps=3,
            initial_step_size=0.2,
            convergence_threshold=1e6,
        )

        result_vmap = run_ns_parallel(
            pos_batch,
            setup["types"],
            en_batch,
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_keys=rng_keys_flat,
            max_iterations=n_iters,
            n_mcmc_steps=3,
            initial_step_size=0.2,
            convergence_threshold=1e6,
        )

        log_z_pmap = np.asarray(result_pmap["log_evidence"]).flatten()
        log_z_vmap = np.asarray(result_vmap["log_evidence"])

        assert jnp.all(jnp.isfinite(jnp.asarray(log_z_pmap)))
        assert jnp.all(jnp.isfinite(jnp.asarray(log_z_vmap)))

        # The measured gap is below 3e-7.
        np.testing.assert_allclose(log_z_pmap, log_z_vmap, atol=1e-4)
        np.testing.assert_array_equal(
            np.asarray(result_pmap["n_dead"]).flatten(),
            np.asarray(result_vmap["n_dead"]),
        )


class MockCallback:
    """Callback that records every on_iteration call."""

    def __init__(self):
        self.calls: list[tuple[int, object, dict]] = []

    def on_iteration(
        self, iteration: int, ns_state: object, info: dict
    ) -> None:
        self.calls.append((iteration, ns_state, info))


class TestRunNsMultiGpuCallbacks:
    """Check the callbacks that run_ns_multi_gpu calls."""

    def test_callbacks_are_called_with_a_batched_descriptor(self):
        """on_iteration runs once per iteration with a batched '_batcher'."""
        n_gpu, n_per_gpu = 1, 2
        n_total = n_gpu * n_per_gpu
        n_iters = 3
        n_walkers = 12
        setup = _make_harmonic_problem(seed=77, n_walkers=n_walkers)

        rng_keys = jax.random.split(jax.random.key(77), n_total)
        pos_batch = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_batch = jnp.broadcast_to(
            setup["energies"][None], (n_total, n_walkers)
        )

        cb = MockCallback()
        run_ns_multi_gpu(
            pos_batch,
            setup["types"],
            en_batch,
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_keys=rng_keys,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
            max_iterations=n_iters,
            n_mcmc_steps=3,
            initial_step_size=0.2,
            convergence_threshold=1e6,
            callbacks=[cb],
        )

        assert len(cb.calls) == n_iters
        for _iter, _state, info in cb.calls:
            assert info["_batcher"].is_batched is True
