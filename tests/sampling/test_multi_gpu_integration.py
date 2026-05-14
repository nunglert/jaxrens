"""Integration tests for the multi-GPU NS branch: build_adapt_step pmap
path, init_ns_multi_gpu, and run_ns_multi_gpu.

All tests use n_gpu=1 (single-device constraint); the implementation itself
does not hard-code this. PmapVmapRuns descriptor unit tests live in
test_batch_descriptor.py.

Coverage:
- build_adapt_step with PmapVmapRuns: closure shapes
- init_ns_multi_gpu: state shapes
- run_ns_multi_gpu: smoke test, finite log_evidence, shapes (1, P)
- Parity: PmapVmapRuns(n_gpu=1, n_per_gpu=P) vs VmapRuns(n_runs=P)
- Restart fresh-start (restart_states=None)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, VmapRuns
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns,
    init_ns_multi_gpu,
    init_ns_parallel,
    run_ns_multi_gpu,
    run_ns_parallel,
)
from jaxrens.sampling.termination import IterationTermination, PriorMassTermination


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------


def _require_gpu():
    """Skip test if no devices available (defensive; should never trigger)."""
    if len(jax.devices()) < 1:
        pytest.skip("No JAX devices available")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_harmonic_problem(
    seed: int = 42,
    n_walkers: int = 20,
    n_atoms: int = 1,
):
    """Return minimal harmonic NS problem arrays + fns."""
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw", random_walk.build_kernel,
            step_size=0.2, step_size_max=5.0,
            min_rate=0.2, max_rate=0.7,
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


# ---------------------------------------------------------------------------
# build_adapt_step with PmapVmapRuns
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pmap_harmonic_setup():
    """Build harmonic setup with (G, P, K, ...) population."""
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

    # Build (G*P, K, ...) parallel state then reshape to (G, P, K, ...).
    n_total = n_gpu * n_per_gpu
    rng_keys_flat = jax.random.split(jax.random.key(42), n_total)
    pos_batch = jnp.broadcast_to(positions[None], (n_total, n_walkers, 1, 3))
    en_batch = jnp.broadcast_to(energies[None], (n_total, n_walkers))

    flat_states = init_ns_parallel(
        init_fn, pos_batch, types, en_batch,
        cells=None, rng_keys=rng_keys_flat,
    )

    # Reshape (G*P, K, ...) -> (G, P, K, ...).
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
    """``build_adapt_step`` closure with PmapVmapRuns."""

    def _build(self, setup):
        return build_adapt_step(
            move_descriptors=setup["descriptors"],
            per_move_fns=setup["per_move_fns"],
            batcher=PmapVmapRuns(
                n_gpu=setup["n_gpu"], n_per_gpu=setup["n_per_gpu"],
            ),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=50,
        )

    def test_closure_no_error(self, pmap_harmonic_setup):
        """closure runs without error for PmapVmapRuns."""
        _require_gpu()
        setup = pmap_harmonic_setup
        adapt = self._build(setup)
        assert adapt is not None
        ns_states = setup["batched_states"]
        n_gpu, n_per_gpu = setup["n_gpu"], setup["n_per_gpu"]

        emax = jnp.max(ns_states.population.energy, axis=2)  # (G, P)
        run_keys = jax.random.split(
            jax.random.key(10), n_gpu * n_per_gpu,
        ).reshape(n_gpu, n_per_gpu)

        new_state, diag, new_key = adapt(ns_states, emax, run_keys)

        assert new_key.shape == (n_gpu, n_per_gpu)
        # step_sizes pulled out via the same axis as the input.
        assert new_state.population.step_sizes.shape \
            == ns_states.population.step_sizes.shape

    def test_diag_output_shapes(self, pmap_harmonic_setup):
        """Diagnostics have shape (G, P, n_moves) for PmapVmapRuns."""
        _require_gpu()
        setup = pmap_harmonic_setup
        adapt = self._build(setup)
        ns_states = setup["batched_states"]
        n_gpu, n_per_gpu = setup["n_gpu"], setup["n_per_gpu"]
        n_moves = 1

        emax = jnp.max(ns_states.population.energy, axis=2)
        run_keys = jax.random.split(
            jax.random.key(11), n_gpu * n_per_gpu,
        ).reshape(n_gpu, n_per_gpu)

        _, diag, _ = adapt(ns_states, emax, run_keys)

        assert diag["rate"].shape == (n_gpu, n_per_gpu, n_moves)
        assert diag["counts"].shape == (n_gpu, n_per_gpu, n_moves, 4)
        assert diag["n_rounds"].shape == (n_gpu, n_per_gpu, n_moves)
        assert diag["converged"].shape == (n_gpu, n_per_gpu, n_moves)

    def test_step_sizes_positive_finite(self, pmap_harmonic_setup):
        """Returned step sizes must be positive and finite."""
        _require_gpu()
        setup = pmap_harmonic_setup
        adapt = self._build(setup)
        ns_states = setup["batched_states"]
        n_gpu, n_per_gpu = setup["n_gpu"], setup["n_per_gpu"]

        emax = jnp.max(ns_states.population.energy, axis=2)
        run_keys = jax.random.split(
            jax.random.key(12), n_gpu * n_per_gpu,
        ).reshape(n_gpu, n_per_gpu)

        new_state, _, _ = adapt(ns_states, emax, run_keys)
        ss = new_state.population.step_sizes
        assert jnp.all(ss > 0.0)
        assert jnp.all(jnp.isfinite(ss))

# ---------------------------------------------------------------------------
# init_ns_multi_gpu
# ---------------------------------------------------------------------------


class TestInitNsMultiGpu:
    """Tests for the init_ns_multi_gpu helper."""

    def test_state_shapes(self):
        """NSState fields have (G, P, ...) leading shape after init."""
        _require_gpu()
        n_gpu, n_per_gpu = 1, 2
        n_walkers = 10
        setup = _make_harmonic_problem(seed=7, n_walkers=n_walkers)

        n_total = n_gpu * n_per_gpu
        rng_keys_flat = jax.random.split(jax.random.key(5), n_total)
        pos_flat = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_flat = jnp.broadcast_to(setup["energies"][None], (n_total, n_walkers))

        state = init_ns_multi_gpu(
            setup["init_fn"], pos_flat, setup["types"], en_flat,
            cells=None, rng_keys=rng_keys_flat,
            n_gpu=n_gpu, n_per_gpu=n_per_gpu,
        )

        # Check leading shape
        assert state.log_evidence.shape == (n_gpu, n_per_gpu)
        assert state.iteration.shape == (n_gpu, n_per_gpu)
        assert state.population.energy.shape[:2] == (n_gpu, n_per_gpu)


# ---------------------------------------------------------------------------
# run_ns_multi_gpu smoke tests
# ---------------------------------------------------------------------------


class TestRunNsMultiGpu:
    """End-to-end smoke tests for run_ns_multi_gpu."""

    def _run_short(self, seed=42, n_per_gpu=2, n_iters=5, n_walkers=15):
        _require_gpu()
        n_gpu = 1
        n_total = n_gpu * n_per_gpu
        setup = _make_harmonic_problem(seed=seed, n_walkers=n_walkers)
        rng_keys = jax.random.split(jax.random.key(seed + 1), n_total)

        pos_batch = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_batch = jnp.broadcast_to(setup["energies"][None], (n_total, n_walkers))

        result = run_ns_multi_gpu(
            pos_batch, setup["types"], en_batch,
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
        )
        return result

    def test_smoke_finite_log_evidence(self):
        """run_ns_multi_gpu produces finite log_evidence."""
        result = self._run_short(seed=42, n_per_gpu=2, n_iters=5)
        assert jnp.all(jnp.isfinite(result["log_evidence"])), (
            f"Non-finite log_evidence: {result['log_evidence']}"
        )

    def test_output_shapes(self):
        """Output arrays have (1, P) leading shape."""
        n_gpu, n_per_gpu = 1, 2
        result = self._run_short(seed=10, n_per_gpu=n_per_gpu, n_iters=5)
        assert result["log_evidence"].shape == (n_gpu, n_per_gpu), (
            f"Expected ({n_gpu}, {n_per_gpu}), got {result['log_evidence'].shape}"
        )
        assert result["n_dead"].shape == (n_gpu, n_per_gpu)
        assert result["iteration"].shape == (n_gpu, n_per_gpu)
        assert result["n_gpu"] == n_gpu
        assert result["n_per_gpu"] == n_per_gpu

    def test_no_nans_in_log_evidence(self):
        """No NaNs in log_evidence after run."""
        result = self._run_short(seed=7, n_per_gpu=2, n_iters=5)
        assert not jnp.any(jnp.isnan(result["log_evidence"]))

    def test_n_dead_positive(self):
        """Each run must have > 0 dead points after 5 iterations."""
        result = self._run_short(seed=13, n_per_gpu=2, n_iters=5)
        assert jnp.all(result["n_dead"] > 0)

    def test_restart_fresh_start(self):
        """run_ns_multi_gpu with restart_states=None runs without error."""
        _require_gpu()
        n_gpu, n_per_gpu = 1, 2
        n_total = n_gpu * n_per_gpu
        setup = _make_harmonic_problem(seed=55, n_walkers=12)
        rng_keys = jax.random.split(jax.random.key(55), n_total)
        pos_batch = jnp.broadcast_to(setup["positions"][None], (n_total, 12, 1, 3))
        en_batch = jnp.broadcast_to(setup["energies"][None], (n_total, 12))

        result = run_ns_multi_gpu(
            pos_batch, setup["types"], en_batch,
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_keys=rng_keys,
            n_gpu=n_gpu, n_per_gpu=n_per_gpu,
            max_iterations=5,
            n_mcmc_steps=3,
            initial_step_size=0.2,
            convergence_threshold=1e6,
            restart_states=None,
        )
        assert jnp.all(jnp.isfinite(result["log_evidence"]))


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestRunNsMultiGpuValidation:
    """run_ns_multi_gpu raises ValueError on bad args."""

    def test_n_gpu_zero(self):
        _require_gpu()
        with pytest.raises(ValueError, match="n_gpu must be >= 1"):
            run_ns_multi_gpu(
                jnp.zeros((2, 10, 1, 3)), jnp.zeros((1,), dtype=jnp.int32),
                jnp.zeros((2, 10)), None,
                init_fn=lambda *a, **kw: None, step_fn=lambda *a, **kw: None,
                rng_keys=jax.random.split(jax.random.key(0), 2),
                n_gpu=0, n_per_gpu=2,
            )

    def test_n_per_gpu_zero(self):
        _require_gpu()
        with pytest.raises(ValueError, match="n_per_gpu must be >= 1"):
            run_ns_multi_gpu(
                jnp.zeros((1, 10, 1, 3)), jnp.zeros((1,), dtype=jnp.int32),
                jnp.zeros((1, 10)), None,
                init_fn=lambda *a, **kw: None, step_fn=lambda *a, **kw: None,
                rng_keys=jax.random.split(jax.random.key(0), 1),
                n_gpu=1, n_per_gpu=0,
            )

    def test_n_gpu_exceeds_devices(self):
        _require_gpu()
        n_available = len(jax.devices())
        with pytest.raises(ValueError, match="exceeds available devices"):
            run_ns_multi_gpu(
                jnp.zeros((n_available + 1, 10, 1, 3)),
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.zeros((n_available + 1, 10)), None,
                init_fn=lambda *a, **kw: None, step_fn=lambda *a, **kw: None,
                rng_keys=jax.random.split(jax.random.key(0), n_available + 1),
                n_gpu=n_available + 1, n_per_gpu=1,
            )


# ---------------------------------------------------------------------------
# Parity: PmapVmapRuns(n_gpu=1, n_per_gpu=P) vs VmapRuns(n_runs=P)
# ---------------------------------------------------------------------------


class TestParityPmapVmapVsVmap:
    """PmapVmapRuns(n_gpu=1, P) and VmapRuns(n_runs=P) should produce
    log_evidence within 5 log-units on the same problem (physicist-generous
    tolerance matching commit 4's test_run_loop_equivalence)."""

    def test_parity_within_tolerance(self):
        """Both variants run the same toy problem; results are close."""
        _require_gpu()
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
        en_batch = jnp.broadcast_to(setup["energies"][None], (n_total, n_walkers))

        # PmapVmapRuns(n_gpu=1, n_per_gpu=2) run
        result_pmap = run_ns_multi_gpu(
            pos_batch, setup["types"], en_batch,
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_keys=rng_keys_flat,
            n_gpu=n_gpu, n_per_gpu=n_per_gpu,
            max_iterations=n_iters,
            n_mcmc_steps=3,
            initial_step_size=0.2,
            convergence_threshold=1e6,
        )

        # VmapRuns(n_runs=2) run with same keys
        result_vmap = run_ns_parallel(
            pos_batch, setup["types"], en_batch,
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_keys=rng_keys_flat,
            max_iterations=n_iters,
            n_mcmc_steps=3,
            initial_step_size=0.2,
            convergence_threshold=1e6,
        )

        # Flatten (1, 2) -> (2,) for comparison
        log_z_pmap = np.asarray(result_pmap["log_evidence"]).flatten()
        log_z_vmap = np.asarray(result_vmap["log_evidence"])

        assert jnp.all(jnp.isfinite(jnp.asarray(log_z_pmap)))
        assert jnp.all(jnp.isfinite(jnp.asarray(log_z_vmap)))

        # Both runs must be within 5 log-units of each other on average
        # (physicist-generous; exact agreement not enforced because
        # pmap execution order may differ from vmap).
        for i in range(n_per_gpu):
            diff = abs(float(log_z_pmap[i]) - float(log_z_vmap[i]))
            assert diff < 5.0, (
                f"Run {i}: pmap log_Z={log_z_pmap[i]:.3f} "
                f"vs vmap log_Z={log_z_vmap[i]:.3f} (diff={diff:.3f} > 5.0)"
            )


# ---------------------------------------------------------------------------
# Task A: callbacks wiring in run_ns_multi_gpu
# ---------------------------------------------------------------------------


class MockCallback:
    """Minimal callback that records on_iteration calls for testing."""

    def __init__(self):
        self.calls: list[tuple[int, object, dict]] = []

    def on_iteration(self, iteration: int, ns_state: object, info: dict) -> None:
        self.calls.append((iteration, ns_state, info))


class TestRunNsMultiGpuCallbacks:
    """Verify callbacks are wired through run_ns_multi_gpu."""

    def test_callbacks_called_correct_count(self):
        """MockCallback.on_iteration is called once per iteration (n_iters times)."""
        _require_gpu()
        n_gpu, n_per_gpu = 1, 2
        n_iters = 3
        n_walkers = 12
        n_total = n_gpu * n_per_gpu
        setup = _make_harmonic_problem(seed=77, n_walkers=n_walkers)

        rng_keys = jax.random.split(jax.random.key(77), n_total)
        pos_batch = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_batch = jnp.broadcast_to(setup["energies"][None], (n_total, n_walkers))

        cb = MockCallback()
        run_ns_multi_gpu(
            pos_batch, setup["types"], en_batch,
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

        assert len(cb.calls) == n_iters, (
            f"Expected {n_iters} callback calls, got {len(cb.calls)}"
        )

    def test_callbacks_info_has_batch_key(self):
        """Each on_iteration info dict carries '_batcher' key."""
        _require_gpu()
        n_gpu, n_per_gpu = 1, 2
        n_total = n_gpu * n_per_gpu
        n_iters = 3
        n_walkers = 12
        setup = _make_harmonic_problem(seed=78, n_walkers=n_walkers)

        rng_keys = jax.random.split(jax.random.key(78), n_total)
        pos_batch = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_batch = jnp.broadcast_to(setup["energies"][None], (n_total, n_walkers))

        cb = MockCallback()
        run_ns_multi_gpu(
            pos_batch, setup["types"], en_batch,
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

        for _iter, _state, info in cb.calls:
            assert "_batcher" in info, (
                f"'_batcher' missing from info at iteration {_iter}"
            )

    def test_callbacks_batch_descriptor_is_batched(self):
        """info['_batcher'].is_batched is True for all calls."""
        _require_gpu()
        n_gpu, n_per_gpu = 1, 2
        n_total = n_gpu * n_per_gpu
        n_iters = 3
        n_walkers = 12
        setup = _make_harmonic_problem(seed=79, n_walkers=n_walkers)

        rng_keys = jax.random.split(jax.random.key(79), n_total)
        pos_batch = jnp.broadcast_to(
            setup["positions"][None], (n_total, n_walkers, 1, 3)
        )
        en_batch = jnp.broadcast_to(setup["energies"][None], (n_total, n_walkers))

        cb = MockCallback()
        run_ns_multi_gpu(
            pos_batch, setup["types"], en_batch,
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
            assert info["_batcher"].is_batched is True, (
                f"is_batched should be True at iter {_iter}, "
                f"got {info['_batcher'].is_batched}"
            )
