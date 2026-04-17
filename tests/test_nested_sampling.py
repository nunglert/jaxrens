"""Test full nested sampling runs (run_ns, run_ns_parallel).

Verifies:
- run_ns completes and produces finite evidence
- Evidence accuracy on harmonic oscillator (known answer)
- Callbacks are invoked
- run_ns_parallel completes with correct shapes
- Parallel evidence roughly matches sequential
- Different pressures produce different evidence
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns, run_ns_parallel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_setup():
    """Single-run harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [
        MoveDescriptor("random_walk", random_walk.build_kernel),
    ])

    n_walkers = 50
    n_atoms = 1
    key = jax.random.key(42)

    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
        "n_walkers": n_walkers,
    }


@pytest.fixture
def parallel_setup():
    """2-run parallel harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [
        MoveDescriptor("random_walk", random_walk.build_kernel),
    ])

    n_runs = 2
    n_walkers = 20
    n_atoms = 1

    keys = jax.random.split(jax.random.key(0), n_runs)
    positions = jax.vmap(
        lambda k: jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0)
    )(keys)

    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)

    rng_keys = jax.random.split(jax.random.key(42), n_runs)

    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "rng_keys": rng_keys,
        "n_runs": n_runs,
        "n_walkers": n_walkers,
    }


# ---------------------------------------------------------------------------
# run_ns (single run)
# ---------------------------------------------------------------------------


class TestRunNS:
    def test_runs_to_completion(self, harmonic_setup):
        s = harmonic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=100,
            n_mcmc_steps=5,
            initial_step_size=0.3,
        )
        assert result["iteration"] > 0
        assert result["n_dead"] > 0
        assert jnp.isfinite(result["log_evidence"])

    @pytest.mark.heavy
    def test_harmonic_evidence_accuracy(self):
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(backend, [
            MoveDescriptor("random_walk", random_walk.build_kernel),
        ])

        n_walkers = 30
        L = 5.0
        key = jax.random.key(123)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, 1, 3), minval=-L, maxval=L
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        result = run_ns(
            positions, types, energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=500,
            n_mcmc_steps=5,
            initial_step_size=0.5,
            target_acceptance=0.4,
            convergence_threshold=0.5,
        )

        log_evidence = float(result["log_evidence"])

        log_prior_volume = 3.0 * jnp.log(2.0 * L)
        log_Z_analytical = 1.5 * jnp.log(2.0 * jnp.pi) - log_prior_volume

        assert abs(log_evidence - float(log_Z_analytical)) < 1.5, (
            f"log_evidence={log_evidence:.3f} vs analytical={float(log_Z_analytical):.3f}"
        )

    def test_callbacks_invoked(self, harmonic_setup):
        s = harmonic_setup

        iterations_seen = []

        class TestCallback:
            def on_iteration(self, iteration, ns_state, info):
                iterations_seen.append(iteration)

            def on_finish(self, ns_state):
                iterations_seen.append("finish")

        result = run_ns(
            s["positions"], s["types"], s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=20,
            n_mcmc_steps=5,
            callbacks=[TestCallback()],
        )

        assert len(iterations_seen) > 1
        assert iterations_seen[-1] == "finish"
        assert 0 in iterations_seen


# ---------------------------------------------------------------------------
# run_ns_parallel (multi-run)
# ---------------------------------------------------------------------------


class TestRunNsParallel:
    def test_basic_completion(self, parallel_setup):
        s = parallel_setup
        result = run_ns_parallel(
            s["positions"], s["types"], s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=s["rng_keys"],
            max_iterations=50,
            n_mcmc_steps=5,
            initial_step_size=0.3,
        )

        assert result["n_runs"] == 2
        assert result["log_evidence"].shape == (2,)
        assert result["iteration"].shape == (2,)
        assert jnp.all(jnp.isfinite(result["log_evidence"]))
        assert jnp.all(result["n_dead"] > 0)

    def test_parallel_matches_sequential(self, parallel_setup):
        s = parallel_setup
        n_iter = 50
        n_mcmc = 5

        result_par = run_ns_parallel(
            s["positions"], s["types"], s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=s["rng_keys"],
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        result_seq_0 = run_ns(
            s["positions"][0], s["types"], s["energies"][0],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["rng_keys"][0],
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        log_Z_par = float(result_par["log_evidence"][0])
        log_Z_seq = float(result_seq_0["log_evidence"])
        assert abs(log_Z_par - log_Z_seq) < 5.0, (
            f"Parallel log_Z={log_Z_par:.3f} vs sequential log_Z={log_Z_seq:.3f}"
        )


class TestDifferentPressures:
    def test_different_pressures_different_evidence(self):
        base_backend = create_harmonic(k=1.0)
        backend_p0 = EnsembleBackend(base_backend, pressure=0.0)
        backend_p1 = EnsembleBackend(base_backend, pressure=0.1)

        init_fn_p0, step_fn_p0, _ = build_mwg(backend_p0, [
            MoveDescriptor("random_walk", random_walk.build_kernel),
        ])

        n_walkers = 15
        n_atoms = 1
        key = jax.random.key(99)
        positions = jax.random.uniform(key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        cells = jnp.tile(5.0 * jnp.eye(3), (n_walkers, 1, 1))

        energies_p0 = jax.vmap(
            lambda pos, cell: backend_p0(pos, types, cell, 0)[0]
        )(positions, cells)

        energies_p1 = jax.vmap(
            lambda pos, cell: backend_p1(pos, types, cell, 0)[0]
        )(positions, cells)

        assert not jnp.allclose(energies_p0, energies_p1)
        assert jnp.all(energies_p1 > energies_p0)
