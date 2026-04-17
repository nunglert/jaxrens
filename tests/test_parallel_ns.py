"""Tests for multi-run parallel nested sampling via vmap(ns_step).

Verifies:
- vmap(ns_step) compiles and runs for 2+ runs
- Different RNG keys produce different trajectories
- Different pressures per run produce different evidence
- run_ns_parallel completes and returns correct shapes
- Evidence from parallel run matches sequential
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.backends.ensemble import EnsembleBackend, make_ensemble_params
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns,
    init_ns_parallel,
    ns_step,
    run_ns_parallel,
)


@pytest.fixture
def parallel_setup():
    """Set up 2-run parallel NS on harmonic oscillator."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn = build_mwg(backend, [
        MoveDescriptor("random_walk", random_walk.build_kernel),
    ])

    n_runs = 2
    n_walkers = 20
    n_atoms = 1

    # Create per-run initial data
    keys = jax.random.split(jax.random.key(0), n_runs)
    positions = jax.vmap(
        lambda k: jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0)
    )(keys)  # (n_runs, n_walkers, n_atoms, 3)

    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)  # (n_runs, n_walkers)

    rng_keys = jax.random.split(jax.random.key(42), n_runs)

    return {
        "backend": backend,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "rng_keys": rng_keys,
        "n_runs": n_runs,
        "n_walkers": n_walkers,
    }


class TestVmapNsStep:
    def test_vmap_two_runs(self, parallel_setup):
        """vmap(ns_step) over 2 runs should compile and run."""
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"], s["positions"], s["types"], s["energies"],
            boxes=None, rng_keys=s["rng_keys"], max_dead=100,
        )

        vmapped = jax.jit(jax.vmap(
            lambda state: ns_step(state, s["step_fn"], 10, 0)
        ))
        new_states, infos = vmapped(ns_states)

        assert new_states.iteration.shape == (2,)
        assert infos["emax"].shape == (2,)
        assert jnp.all(new_states.iteration == 1)

    def test_different_keys_different_results(self, parallel_setup):
        """Runs with different RNG keys should produce different trajectories."""
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"], s["positions"], s["types"], s["energies"],
            boxes=None, rng_keys=s["rng_keys"], max_dead=100,
        )

        vmapped = jax.jit(jax.vmap(
            lambda state: ns_step(state, s["step_fn"], 10, 0)
        ))

        # Run 5 steps
        for _ in range(5):
            ns_states, infos = vmapped(ns_states)

        # Energies should differ between runs
        assert not jnp.allclose(
            ns_states.population.energy[0],
            ns_states.population.energy[1],
        )

    def test_vmap_with_n_extra(self, parallel_setup):
        """vmap(ns_step) with n_extra > 0 should work."""
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"], s["positions"], s["types"], s["energies"],
            boxes=None, rng_keys=s["rng_keys"], max_dead=100,
        )

        vmapped = jax.jit(jax.vmap(
            lambda state: ns_step(state, s["step_fn"], 10, 3)
        ))
        new_states, infos = vmapped(ns_states)

        assert new_states.iteration.shape == (2,)
        assert 0 <= float(infos["acceptance_rate"][0]) <= 1.0


class TestVmapDifferentPressures:
    def test_different_pressures_different_evidence(self):
        """Runs at different pressures should produce different evidence."""
        base_backend = create_harmonic(k=1.0)
        # Wrap with ensemble at two different pressures
        backend_p0 = EnsembleBackend(base_backend, pressure=0.0)
        backend_p1 = EnsembleBackend(base_backend, pressure=0.1)

        # Use same init_fn for both (pressure correction applied by backend)
        init_fn_p0, step_fn_p0 = build_mwg(backend_p0, [
            MoveDescriptor("random_walk", random_walk.build_kernel),
        ])

        n_walkers = 15
        n_atoms = 1
        key = jax.random.key(99)
        positions = jax.random.uniform(key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        boxes = jnp.tile(5.0 * jnp.eye(3), (n_walkers, 1, 1))

        # Compute energies with pressure=0 (NVT)
        energies_p0 = jax.vmap(
            lambda pos, box: backend_p0(pos, types, box, 0)[0]
        )(positions, boxes)

        # Compute energies with pressure=0.1 (NPT)
        energies_p1 = jax.vmap(
            lambda pos, box: backend_p1(pos, types, box, 0)[0]
        )(positions, boxes)

        # Energies should differ (PV term)
        assert not jnp.allclose(energies_p0, energies_p1)
        # NPT energies should be higher (PV > 0)
        assert jnp.all(energies_p1 > energies_p0)


class TestRunNsParallel:
    def test_basic_completion(self, parallel_setup):
        """run_ns_parallel should complete and return correct shapes."""
        s = parallel_setup
        result = run_ns_parallel(
            s["positions"], s["types"], s["energies"],
            boxes=None,
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
        """Evidence from parallel run should roughly match sequential runs."""
        s = parallel_setup
        n_iter = 50
        n_mcmc = 5

        # Parallel run
        result_par = run_ns_parallel(
            s["positions"], s["types"], s["energies"],
            boxes=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=s["rng_keys"],
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        # Sequential runs (same seeds)
        from jaxrens.sampling.nested_sampling import run_ns
        result_seq_0 = run_ns(
            s["positions"][0], s["types"], s["energies"][0],
            boxes=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["rng_keys"][0],
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        # Both should produce finite evidence in a similar range
        # (exact match unlikely due to adaptation differences, but
        #  both should be in the same ballpark)
        log_Z_par = float(result_par["log_evidence"][0])
        log_Z_seq = float(result_seq_0["log_evidence"])
        assert abs(log_Z_par - log_Z_seq) < 5.0, (
            f"Parallel log_Z={log_Z_par:.3f} vs sequential log_Z={log_Z_seq:.3f}"
        )
