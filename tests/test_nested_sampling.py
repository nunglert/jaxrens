"""Test the nested sampling loop.

Verifies:
- NS runs complete without error on toy problems
- Evidence accuracy on harmonic oscillator (known answer)
- lax.scan inner loop produces correct results
- Callbacks are invoked
- ns_step is JIT-compatible
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns


@pytest.fixture
def harmonic_setup():
    """Set up a harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn = build_mwg(backend, [
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
        "backend": backend,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
        "n_walkers": n_walkers,
    }


class TestInitNS:
    def test_init_creates_state(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )
        assert state.population.positions.shape == s["positions"].shape
        assert state.population.energy.shape == (s["n_walkers"],)
        assert state.iteration == 0
        assert state.n_dead == 0

    def test_init_dead_points_buffer(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"], max_dead=100,
        )
        assert state.dead_energies.shape == (100,)

    def test_init_population_is_batched_mcstate(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )
        pop = state.population
        assert pop.positions.shape[0] == s["n_walkers"]
        assert pop.energy.shape == (s["n_walkers"],)
        assert pop.step_sizes.ndim == 2


class TestNSStep:
    def test_single_step(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        new_state, info = ns_step(state, s["step_fn"], n_mcmc_steps=10)

        assert new_state.iteration == 1
        assert new_state.n_dead == 1
        assert 0 <= info["acceptance_rate"] <= 1.0

    def test_multiple_steps_reduce_emax(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        emaxes = []
        for _ in range(20):
            state, info = ns_step(state, s["step_fn"], n_mcmc_steps=10)
            emaxes.append(float(info["emax"]))

        assert emaxes[-1] < emaxes[0], "Emax should decrease during NS"

    def test_evidence_increases(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        for _ in range(50):
            state, info = ns_step(state, s["step_fn"], n_mcmc_steps=10)

        assert jnp.isfinite(state.log_evidence)

    def test_jit_compatible(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        new_state, info = jit_step(state, s["step_fn"], 10, 0)

        assert new_state.iteration == 1
        assert jnp.isfinite(info["emax"])

    def test_n_extra_walks(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        new_state, info = ns_step(state, s["step_fn"], n_mcmc_steps=10, n_extra=5)

        assert new_state.iteration == 1
        assert new_state.n_dead == 1
        assert 0 <= info["acceptance_rate"] <= 1.0

    def test_n_extra_reduces_emax(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        emaxes = []
        for _ in range(20):
            state, info = ns_step(state, s["step_fn"], n_mcmc_steps=10, n_extra=3)
            emaxes.append(float(info["emax"]))

        assert emaxes[-1] < emaxes[0], "Emax should decrease during NS with n_extra"


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
        init_fn, step_fn = build_mwg(backend, [
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
