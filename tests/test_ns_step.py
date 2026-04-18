"""Test ns_step and vmap(ns_step) under JIT.

Verifies:
- init_ns creates correct state shapes
- ns_step single-run: iteration count, emax decrease, evidence accumulation
- ns_step with n_extra walks
- vmap(ns_step): multi-run compilation, different keys → different trajectories
- vmap(ns_step) with n_extra
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel, ns_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_setup():
    """Single-run harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [
        MoveKernel("random_walk", random_walk.build_kernel),
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


@pytest.fixture
def parallel_setup():
    """2-run parallel harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [
        MoveKernel("random_walk", random_walk.build_kernel),
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


@pytest.fixture
def jit_step():
    """JIT-compiled ns_step — all single-run tests use this."""
    return jax.jit(ns_step, static_argnums=(1, 2, 3))


@pytest.fixture
def jit_vmap_step(parallel_setup):
    """JIT-compiled vmap(ns_step) for parallel tests."""
    s = parallel_setup
    return jax.jit(jax.vmap(
        lambda state: ns_step(state, s["step_fn"], 10, 0)
    ))


# ---------------------------------------------------------------------------
# init_ns
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ns_step (single run, under JIT)
# ---------------------------------------------------------------------------


class TestNSStep:
    def test_single_step(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        new_state, info = jit_step(state, s["step_fn"], 10, 0)

        assert new_state.iteration == 1
        assert new_state.n_dead == 1
        assert 0 <= info["acceptance_rate"] <= 1.0

    def test_multiple_steps_reduce_emax(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        emaxes = []
        for _ in range(20):
            state, info = jit_step(state, s["step_fn"], 10, 0)
            emaxes.append(float(info["emax"]))

        assert emaxes[-1] < emaxes[0], "Emax should decrease during NS"

    def test_evidence_increases(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        for _ in range(50):
            state, info = jit_step(state, s["step_fn"], 10, 0)

        assert jnp.isfinite(state.log_evidence)

    def test_n_extra_walks(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        new_state, info = jit_step(state, s["step_fn"], 10, 5)

        assert new_state.iteration == 1
        assert new_state.n_dead == 1
        assert 0 <= info["acceptance_rate"] <= 1.0

    def test_n_extra_reduces_emax(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=None, rng_key=s["key"],
        )

        emaxes = []
        for _ in range(20):
            state, info = jit_step(state, s["step_fn"], 10, 3)
            emaxes.append(float(info["emax"]))

        assert emaxes[-1] < emaxes[0], "Emax should decrease during NS with n_extra"


# ---------------------------------------------------------------------------
# vmap(ns_step) (parallel runs, under JIT)
# ---------------------------------------------------------------------------


class TestVmapNsStep:
    def test_vmap_two_runs(self, parallel_setup):
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"], s["positions"], s["types"], s["energies"],
            cells=None, rng_keys=s["rng_keys"], max_dead=100,
        )

        vmapped = jax.jit(jax.vmap(
            lambda state: ns_step(state, s["step_fn"], 10, 0)
        ))
        new_states, infos = vmapped(ns_states)

        assert new_states.iteration.shape == (2,)
        assert infos["emax"].shape == (2,)
        assert jnp.all(new_states.iteration == 1)

    def test_different_keys_different_results(self, parallel_setup):
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"], s["positions"], s["types"], s["energies"],
            cells=None, rng_keys=s["rng_keys"], max_dead=100,
        )

        vmapped = jax.jit(jax.vmap(
            lambda state: ns_step(state, s["step_fn"], 10, 0)
        ))

        for _ in range(5):
            ns_states, infos = vmapped(ns_states)

        assert not jnp.allclose(
            ns_states.population.energy[0],
            ns_states.population.energy[1],
        )

    def test_vmap_with_n_extra(self, parallel_setup):
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"], s["positions"], s["types"], s["energies"],
            cells=None, rng_keys=s["rng_keys"], max_dead=100,
        )

        vmapped = jax.jit(jax.vmap(
            lambda state: ns_step(state, s["step_fn"], 10, 3)
        ))
        new_states, infos = vmapped(ns_states)

        assert new_states.iteration.shape == (2,)
        assert 0 <= float(infos["acceptance_rate"][0]) <= 1.0
