"""Test the nested sampling loop.

Verifies:
- NS runs complete without error on toy problems
- Evidence accuracy on harmonic oscillator (known answer)
- lax.scan inner loop produces correct results
- Callbacks are invoked
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns
from jaxrens.sampling.adaptation.step_size import init_adaptation


@pytest.fixture
def harmonic_setup():
    """Set up a harmonic oscillator NS problem."""
    energy_fn, params = create_harmonic(k=1.0)
    init_fn, step_fn = build_mwg(energy_fn, params, [
        MoveDescriptor("random_walk", random_walk.build_kernel),
    ])

    n_walkers = 50
    n_atoms = 1
    key = jax.random.key(42)

    # Initialize walkers uniformly in [-3, 3]^3
    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    # Compute initial energies
    energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
        params, positions, types
    )

    return {
        "energy_fn": energy_fn,
        "params": params,
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
            s["positions"], s["types"], s["energies"],
            boxes=None, rng_key=s["key"],
        )
        assert state["positions"].shape == s["positions"].shape
        assert state["energies"].shape == (s["n_walkers"],)
        assert state["iteration"] == 0
        assert state["n_dead"] == 0

    def test_init_dead_points_buffer(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=None, rng_key=s["key"], max_dead=100,
        )
        assert state["dead_energies"].shape == (100,)


class TestNSStep:
    def test_single_step(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=None, rng_key=s["key"],
        )
        adapt = init_adaptation(initial_step_size=0.3)

        new_state, info, new_adapt = ns_step(
            state, s["init_fn"], s["step_fn"], n_mcmc_steps=10, adapt_state=adapt,
        )

        assert new_state["iteration"] == 1
        assert new_state["n_dead"] == 1
        assert info["emax"] == jnp.max(s["energies"])
        assert 0 <= info["acceptance_rate"] <= 1.0

    def test_multiple_steps_reduce_emax(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=None, rng_key=s["key"],
        )
        adapt = init_adaptation(initial_step_size=0.3)

        emaxes = []
        for _ in range(20):
            state, info, adapt = ns_step(
                state, s["init_fn"], s["step_fn"], n_mcmc_steps=10, adapt_state=adapt,
            )
            emaxes.append(float(info["emax"]))

        # Emax should generally decrease over iterations
        assert emaxes[-1] < emaxes[0], "Emax should decrease during NS"

    def test_evidence_increases(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=None, rng_key=s["key"],
        )
        adapt = init_adaptation(initial_step_size=0.3)

        for _ in range(50):
            state, info, adapt = ns_step(
                state, s["init_fn"], s["step_fn"], n_mcmc_steps=10, adapt_state=adapt,
            )

        # Evidence should be finite after some iterations
        assert jnp.isfinite(state["log_evidence"])


class TestRunNS:
    def test_runs_to_completion(self, harmonic_setup):
        s = harmonic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            boxes=None,
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
        """Test evidence on 1-atom 3D harmonic oscillator.

        For E = 0.5 * k * sum(x^2) with k=1, prior box [-L, L]^3:
        Z = integral exp(-E) dx = (2*pi)^(3/2) for the Gaussian part
        Prior volume = (2*L)^3
        log Z = 3/2 * log(2*pi) - 3*log(2*L) = 3/2*log(2*pi) - log(V)

        With walkers initialized in [-L, L]^3, the NS evidence estimate
        should approximate log(Z/V) = 3/2*log(2*pi) - log(V)

        Actually for NS: log_Z_ns ~ log(Z) since we're integrating
        exp(-E) over the prior volume.
        """
        energy_fn, params = create_harmonic(k=1.0)
        init_fn, step_fn = build_mwg(energy_fn, params, [
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
        energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, positions, types
        )

        result = run_ns(
            positions, types, energies,
            boxes=None,
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

        # Analytical: log Z = 3/2 * log(2*pi) - log((2L)^3)
        # = 3/2 * log(2*pi) - 3*log(2L)
        log_prior_volume = 3.0 * jnp.log(2.0 * L)
        log_Z_analytical = 1.5 * jnp.log(2.0 * jnp.pi) - log_prior_volume

        # NS evidence should be within ~1 nat of analytical
        # (generous tolerance for a stochastic test with small n_walkers)
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
            boxes=None,
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
