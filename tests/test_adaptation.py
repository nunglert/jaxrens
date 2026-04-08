"""Test adaptation wrappers: step size dual averaging.

Verifies that adaptation converges toward target acceptance rate
on a toy problem.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.adaptation.step_size import (
    AdaptationState,
    init_adaptation,
    dual_averaging_update,
    get_step_size,
)
from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.moves.random_walk import (
    init as rw_init,
    build_kernel as rw_build_kernel,
    RandomWalkState,
)


class TestAdaptationState:
    def test_init(self):
        state = init_adaptation(initial_step_size=0.1, target_acceptance=0.5)
        assert isinstance(state, AdaptationState)
        assert jnp.allclose(get_step_size(state), 0.1)

    def test_step_size_increases_on_accept(self):
        state = init_adaptation(initial_step_size=0.1, target_acceptance=0.5)
        # Many accepts -> step size should increase
        for _ in range(50):
            state = dual_averaging_update(state, accepted=jnp.array(True), target_acceptance=0.5)
        assert get_step_size(state) > 0.1

    def test_step_size_decreases_on_reject(self):
        state = init_adaptation(initial_step_size=0.1, target_acceptance=0.5)
        # Many rejects -> step size should decrease
        for _ in range(50):
            state = dual_averaging_update(state, accepted=jnp.array(False), target_acceptance=0.5)
        assert get_step_size(state) < 0.1

    def test_averaged_step_size_stable(self):
        state = init_adaptation(initial_step_size=0.1)
        # Alternating accept/reject at 50% rate
        for i in range(200):
            acc = jnp.array(i % 2 == 0)
            state = dual_averaging_update(state, accepted=acc)
        # Averaged step size should be reasonably close to initial
        avg = get_step_size(state, use_averaged=True)
        assert 0.01 < float(avg) < 10.0

    def test_jit_compatible(self):
        state = init_adaptation()

        @jax.jit
        def update(s, acc):
            return dual_averaging_update(s, acc)

        new_state = update(state, jnp.array(True))
        assert new_state.step == 1


class TestAdaptationConvergence:
    """Integration test: adaptation + random walk should converge."""

    def test_converges_to_target_acceptance(self):
        """Run adaptation with random walk on harmonic potential.

        After warmup, the acceptance rate should be near the target.
        """
        energy_fn, params = create_harmonic(k=1.0)
        positions = jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        types = jnp.array([0, 0])

        target_acc = 0.4
        adapt_state = init_adaptation(initial_step_size=0.1, target_acceptance=target_acc)

        # Warmup: 200 steps with adaptation
        step_fn = jax.jit(rw_build_kernel(energy_fn, params))
        rw_state = rw_init(positions, types, energy=0.25, step_size=0.1)
        key = jax.random.key(42)

        for i in range(200):
            key, subkey = jax.random.split(key)
            # Set step size from adaptation
            current_ss = get_step_size(adapt_state)
            rw_state = rw_state._replace(step_size=current_ss)
            rw_state, info = step_fn(subkey, rw_state, likelihood_constraint=3.0)
            adapt_state = dual_averaging_update(
                adapt_state, info.accepted, target_acceptance=target_acc
            )

        # Production: 200 steps with fixed step size, measure acceptance
        final_ss = get_step_size(adapt_state, use_averaged=True)
        rw_state = rw_state._replace(step_size=final_ss)

        n_accepted = 0
        n_steps = 200
        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            rw_state, info = step_fn(subkey, rw_state, likelihood_constraint=3.0)
            n_accepted += int(info.accepted)

        rate = n_accepted / n_steps
        # Should be within 0.15 of target (generous tolerance for stochastic test)
        assert abs(rate - target_acc) < 0.2, (
            f"Acceptance rate {rate:.2f} too far from target {target_acc}"
        )

    def test_scan_compatible(self):
        """Adaptation should work inside lax.scan."""
        adapt_state = init_adaptation()

        def scan_step(adapt_state, accepted):
            new_state = dual_averaging_update(adapt_state, accepted)
            return new_state, get_step_size(new_state)

        accepts = jnp.array([True, True, False, True, False] * 4)
        final_state, step_sizes = jax.lax.scan(scan_step, adapt_state, accepts)

        assert step_sizes.shape == (20,)
        assert final_state.step == 20
