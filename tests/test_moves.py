"""Test move kernels: random walk and Galilean Monte Carlo.

For each move: (a) init returns valid state, (b) step produces expected output,
(c) JIT compilation succeeds, (d) vmap over walkers works, (e) acceptance
rate is reasonable on toy problem.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.moves.random_walk import (
    RandomWalkState,
    init as rw_init,
    build_kernel as rw_build_kernel,
    as_top_level_api as rw_api,
)
from jaxrens.sampling.moves.galilean import (
    GalileanState,
    init as gmc_init,
    build_kernel as gmc_build_kernel,
    as_top_level_api as gmc_api,
)


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


@pytest.fixture
def positions():
    return jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])


@pytest.fixture
def types():
    return jnp.array([0, 0])


# ---------------------------------------------------------------------------
# Random Walk
# ---------------------------------------------------------------------------


class TestRandomWalkInit:
    def test_returns_named_tuple(self, positions, types):
        state = rw_init(positions, types, energy=-1.0, step_size=0.1)
        assert isinstance(state, RandomWalkState)
        assert jnp.array_equal(state.positions, positions)
        assert jnp.allclose(state.energy, -1.0)
        assert jnp.allclose(state.step_size, 0.1)

    def test_with_box(self, positions, types):
        box = 5.0 * jnp.eye(3)
        state = rw_init(positions, types, energy=-1.0, box=box)
        assert state.box is not None


class TestRandomWalkStep:
    def test_step_returns_state_and_info(self, harmonic, positions, types):
        energy_fn, params = harmonic
        state = rw_init(positions, types, energy=1.0, step_size=0.05)
        step = jax.jit(rw_build_kernel(energy_fn, params))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=10.0)

        assert isinstance(new_state, RandomWalkState)
        assert hasattr(info, "accepted")
        assert hasattr(info, "n_evaluations")

    def test_accepts_below_constraint(self, harmonic, positions, types):
        energy_fn, params = harmonic
        # Start near origin with high Emax -> should always accept
        state = rw_init(positions, types, energy=0.25, step_size=0.01)
        step = jax.jit(rw_build_kernel(energy_fn, params))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=100.0)
        assert info.accepted

    def test_rejects_above_constraint(self, harmonic, positions, types):
        energy_fn, params = harmonic
        # Very tight constraint with large step -> likely to reject
        state = rw_init(positions, types, energy=0.001, step_size=100.0)
        step = jax.jit(rw_build_kernel(energy_fn, params))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.002)
        # With step_size=100, proposed energy will far exceed 0.002
        assert not info.accepted
        # Rejected -> positions unchanged
        assert jnp.array_equal(new_state.positions, state.positions)

    def test_jit(self, harmonic, positions, types):
        energy_fn, params = harmonic
        state = rw_init(positions, types, energy=1.0, step_size=0.1)
        step = rw_build_kernel(energy_fn, params)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 10.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, harmonic, positions, types):
        energy_fn, params = harmonic
        step = jax.jit(rw_build_kernel(energy_fn, params))

        # Create batch of 4 walkers
        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([1.0, 2.0, 3.0, 4.0])
        batch_step_size = jnp.array([0.1, 0.1, 0.1, 0.1])

        batch_state = RandomWalkState(
            positions=batch_pos,
            types=batch_types,
            energy=batch_energy,
            box=None,
            step_size=batch_step_size,
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 10.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_acceptance_rate_reasonable(self, harmonic, positions, types):
        """Run 200 steps, expect reasonable acceptance rate on easy problem."""
        energy_fn, params = harmonic
        state = rw_init(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(rw_build_kernel(energy_fn, params))

        n_steps = 200
        n_accepted = 0
        key = jax.random.key(123)

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=5.0)
            n_accepted += int(info.accepted)

        rate = n_accepted / n_steps
        # With step_size=0.05 and Emax=5.0, should accept most proposals
        assert 0.1 < rate < 1.0, f"Acceptance rate {rate} out of expected range"

    def test_scan_compatible(self, harmonic, positions, types):
        """lax.scan should work with random walk step."""
        energy_fn, params = harmonic
        state = rw_init(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(rw_build_kernel(energy_fn, params))

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 5.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)


# ---------------------------------------------------------------------------
# Galilean Monte Carlo
# ---------------------------------------------------------------------------


class TestGalileanInit:
    def test_returns_named_tuple(self, positions, types):
        state = gmc_init(positions, types, energy=-1.0, step_size=0.1)
        assert isinstance(state, GalileanState)
        assert jnp.array_equal(state.positions, positions)
        assert state.direction.shape == positions.shape

    def test_zero_direction_default(self, positions, types):
        state = gmc_init(positions, types, energy=-1.0)
        assert jnp.allclose(state.direction, 0.0)


class TestGalileanStep:
    def test_step_returns_state_and_info(self, harmonic, positions, types):
        energy_fn, params = harmonic
        state = gmc_init(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(energy_fn, params, n_reflect=3))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=10.0)

        assert isinstance(new_state, GalileanState)
        assert hasattr(info, "accepted")

    def test_direction_initialized_on_first_call(self, harmonic, positions, types):
        """Direction should be randomized if initially zero."""
        energy_fn, params = harmonic
        state = gmc_init(positions, types, energy=0.25, step_size=0.05)
        assert jnp.allclose(state.direction, 0.0)

        step = jax.jit(gmc_build_kernel(energy_fn, params, n_reflect=3))
        key = jax.random.key(0)
        new_state, _ = step(key, state, likelihood_constraint=10.0)

        # Direction should now be non-zero
        dir_norm = jnp.sqrt(jnp.sum(new_state.direction**2))
        assert dir_norm > 0.1

    def test_jit(self, harmonic, positions, types):
        energy_fn, params = harmonic
        state = gmc_init(positions, types, energy=0.25, step_size=0.05)
        step = gmc_build_kernel(energy_fn, params, n_reflect=3)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 10.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, harmonic, positions, types):
        energy_fn, params = harmonic
        step = jax.jit(gmc_build_kernel(energy_fn, params, n_reflect=3))

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.25, 0.5, 0.75, 1.0])
        batch_step_size = jnp.full(4, 0.05)
        batch_dir = jnp.zeros((4, *positions.shape))

        batch_state = GalileanState(
            positions=batch_pos,
            types=batch_types,
            energy=batch_energy,
            box=None,
            direction=batch_dir,
            step_size=batch_step_size,
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 10.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_reflection_keeps_energy_below_constraint(self, harmonic, positions, types):
        """After GMC step, accepted states should have energy < Emax."""
        energy_fn, params = harmonic
        state = gmc_init(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(energy_fn, params, n_reflect=5))

        key = jax.random.key(42)
        for i in range(20):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=2.0)
            if info.accepted:
                assert state.energy < 2.0

    def test_acceptance_rate_reasonable(self, harmonic, positions, types):
        energy_fn, params = harmonic
        state = gmc_init(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(energy_fn, params, n_reflect=5))

        n_steps = 100
        n_accepted = 0
        key = jax.random.key(123)

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=2.0)
            n_accepted += int(info.accepted)

        rate = n_accepted / n_steps
        assert 0.1 < rate <= 1.0, f"Acceptance rate {rate} out of expected range"

    def test_scan_compatible(self, harmonic, positions, types):
        energy_fn, params = harmonic
        state = gmc_init(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(energy_fn, params, n_reflect=3))

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 2.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_without_forces(self, harmonic, positions, types):
        """use_forces=False should still work (random reflection)."""
        energy_fn, params = harmonic
        state = gmc_init(positions, types, energy=0.25, step_size=0.05)
        step = gmc_build_kernel(
            energy_fn, params, n_reflect=3, use_forces=False
        )

        key = jax.random.key(0)
        new_state, info = jax.jit(step)(key, state, 10.0)
        assert new_state.energy.shape == ()


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


class TestTopLevelAPI:
    def test_random_walk_api(self, harmonic, positions, types):
        energy_fn, params = harmonic
        kernel = rw_api(energy_fn, params, step_size=0.05)
        state = kernel.init(positions, types, 0.25)
        new_state, info = kernel.step(jax.random.key(0), state, 5.0)
        assert isinstance(new_state, RandomWalkState)

    def test_galilean_api(self, harmonic, positions, types):
        energy_fn, params = harmonic
        kernel = gmc_api(energy_fn, params, step_size=0.05, n_reflect=3)
        state = kernel.init(positions, types, 0.25)
        new_state, info = kernel.step(jax.random.key(0), state, 5.0)
        assert isinstance(new_state, GalileanState)
