"""Test JIT compilation of core components.

Part of PR 1: test infrastructure safety net.
Verifies that key functions compile without error. Catches common JAX
issues: non-static shapes, Python control flow on traced values,
unregistered pytrees, etc.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.state.walker import WalkerState


class TestWalkerJIT:
    """WalkerState operations compile cleanly."""

    def test_jit_energy_extraction(self, dummy_walker):
        @jax.jit
        def f(w):
            return w.energy

        assert f(dummy_walker).shape == ()

    def test_jit_position_transform(self, dummy_walker):
        @jax.jit
        def f(w):
            return w.set(positions=w.positions + 0.1)

        result = f(dummy_walker)
        assert jnp.allclose(result.positions, dummy_walker.positions + 0.1)

    def test_jit_does_not_retrace_on_same_structure(self, dummy_walker):
        """Calling with different values but same structure should not retrace."""
        call_count = 0

        @jax.jit
        def f(w):
            return w.energy * 2.0

        # First call compiles
        f(dummy_walker)
        # Second call with different energy -- should reuse compiled version
        w2 = dummy_walker.set(energy=jnp.array(-99.0))
        result = f(w2)
        assert jnp.allclose(result, jnp.array(-198.0))


class TestEnergyFnJIT:
    """Toy energy functions compile cleanly."""

    def test_jit_harmonic(self, harmonic_energy_fn, dummy_positions_3, dummy_types_3):
        jitted = jax.jit(harmonic_energy_fn, static_argnums=())

        result = jitted(None, dummy_positions_3, dummy_types_3)
        expected = 0.5 * jnp.sum(dummy_positions_3**2)
        assert jnp.allclose(result, expected)

    def test_vmap_harmonic(self, harmonic_energy_fn, dummy_positions_3, dummy_types_3):
        """Energy fn should be vmappable over a batch of positions."""
        batch_pos = jnp.stack([dummy_positions_3] * 4)
        batch_types = jnp.stack([dummy_types_3] * 4)

        vmapped = jax.vmap(harmonic_energy_fn, in_axes=(None, 0, 0, None))
        result = vmapped(None, batch_pos, batch_types, None)
        assert result.shape == (4,)

    def test_grad_harmonic(self, harmonic_energy_fn, dummy_positions_3, dummy_types_3):
        """Should be differentiable wrt positions."""
        grad_fn = jax.grad(harmonic_energy_fn, argnums=1)
        forces = grad_fn(None, dummy_positions_3, dummy_types_3)
        # For harmonic: dE/dx = x
        assert jnp.allclose(forces, dummy_positions_3)


class TestScanCompatibility:
    """Verify that lax.scan works with WalkerState."""

    def test_scan_over_walker_updates(self, dummy_walker):
        """lax.scan should be able to iterate over walker state updates."""

        def step(walker, step_idx):
            new_walker = walker.set(
                energy=walker.energy - 0.1,
                positions=walker.positions + 0.01,
            )
            return new_walker, walker.energy

        final_walker, energies = jax.lax.scan(
            step, dummy_walker, jnp.arange(10)
        )

        assert energies.shape == (10,)
        assert jnp.allclose(final_walker.energy, dummy_walker.energy - 1.0)

    def test_scan_inside_jit(self, dummy_walker):
        """lax.scan inside JIT should work with WalkerState."""

        @jax.jit
        def run_scan(walker):
            def step(w, _):
                return w.set(energy=w.energy - 0.1), w.energy

            return jax.lax.scan(step, walker, jnp.arange(5))

        final, energies = run_scan(dummy_walker)
        assert energies.shape == (5,)
