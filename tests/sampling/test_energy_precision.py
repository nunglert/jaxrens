"""Tests for the worst-walker selection in `_find_worst_walker`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.nested_sampling import _find_worst_walker


class TestFindWorstWalker:
    """Check that _find_worst_walker returns the maximum potential.

    Ties are exact equalities, broken uniformly at random.  The returned value
    is always the strict ``jnp.max``, so no survivor can sit above it.
    """

    def test_basic_argmax(self):
        """With a key, finds the maximum."""
        energies = jnp.array([1.0, 5.0, 3.0, 2.0])
        key = jax.random.key(0)
        idx, val = _find_worst_walker(energies, rng_key=key)
        assert idx == 1
        assert val == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "energies,tied",
        [
            ([3.0, 5.0, 5.0, 5.0, 1.0], {1, 2, 3}),
            ([2.0, 2.0, 2.0, 2.0], {0, 1, 2, 3}),
            ([0.0, 0.0], {0, 1}),
        ],
    )
    def test_tie_breaking_covers_all_tied(self, energies, tied):
        """Different keys pick every walker tied at the maximum, only those."""
        energies = jnp.array(energies)
        selections = set()
        for seed in range(200):
            idx, val = _find_worst_walker(
                energies, rng_key=jax.random.key(seed)
            )
            assert float(val) == float(jnp.max(energies))
            selections.add(int(idx))
        assert selections == tied

    def test_near_ties_not_collapsed(self):
        """Walkers within float32 rtol but not exactly equal are not tied."""
        energies = jnp.array([-105.213005, -105.212997], dtype=jnp.float32)
        strict_max = jnp.max(energies)
        for seed in range(50):
            key = jax.random.key(seed)
            idx, val = _find_worst_walker(energies, rng_key=key)
            assert int(idx) == int(jnp.argmax(energies))
            assert float(val) == float(strict_max)

    def test_jit_compatible(self):
        """_find_worst_walker works under JIT."""
        energies = jnp.array([1.0, 3.0, 2.0])
        key = jax.random.key(7)

        @jax.jit
        def find(e, k):
            return _find_worst_walker(e, rng_key=k)

        idx, val = find(energies, key)
        assert idx == 1
        assert val == pytest.approx(3.0)
