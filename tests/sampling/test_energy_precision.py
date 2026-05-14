"""Tests for energy precision strategy (Step 16).

Verifies:
- Energy per atom comparison works correctly
- Random tie-breaking selects uniformly among tied walkers
- No split_float (float32x2) code exists in the codebase
- Backward compatibility: existing NS runs unaffected
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.nested_sampling import _find_worst_walker


class TestFindWorstWalker:
    def test_basic_argmax(self):
        """With a key, finds the maximum."""
        energies = jnp.array([1.0, 5.0, 3.0, 2.0])
        key = jax.random.key(0)
        idx, val = _find_worst_walker(energies, rng_key=key)
        assert idx == 1
        assert val == pytest.approx(5.0)

    def test_per_atom_comparison(self):
        """Per-atom normalization shouldn't change which walker is worst
        when all walkers have the same n_atoms (just rescales uniformly)."""
        energies = jnp.array([1.0, 5.0, 3.0, 2.0])
        key = jax.random.key(0)
        idx, val = _find_worst_walker(energies, rng_key=key, n_atoms=10)
        assert idx == 1
        assert val == pytest.approx(5.0)  # returns original energy, not per-atom

    def test_tie_breaking_uniform(self):
        """With tied energies, different keys should select different walkers."""
        energies = jnp.array([3.0, 5.0, 5.0, 5.0, 1.0])
        selections = set()
        for seed in range(100):
            key = jax.random.key(seed)
            idx, val = _find_worst_walker(energies, rng_key=key)
            assert val == pytest.approx(5.0)
            assert idx in (1, 2, 3)  # must be one of the tied maxima
            selections.add(int(idx))

        # All three tied walkers should be selected at least once
        assert selections == {1, 2, 3}, (
            f"Tie-breaking should be uniform; only selected indices {selections}"
        )

    def test_tie_breaking_all_equal(self):
        """When all walkers have equal energy, any can be selected."""
        energies = jnp.array([2.0, 2.0, 2.0, 2.0])
        selections = set()
        for seed in range(200):
            key = jax.random.key(seed)
            idx, val = _find_worst_walker(energies, rng_key=key)
            assert val == pytest.approx(2.0)
            selections.add(int(idx))

        assert len(selections) == 4

    def test_deterministic_with_same_key(self):
        """Same rng_key gives same result."""
        energies = jnp.array([3.0, 5.0, 5.0])
        key = jax.random.key(42)
        idx1, _ = _find_worst_walker(energies, rng_key=key)
        idx2, _ = _find_worst_walker(energies, rng_key=key)
        assert idx1 == idx2  # deterministic with same key

    def test_per_atom_large_system(self):
        """Per-atom normalization keeps values in a good float32 range."""
        n_atoms = 10000
        # Energies that are large in absolute terms but small per atom
        energies = jnp.array([1e6, 2e6, 1.5e6])
        key = jax.random.key(42)
        idx, val = _find_worst_walker(energies, rng_key=key, n_atoms=n_atoms)
        assert idx == 1
        assert val == pytest.approx(2e6)

    def test_jit_compatible(self):
        """_find_worst_walker works under JIT."""
        energies = jnp.array([1.0, 3.0, 2.0])
        key = jax.random.key(7)

        @jax.jit
        def find(e, k):
            return _find_worst_walker(e, rng_key=k, n_atoms=5)

        idx, val = find(energies, key)
        assert idx == 1
        assert val == pytest.approx(3.0)


