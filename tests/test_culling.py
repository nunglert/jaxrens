"""Tests for worst-walker selection utilities.

Verifies:
- _find_worst_walkers returns correct top-k
- Tie-breaking works with and without rng_key
- Results are sorted descending

Note: ns_step_multi_cull has been removed in favor of a future
JIT-compatible multi-cull implementation. These tests cover the
standalone _find_worst_walkers utility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.nested_sampling import _find_worst_walkers


class TestFindWorstWalkers:
    def test_top_k_correct(self):
        energies = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        indices, values = _find_worst_walkers(energies, n_cull=2)
        # Top-2 are indices 1 (5.0) and 4 (4.0)
        assert set(int(i) for i in indices) == {1, 4}
        assert jnp.max(values) == pytest.approx(5.0)

    def test_top_k_single(self):
        energies = jnp.array([1.0, 5.0, 3.0])
        indices, values = _find_worst_walkers(energies, n_cull=1)
        assert int(indices[0]) == 1
        assert float(values[0]) == pytest.approx(5.0)

    def test_top_k_with_ties(self):
        energies = jnp.array([5.0, 5.0, 5.0, 1.0])
        key = jax.random.key(42)
        indices, values = _find_worst_walkers(energies, n_cull=2, rng_key=key)
        assert len(indices) == 2
        # Both should be from the tied group {0, 1, 2}
        for idx in indices:
            assert int(idx) in {0, 1, 2}

    def test_sorted_descending(self):
        energies = jnp.array([1.0, 4.0, 2.0, 5.0, 3.0])
        indices, values = _find_worst_walkers(energies, n_cull=3)
        # Values should be in descending order
        assert float(values[0]) >= float(values[1]) >= float(values[2])
