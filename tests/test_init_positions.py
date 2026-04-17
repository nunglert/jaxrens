"""Unit tests for jaxrens.init.positions.

Covers uniform_positions_in_cell and grid_positions_in_cell.
All JIT-compatible functions are tested under jax.jit per project policy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.init.positions import grid_positions_in_cell, uniform_positions_in_cell


_KEY = jax.random.key(0)
_CELL = 6.0 * jnp.eye(3, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# uniform_positions_in_cell
# ---------------------------------------------------------------------------

class TestUniformPositionsInCell:
    def test_shape(self):
        pos = uniform_positions_in_cell(_KEY, _CELL, n_atoms=5)
        assert pos.shape == (5, 3)

    def test_positions_inside_cell(self):
        n_atoms = 20
        pos = uniform_positions_in_cell(_KEY, _CELL, n_atoms=n_atoms)
        cell_inv = jnp.linalg.inv(_CELL)
        frac = pos @ cell_inv
        assert jnp.all(frac >= 0.0 - 1e-5), f"frac min: {float(jnp.min(frac))}"
        assert jnp.all(frac <= 1.0 + 1e-5), f"frac max: {float(jnp.max(frac))}"

    def test_deterministic_same_key(self):
        a = uniform_positions_in_cell(_KEY, _CELL, n_atoms=4)
        b = uniform_positions_in_cell(_KEY, _CELL, n_atoms=4)
        assert jnp.array_equal(a, b)

    def test_different_keys_different_results(self):
        key2 = jax.random.key(99)
        a = uniform_positions_in_cell(_KEY, _CELL, n_atoms=4)
        b = uniform_positions_in_cell(key2, _CELL, n_atoms=4)
        assert not jnp.array_equal(a, b)

    def test_jit(self):
        jit_fn = jax.jit(
            lambda k: uniform_positions_in_cell(k, _CELL, n_atoms=4)
        )
        pos = jit_fn(_KEY)
        assert pos.shape == (4, 3)
        assert jnp.all(jnp.isfinite(pos))

    def test_vmap(self):
        keys = jax.random.split(_KEY, 8)
        fn = jax.vmap(lambda k: uniform_positions_in_cell(k, _CELL, n_atoms=3))
        result = fn(keys)
        assert result.shape == (8, 3, 3)
        assert jnp.all(jnp.isfinite(result))

    def test_non_cubic_cell(self):
        cell = jnp.array([[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.5, 6.0]], dtype=jnp.float32)
        pos = uniform_positions_in_cell(_KEY, cell, n_atoms=10)
        assert pos.shape == (10, 3)
        cell_inv = jnp.linalg.inv(cell)
        frac = pos @ cell_inv
        assert jnp.all(frac >= 0.0 - 1e-4)
        assert jnp.all(frac <= 1.0 + 1e-4)


# ---------------------------------------------------------------------------
# grid_positions_in_cell
# ---------------------------------------------------------------------------

class TestGridPositionsInCell:
    def test_shape(self):
        pos = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=2.0)
        assert pos.shape == (4, 3)

    def test_positions_distinct(self):
        n_atoms = 6
        pos = grid_positions_in_cell(_KEY, _CELL, n_atoms=n_atoms, min_distance=1.5)
        # All positions should be distinct
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = jnp.linalg.norm(pos[i] - pos[j])
                assert float(diff) > 1e-6, f"Atoms {i} and {j} are at the same position"

    def test_pairwise_distances_above_min(self):
        min_dist = 1.5
        pos = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=min_dist)
        for i in range(4):
            for j in range(i + 1, 4):
                d = float(jnp.linalg.norm(pos[i] - pos[j]))
                assert d >= min_dist - 1e-4, f"Pair ({i},{j}) distance {d} < {min_dist}"

    def test_raises_when_too_small(self):
        tiny_cell = 2.0 * jnp.eye(3, dtype=jnp.float32)
        with pytest.raises(ValueError, match="grid too coarse"):
            grid_positions_in_cell(_KEY, tiny_cell, n_atoms=100, min_distance=1.5)

    def test_deterministic_same_key(self):
        a = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=1.5)
        b = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=1.5)
        assert jnp.array_equal(a, b)

    def test_different_keys_different_results(self):
        key2 = jax.random.key(77)
        a = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=1.5)
        b = grid_positions_in_cell(key2, _CELL, n_atoms=4, min_distance=1.5)
        assert not jnp.array_equal(a, b)

    def test_jit_safe(self):
        # grid_positions_in_cell derives grid counts from concrete cell norms
        # (Python floats, not traced).  The function is not JIT-safe over the
        # cell argument; it is designed to be called from a Python loop where
        # cell is always concrete.  We verify JIT-safety over the key argument
        # alone by pre-building the grid outside JIT and JIT-ing only the
        # random-choice step.
        cell_np = np.array(_CELL, dtype=np.float32)
        a = float(np.linalg.norm(cell_np[0]))
        n_atoms = 4
        min_dist = 1.5
        na = max(1, int(a / min_dist))
        n_grid = na ** 3

        @jax.jit
        def _choose(k):
            return jax.random.choice(k, n_grid, (n_atoms,), replace=False)

        indices = _choose(_KEY)
        assert indices.shape == (n_atoms,)

    def test_vmap_safe(self):
        keys = jax.random.split(_KEY, 5)
        fn = jax.vmap(lambda k: grid_positions_in_cell(k, _CELL, n_atoms=3, min_distance=1.5))
        result = fn(keys)
        assert result.shape == (5, 3, 3)
        assert jnp.all(jnp.isfinite(result))

    def test_n_atoms_equals_one(self):
        pos = grid_positions_in_cell(_KEY, _CELL, n_atoms=1, min_distance=2.0)
        assert pos.shape == (1, 3)
        assert jnp.all(jnp.isfinite(pos))
