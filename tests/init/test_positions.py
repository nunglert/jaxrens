"""Tests for uniform_positions_in_cell and grid_positions_in_cell.

Both live in jaxrens.init.positions.  The last test runs the grid
initialisation end to end through resolve and run_from_config.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.cli.resolve import resolve
from jaxrens.cli.run import run_from_config
from jaxrens.cli.schema import RootSpec
from jaxrens.init.positions import (
    grid_positions_in_cell,
    uniform_positions_in_cell,
)

_KEY = jax.random.key(0)
_CELL = 6.0 * jnp.eye(3, dtype=jnp.float32)


class TestUniformPositionsInCell:
    """Check uniform_positions_in_cell.

    The tests cover output shape, positions staying inside the cell (cubic
    and non-cubic), determinism for a fixed key, and jit/vmap compatibility.
    """

    def test_shape(self):
        pos = uniform_positions_in_cell(_KEY, _CELL, n_atoms=5)
        assert pos.shape == (5, 3)

    def test_positions_inside_cell(self):
        n_atoms = 20
        pos = uniform_positions_in_cell(_KEY, _CELL, n_atoms=n_atoms)
        cell_inv = jnp.linalg.inv(_CELL)
        frac = pos @ cell_inv
        assert jnp.all(frac >= 0.0 - 1e-5)
        assert jnp.all(frac <= 1.0 + 1e-5)

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
        cell = jnp.array(
            [[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.5, 6.0]],
            dtype=jnp.float32,
        )
        pos = uniform_positions_in_cell(_KEY, cell, n_atoms=10)
        assert pos.shape == (10, 3)
        cell_inv = jnp.linalg.inv(cell)
        frac = pos @ cell_inv
        assert jnp.all(frac >= 0.0 - 1e-4)
        assert jnp.all(frac <= 1.0 + 1e-4)


class TestGridPositionsInCell:
    """Check grid_positions_in_cell.

    The tests cover output shape, the minimum pairwise distance (cubic and
    oblique cells), the error when the grid is too coarse, determinism for a
    fixed key, and vmap compatibility.
    """

    def test_shape(self):
        pos = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=2.0)
        assert pos.shape == (4, 3)

    def test_pairwise_distances_above_min(self):
        n_atoms = 6
        min_dist = 1.5
        pos = grid_positions_in_cell(
            _KEY, _CELL, n_atoms=n_atoms, min_distance=min_dist
        )
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = float(jnp.linalg.norm(pos[i] - pos[j]))
                assert d >= min_dist - 1e-4

    @pytest.mark.xfail(
        strict=True,
        reason="grid counts use edge lengths, so sites of an oblique cell "
        "can be closer than min_distance",
    )
    def test_pairwise_distances_above_min_oblique_cell(self):
        # Edges a and b are nearly parallel: the sites (i, j) and (i+1, j-1)
        # differ by a/4 - b/4, about 0.3 long.  Asking for every site makes
        # the close pair part of the result.
        cell = jnp.array(
            [[6.0, 0.0, 0.0], [5.9, 1.2, 0.0], [0.0, 0.0, 6.0]],
            dtype=jnp.float32,
        )
        n_atoms = 64
        min_dist = 1.5
        pos = grid_positions_in_cell(
            _KEY, cell, n_atoms=n_atoms, min_distance=min_dist
        )
        dists = jnp.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        dists = dists[jnp.triu_indices(n_atoms, k=1)]
        assert float(jnp.min(dists)) >= min_dist - 1e-4

    def test_raises_when_too_small(self):
        tiny_cell = 2.0 * jnp.eye(3, dtype=jnp.float32)
        with pytest.raises(ValueError, match="grid too coarse"):
            grid_positions_in_cell(
                _KEY, tiny_cell, n_atoms=100, min_distance=1.5
            )

    def test_deterministic_same_key(self):
        a = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=1.5)
        b = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=1.5)
        assert jnp.array_equal(a, b)

    def test_different_keys_different_results(self):
        key2 = jax.random.key(77)
        a = grid_positions_in_cell(_KEY, _CELL, n_atoms=4, min_distance=1.5)
        b = grid_positions_in_cell(key2, _CELL, n_atoms=4, min_distance=1.5)
        assert not jnp.array_equal(a, b)

    def test_vmap_safe(self):
        keys = jax.random.split(_KEY, 5)
        fn = jax.vmap(
            lambda k: grid_positions_in_cell(
                k, _CELL, n_atoms=3, min_distance=1.5
            )
        )
        result = fn(keys)
        assert result.shape == (5, 3, 3)
        assert jnp.all(jnp.isfinite(result))

    def test_n_atoms_equals_one(self):
        pos = grid_positions_in_cell(_KEY, _CELL, n_atoms=1, min_distance=2.0)
        assert pos.shape == (1, 3)
        assert jnp.all(jnp.isfinite(pos))


def test_start_species_e2e_run_ns(tmp_path):
    """Resolver output feeds directly into run_from_config without error."""
    d = {
        "run": {
            "n_live": 6,
            "max_iterations": 5,
            "n_mcmc_steps": 3,
            "seed": 0,
        },
        "moves": [{"move_type": "random_walk", "step_size": 0.3}],
        "backend": {"backend_type": "harmonic"},
        "output": {
            "format": "none",
            "working_dir": str(tmp_path),
            "info_interval": 999,
        },
        "init": {
            "start_species": "1 2",
            "random_initialise_pos": True,
            "random_initialise_cell": False,
            "pos_randomization_mode": "grid",
            "grid_distance": 1.5,
            "init_distance_criterion": 0.5,
            "random_init_max_n_tries": 50,
            "start_energy_ceiling_per_atom": 1e6,
            "pos_autoscale_cells": False,
        },
    }
    root = RootSpec.model_validate(d)
    resolved = resolve(root)

    result = run_from_config(
        resolved.ns,
        list(resolved.moves),
        resolved.backend,
        resolved.output,
        initial_positions=resolved.init.initial_positions,
        initial_types=resolved.init.initial_types,
        initial_energies=resolved.init.initial_energies,
        initial_cells=resolved.init.initial_cells,
    )
    assert result["iteration"] > 0
    assert jnp.isfinite(result["log_evidence"])
