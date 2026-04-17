"""Pure-JAX position generators for walker initialization.

Two public functions:
  - uniform_positions_in_cell: uniform fractional draw -> Cartesian
  - grid_positions_in_cell: regular grid spaced by min_distance -> Cartesian
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


def uniform_positions_in_cell(
    key: jax.Array,
    cell: jnp.ndarray,
    n_atoms: int,
) -> jnp.ndarray:
    """Uniform draw in fractional coords, then fractional -> Cartesian.

    Args:
        key: JAX PRNG key.
        cell: (3, 3) cell matrix, rows are lattice vectors.
        n_atoms: Number of atoms to place.

    Returns:
        positions: (n_atoms, 3) Cartesian positions.
    """
    frac = jax.random.uniform(key, (n_atoms, 3))
    return frac @ cell


def grid_positions_in_cell(
    key: jax.Array,
    cell: jnp.ndarray,
    n_atoms: int,
    min_distance: float,
) -> jnp.ndarray:
    """Pick n_atoms sites on a regular grid spaced by min_distance.

    Grid counts per axis are derived from cell edge lengths (||a||, ||b||, ||c||)
    divided by min_distance, floored to an integer.  Grid sites are built in
    fractional coordinates using numpy (the integer index ranges are Python-side
    scalars), then jax.random.choice selects n_atoms sites without replacement.
    Conversion to Cartesian uses selected @ cell.

    Args:
        key: JAX PRNG key.
        cell: (3, 3) cell matrix, rows are lattice vectors.
        n_atoms: Number of atoms to place.
        min_distance: Minimum spacing between grid points (in same units as cell).

    Returns:
        positions: (n_atoms, 3) Cartesian positions.

    Raises:
        ValueError: If the grid cannot fit n_atoms at the given min_distance.
    """
    a = float(np.linalg.norm(np.array(cell[0])))
    b = float(np.linalg.norm(np.array(cell[1])))
    c = float(np.linalg.norm(np.array(cell[2])))

    na = max(1, int(a / min_distance))
    nb = max(1, int(b / min_distance))
    nc = max(1, int(c / min_distance))

    n_grid = na * nb * nc
    if n_grid < n_atoms:
        raise ValueError(
            f"grid too coarse: {n_grid} sites available ({na}x{nb}x{nc}), "
            f"{n_atoms} atoms requested; increase cell volume or decrease grid_distance"
        )

    # Build fractional grid using numpy index ranges (Python-side, not traced).
    fa = (np.arange(na) + 0.5) / na
    fb = (np.arange(nb) + 0.5) / nb
    fc = (np.arange(nc) + 0.5) / nc

    ga, gb, gc = np.meshgrid(fa, fb, fc, indexing="ij")
    grid_frac = jnp.array(
        np.stack([ga.ravel(), gb.ravel(), gc.ravel()], axis=1), dtype=jnp.float32
    )

    indices = jax.random.choice(key, n_grid, (n_atoms,), replace=False)
    selected_frac = grid_frac[indices]
    return selected_frac @ cell
