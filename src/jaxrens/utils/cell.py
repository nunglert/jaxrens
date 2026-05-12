"""Cell/box utility functions for periodic systems.

All functions are jit-compatible: no Python conditionals on traced values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def get_volume(cell: jnp.ndarray) -> jnp.ndarray:
    """Return the volume of a cell: |det(cell)|."""
    return jnp.abs(jnp.linalg.det(cell))


def min_perpendicular_distance(cell: jnp.ndarray) -> jnp.ndarray:
    """Minimum perpendicular distance between opposite faces of the parallelepiped.

    For each pair of opposite faces (perpendicular to lattice vector i), the
    inter-face distance is V / ||b_j × b_k||. Returns the minimum over the
    three pairs — this is the tightest dimension of the cell, the relevant
    quantity for the minimum image convention (MIC needs r_cut ≤ ½ · this).
    """
    volume = get_volume(cell)
    cross_bc = jnp.cross(cell[1], cell[2])
    cross_ca = jnp.cross(cell[2], cell[0])
    cross_ab = jnp.cross(cell[0], cell[1])
    norms = jnp.stack([
        jnp.linalg.norm(cross_bc),
        jnp.linalg.norm(cross_ca),
        jnp.linalg.norm(cross_ab),
    ])
    return volume / jnp.max(norms)


def min_aspect_ratio(cell: jnp.ndarray, volume: jnp.ndarray) -> jnp.ndarray:
    """Minimum aspect ratio across 3 cell vector pairs.

    For each pair (i, j) of adjacent cell vectors, compute the cross product,
    normalize it, dot with the remaining cell vector, and divide by V^(1/3).

    Args:
        cell: (3, 3) cell matrix, rows are lattice vectors.
        volume: scalar cell volume.

    Returns:
        Scalar minimum aspect ratio.
    """
    v_cbrt = volume ** (1.0 / 3.0)
    # Pairs: (0,1)->2, (1,2)->0, (0,2)->1
    pairs = [(0, 1, 2), (1, 2, 0), (0, 2, 1)]
    ratios = []
    for i, j, k in pairs:
        cross = jnp.cross(cell[i], cell[j])
        cross_norm = cross / jnp.linalg.norm(cross)
        height = jnp.abs(jnp.dot(cross_norm, cell[k]))
        ratios.append(height / v_cbrt)
    return jnp.min(jnp.array(ratios))


def check_cell_shape(
    cell: jnp.ndarray,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
) -> jnp.ndarray:
    """Check whether cell shape satisfies geometric criteria.

    Returns a boolean scalar (True = valid). Fully jit-compatible.
    """
    volume = get_volume(cell)
    vol_per_atom = volume / n_atoms
    aspect = min_aspect_ratio(cell, volume)

    valid = (
        (vol_per_atom <= max_vol_per_atom)
        & (vol_per_atom >= min_vol_per_atom)
        & (aspect >= min_aspect)
    )
    return valid


def get_cell_transformation(
    new_cell: jnp.ndarray, old_cell: jnp.ndarray
) -> jnp.ndarray:
    """Compute transformation matrix T such that new_cell = old_cell @ T.

    Returns T = solve(old_cell, new_cell).
    """
    return jnp.linalg.solve(old_cell, new_cell)


def transform_positions(
    positions: jnp.ndarray,
    old_cell: jnp.ndarray,
    new_cell: jnp.ndarray,
) -> jnp.ndarray:
    """Transform positions from old cell to new cell.

    positions_new = positions @ T where T = get_cell_transformation(new_cell, old_cell).
    """
    T = get_cell_transformation(new_cell, old_cell)
    # HIGHEST precision: TF32 (10-bit mantissa on GPU) corrupts positions@T
    # by ~3.7e-3 even for identity T, spiking LJ energy at dense packing.
    return jnp.einsum("ij,jk->ik", positions, T, precision=jax.lax.Precision.HIGHEST)
