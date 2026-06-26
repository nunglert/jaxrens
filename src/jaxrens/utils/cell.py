"""Cell/box utility functions for periodic systems.

All functions are jit-compatible: no Python conditionals on traced values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def get_volume(cell: jnp.ndarray) -> jnp.ndarray:
    """Return the volume of a cell: |det(cell)|."""
    return jnp.abs(jnp.linalg.det(cell))


def wrap_positions(positions: jnp.ndarray, cell: jnp.ndarray) -> jnp.ndarray:
    """Wrap atom positions into the home unit cell (fractional coords in [0, 1)).

    Graph backends (MACE/nequix) find neighbors via a *finite* supercell image
    set (e.g. ``n in {-1,0,1}^3``).  That set is only complete when every atom
    sits in (or within one cell of) the home cell, so atoms that have drifted
    out of the cell — which they do under unwrapped Cartesian moves over a long
    NS run — would silently lose edges and get a wrong, position-origin-dependent
    energy.  Wrapping restores the lattice-translation invariance the periodic
    energy must have.

    Convention matches the rest of this module: ``cell`` rows are lattice
    vectors and positions are row vectors, so ``frac = positions @ inv(cell)``
    and ``positions = frac @ cell``.

    Properties:
    * **Grad-safe:** ``floor`` has zero gradient, so ``d(wrapped)/d(positions)``
      is the identity almost everywhere — forces (galilean/HMC) are unchanged.
    * **Degenerate-cell-safe:** non-periodic systems pass ``cell == 0`` (or a
      dummy ``safe_cell``); for any singular cell this is a no-op, and the
      ``inv`` of the singular branch is masked so no NaNs leak through.

    Args:
        positions: (N, 3) Cartesian positions.
        cell: (3, 3) cell matrix, rows are lattice vectors.

    Returns:
        (N, 3) positions wrapped into the home cell, or ``positions`` unchanged
        when ``cell`` is degenerate (|det| ~ 0).
    """
    det = jnp.linalg.det(cell)
    periodic = jnp.abs(det) > 1e-12
    safe_cell = jnp.where(periodic, cell, jnp.eye(3))
    inv_cell = jnp.linalg.inv(safe_cell)
    frac = jnp.einsum(
        "ij,jk->ik", positions, inv_cell, precision=jax.lax.Precision.HIGHEST
    )
    frac = frac - jnp.floor(frac)
    wrapped = jnp.einsum(
        "ij,jk->ik", frac, cell, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.where(periodic, wrapped, positions)


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
    return jnp.einsum(
        "ij,jk->ik", positions, T, precision=jax.lax.Precision.HIGHEST
    )
