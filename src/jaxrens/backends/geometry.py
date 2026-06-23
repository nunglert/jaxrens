"""Geometry helpers shared by energy backends and configuration constraints.

The minimum-image pairwise-distance computation lives here so that the
soft-core repulsion wrapper (:mod:`jaxrens.backends.softcore`) and the
configuration constraints (:mod:`jaxrens.constraints`) share a single,
tested implementation rather than each carrying their own copy.

All functions are jit/vmap/pmap safe: no Python conditionals on traced
values.
"""

from __future__ import annotations

import jax.numpy as jnp


def pairwise_distances(
    positions: jnp.ndarray, cell: jnp.ndarray
) -> jnp.ndarray:
    """Full ``(N, N)`` matrix of inter-atomic distances.

    Uses the minimum-image convention when ``cell`` is a real
    (non-degenerate) lattice, so close contacts across a periodic boundary
    are measured correctly; for a non-periodic system (``cell`` all zeros /
    singular) raw Cartesian distances are used. MIC is single-image ("111",
    O(N^2), no supercell expansion) — exact as long as the relevant cutoff
    stays below half the shortest perpendicular cell width (see
    :func:`jaxrens.utils.cell.min_perpendicular_distance`).

    The diagonal is the self-distance and is therefore exactly 0; callers
    that sum or threshold over pairs must mask it themselves.

    Args:
        positions: ``(N, 3)`` atomic positions.
        cell: ``(3, 3)`` lattice matrix (rows are lattice vectors); zeros
            for a non-periodic system.

    Returns:
        ``(N, N)`` symmetric distance matrix.
    """
    raw_delta = positions[:, None, :] - positions[None, :, :]

    # Minimum-image displacement when a real cell is present; raw
    # displacement otherwise. ``safe_cell`` keeps the inverse finite on the
    # non-periodic branch so the discarded MIC values never poison gradients.
    periodic = jnp.abs(jnp.linalg.det(cell)) > 1e-10
    safe_cell = jnp.where(periodic, cell, jnp.eye(3, dtype=cell.dtype))
    frac = positions @ jnp.linalg.inv(safe_cell)
    df = frac[:, None, :] - frac[None, :, :]
    df = df - jnp.round(df)
    mic_delta = df @ safe_cell

    delta = jnp.where(periodic, mic_delta, raw_delta)
    return jnp.linalg.norm(delta, axis=-1)
