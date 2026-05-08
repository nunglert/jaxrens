"""Shared JIT-compatible supercell edge finding for graph backends.

Both ``mace.py`` and ``nequix.py`` need the same neighbor-finding primitive:
expand the unit cell by the configured supercell transform, find pairs
within ``r_cutoff``, and return a fixed-size buffer of (sender, receiver,
shift) triples plus the true per-atom max neighbor count for outer-loop
overflow escalation.

Kept private (leading underscore) — it's an implementation detail of the
graph-NN backends, not a public API.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _make_image_offsets(sc_a: int, sc_b: int, sc_c: int) -> np.ndarray:
    """Generate integer image offset vectors for a centered supercell.

    Offsets range from -sc//2 to +sc//2 in each direction, ensuring
    neighbors are found in all periodic directions.

    Returns:
        (sc_dim, 3) integer array of image offsets.
    """
    a = np.arange(-(sc_a // 2), sc_a // 2 + 1)
    b = np.arange(-(sc_b // 2), sc_b // 2 + 1)
    c = np.arange(-(sc_c // 2), sc_c // 2 + 1)
    grid = np.stack(np.meshgrid(a, b, c, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def _max_neighbor_count_from_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """Return the per-atom max neighbor count from a (N, sc_dim*N) mask."""
    return jnp.max(jnp.sum(mask, axis=1))


def _neighbor_mask(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    r_cutoff: float,
    image_offsets: jnp.ndarray,
) -> jnp.ndarray:
    """Build the per-atom neighbor boolean mask, shape (N, sc_dim*N).

    Extracted from ``_supercell_edges`` so init-time bucket sizing can
    compute true neighbor counts without allocating the edge buffer or
    running the GNN forward.
    """
    cart_shifts = image_offsets @ cell
    super_positions = (
        positions[None, :, :] + cart_shifts[:, None, :]
    ).reshape(-1, 3)
    delta = super_positions[None, :, :] - positions[:, None, :]
    distances = jnp.linalg.norm(delta, axis=-1)
    return (distances > 1e-10) & (distances < r_cutoff)


def _compute_true_max_neighbors(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    r_cutoff: float,
    image_offsets: jnp.ndarray,
) -> jnp.ndarray:
    """Per-atom max neighbor count for one walker, geometry-only.

    Cheap enough to vmap over the full walker array at init time so the
    NS loop can start with a correctly-sized bucket and accurate
    per-walker ``max_neighbor_count``.
    """
    return _max_neighbor_count_from_mask(
        _neighbor_mask(positions, cell, r_cutoff, image_offsets)
    )


def _supercell_edges(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    r_cutoff: float,
    max_edges: int,
    image_offsets: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find edges within cutoff using supercell expansion. JIT-compatible.

    Args:
        positions: (N, 3) atom positions in the unit cell.
        cell: (3, 3) unit cell matrix (rows are lattice vectors).
        r_cutoff: Cutoff radius.
        max_edges: Static size for edge buffer.
        image_offsets: (sc_dim, 3) integer image offsets (static).

    Returns:
        senders: (max_edges,) sender indices in [0, N).
        receivers: (max_edges,) receiver indices in [0, N).
        shifts: (max_edges, 3) Cartesian shift vectors.
        n_actual: Scalar, actual number of edges found.
        overflow: Bool, True if n_actual > max_edges.
        true_max_per_atom: Scalar, max neighbor count per atom computed from
            the full mask (before truncation to ``max_edges``).  Safe to use
            for outer-loop overflow escalation.
    """
    n_atoms = positions.shape[0]
    sc_dim = image_offsets.shape[0]

    # Cartesian shift for each image: (sc_dim, 3).  Rebuilt below alongside
    # the mask; kept out of _neighbor_mask so it's available for edge shifts.
    cart_shifts = image_offsets @ cell

    # Edge mask: (N, sc_dim * N), within cutoff and not self-interaction.
    mask = _neighbor_mask(positions, cell, r_cutoff, image_offsets)

    # True max neighbor count per atom — derived from the full mask BEFORE
    # the flat-nonzero truncation below.  The outer NS loop escalates
    # max_neighbors based on this value; using the post-truncation count
    # would saturate at the current bucket and stall escalation.
    true_max_per_atom = _max_neighbor_count_from_mask(mask)

    # Static-shape edge extraction
    flat_mask = mask.ravel()
    n_total = n_atoms * sc_dim * n_atoms
    fill_val = n_total  # out-of-bounds sentinel
    flat_indices = jnp.nonzero(flat_mask, size=max_edges, fill_value=fill_val)[0]

    n_actual = jnp.sum(flat_mask)
    overflow = n_actual > max_edges

    # Decode flat indices -> (sender_in_unit_cell, supercell_atom_index)
    # flat index k maps to: sender = k // (sc_dim * N), j_sc = k % (sc_dim * N)
    sc_n = sc_dim * n_atoms
    senders = flat_indices // sc_n
    j_sc = flat_indices % sc_n

    # j_sc -> (image_index, receiver_in_unit_cell)
    receivers = j_sc % n_atoms
    image_idx = j_sc // n_atoms

    # Cartesian shifts for each edge
    shifts = cart_shifts[image_idx]

    # Ghost handling: sentinel indices get sender=receiver=N (ghost node), shift=0
    is_ghost = flat_indices >= n_total
    senders = jnp.where(is_ghost, n_atoms, senders)
    receivers = jnp.where(is_ghost, n_atoms, receivers)
    shifts = jnp.where(is_ghost[:, None], 0.0, shifts)

    return senders, receivers, shifts, n_actual, overflow, true_max_per_atom
