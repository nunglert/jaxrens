"""Affected-atom-set queries for local ("few atoms changed") move kernels.

Every local move (``single_atom_swap``, ``alchemical_morph``,
``single_atom``, ``single_atom_sweep``) recomputes its affected-atom set
fresh, in JAX, on every proposal, from the CURRENT ``state.positions``/
``state.cell`` — never from a cached/precomputed structure. This is
deliberate: an earlier version cached a neighbor table once from a fixed
reference geometry for the moves that don't move atoms (swap/morph), but
that table silently went stale the moment any OTHER move in the same run
touched positions or cell. Recomputing fresh is also deliberately NOT a
Verlet-style skin-buffered list rebuilt every few steps: the geometric
search itself (plain distance math, no neural-network forward pass) is
already cheap relative to the energy evaluation it enables skipping, so
the extra machinery of a skin margin + rebuild schedule + staleness
bookkeeping would add real bug surface (see the two rounds of periodic-
image bugs this module already went through, below) for a marginal speed
gain.

Two flavors, depending on whether the move itself moves the touched
atom(s):

- **Species-only** (:func:`local_affected_mask`): for moves that relabel
  species without moving positions (``single_atom_swap``,
  ``alchemical_morph``) — a single neighbor search against the current
  positions is enough, since the touched atom(s) don't move.
- **Position-changing** (:func:`dynamic_affected_mask`): for moves that
  move one atom (``single_atom``, ``single_atom_sweep``) — checks both the
  old and new position, since a neighbor can enter or leave the cutoff
  shell as the touched atom moves.

Both take a precomputed ``image_offsets`` array (see
:func:`compute_periodic_image_offsets`) for the periodic-image search —
see that function's docstring for why a plain single-nearest-image fold is
not sufficient in general.

All outputs are fixed-size and sentinel-padded (sentinel = ``n_atoms``, the
one-past-the-end "ghost" index, matching the convention in
``backends/_graph_neighbors.py``) so this is jit/vmap/scan compatible: no
data-dependent shapes anywhere.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _perpendicular_widths(cell: np.ndarray) -> np.ndarray:
    """Face-to-face (perpendicular) width of the cell along each lattice
    vector direction: ``volume / (area of the opposite face)``.

    This is the right quantity for "how many periodic repeats does a
    sphere of radius r_cutoff span along this direction" — the naive
    lattice-vector *length* overestimates that (a skewed/triclinic cell can
    be "long" along a vector direction but "narrow" perpendicular to the
    opposite face), while the perpendicular width is exactly the spacing
    between successive periodic copies of a plane normal to that face.
    """
    a, b, c = cell[0], cell[1], cell[2]
    volume = abs(np.linalg.det(cell))
    areas = np.array(
        [
            np.linalg.norm(np.cross(b, c)),
            np.linalg.norm(np.cross(a, c)),
            np.linalg.norm(np.cross(a, b)),
        ]
    )
    return np.where(
        areas > 1e-12, volume / np.where(areas > 1e-12, areas, 1.0), np.inf
    )


def compute_periodic_image_offsets(
    cell: np.ndarray, r_cutoff: float
) -> np.ndarray:
    """Host-side: integer periodic-image offsets guaranteed to cover every
    true neighbor within ``r_cutoff``, for a FIXED ``cell``.

    Shared by the static table builder and the dynamic per-step search —
    both need "how many periodic repeats can a neighbor be at most this
    many cells away" for the same ``cell``/``r_cutoff`` pair, computed the
    same (correct, ratio-independent) way. See :func:`_perpendicular_widths`
    for why perpendicular width (not raw lattice-vector length) is the
    right denominator.

    Returns:
        ``(n_images, 3)`` int32 array of integer lattice-vector multiples.
        ``[[0, 0, 0]]`` for a non-periodic (degenerate/zero) cell.
    """
    cell = np.asarray(cell, dtype=np.float64)
    if abs(np.linalg.det(cell)) <= 1e-10:
        return np.zeros((1, 3), dtype=np.int32)

    perp_widths = _perpendicular_widths(cell)
    # +1 for safety margin against floating-point edge cases at an exact
    # multiple of the perpendicular width.
    n_images = np.ceil(r_cutoff / perp_widths).astype(int) + 1
    na, nb, nc = (int(n) for n in n_images)
    ia = np.arange(-na, na + 1)
    ib = np.arange(-nb, nb + 1)
    ic = np.arange(-nc, nc + 1)
    return (
        np.stack(np.meshgrid(ia, ib, ic, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        .astype(np.int32)
    )


def initial_image_bucket_for_cell(cell: np.ndarray, r_cutoff: float) -> int:
    """Host-side: a reasonable STARTING value for the symmetric per-axis
    image-bucket ladder (see :func:`build_symmetric_image_offsets` and
    ``sampling/bucket_manager.py``), for a single FIXED ``cell``.

    Unlike :func:`compute_periodic_image_offsets` (which returns a
    possibly-anisotropic per-axis image count, exact for that one cell),
    the runtime bucket-ladder mechanism uses a single symmetric half-width
    for all three axes — so the right starting point is the MAX over the
    three axes' individual requirements, guaranteeing no axis is
    under-covered even though some axes may end up over-covered (cheap:
    extra candidate images are filtered out by the cutoff check, not a
    correctness risk). Only used to pick a sensible ladder entry at
    resolve time — the overflow-and-retry mechanism keeps this correct
    for the whole run even if the cell later changes (a volume move, or
    per-walker cell diversity), so this value never needs to be exact.

    Returns 0 for a non-periodic (degenerate/zero) cell.
    """
    cell = np.asarray(cell, dtype=np.float64)
    if abs(np.linalg.det(cell)) <= 1e-10:
        return 0
    perp_widths = _perpendicular_widths(cell)
    n_per_axis = np.ceil(r_cutoff / perp_widths).astype(int) + 1
    return int(n_per_axis.max())


def perpendicular_widths_jax(cell: jnp.ndarray) -> jnp.ndarray:
    """JIT-compatible twin of :func:`_perpendicular_widths`, for use INSIDE
    move kernels (where ``cell`` is a traced array, not a host numpy array
    known at kernel-build time) — see :func:`image_count_needed_for`."""
    a, b, c = cell[0], cell[1], cell[2]
    volume = jnp.abs(jnp.linalg.det(cell))
    areas = jnp.array(
        [
            jnp.linalg.norm(jnp.cross(b, c)),
            jnp.linalg.norm(jnp.cross(a, c)),
            jnp.linalg.norm(jnp.cross(a, b)),
        ]
    )
    return jnp.where(
        areas > 1e-12, volume / jnp.where(areas > 1e-12, areas, 1.0), jnp.inf
    )


def image_count_needed_for(cell: jnp.ndarray, r_cutoff: float) -> jnp.ndarray:
    """JIT-compatible: scalar int32, the TRUE symmetric per-axis image
    half-width needed to cover ``r_cutoff`` for the CURRENT ``cell`` — the
    max over the three axes' individual (possibly anisotropic)
    requirements (see :func:`initial_image_bucket_for_cell` for why max,
    not per-axis).

    Called fresh, every proposal, by every local-update move kernel, and
    compared against the static ``state.image_bucket`` to detect whether
    the currently-compiled :func:`build_symmetric_image_offsets` array is
    still large enough for whatever the cell is right now (it may have
    changed since the run started — a volume move, or per-walker cell
    diversity at init). See ``sampling/bucket_manager.py`` for how a
    "too small" result here triggers a recompile-and-retry, not a reject.

    Returns 0 for a non-periodic (degenerate/zero) cell.
    """
    degenerate = jnp.abs(jnp.linalg.det(cell)) <= 1e-10
    perp_widths = perpendicular_widths_jax(cell)
    n_per_axis = jnp.ceil(r_cutoff / perp_widths).astype(jnp.int32) + 1
    return jnp.where(degenerate, jnp.int32(0), jnp.max(n_per_axis))


def build_symmetric_image_offsets(n: int) -> jnp.ndarray:
    """Build the ``((2n+1)**3, 3)`` periodic-image offset array for a
    STATIC, symmetric per-axis half-width ``n`` (same half-width on all
    three axes) — ``n`` must be a Python int known at trace time (e.g.
    ``state.image_bucket``, a static ``MCState`` field), not a traced
    value, since it determines the output's shape.

    This is the runtime counterpart to :func:`compute_periodic_image_offsets`:
    that function computes the exact (possibly anisotropic) image count
    for one FIXED reference cell, host-side, once; this one builds a
    (possibly more generous, always symmetric) array from whatever the
    CURRENT bucket happens to be, cheaply, inside JIT, on every proposal
    — paired with :func:`image_count_needed_for` to detect when the
    bucket itself needs to grow.
    """
    r = jnp.arange(-n, n + 1)
    return jnp.stack(jnp.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def local_affected_mask(
    touched_idx: jnp.ndarray,
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    r_cutoff: float,
    image_offsets: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compatible boolean ``(n_atoms,)`` affected-set mask for moves
    that relabel species WITHOUT moving positions (``single_atom_swap``,
    ``alchemical_morph``).

    Recomputed fresh on every proposal from CURRENT ``positions``/``cell``
    — see module docstring for why. Unlike :func:`dynamic_affected_mask`,
    there is no "before/after" distinction to check: the touched atom(s)
    don't move, so a single neighbor search against the current positions
    is enough. ``touched_idx`` is a small fixed-size array of the atom
    indices whose species the move may change (shape ``(2,)`` for swap,
    ``(1,)`` for morph) — every touched atom's own index is always
    included, plus every atom within ``r_cutoff`` of ANY touched atom
    (checked across all ``image_offsets`` periodic images). A duplicated
    entry in ``touched_idx`` is harmless (mask semantics, not a list).

    ``image_offsets`` must be precomputed (see
    :func:`compute_periodic_image_offsets`) and passed in as a static-shape
    array.
    """
    n_atoms = positions.shape[0]
    cart_shifts = image_offsets @ cell  # (n_images, 3)
    shifted = (
        positions[None, :, :] + cart_shifts[:, None, :]
    )  # (n_images, n_atoms, 3)

    query_positions = positions[touched_idx]  # (K, 3)
    delta = (
        query_positions[:, None, None, :] - shifted[None, :, :, :]
    )  # (K, n_images, n_atoms, 3)
    dist = jnp.linalg.norm(delta, axis=-1)  # (K, n_images, n_atoms)
    within_cutoff = (dist > 1e-10) & (dist < r_cutoff)
    mask = jnp.any(within_cutoff, axis=(0, 1))  # (n_atoms,)
    mask = mask.at[touched_idx].set(True)
    return mask


def affected_indices_from_mask(
    mask: jnp.ndarray,
    max_affected: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract a fixed-size, sentinel-padded index array from a boolean mask.

    Mirrors the existing ``jnp.nonzero(flat_mask, size=max_edges,
    fill_value=...)`` pattern in ``backends/_graph_neighbors.py::_supercell_edges``
    — same style, same overflow semantics (silent truncation if the true
    count exceeds ``max_affected``, reported via the returned ``overflow``
    flag for the caller to fold into the existing overflow/escalation
    control plane).

    Returns:
        (indices, overflow): ``indices`` is ``(max_affected,)`` int32,
        sentinel ``n_atoms`` (= ``mask.shape[0]``) for unused slots.
        ``overflow`` is a scalar bool, True if the true affected count
        exceeded ``max_affected``.
    """
    n_atoms = mask.shape[0]
    n_actual = jnp.sum(mask)
    overflow = n_actual > max_affected
    indices = jnp.nonzero(mask, size=max_affected, fill_value=n_atoms)[
        0
    ].astype(jnp.int32)
    return indices, overflow


def dynamic_affected_mask(
    atom_idx: jnp.ndarray,
    positions_before: jnp.ndarray,
    positions_after: jnp.ndarray,
    cell: jnp.ndarray,
    r_cutoff: float,
    image_offsets: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compatible boolean ``(n_atoms,)`` affected-set mask for a single
    moved atom (``single_atom``, and per-iteration inside
    ``single_atom_sweep``'s scan).

    Recomputed fresh on every proposal — see this module's docstring for
    why (no Verlet-style skin buffer/rebuild schedule). The affected set is
    the touched atom itself, plus everyone within ``r_cutoff`` of it under
    EITHER its old or new position — an atom can enter or leave the cutoff
    shell as the touched atom moves, so both must be checked, not just one.

    ``positions_before``/``positions_after`` differ only at ``atom_idx``;
    both full ``(n_atoms, 3)`` arrays are passed (rather than just the
    single moved position) because every *other* atom's position is needed
    as the neighbor-search reference set.

    ``image_offsets`` must be precomputed host-side via
    :func:`compute_periodic_image_offsets` from the (fixed — this move
    never mutates "cell") reference cell and ``r_cutoff``, and passed in as
    a static-shape array; see that function's docstring for why a plain
    single-nearest-image fold is not sufficient in general.
    """
    n_atoms = positions_before.shape[0]
    cart_shifts = image_offsets @ cell  # (n_images, 3)

    def within_cutoff(
        query_pos: jnp.ndarray, ref_positions: jnp.ndarray
    ) -> jnp.ndarray:
        shifted = (
            ref_positions[None, :, :] + cart_shifts[:, None, :]
        )  # (n_images, n_atoms, 3)
        delta = query_pos[None, None, :] - shifted
        dist = jnp.linalg.norm(delta, axis=-1)  # (n_images, n_atoms)
        return jnp.any(
            (dist > 1e-10) & (dist < r_cutoff), axis=0
        )  # (n_atoms,)

    mask_old = within_cutoff(positions_before[atom_idx], positions_before)
    mask_new = within_cutoff(positions_after[atom_idx], positions_after)
    mask = mask_old | mask_new
    mask = mask.at[atom_idx].set(True)
    return mask
