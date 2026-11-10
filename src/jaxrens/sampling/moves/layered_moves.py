"""Moves for layered hosts (e.g. intercalated graphite).

The host atoms (C in graphite) form flat sheets parallel to the (a, b) plane
of the cell.  Sheets are identified once per walker, when the state is built,
and stored in the ``sheet_id`` state field; galleries (the space between two
consecutive sheets) are recomputed from the sheet heights on every call.

Heights are fractional coordinates along the sheet normal,
``s3 = (r · n) / h mod 1`` with ``n = a × b / |a × b|`` and ``h = c · n``.
They depend only on the component of ``c`` along the normal, so they are
unchanged by in-plane translations and by tilting ``c`` within the plane.

- fractional_heights: s3 of every atom
- make_sheet_id_initializer: ``sheet_id`` state-field initializer
- sheet_heights: circular mean height of each sheet
- galleries: lower bound, width and centre of each gallery
- gallery_of: gallery containing each atom
- check_sheets: resolver-side validation of the starting walkers
- build_gallery_hop_kernel: move one atom to another gallery or site
- build_layer_slide_kernel: slide one sheet, or one gallery's atoms, in plane
- build_stacking_shift_kernel: change the registry across one gallery by
  tilting c
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxrens.constraints.cell_geometry import build_cell_geometry
from jaxrens.sampling.base import MoveInfo
from jaxrens.sampling.moves._common import finalize_cell_move

SHEET_GAP = 1.0
"""Minimum separation (Angstrom) along the normal between two sheets."""

PLANARITY_TOL = 0.1
"""Maximum distance (Angstrom) of a host atom from its sheet at start."""


def _normal_and_height(cell):
    """Unit normal to the (a, b) plane and the period ``h = c · n``."""
    a, b, c = cell[0], cell[1], cell[2]
    n = jnp.cross(a, b)
    n = n / jnp.linalg.norm(n)
    return n, jnp.dot(c, n)


def fractional_heights(positions, cell):
    """Fractional height of every atom along the sheet normal, in [0, 1).

    Positions need not be wrapped into the cell.
    """
    n, h = _normal_and_height(cell)
    return jnp.mod(positions @ n / h, 1.0)


def make_sheet_id_initializer(host_codes: tuple[int, ...]) -> Callable:
    """Build the initializer of the ``sheet_id`` state field.

    Host atoms are sorted by height and split into sheets wherever the gap
    to the previous host atom exceeds ``SHEET_GAP``.  A sheet straddling
    ``s3 = 0`` is merged into sheet 0.  Sheets are numbered 0..n-1 by
    increasing height; non-host atoms get -1.

    Args:
        host_codes: Type codes of the atoms forming the sheets.

    Returns:
        ``init(positions, types, cell) -> sheet_id`` (int32, one per atom).
        Uses only fixed-shape ``jnp`` operations.
    """
    codes = tuple(host_codes)

    def init(positions, types, cell):
        cell = jnp.asarray(cell)
        _, h = _normal_and_height(cell)
        s3 = fractional_heights(positions, cell)
        is_host = jnp.isin(types, jnp.asarray(codes, types.dtype))
        n_host = jnp.sum(is_host)

        # Host atoms by increasing height, the others at the end.
        key = jnp.where(is_host, s3, 2.0)
        order = jnp.argsort(key)
        z = key[order]
        idx = jnp.arange(z.shape[0])
        in_host = idx < n_host

        # A new sheet starts wherever the gap to the previous one is large.
        gap_before = jnp.diff(z, prepend=z[0])
        new_sheet = in_host & (idx > 0) & (gap_before * jnp.abs(h) > SHEET_GAP)
        label = jnp.cumsum(new_sheet)

        # A sheet straddling s3 = 0 shows up as the first and last group.
        n_groups = jnp.max(jnp.where(in_host, label, 0)) + 1
        wrap_gap = z[0] + 1.0 - z[n_host - 1]
        merge = (wrap_gap * jnp.abs(h) <= SHEET_GAP) & (n_groups > 1)
        label = jnp.where(merge & (label == n_groups - 1), 0, label)

        label = jnp.where(in_host, label, -1).astype(jnp.int32)
        return jnp.full(types.shape, -1, jnp.int32).at[order].set(label)

    return init


def sheet_heights(s3, sheet_id, n_sheets: int):
    """Circular mean height of each sheet, in [0, 1).

    Args:
        s3: Fractional heights of all atoms, shape ``(n_atoms,)``.
        sheet_id: Sheet of every atom, -1 for non-host atoms.
        n_sheets: Number of sheets (static).

    Returns:
        Array of shape ``(n_sheets,)``.
    """
    angle = 2.0 * jnp.pi * s3
    # Non-host atoms go to an extra segment that is dropped.
    seg = jnp.where(sheet_id >= 0, sheet_id, n_sheets)
    cos = jax.ops.segment_sum(jnp.cos(angle), seg, num_segments=n_sheets + 1)
    sin = jax.ops.segment_sum(jnp.sin(angle), seg, num_segments=n_sheets + 1)
    mean = jnp.arctan2(sin[:n_sheets], cos[:n_sheets]) / (2.0 * jnp.pi)
    return jnp.mod(mean, 1.0)


def galleries(heights):
    """Galleries between consecutive sheets, cyclically.

    Gallery ``g`` runs upward from sheet ``g`` to sheet ``g + 1`` (sheet 0
    for the last one).

    Returns:
        ``(lower, width, centre)``, fractional, each of shape
        ``(n_sheets,)``.
    """
    lower = heights
    width = jnp.mod(jnp.roll(heights, -1) - heights, 1.0)
    centre = jnp.mod(lower + 0.5 * width, 1.0)
    return lower, width, centre


def gallery_of(s3, lower, width):
    """Index of the gallery containing each height in ``s3``."""
    d = jnp.mod(s3[:, None] - lower[None, :], 1.0)
    return jnp.argmax(d < width[None, :], axis=1)


@dataclass(frozen=True)
class SheetInfo:
    """Sheet layout shared by all starting walkers."""

    n_sheets: int
    sheet_size: int


def check_sheets(
    positions, types, cells, host_codes: tuple[int, ...]
) -> SheetInfo:
    """Validate the sheets of the starting walkers.

    Args:
        positions: Shape ``(..., n_atoms, 3)``; leading axes (walkers,
            replicas) are flattened.
        types: Shape ``(..., n_atoms)``, broadcast against ``positions``.
        cells: Shape ``(..., 3, 3)``, or ``None`` for a non-periodic system.
        host_codes: Type codes of the atoms forming the sheets.

    Returns:
        The number of sheets and the number of host atoms per sheet, the
        same in every walker.

    Raises:
        ValueError: If a cell is not periodic in the (a, b) plane, a walker
            has fewer than two sheets or sheets of different sizes, the
            number of sheets differs between walkers, or a sheet is not
            flat within ``PLANARITY_TOL``.
    """
    positions = np.asarray(positions)
    n_atoms = positions.shape[-2]
    positions = positions.reshape(-1, n_atoms, 3)
    types = np.broadcast_to(np.asarray(types), positions.shape[:-1]).reshape(
        -1, n_atoms
    )
    if cells is None:
        cells = np.zeros((3, 3))
    cells = np.asarray(cells).reshape(-1, 3, 3)
    cells = np.broadcast_to(cells, (positions.shape[0], 3, 3))
    init = jax.jit(make_sheet_id_initializer(host_codes))
    heights = jax.jit(fractional_heights)
    n_ref = None
    for k in range(positions.shape[0]):
        cell = cells[k]
        if np.linalg.norm(np.cross(cell[0], cell[1])) < 1e-8:
            raise ValueError(
                f"walker {k}: layered moves need a cell that is periodic in "
                "the sheet plane (a x b != 0)."
            )
        _, h = _normal_and_height(cell)
        s3 = np.asarray(heights(positions[k], cell))
        sid = np.asarray(init(positions[k], types[k], cell))
        host = sid >= 0
        n = int(sid.max()) + 1
        if n < 2:
            raise ValueError(
                f"walker {k}: found {n} sheet(s) of host atoms; layered "
                "moves need at least 2, parallel to the (a, b) plane of "
                "the cell."
            )
        sizes = np.bincount(sid[host], minlength=n)
        if len(set(sizes.tolist())) > 1:
            raise ValueError(
                f"walker {k}: sheets have different numbers of host atoms "
                f"{sizes.tolist()}."
            )
        if n_ref is None:
            n_ref = n
        elif n != n_ref:
            raise ValueError(
                f"walker {k}: {n} sheets, but walker 0 has {n_ref}. All "
                "starting walkers must have the same number of sheets."
            )
        m = np.asarray(sheet_heights(s3, sid, n))
        dev = (s3[host] - m[sid[host]] + 0.5) % 1.0 - 0.5
        worst = float(np.max(np.abs(dev)) * abs(float(h)))
        if worst > PLANARITY_TOL:
            raise ValueError(
                f"walker {k}: a sheet is not flat within {PLANARITY_TOL} A "
                f"(largest deviation {worst:.3f} A). Sheets must lie "
                "parallel to the (a, b) plane of the cell."
            )
    return SheetInfo(n_sheets=n_ref, sheet_size=int(sizes[0]))


def build_gallery_hop_kernel(
    backend: Any,
    species: tuple[int, ...],
    host_codes: tuple[int, ...],
    n_sheets: int,
    in_plane_repeats: tuple[int, int],
):
    """Build a kernel moving one atom to another gallery or in-plane site.

    One atom of ``species`` is picked uniformly.  A hop ``(dg, i, j)`` is
    drawn uniformly from ``{0..n_sheets-1} x {0..m_a-1} x {0..m_b-1}``
    without ``(0, 0, 0)``, and the atom is displaced by

    * ``(centre[g1] - centre[g0]) * c``: from the centre of its gallery
      ``g0`` to the centre of ``g1 = g0 + dg``, keeping its offset from the
      centre and its in-plane fractional coordinates;
    * ``i * a / m_a + j * b / m_b``: a host lattice vector, hollow to hollow;
    * ``step_size * N(0, 1)`` in 3D.

    The proposal is rejected (reason 3) if the walker has no atom of
    ``species`` or if the noise moved the atom out of ``g1``; the latter
    keeps the proposal symmetric, since the reverse hop starts from ``g1``.
    Sheets do not move, so gallery centres are the same before and after.

    Args:
        backend: EnergyBackend instance.
        species: Type codes of the atoms that may hop.
        host_codes: Type codes of the sheet atoms (used by ``sheet_id``).
        n_sheets: Number of sheets, from ``check_sheets``.
        in_plane_repeats: ``(m_a, m_b)``, host primitive cells along a, b.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo).
        The state must carry the ``sheet_id`` field.
    """
    del host_codes  # carried for the descriptor; sheet_id already uses it
    m_a, m_b = in_plane_repeats
    n_sites = m_a * m_b
    n_hops = n_sheets * n_sites

    def step(rng_key, state, likelihood_constraint):
        k_atom, k_hop, k_noise = jax.random.split(rng_key, 3)
        cell = state.cell

        codes = jnp.asarray(species, dtype=state.types.dtype)
        mask = jnp.isin(state.types, codes)
        atom = jax.random.categorical(k_atom, jnp.where(mask, 0.0, -jnp.inf))
        has_atom = jnp.any(mask)

        s3 = fractional_heights(state.positions, cell)
        heights = sheet_heights(s3, state.sheet_id, n_sheets)
        lower, width, centre = galleries(heights)
        g0 = gallery_of(s3[atom][None], lower, width)[0]

        hop = jax.random.randint(k_hop, (), 1, n_hops)
        dg, site = hop // n_sites, hop % n_sites
        i, j = site // m_b, site % m_b
        g1 = (g0 + dg) % n_sheets

        disp = (
            (centre[g1] - centre[g0]) * cell[2]
            + i * cell[0] / m_a
            + j * cell[1] / m_b
            + state.step_size * jax.random.normal(k_noise, (3,))
        )
        new_positions = state.positions.at[atom].add(disp)

        s3_new = fractional_heights(new_positions[atom][None], cell)
        arrived = gallery_of(s3_new, lower, width)[0] == g1

        result = backend(
            new_positions,
            state.types,
            cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        valid = has_atom & arrived
        accepted = valid & (result.energy < likelihood_constraint)

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, result.energy, state.energy),
            max_neighbor_count=jnp.maximum(
                state.max_neighbor_count, result.max_neighbor_count
            ),
            overflow=state.overflow | result.overflow,
        )
        reason = jnp.where(accepted, 0, jnp.where(valid, 1, 3))
        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
            reject_reason=reason.astype(jnp.int32),
        )
        return new_state, info

    return step


def build_layer_slide_kernel(
    backend: Any,
    species: tuple[int, ...],
    host_codes: tuple[int, ...],
    n_sheets: int,
    in_plane_repeats: tuple[int, int],
    subdivisions: int,
    slide_host: bool,
):
    """Build a kernel translating one sheet, or one gallery's atoms, in plane.

    A group is picked uniformly among the ``n_sheets`` sheets
    (``slide_host``) or galleries (otherwise) and translated rigidly by

        t = i a / (m_a k) + j b / (m_b k) + step_size (e0 u + e1 v)

    with ``(i, j)`` uniform on ``{0..m_a k - 1} x {0..m_b k - 1}`` without
    ``(0, 0)``, ``k = subdivisions``, and ``u, v`` an orthonormal basis of
    the sheet plane.  The translation is in plane, so no height changes:
    the sheet (fixed by ``sheet_id``) or the gallery's content is the same
    before and after, and the proposal is symmetric.  An empty gallery
    rejects the proposal (reason 3).

    Args:
        backend: EnergyBackend instance.
        species: Type codes of the atoms that slide.  For ``slide_host``
            these are the host codes and the group is given by ``sheet_id``.
        host_codes: Type codes of the sheet atoms (used by ``sheet_id``).
        n_sheets: Number of sheets, from ``check_sheets``.
        in_plane_repeats: ``(m_a, m_b)``, host primitive cells along a, b.
        subdivisions: Grid steps ``k`` per host lattice vector.
        slide_host: Slide a sheet (True) or a gallery's atoms (False).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo).
        The state must carry the ``sheet_id`` field.
    """
    del host_codes  # carried for the descriptor; sheet_id already uses it
    m_a, m_b = in_plane_repeats
    n_a, n_b = m_a * subdivisions, m_b * subdivisions
    n_grid = n_a * n_b

    def step(rng_key, state, likelihood_constraint):
        k_group, k_t, k_noise = jax.random.split(rng_key, 3)
        cell, positions = state.cell, state.positions

        group = jax.random.randint(k_group, (), 0, n_sheets)
        if slide_host:
            members = state.sheet_id == group
            valid = True
        else:
            s3 = fractional_heights(positions, cell)
            heights = sheet_heights(s3, state.sheet_id, n_sheets)
            lower, width, _ = galleries(heights)
            codes = jnp.asarray(species, dtype=state.types.dtype)
            members = jnp.isin(state.types, codes) & (
                gallery_of(s3, lower, width) == group
            )
            valid = jnp.any(members)

        t_idx = jax.random.randint(k_t, (), 1, n_grid)
        i, j = t_idx // n_b, t_idx % n_b
        n_hat, _ = _normal_and_height(cell)
        u = cell[0] / jnp.linalg.norm(cell[0])
        v = jnp.cross(n_hat, u)
        eps = jax.random.normal(k_noise, (2,))
        t = (
            i * cell[0] / n_a
            + j * cell[1] / n_b
            + state.step_size * (eps[0] * u + eps[1] * v)
        )
        new_positions = positions + members[:, None] * t

        result = backend(
            new_positions,
            state.types,
            cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        accepted = valid & (result.energy < likelihood_constraint)

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, positions),
            energy=jnp.where(accepted, result.energy, state.energy),
            max_neighbor_count=jnp.maximum(
                state.max_neighbor_count, result.max_neighbor_count
            ),
            overflow=state.overflow | result.overflow,
        )
        reason = jnp.where(accepted, 0, jnp.where(valid, 1, 3))
        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
            reject_reason=reason.astype(jnp.int32),
        )
        return new_state, info

    return step


# Centred grid of in-plane shifts for stacking_shift: (i, j) in {-1, 0, 1}^2
# without (0, 0), so that every shift's opposite is also on the grid.
_SHIFTS = np.array(
    [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
)


def build_stacking_shift_kernel(
    backend: Any,
    host_codes: tuple[int, ...],
    n_sheets: int,
    in_plane_repeats: tuple[int, int],
    subdivisions: int,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
    restrict_c_in_plane: bool = True,
):
    """Build a kernel changing the stacking registry across one gallery.

    1. Pick a gallery ``g`` uniformly; the cut is at its centre.
    2. Move every atom by an integer multiple of ``c`` so that its height is
       in ``[cut, cut + 1)``: the cell boundary now lies in gallery ``g``.
       This only changes the representation of the configuration.
    3. Tilt the cell, ``c' = c + t``, keeping the atoms fixed in Cartesian
       space, with ``t = i a / (m_a k) + j b / (m_b k)`` plus in-plane
       Gaussian noise of ``step_size``, ``(i, j)`` uniform in
       ``{-1, 0, 1}^2`` without ``(0, 0)``.  The periodic images above the
       boundary shift by ``t`` relative to the atoms below it, so only the
       registry across gallery ``g`` changes.

    The volume and every height are unchanged (``t`` is in plane), so the
    reverse move sees the same galleries and cut, and undoes the tilt with
    ``-t``, which the centred grid contains.  The map has unit Jacobian in
    the cell and fractional coordinates, so the prior measure is preserved.
    Rejection follows the cell moves: energy (1) or cell geometry (2).

    Args:
        backend: EnergyBackend instance.
        host_codes: Type codes of the sheet atoms (used by ``sheet_id``).
        n_sheets: Number of sheets, from ``check_sheets``.
        in_plane_repeats: ``(m_a, m_b)``, host primitive cells along a, b.
        subdivisions: Grid steps ``k`` per host lattice vector: 1 for host
            lattice vectors, 3 for the AA/AB/AC shifts.
        n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect,
            restrict_c_in_plane: Cell-geometry guard, as for the other cell
            moves.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo).
        The state must carry the ``sheet_id`` field.
    """
    del host_codes  # carried for the descriptor; sheet_id already uses it
    m_a, m_b = in_plane_repeats
    n_a, n_b = m_a * subdivisions, m_b * subdivisions
    shifts = jnp.asarray(_SHIFTS)
    cell_geometry = build_cell_geometry(
        n_atoms,
        max_vol_per_atom,
        min_vol_per_atom,
        min_aspect,
        restrict_c_in_plane,
    )

    def step(rng_key, state, likelihood_constraint):
        k_g, k_t, k_noise = jax.random.split(rng_key, 3)
        cell, positions = state.cell, state.positions

        s3 = fractional_heights(positions, cell)
        heights = sheet_heights(s3, state.sheet_id, n_sheets)
        _, _, centre = galleries(heights)
        cut = centre[jax.random.randint(k_g, (), 0, n_sheets)]

        n_hat, h = _normal_and_height(cell)
        shift = jnp.floor(positions @ n_hat / h - cut)
        wrapped = positions - shift[:, None] * cell[2]

        i, j = shifts[jax.random.randint(k_t, (), 0, shifts.shape[0])]
        u = cell[0] / jnp.linalg.norm(cell[0])
        v = jnp.cross(n_hat, u)
        eps = jax.random.normal(k_noise, (2,))
        t = (
            i * cell[0] / n_a
            + j * cell[1] / n_b
            + state.step_size * (eps[0] * u + eps[1] * v)
        )
        new_cell = cell.at[2].add(t)

        result = backend(
            wrapped,
            state.types,
            new_cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        cell_valid = cell_geometry(wrapped, state.types, new_cell)
        return finalize_cell_move(
            state,
            wrapped,
            new_cell,
            result.energy,
            result.max_neighbor_count,
            result.overflow,
            cell_valid,
            result.energy < likelihood_constraint,
        )

    return step
