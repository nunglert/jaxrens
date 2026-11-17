"""Structural observables of one configuration, in NumPy.

Each function maps one configuration to a dict ``name -> value``, where a
value is a float or a 1-D array (one entry per gallery).  Evaluated on
every point of a run, each name gives the per-point array that
:meth:`jaxrens.postprocess.Monitor.expectation` and friends average.

Lengths are in Angstrom, volumes in Angstrom^3, angles in degrees.

- cell_observables: lattice parameters and volume
- gallery_order: order of the galleries in every per-gallery observable
- layer_observables: gallery widths and occupancies of a layered host
- registry_observables: stacking registry across each gallery
- li_order_observables: sqrt(3) x sqrt(3) order of the intercalant
"""

from __future__ import annotations

import numpy as np

from jaxrens.postprocess.layers import (
    LayerGeometry,
    normal_and_height,
    unwrap_to_height,
)


def cell_observables(
    positions, cell, *, in_plane_repeats: tuple[int, int] = (3, 3)
) -> dict[str, float]:
    """Lattice parameters of one configuration.

    Args:
        positions: Shape ``(n_atoms, 3)``, only the number of atoms is used.
        cell: Shape ``(3, 3)``, cell vectors as rows; periodic in the
            (a, b) plane.
        in_plane_repeats: Primitive cells of the host along a and b.

    Returns:
        ``a_eq``, ``b_eq``: ``|a| / m_a`` and ``|b| / m_b``;
        ``gamma``: angle between a and b;
        ``h``: period along the sheet normal, ``c · n``;
        ``c_shift``: length of the in-plane component of c;
        ``volume_per_atom``: ``V / N``.
    """
    cell = np.asarray(cell, dtype=float)
    a, b, c = cell
    m_a, m_b = in_plane_repeats
    n, h = normal_and_height(cell)
    cos_gamma = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "a_eq": float(np.linalg.norm(a) / m_a),
        "b_eq": float(np.linalg.norm(b) / m_b),
        "gamma": float(np.degrees(np.arccos(np.clip(cos_gamma, -1, 1)))),
        "h": h,
        "c_shift": float(np.linalg.norm(c - (c @ n) * n)),
        "volume_per_atom": float(
            abs(np.linalg.det(cell)) / len(np.asarray(positions))
        ),
    }


def gallery_occupancy(geo: LayerGeometry, intercalant) -> np.ndarray:
    """Number of intercalated atoms in each gallery, ``(n_galleries,)``.

    Args:
        geo: Sheets and galleries, from
            :func:`jaxrens.postprocess.layers.layer_geometry`.
        intercalant: Boolean mask of the intercalated atoms (Li),
            shape ``(n_atoms,)``.
    """
    gal = geo.gallery[np.asarray(intercalant, dtype=bool)]
    if (gal < 0).any():
        raise ValueError("an intercalated atom is also a host atom.")
    return np.bincount(gal, minlength=geo.n_sheets)


def gallery_order(geo: LayerGeometry, intercalant) -> np.ndarray:
    """Order in which every per-gallery observable lists the galleries.

    Galleries have no identity across configurations (which one is
    "gallery 0" is arbitrary), so they are sorted by occupancy, largest
    first, and by width, largest first, among galleries with the same
    occupancy: the first one is always the fullest gallery.

    Returns:
        Permutation of the gallery indices, ``(n_galleries,)``.
    """
    occupancy = gallery_occupancy(geo, intercalant)
    return np.lexsort((-geo.width, -occupancy))


def layer_observables(geo: LayerGeometry, intercalant) -> dict:
    """Gallery widths and occupancies of one configuration.

    Per-gallery vectors are in :func:`gallery_order`.

    Args:
        geo: Sheets and galleries, from
            :func:`jaxrens.postprocess.layers.layer_geometry`.
        intercalant: Boolean mask of the intercalated atoms (Li),
            shape ``(n_atoms,)``.

    Returns:
        ``gallery_width``: width of each gallery, ``(n_galleries,)``;
        ``gallery_occupancy``: intercalated atoms in each gallery,
        ``(n_galleries,)``;
        ``n_empty``: number of galleries without intercalated atoms;
        ``d_occupied``: mean width of the galleries with intercalated
        atoms (NaN if none);
        ``d_empty``: mean width of the empty galleries (NaN if none).
    """
    order = gallery_order(geo, intercalant)
    occupancy = gallery_occupancy(geo, intercalant)[order]
    width = (geo.width * abs(geo.height))[order]
    empty = occupancy == 0
    return {
        "gallery_width": width,
        "gallery_occupancy": occupancy,
        "n_empty": int(empty.sum()),
        "d_occupied": _mean_or_nan(width[~empty]),
        "d_empty": _mean_or_nan(width[empty]),
    }


def _mean_or_nan(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


AA, AB = 0, 1
"""Labels of ``stacking``: AB also covers AC (the opposite shift)."""


def registry_observables(
    positions,
    cell,
    geo: LayerGeometry,
    intercalant,
    *,
    in_plane_repeats: tuple[int, int] = (3, 3),
) -> dict:
    """Stacking registry of the two sheets bounding each gallery.

    The in-plane shift ``dt`` of the upper sheet relative to the lower one
    is read from the phases of the sheets' structure factors
    ``S(G) = sum_j exp(i G . r_j)`` at two first-shell reciprocal vectors
    ``G1, G2`` of the host lattice: a rigid shift ``t`` rotates ``S(G)`` by
    ``G . t``, so ``G_k . dt`` is the phase of ``S_up(G_k) S_low(G_k)*``.
    ``dt`` is known modulo the host lattice and reduced to its shortest
    image.  AA is ``dt = 0``; AB and AC are the two points with
    ``G1 . dt = G2 . dt = +-120 deg``, one C-C bond away.

    The host lattice is taken from the cell, ``a / m_a`` and ``b / m_b``:
    the cell must be an ``m_a x m_b`` supercell of the host lattice.  The
    upper sheet of the gallery that crosses the cell boundary is the image
    shifted by ``c``.

    Args:
        positions: Shape ``(n_atoms, 3)``; need not be wrapped.
        cell: Shape ``(3, 3)``, cell vectors as rows.
        geo: Sheets and galleries of this configuration.
        intercalant: Boolean mask of the intercalated atoms, only used for
            :func:`gallery_order`.
        in_plane_repeats: Primitive cells of the host along a and b.

    Returns:
        ``stacking_shift``: ``|dt|`` of each gallery (Angstrom), 0 for AA
        and one C-C bond (1.42 for graphite) for AB/AC;
        ``stacking``: nearest high-symmetry registry of each gallery,
        ``AA`` (0) or ``AB`` (1, either AB or AC).
        Both ``(n_galleries,)``, in :func:`gallery_order`.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    g1, g2, e = _host_reciprocal(cell, geo.normal, in_plane_repeats)
    gs = np.stack([g1, g2])
    shift = np.empty(geo.n_sheets)
    label = np.empty(geo.n_sheets, dtype=int)
    for g in range(geo.n_sheets):
        up = (g + 1) % geo.n_sheets
        top = geo.lower[g] + geo.width[g]
        s_low = _structure_factor(positions, cell, geo, g, geo.heights[g], gs)
        s_up = _structure_factor(positions, cell, geo, up, top, gs)
        frac = np.angle(s_up * np.conj(s_low)) / (2.0 * np.pi)
        dist = [
            _min_image(frac - ref, e)
            for ref in (np.zeros(2), np.full(2, 1 / 3), np.full(2, -1 / 3))
        ]
        shift[g] = dist[0]
        label[g] = AA if np.argmin(dist) == 0 else AB
    order = gallery_order(geo, intercalant)
    return {"stacking_shift": shift[order], "stacking": label[order]}


ALIGNED, SHIFTED = 0, 1
"""Labels of ``li_stacking``: intercalant above intercalant, or shifted."""


def li_order_observables(
    positions,
    cell,
    geo: LayerGeometry,
    intercalant,
    *,
    in_plane_repeats: tuple[int, int] = (3, 3),
) -> dict:
    """In-plane sqrt(3) x sqrt(3) order of the intercalant in each gallery.

    Each intercalated atom gets the unit phasor ``exp(i K . r)``, with ``K``
    a corner of the host Brillouin zone, ``K = (G1 - G2) / 3``.  ``K . r``
    grows by 120 deg from one hexagon to the next, so the hexagon centres
    split into three sublattices (alpha, beta, gamma) with phasors 1,
    ``w``, ``w^2``.  The order parameter of a gallery is
    ``m = |Z|``, ``Z = sum_j exp(i K . r_j) / N``: 1 when all atoms sit on
    one sublattice (sqrt(3) x sqrt(3) order), small when they are spread
    over the three (3-state Potts magnetisation).  Few atoms per gallery
    keep ``m`` well above 0 even when disordered.

    The phase of ``Z`` compares galleries: equal phases mean the atoms of
    one gallery sit right above those of the other (alpha-alpha), 120 deg
    apart mean shifted sublattices (alpha-beta).  Positions are Cartesian,
    each gallery's atoms unwrapped along c to the gallery.

    Args:
        positions: Shape ``(n_atoms, 3)``; need not be wrapped.
        cell: Shape ``(3, 3)``, cell vectors as rows; an
            ``m_a x m_b`` supercell of the host lattice.
        geo: Sheets and galleries of this configuration.
        intercalant: Boolean mask of the intercalated atoms.
        in_plane_repeats: Primitive cells of the host along a and b; both
            multiples of 3, or ``K`` is not commensurate with the cell.

    Returns:
        ``li_order``: ``m`` of each gallery, NaN with fewer than 2 atoms,
        ``(n_galleries,)`` in :func:`gallery_order`;
        ``li_phase_shift``: ``|arg(Z_1 Z_0*)|`` (degrees, [0, 180]) of the
        two fullest galleries, NaN if either has fewer than 2 atoms;
        ``li_stacking``: ``ALIGNED`` (0) if ``li_phase_shift < 60``, else
        ``SHIFTED`` (1); NaN when the shift is.

    Raises:
        ValueError: If ``in_plane_repeats`` are not multiples of 3.
    """
    if any(m % 3 for m in in_plane_repeats):
        raise ValueError(
            f"in_plane_repeats {tuple(in_plane_repeats)}: the sqrt(3) x "
            "sqrt(3) order needs multiples of 3 along a and b."
        )
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    intercalant = np.asarray(intercalant, dtype=bool)
    g1, g2, _ = _host_reciprocal(cell, geo.normal, in_plane_repeats)
    k = (g1 - g2) / 3.0
    z = np.full(geo.n_sheets, np.nan, dtype=complex)
    for g in range(geo.n_sheets):
        atoms = intercalant & (geo.gallery == g)
        if atoms.sum() < 2:
            continue
        # Just above the gallery's lower sheet: the boundary gallery is not
        # wrapped back by c, which would add K . c to its phase.
        mid = geo.lower[g] + 0.5 * geo.width[g]
        pos = unwrap_to_height(positions[atoms], cell, mid)
        z[g] = np.exp(1j * pos @ k).mean()
    z = z[gallery_order(geo, intercalant)]
    shift = (
        float(np.degrees(np.abs(np.angle(z[1] * np.conj(z[0])))))
        if geo.n_sheets >= 2
        else float("nan")
    )
    return {
        "li_order": np.abs(z),
        "li_phase_shift": shift,
        "li_stacking": (
            float("nan")
            if np.isnan(shift)
            else float(ALIGNED if shift < 60.0 else SHIFTED)
        ),
    }


def _host_reciprocal(cell, normal, in_plane_repeats):
    """Two first-shell reciprocal vectors of the host lattice, 120 deg
    apart (counter-clockwise about ``normal``), and the dual lattice basis
    ``e`` (rows) with ``G_k . e_j = 2 pi delta_kj``."""
    m_a, m_b = in_plane_repeats
    a1, a2 = cell[0] / m_a, cell[1] / m_b
    area = np.cross(a1, a2) @ normal
    b1 = 2.0 * np.pi * np.cross(a2, normal) / area
    b2 = 2.0 * np.pi * np.cross(normal, a1) / area
    cand = [
        i * b1 + j * b2
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    ]
    g1 = min(cand, key=np.linalg.norm)
    angle = [np.arctan2(np.cross(g1, g) @ normal, g1 @ g) for g in cand]
    g2 = cand[int(np.argmin(np.abs(np.asarray(angle) - 2 * np.pi / 3)))]
    # Dual basis, in the plane: e_j . G_k = 2 pi delta_jk.
    gmat = np.stack([g1, g2, normal])
    e = 2.0 * np.pi * np.linalg.inv(gmat).T[:2]
    return g1, g2, e


def _structure_factor(positions, cell, geo, sheet, height, gs):
    """``S(G)`` of one sheet, its atoms brought to the image of the sheet
    at fractional ``height`` (unwrapped along c)."""
    atoms = geo.sheet_id == sheet
    pos = unwrap_to_height(positions[atoms], cell, height)
    return np.exp(1j * pos @ gs.T).sum(axis=0)


def _min_image(frac, e) -> float:
    """Length of the shortest lattice image of ``frac @ e``."""
    frac = frac - np.round(frac)
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    return float(np.min(np.linalg.norm((frac + shifts) @ e, axis=1)))
