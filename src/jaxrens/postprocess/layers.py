"""Sheet and gallery geometry of one layered configuration, in NumPy.

The post-processing counterpart of the geometry in
:mod:`jaxrens.sampling.moves.layered_moves`: same definitions, same
algorithm, but plain NumPy so that analysing a run does not import JAX.

Heights are fractional coordinates along the sheet normal,
``s3 = (r · n) / h mod 1`` with ``n = a × b / |a × b|`` and ``h = c · n``.
Host atoms (C in graphite) are sorted by height and split into sheets
wherever the gap between two consecutive ones exceeds ``SHEET_GAP``; a
gallery is the space between two consecutive sheets.

Unlike the moves, which identify the sheets once per walker, the sheets
are found again in every configuration.  :func:`layer_geometry` checks that
they are still the expected flat sheets and raises otherwise: the layered
runs move the host atoms only rigidly, so a broken sheet means the run is
not the kind this analysis assumes.

- normal_and_height: unit normal to the sheets and the period along it
- fractional_heights: s3 of every atom
- find_sheets: sheet index of every atom
- sheet_heights: circular mean height of each sheet
- galleries: lower bound, width and centre of each gallery
- gallery_of: gallery containing each height
- unwrap_to_height: bring atoms to the image of c nearest to a height
- layer_geometry: all of the above for one configuration, checked
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHEET_GAP = 1.0
"""Minimum separation (Angstrom) along the normal between two sheets.

Same value as ``jaxrens.sampling.moves.layered_moves.SHEET_GAP``."""

PLANARITY_TOL = 0.1
"""Maximum distance (Angstrom) of a host atom from its sheet.

Same value as ``jaxrens.sampling.moves.layered_moves.PLANARITY_TOL``."""


def normal_and_height(cell) -> tuple[np.ndarray, float]:
    """Unit normal to the (a, b) plane and the period ``h = c · n``.

    Raises:
        ValueError: If the cell is not periodic in the (a, b) plane.
    """
    cell = np.asarray(cell, dtype=float)
    n = np.cross(cell[0], cell[1])
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        raise ValueError(
            "layer geometry needs a cell that is periodic in the sheet "
            "plane (a x b != 0)."
        )
    n = n / norm
    return n, float(cell[2] @ n)


def fractional_heights(positions, cell) -> np.ndarray:
    """Fractional height of every atom along the sheet normal, in [0, 1).

    Positions need not be wrapped into the cell.
    """
    n, h = normal_and_height(cell)
    return np.mod(np.asarray(positions, dtype=float) @ n / h, 1.0)


def find_sheets(positions, cell, host) -> np.ndarray:
    """Sheet index of every atom.

    Host atoms are sorted by height and split into sheets wherever the gap
    to the previous host atom exceeds ``SHEET_GAP``.  A sheet straddling
    ``s3 = 0`` is merged into sheet 0.  Sheets are numbered 0..n-1 by
    increasing height; non-host atoms get -1.

    Args:
        positions: Shape ``(n_atoms, 3)``.
        cell: Shape ``(3, 3)``, cell vectors as rows.
        host: Boolean mask of the host atoms, shape ``(n_atoms,)``.

    Returns:
        Integer array of shape ``(n_atoms,)``.
    """
    _, h = normal_and_height(cell)
    s3 = fractional_heights(positions, cell)
    host = np.asarray(host, dtype=bool)
    sheet_id = np.full(s3.shape, -1, dtype=int)
    idx = np.flatnonzero(host)
    if idx.size == 0:
        return sheet_id
    order = idx[np.argsort(s3[idx], kind="stable")]
    z = s3[order]
    label = np.concatenate([[0], np.cumsum(np.diff(z) * abs(h) > SHEET_GAP)])
    n_groups = int(label[-1]) + 1
    wrap_gap = z[0] + 1.0 - z[-1]
    if n_groups > 1 and wrap_gap * abs(h) <= SHEET_GAP:
        label[label == n_groups - 1] = 0
    sheet_id[order] = label
    return sheet_id


def sheet_heights(s3, sheet_id, n_sheets: int) -> np.ndarray:
    """Circular mean height of each sheet, in [0, 1).

    Args:
        s3: Fractional heights of all atoms, shape ``(n_atoms,)``.
        sheet_id: Sheet of every atom, -1 for non-host atoms.
        n_sheets: Number of sheets.

    Returns:
        Array of shape ``(n_sheets,)``.
    """
    s3 = np.asarray(s3, dtype=float)
    sheet_id = np.asarray(sheet_id)
    host = sheet_id >= 0
    angle = 2.0 * np.pi * s3[host]
    cos = np.bincount(sheet_id[host], np.cos(angle), minlength=n_sheets)
    sin = np.bincount(sheet_id[host], np.sin(angle), minlength=n_sheets)
    mean = np.arctan2(sin[:n_sheets], cos[:n_sheets]) / (2.0 * np.pi)
    return np.mod(mean, 1.0)


def galleries(heights) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Galleries between consecutive sheets, cyclically.

    Gallery ``g`` runs upward from sheet ``g`` to sheet ``g + 1`` (sheet 0
    for the last one).

    Returns:
        ``(lower, width, centre)``, fractional, each of shape
        ``(n_sheets,)``.
    """
    lower = np.asarray(heights, dtype=float)
    width = np.mod(np.roll(lower, -1) - lower, 1.0)
    centre = np.mod(lower + 0.5 * width, 1.0)
    return lower, width, centre


def gallery_of(s3, lower, width) -> np.ndarray:
    """Index of the gallery containing each height in ``s3``."""
    s3 = np.asarray(s3, dtype=float)
    d = np.mod(s3[:, None] - np.asarray(lower)[None, :], 1.0)
    return np.argmax(d < np.asarray(width)[None, :], axis=1)


def unwrap_to_height(positions, cell, height) -> np.ndarray:
    """Shift atoms by multiples of c to the image nearest to ``height``.

    Args:
        positions: Shape ``(n_atoms, 3)``.
        cell: Shape ``(3, 3)``, cell vectors as rows.
        height: Target fractional height along the normal, scalar or
            ``(n_atoms,)``; may lie outside [0, 1).

    Returns:
        Positions whose fractional heights lie within 1/2 of ``height``.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n, h = normal_and_height(cell)
    k = np.round(positions @ n / h - np.asarray(height))
    return positions - k[:, None] * cell[2]


@dataclass(frozen=True)
class LayerGeometry:
    """Sheets and galleries of one configuration.

    Fractional quantities are along the sheet normal, in units of
    ``height``.  Gallery ``g`` runs upward from sheet ``g`` to the next.

    Attributes:
        normal: Unit normal to the sheets, shape ``(3,)``.
        height: Period along the normal, ``h = c · n`` (Angstrom).
        s3: Fractional height of every atom, shape ``(n_atoms,)``.
        sheet_id: Sheet of every atom, -1 for non-host atoms.
        sheet_size: Number of host atoms per sheet.
        heights: Fractional height of each sheet, shape ``(n_sheets,)``.
        lower: Fractional lower bound of each gallery.
        width: Fractional width of each gallery.
        centre: Fractional centre of each gallery.
        gallery: Gallery of every non-host atom, -1 for host atoms.
    """

    normal: np.ndarray
    height: float
    s3: np.ndarray
    sheet_id: np.ndarray
    sheet_size: int
    heights: np.ndarray
    lower: np.ndarray
    width: np.ndarray
    centre: np.ndarray
    gallery: np.ndarray

    @property
    def n_sheets(self) -> int:
        """Number of sheets, equal to the number of galleries."""
        return len(self.heights)


def layer_geometry(
    positions,
    cell,
    host,
    *,
    n_sheets: int | None = None,
    sheet_size: int | None = None,
) -> LayerGeometry:
    """Sheets and galleries of one configuration, checked.

    Args:
        positions: Shape ``(n_atoms, 3)``; need not be wrapped.
        cell: Shape ``(3, 3)``, cell vectors as rows.
        host: Boolean mask of the host atoms, shape ``(n_atoms,)``.
        n_sheets: Expected number of sheets; any number >= 2 if None.
        sheet_size: Expected host atoms per sheet; any if None.

    Returns:
        The :class:`LayerGeometry` of the configuration.

    Raises:
        ValueError: If the cell is not periodic in the (a, b) plane, there
            are fewer than two sheets, the sheets have different sizes or
            not the expected number or size, or a sheet is not flat within
            ``PLANARITY_TOL``.
    """
    normal, h = normal_and_height(cell)
    s3 = fractional_heights(positions, cell)
    sheet_id = find_sheets(positions, cell, host)
    host = sheet_id >= 0
    n = int(sheet_id.max()) + 1 if host.any() else 0
    if n < 2:
        raise ValueError(
            f"found {n} sheet(s) of host atoms; layer geometry needs at "
            "least 2, parallel to the (a, b) plane of the cell."
        )
    if n_sheets is not None and n != n_sheets:
        raise ValueError(f"found {n} sheets, expected {n_sheets}.")
    sizes = np.bincount(sheet_id[host], minlength=n)
    if len(set(sizes.tolist())) > 1:
        raise ValueError(
            f"sheets have different numbers of host atoms {sizes.tolist()}."
        )
    if sheet_size is not None and sizes[0] != sheet_size:
        raise ValueError(
            f"sheets have {int(sizes[0])} host atoms, expected {sheet_size}."
        )
    heights = sheet_heights(s3, sheet_id, n)
    dev = (s3[host] - heights[sheet_id[host]] + 0.5) % 1.0 - 0.5
    worst = float(np.max(np.abs(dev)) * abs(h))
    if worst > PLANARITY_TOL:
        raise ValueError(
            f"a sheet is not flat within {PLANARITY_TOL} A (largest "
            f"deviation {worst:.3f} A)."
        )
    lower, width, centre = galleries(heights)
    gallery = np.where(host, -1, gallery_of(s3, lower, width))
    return LayerGeometry(
        normal=normal,
        height=h,
        s3=s3,
        sheet_id=sheet_id,
        sheet_size=int(sizes[0]),
        heights=heights,
        lower=lower,
        width=width,
        centre=centre,
        gallery=gallery,
    )
