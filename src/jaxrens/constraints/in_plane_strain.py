"""In-plane strain constraint on the (a, b) lattice of a layered cell.

Rejects cells whose (a, b) lattice is strained, with respect to a reference
lattice, beyond a window on its principal stretches.  Meant for layered
systems whose sheets only move rigidly with the cell (no local moves on the
sheet atoms): there, every volume / shear / stretch move maps the sheets
affinely, so the in-plane strain of the cell *is* the strain of the sheets,
and nothing else stops a cell move from turning graphene into a
4-coordinated net (e.g. principal stretches 1/sqrt(3) and sqrt(3), where
second neighbours land at the bond length).

The strain is measured on the 2-D lattice only.  ``a`` and ``b`` are written
in an orthonormal basis of their own plane, which makes the measure blind to
rigid rotations (also to a tilt of the sheets after a shear of a or b) and
to any change of ``c``.  With ``R`` and ``A`` the 2x2 matrices whose rows
are the reference and the current (a, b), the deformation gradient is
``F = A^T R^-T`` and its singular values, the principal stretches
``lambda_1 <= lambda_2``, are 1 for an undeformed lattice.  The cell is
valid when ``lambda_min <= lambda_1`` and ``lambda_2 <= lambda_max``.

``F`` compares labelled vectors: the reference ``a`` is matched to the
current ``a``.  The cell moves never relabel or reduce the cell vectors, so
this is the right match for a run started from cells with the reference
(a, b).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxrens.constraints.base import ConstraintDescriptor


def reference_lattice_2d(a: float, b: float, gamma: float) -> np.ndarray:
    """Rows (a, b) of a 2-D lattice from |a|, |b| and their angle (deg)."""
    g = np.deg2rad(gamma)
    return np.array([[a, 0.0], [b * np.cos(g), b * np.sin(g)]])


def lattice_2d(cell: jnp.ndarray) -> jnp.ndarray:
    """Rows (a, b) of ``cell`` in an orthonormal basis of their plane."""
    a, b = cell[0], cell[1]
    u = a / jnp.linalg.norm(a)
    n = jnp.cross(a, b)
    v = jnp.cross(n / jnp.linalg.norm(n), u)
    return jnp.array([[a @ u, a @ v], [b @ u, b @ v]])


def principal_stretches(
    cell: jnp.ndarray, ref_inv_t: jnp.ndarray
) -> jnp.ndarray:
    """Sorted principal stretches of the in-plane lattice of ``cell``.

    Args:
        cell: (3, 3) cell, rows a, b, c.
        ref_inv_t: ``inv(R).T`` of the reference 2-D lattice R (rows a, b).

    Returns:
        (2,) array ``[lambda_1, lambda_2]``, ``lambda_1 <= lambda_2``.
        Closed form for a 2x2 matrix: the eigenvalues of ``F^T F`` are
        ``t / 2 -+ sqrt(t^2 / 4 - d)`` with ``t`` its trace and
        ``d = det(F)^2``.
    """
    f = lattice_2d(cell).T @ ref_inv_t
    t = jnp.sum(f * f)
    d = jnp.linalg.det(f) ** 2
    disc = jnp.sqrt(jnp.maximum(0.25 * t * t - d, 0.0))
    lam2 = jnp.stack([0.5 * t - disc, 0.5 * t + disc])
    return jnp.sqrt(jnp.maximum(lam2, 0.0))


def build_in_plane_strain(
    ref_2d, lambda_min: float, lambda_max: float
) -> object:
    """Build the in-plane strain predicate.

    Args:
        ref_2d: (2, 2) reference lattice, rows (a, b) in any orthonormal
            basis of their plane (see :func:`reference_lattice_2d`).
        lambda_min: Smallest allowed principal stretch (< 1).
        lambda_max: Largest allowed principal stretch (> 1).

    Returns:
        ``is_valid(positions, types, cell) -> bool`` scalar predicate.  Only
        ``cell`` is read.
    """
    ref_inv_t = jnp.asarray(np.linalg.inv(np.asarray(ref_2d)).T)

    def is_valid(positions, types, cell):
        lam = principal_stretches(cell, ref_inv_t)
        return (lam[0] >= lambda_min) & (lam[1] <= lambda_max)

    return is_valid


def in_plane_strain_descriptor(
    ref_2d, lambda_min: float, lambda_max: float
) -> ConstraintDescriptor:
    """Wrap the in-plane strain predicate as a gate constraint.

    Depends on the cell only, so the gate checks it after the moves that
    change the cell (volume, normal_strain, shear, stretch, stacking_shift,
    gallery_strain) and skips the others.  Rejections share bucket 4 with the other configuration
    constraints.
    """
    return ConstraintDescriptor(
        name="in_plane_strain",
        depends_on=frozenset({"cell"}),
        build=build_in_plane_strain,
        build_kwargs={
            "ref_2d": np.asarray(ref_2d, dtype=np.float64),
            "lambda_min": float(lambda_min),
            "lambda_max": float(lambda_max),
        },
        reject_reason=4,
    )
