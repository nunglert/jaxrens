"""Cell/box utility functions for periodic systems.

All functions are jit-compatible: no Python conditionals on traced values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def get_volume(cell: jnp.ndarray) -> jnp.ndarray:
    """Return the volume of a cell: |det(cell)|."""
    return jnp.abs(jnp.linalg.det(cell))


def nonperiodic_safe_cell(
    cell: jnp.ndarray, box: float = 1000.0
) -> jnp.ndarray:
    """Substitute a large cubic dummy box when ``cell`` is degenerate.

    Non-periodic systems pass ``cell == 0``; backends that always build a
    periodic image geometry need *some* finite cell, so they get a
    ``box``-sized cube in which the system is an isolated cluster.

    The test is on the determinant, not the trace: a valid periodic cell can
    have a zero diagonal (e.g. the fcc primitive ``[[0,a,a],[a,0,a],[a,a,0]]``
    and its diagonal supercells), and a trace test would silently evaluate
    such crystals as clusters in vacuum.  The threshold matches
    :func:`wrap_positions`.
    """
    periodic = jnp.abs(jnp.linalg.det(cell)) > 1e-12
    return jnp.where(periodic, cell, box * jnp.eye(3, dtype=cell.dtype))


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


def _in_plane_lattice_vectors(cell: jnp.ndarray) -> jnp.ndarray:
    """``a, b, a + b, a - b`` and their negatives, shape (8, 3)."""
    a, b = cell[0], cell[1]
    vecs = jnp.stack([a, b, a + b, a - b])
    return jnp.concatenate([vecs, -vecs])


def _split_c(cell: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unit normal of the (a, b) plane and the in-plane part of c."""
    n = jnp.cross(cell[0], cell[1])
    n = n / jnp.linalg.norm(n)
    c = cell[2]
    return n, c - jnp.dot(c, n) * n


def c_in_plane_reduced(cell: jnp.ndarray) -> jnp.ndarray:
    """Whether the in-plane part of c lies in the Wigner-Seitz cell of (a, b).

    ``c`` and ``c + p a + q b`` describe the same crystal.  Within the
    Wigner-Seitz cell, the in-plane part ``c_par`` is the shortest of these
    representatives: ``|c_par . L| <= |L|^2 / 2`` for ``L`` in
    ``a, b, a + b, a - b``.

    Returns a boolean scalar.  Fully jit-compatible.
    """
    _, c_par = _split_c(cell)
    lat = _in_plane_lattice_vectors(cell)
    half = 0.5 * jnp.sum(lat * lat, axis=1)
    return jnp.all(lat @ c_par <= half * (1.0 + 1e-6))


def _ws_vertex_candidates(cell: jnp.ndarray):
    """Candidate vertices of the Wigner-Seitz cell of (a, b), in 3D.

    The cell is the intersection of the half-planes ``x . L <= |L|^2 / 2``
    over the 8 vectors of :func:`_in_plane_lattice_vectors`; its vertices are
    among the 28 pairwise intersections of their boundary lines.

    Returns:
        ``(points, valid)``: shape (28, 3) and (28,), ``valid`` marking the
        intersections that satisfy all half-planes (the actual vertices).
    """
    n = jnp.cross(cell[0], cell[1])
    n = n / jnp.linalg.norm(n)
    e1 = cell[0] / jnp.linalg.norm(cell[0])
    e2 = jnp.cross(n, e1)
    lat3 = _in_plane_lattice_vectors(cell)
    lat = jnp.stack([lat3 @ e1, lat3 @ e2], axis=1)  # (8, 2)
    half = 0.5 * jnp.sum(lat * lat, axis=1)
    ii, jj = jnp.triu_indices(8, k=1)
    li, lj = lat[ii], lat[jj]
    det = li[:, 0] * lj[:, 1] - li[:, 1] * lj[:, 0]
    ok = jnp.abs(det) > 1e-9 * half[ii]
    safe = jnp.where(ok, det, 1.0)
    # Cramer's rule for x . li = half_i, x . lj = half_j.
    x = (half[ii] * lj[:, 1] - half[jj] * li[:, 1]) / safe
    y = (li[:, 0] * half[jj] - lj[:, 0] * half[ii]) / safe
    pts2 = jnp.stack([x, y], axis=1)
    inside = jnp.all(pts2 @ lat.T <= half * (1.0 + 1e-6), axis=1)
    points = pts2[:, :1] * e1 + pts2[:, 1:] * e2
    return points, ok & inside


def worst_tilt_aspect(cell: jnp.ndarray) -> jnp.ndarray:
    """Smallest aspect ratio over all in-plane tilts of c in the WS cell.

    Keeps ``a``, ``b`` and the height of ``c`` above the (a, b) plane, and
    puts the in-plane part of ``c`` at each vertex of the Wigner-Seitz cell
    of (a, b).  The heights of a and b are ``V / |b x c|`` and
    ``V / |a x c|``; both cross products are convex in the in-plane part of
    c, so the minimum over the cell is reached at a vertex.  The result
    depends on the cell shape only, not on where c's tilt currently is.

    Returns a scalar.  Fully jit-compatible.
    """
    n, c_par = _split_c(cell)
    c_perp = cell[2] - c_par
    volume = get_volume(cell)
    points, valid = _ws_vertex_candidates(cell)

    def aspect_at(p):
        tilted = cell.at[2].set(c_perp + p)
        return min_aspect_ratio(tilted, volume)

    aspects = jax.vmap(aspect_at)(points)
    return jnp.min(jnp.where(valid, aspects, jnp.inf))


def check_cell_shape(
    cell: jnp.ndarray,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
    restrict_c_in_plane: bool = False,
) -> jnp.ndarray:
    """Check whether cell shape satisfies geometric criteria.

    With ``restrict_c_in_plane`` the in-plane part of c must lie in the
    Wigner-Seitz cell of (a, b) (see :func:`c_in_plane_reduced`), and the
    aspect ratio is taken at the worst tilt in that cell
    (:func:`worst_tilt_aspect`) instead of the current one.  The verdict then
    depends on the cell shape only, so every tilt of c, i.e. every stacking
    registry, is allowed or excluded together.

    Returns a boolean scalar (True = valid). Fully jit-compatible.
    """
    volume = get_volume(cell)
    vol_per_atom = volume / n_atoms
    if restrict_c_in_plane:
        aspect = worst_tilt_aspect(cell)
    else:
        aspect = min_aspect_ratio(cell, volume)

    valid = (
        (vol_per_atom <= max_vol_per_atom)
        & (vol_per_atom >= min_vol_per_atom)
        & (aspect >= min_aspect)
    )
    if restrict_c_in_plane:
        valid = valid & c_in_plane_reduced(cell)
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
