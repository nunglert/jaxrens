"""Tests for jaxrens.utils.cell: wrap_positions and the c-tilt restriction.

The graph backends (MACE/nequix) find neighbors with a *finite* supercell
image set, which is only complete when atoms sit in the home cell.  Atoms
drift out of the cell under unwrapped Cartesian moves, so ``wrap_positions``
must (a) put them back, (b) be invariant under whole-lattice-vector shifts,
(c) preserve gradients (forces), and (d) leave non-periodic systems alone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends._graph_neighbors import (
    _make_image_offsets,
    _neighbor_mask,
)
from jaxrens.utils.cell import (
    _ws_vertex_candidates,
    c_in_plane_reduced,
    check_cell_shape,
    get_volume,
    min_aspect_ratio,
    nonperiodic_safe_cell,
    worst_tilt_aspect,
    wrap_positions,
)


def _ortho_cell(a=4.0, b=5.0, c=6.0):
    return jnp.diag(jnp.array([a, b, c], dtype=jnp.float32))


def _triclinic_cell():
    # Non-orthogonal but well-conditioned (rows are lattice vectors).
    return jnp.array(
        [[5.0, 0.0, 0.0], [1.2, 4.5, 0.0], [0.7, 0.9, 5.5]], dtype=jnp.float32
    )


# ---------------------------------------------------------------------------
# Basic wrapping behaviour
# ---------------------------------------------------------------------------


class TestWrapPositions:
    def test_in_cell_positions_unchanged(self):
        cell = _ortho_cell()
        pos = jnp.array([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0]], dtype=jnp.float32)
        out = wrap_positions(pos, cell)
        np.testing.assert_allclose(np.asarray(out), np.asarray(pos), atol=1e-5)

    def test_wraps_into_fractional_unit_box(self):
        cell = _triclinic_cell()
        rng = np.random.default_rng(0)
        # Positions spread several cells out in every direction.
        frac = rng.uniform(-3.0, 4.0, size=(12, 3)).astype(np.float32)
        pos = jnp.asarray(frac) @ cell
        out = wrap_positions(pos, cell)
        # Recover fractional coords of the wrapped positions.
        out_frac = np.asarray(out) @ np.linalg.inv(np.asarray(cell))
        assert np.all(out_frac >= -1e-4)
        assert np.all(out_frac < 1.0 + 1e-4)

    def test_idempotent(self):
        cell = _triclinic_cell()
        rng = np.random.default_rng(1)
        pos = (
            jnp.asarray(rng.uniform(-2.0, 2.0, (8, 3)).astype(np.float32))
            @ cell
        )
        once = wrap_positions(pos, cell)
        twice = wrap_positions(once, cell)
        np.testing.assert_allclose(
            np.asarray(once), np.asarray(twice), atol=1e-5
        )

    def test_lattice_translation_invariance(self):
        """Shifting an atom by any integer combination of lattice vectors is a
        no-op after wrapping — the core invariant the energy must respect."""
        cell = _triclinic_cell()
        rng = np.random.default_rng(2)
        pos = (
            jnp.asarray(rng.uniform(0.0, 1.0, (6, 3)).astype(np.float32))
            @ cell
        )
        n = np.array(
            [
                [2, -1, 3],
                [0, 0, 0],
                [-3, 4, -2],
                [1, 1, 1],
                [5, 0, -4],
                [0, -2, 1],
            ],
            dtype=np.float32,
        )
        shifted = pos + jnp.asarray(n) @ cell
        np.testing.assert_allclose(
            np.asarray(wrap_positions(pos, cell)),
            np.asarray(wrap_positions(shifted, cell)),
            atol=2e-4,
        )


# ---------------------------------------------------------------------------
# Non-periodic / degenerate cell
# ---------------------------------------------------------------------------


class TestNonPeriodic:
    def test_zero_cell_is_noop(self):
        cell = jnp.zeros((3, 3), dtype=jnp.float32)
        pos = jnp.array(
            [[-7.0, 3.0, 9.0], [12.0, -1.0, 0.5]], dtype=jnp.float32
        )
        out = wrap_positions(pos, cell)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(pos))

    def test_zero_cell_no_nans(self):
        cell = jnp.zeros((3, 3), dtype=jnp.float32)
        pos = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        out = wrap_positions(pos, cell)
        assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# Gradient safety (forces unchanged)
# ---------------------------------------------------------------------------


class TestGradient:
    def test_jacobian_is_identity_within_cell(self):
        """d(wrapped)/d(positions) is the identity a.e. — floor has zero grad,
        so galilean/HMC forces pass through unchanged."""
        cell = _triclinic_cell()
        # A point safely inside the cell (away from face boundaries).
        pos = jnp.array([[0.31, 0.42, 0.55]], dtype=jnp.float32) @ cell

        def wrap_one(p):
            return wrap_positions(p, cell)

        jac = jax.jacobian(wrap_one)(pos)  # shape (1,3,1,3)
        jac = np.asarray(jac).reshape(3, 3)
        np.testing.assert_allclose(jac, np.eye(3), atol=1e-4)


# ---------------------------------------------------------------------------
# Regression: neighbor completeness under drift
# ---------------------------------------------------------------------------


class TestNeighborInvariance:
    """The actual bug: ``_neighbor_mask`` on raw drifted positions loses edges;
    wrapping first restores lattice-translation invariance of the graph."""

    def _setup(self):
        cell = _ortho_cell(4.0, 4.0, 4.0)
        # Two atoms that are genuine neighbors across a face (r ~ 1.0 < cutoff).
        pos = jnp.array([[0.5, 0.5, 0.5], [3.5, 0.5, 0.5]], dtype=jnp.float32)
        offsets = jnp.asarray(_make_image_offsets(2, 2, 2))  # 27 images
        r_cutoff = 1.5
        return cell, pos, offsets, r_cutoff

    def test_raw_mask_changes_when_atom_drifts(self):
        """Sanity: without wrapping, drifting an atom by a lattice vector
        changes the neighbor count — demonstrating the bug exists."""
        cell, pos, offsets, r_cutoff = self._setup()
        base = _neighbor_mask(pos, cell, r_cutoff, offsets).sum()
        # Push atom 1 three cells along x — true neighbor relation unchanged,
        # but its nearest image now sits outside the {-1,0,1} offsets.
        drifted = pos.at[1].add(jnp.array([3 * 4.0, 0.0, 0.0]))
        drifted_count = _neighbor_mask(drifted, cell, r_cutoff, offsets).sum()
        assert int(base) != int(drifted_count)

    def test_wrapped_mask_invariant_to_drift(self):
        """With wrapping, the neighbor count is invariant to the drift."""
        cell, pos, offsets, r_cutoff = self._setup()
        base = _neighbor_mask(
            wrap_positions(pos, cell), cell, r_cutoff, offsets
        ).sum()
        drifted = pos.at[1].add(jnp.array([3 * 4.0, 0.0, 0.0]))
        wrapped_count = _neighbor_mask(
            wrap_positions(drifted, cell), cell, r_cutoff, offsets
        ).sum()
        assert int(base) == int(wrapped_count)
        # And the bond is actually present (each atom sees the other).
        assert int(base) == 2


class TestNonperiodicSafeCell:
    """Only a *degenerate* cell is replaced by the dummy box.

    Regression: the backends used to test ``trace(cell) == 0``, which also
    matches the zero-diagonal fcc primitive cell and its diagonal
    supercells, so such crystals were silently evaluated as vacuum clusters.
    """

    def test_zero_cell_gets_dummy_box(self):
        out = nonperiodic_safe_cell(jnp.zeros((3, 3)))
        np.testing.assert_allclose(np.asarray(out), 1000.0 * np.eye(3))

    @pytest.mark.parametrize("n", [1, 3])
    def test_zero_diagonal_fcc_primitive_kept(self, n):
        a = 4.0
        cell = (
            n
            * 0.5
            * a
            * jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        )
        assert float(jnp.trace(cell)) == 0.0
        np.testing.assert_array_equal(
            np.asarray(nonperiodic_safe_cell(cell)), np.asarray(cell)
        )

    def test_regular_cells_kept(self):
        for cell in (_ortho_cell(), _triclinic_cell()):
            np.testing.assert_array_equal(
                np.asarray(nonperiodic_safe_cell(cell)), np.asarray(cell)
            )


class TestRestrictCInPlane:
    """In-plane tilt of c restricted to the Wigner-Seitz cell of (a, b)."""

    A = 7.38

    def _hex(self, cz=6.7, cpar=(0.0, 0.0), sx=1.0):
        a = np.array([self.A * sx, 0.0, 0.0])
        b = np.array([self.A / 2 * sx, self.A * np.sqrt(3) / 2, 0.0])
        return jnp.asarray(np.array([a, b, [cpar[0], cpar[1], cz]]))

    def _domain_grid(self, cell, n=41):
        """Points of a dense grid inside the WS cell of (a, b)."""
        a, b = np.asarray(cell[0]), np.asarray(cell[1])
        pts = []
        for x in np.linspace(-1, 1, n):
            for y in np.linspace(-1, 1, n):
                p = x * a + y * b
                if all(
                    abs(p @ L) <= 0.5 * (L @ L) for L in (a, b, a + b, a - b)
                ):
                    pts.append(p)
        return np.array(pts)

    def test_reduced_cases(self):
        a = np.asarray(self._hex()[0])
        b = np.asarray(self._hex()[1])
        ab = (a + b) / 9  # (a1 + a2) / 3, AB registry
        assert bool(c_in_plane_reduced(self._hex()))
        assert bool(c_in_plane_reduced(self._hex(cpar=ab[:2])))
        assert bool(c_in_plane_reduced(self._hex(cpar=(2 * ab)[:2])))
        assert not bool(c_in_plane_reduced(self._hex(cpar=(ab + a)[:2])))
        # Just inside / outside a vertex of the hexagon, at |a| / sqrt(3).
        v = (a + b) / 3
        assert bool(c_in_plane_reduced(self._hex(cpar=(0.999 * v)[:2])))
        assert not bool(c_in_plane_reduced(self._hex(cpar=(1.01 * v)[:2])))

    def test_vertices_of_hexagonal_and_rectangular_lattices(self):
        pts, valid = _ws_vertex_candidates(self._hex())
        pts = np.asarray(pts)[np.asarray(valid)]
        r = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(r, self.A / np.sqrt(3), rtol=1e-5)
        assert len(np.unique(np.round(pts, 4), axis=0)) == 6
        rect = jnp.asarray(np.diag([4.0, 6.0, 7.0]))
        pts, valid = _ws_vertex_candidates(rect)
        corners = np.unique(
            np.round(np.asarray(pts)[np.asarray(valid)], 4), axis=0
        )
        np.testing.assert_allclose(np.abs(corners[:, :2]), [[2.0, 3.0]] * 4)

    @pytest.mark.parametrize(
        "cz, sx", [(6.7, 1.0), (13.4, 1.0), (22.0, 1.0), (9.0, 0.7)]
    )
    def _boundary(self, cell, n_angles=1440):
        """Points on the boundary of the WS cell, ray by ray.

        Along direction d, the boundary is at the smallest
        ``(|L|^2 / 2) / (d . L)`` over the lattice vectors with d . L > 0.
        """
        a, b = np.asarray(cell[0]), np.asarray(cell[1])
        lat = [a, b, a + b, a - b, -a, -b, -a - b, b - a]
        pts = []
        for t in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            d = np.array([np.cos(t), np.sin(t), 0.0])
            r = min(0.5 * (L @ L) / (d @ L) for L in lat if d @ L > 1e-12)
            pts.append(r * d)
        return np.array(pts)

    @pytest.mark.parametrize(
        "cz, sx", [(6.7, 1.0), (13.4, 1.0), (22.0, 1.0), (9.0, 0.7)]
    )
    def test_worst_tilt_matches_boundary_search(self, cz, sx):
        cell = self._hex(cz=cz, sx=sx)
        vol = get_volume(cell)
        brute = min(
            float(
                min_aspect_ratio(
                    cell.at[2].set(jnp.array([p[0], p[1], cz])), vol
                )
            )
            for p in self._boundary(cell)
        )
        assert float(worst_tilt_aspect(cell)) == pytest.approx(brute, abs=1e-4)

    def test_worst_tilt_ignores_current_tilt(self):
        cell = self._hex(cz=13.4)
        ref = float(worst_tilt_aspect(cell))
        for p in self._domain_grid(cell, n=9):
            tilted = cell.at[2].set(jnp.array([p[0], p[1], 13.4]))
            assert float(worst_tilt_aspect(tilted)) == pytest.approx(
                ref, rel=1e-5
            )

    @pytest.mark.parametrize("min_aspect", [0.5, 0.7, 0.75, 0.8, 0.9])
    @pytest.mark.parametrize("cz", [6.7, 10.05, 13.4, 14.8])
    def test_same_verdict_for_every_tilt(self, cz, min_aspect):
        cell = self._hex(cz=cz)
        verdicts = {
            bool(
                check_cell_shape(
                    cell.at[2].set(jnp.array([p[0], p[1], cz])),
                    39,
                    1e4,
                    0.1,
                    min_aspect,
                    restrict_c_in_plane=True,
                )
            )
            for p in self._domain_grid(cell, n=15)
        }
        assert len(verdicts) == 1

    def test_outside_domain_is_invalid(self):
        a = np.asarray(self._hex()[0])
        cell = self._hex(cpar=(a + 0.1)[:2])
        assert bool(check_cell_shape(cell, 39, 1e4, 0.1, 0.3))
        assert not bool(
            check_cell_shape(cell, 39, 1e4, 0.1, 0.3, restrict_c_in_plane=True)
        )

    def test_flag_off_is_unchanged(self):
        # 0.8 cuts the hexagon of a two-sheet cell: centre valid, worst
        # vertex not.
        straight = self._hex()
        vol = get_volume(straight)
        pts, valid = _ws_vertex_candidates(straight)
        verts = np.asarray(pts)[np.asarray(valid)]
        tilted = [
            straight.at[2].set(jnp.array([v[0], v[1], 6.7])) for v in verts
        ]
        worst = min(tilted, key=lambda c: float(min_aspect_ratio(c, vol)))
        assert bool(check_cell_shape(straight, 39, 1e4, 0.1, 0.8))
        assert not bool(check_cell_shape(worst, 39, 1e4, 0.1, 0.8))
        # With the restriction the whole shape is excluded.
        for c in (straight, worst):
            assert not bool(
                check_cell_shape(
                    c, 39, 1e4, 0.1, 0.8, restrict_c_in_plane=True
                )
            )
