"""Tests for jaxrens.utils.cell.wrap_positions.

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
from jaxrens.utils.cell import wrap_positions


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
