"""Tests for ``wrap_positions`` and ``nonperiodic_safe_cell`` in
``jaxrens.utils.cell``.

The graph backends (MACE, nequix) find neighbors with a finite set of periodic
images, which is complete only when atoms sit in the home cell. Atoms drift
out of it under unwrapped Cartesian moves, so ``wrap_positions`` has to put
them back without changing gradients, and leave non-periodic systems alone.
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
from jaxrens.utils.cell import nonperiodic_safe_cell, wrap_positions


def _ortho_cell(a=4.0, b=5.0, c=6.0):
    return jnp.diag(jnp.array([a, b, c], dtype=jnp.float32))


def _triclinic_cell():
    # Non-orthogonal but well-conditioned (rows are lattice vectors).
    return jnp.array(
        [[5.0, 0.0, 0.0], [1.2, 4.5, 0.0], [0.7, 0.9, 5.5]], dtype=jnp.float32
    )


class TestWrapPositions:
    """Check that positions wrap into the unit cell, that wrapping twice
    changes nothing, and that lattice translations don't matter."""

    def test_in_cell_positions_unchanged(self):
        cell = _ortho_cell()
        pos = jnp.array([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0]], dtype=jnp.float32)
        out = wrap_positions(pos, cell)
        np.testing.assert_allclose(np.asarray(out), np.asarray(pos), atol=1e-5)

    def test_wraps_into_fractional_unit_box(self):
        cell = _triclinic_cell()
        rng = np.random.default_rng(0)
        frac = rng.uniform(-3.0, 4.0, size=(12, 3)).astype(np.float32)
        pos = jnp.asarray(frac) @ cell
        out = wrap_positions(pos, cell)
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
        """Shifting an atom by any integer combination of lattice vectors is
        a no-op after wrapping, the invariant the energy must respect."""
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


class TestNonPeriodic:
    """Check that a zero (non-periodic) cell leaves positions untouched and
    keeps the gradient free of NaNs."""

    def test_zero_cell_is_noop(self):
        cell = jnp.zeros((3, 3), dtype=jnp.float32)
        pos = jnp.array(
            [[-7.0, 3.0, 9.0], [12.0, -1.0, 0.5]], dtype=jnp.float32
        )
        out = wrap_positions(pos, cell)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(pos))

    def test_zero_cell_gradient_is_finite(self):
        cell = jnp.zeros((3, 3), dtype=jnp.float32)
        pos = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        # inv of the zero cell is inf/NaN. The forward value hides it behind
        # ``where``, but without the safe-cell mask it leaks into the gradient.
        jac = jax.jacobian(lambda p: wrap_positions(p, cell))(pos)
        assert np.all(np.isfinite(np.asarray(jac)))


class TestGradient:
    """Check that wrapping leaves gradients unchanged, so forces pass
    through."""

    def test_jacobian_is_identity_within_cell(self):
        """``d(wrapped)/d(positions)`` is the identity almost everywhere:
        ``floor`` has zero gradient, so galilean/HMC forces are unchanged."""
        cell = _triclinic_cell()
        # Away from the cell faces, where floor is discontinuous.
        pos = jnp.array([[0.31, 0.42, 0.55]], dtype=jnp.float32) @ cell

        def wrap_one(p):
            return wrap_positions(p, cell)

        jac = jax.jacobian(wrap_one)(pos)  # shape (1, 3, 1, 3)
        jac = np.asarray(jac).reshape(3, 3)
        np.testing.assert_allclose(jac, np.eye(3), atol=1e-4)


class TestNeighborInvariance:
    """Check that wrapping fixes the bug where an atom that drifted by a
    lattice vector loses its edges in ``_neighbor_mask``, which only sees a
    finite set of periodic images."""

    def _setup(self):
        cell = _ortho_cell(4.0, 4.0, 4.0)
        # Two atoms that are real neighbors across a face (r ~ 1.0 < cutoff).
        pos = jnp.array([[0.5, 0.5, 0.5], [3.5, 0.5, 0.5]], dtype=jnp.float32)
        offsets = jnp.asarray(_make_image_offsets(2, 2, 2))  # 27 images
        r_cutoff = 1.5
        return cell, pos, offsets, r_cutoff

    def test_raw_mask_changes_when_atom_drifts(self):
        """Without wrapping, drifting an atom by a lattice vector changes the
        neighbor count, which shows the bug exists."""
        cell, pos, offsets, r_cutoff = self._setup()
        base = _neighbor_mask(pos, cell, r_cutoff, offsets).sum()
        # Push atom 1 three cells along x. The true neighbor relation is
        # unchanged, but its nearest image now sits outside the {-1,0,1}
        # offsets.
        drifted = pos.at[1].add(jnp.array([3 * 4.0, 0.0, 0.0]))
        drifted_count = _neighbor_mask(drifted, cell, r_cutoff, offsets).sum()
        assert int(base) != int(drifted_count)

    def test_wrapped_mask_invariant_to_drift(self):
        """With wrapping, the neighbor count doesn't depend on the drift."""
        cell, pos, offsets, r_cutoff = self._setup()
        base = _neighbor_mask(
            wrap_positions(pos, cell), cell, r_cutoff, offsets
        ).sum()
        drifted = pos.at[1].add(jnp.array([3 * 4.0, 0.0, 0.0]))
        wrapped_count = _neighbor_mask(
            wrap_positions(drifted, cell), cell, r_cutoff, offsets
        ).sum()
        assert int(base) == int(wrapped_count)
        # The bond exists: each atom sees the other.
        assert int(base) == 2


class TestNonperiodicSafeCell:
    """Check that only a degenerate cell is replaced by the dummy box.

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
