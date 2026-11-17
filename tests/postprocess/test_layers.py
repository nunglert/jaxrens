"""Sheet and gallery geometry of single configurations (NumPy)."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.postprocess import layers
from jaxrens.postprocess.layers import (
    find_sheets,
    fractional_heights,
    galleries,
    gallery_of,
    layer_geometry,
    normal_and_height,
    sheet_heights,
)
from jaxrens.sampling.moves import layered_moves
from tests._layered import LI, C, graphite


def _perturbed(spacings, li, *, z0=0.0, tilt=(0.0, 0.0), seed=0):
    """Graphite with in-plane noise on every atom, out-of-plane noise below
    ``PLANARITY_TOL`` on C, c tilted by ``tilt`` (Angstrom, in plane), and
    some atoms shifted by a lattice vector (unwrapped)."""
    pos, typ, cell = graphite(spacings, li, z0=z0)
    rng = np.random.default_rng(seed)
    cell[2, :2] = tilt
    pos = pos + rng.uniform(-0.3, 0.3, pos.shape) * [1.0, 1.0, 0.0]
    pos[:, 2] += np.where(typ == C, rng.uniform(-0.04, 0.04, len(pos)), 0.0)
    shift = rng.integers(-1, 2, (len(pos), 3))
    return pos + shift @ cell, typ, cell


CASES = {
    "bilayer": dict(spacings=[3.7, 3.7], li=(3, 3), z0=1.0),
    "uneven": dict(spacings=[3.4, 4.1], li=(3,)),
    "straddling": dict(spacings=[3.7, 3.7], li=(3, 3), z0=0.0),
    "tilted": dict(spacings=[3.7, 3.7], li=(2, 1), tilt=(1.3, -0.7)),
    "three_sheets": dict(spacings=[3.5, 3.5, 3.5], li=(1, 2, 3)),
}


def test_constants_match_layered_moves():
    assert layers.SHEET_GAP == layered_moves.SHEET_GAP
    assert layers.PLANARITY_TOL == layered_moves.PLANARITY_TOL


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
class TestMatchesLayeredMoves:
    """Same results as the JAX functions the layered moves use."""

    def _setup(self, name, seed):
        pos, typ, cell = _perturbed(**CASES[name], seed=seed)
        return pos, typ, cell

    def test_normal_and_height(self, name, seed):
        _, _, cell = self._setup(name, seed)
        n, h = normal_and_height(cell)
        n_ref, h_ref = layered_moves._normal_and_height(jnp.asarray(cell))
        np.testing.assert_allclose(n, n_ref, atol=1e-6)
        assert h == pytest.approx(float(h_ref), rel=1e-6)

    def test_sheets(self, name, seed):
        pos, typ, cell = self._setup(name, seed)
        sid = find_sheets(pos, cell, typ == C)
        init = layered_moves.make_sheet_id_initializer((C,))
        ref = np.asarray(init(jnp.asarray(pos), jnp.asarray(typ), cell))
        np.testing.assert_array_equal(sid, ref)

    def test_heights_and_galleries(self, name, seed):
        pos, typ, cell = self._setup(name, seed)
        sid = find_sheets(pos, cell, typ == C)
        n = int(sid.max()) + 1
        s3 = fractional_heights(pos, cell)
        s3_ref = layered_moves.fractional_heights(
            jnp.asarray(pos), jnp.asarray(cell)
        )
        np.testing.assert_allclose(s3, s3_ref, atol=1e-5)
        heights = sheet_heights(s3, sid, n)
        ref = layered_moves.sheet_heights(jnp.asarray(s3), jnp.asarray(sid), n)
        np.testing.assert_allclose(heights, ref, atol=1e-5)
        got = galleries(heights)
        for x, y in zip(got, layered_moves.galleries(jnp.asarray(heights))):
            np.testing.assert_allclose(x, y, atol=1e-6)
        lower, width, _ = got
        np.testing.assert_array_equal(
            gallery_of(s3, lower, width),
            layered_moves.gallery_of(
                jnp.asarray(s3), jnp.asarray(lower), jnp.asarray(width)
            ),
        )


class TestLayerGeometry:
    def test_bilayer(self):
        pos, typ, cell = graphite([3.4, 4.0], (3, 2))
        geo = layer_geometry(pos, cell, typ == C)
        assert geo.n_sheets == 2
        assert geo.sheet_size == 18
        assert geo.height == pytest.approx(7.4)
        np.testing.assert_allclose(geo.normal, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(geo.heights, [0.0, 3.4 / 7.4], atol=1e-12)
        np.testing.assert_allclose(geo.width * geo.height, [3.4, 4.0])
        np.testing.assert_allclose(geo.centre * geo.height, [1.7, 3.4 + 2.0])

    def test_gallery_of_every_atom(self):
        pos, typ, cell = graphite([3.4, 4.0], (3, 2))
        geo = layer_geometry(pos, cell, typ == C)
        assert (geo.gallery[typ == C] == -1).all()
        np.testing.assert_array_equal(
            geo.gallery[typ == LI], [0] * 3 + [1] * 2
        )

    def test_tilted_c_changes_nothing(self):
        pos, typ, cell = graphite([3.4, 4.0], (3, 2))
        ref = layer_geometry(pos, cell, typ == C)
        cell[2, :2] = [2.0, -1.1]
        geo = layer_geometry(pos, cell, typ == C)
        assert geo.height == pytest.approx(ref.height)
        np.testing.assert_allclose(geo.width, ref.width)
        np.testing.assert_array_equal(geo.gallery, ref.gallery)

    def test_unwrapped_positions_change_nothing(self):
        pos, typ, cell = graphite([3.4, 4.0], (3, 2))
        ref = layer_geometry(pos, cell, typ == C)
        shift = np.random.default_rng(0).integers(-2, 3, (len(pos), 3))
        geo = layer_geometry(pos + shift @ cell, cell, typ == C)
        np.testing.assert_allclose(geo.heights, ref.heights, atol=1e-12)
        np.testing.assert_array_equal(geo.gallery, ref.gallery)

    def test_expected_layout_accepted(self):
        pos, typ, cell = graphite([3.7, 3.7], (3, 3))
        geo = layer_geometry(pos, cell, typ == C, n_sheets=2, sheet_size=18)
        assert geo.n_sheets == 2


class TestLayerGeometryErrors:
    def test_carbon_in_a_gallery(self):
        pos, typ, cell = graphite([3.7, 3.7], (3, 3))
        pos[0, 2] += 1.85
        with pytest.raises(ValueError, match="different numbers"):
            layer_geometry(pos, cell, typ == C)

    def test_unexpected_number_of_sheets(self):
        pos, typ, cell = graphite([3.5, 3.5, 3.5])
        with pytest.raises(ValueError, match="found 3 sheets, expected 2"):
            layer_geometry(pos, cell, typ == C, n_sheets=2)

    def test_unexpected_sheet_size(self):
        pos, typ, cell = graphite([3.7, 3.7])
        with pytest.raises(ValueError, match="expected 16"):
            layer_geometry(pos, cell, typ == C, sheet_size=16)

    def test_sheet_not_flat(self):
        pos, typ, cell = graphite([3.7, 3.7])
        pos[0, 2] += 0.3
        with pytest.raises(ValueError, match="not flat"):
            layer_geometry(pos, cell, typ == C)

    def test_single_sheet(self):
        pos, typ, cell = graphite([3.7])
        with pytest.raises(ValueError, match="found 1 sheet"):
            layer_geometry(pos, cell, typ == C)

    def test_non_periodic_cell(self):
        pos, typ, _ = graphite([3.7, 3.7])
        with pytest.raises(ValueError, match="periodic"):
            layer_geometry(pos, np.zeros((3, 3)), typ == C)
