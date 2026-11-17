"""Structural observables of single configurations."""

import numpy as np
import pytest

from jaxrens.postprocess.layers import layer_geometry
from jaxrens.postprocess.observables import (
    AA,
    AB,
    cell_observables,
    layer_observables,
    li_order_observables,
    registry_observables,
)
from tests._layered import A1, A2, HOLLOW, LI, C, graphite


class TestCellObservables:
    def test_graphite(self):
        pos, _, cell = graphite([3.4, 4.0])
        obs = cell_observables(pos, cell)
        assert obs["a_eq"] == pytest.approx(2.46)
        assert obs["b_eq"] == pytest.approx(2.46)
        assert obs["gamma"] == pytest.approx(60.0)
        assert obs["h"] == pytest.approx(7.4)
        assert obs["c_shift"] == pytest.approx(0.0, abs=1e-12)
        area = 3 * 2.46 * 3 * 2.46 * np.sqrt(3) / 2
        assert obs["volume_per_atom"] == pytest.approx(area * 7.4 / 36)

    def test_in_plane_repeats(self):
        pos, _, cell = graphite([3.7, 3.7], reps=2)
        obs = cell_observables(pos, cell, in_plane_repeats=(2, 2))
        assert obs["a_eq"] == pytest.approx(2.46)

    def test_tilted_c(self):
        pos, _, cell = graphite([3.4, 4.0])
        ref = cell_observables(pos, cell)
        cell[2, :2] = [1.2, -0.5]
        obs = cell_observables(pos, cell)
        assert obs["h"] == pytest.approx(ref["h"])
        assert obs["c_shift"] == pytest.approx(1.3)
        assert obs["volume_per_atom"] == pytest.approx(ref["volume_per_atom"])


def _layer_obs(spacings, li):
    pos, typ, cell = graphite(spacings, li)
    geo = layer_geometry(pos, cell, typ == C)
    return layer_observables(geo, typ == LI)


class TestLayerObservables:
    def test_stage_one(self):
        obs = _layer_obs([3.7, 3.6], (3, 3))
        np.testing.assert_array_equal(obs["gallery_occupancy"], [3, 3])
        # Same occupancy: wider gallery first.
        np.testing.assert_allclose(obs["gallery_width"], [3.7, 3.6])
        assert obs["n_empty"] == 0
        assert obs["d_occupied"] == pytest.approx(3.65)
        assert np.isnan(obs["d_empty"])

    def test_stage_two(self):
        # The empty gallery is gallery 0: sorting puts the full one first.
        obs = _layer_obs([3.35, 3.7], (0, 3))
        np.testing.assert_array_equal(obs["gallery_occupancy"], [3, 0])
        np.testing.assert_allclose(obs["gallery_width"], [3.7, 3.35])
        assert obs["n_empty"] == 1
        assert obs["d_occupied"] == pytest.approx(3.7)
        assert obs["d_empty"] == pytest.approx(3.35)

    def test_dilute_stage_one(self):
        obs = _layer_obs([3.5, 3.6], (1, 2))
        np.testing.assert_array_equal(obs["gallery_occupancy"], [2, 1])
        np.testing.assert_allclose(obs["gallery_width"], [3.6, 3.5])
        assert obs["n_empty"] == 0
        assert obs["d_occupied"] == pytest.approx(3.55)

    def test_no_intercalant(self):
        obs = _layer_obs([3.35, 3.35], ())
        assert obs["n_empty"] == 2
        assert np.isnan(obs["d_occupied"])
        assert obs["d_empty"] == pytest.approx(3.35)

    def test_three_galleries(self):
        obs = _layer_obs([3.4, 3.7, 3.5], (0, 3, 1))
        np.testing.assert_array_equal(obs["gallery_occupancy"], [3, 1, 0])
        np.testing.assert_allclose(obs["gallery_width"], [3.7, 3.5, 3.4])
        assert obs["d_occupied"] == pytest.approx(3.6)

    def test_host_atom_as_intercalant(self):
        pos, typ, cell = graphite([3.7, 3.7], (3, 3))
        geo = layer_geometry(pos, cell, typ == C)
        with pytest.raises(ValueError, match="also a host atom"):
            layer_observables(geo, np.ones(len(typ), dtype=bool))


BOND = 2.46 / np.sqrt(3)
DELTA = (A1 + A2) / 3.0  # one C-C bond: AA -> AB


def _registry(pos, typ, cell):
    geo = layer_geometry(pos, cell, typ == C)
    return registry_observables(pos, cell, geo, typ == LI)


def _shift_sheet(pos, typ, sheet, vec, size=18):
    pos = pos.copy()
    idx = np.flatnonzero(typ == C)[sheet * size : (sheet + 1) * size]
    pos[idx] += vec
    return pos


def _wrap(pos, cell):
    frac = np.mod(pos @ np.linalg.inv(cell), 1.0)
    return frac @ cell


class TestRegistryObservables:
    def test_aa(self):
        obs = _registry(*graphite([3.7, 3.7], (3, 3)))
        np.testing.assert_allclose(obs["stacking_shift"], 0.0, atol=1e-9)
        np.testing.assert_array_equal(obs["stacking"], [AA, AA])

    def test_ab(self):
        pos, typ, cell = graphite([3.35, 3.35])
        pos = _shift_sheet(pos, typ, 1, DELTA)
        obs = _registry(pos, typ, cell)
        # Gallery 0 is AB, gallery 1 (back to sheet 0) is AC: both "AB".
        np.testing.assert_allclose(obs["stacking_shift"], BOND)
        np.testing.assert_array_equal(obs["stacking"], [AB, AB])

    @pytest.mark.parametrize("frac, label", [(0.4, AA), (0.6, AB)])
    def test_intermediate_shift(self, frac, label):
        pos, typ, cell = graphite([3.5, 3.5])
        pos = _shift_sheet(pos, typ, 1, frac * DELTA)
        obs = _registry(pos, typ, cell)
        np.testing.assert_allclose(obs["stacking_shift"], frac * BOND)
        np.testing.assert_array_equal(obs["stacking"], [label, label])

    def test_tilted_c_shifts_the_boundary_gallery(self):
        # AA sheets in the cell, c tilted by one bond: the gallery that
        # crosses the cell boundary (gallery 1, empty) is AB.
        pos, typ, cell = graphite([3.7, 3.35], (3,))
        cell[2] += DELTA
        obs = _registry(pos, typ, cell)
        np.testing.assert_allclose(
            obs["stacking_shift"], [0.0, BOND], atol=1e-9
        )
        np.testing.assert_array_equal(obs["stacking"], [AA, AB])

    def test_follows_gallery_order(self):
        # Same structure with the Li in the boundary gallery: it comes first.
        pos, typ, cell = graphite([3.35, 3.7], (0, 3))
        cell[2] += DELTA
        obs = _registry(pos, typ, cell)
        np.testing.assert_array_equal(obs["stacking"], [AB, AA])

    def test_invariances(self):
        pos, typ, cell = graphite([3.7, 3.35], (3,))
        cell[2] += 0.7 * DELTA
        pos = _shift_sheet(pos, typ, 1, 0.3 * DELTA)
        ref = _registry(pos, typ, cell)
        # 120-degree basis, rigid translation, sheet 0 straddling s3 = 0,
        # positions wrapped into the cell.
        cell2 = cell.copy()
        cell2[1] = cell[1] - cell[0]
        moved = _wrap(pos + [0.8, -0.3, -0.02], cell2)
        obs = _registry(moved, typ, cell2)
        np.testing.assert_allclose(
            obs["stacking_shift"], ref["stacking_shift"]
        )
        np.testing.assert_array_equal(obs["stacking"], ref["stacking"])


ALPHA = [(0, 0), (1, 1), (2, 2)]
BETA = [(1, 0), (2, 1), (0, 2)]
ONE_EACH = [(0, 0), (1, 0), (0, 1)]
TWO_ONE = [(0, 0), (1, 1), (1, 0)]


def _lic(sites, spacings=(3.7, 3.7), z0=0.0, reps=3):
    """Graphite with Li on the given hexagons ``(i, j)`` of each gallery,
    at mid-gallery; hexagon (i, j) is on sublattice ``(i - j) mod 3``."""
    pos, typ, cell = graphite(list(spacings), z0=z0, reps=reps)
    tops = z0 + np.concatenate([[0.0], np.cumsum(spacings)])
    li = [
        HOLLOW + i * A1 + j * A2 + [0.0, 0.0, 0.5 * (tops[g] + tops[g + 1])]
        for g, gallery in enumerate(sites)
        for i, j in gallery
    ]
    pos = np.concatenate([pos, li]) if li else pos
    typ = np.concatenate([typ, np.full(len(li), LI, dtype=typ.dtype)])
    return pos, typ, cell


def _li_order(pos, typ, cell, **kw):
    geo = layer_geometry(pos, cell, typ == C)
    return li_order_observables(pos, cell, geo, typ == LI, **kw)


class TestLiOrderObservables:
    def test_alpha_alpha(self):
        obs = _li_order(*_lic([ALPHA, ALPHA]))
        np.testing.assert_allclose(obs["li_order"], [1.0, 1.0])
        assert obs["li_alignment"] == pytest.approx(1.0)
        assert obs["li_phase_shift"] == pytest.approx(0.0, abs=1e-6)

    def test_alpha_beta(self):
        obs = _li_order(*_lic([ALPHA, BETA]))
        np.testing.assert_allclose(obs["li_order"], [1.0, 1.0])
        assert obs["li_alignment"] == pytest.approx(-0.5)
        assert obs["li_phase_shift"] == pytest.approx(120.0)

    @pytest.mark.parametrize(
        "sites, m, alignment",
        [
            # No preferred sublattice: no alignment, and no phase at all.
            (ONE_EACH, 0.0, 0.0),
            # Z = (2 + w) / 3 next to Z = 1: C = (2 + cos 120) / 3.
            (TWO_ONE, 1 / np.sqrt(3), 0.5),
        ],
    )
    def test_partial_order(self, sites, m, alignment):
        obs = _li_order(*_lic([sites, ALPHA]))
        np.testing.assert_allclose(
            np.sort(obs["li_order"]), [m, 1.0], atol=1e-9
        )
        assert obs["li_alignment"] == pytest.approx(alignment, abs=1e-9)
        assert np.isnan(obs["li_phase_shift"]) == (m == 0.0)

    def test_off_site_atoms_reach_minus_one(self):
        # Gallery 1 shifted by 3/2 a2, off the hexagon centres:
        # K . (3/2 a2) = 180 deg, so C = -1, below the alpha-beta -1/2.
        pos, typ, cell = _lic([ALPHA, ALPHA])
        top = np.flatnonzero(typ == LI)[3:]
        pos[top] += 1.5 * A2
        obs = _li_order(pos, typ, cell)
        assert obs["li_alignment"] == pytest.approx(-1.0)
        assert obs["li_phase_shift"] == pytest.approx(180.0)

    def test_single_atom_gallery(self):
        obs = _li_order(*_lic([[(0, 0)], [(0, 0), (1, 1)]]))
        assert obs["li_order"][0] == pytest.approx(1.0)
        assert np.isnan(obs["li_order"][1])
        assert np.isnan(obs["li_phase_shift"])
        assert np.isnan(obs["li_alignment"])

    def test_stage_two(self):
        obs = _li_order(*_lic([[], ALPHA], spacings=(3.35, 3.7)))
        assert obs["li_order"][0] == pytest.approx(1.0)
        assert np.isnan(obs["li_order"][1])
        assert np.isnan(obs["li_phase_shift"])

    def test_boundary_gallery_unwrapped(self):
        # Gallery 1 crosses the top of the cell; with c tilted, wrapping its
        # Li back by c must not change the phase.
        pos, typ, cell = _lic([ALPHA, BETA], z0=2.0)
        ref = _li_order(pos, typ, cell)
        cell[2] += 0.5 * A1
        wrapped = np.mod(pos @ np.linalg.inv(cell), 1.0) @ cell
        obs = _li_order(wrapped, typ, cell)
        np.testing.assert_allclose(obs["li_order"], ref["li_order"])
        assert obs["li_phase_shift"] == pytest.approx(ref["li_phase_shift"])
        assert obs["li_phase_shift"] == pytest.approx(120.0)
        assert obs["li_alignment"] == pytest.approx(-0.5)

    def test_needs_multiples_of_three(self):
        pos, typ, cell = _lic([[(0, 0), (1, 1)], []], reps=2)
        with pytest.raises(ValueError, match="multiples of 3"):
            _li_order(pos, typ, cell, in_plane_repeats=(2, 2))
