"""Sheet and gallery identification for layered moves."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest
from scipy import stats

from jaxrens.backends.lj import create_lj
from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.resolve import _resolve_sheets, resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.moves import (
    GalleryHopMoveSpec,
    LayerSlideMoveSpec,
    ShearMoveSpec,
    SingleAtomMoveSpec,
    StackingShiftMoveSpec,
    StretchMoveSpec,
    VolumeMoveSpec,
)
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import single_atom
from jaxrens.sampling.moves.layered_moves import (
    SheetInfo,
    build_gallery_hop_kernel,
    build_layer_slide_kernel,
    build_stacking_shift_kernel,
    check_sheets,
    fractional_heights,
    galleries,
    gallery_of,
    make_sheet_id_initializer,
    sheet_heights,
)
from jaxrens.sampling.mwg import build_mwg
from jaxrens.utils.cell import c_in_plane_reduced

C, LI, NA, B = 0, 1, 2, 3
A1 = np.array([2.46, 0.0, 0.0])
A2 = np.array([1.23, 2.46 * np.sqrt(3) / 2, 0.0])
HOLLOW = 2.0 * (A1 + A2) / 3.0


def _graphite(spacings, li_per_gallery=(), z0=0.0, reps=3):
    """AA-stacked graphite, ``reps`` x ``reps`` in plane.

    Args:
        spacings: Gallery heights (Angstrom), one per sheet.
        li_per_gallery: Number of Li at hollow sites in mid-gallery, per
            gallery (missing entries mean none).
        z0: Height of sheet 0.

    Returns:
        ``(positions, types, cell)`` as numpy arrays; C atoms come first,
        sheet by sheet.
    """
    a, b = reps * A1, reps * A2
    cell = np.array([a, b, [0.0, 0.0, float(sum(spacings))]])
    base = [i * A1 + j * A2 for i in range(reps) for j in range(reps)]
    tops = np.concatenate(
        [np.cumsum([0.0, *spacings[:-1]]) + z0, [z0 + sum(spacings)]]
    )
    pos, typ = [], []
    for l in range(len(spacings)):
        for p in base:
            for d in (np.zeros(3), (A1 + A2) / 3.0):
                pos.append(p + d + [0.0, 0.0, tops[l]])
                typ.append(C)
    for g, n_li in enumerate(li_per_gallery):
        mid = 0.5 * (tops[g] + tops[g + 1])
        for p in base[:n_li]:
            pos.append(p + HOLLOW + [0.0, 0.0, mid])
            typ.append(LI)
    return np.array(pos), np.array(typ, dtype=np.int32), cell


def _sheet_id(pos, typ, cell):
    init = make_sheet_id_initializer((C,))
    return np.asarray(init(jnp.asarray(pos), jnp.asarray(typ), cell))


class TestSheetId:
    def test_bilayer(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        sid = _sheet_id(pos, typ, cell)
        np.testing.assert_array_equal(sid[:18], 0)
        np.testing.assert_array_equal(sid[18:36], 1)
        np.testing.assert_array_equal(sid[36:], -1)

    @pytest.mark.parametrize("n", [3, 4])
    def test_labels_increase_with_height(self, n):
        pos, typ, cell = _graphite([3.4] * n, z0=0.5)
        sid = _sheet_id(pos, typ, cell)
        np.testing.assert_array_equal(sid, np.repeat(np.arange(n), 18))

    def test_sheet_straddling_cell_boundary(self):
        pos, typ, cell = _graphite([3.35, 3.35])
        # Sheet 0 at z = 0: push half of its atoms just below the boundary.
        pos[:9, 2] -= 0.05
        sid = _sheet_id(pos, typ, cell)
        np.testing.assert_array_equal(sid[:18], 0)
        np.testing.assert_array_equal(sid[18:], 1)

    def test_atom_order_is_irrelevant(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(2, 1))
        perm = np.random.default_rng(0).permutation(len(typ))
        sid = _sheet_id(pos, typ, cell)
        sid_perm = _sheet_id(pos[perm], typ[perm], cell)
        np.testing.assert_array_equal(sid_perm, sid[perm])

    def test_unwrapped_positions(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        shifts = np.random.default_rng(1).integers(-2, 3, len(typ))
        unwrapped = pos + shifts[:, None] * cell[2]
        np.testing.assert_array_equal(
            _sheet_id(unwrapped, typ, cell), _sheet_id(pos, typ, cell)
        )

    def test_tilted_c(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        tilted = cell.copy()
        tilted[2] += 0.7 * A1 + 0.4 * A2
        s3 = fractional_heights(jnp.asarray(pos), tilted)
        np.testing.assert_allclose(
            s3, fractional_heights(jnp.asarray(pos), cell), atol=1e-6
        )
        np.testing.assert_array_equal(
            _sheet_id(pos, typ, tilted), _sheet_id(pos, typ, cell)
        )


class TestGalleries:
    def _setup(self, spacings, li):
        pos, typ, cell = _graphite(spacings, li_per_gallery=li, z0=0.2)
        sid = jnp.asarray(_sheet_id(pos, typ, cell))
        s3 = fractional_heights(jnp.asarray(pos), cell)
        heights = sheet_heights(s3, sid, len(spacings))
        return pos, typ, cell, sid, s3, heights

    def test_heights_widths_centres(self):
        *_, heights = self._setup([3.5, 3.2], (3,))
        h = 6.7
        np.testing.assert_allclose(heights, [0.2 / h, 3.7 / h], atol=1e-6)
        lower, width, centre = galleries(heights)
        np.testing.assert_allclose(width * h, [3.5, 3.2], atol=1e-5)
        np.testing.assert_allclose(
            centre * h, [0.2 + 1.75, 3.7 + 1.6], atol=1e-5
        )

    def test_li_assigned_to_their_gallery(self):
        pos, typ, cell, sid, s3, heights = self._setup([3.5, 3.2], (2, 1))
        lower, width, _ = galleries(heights)
        g = np.asarray(gallery_of(s3, lower, width))
        np.testing.assert_array_equal(g[typ == LI], [0, 0, 1])

    def test_gallery_across_cell_boundary(self):
        # Sheets at 1.0 and 4.5 A in a 6.7 A cell: gallery 1 wraps around.
        pos, typ, cell = _graphite([3.5, 3.2], li_per_gallery=(0, 2), z0=1.0)
        pos[typ == LI, 2] = [6.6, 0.1]  # either side of the boundary
        sid = jnp.asarray(_sheet_id(pos, typ, cell))
        s3 = fractional_heights(jnp.asarray(pos), cell)
        lower, width, _ = galleries(sheet_heights(s3, sid, 2))
        g = np.asarray(gallery_of(s3, lower, width))
        np.testing.assert_array_equal(g[typ == LI], [1, 1])

    def test_in_plane_translation_changes_nothing(self):
        pos, typ, cell, sid, s3, heights = self._setup([3.5, 3.2], (2, 1))
        moved = pos.copy()
        moved[:18] += 0.37 * A1 - 0.81 * A2  # slide sheet 0
        moved[typ == LI] += 1.3 * A2  # and the Li
        s3_new = fractional_heights(jnp.asarray(moved), cell)
        np.testing.assert_allclose(s3_new, s3, atol=1e-6)
        np.testing.assert_allclose(
            sheet_heights(s3_new, sid, 2), heights, atol=1e-6
        )


class TestCheckSheets:
    def _batch(self, *frames):
        pos, typ, cell = zip(*frames)
        return np.stack(pos), np.stack(typ), np.stack(cell)

    def test_returns_number_of_sheets(self):
        f2 = _graphite([3.35, 3.35], li_per_gallery=(3,))
        assert check_sheets(*self._batch(f2, f2), (C,)) == SheetInfo(2, 18)
        f4 = _graphite([3.4] * 4)
        assert check_sheets(*self._batch(f4), (C,)) == SheetInfo(4, 18)

    def test_replica_axes_are_flattened(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        stacked = np.broadcast_to(pos, (2, 3, *pos.shape))
        cells = np.broadcast_to(cell, (2, 3, 3, 3))
        assert check_sheets(stacked, typ, cells, (C,)) == SheetInfo(2, 18)
        # Types with the same leading axes, as the multi-replica resolver
        # passes them.
        types = np.broadcast_to(typ, (2, 3, len(typ)))
        assert check_sheets(stacked, types, cells, (C,)) == SheetInfo(2, 18)

    def test_non_periodic_cell(self):
        pos, typ, cell = _graphite([3.35, 3.35])
        with pytest.raises(ValueError, match="periodic"):
            check_sheets(pos[None], typ[None], np.zeros((1, 3, 3)), (C,))

    def test_single_sheet(self):
        pos, typ, cell = _graphite([6.7])
        with pytest.raises(ValueError, match="at least 2"):
            check_sheets(pos[None], typ[None], cell[None], (C,))

    def test_sheets_of_different_size(self):
        pos, typ, cell = _graphite([3.35, 3.35])
        with pytest.raises(ValueError, match="different numbers"):
            check_sheets(pos[None, 1:], typ[None, 1:], cell[None], (C,))

    def test_number_of_sheets_differs_between_walkers(self):
        pos, typ, cell = _graphite([3.35, 3.35])
        # Same 36 C regrouped into three sheets of 12.
        pos3 = pos.copy()
        pos3[:, 2] = np.repeat([0.0, 2.2, 4.4], 12)
        with pytest.raises(ValueError, match="same number of sheets"):
            check_sheets(
                *self._batch((pos, typ, cell), (pos3, typ, cell)), (C,)
            )

    def test_sheet_not_flat(self):
        pos, typ, cell = _graphite([3.35, 3.35])
        pos[0, 2] += 0.3
        with pytest.raises(ValueError, match="not flat"):
            check_sheets(pos[None], typ[None], cell[None], (C,))


class TestStateField:
    def _mwg(self, init_extra):
        desc = MoveKernel(
            name="single_atom",
            build_kernel=single_atom.build_kernel,
            extra_state_fields=init_extra,
        )
        return build_mwg(create_harmonic(), [desc])

    def test_sheet_id_built_from_walker_cell(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        init_fn, *_ = self._mwg(
            {"sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))}
        )
        state = init_fn(
            jnp.asarray(pos), jnp.asarray(typ), 0.0, jnp.asarray(cell)
        )
        np.testing.assert_array_equal(
            state.sheet_id, _sheet_id(pos, typ, cell)
        )

    def test_every_initializer_receives_the_cell(self):
        pos, typ, cell = _graphite([3.35, 3.35])
        seen = {}

        def init(positions, types, cell):
            seen["cell"] = cell
            return jnp.zeros_like(positions)

        init_fn, *_ = self._mwg({"direction": (jnp.ndarray, init)})
        init_fn(jnp.asarray(pos), jnp.asarray(typ), 0.0, jnp.asarray(cell))
        np.testing.assert_allclose(seen["cell"], cell, rtol=1e-6)

    def test_clone_copies_sheet_id(self):
        frames = [
            _graphite([3.35, 3.35], li_per_gallery=(3,)),
            _graphite([3.35, 3.35], li_per_gallery=(3,), z0=3.0),
        ]
        init_fn, *_ = self._mwg(
            {"sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))}
        )
        walkers = [
            init_fn(jnp.asarray(p), jnp.asarray(t), 0.0, jnp.asarray(c))
            for p, t, c in frames
        ]
        pop = jax.tree.map(lambda *x: jnp.stack(x), *walkers)
        cloned = jax.tree.map(lambda x: x[jnp.array([1, 1])], pop)
        np.testing.assert_array_equal(cloned.sheet_id[0], walkers[1].sheet_id)
        np.testing.assert_array_equal(
            cloned.positions[0], walkers[1].positions
        )


# ---------------------------------------------------------------------------
# gallery_hop
# ---------------------------------------------------------------------------


def _hop_setup(spacings=(3.5, 3.2), li=(2, 1), tilt=None, sigma=0.0):
    """Kernel and state for gallery_hop on a graphite bilayer with Li."""
    pos, typ, cell = _graphite(list(spacings), li_per_gallery=li, z0=0.2)
    if tilt is not None:
        cell = cell.copy()
        cell[2] += tilt
    desc = MoveKernel(
        name="gallery_hop_Li",
        build_kernel=build_gallery_hop_kernel,
        kernel_kwargs=dict(
            species=(LI,),
            host_codes=(C,),
            n_sheets=len(spacings),
            in_plane_repeats=(3, 3),
        ),
        extra_state_fields={
            "sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))
        },
    )
    init_fn, _, (step,) = build_mwg(create_harmonic(), [desc])
    state = init_fn(jnp.asarray(pos), jnp.asarray(typ), 0.0, jnp.asarray(cell))
    state = state.set(step_sizes=jnp.full(1, sigma))
    return step, state, cell


def _frac(positions, cell):
    return np.asarray(positions) @ np.linalg.inv(cell)


def _propose(step, state, n, seed=0):
    keys = jax.random.split(jax.random.key(seed), n)
    return jax.vmap(lambda k: step(state, k, 1e12))(keys)


def _decode_hops(state, new, cell, n_sheets=2):
    """(atom, dg, i, j) of each accepted proposal, from the displacement."""
    s3 = np.asarray(fractional_heights(state.positions, cell))
    lower, width, _ = galleries(
        sheet_heights(jnp.asarray(s3), state.sheet_id, n_sheets)
    )
    g_old = np.asarray(gallery_of(jnp.asarray(s3), lower, width))
    out = []
    for pos in np.asarray(new.positions):
        moved = np.flatnonzero(np.any(pos != state.positions, axis=1))
        assert len(moved) == 1
        a = moved[0]
        s3_new = fractional_heights(jnp.asarray(pos[a : a + 1]), cell)
        g_new = int(gallery_of(s3_new, lower, width)[0])
        ds = _frac(pos[a] - np.asarray(state.positions[a]), cell)
        i, j = np.rint(3 * ds[:2]).astype(int) % 3
        out.append((a, (g_new - g_old[a]) % n_sheets, i, j))
    return out


class TestGalleryHopKernel:
    @pytest.mark.parametrize("tilt", [None, 0.6 * A1 + 0.3 * A2])
    def test_only_li_move_by_lattice_vectors(self, tilt):
        step, state, cell = _hop_setup(tilt=tilt)
        new, info = _propose(step, state, 200)
        assert bool(jnp.all(info.accepted))
        typ = np.asarray(state.types)
        for a, _, _, _ in _decode_hops(state, new, cell):
            assert typ[a] == LI
        # In-plane fractional shifts are multiples of 1/3 (sigma = 0).
        ds = _frac(np.asarray(new.positions - state.positions), cell)
        np.testing.assert_allclose(
            3 * ds[..., :2], np.rint(3 * ds[..., :2]), atol=1e-4
        )

    def test_offset_from_gallery_centre_is_kept(self):
        step, state, cell = _hop_setup(spacings=(3.5, 3.2))
        li = np.flatnonzero(np.asarray(state.types) == LI)
        state = state.set(positions=state.positions.at[li, 2].add(0.15))
        s3 = fractional_heights(state.positions, cell)
        _, width, centre = galleries(sheet_heights(s3, state.sheet_id, 2))
        lower = galleries(sheet_heights(s3, state.sheet_id, 2))[0]
        new, info = _propose(step, state, 200, seed=1)
        assert bool(jnp.all(info.accepted))
        h = float(cell[2, 2])
        for pos in np.asarray(new.positions):
            s3n = fractional_heights(jnp.asarray(pos), cell)
            g = gallery_of(s3n[li], lower, width)
            off = (np.asarray(s3n[li] - centre[g]) + 0.5) % 1.0 - 0.5
            np.testing.assert_allclose(off * h, 0.15, atol=1e-4)

    def test_hops_are_uniform_and_never_null(self):
        step, state, cell = _hop_setup(li=(1, 0))
        new, info = _propose(step, state, 3400, seed=2)
        hops = [h[1:] for h in _decode_hops(state, new, cell)]
        assert (0, 0, 0) not in hops
        counts = np.zeros((2, 3, 3), int)
        for dg, i, j in hops:
            counts[dg, i, j] += 1
        observed = counts.ravel()[1:]
        assert np.all(observed > 0)
        assert stats.chisquare(observed).pvalue > 1e-3

    def test_reverse_hop_returns_to_start(self):
        step, state, cell = _hop_setup(spacings=(3.5, 3.2), li=(1, 0))
        new, _ = _propose(step, state, 30, seed=3)
        s3 = fractional_heights(state.positions, cell)
        heights = sheet_heights(s3, state.sheet_id, 2)
        lower, width, centre = galleries(heights)
        for pos, (a, dg, i, j) in zip(
            np.asarray(new.positions), _decode_hops(state, new, cell)
        ):
            # Sheets unchanged, so the gallery centres are too.
            s3n = fractional_heights(jnp.asarray(pos), cell)
            np.testing.assert_allclose(
                sheet_heights(s3n, state.sheet_id, 2), heights, atol=1e-6
            )
            g1 = int(gallery_of(s3n[a : a + 1], lower, width)[0])
            g0 = (g1 - dg) % 2
            # Reverse hop (-dg, -i, -j) mod (2, 3, 3): back to the start up
            # to a lattice vector of the cell.
            back = (
                float(centre[g0] - centre[g1]) * cell[2]
                + ((-i) % 3) * cell[0] / 3
                + ((-j) % 3) * cell[1] / 3
            )
            diff = _frac(pos[a] + back - np.asarray(state.positions[a]), cell)
            np.testing.assert_allclose(diff, np.rint(diff), atol=1e-4)

    def test_noise_leaving_target_gallery_is_rejected(self):
        step, state, _ = _hop_setup(sigma=5.0)
        new, info = _propose(step, state, 300, seed=4)
        rejected = ~np.asarray(info.accepted)
        assert rejected.any()
        assert np.all(np.asarray(info.reject_reason)[rejected] == 3)
        np.testing.assert_array_equal(
            np.asarray(new.positions)[rejected],
            np.broadcast_to(state.positions, new.positions.shape)[rejected],
        )

    def test_walker_without_species_rejects(self):
        step, state, _ = _hop_setup(li=())
        new, info = step(state, jax.random.key(5), 1e12)
        assert not bool(info.accepted)
        assert int(info.reject_reason) == 3
        np.testing.assert_array_equal(new.positions, state.positions)


class TestGalleryHopWiring:
    _SYMBOLS = {0: "C", 1: "Li"}

    def test_auto_name_and_species(self):
        spec = GalleryHopMoveSpec(species="Li")
        assert spec._effective_name() == "gallery_hop_Li"
        assert spec.species == ("Li",)
        assert (
            GalleryHopMoveSpec(species="Li", name="hop")._effective_name()
            == "hop"
        )

    def test_species_is_required(self):
        with pytest.raises(pydantic.ValidationError):
            GalleryHopMoveSpec()

    def test_descriptor(self):
        desc = GalleryHopMoveSpec(species="Li").to_descriptor(
            symbol_map=self._SYMBOLS, sheets=SheetInfo(2, 18)
        )
        assert desc.name == "gallery_hop_Li"
        assert desc.kernel_kwargs == {
            "species": (1,),
            "host_codes": (0,),
            "n_sheets": 2,
            "in_plane_repeats": (3, 3),
        }
        assert set(desc.extra_state_fields) == {"sheet_id"}
        assert desc.reject_reasons == {"energy", "prior"}

    def test_sheet_size_must_match_repeats(self):
        spec = GalleryHopMoveSpec(species="Li", in_plane_repeats=(2, 3))
        with pytest.raises(ValueError, match="host atoms per sheet"):
            spec.to_descriptor(
                symbol_map=self._SYMBOLS, sheets=SheetInfo(2, 18)
            )

    def test_sheets_are_required(self):
        with pytest.raises(ValueError, match="sheet layout"):
            GalleryHopMoveSpec(species="Li").to_descriptor(
                symbol_map=self._SYMBOLS
            )

    def _init(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        return SimpleNamespace(
            initial_positions=pos[None],
            initial_types=typ[None],
            initial_cells=cell[None],
            symbol_map=self._SYMBOLS,
        )

    def test_resolver_skips_runs_without_layered_moves(self):
        root = SimpleNamespace(moves=[SingleAtomMoveSpec()])
        assert _resolve_sheets(root, None) is None

    def test_resolver_returns_layout(self):
        root = SimpleNamespace(moves=[GalleryHopMoveSpec(species="Li")])
        assert _resolve_sheets(root, self._init()) == SheetInfo(2, 18)

    def test_resolver_rejects_mixed_hosts(self):
        root = SimpleNamespace(
            moves=[
                GalleryHopMoveSpec(species="Li"),
                GalleryHopMoveSpec(species="Li", host_species=["C", "B"]),
            ]
        )
        with pytest.raises(ValueError, match="disagree on host_species"):
            _resolve_sheets(root, self._init())

    def test_resolver_rejects_absent_host(self):
        root = SimpleNamespace(
            moves=[GalleryHopMoveSpec(species="Li", host_species="B")]
        )
        with pytest.raises(ValueError, match="not present"):
            _resolve_sheets(root, self._init())

    def test_full_resolve(self, tmp_path):
        import ase.io
        from ase import Atoms

        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        atoms = Atoms(
            symbols=["C" if t == C else "Li" for t in typ],
            positions=pos,
            cell=cell,
            pbc=True,
        )
        path = tmp_path / "walkers.extxyz"
        ase.io.write(str(path), [atoms.copy() for _ in range(4)])
        root = RootSpec.model_validate(
            {
                "run": {"n_live": 4, "max_iterations": 2, "seed": 0},
                "moves": [
                    {"type": "single_atom", "species": "Li"},
                    {"type": "gallery_hop", "species": "Li"},
                    {"type": "layer_slide", "species": "C", "subdivisions": 3},
                    {"type": "layer_slide", "species": "Li"},
                ],
                "backend": {"backend_type": "harmonic"},
                "output": {"format": "none", "working_dir": str(tmp_path)},
                "init": {
                    "start_walker_set": str(path),
                    "random_initialise_pos": False,
                    "random_initialise_cell": False,
                },
            }
        )
        resolved = resolve(root)
        hop = resolved.move_descriptors[1]
        assert hop.name == "gallery_hop_Li"
        assert hop.kernel_kwargs["n_sheets"] == 2
        assert "sheets" not in resolved.move_descriptors[0].kernel_kwargs
        names = [d.name for d in resolved.move_descriptors]
        assert names[2:] == ["layer_slide_C", "layer_slide_Li"]
        slide_c, slide_li = resolved.move_descriptors[2:]
        assert slide_c.kernel_kwargs["slide_host"] is True
        assert slide_li.kernel_kwargs["slide_host"] is False
        # Three layered moves, one shared sheet_id field.
        init_fn, *_ = build_mwg(
            create_harmonic(), list(resolved.move_descriptors)
        )
        state = init_fn(
            jnp.asarray(pos), jnp.asarray(typ), 0.0, jnp.asarray(cell)
        )
        np.testing.assert_array_equal(
            state.sheet_id, _sheet_id(pos, typ, cell)
        )


class TestLayeredMovesWithLJ:
    """gallery_hop, layer_slide and stacking_shift in a short NS run on a
    periodic LJ bilayer."""

    def test_li_change_gallery_and_sheets_stay_intact(self):
        from jaxrens.sampling.nested_sampling import run_ns

        # C-C minimum at the bond length.  Li sigma 3 A: the three Li of the
        # full gallery, 2.46 A apart, repel each other, so moving one to the
        # empty gallery lowers the energy.
        backend = create_lj(
            epsilon=(1.0, 1.0), sigma=(1.42 / 2 ** (1 / 6), 3.0), cutoff=3.0
        )
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        n_live = 8
        rng = np.random.default_rng(0)
        positions = pos + 0.02 * rng.standard_normal((n_live, *pos.shape))
        cells = np.broadcast_to(cell, (n_live, 3, 3))
        hop = MoveKernel(
            name="gallery_hop_Li",
            build_kernel=build_gallery_hop_kernel,
            kernel_kwargs=dict(
                species=(LI,),
                host_codes=(C,),
                n_sheets=2,
                in_plane_repeats=(3, 3),
            ),
            extra_state_fields={
                "sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))
            },
            step_size=0.05,
        )
        local = MoveKernel(
            name="single_atom_Li",
            build_kernel=single_atom.build_kernel,
            kernel_kwargs={"species": (LI,)},
            step_size=0.1,
        )
        slide = MoveKernel(
            name="layer_slide_C",
            build_kernel=build_layer_slide_kernel,
            kernel_kwargs=dict(
                species=(C,),
                host_codes=(C,),
                n_sheets=2,
                in_plane_repeats=(3, 3),
                subdivisions=3,
                slide_host=True,
            ),
            extra_state_fields={
                "sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))
            },
            step_size=0.02,
        )
        shift = MoveKernel(
            name="stacking_shift_3",
            build_kernel=build_stacking_shift_kernel,
            kernel_kwargs=dict(
                host_codes=(C,),
                n_sheets=2,
                in_plane_repeats=(3, 3),
                subdivisions=3,
                n_atoms=len(typ),
                max_vol_per_atom=1e4,
                min_vol_per_atom=0.1,
                min_aspect=0.5,
                restrict_c_in_plane=True,
            ),
            extra_state_fields={
                "sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))
            },
            step_size=0.02,
        )
        init_fn, step_fn, _ = build_mwg(backend, [hop, local, slide, shift])
        types = jnp.asarray(typ)
        energies = jax.vmap(lambda p, c: backend(p, types, c, 0).energy)(
            jnp.asarray(positions), jnp.asarray(cells)
        )

        result = run_ns(
            positions=jnp.asarray(positions),
            types=types,
            energies=energies,
            cells=jnp.asarray(cells),
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=jax.random.key(0),
            max_iterations=30,
            n_mcmc_steps=10,
            move_descriptors=[hop, local, slide, shift],
        )
        assert np.isfinite(float(result["log_evidence"]))

        final_pos = np.asarray(result["positions"])
        final_cells = np.asarray(result["cells"])
        assert check_sheets(final_pos, typ, final_cells, (C,)) == SheetInfo(
            2, 18
        )
        for c in final_cells:
            assert bool(c_in_plane_reduced(jnp.asarray(c)))
        occupancy = set()
        for p, c in zip(final_pos, final_cells):
            sid = jnp.asarray(_sheet_id(p, typ, c))
            s3 = fractional_heights(jnp.asarray(p), c)
            lower, width, _ = galleries(sheet_heights(s3, sid, 2))
            g = np.asarray(gallery_of(s3, lower, width))[typ == LI]
            occupancy.add(tuple(np.bincount(g, minlength=2)))
        # Every walker started with (3, 0); hops must have changed that.
        assert occupancy != {(3, 0)}


# ---------------------------------------------------------------------------
# layer_slide
# ---------------------------------------------------------------------------


def _slide_setup(
    species,
    slide_host,
    k=1,
    li=(3, 3),
    tilt=None,
    sigma=0.0,
    host=(C,),
    retype=None,
):
    """Kernel and state for layer_slide on a graphite bilayer.

    ``retype`` maps atom index -> new type code (Na in a gallery, B dopant).
    """
    pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=li)
    for idx, code in (retype or {}).items():
        typ[idx] = code
    if tilt is not None:
        cell = cell.copy()
        cell[2] += tilt
    desc = MoveKernel(
        name="layer_slide",
        build_kernel=build_layer_slide_kernel,
        kernel_kwargs=dict(
            species=species,
            host_codes=host,
            n_sheets=2,
            in_plane_repeats=(3, 3),
            subdivisions=k,
            slide_host=slide_host,
        ),
        extra_state_fields={
            "sheet_id": (jnp.ndarray, make_sheet_id_initializer(host))
        },
    )
    init_fn, _, (step,) = build_mwg(create_harmonic(), [desc])
    state = init_fn(jnp.asarray(pos), jnp.asarray(typ), 0.0, jnp.asarray(cell))
    state = state.set(step_sizes=jnp.full(1, sigma))
    return step, state, cell


def _moved(state, new):
    """Boolean (n_proposals, n_atoms) mask of displaced atoms."""
    return np.any(np.asarray(new.positions) != np.asarray(state.positions), -1)


def _grid_index(state, new, cell, n):
    """(i, j) on an n x n grid of each proposal's common translation."""
    d = np.asarray(new.positions - state.positions)
    moved = _moved(state, new)
    out = []
    for dk, mk in zip(d, moved):
        t = dk[mk]
        np.testing.assert_allclose(
            t, np.broadcast_to(t[0], t.shape), atol=1e-5
        )
        ds = _frac(t[0], cell)
        out.append(tuple(np.rint(n * ds[:2]).astype(int) % n))
        np.testing.assert_allclose(n * ds[:2], np.rint(n * ds[:2]), atol=1e-4)
    return out


class TestLayerSlideKernel:
    def test_one_sheet_moves_rigidly(self):
        step, state, cell = _slide_setup((C,), True, k=3)
        new, info = _propose(step, state, 200)
        assert bool(jnp.all(info.accepted))
        moved = _moved(state, new)
        sid = np.asarray(state.sheet_id)
        for mk in moved:
            assert mk.sum() == 18
            assert len(set(sid[mk])) == 1
        _grid_index(state, new, cell, 9)  # rigid and on the grid
        # Both sheets are picked.
        first = [sid[mk][0] for mk in moved]
        assert set(first) == {0, 1}
        np.testing.assert_array_equal(
            np.asarray(new.cell), np.broadcast_to(state.cell, new.cell.shape)
        )

    @pytest.mark.parametrize("tilt", [None, 0.6 * A1 + 0.3 * A2])
    def test_heights_unchanged(self, tilt):
        step, state, cell = _slide_setup((C,), True, k=3, tilt=tilt, sigma=0.3)
        new, _ = _propose(step, state, 50, seed=1)
        s3 = fractional_heights(state.positions, cell)
        for p in np.asarray(new.positions):
            np.testing.assert_allclose(
                fractional_heights(jnp.asarray(p), cell), s3, atol=1e-5
            )

    def test_grid_is_uniform_and_reaches_ab(self):
        step, state, cell = _slide_setup((C,), True, k=3)
        new, _ = _propose(step, state, 8000, seed=2)
        idx = _grid_index(state, new, cell, 9)
        assert (0, 0) not in idx
        counts = np.zeros((9, 9), int)
        for i, j in idx:
            counts[i, j] += 1
        observed = counts.ravel()[1:]
        assert np.all(observed > 0)
        assert stats.chisquare(observed).pvalue > 1e-3
        # (1, 1) on the k = 3 grid is (a1 + a2) / 3: AA -> AB.
        assert counts[1, 1] > 0 and counts[2, 2] > 0

    def test_continuous_slide_is_a_small_in_plane_gaussian(self):
        sigma = 0.05
        step, state, cell = _slide_setup((C,), True, k=0, sigma=sigma)
        new, info = _propose(step, state, 2000, seed=9)
        assert bool(jnp.all(info.accepted))
        d = np.asarray(new.positions - state.positions)
        moved = _moved(state, new)
        sid = np.asarray(state.sheet_id)
        ts = []
        for dk, mk in zip(d, moved):
            assert mk.sum() == 18 and len(set(sid[mk])) == 1
            t = dk[mk]
            np.testing.assert_allclose(
                t, np.broadcast_to(t[0], t.shape), atol=1e-6
            )
            ts.append(t[0])
        ts = np.array(ts)
        np.testing.assert_allclose(ts[:, 2], 0.0, atol=1e-6)  # in plane
        # Centred Gaussian of width sigma in each in-plane direction: no
        # grid jump.
        np.testing.assert_allclose(
            ts[:, :2].mean(axis=0), 0.0, atol=4 * sigma / np.sqrt(2000)
        )
        np.testing.assert_allclose(ts[:, :2].std(axis=0), sigma, rtol=0.1)
        assert np.abs(ts[:, :2]).max() < 6 * sigma

    def test_continuous_slide_of_a_gallery(self):
        step, state, _ = _slide_setup((LI,), False, k=0, li=(3, 3), sigma=0.05)
        new, info = _propose(step, state, 100, seed=10)
        typ = np.asarray(state.types)
        for mk in _moved(state, new):
            assert mk.sum() == 3 and np.all(typ[mk] == LI)

    def test_reverse_translation_returns_to_start(self):
        step, state, cell = _slide_setup((C,), True, k=3)
        new, _ = _propose(step, state, 20, seed=3)
        for p, mk, (i, j) in zip(
            np.asarray(new.positions),
            _moved(state, new),
            _grid_index(state, new, cell, 9),
        ):
            back = ((-i) % 9) * cell[0] / 9 + ((-j) % 9) * cell[1] / 9
            diff = _frac(p[mk] + back - np.asarray(state.positions)[mk], cell)
            np.testing.assert_allclose(diff, np.rint(diff), atol=1e-4)

    def test_gallery_li_move_together(self):
        step, state, cell = _slide_setup((LI,), False, k=1, li=(3, 3))
        new, info = _propose(step, state, 200, seed=4)
        assert bool(jnp.all(info.accepted))
        typ = np.asarray(state.types)
        li = np.flatnonzero(typ == LI)
        for mk in _moved(state, new):
            assert np.all(typ[mk] == LI)
            assert mk.sum() == 3
            assert set(np.flatnonzero(mk)) in (set(li[:3]), set(li[3:]))
        _grid_index(state, new, cell, 3)  # host lattice vectors only

    def test_empty_gallery_rejects(self):
        step, state, _ = _slide_setup((LI,), False, li=(3, 0))
        new, info = _propose(step, state, 400, seed=5)
        acc = np.asarray(info.accepted)
        assert 0.35 < acc.mean() < 0.65
        assert np.all(np.asarray(info.reject_reason)[~acc] == 3)
        assert not _moved(state, new)[~acc].any()

    def test_third_species_in_galleries(self):
        # Gallery 0: 3 Li; gallery 1: 3 Na (the second batch retyped).
        retype = {39: NA, 40: NA, 41: NA}
        step, state, _ = _slide_setup((LI,), False, retype=retype)
        new, info = _propose(step, state, 300, seed=6)
        typ = np.asarray(state.types)
        moved = _moved(state, new)
        assert not moved[:, typ == NA].any()
        acc = np.asarray(info.accepted)
        assert np.all(np.asarray(info.reject_reason)[~acc] == 3)

        step, state, _ = _slide_setup((LI, NA), False, retype=retype)
        new, info = _propose(step, state, 300, seed=6)
        assert bool(jnp.all(info.accepted))
        moved = _moved(state, new)
        assert moved[:, typ == NA].any() and moved[:, typ == LI].any()

    def test_doped_sheet_moves_whole(self):
        # One B per sheet: sheets are C + B.
        retype = {0: B, 18: B}
        step, state, _ = _slide_setup(
            (C, B), True, k=3, host=(C, B), retype=retype
        )
        new, info = _propose(step, state, 100, seed=7)
        assert bool(jnp.all(info.accepted))
        for mk in _moved(state, new):
            assert mk.sum() == 18
            assert mk[0] or mk[18]  # the dopant moves with its sheet


class TestLayerSlideSpec:
    _SYMBOLS = {0: "C", 1: "Li", 2: "Na", 3: "B"}

    def test_names_and_modes(self):
        c = LayerSlideMoveSpec(species="C", subdivisions=3)
        li = LayerSlideMoveSpec(species="Li")
        assert c._effective_name() == "layer_slide_C"
        assert li._effective_name() == "layer_slide_Li"
        assert c._slide_host() and not li._slide_host()
        mixed = LayerSlideMoveSpec(species=["Li", "Na"])
        assert not mixed._slide_host()

    def test_mixed_sheet_and_gallery_species_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="mixes sheet"):
            LayerSlideMoveSpec(species=["C", "Li"])
        with pytest.raises(pydantic.ValidationError, match="mixes sheet"):
            LayerSlideMoveSpec(species="C", host_species=["C", "B"])

    def test_doped_host(self):
        spec = LayerSlideMoveSpec(species=["B", "C"], host_species=["C", "B"])
        assert spec._slide_host()
        desc = spec.to_descriptor(
            symbol_map=self._SYMBOLS, sheets=SheetInfo(2, 18)
        )
        assert desc.kernel_kwargs["host_codes"] == (0, 3)

    def test_subdivisions_must_not_be_negative(self):
        with pytest.raises(pydantic.ValidationError):
            LayerSlideMoveSpec(species="C", subdivisions=-1)

    def test_continuous_mode(self):
        spec = LayerSlideMoveSpec(species="C", subdivisions=0)
        assert spec._effective_name() == "layer_slide_C_continuous"
        assert (
            LayerSlideMoveSpec(species="Li", subdivisions=0)._effective_name()
            == "layer_slide_Li_continuous"
        )
        desc = spec.to_descriptor(
            symbol_map=self._SYMBOLS, sheets=SheetInfo(2, 18)
        )
        assert desc.kernel_kwargs["subdivisions"] == 0
        assert desc.kernel_kwargs["slide_host"] is True

    def test_descriptor(self):
        desc = LayerSlideMoveSpec(species="C", subdivisions=3).to_descriptor(
            symbol_map=self._SYMBOLS, sheets=SheetInfo(2, 18)
        )
        assert desc.kernel_kwargs == {
            "species": (0,),
            "subdivisions": 3,
            "slide_host": True,
            "host_codes": (0,),
            "n_sheets": 2,
            "in_plane_repeats": (3, 3),
        }
        assert set(desc.extra_state_fields) == {"sheet_id"}
        assert desc.reject_reasons == {"energy", "prior"}

    def test_gallery_hop_rejects_sheet_species(self):
        with pytest.raises(pydantic.ValidationError, match="form the sheets"):
            GalleryHopMoveSpec(species="C")

    def test_resolver_accepts_same_host_in_any_order(self):
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        typ[0] = B
        typ[18] = B
        init = SimpleNamespace(
            initial_positions=pos[None],
            initial_types=typ[None],
            initial_cells=cell[None],
            symbol_map=self._SYMBOLS,
        )
        root = SimpleNamespace(
            moves=[
                LayerSlideMoveSpec(
                    species=["C", "B"], host_species=["C", "B"]
                ),
                GalleryHopMoveSpec(species="Li", host_species=["B", "C"]),
            ]
        )
        assert _resolve_sheets(root, init) == SheetInfo(2, 18)


# ---------------------------------------------------------------------------
# stacking_shift
# ---------------------------------------------------------------------------


def _shift_setup(k=3, li=(3,), cpar=None, sigma=0.0, backend=None, **guard):
    """Kernel and state for stacking_shift on an AA graphite bilayer."""
    pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=li, z0=0.4)
    if cpar is not None:
        cell = cell.copy()
        cell[2, :2] += cpar
    kwargs = dict(
        host_codes=(C,),
        n_sheets=2,
        in_plane_repeats=(3, 3),
        subdivisions=k,
        n_atoms=len(typ),
        max_vol_per_atom=1e4,
        min_vol_per_atom=0.1,
        min_aspect=0.3,
        restrict_c_in_plane=True,
    )
    kwargs.update(guard)
    desc = MoveKernel(
        name="stacking_shift",
        build_kernel=build_stacking_shift_kernel,
        kernel_kwargs=kwargs,
        extra_state_fields={
            "sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))
        },
    )
    backend = backend or create_harmonic()
    init_fn, _, (step,) = build_mwg(backend, [desc])
    state = init_fn(jnp.asarray(pos), jnp.asarray(typ), 0.0, jnp.asarray(cell))
    state = state.set(step_sizes=jnp.full(1, sigma))
    return step, state


def _registries(pos, cell, n_sheets=2, size=18):
    """In-plane offset across each gallery, in thirds of the host lattice.

    Gallery g: atom 0 of sheet g+1, lifted by a multiple of c to just above
    sheet g, minus atom 0 of sheet g, reduced modulo (a / 3, b / 3).
    """
    pos, cell = np.asarray(pos, float), np.asarray(cell, float)
    n = np.cross(cell[0], cell[1])
    n /= np.linalg.norm(n)
    h = cell[2] @ n
    basis = np.array([cell[0] / 3, cell[1] / 3])
    out = []
    for g in range(n_sheets):
        d = pos[((g + 1) % n_sheets) * size] - pos[g * size]
        d = d - np.floor((d @ n) / h) * cell[2]
        d = d - (d @ n) * n
        f = np.linalg.lstsq(basis.T, d, rcond=None)[0]
        out.append(tuple(np.rint(3 * f).astype(int) % 3))
    return out


def _decode_shift(old_cell, new_cell, n=9):
    t = np.asarray(new_cell[2] - old_cell[2], float)
    basis = np.asarray(old_cell[:2], float) / n
    f = np.linalg.lstsq(basis.T, t, rcond=None)[0]
    return tuple(np.rint(f).astype(int))


class TestStackingShiftKernel:
    def test_only_the_cut_gallery_changes(self):
        step, state = _shift_setup(k=3)
        before = _registries(state.positions, state.cell)
        assert before == [(0, 0), (0, 0)]
        new, info = _propose(step, state, 200)
        assert bool(jnp.all(info.accepted))
        changed = set()
        for pos, cell in zip(np.asarray(new.positions), np.asarray(new.cell)):
            i, j = _decode_shift(state.cell, cell)
            after = _registries(pos, cell)
            diff = [g for g in range(2) if after[g] != before[g]]
            assert len(diff) == 1
            g = diff[0]
            assert after[g] == ((i % 3), (j % 3))
            changed.add(g)
        assert changed == {0, 1}  # both galleries get cut

    def test_volume_and_heights_unchanged(self):
        step, state = _shift_setup(k=3, sigma=0.2)
        new, _ = _propose(step, state, 50, seed=1)
        s3 = np.asarray(fractional_heights(state.positions, state.cell))
        vol = abs(np.linalg.det(np.asarray(state.cell)))
        for pos, cell in zip(np.asarray(new.positions), np.asarray(new.cell)):
            assert abs(np.linalg.det(cell)) == pytest.approx(vol, rel=1e-5)
            s3n = np.asarray(fractional_heights(jnp.asarray(pos), cell))
            d = (s3n - s3 + 0.5) % 1.0 - 0.5
            np.testing.assert_allclose(d, 0.0, atol=1e-5)

    def test_shifts_are_uniform_on_centred_grid(self):
        step, state = _shift_setup(k=3)
        new, _ = _propose(step, state, 4000, seed=2)
        shifts = [_decode_shift(state.cell, c) for c in np.asarray(new.cell)]
        grid = [
            (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)
        ]
        assert set(shifts) == set(grid)
        counts = [shifts.count(s) for s in grid]
        assert stats.chisquare(counts).pvalue > 1e-3

    def test_reverse_proposal_restores_the_cell(self):
        step, state = _shift_setup(k=3)
        fwd, _ = step(state, jax.random.key(3), 1e12)
        back, _ = _propose(step, fwd, 400, seed=4)
        cells = np.asarray(back.cell)
        hit = np.flatnonzero(
            np.all(
                np.isclose(cells, np.asarray(state.cell), atol=1e-5), (1, 2)
            )
        )
        assert hit.size > 0
        pos = np.asarray(back.positions)[hit[0]]
        # Same configuration as the start, up to lattice vectors of the cell.
        f = (pos - np.asarray(state.positions)) @ np.linalg.inv(
            np.asarray(state.cell)
        )
        np.testing.assert_allclose(f, np.rint(f), atol=1e-4)

    def test_unwrapped_atoms_give_the_same_crystal(self):
        """Atoms stored at r - c or r + 2c must be wrapped before tilting."""
        backend = create_lj(
            sigma=(1.42 / 2 ** (1 / 6), 3.0), epsilon=(1, 1), cutoff=3.0
        )
        step, state = _shift_setup(k=3, backend=backend)
        energy = backend(state.positions, state.types, state.cell, 0).energy
        state = state.set(energy=energy)
        shifted = state.positions.at[:5].add(-state.cell[2])
        shifted = shifted.at[20:25].add(2 * state.cell[2])
        state_u = state.set(positions=shifted)
        key = jax.random.key(5)
        new, _ = step(state, key, 1e12)
        new_u, _ = step(state_u, key, 1e12)
        np.testing.assert_allclose(new_u.cell, new.cell)
        assert float(new_u.energy) == pytest.approx(
            float(new.energy), rel=1e-5
        )
        # Without wrapping, tilting the unwrapped atoms is another crystal.
        naive = float(backend(shifted, state.types, new.cell, 0).energy)
        assert not np.isclose(naive, float(new.energy), rtol=1e-3)

    def test_leaving_the_domain_is_rejected(self):
        # Start next to a vertex of the hexagon, at |a| / sqrt(3).
        v = 0.97 * (3 * A1 + 3 * A2) / 3
        step, state = _shift_setup(k=1, cpar=v[:2])
        new, info = _propose(step, state, 200, seed=6)
        acc = np.asarray(info.accepted)
        assert acc.any() and (~acc).any()
        assert np.all(np.asarray(info.reject_reason)[~acc] == 2)
        for cell in np.asarray(new.cell)[acc]:
            assert bool(c_in_plane_reduced(jnp.asarray(cell)))

    def test_minimum_image_is_exact_inside_the_domain(self):
        """Tilts inside the hexagon need no extra periodic images.

        With the cutoff at most half the smallest cell width at the worst
        tilt, the minimum-image LJ energy equals a reference with many
        explicit images; the equivalent cells c + p a + q b outside the
        hexagon are so tilted that the minimum image fails.
        """
        sig = (1.42 / 2 ** (1 / 6), 3.0)
        mic = create_lj(sigma=sig, epsilon=(1, 1), cutoff=2.6)
        ref = create_lj(
            sigma=sig, epsilon=(1, 1), cutoff=2.6, supercell_trafo=(3, 3, 3)
        )
        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        pos, typ = jnp.asarray(pos), jnp.asarray(typ)
        rng = np.random.default_rng(7)
        vertex = 0.999 * (A1 + A2)  # |a| / sqrt(3) for the 3 x 3 cell
        tilts = [np.zeros(3), (A1 + A2) / 3, vertex]
        tilts += [
            p
            for p in rng.uniform(-4.3, 4.3, (40, 3)) * [1, 1, 0]
            if bool(
                c_in_plane_reduced(
                    jnp.asarray(
                        np.array([cell[0], cell[1], [p[0], p[1], 6.7]])
                    )
                )
            )
        ][:10]
        for t in tilts:
            tilted = cell.copy()
            tilted[2, :2] = t[:2]
            e_mic = float(mic(pos, typ, jnp.asarray(tilted), 0).energy)
            e_ref = float(ref(pos, typ, jnp.asarray(tilted), 0).energy)
            assert e_mic == pytest.approx(e_ref, rel=1e-5)
        far = cell.copy()
        far[2, :2] = (vertex + cell[0])[:2]
        e_far = float(mic(pos, typ, jnp.asarray(far), 0).energy)
        e_far_ref = float(ref(pos, typ, jnp.asarray(far), 0).energy)
        assert e_far != pytest.approx(e_far_ref, rel=1e-3)


class _FlatBackend:
    """Zero energy everywhere: every proposal passes the energy test."""

    def __call__(self, positions, types, cell, max_neighbors=0, **_):
        from jaxrens.backends.base import BackendResult

        return BackendResult(energy=jnp.zeros((), positions.dtype))


class TestStackingShiftPrior:
    def test_tilt_is_uniform_on_the_hexagon(self):
        """With zero energy the chain samples the prior: for each cell shape,
        the tilt of c is uniform on the hexagon."""
        from jaxrens.sampling.moves import shear

        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        guard = dict(
            n_atoms=len(typ),
            max_vol_per_atom=1e4,
            min_vol_per_atom=0.1,
            min_aspect=0.3,
            restrict_c_in_plane=True,
        )
        descs = [
            MoveKernel(
                name="stacking_shift_3",
                build_kernel=build_stacking_shift_kernel,
                kernel_kwargs=dict(
                    host_codes=(C,),
                    n_sheets=2,
                    in_plane_repeats=(3, 3),
                    subdivisions=3,
                    **guard,
                ),
                extra_state_fields={
                    "sheet_id": (
                        jnp.ndarray,
                        make_sheet_id_initializer((C,)),
                    )
                },
                step_size=0.3,
            ),
            MoveKernel(
                name="shear",
                build_kernel=shear.build_kernel,
                kernel_kwargs=guard,
                step_size=0.05,
            ),
        ]
        init_fn, step_fn, _ = build_mwg(_FlatBackend(), descs)
        state = init_fn(
            jnp.asarray(pos), jnp.asarray(typ), 0.0, jnp.asarray(cell)
        )
        n_chains, n_steps = 256, 1500
        pop = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_chains, *jnp.shape(x))), state
        )

        def chain(st, key):
            def body(st, k):
                k1, k2 = jax.random.split(k)
                st, _ = step_fn(k1, st, 1.0, k2)
                return st, None

            st, _ = jax.lax.scan(body, st, jax.random.split(key, n_steps))
            return st

        keys = jax.random.split(jax.random.key(8), n_chains)
        final = jax.jit(jax.vmap(chain))(pop, keys)

        # Where is each chain's tilt, relative to its own hexagon?
        inner_half, inner_quarter, sextant = [], [], []
        for cl in np.asarray(final.cell):
            a, b, c = cl
            n = np.cross(a, b)
            n /= np.linalg.norm(n)
            c_par = c - (c @ n) * n

            def inside(scale):
                lat = [a, b, a + b, a - b]
                return all(
                    abs(c_par @ L) <= scale * 0.5 * (L @ L) for L in lat
                )

            assert inside(1.0 + 1e-5)
            inner_half.append(inside(1 / np.sqrt(2)))
            inner_quarter.append(inside(0.5))
            e1 = a / np.linalg.norm(a)
            ang = np.arctan2(c_par @ np.cross(n, e1), c_par @ e1)
            sextant.append(
                int(((ang + np.pi / 6) % (2 * np.pi)) // (np.pi / 3))
            )
        # Scaled hexagons hold 1/2 and 1/4 of the area; sextants 1/6 each.
        assert stats.binomtest(sum(inner_half), n_chains, 0.5).pvalue > 1e-3
        assert (
            stats.binomtest(sum(inner_quarter), n_chains, 0.25).pvalue > 1e-3
        )
        counts = np.bincount(sextant, minlength=6)
        assert stats.chisquare(counts).pvalue > 1e-3


class TestStackingShiftContinuous:
    def test_continuous_tilt_is_a_small_in_plane_gaussian(self):
        sigma = 0.05
        step, state = _shift_setup(k=0, sigma=sigma)
        new, info = _propose(step, state, 2000, seed=11)
        assert bool(jnp.all(info.accepted))
        t = np.asarray(new.cell[:, 2] - state.cell[2])
        np.testing.assert_allclose(t[:, 2], 0.0, atol=1e-6)  # in plane
        np.testing.assert_allclose(
            t[:, :2].mean(axis=0), 0.0, atol=4 * sigma / np.sqrt(2000)
        )
        np.testing.assert_allclose(t[:, :2].std(axis=0), sigma, rtol=0.1)
        vol = abs(np.linalg.det(np.asarray(state.cell)))
        np.testing.assert_allclose(
            np.abs(np.linalg.det(np.asarray(new.cell))), vol, rtol=1e-5
        )
        # The tilt is in plane: every height is unchanged.
        s3 = np.asarray(fractional_heights(state.positions, state.cell))
        for pos, cell in zip(
            np.asarray(new.positions)[:20], np.asarray(new.cell)[:20]
        ):
            s3n = np.asarray(fractional_heights(jnp.asarray(pos), cell))
            d = (s3n - s3 + 0.5) % 1.0 - 0.5
            np.testing.assert_allclose(d, 0.0, atol=1e-5)

    def test_spec(self):
        from jaxrens.cli.schema.cell import CellSpec

        spec = StackingShiftMoveSpec(subdivisions=0)
        assert spec._effective_name() == "stacking_shift_continuous"
        desc = spec.to_descriptor(
            n_atoms=39,
            cell_cfg=CellSpec(restrict_c_in_plane=True, min_aspect_ratio=0.5),
            symbol_map={0: "C", 1: "Li"},
            sheets=SheetInfo(2, 18),
        )
        assert desc.kernel_kwargs["subdivisions"] == 0
        with pytest.raises(pydantic.ValidationError):
            StackingShiftMoveSpec(subdivisions=-1)


class TestStackingShiftSpec:
    _SYMBOLS = {0: "C", 1: "Li"}

    def _cell(self, restrict=True):
        from jaxrens.cli.schema.cell import CellSpec

        return CellSpec(restrict_c_in_plane=restrict, min_aspect_ratio=0.5)

    def test_names(self):
        assert StackingShiftMoveSpec()._effective_name() == "stacking_shift_1"
        assert (
            StackingShiftMoveSpec(subdivisions=3)._effective_name()
            == "stacking_shift_3"
        )

    def test_requires_restrict_c_in_plane(self):
        with pytest.raises(ValueError, match="restrict_c_in_plane"):
            StackingShiftMoveSpec().to_descriptor(
                n_atoms=39,
                cell_cfg=self._cell(restrict=False),
                symbol_map=self._SYMBOLS,
                sheets=SheetInfo(2, 18),
            )

    def test_descriptor(self):
        desc = StackingShiftMoveSpec(subdivisions=3).to_descriptor(
            n_atoms=39,
            cell_cfg=self._cell(),
            symbol_map=self._SYMBOLS,
            sheets=SheetInfo(2, 18),
        )
        assert desc.name == "stacking_shift_3"
        assert desc.kernel_kwargs["subdivisions"] == 3
        assert desc.kernel_kwargs["restrict_c_in_plane"] is True
        assert desc.kernel_kwargs["n_sheets"] == 2
        assert set(desc.extra_state_fields) == {"sheet_id"}
        assert desc.mutates == {"positions", "cell"}
        assert desc.reject_reasons == {"energy", "cell"}

    @pytest.mark.parametrize(
        "spec_cls", [VolumeMoveSpec, ShearMoveSpec, StretchMoveSpec]
    )
    def test_cell_moves_carry_the_restriction_only_when_set(self, spec_cls):
        off = spec_cls().to_descriptor(n_atoms=39, cell_cfg=self._cell(False))
        on = spec_cls().to_descriptor(n_atoms=39, cell_cfg=self._cell(True))
        assert "restrict_c_in_plane" not in off.kernel_kwargs
        assert on.kernel_kwargs["restrict_c_in_plane"] is True

    def _root(self, tmp_path, cpar=(0.0, 0.0), restrict=True):
        import ase.io
        from ase import Atoms

        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        cell = cell.copy()
        cell[2, :2] = cpar
        atoms = Atoms(
            symbols=["C" if t == C else "Li" for t in typ],
            positions=pos,
            cell=cell,
            pbc=True,
        )
        path = tmp_path / "walkers.extxyz"
        ase.io.write(str(path), [atoms.copy() for _ in range(4)])
        return RootSpec.model_validate(
            {
                "run": {"n_live": 4, "max_iterations": 2, "seed": 0},
                "moves": [
                    {"type": "gallery_hop", "species": "Li"},
                    {"type": "layer_slide", "species": "C", "subdivisions": 3},
                    {"type": "stacking_shift", "subdivisions": 3},
                    {"type": "shear"},
                ],
                "backend": {"backend_type": "harmonic"},
                "cell": {
                    "restrict_c_in_plane": restrict,
                    "min_aspect_ratio": 0.5,
                    "min_volume_per_atom": 1.0,
                    "max_volume_per_atom": 50.0,
                },
                "output": {"format": "none", "working_dir": str(tmp_path)},
                "init": {
                    "start_walker_set": str(path),
                    "random_initialise_pos": False,
                    "random_initialise_cell": False,
                },
            }
        )

    def test_full_resolve(self, tmp_path):
        resolved = resolve(self._root(tmp_path))
        names = [d.name for d in resolved.move_descriptors]
        assert names == [
            "gallery_hop_Li",
            "layer_slide_C",
            "stacking_shift_3",
            "shear",
        ]
        assert resolved.move_descriptors[3].kernel_kwargs[
            "restrict_c_in_plane"
        ]

    def test_resolve_errors_without_restriction(self, tmp_path):
        with pytest.raises(ValueError, match="restrict_c_in_plane"):
            resolve(self._root(tmp_path, restrict=False))

    def test_starting_cell_outside_domain_is_an_error(self, tmp_path):
        a = 3 * A1
        with pytest.raises(RuntimeError, match="Wigner-Seitz"):
            resolve(self._root(tmp_path, cpar=(a + 0.5 * A2)[:2]))


class TestSheetIdInReplicaExchange:
    def test_sheet_id_follows_its_configuration(self):
        """sheet_id describes the configuration, so a replica-exchange swap
        must move it with the positions to the other run."""
        from jaxrens.sampling.batch_descriptor import VmapRuns
        from jaxrens.sampling.inter_re_manager import InterREManager
        from jaxrens.sampling.moves.replica_exchange import EnsembleSwap
        from jaxrens.sampling.nested_sampling import init_ns_parallel

        pos, typ, cell = _graphite([3.35, 3.35], li_per_gallery=(3,))
        desc = MoveKernel(
            name="layer_slide_C",
            build_kernel=build_layer_slide_kernel,
            kernel_kwargs=dict(
                species=(C,),
                host_codes=(C,),
                n_sheets=2,
                in_plane_repeats=(3, 3),
                subdivisions=3,
                slide_host=True,
            ),
            extra_state_fields={
                "sheet_id": (jnp.ndarray, make_sheet_id_initializer((C,)))
            },
        )
        init_fn, _, _ = build_mwg(create_harmonic(), [desc])
        # Run 1 holds the same crystal shifted up by half a cell, so its
        # sheets are numbered the other way round.
        shifted = pos + [0.0, 0.0, 0.5 * cell[2, 2]]
        positions = jnp.asarray(np.stack([pos[None], shifted[None]]))
        cells = jnp.asarray(np.broadcast_to(cell, (2, 1, 3, 3)))
        ns_state = init_ns_parallel(
            init_fn,
            positions,
            jnp.asarray(typ),
            jnp.array([[0.0], [1.0]]),
            cells,
            jax.random.split(jax.random.key(0), 2),
            step_sizes=jnp.array([0.02]),
            ensemble_params_per_run=[
                {"pressure": jnp.array(0.0)},
                {"pressure": jnp.array(0.0)},
            ],
        )
        ns_state = ns_state.set(emax=jnp.array([1e6, 1e6]))
        old = ns_state.population
        assert not jnp.array_equal(old.sheet_id[0], old.sheet_id[1])

        mgr = InterREManager(
            EnsembleSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        new_state, stats, _ = mgr.apply(ns_state, jax.random.key(1))
        new = new_state.population
        assert stats["n_swap_pairs_accepted"] == 1
        for name in ("positions", "sheet_id"):
            got, want = getattr(new, name), getattr(old, name)
            assert jnp.array_equal(got[0], want[1]), name
            assert jnp.array_equal(got[1], want[0]), name
