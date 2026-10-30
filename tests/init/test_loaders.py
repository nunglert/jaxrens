"""Tests for load_structure and load_walker_set (jaxrens.init).

load_structure (structure.py) reads a single-frame file through ASE.
load_walker_set (walker_set.py) reads a multi-frame extxyz or an HDF5 file.
The last class checks how _resolve_init (jaxrens.cli.resolve) uses a walker
set from a file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ase
import ase.io
import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.resolve import ResolvedInit, _resolve_init
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.init import InitSpec
from jaxrens.init.structure import load_structure
from jaxrens.init.walker_set import WalkerSet, load_walker_set


def _make_frames(
    n_frames: int,
    symbols: list[str],
    cell_size: float = 6.0,
) -> list:
    """Return a list of ase.Atoms with distinct (jittered) positions."""
    frames = []
    n_atoms = len(symbols)
    cell = np.eye(3) * cell_size
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        pos = rng.uniform(0.5, cell_size - 0.5, size=(n_atoms, 3)).astype(
            np.float32
        )
        atoms = ase.Atoms(
            symbols=list(symbols), positions=pos, cell=cell, pbc=True
        )
        frames.append(atoms)
    return frames


def _write_extxyz(
    tmp_path: Path, frames, name: str = "walkers.extxyz"
) -> Path:
    p = tmp_path / name
    ase.io.write(str(p), frames, format="extxyz")
    return p


def _write_single_extxyz(
    path: Path, symbols: list[str], positions: np.ndarray, cell: np.ndarray
) -> None:
    """Write a single-frame extxyz file."""
    atoms = ase.Atoms(
        symbols=symbols, positions=positions, cell=cell, pbc=True
    )
    ase.io.write(str(path), atoms)


def _write_hdf5(
    tmp_path: Path,
    positions: np.ndarray,
    types: np.ndarray,
    cells: np.ndarray,
    symbol_map: dict[int, str] | None = None,
    name: str = "walkers.h5",
) -> Path:
    p = tmp_path / name
    with h5py.File(p, "w") as f:
        f.create_dataset("positions", data=positions)
        f.create_dataset("types", data=types)
        f.create_dataset("cells", data=cells)
        if symbol_map is not None:
            f.attrs["symbol_map"] = json.dumps(
                {str(k): v for k, v in symbol_map.items()}
            )
    return p


def _cell_cfg_permissive():
    return CellSpec(
        max_volume_per_atom=10000.0,
        min_volume_per_atom=0.01,
        min_aspect_ratio=0.001,
    )


class TestLoadStructureRoundTrip:
    """Check that load_structure returns what was written to a single frame.

    The tests cover positions, cell, dtypes, the symbol map for one element,
    a single atom, and a path given as a string.
    """

    def test_positions_values(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]], dtype=np.float32)
        cell = np.diag([5.0, 5.0, 5.0]).astype(np.float32)
        p = tmp_path / "si.extxyz"
        _write_single_extxyz(p, ["Si", "Si"], pos, cell)
        positions, _, _, _ = load_structure(p)
        np.testing.assert_allclose(np.array(positions), pos, atol=1e-5)

    def test_cell_values(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        cell = np.diag([4.0, 5.0, 6.0]).astype(np.float32)
        p = tmp_path / "single.extxyz"
        _write_single_extxyz(p, ["Si"], pos, cell)
        _, _, loaded_cell, _ = load_structure(p)
        np.testing.assert_allclose(np.array(loaded_cell), cell, atol=1e-5)

    def test_dtypes(self, tmp_path):
        pos = np.zeros((2, 3), dtype=np.float64)
        cell = np.eye(3) * 5.0
        p = tmp_path / "dtype.extxyz"
        _write_single_extxyz(p, ["Si", "O"], pos, cell)
        positions, types, loaded_cell, _ = load_structure(p)
        assert positions.dtype == jnp.float32
        assert types.dtype == jnp.int32
        assert loaded_cell.dtype == jnp.float32

    def test_symbol_map_single_element(self, tmp_path):
        pos = np.zeros((3, 3))
        cell = np.eye(3) * 5.0
        p = tmp_path / "three_si.extxyz"
        _write_single_extxyz(p, ["Si", "Si", "Si"], pos, cell)
        _, _, _, symbol_map = load_structure(p)
        assert symbol_map == {0: "Si"}

    def test_single_atom(self, tmp_path):
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        p = tmp_path / "one.extxyz"
        _write_single_extxyz(p, ["Si"], pos, cell)
        positions, types, _, symbol_map = load_structure(p)
        assert positions.shape == (1, 3)
        assert types.shape == (1,)
        assert symbol_map == {0: "Si"}

    def test_path_as_string(self, tmp_path):
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        p = tmp_path / "str.extxyz"
        _write_single_extxyz(p, ["Si"], pos, cell)
        positions, _, _, _ = load_structure(str(p))
        assert positions.shape == (1, 3)


class TestLoadStructureMultiElement:
    """Check the symbol map and type indices for several elements.

    Element types are numbered in order of first appearance in the file.
    """

    def test_si2o3_symbol_map(self, tmp_path):
        pos = np.zeros((5, 3))
        pos[1, 0] = 1.5
        pos[2, 1] = 1.5
        pos[3, 2] = 1.5
        pos[4, 0] = 3.0
        cell = np.eye(3) * 6.0
        p = tmp_path / "si2o3.extxyz"
        _write_single_extxyz(p, ["Si", "Si", "O", "O", "O"], pos, cell)
        _, types, _, symbol_map = load_structure(p)
        assert symbol_map == {0: "Si", 1: "O"}
        assert list(np.array(types)) == [0, 0, 1, 1, 1]

    def test_noncontiguous_ordering_first_appearance(self, tmp_path):
        pos = np.zeros((4, 3))
        for i in range(4):
            pos[i, 0] = i * 1.5
        cell = np.eye(3) * 8.0
        p = tmp_path / "noncontig.extxyz"
        _write_single_extxyz(p, ["Si", "O", "Si", "O"], pos, cell)
        _, types, _, symbol_map = load_structure(p)
        assert symbol_map[0] == "Si"
        assert symbol_map[1] == "O"
        assert list(np.array(types)) == [0, 1, 0, 1]

    def test_three_species(self, tmp_path):
        pos = np.zeros((3, 3))
        for i in range(3):
            pos[i, 0] = i * 2.0
        cell = np.eye(3) * 7.0
        p = tmp_path / "ternary.extxyz"
        _write_single_extxyz(p, ["Si", "O", "N"], pos, cell)
        _, types, _, symbol_map = load_structure(p)
        assert symbol_map == {0: "Si", 1: "O", 2: "N"}
        assert list(np.array(types)) == [0, 1, 2]


class TestLoadStructureErrors:
    """Check the errors raised by load_structure.

    The tests cover a missing file, a multi-frame file, a file without a
    cell, and the contents of the error messages.
    """

    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_structure(tmp_path / "does_not_exist.extxyz")

    def test_multi_frame_raises_value_error(self, tmp_path):
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        atoms = ase.Atoms(["Si"], positions=pos, cell=cell, pbc=True)
        p = tmp_path / "multi.extxyz"
        ase.io.write(str(p), [atoms, atoms])
        with pytest.raises(ValueError, match="single-frame"):
            load_structure(p)

    def test_zero_cell_raises_value_error(self, tmp_path):
        pos = np.zeros((1, 3))
        atoms = ase.Atoms(["Si"], positions=pos)
        p = tmp_path / "nocell.extxyz"
        ase.io.write(str(p), atoms)
        with pytest.raises(ValueError, match="no simulation cell"):
            load_structure(p)

    def test_error_message_includes_path(self, tmp_path):
        missing = tmp_path / "missing_file.xyz"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_structure(missing)
        assert "missing_file.xyz" in str(exc_info.value)

    def test_multi_frame_error_message_includes_count(self, tmp_path):
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        atoms = ase.Atoms(["Si"], positions=pos, cell=cell, pbc=True)
        p = tmp_path / "three_frames.extxyz"
        ase.io.write(str(p), [atoms, atoms, atoms])
        with pytest.raises(ValueError, match="3"):
            load_structure(p)


class TestExtxyzRoundTrip:
    """Check that load_walker_set reads a multi-frame extxyz file.

    The tests cover array shapes, dtypes, position values, and the returned
    WalkerSet type.
    """

    def test_positions_shape(self, tmp_path):
        frames = _make_frames(4, ["Si", "Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.shape == (4, 2, 3)

    def test_types_shape(self, tmp_path):
        frames = _make_frames(4, ["Si", "Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.types.shape == (4, 2)

    def test_cells_shape(self, tmp_path):
        frames = _make_frames(4, ["Si", "Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.cells.shape == (4, 3, 3)

    def test_dtypes(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.dtype == jnp.float32
        assert ws.types.dtype == jnp.int32
        assert ws.cells.dtype == jnp.float32

    def test_positions_values_match(self, tmp_path):
        frames = _make_frames(4, ["Si", "O"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        for i, atoms in enumerate(frames):
            np.testing.assert_allclose(
                np.array(ws.positions[i]),
                np.array(atoms.get_positions(), dtype=np.float32),
                atol=1e-5,
            )


class TestExtxyzSymbolMap:
    """Check the symbol map built from an extxyz walker set.

    Element types are numbered in order of first appearance in the first
    frame.
    """

    def test_si2o3_symbol_map_first_appearance(self, tmp_path):
        frames = _make_frames(3, ["Si", "Si", "O", "O", "O"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=3)
        assert ws.symbol_map == {0: "Si", 1: "O"}

    def test_si2o3_types_values(self, tmp_path):
        frames = _make_frames(3, ["Si", "Si", "O", "O", "O"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=3)
        for i in range(3):
            assert list(np.array(ws.types[i])) == [0, 0, 1, 1, 1]

    def test_single_element_symbol_map(self, tmp_path):
        frames = _make_frames(2, ["Si", "Si", "Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=2)
        assert ws.symbol_map == {0: "Si"}

    def test_three_species_first_appearance(self, tmp_path):
        frames = _make_frames(2, ["Si", "O", "N"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=2)
        assert ws.symbol_map == {0: "Si", 1: "O", 2: "N"}


class TestExtxyzErrors:
    """Check the errors raised for an inconsistent extxyz walker set.

    The tests cover a wrong frame count, frames with different atom counts or
    compositions, and a frame without a cell.
    """

    def test_frame_count_mismatch_raises(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        with pytest.raises(ValueError, match="n_live=5"):
            load_walker_set(p, n_live_expected=5)

    def test_differing_atom_counts_raises(self, tmp_path):
        cell = np.eye(3) * 6.0
        a1 = ase.Atoms(
            ["Si", "Si"], positions=[[0, 0, 0], [3, 3, 3]], cell=cell, pbc=True
        )
        a2 = ase.Atoms(
            ["Si", "Si", "Si"],
            positions=[[0, 0, 0], [2, 2, 2], [4, 4, 4]],
            cell=cell,
            pbc=True,
        )
        p = tmp_path / "bad.extxyz"
        ase.io.write(str(p), [a1, a2])
        with pytest.raises(ValueError, match="atom counts"):
            load_walker_set(p, n_live_expected=2)

    def test_divergent_compositions_raises(self, tmp_path):
        cell = np.eye(3) * 6.0
        a1 = ase.Atoms(
            ["Si", "Si"], positions=[[0, 0, 0], [3, 3, 3]], cell=cell, pbc=True
        )
        a2 = ase.Atoms(
            ["Si", "O"], positions=[[0, 0, 0], [3, 3, 3]], cell=cell, pbc=True
        )
        p = tmp_path / "comp.extxyz"
        ase.io.write(str(p), [a1, a2])
        with pytest.raises(ValueError, match="divergent compositions"):
            load_walker_set(p, n_live_expected=2)

    def test_zero_cell_raises(self, tmp_path):
        cell = np.eye(3) * 6.0
        a1 = ase.Atoms(["Si"], positions=[[0, 0, 0]], cell=cell, pbc=True)
        a2 = ase.Atoms(["Si"], positions=[[0, 0, 0]])
        p = tmp_path / "nocell.extxyz"
        ase.io.write(str(p), [a1, a2])
        with pytest.raises(ValueError, match="cell"):
            load_walker_set(p, n_live_expected=2)


class TestHDF5RoundTrip:
    """Check that load_walker_set reads an HDF5 walker set.

    The tests cover array shapes, values, dtypes, and the symbol map stored
    as a file attribute.
    """

    def _make_arrays(self, n_live: int = 4, n_atoms: int = 2):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 5, (n_live, n_atoms, 3)).astype(np.float32)
        types = np.zeros((n_live, n_atoms), dtype=np.int32)
        cells = np.stack([np.eye(3) * 5.0 for _ in range(n_live)]).astype(
            np.float32
        )
        symbol_map = {0: "Si"}
        return positions, types, cells, symbol_map

    def test_shapes(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.shape == (4, 2, 3)
        assert ws.types.shape == (4, 2)
        assert ws.cells.shape == (4, 3, 3)

    def test_values_match(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        np.testing.assert_allclose(np.array(ws.positions), pos, atol=1e-5)
        np.testing.assert_array_equal(np.array(ws.types), types)
        np.testing.assert_allclose(np.array(ws.cells), cells, atol=1e-5)

    def test_symbol_map_restored(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.symbol_map == {0: "Si"}

    def test_dtypes(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.dtype == jnp.float32
        assert ws.types.dtype == jnp.int32


class TestHDF5Errors:
    """Check the errors and warnings for an invalid HDF5 walker set.

    The tests cover each missing dataset, a wrong number of walkers, and the
    warning logged when the symbol_map attribute is absent.
    """

    def test_missing_positions_raises(self, tmp_path):
        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("types", data=np.zeros((4, 2), dtype=np.int32))
            f.create_dataset("cells", data=np.stack([np.eye(3)] * 4))
        with pytest.raises(ValueError, match="positions"):
            load_walker_set(p, n_live_expected=4)

    def test_missing_types_raises(self, tmp_path):
        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset(
                "positions", data=np.zeros((4, 2, 3), dtype=np.float32)
            )
            f.create_dataset("cells", data=np.stack([np.eye(3)] * 4))
        with pytest.raises(ValueError, match="types"):
            load_walker_set(p, n_live_expected=4)

    def test_missing_cells_raises(self, tmp_path):
        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset(
                "positions", data=np.zeros((4, 2, 3), dtype=np.float32)
            )
            f.create_dataset("types", data=np.zeros((4, 2), dtype=np.int32))
        with pytest.raises(ValueError, match="cells"):
            load_walker_set(p, n_live_expected=4)

    def test_n_live_mismatch_raises(self, tmp_path):
        pos = np.zeros((4, 2, 3), dtype=np.float32)
        types = np.zeros((4, 2), dtype=np.int32)
        cells = np.stack([np.eye(3)] * 4).astype(np.float32)
        p = _write_hdf5(tmp_path, pos, types, cells, {0: "Si"})
        with pytest.raises(ValueError, match="n_live=6"):
            load_walker_set(p, n_live_expected=6)

    def test_no_symbol_map_synthesizes_integer_codes(self, tmp_path, caplog):
        pos = np.zeros((3, 2, 3), dtype=np.float32)
        types = np.array([[[0, 1]] * 3]).reshape(3, 2).astype(np.int32)
        cells = np.stack([np.eye(3) * 5.0] * 3).astype(np.float32)
        p = _write_hdf5(tmp_path, pos, types, cells, symbol_map=None)

        with caplog.at_level(
            logging.WARNING, logger="jaxrens.init.walker_set"
        ):
            ws = load_walker_set(p, n_live_expected=3)

        assert isinstance(ws.symbol_map, dict)
        assert any("symbol_map" in r.message for r in caplog.records)
        assert 0 in ws.symbol_map
        assert 1 in ws.symbol_map


class TestWalkerSetDispatch:
    """Check how load_walker_set picks a reader from the file extension.

    The tests cover unsupported extensions, a missing file, and the .hdf5 and
    .xyz extensions.
    """

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "walkers.txt"
        p.write_text("dummy")
        with pytest.raises(ValueError, match=r"\.txt"):
            load_walker_set(p, n_live_expected=1)

    def test_unsupported_extension_message_lists_supported(self, tmp_path):
        p = tmp_path / "walkers.npz"
        p.write_bytes(b"dummy")
        with pytest.raises(ValueError) as exc_info:
            load_walker_set(p, n_live_expected=1)
        for ext in (".extxyz", ".xyz", ".h5", ".hdf5"):
            assert ext in str(exc_info.value)

    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        p = tmp_path / "missing.extxyz"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_walker_set(p, n_live_expected=4)

    def test_hdf5_extension_dispatches_correctly(self, tmp_path):
        pos = np.zeros((2, 1, 3), dtype=np.float32)
        types = np.zeros((2, 1), dtype=np.int32)
        cells = np.stack([np.eye(3) * 4.0] * 2).astype(np.float32)
        p = _write_hdf5(
            tmp_path, pos, types, cells, {0: "Si"}, name="walkers.hdf5"
        )
        ws = load_walker_set(p, n_live_expected=2)
        assert ws.positions.shape == (2, 1, 3)

    def test_xyz_extension_dispatches_to_ase(self, tmp_path):
        cell = np.eye(3) * 5.0
        frames = [
            ase.Atoms(["Si"], positions=[[i, i, i]], cell=cell, pbc=True)
            for i in range(2)
        ]
        p = tmp_path / "walkers.xyz"
        ase.io.write(str(p), frames, format="extxyz")
        ws = load_walker_set(p, n_live_expected=2)
        assert ws.positions.shape == (2, 1, 3)


class TestInitSpecResolverModeC:
    """Check _resolve_init for an init spec with start_walker_set.

    The tests cover extxyz and HDF5 files, the warning logged when
    random_initialise_pos or random_initialise_cell is also set, and the
    error for a walker cell outside the CellSpec bounds.
    """

    def test_extxyz_resolved_init_type(self, tmp_path):
        p = _write_extxyz(tmp_path, _make_frames(4, ["Si"]))
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert isinstance(result, ResolvedInit)

    def test_extxyz_positions_shape(self, tmp_path):
        p = _write_extxyz(tmp_path, _make_frames(4, ["Si", "Si"]))
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.initial_positions.shape == (4, 2, 3)

    def test_extxyz_types_shape(self, tmp_path):
        p = _write_extxyz(tmp_path, _make_frames(4, ["Si", "Si"]))
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.initial_types.shape == (4, 2)

    def test_extxyz_symbol_map_correct(self, tmp_path):
        p = _write_extxyz(tmp_path, _make_frames(3, ["Si", "O", "O"]))
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg,
            n_live=3,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.symbol_map == {0: "Si", 1: "O"}

    def test_hdf5_positions_shape(self, tmp_path):
        rng = np.random.default_rng(7)
        positions = rng.uniform(0.5, 5.5, (5, 2, 3)).astype(np.float32)
        types = np.zeros((5, 2), dtype=np.int32)
        cells = np.stack([np.eye(3) * 6.0] * 5).astype(np.float32)
        p = _write_hdf5(tmp_path, positions, types, cells, {0: "Si"})
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg,
            n_live=5,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.initial_positions.shape == (5, 2, 3)

    def test_random_initialise_pos_true_keeps_file_positions(
        self, tmp_path, caplog
    ):
        """With random_initialise_pos=True, positions come from the file."""

        p = _write_extxyz(tmp_path, _make_frames(3, ["Si"]))
        cfg = InitSpec(start_walker_set=p, random_initialise_pos=True)
        ws = load_walker_set(p, n_live_expected=3)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(
                cfg,
                n_live=3,
                seed=0,
                energy_backend=create_harmonic(),
                cell_cfg=_cell_cfg_permissive(),
            )
        assert any(
            "random_initialise_pos" in r.message
            or "randomiz" in r.message.lower()
            for r in caplog.records
        )
        np.testing.assert_allclose(
            np.array(result.initial_positions),
            np.array(ws.positions),
            atol=1e-5,
        )

    def test_random_initialise_cell_true_keeps_file_cells(
        self, tmp_path, caplog
    ):
        """With random_initialise_cell=True, cells still come from the file."""

        p = _write_extxyz(tmp_path, _make_frames(3, ["Si"]))
        cfg = InitSpec(start_walker_set=p, random_initialise_cell=True)
        ws = load_walker_set(p, n_live_expected=3)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(
                cfg,
                n_live=3,
                seed=0,
                energy_backend=create_harmonic(),
                cell_cfg=_cell_cfg_permissive(),
            )
        assert any(
            "random_initialise_cell" in r.message
            or "randomiz" in r.message.lower()
            for r in caplog.records
        )
        np.testing.assert_allclose(
            np.array(result.initial_cells),
            np.array(ws.cells),
            atol=1e-5,
        )

    def test_cell_config_violation_raises(self, tmp_path):
        """A walker cell outside the CellSpec bounds raises RuntimeError."""

        cell = np.eye(3, dtype=np.float32) * 6.0
        atoms = ase.Atoms(
            ["Si"], positions=[[3.0, 3.0, 3.0]], cell=cell, pbc=True
        )
        p = tmp_path / "toosmall.extxyz"
        ase.io.write(str(p), [atoms], format="extxyz")

        strict_cfg = CellSpec(
            max_volume_per_atom=1.0,
            min_volume_per_atom=0.0001,
            min_aspect_ratio=0.001,
        )
        cfg = InitSpec(start_walker_set=p)
        with pytest.raises(RuntimeError):
            _resolve_init(cfg, n_live=1, seed=0, cell_cfg=strict_cfg)
