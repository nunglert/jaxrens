"""Unit tests for jaxrens.init.walker_set.load_walker_set.

Covers extxyz and HDF5 formats, dispatch, error conditions, and round-trip
correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.init.walker_set import WalkerSet, load_walker_set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frames(
    n_frames: int,
    symbols: list[str],
    cell_size: float = 6.0,
) -> list:
    """Return a list of ase.Atoms with distinct (jittered) positions."""
    import ase

    frames = []
    n_atoms = len(symbols)
    cell = np.eye(3) * cell_size
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        pos = rng.uniform(0.5, cell_size - 0.5, size=(n_atoms, 3)).astype(np.float32)
        atoms = ase.Atoms(symbols=list(symbols), positions=pos, cell=cell, pbc=True)
        frames.append(atoms)
    return frames


def _write_extxyz(tmp_path: Path, frames: list, name: str = "walkers.extxyz") -> Path:
    import ase.io

    p = tmp_path / name
    ase.io.write(str(p), frames, format="extxyz")
    return p


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
            f.attrs["symbol_map"] = json.dumps({str(k): v for k, v in symbol_map.items()})
    return p


# ---------------------------------------------------------------------------
# extxyz: round-trip
# ---------------------------------------------------------------------------

class TestExtxyzRoundTrip:
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

    def test_positions_dtype(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.dtype == jnp.float32

    def test_cells_dtype(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.cells.dtype == jnp.float32

    def test_types_dtype(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.types.dtype == jnp.int32

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

    def test_n_live_dimension(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.shape[0] == 4

    def test_returns_walker_set_instance(self, tmp_path):
        frames = _make_frames(3, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        ws = load_walker_set(p, n_live_expected=3)
        assert isinstance(ws, WalkerSet)


# ---------------------------------------------------------------------------
# extxyz: multi-element symbol_map
# ---------------------------------------------------------------------------

class TestExtxyzSymbolMap:
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


# ---------------------------------------------------------------------------
# extxyz: error conditions
# ---------------------------------------------------------------------------

class TestExtxyzErrors:
    def test_frame_count_mismatch_raises(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        with pytest.raises(ValueError, match="4"):
            load_walker_set(p, n_live_expected=5)

    def test_frame_count_mismatch_message_has_expected_count(self, tmp_path):
        frames = _make_frames(4, ["Si"])
        p = _write_extxyz(tmp_path, frames)
        with pytest.raises(ValueError, match="n_live=5"):
            load_walker_set(p, n_live_expected=5)

    def test_differing_atom_counts_raises(self, tmp_path):
        import ase, ase.io

        cell = np.eye(3) * 6.0
        a1 = ase.Atoms(["Si", "Si"], positions=[[0, 0, 0], [3, 3, 3]], cell=cell, pbc=True)
        a2 = ase.Atoms(["Si", "Si", "Si"], positions=[[0, 0, 0], [2, 2, 2], [4, 4, 4]], cell=cell, pbc=True)
        p = tmp_path / "bad.extxyz"
        ase.io.write(str(p), [a1, a2])
        with pytest.raises(ValueError, match="atom counts"):
            load_walker_set(p, n_live_expected=2)

    def test_divergent_compositions_raises(self, tmp_path):
        import ase, ase.io

        cell = np.eye(3) * 6.0
        a1 = ase.Atoms(["Si", "Si"], positions=[[0, 0, 0], [3, 3, 3]], cell=cell, pbc=True)
        a2 = ase.Atoms(["Si", "O"], positions=[[0, 0, 0], [3, 3, 3]], cell=cell, pbc=True)
        p = tmp_path / "comp.extxyz"
        ase.io.write(str(p), [a1, a2])
        with pytest.raises(ValueError, match="divergent compositions"):
            load_walker_set(p, n_live_expected=2)

    def test_zero_cell_raises(self, tmp_path):
        import ase, ase.io

        cell = np.eye(3) * 6.0
        a1 = ase.Atoms(["Si"], positions=[[0, 0, 0]], cell=cell, pbc=True)
        a2 = ase.Atoms(["Si"], positions=[[0, 0, 0]])
        p = tmp_path / "nocell.extxyz"
        ase.io.write(str(p), [a1, a2])
        with pytest.raises(ValueError, match="cell"):
            load_walker_set(p, n_live_expected=2)


# ---------------------------------------------------------------------------
# HDF5: round-trip
# ---------------------------------------------------------------------------

class TestHDF5RoundTrip:
    def _make_arrays(self, n_live: int = 4, n_atoms: int = 2):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 5, (n_live, n_atoms, 3)).astype(np.float32)
        types = np.zeros((n_live, n_atoms), dtype=np.int32)
        cells = np.stack([np.eye(3) * 5.0 for _ in range(n_live)]).astype(np.float32)
        symbol_map = {0: "Si"}
        return positions, types, cells, symbol_map

    def test_positions_shape(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.shape == (4, 2, 3)

    def test_types_shape(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.types.shape == (4, 2)

    def test_cells_shape(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.cells.shape == (4, 3, 3)

    def test_positions_values(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        np.testing.assert_allclose(np.array(ws.positions), pos, atol=1e-5)

    def test_types_values(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        np.testing.assert_array_equal(np.array(ws.types), types)

    def test_cells_values(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        np.testing.assert_allclose(np.array(ws.cells), cells, atol=1e-5)

    def test_symbol_map_restored(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.symbol_map == {0: "Si"}

    def test_positions_dtype(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.positions.dtype == jnp.float32

    def test_types_dtype(self, tmp_path):
        pos, types, cells, sm = self._make_arrays()
        p = _write_hdf5(tmp_path, pos, types, cells, sm)
        ws = load_walker_set(p, n_live_expected=4)
        assert ws.types.dtype == jnp.int32


# ---------------------------------------------------------------------------
# HDF5: error conditions
# ---------------------------------------------------------------------------

class TestHDF5Errors:
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
            f.create_dataset("positions", data=np.zeros((4, 2, 3), dtype=np.float32))
            f.create_dataset("cells", data=np.stack([np.eye(3)] * 4))
        with pytest.raises(ValueError, match="types"):
            load_walker_set(p, n_live_expected=4)

    def test_missing_cells_raises(self, tmp_path):
        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("positions", data=np.zeros((4, 2, 3), dtype=np.float32))
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
        import logging

        pos = np.zeros((3, 2, 3), dtype=np.float32)
        types = np.array([[[0, 1]] * 3]).reshape(3, 2).astype(np.int32)
        cells = np.stack([np.eye(3) * 5.0] * 3).astype(np.float32)
        p = _write_hdf5(tmp_path, pos, types, cells, symbol_map=None)

        with caplog.at_level(logging.WARNING, logger="jaxrens.init.walker_set"):
            ws = load_walker_set(p, n_live_expected=3)

        assert isinstance(ws.symbol_map, dict)
        assert any("symbol_map" in r.message for r in caplog.records)

    def test_no_symbol_map_integer_coded_keys(self, tmp_path, caplog):
        import logging

        pos = np.zeros((2, 2, 3), dtype=np.float32)
        types = np.array([[0, 1], [0, 1]], dtype=np.int32)
        cells = np.stack([np.eye(3) * 5.0] * 2).astype(np.float32)
        p = _write_hdf5(tmp_path, pos, types, cells, symbol_map=None)

        with caplog.at_level(logging.WARNING, logger="jaxrens.init.walker_set"):
            ws = load_walker_set(p, n_live_expected=2)

        assert 0 in ws.symbol_map
        assert 1 in ws.symbol_map


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "walkers.txt"
        p.write_text("dummy")
        with pytest.raises(ValueError, match=r"\.txt"):
            load_walker_set(p, n_live_expected=1)

    def test_unsupported_extension_message_lists_supported(self, tmp_path):
        p = tmp_path / "walkers.npz"
        p.write_bytes(b"dummy")
        with pytest.raises(ValueError, match=r"\.extxyz|\.h5"):
            load_walker_set(p, n_live_expected=1)

    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        p = tmp_path / "missing.extxyz"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_walker_set(p, n_live_expected=4)

    def test_hdf5_extension_dispatches_correctly(self, tmp_path):
        pos = np.zeros((2, 1, 3), dtype=np.float32)
        types = np.zeros((2, 1), dtype=np.int32)
        cells = np.stack([np.eye(3) * 4.0] * 2).astype(np.float32)
        p = _write_hdf5(tmp_path, pos, types, cells, {0: "Si"}, name="walkers.hdf5")
        ws = load_walker_set(p, n_live_expected=2)
        assert ws.positions.shape == (2, 1, 3)

    def test_xyz_extension_dispatches_to_ase(self, tmp_path):
        import ase, ase.io

        cell = np.eye(3) * 5.0
        frames = [
            ase.Atoms(["Si"], positions=[[i, i, i]], cell=cell, pbc=True)
            for i in range(2)
        ]
        p = tmp_path / "walkers.xyz"
        ase.io.write(str(p), frames, format="extxyz")
        ws = load_walker_set(p, n_live_expected=2)
        assert ws.positions.shape == (2, 1, 3)
