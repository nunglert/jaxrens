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


# ---------------------------------------------------------------------------
# Mode C resolver tests (moved from test_schema.py::TestInitConfigResolverModeC)
# ---------------------------------------------------------------------------

def _make_walker_set_extxyz(
    tmp_path: Path,
    n_live: int = 4,
    symbols: list[str] | None = None,
    cell_size: float = 6.0,
    name: str = "walkers.extxyz",
) -> Path:
    """Write a minimal multi-frame extxyz file for Mode C resolver tests."""
    import ase, ase.io as _ase_io

    if symbols is None:
        symbols = ["Si"]
    n_atoms = len(symbols)
    cell = np.eye(3) * cell_size
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(n_live):
        pos = rng.uniform(0.5, cell_size - 0.5, (n_atoms, 3)).astype(np.float32)
        frames.append(ase.Atoms(list(symbols), positions=pos, cell=cell, pbc=True))
    p = tmp_path / name
    _ase_io.write(str(p), frames, format="extxyz")
    return p


def _make_walker_set_hdf5_resolver(
    tmp_path: Path,
    n_live: int = 4,
    n_atoms: int = 1,
    cell_size: float = 6.0,
    symbol_map: dict | None = None,
    name: str = "walkers.h5",
) -> Path:
    import json as _json

    if symbol_map is None:
        symbol_map = {0: "Si"}
    rng = np.random.default_rng(7)
    positions = rng.uniform(0.5, cell_size - 0.5, (n_live, n_atoms, 3)).astype(np.float32)
    types = np.zeros((n_live, n_atoms), dtype=np.int32)
    cells = np.stack([np.eye(3) * cell_size] * n_live).astype(np.float32)
    p = tmp_path / name
    with h5py.File(p, "w") as f:
        f.create_dataset("positions", data=positions)
        f.create_dataset("types", data=types)
        f.create_dataset("cells", data=cells)
        f.attrs["symbol_map"] = _json.dumps({str(k): v for k, v in symbol_map.items()})
    return p


def _cell_cfg_permissive():
    from jaxrens.cli.schema.cell import CellConfig
    return CellConfig(
        max_volume_per_atom=10000.0,
        min_volume_per_atom=0.01,
        min_aspect_ratio=0.001,
    )


class TestInitConfigResolverModeC:
    """Mode C resolver tests: start_walker_set.

    Moved verbatim from test_schema.py::TestInitConfigResolverModeC.
    """

    def test_extxyz_resolved_init_type(self, tmp_path):
        from jaxrens.cli.resolve import ResolvedInit, _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=4, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert isinstance(result, ResolvedInit)

    def test_extxyz_positions_shape(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=4, symbols=["Si", "Si"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.initial_positions.shape == (4, 2, 3)

    def test_extxyz_types_shape(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=4, symbols=["Si", "Si"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.initial_types.shape == (4, 2)

    def test_extxyz_symbol_map_correct(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si", "O", "O"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.symbol_map == {0: "Si", 1: "O"}

    def test_extxyz_energies_recomputed_not_from_file(self, tmp_path):
        """Energies in the extxyz are stale; resolver must recompute with the backend."""
        import ase, ase.io as _ase_io
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        cell = np.eye(3, dtype=np.float32) * 6.0
        rng = np.random.default_rng(0)
        frames = []
        for _ in range(3):
            pos = rng.uniform(0.5, 5.5, (1, 3)).astype(np.float32)
            atoms = ase.Atoms(["Si"], positions=pos, cell=cell, pbc=True)
            atoms.info["energy"] = -9999.0
            frames.append(atoms)
        p = tmp_path / "stale.extxyz"
        _ase_io.write(str(p), frames, format="extxyz")

        cfg = InitConfig(start_walker_set=p)
        backend = create_harmonic()
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=backend, cell_cfg=_cell_cfg_permissive())
        assert result.initial_energies is not None
        assert not jnp.any(jnp.isclose(result.initial_energies, jnp.float32(-9999.0)))

    def test_hdf5_positions_shape(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_hdf5_resolver(tmp_path, n_live=5, n_atoms=2)
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=5, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.initial_positions.shape == (5, 2, 3)

    def test_hdf5_symbol_map_correct(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_hdf5_resolver(tmp_path, n_live=3, n_atoms=1, symbol_map={0: "O"})
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.symbol_map == {0: "O"}

    def test_hdf5_energies_recomputed(self, tmp_path):
        """Resolver must recompute energies regardless of what is stored in the file."""
        import json as _json
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        rng = np.random.default_rng(5)
        positions = rng.uniform(0, 5, (4, 1, 3)).astype(np.float32)
        types = np.zeros((4, 1), dtype=np.int32)
        cells = np.stack([np.eye(3) * 6.0] * 4).astype(np.float32)
        p = tmp_path / "stale.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("positions", data=positions)
            f.create_dataset("types", data=types)
            f.create_dataset("cells", data=cells)
            f.create_dataset("energies", data=np.full(4, -9999.0, dtype=np.float32))
            f.attrs["symbol_map"] = _json.dumps({"0": "Si"})

        cfg = InitConfig(start_walker_set=p)
        backend = create_harmonic()
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=backend, cell_cfg=_cell_cfg_permissive())
        assert result.initial_energies is not None
        assert not jnp.any(jnp.isclose(result.initial_energies, jnp.float32(-9999.0)))

    def test_random_initialise_pos_true_warning(self, tmp_path, caplog):
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_pos=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert any("random_initialise_pos" in r.message or "randomiz" in r.message.lower()
                   for r in caplog.records)

    def test_random_initialise_pos_true_positions_verbatim(self, tmp_path, caplog):
        """With random_initialise_pos=True, positions must still come from the file."""
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_pos=True)
        ws = load_walker_set(p, n_live_expected=3)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        np.testing.assert_allclose(
            np.array(result.initial_positions),
            np.array(ws.positions),
            atol=1e-5,
        )

    def test_random_initialise_cell_true_warning(self, tmp_path, caplog):
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_cell=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert any("random_initialise_cell" in r.message or "randomiz" in r.message.lower()
                   for r in caplog.records)

    def test_random_initialise_cell_true_cells_verbatim(self, tmp_path, caplog):
        """With random_initialise_cell=True, cells must still come from the file."""
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_cell=True)
        ws = load_walker_set(p, n_live_expected=3)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        np.testing.assert_allclose(
            np.array(result.initial_cells),
            np.array(ws.cells),
            atol=1e-5,
        )

    def test_cell_config_violation_raises(self, tmp_path):
        """A walker cell that violates CellConfig bounds must raise RuntimeError."""
        import ase, ase.io as _ase_io
        import pytest as _pytest
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig

        cell = np.eye(3, dtype=np.float32) * 6.0
        atoms = ase.Atoms(["Si"], positions=[[3.0, 3.0, 3.0]], cell=cell, pbc=True)
        p = tmp_path / "toosmall.extxyz"
        _ase_io.write(str(p), [atoms], format="extxyz")

        strict_cfg = CellConfig(
            max_volume_per_atom=1.0,
            min_volume_per_atom=0.0001,
            min_aspect_ratio=0.001,
        )
        cfg = InitConfig(start_walker_set=p)
        with _pytest.raises(RuntimeError):
            _resolve_init(cfg, n_live=1, seed=0, cell_cfg=strict_cfg)

    def test_mode_c_end_to_end_jit(self, tmp_path):
        """Mode C resolver -> run_ns -> ns_step under JIT."""
        import jax
        import jax.numpy as _jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step
        from jaxrens.sampling.move_kernel import MoveKernel
        import jaxrens.sampling.moves.random_walk as rw_mod

        p = _make_walker_set_extxyz(tmp_path, n_live=6, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg,
            n_live=6,
            seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive(),
        )

        desc = MoveKernel(
            name="random_walk",
            build_kernel=rw_mod.build_kernel,
            step_size=0.3,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        )
        init_fn, step_fn, _ = build_mwg(backend, [desc])

        key = jax.random.key(77)
        ns_state = init_ns(
            init_fn,
            result.initial_positions,
            result.initial_types,
            result.initial_energies,
            cells=result.initial_cells,
            rng_key=key,
        )

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, _ = jit_ns_step(ns_state, step_fn, n_mcmc_steps=2)
        assert _jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0
