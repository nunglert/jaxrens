"""Unit tests for jaxrens.init.structure.load_structure.

Covers round-trip correctness, error conditions, and symbol_map ordering.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.init.structure import load_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_extxyz(path: Path, symbols: list[str], positions: np.ndarray, cell: np.ndarray) -> None:
    """Write a minimal extxyz file for testing."""
    import ase
    import ase.io
    atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    ase.io.write(str(path), atoms)


# ---------------------------------------------------------------------------
# Round-trip: positions, types, cell, symbol_map
# ---------------------------------------------------------------------------

class TestLoadStructureRoundTrip:
    def test_positions_shape(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]], dtype=np.float32)
        cell = np.diag([5.0, 5.0, 5.0]).astype(np.float32)
        p = tmp_path / "two_si.extxyz"
        _write_extxyz(p, ["Si", "Si"], pos, cell)
        positions, types, loaded_cell, symbol_map = load_structure(p)
        assert positions.shape == (2, 3)

    def test_positions_values(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]], dtype=np.float32)
        cell = np.diag([5.0, 5.0, 5.0]).astype(np.float32)
        p = tmp_path / "si.extxyz"
        _write_extxyz(p, ["Si", "Si"], pos, cell)
        positions, _, _, _ = load_structure(p)
        np.testing.assert_allclose(np.array(positions), pos, atol=1e-5)

    def test_cell_shape(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        cell = np.diag([4.0, 4.0, 4.0]).astype(np.float32)
        p = tmp_path / "single.extxyz"
        _write_extxyz(p, ["Si"], pos, cell)
        _, _, loaded_cell, _ = load_structure(p)
        assert loaded_cell.shape == (3, 3)

    def test_cell_values(self, tmp_path):
        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        cell = np.diag([4.0, 5.0, 6.0]).astype(np.float32)
        p = tmp_path / "single.extxyz"
        _write_extxyz(p, ["Si"], pos, cell)
        _, _, loaded_cell, _ = load_structure(p)
        np.testing.assert_allclose(np.array(loaded_cell), cell, atol=1e-5)

    def test_types_dtype(self, tmp_path):
        pos = np.zeros((2, 3), dtype=np.float32)
        cell = np.eye(3, dtype=np.float32) * 5.0
        p = tmp_path / "dtype.extxyz"
        _write_extxyz(p, ["Si", "O"], pos, cell)
        _, types, _, _ = load_structure(p)
        assert types.dtype == jnp.int32

    def test_positions_dtype(self, tmp_path):
        pos = np.zeros((2, 3), dtype=np.float64)
        cell = np.eye(3) * 5.0
        p = tmp_path / "dtype.extxyz"
        _write_extxyz(p, ["Si", "O"], pos, cell)
        positions, _, _, _ = load_structure(p)
        assert positions.dtype == jnp.float32

    def test_cell_dtype(self, tmp_path):
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        p = tmp_path / "dtype.extxyz"
        _write_extxyz(p, ["Si"], pos, cell)
        _, _, loaded_cell, _ = load_structure(p)
        assert loaded_cell.dtype == jnp.float32

    def test_symbol_map_single_element(self, tmp_path):
        pos = np.zeros((3, 3))
        cell = np.eye(3) * 5.0
        p = tmp_path / "three_si.extxyz"
        _write_extxyz(p, ["Si", "Si", "Si"], pos, cell)
        _, _, _, symbol_map = load_structure(p)
        assert symbol_map == {0: "Si"}

    def test_single_atom(self, tmp_path):
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        p = tmp_path / "one.extxyz"
        _write_extxyz(p, ["Si"], pos, cell)
        positions, types, _, symbol_map = load_structure(p)
        assert positions.shape == (1, 3)
        assert types.shape == (1,)
        assert symbol_map == {0: "Si"}

    def test_path_as_string(self, tmp_path):
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        p = tmp_path / "str.extxyz"
        _write_extxyz(p, ["Si"], pos, cell)
        positions, types, _, _ = load_structure(str(p))
        assert positions.shape == (1, 3)


# ---------------------------------------------------------------------------
# Multi-element: Si2O3
# ---------------------------------------------------------------------------

class TestLoadStructureMultiElement:
    def test_si2o3_symbol_map(self, tmp_path):
        pos = np.zeros((5, 3))
        pos[1, 0] = 1.5
        pos[2, 1] = 1.5
        pos[3, 2] = 1.5
        pos[4, 0] = 3.0
        cell = np.eye(3) * 6.0
        p = tmp_path / "si2o3.extxyz"
        _write_extxyz(p, ["Si", "Si", "O", "O", "O"], pos, cell)
        _, types, _, symbol_map = load_structure(p)
        assert symbol_map == {0: "Si", 1: "O"}
        assert list(np.array(types)) == [0, 0, 1, 1, 1]

    def test_types_values_multi(self, tmp_path):
        pos = np.zeros((4, 3))
        for i in range(4):
            pos[i, 0] = i * 1.5
        cell = np.eye(3) * 8.0
        p = tmp_path / "mixed.extxyz"
        _write_extxyz(p, ["Si", "O", "Si", "O"], pos, cell)
        _, types, _, symbol_map = load_structure(p)
        # first-appearance: Si=0, O=1
        assert symbol_map == {0: "Si", 1: "O"}
        assert list(np.array(types)) == [0, 1, 0, 1]

    def test_noncontiguous_ordering_first_appearance(self, tmp_path):
        """[Si, O, Si, O] -> symbol_map {0: Si, 1: O} (first appearance)."""
        pos = np.zeros((4, 3))
        for i in range(4):
            pos[i, 0] = i * 1.5
        cell = np.eye(3) * 8.0
        p = tmp_path / "noncontig.extxyz"
        _write_extxyz(p, ["Si", "O", "Si", "O"], pos, cell)
        _, types, _, symbol_map = load_structure(p)
        assert symbol_map[0] == "Si"
        assert symbol_map[1] == "O"
        assert int(types[0]) == 0
        assert int(types[1]) == 1
        assert int(types[2]) == 0
        assert int(types[3]) == 1

    def test_three_species(self, tmp_path):
        pos = np.zeros((3, 3))
        for i in range(3):
            pos[i, 0] = i * 2.0
        cell = np.eye(3) * 7.0
        p = tmp_path / "ternary.extxyz"
        _write_extxyz(p, ["Si", "O", "N"], pos, cell)
        _, types, _, symbol_map = load_structure(p)
        assert symbol_map == {0: "Si", 1: "O", 2: "N"}
        assert list(np.array(types)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

class TestLoadStructureErrors:
    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_structure(tmp_path / "does_not_exist.extxyz")

    def test_multi_frame_raises_value_error(self, tmp_path):
        import ase
        import ase.io
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        atoms = ase.Atoms(["Si"], positions=pos, cell=cell, pbc=True)
        p = tmp_path / "multi.extxyz"
        # Write two frames
        ase.io.write(str(p), [atoms, atoms])
        with pytest.raises(ValueError, match="single-frame"):
            load_structure(p)

    def test_zero_cell_raises_value_error(self, tmp_path):
        import ase
        import ase.io
        pos = np.zeros((1, 3))
        atoms = ase.Atoms(["Si"], positions=pos)
        # No cell set -> all-zero cell
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
        import ase
        import ase.io
        pos = np.zeros((1, 3))
        cell = np.eye(3) * 4.0
        atoms = ase.Atoms(["Si"], positions=pos, cell=cell, pbc=True)
        p = tmp_path / "three_frames.extxyz"
        ase.io.write(str(p), [atoms, atoms, atoms])
        with pytest.raises(ValueError, match="3"):
            load_structure(p)
