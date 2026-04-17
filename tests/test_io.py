"""Test I/O layer: formats, checkpoints, trajectory, energy log."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from jaxrens.state.walker import WalkerState
from jaxrens.io.formats import (
    walker_to_ase_atoms,
    ase_atoms_to_walker,
    walker_to_h5_group,
    h5_group_to_walker,
)
from jaxrens.io.checkpoint import save_checkpoint, load_checkpoint
from jaxrens.io.energy_log import EnergyLogger, EnergyLog
from jaxrens.io.trajectory import (
    NullTrajectoryWriter,
    H5TrajectoryWriter,
    create_trajectory_writer,
)


@pytest.fixture
def symbol_map():
    return {0: "Si", 1: "O"}


@pytest.fixture
def walker():
    return WalkerState(
        positions=jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        types=jnp.array([0, 1]),
        energy=jnp.array(-1.5),
        cell=5.0 * jnp.eye(3),
        n_atoms=2,
    )


@pytest.fixture
def ns_state():
    return {
        "positions": jnp.zeros((10, 2, 3)),
        "types": jnp.zeros((10, 2), dtype=jnp.int32),
        "energies": jnp.arange(10, dtype=jnp.float32),
        "cells": None,
        "dead_energies": jnp.array([5.0, 4.0, 3.0, jnp.inf, jnp.inf]),
        "dead_positions": jnp.zeros((5, 2, 3)),
        "log_evidence": jnp.array(-2.5),
        "iteration": 3,
        "n_dead": 3,
        "n_walkers": 10,
        "rng_key": jax.random.key(42),
    }


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------


class TestFormats:
    def test_walker_to_ase_roundtrip(self, walker, symbol_map):
        atoms = walker_to_ase_atoms(walker, symbol_map)
        assert atoms.get_chemical_symbols() == ["Si", "O"]
        assert np.allclose(atoms.get_positions(), np.array(walker.positions))
        assert atoms.info["ns_energy"] == pytest.approx(-1.5)

        inv_map = {"Si": 0, "O": 1}
        reconstructed = ase_atoms_to_walker(atoms, inv_map)
        assert jnp.allclose(reconstructed.positions, walker.positions)
        assert jnp.array_equal(reconstructed.types, walker.types)
        assert reconstructed.n_atoms == 2

    def test_walker_to_h5_roundtrip(self, walker, tmp_path):
        import h5py

        path = tmp_path / "test.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("walker")
            walker_to_h5_group(grp, walker)

        with h5py.File(path, "r") as f:
            loaded = h5_group_to_walker(f["walker"])

        assert jnp.allclose(loaded.positions, walker.positions)
        assert jnp.array_equal(loaded.types, walker.types)
        assert jnp.allclose(loaded.energy, walker.energy)
        assert loaded.cell is not None

    def test_dict_walker_to_ase(self, symbol_map):
        w = {
            "positions": np.array([[0.0, 0.0, 0.0]]),
            "types": np.array([0]),
            "energy": -1.0,
        }
        atoms = walker_to_ase_atoms(w, symbol_map)
        assert len(atoms) == 1


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load_roundtrip(self, ns_state, tmp_path):
        path = tmp_path / "test.checkpoint.h5"
        save_checkpoint(path, ns_state, symbol_map={0: "Si"})
        assert path.exists()

        loaded = load_checkpoint(path, rng_key=jax.random.key(99))
        assert loaded["iteration"] == 3
        assert loaded["n_dead"] == 3
        assert loaded["n_walkers"] == 10
        assert jnp.allclose(loaded["log_evidence"], ns_state["log_evidence"])
        assert jnp.allclose(
            loaded["dead_energies"][:3], ns_state["dead_energies"][:3]
        )

    def test_save_without_cells(self, ns_state, tmp_path):
        path = tmp_path / "no_cells.h5"
        save_checkpoint(path, ns_state)
        loaded = load_checkpoint(path)
        assert loaded["cells"] is None


# ---------------------------------------------------------------------------
# Energy log
# ---------------------------------------------------------------------------


class TestEnergyLog:
    def test_write_read_roundtrip(self, tmp_path):
        path = tmp_path / "test.energies"
        logger = EnergyLogger(path, n_walkers=50, n_atoms=3)
        logger.write_header()
        for i in range(10):
            logger.write_entry(i, energy=float(10 - i), volume=float(i * 100))
        logger.close()

        log = EnergyLogger.read(path)
        assert log.n_walkers == 50
        assert log.n_atoms == 3
        assert len(log.energies) == 10
        assert log.energies[0] == pytest.approx(10.0)
        assert log.energies[9] == pytest.approx(1.0)
        assert log.volumes[5] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Trajectory writers
# ---------------------------------------------------------------------------


class TestTrajectoryWriters:
    def test_null_writer(self):
        writer = NullTrajectoryWriter()
        writer.write_dead_point(0, {}, 0.0)
        writer.write_walker_snapshot(0, {})
        writer.close()

    def test_h5_writer(self, tmp_path, symbol_map):
        path = tmp_path / "traj.h5"
        writer = H5TrajectoryWriter(path, symbol_map)

        for i in range(3):
            w = {
                "positions": np.array([[float(i), 0.0, 0.0]]),
                "types": np.array([0]),
                "energy": float(-i),
            }
            writer.write_dead_point(i, w, float(-i))
        writer.close()

        import h5py

        with h5py.File(path, "r") as f:
            assert "0" in f
            assert "1" in f
            assert "2" in f

    def test_create_factory(self, tmp_path, symbol_map):
        writer = create_trajectory_writer("none", tmp_path / "x", symbol_map)
        assert isinstance(writer, NullTrajectoryWriter)

        writer = create_trajectory_writer("h5", tmp_path / "y.h5", symbol_map)
        assert isinstance(writer, H5TrajectoryWriter)
        writer.close()

    def test_create_unknown_raises(self, tmp_path, symbol_map):
        with pytest.raises(ValueError):
            create_trajectory_writer("zarr", tmp_path / "z", symbol_map)
