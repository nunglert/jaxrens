"""Tests for the I/O layer.

Covers the WalkerState <-> ASE / HDF5 conversions in io.formats, checkpoint
save/load for single, VmapRuns and PmapVmapRuns states, the .energies log,
and the extxyz / HDF5 / null trajectory writers with their snapshot
retention.
"""

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.io import read

from jaxrens.io.checkpoint import load_checkpoint, save_checkpoint
from jaxrens.io.energy_log import EnergyLog, EnergyLogger
from jaxrens.io.formats import (
    ase_atoms_to_walker,
    h5_group_to_walker,
    walker_to_ase_atoms,
    walker_to_h5_group,
)
from jaxrens.io.trajectory import (
    ExtxyzTrajectoryWriter,
    H5TrajectoryWriter,
    NullTrajectoryWriter,
    create_trajectory_writer,
)
from jaxrens.state.walker import WalkerState


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


class TestFormats:
    """Check the WalkerState <-> ASE Atoms and HDF5 group conversions."""

    def test_walker_to_ase_roundtrip(self, walker, symbol_map):
        atoms = walker_to_ase_atoms(walker, symbol_map)
        assert atoms.get_chemical_symbols() == ["Si", "O"]
        assert np.allclose(atoms.get_positions(), np.array(walker.positions))
        assert atoms.info["ns_energy"] == pytest.approx(-1.5)
        assert atoms.pbc.all()
        assert np.allclose(atoms.cell[:], 5.0 * np.eye(3))

        inv_map = {"Si": 0, "O": 1}
        reconstructed = ase_atoms_to_walker(atoms, inv_map)
        assert jnp.allclose(reconstructed.positions, walker.positions)
        assert jnp.array_equal(reconstructed.types, walker.types)
        assert reconstructed.n_atoms == 2
        assert jnp.allclose(reconstructed.cell, walker.cell)

    def test_walker_to_h5_roundtrip(self, walker, tmp_path):
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
        assert atoms.get_chemical_symbols() == ["Si"]
        assert atoms.info["ns_energy"] == pytest.approx(-1.0)
        # No box: the atoms are non-periodic and come back without a cell.
        assert not atoms.pbc.any()
        assert ase_atoms_to_walker(atoms, {"Si": 0}).cell is None


def _make_pmap_vmap_ns_state(
    G: int = 1,
    P: int = 2,
    n_walkers: int = 4,
    n_atoms: int = 2,
    max_dead: int = 5,
):
    """Build a fake ``(G, P, ...)``-shaped NS state dict, as
    run_ns_multi_gpu returns, without running pmap."""
    key = jax.random.key(0)
    return {
        "positions": jnp.zeros((G, P, n_walkers, n_atoms, 3)),
        "types": jnp.zeros((G, P, n_walkers, n_atoms), dtype=jnp.int32),
        "energies": jax.random.uniform(key, (G, P, n_walkers)),
        "cells": None,
        "dead_energies": jnp.full((G, P, max_dead), jnp.inf)
        .at[:, :, :3]
        .set(jnp.array([5.0, 4.0, 3.0])),
        "dead_positions": jnp.zeros((G, P, max_dead, n_atoms, 3)),
        "log_evidence": jnp.full((G, P), -2.5)
        + jax.random.uniform(key, (G, P)) * 0.1,
        "iteration": jnp.full((G, P), 3, dtype=jnp.int32),
        "n_dead": jnp.full((G, P), 3, dtype=jnp.int32),
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(7),
    }


class TestCheckpoint:
    """Check that save_checkpoint / load_checkpoint round-trip the NS state
    for scalar (SingleRun), (n_runs,) and (G, P) batch shapes."""

    def test_save_load_roundtrip(self, ns_state, tmp_path):
        path = tmp_path / "test.checkpoint.h5"
        save_checkpoint(path, ns_state, symbol_map={0: "Si"})
        assert path.exists()

        loaded = load_checkpoint(path)
        assert loaded["iteration"] == 3
        assert loaded["n_dead"] == 3
        assert loaded["n_walkers"] == 10
        assert loaded["log_evidence"].shape == ()
        assert jnp.allclose(loaded["log_evidence"], ns_state["log_evidence"])
        assert jnp.allclose(
            loaded["dead_energies"][:3], ns_state["dead_energies"][:3]
        )

    def test_save_without_cells(self, ns_state, tmp_path):
        path = tmp_path / "no_cells.h5"
        save_checkpoint(path, ns_state)
        loaded = load_checkpoint(path)
        assert loaded["cells"] is None

    def test_rng_key_roundtrip_preserves_stream(self, ns_state, tmp_path):
        """save_checkpoint persists the PRNG state so load can resume it,
        keeping resumed runs reproducible instead of silently reseeding.
        """
        path = tmp_path / "rng_roundtrip.h5"
        save_checkpoint(path, ns_state)

        loaded = load_checkpoint(path)
        # Raw uint32 buffer must be present and match the saved key's data.
        assert loaded["rng_key_data"] is not None
        original_data = np.asarray(jax.random.key_data(ns_state["rng_key"]))
        assert np.array_equal(loaded["rng_key_data"], original_data)

        # Re-wrapped key must yield identical draws to the original key.
        restored = jax.random.wrap_key_data(
            jnp.asarray(loaded["rng_key_data"])
        )
        s_orig = jax.random.uniform(ns_state["rng_key"], (5,))
        s_restored = jax.random.uniform(restored, (5,))
        assert jnp.array_equal(s_orig, s_restored)

    def test_legacy_checkpoint_without_rng_key(self, ns_state, tmp_path):
        """Older checkpoint files have no rng_key_data → load returns None."""
        path = tmp_path / "legacy.h5"
        legacy = {k: v for k, v in ns_state.items() if k != "rng_key"}
        save_checkpoint(path, legacy)
        loaded = load_checkpoint(path)
        assert loaded["rng_key_data"] is None

    def test_emax_roundtrip(self, ns_state, tmp_path):
        path = tmp_path / "emax.h5"
        save_checkpoint(path, dict(ns_state, emax=jnp.array(1.5)))
        assert load_checkpoint(path)["emax"] == pytest.approx(1.5)

    def test_missing_emax_saved_as_inf(self, ns_state, tmp_path):
        """A state without emax is saved with +inf shaped like
        log_evidence."""
        path = tmp_path / "no_emax.h5"
        save_checkpoint(path, ns_state)
        emax = load_checkpoint(path)["emax"]
        assert emax.shape == ()
        assert np.isinf(emax)

    def test_vmap_1d_log_evidence_roundtrip(self, tmp_path):
        """VmapRuns (n_runs=2) state round-trips shape and values."""
        n_runs, n_walkers, n_atoms = 2, 4, 2
        max_dead = 5
        key = jax.random.key(10)
        state = {
            "positions": jnp.zeros((n_runs, n_walkers, n_atoms, 3)),
            "types": jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32),
            "energies": jax.random.uniform(key, (n_runs, n_walkers)),
            "cells": None,
            "dead_energies": jnp.full((n_runs, max_dead), jnp.inf)
            .at[:, :3]
            .set(jnp.array([5.0, 4.0, 3.0])),
            "dead_positions": jnp.zeros((n_runs, max_dead, n_atoms, 3)),
            "log_evidence": jnp.array([-2.5, -3.0]),
            "iteration": jnp.array([3, 3], dtype=jnp.int32),
            "n_dead": jnp.array([3, 3], dtype=jnp.int32),
            "n_walkers": n_walkers,
        }

        path = tmp_path / "vmap.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert loaded["log_evidence"].shape == (n_runs,)
        assert jnp.allclose(loaded["log_evidence"], state["log_evidence"])
        assert loaded["n_dead"].shape == (n_runs,)
        assert loaded["dead_energies"].shape == (n_runs, max_dead)

    def test_pmap_vmap_roundtrip(self, tmp_path):
        G, P, max_dead = 1, 2, 5
        state = _make_pmap_vmap_ns_state(G=G, P=P, max_dead=max_dead)
        path = tmp_path / "pmap_vmap.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert loaded["log_evidence"].shape == (G, P)
        assert jnp.allclose(loaded["log_evidence"], state["log_evidence"])
        assert np.array_equal(loaded["n_dead"], np.full((G, P), 3))
        assert np.array_equal(loaded["iteration"], np.full((G, P), 3))
        assert loaded["dead_energies"].shape == (G, P, max_dead)
        # array_equal also matches the +inf padding.
        assert np.array_equal(loaded["dead_energies"], state["dead_energies"])
        assert loaded["emax"].shape == (G, P)
        assert np.all(np.isinf(loaded["emax"]))


def test_energy_log_write_read_roundtrip(tmp_path):
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


class TestTrajectoryWriters:
    """Check the trajectory writers, position wrapping and the
    create_trajectory_writer factory."""

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

        with h5py.File(path, "r") as f:
            assert "0" in f
            assert "1" in f
            assert "2" in f

    def _drifted_walker(self):
        # One atom sitting 2 cells (8 Å) outside a 4 Å cubic box on x.
        return {
            "positions": np.array([[9.0, 0.5, 0.5]]),
            "types": np.array([0]),
            "energy": 0.0,
            "box": 4.0 * np.eye(3),
        }

    def _read_h5_pos(self, path):
        with h5py.File(path, "r") as f:
            return np.asarray(f["0"]["positions"][:])

    def test_h5_wrap_true_wraps_into_cell(self, tmp_path, symbol_map):
        path = tmp_path / "wrap.h5"
        writer = H5TrajectoryWriter(path, symbol_map, wrap=True)
        writer.write_dead_point(0, self._drifted_walker(), 0.0)
        writer.close()
        pos = self._read_h5_pos(path)
        # 9.0 mod 4.0 -> 1.0; all coords land in [0, 4).
        np.testing.assert_allclose(pos[0], [1.0, 0.5, 0.5], atol=1e-5)

    def test_h5_wrap_false_keeps_absolute(self, tmp_path, symbol_map):
        path = tmp_path / "nowrap.h5"
        writer = H5TrajectoryWriter(path, symbol_map, wrap=False)
        writer.write_dead_point(0, self._drifted_walker(), 0.0)
        writer.close()
        pos = self._read_h5_pos(path)
        np.testing.assert_allclose(pos[0], [9.0, 0.5, 0.5], atol=1e-5)

    def _read_extxyz_pos(self, path):
        return read(path).get_positions()

    def test_extxyz_wrap_true_wraps_into_cell(self, tmp_path, symbol_map):
        path = tmp_path / "wrap.extxyz"
        writer = ExtxyzTrajectoryWriter(path, symbol_map, wrap=True)
        writer.write_dead_point(0, self._drifted_walker(), 0.0)
        writer.close()
        pos = self._read_extxyz_pos(path)
        np.testing.assert_allclose(pos[0], [1.0, 0.5, 0.5], atol=1e-5)

    def test_extxyz_wrap_false_keeps_absolute(self, tmp_path, symbol_map):
        path = tmp_path / "nowrap.extxyz"
        writer = ExtxyzTrajectoryWriter(path, symbol_map, wrap=False)
        writer.write_dead_point(0, self._drifted_walker(), 0.0)
        writer.close()
        pos = self._read_extxyz_pos(path)
        np.testing.assert_allclose(pos[0], [9.0, 0.5, 0.5], atol=1e-5)

    def test_h5_factory_accepts_wrap(self, tmp_path, symbol_map):
        """The run path passes wrap, mode, restart_iteration and
        clean_snapshots to every format; the h5 writer accepts them all."""
        writer = create_trajectory_writer(
            "h5",
            tmp_path / "f.h5",
            symbol_map,
            wrap=True,
            mode="w",
            restart_iteration=0,
            clean_snapshots=False,
        )
        assert isinstance(writer, H5TrajectoryWriter)
        writer.close()

    def test_create_factory(self, tmp_path, symbol_map):
        writer = create_trajectory_writer("none", tmp_path / "x", symbol_map)
        assert isinstance(writer, NullTrajectoryWriter)

        writer = create_trajectory_writer("h5", tmp_path / "y.h5", symbol_map)
        assert isinstance(writer, H5TrajectoryWriter)
        writer.close()

    def test_create_unknown_raises(self, tmp_path, symbol_map):
        with pytest.raises(ValueError):
            create_trajectory_writer("zarr", tmp_path / "z", symbol_map)


class TestSnapshotClean:
    """Check that ExtxyzTrajectoryWriter keeps only the latest walker
    snapshot file with clean_snapshots=True and all of them otherwise."""

    @staticmethod
    def _snap(i):
        # One walker, one atom: positions (1, 1, 3), per-walker types (1, 1).
        return {
            "positions": np.array([[[float(i), 0.0, 0.0]]]),
            "types": np.array([[0]]),
            "energies": np.array([float(-i)]),
            "cells": None,
        }

    def _snap_files(self, tmp_path):
        return sorted(tmp_path.glob("*.snap.*.extxyz"))

    def test_clean_keeps_only_latest(self, tmp_path, symbol_map):
        writer = ExtxyzTrajectoryWriter(
            tmp_path / "ns.traj.extxyz", symbol_map, clean_snapshots=True
        )
        for i in (10, 20, 30):
            writer.write_walker_snapshot(i, self._snap(i))
            # After each write the new snapshot exists; once past the first,
            # the previous one must be gone — never more than one on disk.
            files = self._snap_files(tmp_path)
            assert len(files) == 1
            assert files[0].name.endswith(f".snap.{i}.extxyz")

    def test_no_clean_accumulates(self, tmp_path, symbol_map):
        writer = ExtxyzTrajectoryWriter(
            tmp_path / "ns.traj.extxyz", symbol_map, clean_snapshots=False
        )
        for i in (10, 20, 30):
            writer.write_walker_snapshot(i, self._snap(i))
        assert len(self._snap_files(tmp_path)) == 3

    def test_clean_survives_missing_previous(self, tmp_path, symbol_map):
        """A previous snapshot deleted by hand does not crash the writer."""
        writer = ExtxyzTrajectoryWriter(
            tmp_path / "ns.traj.extxyz", symbol_map, clean_snapshots=True
        )
        writer.write_walker_snapshot(10, self._snap(10))
        (tmp_path / "ns.traj.snap.10.extxyz").unlink()
        # Should warn-and-continue, leaving the new snapshot in place.
        writer.write_walker_snapshot(20, self._snap(20))
        files = self._snap_files(tmp_path)
        assert len(files) == 1
        assert files[0].name.endswith(".snap.20.extxyz")


class TestH5WalkerSnapshot:
    """Check the content of H5TrajectoryWriter walker snapshots and that
    clean_snapshots keeps only the latest snapshot group."""

    @staticmethod
    def _snap(i, periodic=False):
        # One walker, two atoms.
        d = {
            "positions": np.array([[[float(i), 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            "types": np.array([[0, 1]]),
            "energies": np.array([float(-i)]),
            "cells": None,
        }
        if periodic:
            d["cells"] = np.array([[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]])
        return d

    @staticmethod
    def _snap_groups(writer):
        return sorted(
            k for k in writer._file.keys() if k.startswith("snapshot_")
        )

    def test_writes_full_walker(self, tmp_path, symbol_map):
        """A snapshot stores the whole walker (positions, types, energy,
        box) and round-trips through h5_group_to_walker."""
        path = tmp_path / "traj.h5"
        writer = H5TrajectoryWriter(path, symbol_map)
        writer.write_walker_snapshot(10, self._snap(10, periodic=True))
        writer.close()

        with h5py.File(path, "r") as f:
            grp = f["snapshot_10"]
            assert grp.attrs["n_walkers"] == 1
            wg = grp["walker_0"]
            assert {"positions", "types", "energy", "box"} <= set(wg.keys())
            walker = h5_group_to_walker(wg)
            assert walker.positions.shape == (2, 3)
            assert np.asarray(walker.types).tolist() == [0, 1]
            assert walker.cell is not None
            assert float(walker.energy) == pytest.approx(-10.0)

    def test_clean_keeps_only_latest_group(self, tmp_path, symbol_map):
        writer = H5TrajectoryWriter(
            tmp_path / "traj.h5", symbol_map, clean_snapshots=True
        )
        for i in (10, 20, 30):
            writer.write_walker_snapshot(i, self._snap(i))
            assert self._snap_groups(writer) == [f"snapshot_{i}"]
        writer.close()

    def test_no_clean_accumulates_groups(self, tmp_path, symbol_map):
        writer = H5TrajectoryWriter(
            tmp_path / "traj.h5", symbol_map, clean_snapshots=False
        )
        for i in (10, 20, 30):
            writer.write_walker_snapshot(i, self._snap(i))
        assert len(self._snap_groups(writer)) == 3
        writer.close()
