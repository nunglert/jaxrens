"""Test I/O layer: formats, checkpoints, trajectory, energy log."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _make_pmap_vmap_ns_state(
    G: int = 1,
    P: int = 2,
    n_walkers: int = 4,
    n_atoms: int = 2,
    max_dead: int = 5,
):
    """Construct a fake ``(G, P, ...)``-shaped NS state dict without actually running pmap.

    Used to test checkpoint round-trips for multi-GPU output shapes.
    """
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
    def test_save_load_roundtrip(self, ns_state, tmp_path):
        path = tmp_path / "test.checkpoint.h5"
        save_checkpoint(path, ns_state, symbol_map={0: "Si"})
        assert path.exists()

        loaded = load_checkpoint(path)
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

    def test_rng_key_roundtrip_preserves_stream(self, ns_state, tmp_path):
        """save_checkpoint persists the PRNG state so load can resume it,
        keeping resumed runs reproducible instead of silently reseeding.
        """
        path = tmp_path / "rng_roundtrip.h5"
        save_checkpoint(path, ns_state)

        loaded = load_checkpoint(path)
        # Raw uint32 buffer must be present and match the saved key's data.
        assert (
            loaded["rng_key_data"] is not None
        ), "save_checkpoint must persist rng_key as 'rng_key_data'"
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
        # Build a legacy-style state dict (no rng_key field).
        legacy = {k: v for k, v in ns_state.items() if k != "rng_key"}
        save_checkpoint(path, legacy)
        loaded = load_checkpoint(path)
        assert loaded["rng_key_data"] is None

    def test_scalar_log_evidence_shape_roundtrip(self, ns_state, tmp_path):
        """Scalar log_evidence (SingleRun) round-trips with shape ()."""
        path = tmp_path / "scalar_le.h5"
        save_checkpoint(path, ns_state)
        loaded = load_checkpoint(path)
        assert (
            loaded["log_evidence"].shape == ()
        ), f"Expected scalar, got shape {loaded['log_evidence'].shape}"
        assert jnp.allclose(loaded["log_evidence"], ns_state["log_evidence"])

    # ------------------------------------------------------------------
    # VmapRuns: (n_runs,)-shaped state
    # ------------------------------------------------------------------

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
            "log_evidence": jnp.array([-2.5, -3.0]),  # (n_runs,)
            "iteration": jnp.array([3, 3], dtype=jnp.int32),
            "n_dead": jnp.array([3, 3], dtype=jnp.int32),
            "n_walkers": n_walkers,
        }

        path = tmp_path / "vmap.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert loaded["log_evidence"].shape == (
            n_runs,
        ), f"Expected (n_runs,), got {loaded['log_evidence'].shape}"
        assert jnp.allclose(loaded["log_evidence"], state["log_evidence"])
        assert loaded["n_dead"].shape == (n_runs,)
        assert loaded["dead_energies"].shape == (n_runs, max_dead)

    # ------------------------------------------------------------------
    # PmapVmapRuns: (G, P)-shaped state
    # ------------------------------------------------------------------

    def test_pmap_vmap_log_evidence_shape_roundtrip(self, tmp_path):
        """PmapVmapRuns (G=1, P=2) state: log_evidence shape (1, 2) round-trips."""
        G, P = 1, 2
        state = _make_pmap_vmap_ns_state(G=G, P=P)
        path = tmp_path / "pmap_vmap.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert loaded["log_evidence"].shape == (
            G,
            P,
        ), f"Expected ({G}, {P}), got {loaded['log_evidence'].shape}"

    def test_pmap_vmap_log_evidence_value_roundtrip(self, tmp_path):
        """PmapVmapRuns state: log_evidence values are preserved."""
        state = _make_pmap_vmap_ns_state(G=1, P=2)
        path = tmp_path / "pmap_vmap_val.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert jnp.allclose(loaded["log_evidence"], state["log_evidence"]), (
            f"log_evidence mismatch: stored={state['log_evidence']}, "
            f"loaded={loaded['log_evidence']}"
        )

    def test_pmap_vmap_n_dead_shape_roundtrip(self, tmp_path):
        """PmapVmapRuns state: n_dead has shape (G, P) after round-trip."""
        G, P = 1, 2
        state = _make_pmap_vmap_ns_state(G=G, P=P)
        path = tmp_path / "pmap_vmap_ndead.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        n_dead_np = np.asarray(loaded["n_dead"])
        assert n_dead_np.shape == (
            G,
            P,
        ), f"Expected n_dead shape ({G}, {P}), got {n_dead_np.shape}"

    def test_pmap_vmap_dead_energies_shape_roundtrip(self, tmp_path):
        """PmapVmapRuns state: dead_energies shape (G, P, max_dead) preserved."""
        G, P, max_dead = 1, 2, 5
        state = _make_pmap_vmap_ns_state(G=G, P=P, max_dead=max_dead)
        path = tmp_path / "pmap_vmap_de.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert loaded["dead_energies"].shape == (
            G,
            P,
            max_dead,
        ), f"Expected ({G}, {P}, {max_dead}), got {loaded['dead_energies'].shape}"
        # First 3 entries per run match stored values.
        assert jnp.allclose(
            loaded["dead_energies"][:, :, :3],
            state["dead_energies"][:, :, :3],
        )

    def test_pmap_vmap_dead_energies_value_roundtrip(self, tmp_path):
        """PmapVmapRuns state: dead_energies values round-trip exactly."""
        state = _make_pmap_vmap_ns_state(G=1, P=2)
        path = tmp_path / "pmap_vmap_de_val.checkpoint.h5"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert jnp.allclose(
            loaded["dead_energies"],
            state["dead_energies"],
            equal_nan=True,  # inf == inf under allclose
        ) or jnp.all(
            (loaded["dead_energies"] == state["dead_energies"])
            | (
                jnp.isinf(loaded["dead_energies"])
                & jnp.isinf(state["dead_energies"])
            )
        )


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

    def _drifted_walker(self):
        # One atom sitting 2 cells (8 Å) outside a 4 Å cubic box on x.
        return {
            "positions": np.array([[9.0, 0.5, 0.5]]),
            "types": np.array([0]),
            "energy": 0.0,
            "box": 4.0 * np.eye(3),
        }

    def _read_h5_pos(self, path):
        import h5py

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

    def test_h5_factory_accepts_wrap(self, tmp_path, symbol_map):
        """Regression: the run path passes ``wrap=`` to every format; h5 must
        accept it (it previously raised TypeError)."""
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
    """``ExtxyzTrajectoryWriter`` walker-snapshot retention."""

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
        """A previous snapshot deleted out-of-band must not crash the writer."""
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
    """``H5TrajectoryWriter`` walker-snapshot content + retention."""

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
        """Snapshot must carry the whole walker (positions, types, energy,
        box), not just positions + energies, and round-trip via
        ``h5_group_to_walker``."""
        path = tmp_path / "traj.h5"
        writer = H5TrajectoryWriter(path, symbol_map)
        writer.write_walker_snapshot(10, self._snap(10, periodic=True))
        writer.close()

        import h5py

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
