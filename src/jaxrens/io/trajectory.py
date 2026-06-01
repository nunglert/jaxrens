"""Trajectory writers: pluggable output format implementations.

TrajectoryWriter protocol + ExtxyzTrajectoryWriter, H5TrajectoryWriter,
NullTrajectoryWriter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ExtxyzTrajectoryWriter:
    """Write dead points in extended XYZ format via ASE."""

    def __init__(
        self,
        path: Path | str,
        symbol_map: dict[int, str],
        wrap: bool = True,
        mode: str = "w",
        restart_iteration: int = 0,
        clean_snapshots: bool = False,
    ):
        self.path = Path(path)
        self.symbol_map = symbol_map
        self.wrap = wrap
        self._mode = mode
        self.clean_snapshots = clean_snapshots
        # Path of the most recently written walker snapshot.  When
        # ``clean_snapshots`` is set we delete it as soon as the *next*
        # snapshot is safely on disk (so the directory keeps at most one
        # walker snapshot, but a complete one always exists).
        self._prev_snapshot_path: Path | None = None
        # First write of a run with mode="w" must overwrite any leftover
        # file; subsequent writes within the same run must append so frames
        # accumulate instead of replacing each other.
        self._first_write = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Restart: rewind frames flushed past the checkpoint before appending.
        if mode == "a" and restart_iteration > 0:
            from jaxrens.io.restart_truncate import truncate_extxyz
            truncate_extxyz(self.path, restart_iteration)

    def write_dead_point(
        self, iteration: int, walker: Any, energy: float
    ) -> None:
        from ase.io import write as ase_write
        from jaxrens.io.formats import walker_to_ase_atoms

        atoms = walker_to_ase_atoms(walker, self.symbol_map)
        atoms.info["iter"] = iteration
        atoms.info["ns_energy"] = energy
        if self.wrap and any(atoms.get_pbc()):
            atoms.wrap()
        append = (self._mode == "a") or not self._first_write
        ase_write(str(self.path), atoms, append=append)
        self._first_write = False

    def write_walker_snapshot(
        self, iteration: int, walkers: Any
    ) -> None:
        # Write all walkers as a snapshot file
        snapshot_path = self.path.with_suffix(f".snap.{iteration}.extxyz")
        from ase.io import write as ase_write
        from jaxrens.io.formats import walker_to_ase_atoms

        positions = np.asarray(walkers["positions"])
        types = np.asarray(walkers["types"])
        energies = np.asarray(walkers["energies"])

        atoms_list = []
        for i in range(positions.shape[0]):
            w = {"positions": positions[i], "types": types[i], "energy": energies[i]}
            if walkers.get("cells") is not None:
                w["box"] = np.asarray(walkers["cells"])[i]
            atoms = walker_to_ase_atoms(w, self.symbol_map)
            atoms.info["iter"] = iteration
            atoms.info["walker_idx"] = i
            atoms_list.append(atoms)

        ase_write(str(snapshot_path), atoms_list)

        # snapshot_clean: now that the new snapshot is fully on disk, drop
        # the previous one (the "second last") so the output directory
        # doesn't accumulate one walker dump per interval.  Deleting only
        # after the new write guarantees at least one complete snapshot
        # always exists, even if the run crashes mid-write.
        if (
            self.clean_snapshots
            and self._prev_snapshot_path is not None
            and self._prev_snapshot_path != snapshot_path
        ):
            try:
                self._prev_snapshot_path.unlink()
            except OSError as exc:
                logger.warning(
                    "snapshot_clean: could not delete previous snapshot %s: %s",
                    self._prev_snapshot_path,
                    exc,
                )
        self._prev_snapshot_path = snapshot_path

    def close(self) -> None:
        pass


class H5TrajectoryWriter:
    """Write dead points in HDF5 format."""

    def __init__(
        self,
        path: Path | str,
        symbol_map: dict[int, str],
        mode: str = "w",
        restart_iteration: int = 0,
        clean_snapshots: bool = False,
    ):
        # ``clean_snapshots`` is accepted for signature parity with
        # ``ExtxyzTrajectoryWriter`` but is a no-op here: H5 snapshots are
        # groups inside the single trajectory file, so deleting a group
        # would not reclaim disk space without repacking.
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.symbol_map = symbol_map
        self._mode = mode
        # Restart: drop per-iteration groups flushed past the checkpoint.
        # Done before opening so the append handle sees the rewound file.
        if mode == "a" and restart_iteration > 0:
            from jaxrens.io.restart_truncate import truncate_h5_traj
            truncate_h5_traj(self.path, restart_iteration)
        self._file = h5py.File(self.path, self._mode)
        self._file.attrs["symbol_map"] = str(symbol_map)

    def write_dead_point(
        self, iteration: int, walker: Any, energy: float
    ) -> None:
        from jaxrens.io.formats import walker_to_h5_group

        grp = self._file.create_group(str(iteration))
        walker_to_h5_group(grp, walker)
        grp.attrs["iteration"] = iteration
        grp.attrs["energy"] = energy

    def write_walker_snapshot(
        self, iteration: int, walkers: Any
    ) -> None:
        grp = self._file.create_group(f"snapshot_{iteration}")
        grp.create_dataset("positions", data=np.asarray(walkers["positions"]))
        grp.create_dataset("energies", data=np.asarray(walkers["energies"]))

    def close(self) -> None:
        self._file.close()


class NullTrajectoryWriter:
    """No-op writer for benchmarking or when output is disabled."""

    def write_dead_point(self, *args: Any, **kwargs: Any) -> None:
        pass

    def write_walker_snapshot(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def create_trajectory_writer(
    format: str,
    path: Path | str,
    symbol_map: dict[int, str],
    **kwargs: Any,
) -> ExtxyzTrajectoryWriter | H5TrajectoryWriter | NullTrajectoryWriter:
    """Factory for trajectory writers."""
    match format:
        case "extxyz":
            return ExtxyzTrajectoryWriter(path, symbol_map, **kwargs)
        case "h5":
            return H5TrajectoryWriter(path, symbol_map, **kwargs)
        case "none":
            return NullTrajectoryWriter()  # ignores mode/wrap/restart_iteration kwargs
        case _:
            raise ValueError(f"Unknown trajectory format: {format!r}")
