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
    ):
        self.path = Path(path)
        self.symbol_map = symbol_map
        self.wrap = wrap
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
        ase_write(str(self.path), atoms, append=True)

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
            if walkers.get("boxes") is not None:
                w["box"] = np.asarray(walkers["boxes"])[i]
            atoms = walker_to_ase_atoms(w, self.symbol_map)
            atoms.info["iter"] = iteration
            atoms.info["walker_idx"] = i
            atoms_list.append(atoms)

        ase_write(str(snapshot_path), atoms_list)

    def close(self) -> None:
        pass


class H5TrajectoryWriter:
    """Write dead points in HDF5 format."""

    def __init__(self, path: Path | str, symbol_map: dict[int, str]):
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.symbol_map = symbol_map
        self._file = h5py.File(self.path, "w")
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
            return H5TrajectoryWriter(path, symbol_map)
        case "none":
            return NullTrajectoryWriter()
        case _:
            raise ValueError(f"Unknown trajectory format: {format!r}")
