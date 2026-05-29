"""Energy log: write and read .energies files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EnergyLog:
    """Parsed energy log data."""

    iterations: np.ndarray
    energies: np.ndarray
    volumes: np.ndarray
    n_walkers: int
    n_cull: int
    n_atoms: int
    n_dof: int


class EnergyLogger:
    """Writes and reads the .energies file."""

    def __init__(
        self,
        path: Path | str,
        n_walkers: int,
        n_cull: int = 1,
        n_dof: int = 0,
        n_atoms: int = 0,
        mode: str = "w",
        restart_iteration: int = 0,
    ):
        self.path = Path(path)
        self.n_walkers = n_walkers
        self.n_cull = n_cull
        self.n_dof = n_dof
        self.n_atoms = n_atoms
        self._mode = mode
        self._restart_iteration = restart_iteration
        self._file = None

    def write_header(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Restart: rewind the on-disk log to the checkpoint before appending,
        # dropping any dead-point rows the previous process flushed past it.
        if self._mode == "a" and self._restart_iteration > 0:
            from jaxrens.io.restart_truncate import truncate_energies
            truncate_energies(self.path, self._restart_iteration)
        # mode="a" + existing non-empty file: header already on disk, skip it.
        skip_header = self._mode == "a" and self.path.exists() and self.path.stat().st_size > 0
        self._file = open(self.path, self._mode)
        if not skip_header:
            self._file.write(
                f"{self.n_walkers} {self.n_cull} {self.n_dof} 0.0 {self.n_atoms}\n"
            )
            self._file.flush()

    def write_entry(
        self, iteration: int, energy: float, volume: float = 0.0
    ) -> None:
        if self._file is None:
            self.write_header()
        self._file.write(f"{iteration} {energy:.10e} {volume:.10e}\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def read(path: Path | str) -> EnergyLog:
        """Parse a .energies file into structured data."""
        path = Path(path)
        with open(path) as f:
            header = f.readline().strip().split()
            n_walkers = int(header[0])
            n_cull = int(header[1])
            n_dof = int(header[2])
            n_atoms = int(header[4]) if len(header) > 4 else 0

            iterations = []
            energies = []
            volumes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    iterations.append(int(parts[0]))
                    energies.append(float(parts[1]))
                    volumes.append(float(parts[2]) if len(parts) > 2 else 0.0)

        return EnergyLog(
            iterations=np.array(iterations),
            energies=np.array(energies),
            volumes=np.array(volumes),
            n_walkers=n_walkers,
            n_cull=n_cull,
            n_atoms=n_atoms,
            n_dof=n_dof,
        )
