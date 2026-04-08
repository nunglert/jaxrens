"""Configuration dataclasses for jaxrens.

Frozen dataclasses replace dict/file-based configuration in the library core.
CLI-level file parsing (reading ns.inp, YAML, etc.) is in cli/parser.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class NSConfig:
    """Nested sampling run configuration."""

    n_live: int = 500
    max_iterations: int = 50_000
    convergence_threshold: float = 0.1
    n_mcmc_steps: int = 20
    n_cull: int = 1
    seed: int = 42

    # Ensemble
    pressure: float | None = None  # pressure for NPT ensemble (None = NVT)
    chemical_potential: float | None = None  # for future sGC support

    # Multi-GPU
    platform: str = "gpu"  # "gpu" | "multi-gpu" | "cpu"
    n_runs: int = 1


@dataclass(frozen=True)
class MoveConfig:
    """Move-specific parameters."""

    move_type: str = "galilean"
    step_size: float = 0.1
    n_steps: int = 10
    adaptation_enabled: bool = True
    adaptation_warmup: int = 100
    target_acceptance: float = 0.5


@dataclass(frozen=True)
class BackendConfig:
    """Energy backend configuration."""

    backend_type: str = "lj"
    checkpoint_path: str | None = None
    n_atoms: int = 13
    periodic: bool = False
    cutoff: float | None = None

    # NeuralIL-specific
    max_neighbors_list: list[int] = field(default_factory=lambda: [30, 35, 40, 45, 50])
    max_neighbors_offset: int = 5


@dataclass(frozen=True)
class OutputConfig:
    """I/O and output configuration."""

    format: str = "extxyz"  # "extxyz" | "h5" | "none"
    traj_interval: int = 1
    snapshot_interval: int = 100
    snapshot_clean: bool = True
    checkpoint_interval: int = 100
    checkpoint_keep: int = 3
    info_interval: int = 100
    out_file_prefix: str = "ns"
    working_dir: Path = field(default_factory=lambda: Path("."))
    write_traj_db: bool = False
