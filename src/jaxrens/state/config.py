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


@dataclass(frozen=True)
class MoveConfig:
    """Move-specific parameters."""

    move_type: str = "galilean"
    step_size: float = 0.1
    n_steps: int = 10
    weight: float = 1.0  # relative probability for MWG dispatch
    adaptation_warmup: int = 100
    target_acceptance: float = 0.5


@dataclass(frozen=True)
class BackendConfig:
    """Energy backend configuration.

    ``n_atoms`` is intentionally absent: it is derived from the initial
    walker positions (``positions.shape[-2]``) at run time.  Storing it
    here would duplicate the source of truth and require manual synchronisation
    with the initialisation configuration.
    """

    backend_type: str = "lj"
    checkpoint_path: str | None = None
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
    checkpoint_interval: int = 100
    info_interval: int = 100
    out_file_prefix: str = "ns"
    working_dir: Path = field(default_factory=lambda: Path("."))
    log_level: str = "info"  # "info" | "debug"
