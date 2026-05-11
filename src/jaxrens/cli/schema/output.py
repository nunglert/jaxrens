"""Pydantic schema for the [output] section of a jaxrens YAML config.

DEFERRED fields — accepted and validated but not yet consumed by the runtime.
Users who set these will see a ``logging.warning`` from the resolver:

  - ``snapshot_time``: time-based snapshot interval (runtime has only
    iteration-based snapshots today).
  - ``snapshot_clean``: delete older snapshots on write.
  - ``wrap_atoms``: wrap atom positions into the cell on output.
  - ``save_stepsizes``: write a step-size trajectory file.
  - ``write_traj_db``: write a trajectory database (no DB writer exists yet).
  - ``write_walkers_db``: write a walker database (no walker DB writer exists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class OutputSpec(BaseModel):
    """Mirrors OutputConfig; YAML key is ``output:``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = "extxyz"
    # The four ``*_interval`` fields are widened to int|float so the
    # per-walker interval-unit mode (see RootSpec.interval_units) can accept
    # fractional walker-sweeps (e.g. ``info_interval: 0.2``).  The resolver
    # rounds and casts to int before constructing OutputConfig.
    traj_interval: int | float = 1
    snapshot_interval: int | float = 100
    checkpoint_interval: int | float = 100
    info_interval: int | float = 100
    out_file_prefix: str = "ns"
    working_dir: Path = Path(".")
    log_level: Literal["info", "debug"] = "info"

    # -- Deferred fields (see module docstring) --
    snapshot_time: float | None = None
    snapshot_clean: bool = False
    wrap_atoms: bool = False
    save_stepsizes: bool = False
    write_traj_db: bool = False
    write_walkers_db: bool = False
