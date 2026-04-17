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

from pydantic import BaseModel, ConfigDict


class OutputSchema(BaseModel):
    """Mirrors OutputConfig; YAML key is ``output:``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = "extxyz"
    traj_interval: int = 1
    snapshot_interval: int = 100
    checkpoint_interval: int = 100
    info_interval: int = 100
    out_file_prefix: str = "ns"
    working_dir: Path = Path(".")

    # -- Deferred fields (see module docstring) --
    snapshot_time: float | None = None
    snapshot_clean: bool = False
    wrap_atoms: bool = False
    save_stepsizes: bool = False
    write_traj_db: bool = False
    write_walkers_db: bool = False
