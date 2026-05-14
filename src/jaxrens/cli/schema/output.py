"""Pydantic schema for the [output] section of a jaxrens YAML config.

DEFERRED fields — accepted and validated but not yet consumed by the runtime.
Users who set these will see a ``logging.warning`` from the resolver:

  - ``snapshot_time``: time-based snapshot interval (runtime has only
    iteration-based snapshots today).
  - ``snapshot_clean``: delete older snapshots on write.
  - ``wrap_atoms``: wrap atom positions into the cell on output.
  - ``write_traj_db``: write a trajectory database (no DB writer exists yet).
  - ``write_walkers_db``: write a walker database (no walker DB writer exists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    # Per-iter chain-phase acceptance log.  When ``save_acc_rates`` is
    # True the runtime registers an ``AccRatesCallback`` that writes
    # ``<prefix>.acc_rates.h5`` every ``acc_rates_interval`` iterations.
    # Independent of ``adaptation.full_auto``.
    save_acc_rates: bool = False
    acc_rates_interval: int | float = Field(
        default=1,
        description=(
            "Fire AccRatesCallback every N iterations.  Default 1 (every "
            "iter); set higher for long runs to reduce I/O."
        ),
    )

    # Per-iter neighbor-bucket diagnostic log.  When ``save_max_neighbors``
    # is True the runtime registers a ``MaxNeighborsCallback`` that writes
    # ``<prefix>.max_neighbors.h5`` every ``max_neighbors_interval``
    # iterations.  No-op for backends that don't use neighbor lists
    # (e.g. all-pairs LJ, toy backends).  Honours ``interval_units``.
    save_max_neighbors: bool = False
    max_neighbors_interval: int | float = Field(
        default=1,
        description=(
            "Fire MaxNeighborsCallback every N iterations.  Default 1; "
            "raise to reduce I/O on long runs."
        ),
    )

    # Per-fire inter-RE swap log.  When ``save_re_stats`` is True and
    # ``inter_re`` is configured, the runtime registers an
    # ``RECallback`` that writes ``<prefix>.re_stats.h5`` once per
    # swap fire (cadence is dictated by ``inter_re.re_interval`` —
    # there is intentionally no separate interval here, since the file
    # would just sub-sample an already sub-sampled signal).  No-op
    # when ``inter_re`` is not configured.
    save_re_stats: bool = False

    # Finite-difference temperature estimator (Baldock et al. 2017).
    # ``temperature_lag`` is the length of the Emax FIFO used for the
    # finite difference; ``None`` disables the callback entirely.
    # Both ``temperature_lag`` and ``temperature_interval`` honour
    # ``interval_units`` (``per_walker`` scales by ``n_live``).
    # ``temperature_kB`` defaults to eV/K (ASE convention) — set ``1.0``
    # for reduced-unit backends (LJ, harmonic).
    temperature_lag: int | float | None = 100
    temperature_interval: int | float = 100
    temperature_kB: float = 8.6173324e-5

    # -- Deferred fields (see module docstring) --
    snapshot_time: float | None = None
    snapshot_clean: bool = False
    wrap_atoms: bool = False
    write_traj_db: bool = False
    write_walkers_db: bool = False
