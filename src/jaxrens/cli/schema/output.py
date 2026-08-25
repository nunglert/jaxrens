"""Pydantic schema for the [output] section of a jaxrens YAML config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class OutputSpec(BaseModel):
    """Mirrors OutputConfig; YAML key is ``output:``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = Field(
        default="extxyz",
        description=(
            "Trajectory writer to use: ``extxyz`` (default, ASE-readable "
            "text), ``h5`` (compact binary, better for long runs), or "
            "``none`` to write no trajectory at all."
        ),
    )
    # The four ``*_interval`` fields are widened to int|float so the
    # per-walker interval-unit mode (see RootSpec.interval_units) can accept
    # fractional walker-sweeps (e.g. ``info_interval: 0.2``).  The resolver
    # rounds and casts to int before constructing OutputConfig.
    traj_interval: int | float = Field(
        default=1,
        description=(
            "Write the culled (dead) walkers to the trajectory every N "
            "iterations.  ``1`` (default) keeps every dead point, which is "
            "what the post-processing estimators expect; raise it only if "
            "trajectory size is a problem."
        ),
    )
    snapshot_interval: int | float = Field(
        default=100,
        description=(
            "Dump the full live-walker population to "
            "``<prefix>.snap.<iter>.extxyz`` every N iterations, for crash "
            "inspection.  See ``snapshot_clean``."
        ),
    )
    checkpoint_interval: int | float = Field(
        default=100,
        description=(
            "Write a restartable checkpoint every N iterations.  This is "
            "the cadence that bounds how much work a crash costs you."
        ),
    )
    info_interval: int | float = Field(
        default=100,
        description=(
            "Print a progress line (iteration, ``E_max``, log-evidence, "
            "per-move acceptance) every N iterations."
        ),
    )
    out_file_prefix: str = Field(
        default="ns",
        description=(
            "Basename shared by every output file, e.g. ``ns.traj.extxyz``, "
            "``ns.energies``, ``ns.restart.h5``."
        ),
    )
    working_dir: Path = Field(
        default=Path("."),
        description=(
            "Directory all output is written to.  Created if absent; "
            "guarded by the output-dir gate, which refuses to overwrite a "
            "directory holding a different run unless you restart or "
            "resume into it."
        ),
    )
    log_level: Literal["info", "debug"] = Field(
        default="info",
        description=(
            "Console log verbosity.  ``debug`` adds per-iteration resolver, "
            "bucket-resize, and adaptation detail."
        ),
    )

    # Shared write-buffer flush cadence for trace loggers (acc_rates,
    # max_neighbors, re_stats).  A flush fires once the NS iteration
    # index has advanced by ``flush_interval`` since the previous
    # flush.  Honours ``RootSpec.interval_units`` like every other
    # ``*_interval`` field — set ``per_walker`` to specify in
    # walker-sweeps.  Adaptation events are NOT affected — that log
    # flushes per-event for crash durability.
    flush_interval: int | float = Field(
        default=1000,
        gt=0,
        description=(
            "Flush trace-logger write buffers every ``flush_interval``"
            " NS iterations.  Default 1000.  In ``per_walker`` mode this"
            " is in walker-sweeps (recommended: 2-10)."
        ),
    )

    # Per-iter chain-phase acceptance log.  When ``save_acc_rates`` is
    # True the runtime registers an ``AccRatesCallback`` that writes
    # ``<prefix>.acc_rates.h5`` every ``acc_rates_interval`` iterations.
    # Independent of ``adaptation.full_auto``.
    save_acc_rates: bool = Field(
        default=False,
        description=(
            "Write per-move, per-chain-phase acceptance rates to "
            "``<prefix>.acc_rates.h5``.  The first thing to turn on when "
            "diagnosing a stuck or badly adapted chain.  Independent of "
            "``adaptation.full_auto``."
        ),
    )
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
    save_max_neighbors: bool = Field(
        default=False,
        description=(
            "Write the observed per-iteration maximum neighbour count to "
            "``<prefix>.max_neighbors.h5``, for tuning "
            "``backend.max_neighbors_list``.  No-op for backends without "
            "neighbour lists (all-pairs LJ, toy potentials)."
        ),
    )
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
    save_re_stats: bool = Field(
        default=False,
        description=(
            "Write per-fire inter-replica swap statistics to "
            "``<prefix>.re_stats.h5``.  Cadence follows "
            "``inter_re.re_interval`` — there is deliberately no separate "
            "interval.  No-op when ``inter_re`` is unset."
        ),
    )

    # Finite-difference temperature estimator (Baldock et al. 2017).
    # ``temperature_lag_interval`` is the length of the Emax FIFO used for the
    # finite difference; ``None`` disables the callback entirely.
    # Both ``temperature_lag_interval`` and ``temperature_interval`` honour
    # ``interval_units`` (``per_walker`` scales by ``n_live``).
    # ``temperature_kB`` defaults to eV/K (ASE convention) — set ``1.0``
    # for reduced-unit backends (LJ, harmonic).
    temperature_lag_interval: int | float | None = Field(
        default=100,
        description=(
            "Length of the ``E_max`` FIFO used by the finite-difference "
            "temperature estimator (Baldock et al. 2017).  Longer lags "
            "give a smoother but more delayed estimate.  ``null`` disables "
            "the temperature callback entirely."
        ),
    )
    temperature_interval: int | float = Field(
        default=100,
        description=(
            "Evaluate and log the temperature estimate every N iterations."
        ),
    )
    temperature_kB: float = Field(
        default=8.6173324e-5,
        description=(
            "Boltzmann constant in the backend's energy units.  The "
            "default is eV/K (ASE convention); set ``1.0`` for "
            "reduced-unit backends such as LJ or harmonic, otherwise the "
            "reported temperatures are meaningless."
        ),
    )

    # Pairwise-distance "collision" check.  When ``collision_check_threshold``
    # is set, the runtime registers a ``CollisionCheckCallback`` that every
    # ``collision_check_interval`` iterations computes the minimum
    # interatomic distance per walker (minimum-image when cells are present,
    # all-pairs otherwise) and logs a warning when any walker has atoms
    # closer than the threshold.  Diagnostic only — never feeds back into
    # the run.  ``None`` (default) disables the callback entirely.  Honours
    # ``interval_units`` like the other interval fields.
    collision_check_threshold: float | None = Field(
        default=None,
        description=(
            "Minimum interatomic distance below which ``CollisionCheckCallback``"
            " emits a warning.  Backend length units (typically Å).  ``null``"
            " disables the callback."
        ),
    )
    collision_check_interval: int | float = Field(
        default=100,
        gt=0,
        description=(
            "Fire CollisionCheckCallback every N iterations.  Default 100;"
            " a diagnostic, not a per-iter signal."
        ),
    )

    # Whether trajectory writers wrap atom positions into each frame's own
    # cell before writing.  Default True: atoms drift arbitrarily far from the
    # cell over a run (moves are unwrapped Cartesian), so absolute Cartesians
    # produce trajectories with atoms scattered across many cell lengths.
    # Wrapping is a no-op for non-periodic frames (no cell / det≈0).  Set False
    # to keep absolute Cartesians (e.g. to avoid splitting a molecule that
    # straddles a periodic boundary).
    wrap_atoms: bool = Field(
        default=True,
        description=(
            "Wrap atom positions into each frame's own cell before "
            "writing.  Moves are unwrapped Cartesian, so atoms drift "
            "arbitrarily far over a run and unwrapped output is scattered "
            "across many cell lengths.  No-op for non-periodic frames.  "
            "Set ``false`` to keep absolute Cartesians, e.g. to avoid "
            "splitting a molecule across a boundary."
        ),
    )

    # When True (default), only the most recent walker snapshot
    # (``*.snap.<iter>.extxyz``) is kept: the previous one is deleted as soon
    # as the next is written.  Walker snapshots are a crash-inspection
    # convenience, so this stops the output directory from growing one dump per
    # ``snapshot_interval``.  Set False to retain every snapshot.
    snapshot_clean: bool = Field(
        default=True,
        description=(
            "Keep only the most recent walker snapshot, deleting the "
            "previous one as each new one lands.  Stops the output "
            "directory growing by one dump per ``snapshot_interval``.  Set "
            "``false`` to retain every snapshot."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_temperature_lag(cls, data: Any) -> Any:
        """Map the deprecated ``temperature_lag`` key onto its new name.

        ``temperature_lag`` was renamed to ``temperature_lag_interval`` for
        naming consistency with the other ``*_interval`` fields.  Configs
        written against the old name still load (with a deprecation warning)
        instead of being rejected by ``extra='forbid'``.
        """
        if not isinstance(data, dict) or "temperature_lag" not in data:
            return data
        data = dict(data)
        legacy = data.pop("temperature_lag")
        if "temperature_lag_interval" in data:
            raise ValueError(
                "output: set either 'temperature_lag' (deprecated) or "
                "'temperature_lag_interval', not both."
            )
        logger.warning(
            "output.temperature_lag is deprecated; use temperature_lag_interval. "
            "Mapping the provided value (%r) onto temperature_lag_interval.",
            legacy,
        )
        data["temperature_lag_interval"] = legacy
        return data
