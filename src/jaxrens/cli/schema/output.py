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
    # ``temperature_lag_interval`` is the length of the Emax FIFO used for the
    # finite difference; ``None`` disables the callback entirely.
    # Both ``temperature_lag_interval`` and ``temperature_interval`` honour
    # ``interval_units`` (``per_walker`` scales by ``n_live``).
    # ``temperature_kB`` defaults to eV/K (ASE convention) — set ``1.0``
    # for reduced-unit backends (LJ, harmonic).
    temperature_lag_interval: int | float | None = 100
    temperature_interval: int | float = 100
    temperature_kB: float = 8.6173324e-5

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
            " emits a warning.  Backend length units (typically Å).  ``None``"
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

    # Whether ``ExtxyzTrajectoryWriter`` wraps dead-point atom positions into
    # the dead walker's own cell before writing.  Default False keeps absolute
    # Cartesians so off-the-shelf viewers don't show boundary-wrap artifacts.
    wrap_atoms: bool = False

    # When True (default), only the most recent walker snapshot
    # (``*.snap.<iter>.extxyz``) is kept: the previous one is deleted as soon
    # as the next is written.  Walker snapshots are a crash-inspection
    # convenience, so this stops the output directory from growing one dump per
    # ``snapshot_interval``.  Set False to retain every snapshot.
    snapshot_clean: bool = True

    # Post-hoc committee-uncertainty annotation (Phase 4 / active learning).
    # When ``write_uncertainty`` is True and the backend is an NN committee
    # (ensemble), a post-run step annotates the written trajectory with
    # per-frame ``ns_energy_std`` (+ per-atom ``ns_force_std`` when
    # ``write_force_uncertainty``).  No effect on the run itself; a
    # non-committee backend warns and skips.  The annotation can also be run
    # standalone later via the ``annotate-uncertainty`` CLI subcommand.
    write_uncertainty: bool = False
    write_force_uncertainty: bool = True
    uncertainty_in_place: bool = False

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
