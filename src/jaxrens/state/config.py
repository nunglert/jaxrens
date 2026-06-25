"""Configuration dataclasses for jaxrens.

Frozen dataclasses replace dict/file-based configuration in the library core.
CLI-level YAML parsing and validation lives in cli/schema/ and cli/resolve.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    max_neighbors_list: list[int] = field(
        default_factory=lambda: [30, 35, 40, 45, 50]
    )
    max_neighbors_offset: int = 5

    # Hysteresis-gated bucket shrinking (opt-in).  When ``shrink_dwell == 0``
    # (default), the outer NS loop never downsizes the bucket: behaviour is
    # byte-identical to before this feature was added.  When > 0, the bucket
    # is stepped one entry down after ``shrink_dwell`` consecutive iterations
    # where ``observed_max + offset <= next_smaller_entry``.  Temporal
    # hysteresis comes from ``shrink_dwell``; the ``offset`` slack already
    # plays the role of post-shrink safety margin.
    max_neighbors_shrink_dwell: int = 0

    # Soft-core repulsion wrapper kwargs.  ``None`` disables.  When set,
    # the runtime wraps ``base_backend`` with ``SoftCoreBackend`` before
    # any ``EnsembleBackend`` wrap.  Expected keys: ``a0``, ``b0``,
    # ``d0``, ``r_core_cut``, ``r_core_switch``.  See
    # ``jaxrens.backends.softcore`` and the ``SoftCoreSpec`` schema.
    softcore_repulsion: dict | None = None


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

    # Common write-buffer flush cadence for the per-iter trace loggers
    # (``acc_rates``, ``max_neighbors``, ``re_stats``).  A flush fires once
    # the NS iteration index has advanced by ``flush_interval`` since the
    # previous flush.  Scaled by ``RootSpec.interval_units`` in the resolver
    # exactly like every other ``*_interval`` field — set ``per_walker`` in
    # the YAML to specify the value in walker-sweeps instead of raw iters.
    # Decoupled from per-callback firing intervals so tight logging cadences
    # don't force per-iter I/O.  Does NOT affect the adaptation log, which
    # flushes on every event for crash durability.
    flush_interval: int = 1000

    # Per-iter chain acceptance logging.  ``True`` registers an
    # ``AccRatesCallback`` writing ``<prefix>.acc_rates.h5`` every
    # ``acc_rates_interval`` iterations.  Decoupled from full_auto.
    save_acc_rates: bool = False
    acc_rates_interval: int = 1

    # Per-iter neighbor-bucket diagnostic log.  When ``True`` registers a
    # ``MaxNeighborsCallback`` writing ``<prefix>.max_neighbors.h5`` every
    # ``max_neighbors_interval`` iterations.  Captures the full per-walker
    # ``max_neighbor_count`` distribution plus the current bucket and
    # overflow flag — useful for diagnosing bucket oscillations and for
    # tuning ``max_neighbors_list`` / ``shrink_dwell``.  Default-off; no I/O
    # overhead unless enabled.
    save_max_neighbors: bool = False
    max_neighbors_interval: int = 1

    # Per-fire inter-RE swap log; cadence is upstream
    # ``inter_re.re_interval``.  No-op when ``inter_re`` is not configured.
    save_re_stats: bool = False

    # Finite-difference temperature estimator (Baldock et al. 2017).
    # ``temperature_lag_interval`` is the length of the Emax FIFO used for the
    # finite difference; ``None`` disables the callback entirely.
    # Both ``temperature_lag_interval`` and ``temperature_interval`` are scaled by
    # ``RootSpec.interval_units`` (``per_walker`` → multiply by ``n_live``)
    # in the resolver before reaching this runtime dataclass.
    # ``temperature_kB`` defaults to eV/K (ASE convention) — set ``1.0``
    # for reduced-unit backends (LJ, harmonic).
    temperature_lag_interval: int | None = 100
    temperature_interval: int = 100
    temperature_kB: float = 8.6173324e-5

    # Pairwise-distance "collision" check.  When ``collision_check_threshold``
    # is set (non-None and > 0), a ``CollisionCheckCallback`` fires every
    # ``collision_check_interval`` iterations: it computes the minimum
    # interatomic distance per walker (minimum-image convention if cells are
    # present, all-pairs otherwise) and emits a logging warning when any
    # walker has atoms closer than the threshold.  Diagnostic only — never
    # feeds back into the run.  Default-off (no overhead unless enabled).
    collision_check_threshold: float | None = None
    collision_check_interval: int = 100

    # Whether trajectory writers wrap atom positions into each frame's own
    # cell before writing.  Default True because atoms drift far from the cell
    # over a run (moves are unwrapped Cartesian); no-op for non-periodic
    # frames.  Set False to keep absolute Cartesians.
    wrap_atoms: bool = True

    # When True, ``ExtxyzTrajectoryWriter`` deletes the previous walker
    # snapshot right after the next one is written, keeping at most one
    # ``*.snap.<iter>.extxyz`` file on disk.  Walker snapshots are a crash-
    # inspection convenience; cleaning them stops the output directory from
    # growing one dump per ``snapshot_interval``.  No-op for the h5/none
    # writers.
    snapshot_clean: bool = True

    # Post-hoc committee-uncertainty annotation (Phase 4 / active learning).
    # When ``write_uncertainty`` is True and the backend is an NN committee
    # (ensemble), a post-run step annotates the written trajectory with
    # per-frame ``ns_energy_std`` (and, when ``write_force_uncertainty``,
    # per-atom ``ns_force_std``).  No effect on the sampling run itself; a
    # non-committee backend just emits a warning and skips.
    write_uncertainty: bool = False
    write_force_uncertainty: bool = True
    uncertainty_in_place: bool = False


@dataclass(frozen=True)
class InterREConfig:
    """Configuration for inter-replica-exchange (inter-RE) swaps.

    When attached to ``NSConfig.inter_re``, ``run_ns_parallel`` and
    ``run_ns_multi_gpu`` will perform replica-exchange swap passes after each
    ``ns_step`` call.  ``run_ns`` (single run) silently ignores this field.

    Attributes:
        flavor: Swap kernel flavor.  ``"pressure"`` (pressure-RENS, identity
            proposal), ``"xrens"`` (composition-morphing), or
            ``"semi_grand"`` (chemical-potential assignment swap, zero backend
            calls).
        re_interval: Fire a swap pass every this many NS iterations (1 = every iter).
        n_swap_cycles: Number of even+odd swap phases per fire.
        composition_targets: XRENS-only — per-run target compositions,
            shape ``(n_runs, n_species)``.  ``None`` for other flavors.
        chemical_potentials: Semi-grand-only — per-run per-species chemical
            potentials, shape ``(n_runs, n_species)``.  Required when
            ``flavor == "semi_grand"``; ``None`` for other flavors.
    """

    flavor: str = "pressure"
    re_interval: int = 1
    n_swap_cycles: int = 1
    # XRENS-only: per-run target compositions, shape (n_runs, n_species).
    # Required when flavor == "xrens"; each row must sum to n_atoms and
    # the number of rows must equal n_runs.
    composition_targets: tuple[tuple[int, ...], ...] | None = None
    # semi_grand-only: per-run per-species chemical potentials.
    # Shape (n_runs, n_species) encoded as tuple-of-tuple.
    # Required when flavor == "semi_grand".  All rows must have the same
    # length (= n_species) and the row count must equal n_runs.
    chemical_potentials: tuple[tuple[float, ...], ...] | None = None


@dataclass(frozen=True)
class NSConfig:
    """Nested sampling run configuration.

    ``max_iterations`` is optional — when ``None``, the resolver does not add
    an ``IterationTermination`` to the default termination tuple, and the loop
    runs until another criterion (e.g. ``PriorMassTermination``) fires.  No
    static dead-array cap exists: dead-point history is streamed to disk by
    per-iteration callbacks, never held in memory.
    """

    n_live: int = 500
    max_iterations: int | None = None
    convergence_threshold: float = 0.1
    n_mcmc_steps: int = 20
    n_extra: int = 0
    n_cull: int = 1
    seed: int = 42

    # Inter-replica exchange (optional; None → disabled)
    inter_re: InterREConfig | None = None

    # Multi-run / multi-GPU topology (populated by the resolver, not the YAML).
    # Both default to 1; values > 1 indicate that the CLI dispatched through
    # ``run_ns_multi_gpu`` with ``(n_gpu, n_per_gpu)`` replicas.  n_gpu comes
    # from ``len(jax.local_devices())``; n_per_gpu is derived from the
    # replica-axis list length.
    n_gpu: int = 1
    n_per_gpu: int = 1
