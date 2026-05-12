"""Configuration dataclasses for jaxrens.

Frozen dataclasses replace dict/file-based configuration in the library core.
CLI-level file parsing (reading ns.inp, YAML, etc.) is in cli/parser.py.
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
    # Per-iter chain acceptance logging.  ``True`` registers an
    # ``AccRatesCallback`` writing ``<prefix>.acc_rates.h5`` every
    # ``acc_rates_interval`` iterations.  Decoupled from full_auto.
    save_acc_rates: bool = False
    acc_rates_interval: int = 1


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
        every: Fire a swap pass every this many NS iterations (1 = every iter).
        n_swap_cycles: Number of even+odd swap phases per fire.
        composition_targets: XRENS-only — per-run target compositions,
            shape ``(n_runs, n_species)``.  ``None`` for other flavors.
        chemical_potentials: Semi-grand-only — per-run per-species chemical
            potentials, shape ``(n_runs, n_species)``.  Required when
            ``flavor == "semi_grand"``; ``None`` for other flavors.
    """

    flavor: str = "pressure"
    every: int = 1
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

    # Ensemble
    pressure: float | None = None  # pressure for NPT ensemble (None = NVT)

    # Inter-replica exchange (optional; None → disabled)
    inter_re: InterREConfig | None = None

    # Multi-run / multi-GPU topology (populated by the resolver, not the YAML).
    # Both default to 1; values > 1 indicate that the CLI dispatched through
    # ``run_ns_multi_gpu`` with ``(n_gpu, n_per_gpu)`` replicas.  n_gpu comes
    # from ``len(jax.local_devices())``; n_per_gpu is derived from the
    # replica-axis list length.
    n_gpu: int = 1
    n_per_gpu: int = 1
