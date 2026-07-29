"""Main entry point for running a nested sampling calculation.

Wires together: config loading, backend creation, move kernel construction,
NS loop execution, and I/O callbacks.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op
from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.backends.loader import load_backend
from jaxrens.cli.monitor import (
    AdaptationCallback,
    BatchedTrajectoryCallback,
    CheckpointCallback,
    CollisionCheckCallback,
    EnergyCheckCallback,
    MemProfileCallback,
    ProgressCallback,
    TemperatureCallback,
    TrajectoryCallback,
)
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.trajectory import create_trajectory_writer
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns_multi_gpu,
    init_ns_parallel,
    run_ns,
    run_ns_multi_gpu,
    run_ns_sharded,
)
from jaxrens.state.config import (
    BackendConfig,
    MoveConfig,
    NSConfig,
    OutputConfig,
)

logger = logging.getLogger(__name__)


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _to_runtime_ensemble_params(params: dict | None) -> dict | None:
    """Normalise a config ensemble-params dict for the JIT'd NS step.

    Generic: list/tuple leaves (e.g. ``chemical_potentials``) become float32
    arrays; scalar leaves (e.g. ``pressure``) pass through unchanged — matching
    the per-run dicts the multi-replica resolver builds.  Returns ``None`` for
    an empty/``None`` dict (NVT) so callers can skip the EnsembleBackend wrap.
    """
    if not params:
        return None
    return {
        k: (
            jnp.asarray(v, dtype=jnp.float32)
            if isinstance(v, (list, tuple))
            else v
        )
        for k, v in params.items()
    }


def configure_file_logging(
    *,
    working_dir: Path,
    prefix: str,
    level: str,
    mode: str = "w",
) -> None:
    """Attach a file handler to the ``jaxrens`` logger.

    Writes to a single ``<working_dir>/<prefix>.log``.  The threshold
    follows ``level``: INFO+ normally, DEBUG+ when ``level`` is ``debug``
    (so debug output lands in the same file — there is no separate
    ``.debug.log``).

    ``mode`` mirrors the ``writer_mode`` plumbed through to the I/O
    writers: ``"w"`` on a fresh run, ``"a"`` on restart so the prior
    run's diagnostic log is preserved.

    Should be called once per process, early.  ``cli._cmd_run`` hoists
    this before the resolver so resolver-phase logs (which can take
    many minutes on heavy backends) reach the file from second 1.
    Direct callers of ``run_*_from_config`` (tests, scripts) may call
    this themselves if they want file output.
    """
    root = logging.getLogger("jaxrens")
    root.setLevel(logging.DEBUG if level == "debug" else logging.INFO)

    for h in list(root.handlers):
        if getattr(h, "_jaxrens_managed", False):
            root.removeHandler(h)
            h.close()

    file_h = logging.FileHandler(working_dir / f"{prefix}.log", mode=mode)
    file_h.setLevel(logging.DEBUG if level == "debug" else logging.INFO)
    file_h.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_h._jaxrens_managed = True  # type: ignore[attr-defined]
    root.addHandler(file_h)

    # stream_h = logging.StreamHandler(sys.stderr)
    # stream_h.setLevel(logging.INFO)
    # stream_h.setFormatter(logging.Formatter(_LOG_FORMAT))
    # stream_h._jaxrens_managed = True  # type: ignore[attr-defined]
    # root.addHandler(stream_h)


def _barrier(label: str, *arrays: Any) -> None:
    """Force materialisation on JAX arrays and log timing under ``label``.

    Use as a stage boundary in the multi-GPU dispatcher: any OOM during the
    just-completed stage's compile/execute will surface inside this call,
    not deferred to the next materialisation point.  No-op for non-array
    inputs and ``None``.  Only emits at DEBUG level — zero cost when DEBUG
    logging is not enabled.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    t0 = time.perf_counter()
    for a in arrays:
        if a is None:
            continue
        if hasattr(a, "block_until_ready"):
            a.block_until_ready()
        else:
            jax.block_until_ready(a)
    logger.debug("[barrier] %s (%.2fs)", label, time.perf_counter() - t0)


def _recompute_max_neighbor_counts(
    backend: Any,
    positions: jnp.ndarray,
    cells: jnp.ndarray | None,
) -> jnp.ndarray | None:
    """Refresh per-walker max neighbor counts after burn-in.

    Burn-in drifts positions (and cells in NPT), so the resolver's
    pre-burn-in counts no longer describe the state entering the NS
    loop.  Recompute via ``backend.max_neighbors_for`` so the NS loop
    picks the correct starting bucket.  Returns ``None`` for backends
    without the helper (LJ / toy — they ignore ``max_neighbors`` anyway).

    Handles any leading batch shape by flattening + a sequential
    ``jax.lax.map``.  Sequential (not vmap) bounds peak memory at one
    walker's pairwise tensor — vmapping would allocate
    ``(W, N, sc_dim*N, 3)`` and OOM on MACE-sized systems.
    """
    if cells is None or not hasattr(backend, "max_neighbors_for"):
        return None
    leading_shape = positions.shape[:-2]
    flat_pos = positions.reshape(-1, *positions.shape[-2:])
    flat_cells = cells.reshape(-1, 3, 3)

    def _per_walker(args):
        pos, cell = args
        return backend.max_neighbors_for(pos, cell)

    flat_counts = jax.lax.map(_per_walker, (flat_pos, flat_cells))
    return flat_counts.reshape(leading_shape)


def _move_config_to_descriptor(mc: MoveConfig) -> MoveKernel:
    """Convert a ``MoveConfig`` dataclass to a ``MoveKernel``.

    Delegates to the corresponding ``*MoveSpec`` class so that the spec
    classes remain the single source of truth for kernel kwargs and
    extra state fields.  Specs that require fields absent from
    ``MoveConfig`` (e.g. ``n_atoms`` for volume/shear/stretch) cannot be
    constructed this way; callers that need those moves should build
    descriptors via ``ResolvedConfig.move_descriptors`` instead.
    """
    from jaxrens.cli.schema.moves import (
        AlchemicalShiftMoveSpec,
        GMCMoveSpec,
        HMCMoveSpec,
        RandomWalkMoveSpec,
        SingleAtomMoveSpec,
        SingleAtomSwapMoveSpec,
    )

    _SIMPLE_SPEC_MAP: dict[str, Any] = {
        "random_walk": RandomWalkMoveSpec,
        "galilean": GMCMoveSpec,
        "gmc": GMCMoveSpec,
        "hmc": HMCMoveSpec,
        "single_atom": SingleAtomMoveSpec,
        "single_atom_swap": SingleAtomSwapMoveSpec,
        "alchemical_shift": AlchemicalShiftMoveSpec,
    }

    spec_cls = _SIMPLE_SPEC_MAP.get(mc.move_type)
    if spec_cls is None:
        raise ValueError(
            f"Unknown move type: {mc.move_type!r}. "
            f"Available via MoveConfig: {list(_SIMPLE_SPEC_MAP)}. "
            f"For volume/shear/stretch/single_atom_sweep/alchemical_morph use "
            f"ResolvedConfig.move_descriptors (they require n_atoms/n_species)."
        )

    common = dict(
        step_size=mc.step_size,
        weight=mc.weight,
        adaptation_warmup=mc.adaptation_warmup,
        target_acceptance=mc.target_acceptance,
    )
    if mc.move_type in ("galilean", "gmc"):
        spec = spec_cls(n_reflect=mc.n_steps, **common)
    elif mc.move_type == "hmc":
        spec = spec_cls(n_leapfrog=mc.n_steps, **common)
    else:
        spec = spec_cls(**common)

    return spec.to_descriptor()


def setup_mwg(
    move_configs: list[MoveConfig] | MoveConfig,
    backend: Any,
):
    """Create MWG init_fn and step_fn from move config(s).

    Args:
        move_configs: Single ``MoveConfig`` or list of ``MoveConfig`` objects.
            For move types that require ``n_atoms`` or ``n_species``
            (volume, shear, stretch, single_atom_sweep, alchemical_morph)
            use ``build_mwg`` directly with pre-built ``MoveKernel`` instances.
        backend: EnergyBackend instance.

    Returns:
        (init_fn, step_fn, per_move_fns) from build_mwg.
    """
    if isinstance(move_configs, MoveConfig):
        move_configs = [move_configs]

    descriptors = [_move_config_to_descriptor(mc) for mc in move_configs]
    return build_mwg(backend, descriptors)


def run_from_config(
    ns_config: NSConfig,
    move_config: MoveConfig | list[MoveConfig],
    backend_config: BackendConfig,
    output_config: OutputConfig,
    initial_positions: jnp.ndarray,
    initial_types: jnp.ndarray,
    initial_energies: jnp.ndarray | None = None,
    initial_cells: jnp.ndarray | None = None,
    initial_max_neighbor_counts: jnp.ndarray | None = None,
    symbol_map: dict[int, str] | None = None,
    termination_criteria: list | None = None,
    restart_state=None,
    initial_walk_config=None,
    adaptation_config=None,
    move_descriptors=None,
    constraint_descriptors: tuple = (),
    base_backend: Any = None,
    writer_mode: str = "w",
    ensemble_params: dict | None = None,
) -> dict:
    """Run NS from typed config objects.

    ``base_backend`` (preferred): a pre-built ``EnergyBackend``, e.g. the one
    populated on ``ResolvedConfig.energy_backend`` by the resolver via
    ``BackendSpec.build_backend()``.  When provided, it's used directly —
    avoids rebuilding via ``load_backend(backend_type, **backend_kwargs)`` from
    the flat ``BackendConfig``, whose ``checkpoint_path -> pickle_file``
    translation does not match every backend's ``create_*`` kwargs (MACE
    expects ``model_path``, not ``pickle_file``).  The fallback path is kept
    for legacy callers that only have a ``BackendConfig``.
    """
    if symbol_map is None:
        symbol_map = {0: "X"}

    if base_backend is None:
        # Legacy path: rebuild from the flat BackendConfig.  Note: the
        # checkpoint_path -> pickle_file mapping below is correct for NeuralIL
        # but wrong for MACE (which expects model_path).  Prefer passing
        # ``base_backend`` directly from the resolver.
        backend_kwargs = {}
        if backend_config.checkpoint_path:
            backend_kwargs["pickle_file"] = backend_config.checkpoint_path
        if backend_config.cutoff is not None:
            backend_kwargs["cutoff"] = backend_config.cutoff

        base_backend = load_backend(
            backend_config.backend_type, **backend_kwargs
        )

    # Soft-core wrapper applied first (closest to the bare backend) so
    # the EnsembleBackend PV correction sits outside it.
    if backend_config.softcore_repulsion is not None:
        from jaxrens.backends.softcore import SoftCoreBackend

        base_backend = SoftCoreBackend(
            base_backend,
            **backend_config.softcore_repulsion,
        )

    # Ensemble corrections (P·V, -μ·N, ...) are driven entirely by the generic
    # per-call ``ensemble_params`` dict — same contract the multi-replica runner
    # uses.  Wrap with neutral defaults when any are configured; the dict's
    # array-valued leaves (e.g. chemical_potentials) are normalised for the
    # JIT'd step.  An empty/None dict means NVT → no wrap, zero overhead.
    ensemble_params = _to_runtime_ensemble_params(ensemble_params)
    if ensemble_params is not None:
        backend = EnsembleBackend(base_backend, pressure=0.0)
    else:
        backend = base_backend

    if initial_energies is None:
        logger.debug(
            "initial_energies missing from ResolvedInit; computing in run_from_config. "
            "Prefer populating ResolvedInit.initial_energies via the resolver."
        )

        def eval_one(pos):
            return backend(
                pos,
                initial_types,
                initial_cells[0]
                if initial_cells is not None
                else jnp.zeros((3, 3)),
                0,
            ).energy

        initial_energies = jax.vmap(eval_one)(initial_positions)

    # Build MWG sampler. Prefer pre-built MoveKernels when available — they
    # carry n_atoms / n_species that MoveConfig can't express (volume, shear,
    # stretch, single_atom_sweep, alchemical_morph).
    if move_descriptors is not None:
        init_fn, step_fn, per_move_fns = build_mwg(
            backend, list(move_descriptors), tuple(constraint_descriptors)
        )
    else:
        init_fn, step_fn, per_move_fns = setup_mwg(move_config, backend)

    working_dir = output_config.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    # File logging is configured once, early, by ``cli._cmd_run`` so the
    # resolver phase reaches the log file.  Direct callers (tests,
    # scripts) that want a log file should call ``configure_file_logging``
    # before invoking this function.

    # Set up callbacks
    callbacks = [
        ProgressCallback(info_interval=output_config.info_interval),
        EnergyCheckCallback(),
    ]

    if output_config.temperature_lag_interval is not None:
        callbacks.append(
            TemperatureCallback(
                n_live=ns_config.n_live,
                n_cull=ns_config.n_cull,
                lag=output_config.temperature_lag_interval,
                interval=output_config.temperature_interval,
                kB=output_config.temperature_kB,
            )
        )

    if output_config.collision_check_threshold is not None:
        callbacks.append(
            CollisionCheckCallback(
                threshold=output_config.collision_check_threshold,
                interval=output_config.collision_check_interval,
            )
        )

    callbacks.append(
        CheckpointCallback(
            working_dir=working_dir,
            interval=output_config.checkpoint_interval,
            prefix=output_config.out_file_prefix,
            symbol_map=symbol_map,
        )
    )

    if memprof := os.environ.get("JAXRENS_MEMPROF"):
        callbacks.append(MemProfileCallback(working_dir / memprof))

    # On restart, the global iteration the checkpoint was taken at — every
    # output stream is rewound to drop records the previous process flushed
    # past it (see io/restart_truncate.py).  0 for a fresh run (no-op).
    restart_iteration = (
        int(restart_state.iteration) if restart_state is not None else 0
    )

    traj_path = (
        working_dir
        / f"{output_config.out_file_prefix}.traj.{output_config.format}"
    )
    writer = create_trajectory_writer(
        output_config.format,
        traj_path,
        symbol_map,
        mode=writer_mode,
        restart_iteration=restart_iteration,
        wrap=output_config.wrap_atoms,
        clean_snapshots=output_config.snapshot_clean,
    )
    energy_logger = EnergyLogger(
        working_dir / f"{output_config.out_file_prefix}.energies",
        n_walkers=ns_config.n_live,
        n_atoms=initial_positions.shape[-2],
        mode=writer_mode,
        restart_iteration=restart_iteration,
    )
    callbacks.append(
        TrajectoryCallback(
            writer=writer,
            energy_logger=energy_logger,
            traj_interval=output_config.traj_interval,
            snapshot_interval=output_config.snapshot_interval,
        )
    )

    # Wire the adaptation logger unconditionally when move descriptors
    # exist.  The callback's ``on_start`` writes a mandatory iter-0
    # baseline row carrying the initial step sizes, so a non-full_auto
    # run still produces a useful 1-row artefact documenting the
    # constant ss; full_auto runs append on every bisection event.
    if move_descriptors is not None:
        from jaxrens.io.adaptation_log import AdaptationLogger

        adapt_log_path = (
            working_dir / f"{output_config.out_file_prefix}.adaptation.h5"
        )
        move_name_list = [d.name for d in move_descriptors]
        adaptation_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=move_name_list,
            n_runs=1,
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(AdaptationCallback(adaptation_logger))

    # Wire per-iter chain-acceptance logger when requested (independent
    # of full_auto — the chain counters are populated by ns_step every
    # iter regardless of adaptation).
    if move_descriptors is not None and getattr(
        output_config, "save_acc_rates", False
    ):
        from jaxrens.cli.monitor import AccRatesCallback
        from jaxrens.io.acc_rates_log import AccRatesLogger

        acc_log_path = (
            working_dir / f"{output_config.out_file_prefix}.acc_rates.h5"
        )
        acc_logger = AccRatesLogger(
            path=acc_log_path,
            move_names=[d.name for d in move_descriptors],
            n_runs=1,
            flush_interval=int(getattr(output_config, "flush_interval", 1000)),
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(
            AccRatesCallback(
                acc_logger,
                interval=int(getattr(output_config, "acc_rates_interval", 1)),
            )
        )

    if getattr(output_config, "save_max_neighbors", False):
        from jaxrens.cli.monitor import MaxNeighborsCallback
        from jaxrens.io.max_neighbors_log import MaxNeighborsLogger

        mn_log_path = (
            working_dir / f"{output_config.out_file_prefix}.max_neighbors.h5"
        )
        mn_logger = MaxNeighborsLogger(
            path=mn_log_path,
            n_runs=1,
            n_walkers=ns_config.n_live,
            flush_interval=int(getattr(output_config, "flush_interval", 1000)),
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(
            MaxNeighborsCallback(
                mn_logger,
                interval=int(
                    getattr(output_config, "max_neighbors_interval", 1)
                ),
            )
        )

    first_mc = move_config[0] if isinstance(move_config, list) else move_config

    key = jax.random.key(ns_config.seed)

    # --- Burn-in: fixed-Emax relaxation before NS proper ---
    # Skip for Mode D (restart) — the checkpoint is already at the right level.
    burn_in_cfg = initial_walk_config
    do_burn_in = (
        burn_in_cfg is not None
        and burn_in_cfg.n_walks > 0
        and restart_state is None
    )
    if do_burn_in:
        from jaxrens.init.burn_in import initial_walk
        from jaxrens.sampling.nested_sampling import (
            _choose_starting_bucket,
            init_ns,
        )

        logger.info(
            "Running initial-walk burn-in: %d walks x %d steps/walk",
            burn_in_cfg.n_walks,
            burn_in_cfg.walklength,
        )

        n_atoms = (
            initial_positions.shape[1] if initial_positions.ndim >= 2 else 1
        )

        _ladder = tuple(int(x) for x in backend_config.max_neighbors_list)
        _offset = int(backend_config.max_neighbors_offset)
        starting_bucket = _choose_starting_bucket(
            initial_max_neighbor_counts,
            _ladder,
            _offset,
        )

        key, key_init, key_burn = jax.random.split(key, 3)
        ns_state_burn = init_ns(
            init_fn,
            initial_positions,
            initial_types,
            initial_energies,
            initial_cells,
            key_init,
            ensemble_params=ensemble_params,
            max_neighbors=starting_bucket,
            max_neighbor_counts=initial_max_neighbor_counts,
        )

        # Build per-move adaptation data if available.
        burn_per_move_fns = None
        burn_adaptation_policies = None
        if adaptation_config is not None and move_descriptors is not None:
            burn_per_move_fns = per_move_fns
            burn_adaptation_policies = tuple(
                adaptation_config.resolve_for(d.name) for d in move_descriptors
            )

        ns_state_burn = initial_walk(
            key=key_burn,
            ns_state=ns_state_burn,
            step_fn=step_fn,
            n_walks=burn_in_cfg.n_walks,
            walklength=burn_in_cfg.walklength,
            adjust_interval=burn_in_cfg.adjust_interval,
            emax_offset_per_atom=burn_in_cfg.emax_offset_per_atom,
            n_atoms=n_atoms,
            walker_batch_size=getattr(burn_in_cfg, "walker_batch_size", None),
            per_move_fns=burn_per_move_fns,
            adaptation_policies=burn_adaptation_policies,
            move_names=(
                tuple(d.name for d in move_descriptors)
                if move_descriptors is not None
                else None
            ),
            adjust_n_samples=getattr(adaptation_config, "adjust_n_samples", 50)
            if adaptation_config is not None
            else 50,
            adjust_max_rounds=getattr(
                adaptation_config, "adjust_max_rounds", 15
            )
            if adaptation_config is not None
            else 15,
            max_neighbors_list=tuple(backend_config.max_neighbors_list),
            max_neighbors_offset=backend_config.max_neighbors_offset,
            max_neighbors_shrink_dwell=backend_config.max_neighbors_shrink_dwell,
        )

        # Extract burned-in walker arrays to re-seed run_ns.
        pop = ns_state_burn.population
        initial_positions = pop.positions
        initial_energies = pop.energy
        initial_cells = pop.cell
        logger.info("Burn-in complete")

        # Refresh counts for the NS loop — burn-in drifted positions/cells.
        if hasattr(backend, "max_neighbors_for"):
            logger.info(
                "Recomputing post-burn-in max neighbor counts (n_walkers=%d)",
                initial_positions.shape[0],
            )
        initial_max_neighbor_counts = _recompute_max_neighbor_counts(
            backend,
            initial_positions,
            initial_cells,
        )

        if burn_in_cfg.write_initial_walkers:
            logger.warning(
                "initial_walk.write_initial_walkers=True is set but not yet "
                "consumed by the runtime — field is a deferred placeholder."
            )

        if getattr(burn_in_cfg, "only", None):
            raise NotImplementedError(
                "initial_walk.only=True is not yet implemented. "
                "The run-ns-skip path is deferred."
            )

    full_auto_kwargs: dict[str, Any] = {}
    if adaptation_config is not None and adaptation_config.full_auto:
        adjust_factor = (
            adaptation_config.defaults.adjust_factor
            if adaptation_config.defaults.adjust_factor is not None
            else 1.5
        )
        full_auto_kwargs = dict(
            per_move_fns=per_move_fns,
            move_descriptors=list(move_descriptors)
            if move_descriptors is not None
            else None,
            adjust_interval=adaptation_config.adjust_interval,
            adjust_n_samples=adaptation_config.adjust_n_samples,
            adjust_max_rounds=adaptation_config.adjust_max_rounds,
            adjust_factor=adjust_factor,
            trial_batch_size=adaptation_config.trial_batch_size,
        )

    result = run_ns(
        positions=initial_positions,
        types=initial_types,
        energies=initial_energies,
        cells=initial_cells,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_key=key,
        max_iterations=ns_config.max_iterations,
        n_mcmc_steps=ns_config.n_mcmc_steps,
        n_extra=ns_config.n_extra,
        convergence_threshold=ns_config.convergence_threshold,
        initial_step_size=first_mc.step_size,
        target_acceptance=first_mc.target_acceptance,
        callbacks=callbacks,
        ensemble_params=ensemble_params,
        termination_criteria=termination_criteria,
        restart_state=restart_state,
        max_neighbors_list=tuple(backend_config.max_neighbors_list),
        max_neighbors_offset=backend_config.max_neighbors_offset,
        max_neighbors_shrink_dwell=backend_config.max_neighbors_shrink_dwell,
        initial_max_neighbor_counts=initial_max_neighbor_counts,
        **full_auto_kwargs,
    )

    return result


def _restart_iteration_from(restart_state: Any) -> int:
    """Global iteration to rewind output streams to on restart.

    Accepts the single-run ``RestartBundle``, the multi-GPU 2-D nested
    bundle list, or a flat list — every replica shares one checkpoint, so
    the first non-``None`` bundle's ``iteration`` is authoritative.  Returns
    0 for a fresh run (``restart_state is None``), making truncation a no-op.
    """
    if restart_state is None:
        return 0
    stack = [restart_state]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, list):
            stack.extend(item)
        elif hasattr(item, "iteration"):
            return int(item.iteration)
    return 0


# ---------------------------------------------------------------------------
# Multi-run (multi-GPU) dispatch
# ---------------------------------------------------------------------------


def run_multi_gpu_from_config(resolved, *, writer_mode: str = "w") -> dict:
    """Execute a multi-replica NS dispatch from a multi-replica ``ResolvedConfig``.

    Mirrors :func:`run_from_config` but ends in ``run_ns_multi_gpu`` with
    per-replica ``ensemble_params_per_run``.  Wires all five callbacks with
    batched-aware variants where needed:

    * ``ProgressCallback``, ``AdaptationCallback``, ``EnergyCheckCallback`` —
      already (G, P)-safe.
    * ``CheckpointCallback`` — saves HDF5 with batched shapes via the
      already-batched-safe ``io/checkpoint.py`` path.
    * ``BatchedTrajectoryCallback`` — one writer + energy logger per replica,
      file suffix ``.run{r:02d}``.

    Burn-in runs via ``initial_walk(batched=True)`` on the stacked
    ``(n_total, K, ...)``-shaped NSState, before entering
    ``run_ns_multi_gpu``.
    """
    from jaxrens.io.adaptation_log import AdaptationLogger
    from jaxrens.sampling.termination import (
        IterationTermination,
        PriorMassTermination,
    )
    from jaxrens.state.ns import NSState as _NSState  # for isinstance

    ns = resolved.ns
    n_gpu = ns.n_gpu
    n_per_gpu = ns.n_per_gpu
    n_total = n_gpu * n_per_gpu

    # --- Backend: one base, wrapped once with EnsembleBackend --------------
    # Per-call ``ensemble_params`` dicts override the wrapper's closured
    # defaults (see backends/ensemble.py __call__).  The resolver already
    # applied the soft-core wrapper (if configured) to ``base_backend``
    # before stashing it on ``resolved`` (see ``_resolve_multi_replica``),
    # so no additional wrap is needed here.
    base_backend = resolved.base_backend
    backend = EnsembleBackend(base_backend, pressure=0.0)

    init_fn, step_fn, per_move_fns = build_mwg(
        backend,
        list(resolved.move_descriptors),
        resolved.constraint_descriptors,
    )

    working_dir = resolved.output.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    # File logging is configured once, early, by ``cli._cmd_run`` so the
    # resolver phase reaches the log file.  Direct callers (tests,
    # scripts) that want a log file should call ``configure_file_logging``
    # before invoking this function.

    n_live = ns.n_live
    positions = resolved.init.initial_positions  # (n_total, K, A, 3)
    types = resolved.init.initial_types  # (A,)
    cells = resolved.init.initial_cells  # (n_total, K, 3, 3) | None
    energies = resolved.init.initial_energies  # (n_total, K)

    if positions.shape[0] != n_total:
        raise RuntimeError(
            f"run_multi_gpu_from_config: init positions axis 0 is "
            f"{positions.shape[0]} but n_total={n_total}."
        )

    key = jax.random.key(ns.seed)

    # Starting bucket for burn-in and NS — derived from the per-walker
    # neighbor counts captured in the resolver.  Without this, burn-in
    # would run with max_neighbors=0 (static field default), causing
    # MACE to evaluate on an edge-less graph and overwrite the correct
    # post-resolver energies with garbage.
    from jaxrens.sampling.nested_sampling import _choose_starting_bucket

    _ladder = tuple(int(x) for x in resolved.backend.max_neighbors_list)
    _offset = int(resolved.backend.max_neighbors_offset)
    _init_counts = resolved.init.initial_max_neighbor_counts
    starting_bucket = _choose_starting_bucket(_init_counts, _ladder, _offset)
    logger.debug(
        "[stage] resolver -> dispatcher: positions=%s cells=%s energies=%s "
        "starting_bucket=%d ladder=%s offset=%d",
        positions.shape,
        None if cells is None else cells.shape,
        energies.shape,
        starting_bucket,
        _ladder,
        _offset,
    )

    # --- Burn-in (batched=True) -------------------------------------------
    # Skip burn-in on the restart branch: the checkpoint already holds
    # walkers from the prior run's NS loop (well past the burn-in level).
    burn_in_cfg = resolved.initial_walk_config
    do_burn_in = (
        burn_in_cfg is not None
        and burn_in_cfg.n_walks > 0
        and resolved.init.restart_state is None
    )
    if do_burn_in:
        from jaxrens.init.burn_in import initial_walk

        key, key_init, key_burn = jax.random.split(key, 3)
        rng_keys = jax.random.split(key_init, n_total)
        step_sizes = jnp.full(
            len(resolved.move_descriptors), resolved.moves[0].step_size
        )

        logger.debug("[stage] init_ns_multi_gpu (burn-in NSState): starting")
        ns_state_burn = init_ns_multi_gpu(
            init_fn,
            positions,
            types,
            energies,
            cells,
            rng_keys,
            n_gpu,
            n_per_gpu,
            step_sizes=step_sizes,
            ensemble_params_per_run=list(resolved.ensemble_params_per_run),
            max_neighbors=starting_bucket,
            max_neighbor_counts=_init_counts,
        )
        _barrier("init_ns_multi_gpu", ns_state_burn.population.positions)

        n_atoms = positions.shape[-2]
        adaptation_policies = resolved.adaptation_policies
        burn_per_move_fns = per_move_fns

        logger.info(
            "Starting initial burn-in: n_walks=%d, walklength=%d, "
            "adjust_interval=%d, walker_batch_size=%s, n_atoms=%d",
            burn_in_cfg.n_walks,
            burn_in_cfg.walklength,
            burn_in_cfg.adjust_interval,
            getattr(burn_in_cfg, "walker_batch_size", None),
            n_atoms,
        )

        ns_state_burn = initial_walk(
            key=key_burn,
            ns_state=ns_state_burn,
            step_fn=step_fn,
            n_walks=burn_in_cfg.n_walks,
            walklength=burn_in_cfg.walklength,
            adjust_interval=burn_in_cfg.adjust_interval,
            emax_offset_per_atom=burn_in_cfg.emax_offset_per_atom,
            n_atoms=n_atoms,
            batcher=resolved.batcher,
            walker_batch_size=getattr(burn_in_cfg, "walker_batch_size", None),
            per_move_fns=burn_per_move_fns,
            adaptation_policies=adaptation_policies,
            move_names=tuple(d.name for d in resolved.move_descriptors),
            adjust_n_samples=getattr(
                resolved.adaptation_cfg, "adjust_n_samples", 50
            ),
            adjust_max_rounds=getattr(
                resolved.adaptation_cfg, "adjust_max_rounds", 15
            ),
            max_neighbors_list=tuple(resolved.backend.max_neighbors_list),
            max_neighbors_offset=resolved.backend.max_neighbors_offset,
            max_neighbors_shrink_dwell=resolved.backend.max_neighbors_shrink_dwell,
        )
        pop = ns_state_burn.population

        # Burn-in operated on (G, P, K, ...) state via the PmapVmapRuns batcher.
        # Downstream code (``_recompute_max_neighbor_counts``, ``run_ns_multi_gpu``
        # input adapters) consumes the flat ``(n_total, K, ...)`` form, so flatten
        # the leading G×P axes back here.
        def _flatten_GP(arr):
            if arr is None:
                return None
            return arr.reshape((n_total,) + arr.shape[2:])

        positions = _flatten_GP(pop.positions)
        energies = _flatten_GP(pop.energy)
        cells = _flatten_GP(pop.cell)
        _barrier("initial_walk", positions, energies, cells)
        logger.info("Burn-in complete")

    # Refresh per-walker neighbor counts after burn-in — cells/positions
    # have drifted and the resolver's pre-burn-in counts no longer
    # describe the state that will enter the NS loop.
    if do_burn_in:
        logger.info(
            "Recomputing post-burn-in max neighbor counts (n_walkers_total=%d)",
            positions.reshape(-1, *positions.shape[-2:]).shape[0],
        )
    post_burn_in_counts = (
        _recompute_max_neighbor_counts(
            base_backend,
            positions,
            cells,
        )
        if do_burn_in
        else _init_counts
    )
    _barrier("_recompute_max_neighbor_counts", post_burn_in_counts)

    # --- PRNG keys for the multi-GPU dispatch -----------------------------
    key, key_run = jax.random.split(key)
    rng_keys_flat = jax.random.split(key_run, n_total)

    # --- Callbacks ---------------------------------------------------------
    callbacks: list[Any] = [
        ProgressCallback(info_interval=resolved.output.info_interval),
        EnergyCheckCallback(),
    ]

    if resolved.output.temperature_lag_interval is not None:
        callbacks.append(
            TemperatureCallback(
                n_live=ns.n_live,
                n_cull=ns.n_cull,
                lag=resolved.output.temperature_lag_interval,
                interval=resolved.output.temperature_interval,
                kB=resolved.output.temperature_kB,
            )
        )

    if resolved.output.collision_check_threshold is not None:
        callbacks.append(
            CollisionCheckCallback(
                threshold=resolved.output.collision_check_threshold,
                interval=resolved.output.collision_check_interval,
            )
        )

    symbol_map = resolved.init.symbol_map
    callbacks.append(
        CheckpointCallback(
            working_dir=working_dir,
            interval=resolved.output.checkpoint_interval,
            prefix=resolved.output.out_file_prefix,
            symbol_map=symbol_map,
        )
    )

    if memprof := os.environ.get("JAXRENS_MEMPROF"):
        callbacks.append(MemProfileCallback(working_dir / memprof))

    # Rewind every output stream to the checkpoint on restart (0 = fresh).
    restart_iteration = _restart_iteration_from(resolved.init.restart_state)

    n_atoms = positions.shape[-2]
    writers = []
    energy_loggers = []
    for r in range(n_total):
        traj_path = (
            working_dir
            / f"{resolved.output.out_file_prefix}.run{r:02d}.traj.{resolved.output.format}"
        )
        writers.append(
            create_trajectory_writer(
                resolved.output.format,
                traj_path,
                symbol_map,
                mode=writer_mode,
                restart_iteration=restart_iteration,
                wrap=resolved.output.wrap_atoms,
                clean_snapshots=resolved.output.snapshot_clean,
            )
        )
        energy_path = (
            working_dir
            / f"{resolved.output.out_file_prefix}.run{r:02d}.energies"
        )
        energy_loggers.append(
            EnergyLogger(
                energy_path,
                n_walkers=n_live,
                n_atoms=n_atoms,
                mode=writer_mode,
                restart_iteration=restart_iteration,
            )
        )
    callbacks.append(
        BatchedTrajectoryCallback(
            writers=writers,
            energy_loggers=energy_loggers,
            traj_interval=resolved.output.traj_interval,
            snapshot_interval=resolved.output.snapshot_interval,
        )
    )

    # Register unconditionally — the iter-0 baseline row makes the
    # adaptation file useful even when full_auto is off (single-row
    # artefact documenting the constant step size).  See
    # ``run_from_config`` for the matching wire-up on the single-run path.
    if resolved.move_descriptors:
        adapt_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.adaptation.h5"
        )
        move_name_list = [d.name for d in resolved.move_descriptors]
        adaptation_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=move_name_list,
            n_runs=n_total,
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(AdaptationCallback(adaptation_logger))

    # Per-iter chain-acceptance logger (multi-run path).
    if resolved.move_descriptors and getattr(
        resolved.output,
        "save_acc_rates",
        False,
    ):
        from jaxrens.cli.monitor import AccRatesCallback
        from jaxrens.io.acc_rates_log import AccRatesLogger

        acc_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.acc_rates.h5"
        )
        acc_logger = AccRatesLogger(
            path=acc_log_path,
            move_names=[d.name for d in resolved.move_descriptors],
            n_runs=n_total,
            flush_interval=int(
                getattr(resolved.output, "flush_interval", 1000)
            ),
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(
            AccRatesCallback(
                acc_logger,
                interval=int(
                    getattr(resolved.output, "acc_rates_interval", 1)
                ),
            )
        )

    if getattr(resolved.output, "save_max_neighbors", False):
        from jaxrens.cli.monitor import MaxNeighborsCallback
        from jaxrens.io.max_neighbors_log import MaxNeighborsLogger

        mn_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.max_neighbors.h5"
        )
        mn_logger = MaxNeighborsLogger(
            path=mn_log_path,
            n_runs=n_total,
            n_walkers=n_live,
            flush_interval=int(
                getattr(resolved.output, "flush_interval", 1000)
            ),
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(
            MaxNeighborsCallback(
                mn_logger,
                interval=int(
                    getattr(resolved.output, "max_neighbors_interval", 1)
                ),
            )
        )

    # Per-fire inter-RE swap counts (multi-run + inter_re only).
    if resolved.inter_re_config is not None and getattr(
        resolved.output, "save_re_stats", False
    ):
        from jaxrens.cli.monitor import RECallback
        from jaxrens.io.re_stats_log import RELogger

        re_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.re_stats.h5"
        )
        re_logger = RELogger(
            path=re_log_path,
            n_pairs=max(n_total - 1, 0),
            flavor=resolved.inter_re_config.flavor,
            flush_interval=int(
                getattr(resolved.output, "flush_interval", 1000)
            ),
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(RECallback(re_logger))

    first_mc = resolved.moves[0]
    full_auto_kwargs: dict[str, Any] = {}
    if (
        resolved.adaptation_cfg is not None
        and resolved.adaptation_cfg.full_auto
    ):
        adjust_factor = (
            resolved.adaptation_cfg.defaults.adjust_factor
            if resolved.adaptation_cfg.defaults.adjust_factor is not None
            else 1.5
        )
        full_auto_kwargs = dict(
            per_move_fns=per_move_fns,
            adjust_interval=resolved.adaptation_cfg.adjust_interval,
            adjust_n_samples=resolved.adaptation_cfg.adjust_n_samples,
            adjust_max_rounds=resolved.adaptation_cfg.adjust_max_rounds,
            adjust_factor=adjust_factor,
            trial_batch_size=resolved.adaptation_cfg.trial_batch_size,
        )

    logger.info(
        "Starting multi-GPU NS: n_gpu=%d, n_per_gpu=%d (n_total=%d), "
        "n_walkers=%d, n_mcmc=%d, max_iter=%s",
        n_gpu,
        n_per_gpu,
        n_total,
        n_live,
        ns.n_mcmc_steps,
        ns.max_iterations,
    )

    # Multi-replica restart: the resolver attached ``list[list[RestartBundle]]``
    # of shape (n_gpu, n_per_gpu) when ``init.restart_file`` was set; pass it
    # through so ``run_ns_multi_gpu`` seeds per-replica NSState from the
    # checkpoint instead of from a fresh ``init_ns``.
    restart_states_2d = resolved.init.restart_state
    if restart_states_2d is not None and not (
        isinstance(restart_states_2d, list)
        and len(restart_states_2d) > 0
        and isinstance(restart_states_2d[0], list)
    ):
        raise RuntimeError(
            f"run_multi_gpu_from_config: expected resolved.init.restart_state "
            f"to be list[list[RestartBundle]] for the multi-replica path, "
            f"got {type(restart_states_2d).__name__}."
        )

    result = run_ns_multi_gpu(
        positions=positions,
        types=types,
        energies=energies,
        cells=cells,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys_flat,
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
        n_walkers=n_live,
        max_iterations=ns.max_iterations,
        n_mcmc_steps=ns.n_mcmc_steps,
        n_extra=ns.n_extra,
        convergence_threshold=ns.convergence_threshold,
        initial_step_size=first_mc.step_size,
        target_acceptance=first_mc.target_acceptance,
        callbacks=callbacks,
        termination_criteria=list(resolved.termination),
        ensemble_params_per_run=list(resolved.ensemble_params_per_run),
        move_descriptors=list(resolved.move_descriptors),
        inter_re_config=resolved.inter_re_config,
        backend=base_backend,
        max_neighbors_list=tuple(resolved.backend.max_neighbors_list),
        max_neighbors_offset=resolved.backend.max_neighbors_offset,
        max_neighbors_shrink_dwell=resolved.backend.max_neighbors_shrink_dwell,
        initial_max_neighbor_counts=post_burn_in_counts,
        batcher=resolved.batcher,
        restart_states=restart_states_2d,
        **full_auto_kwargs,
    )

    return result


def run_sharded_from_config(resolved, *, writer_mode: str = "w") -> dict:
    """Execute a sharded-single NS dispatch from a sharded ``ResolvedConfig``.

    One logical NS run with its ``n_live`` walker population sharded
    across ``shard_n_gpu`` GPUs.  Sibling of :func:`run_from_config`
    (one population on one device) and :func:`run_multi_gpu_from_config`
    (G*P independent populations).

    The resolver populates ``resolved.batcher`` with a
    :class:`ShardedSingleRun(n_gpu=...)` and keeps the init arrays in
    the flat ``(K, ...)`` layout — ``init_ns_sharded`` (inside
    ``run_ns_sharded``) does the reshape and per-device placement.

    Burn-in (`initial_walk`) runs through ``ShardedSingleRun`` natively
    — adaptation goes through :func:`build_adapt_step` (which
    dispatches to ``adjust_step_size_sharded``) and the walking step
    pmaps with per-shard distinct keys.  After burn-in the sharded
    NSState's ``(G, K/G, ...)`` population is flattened back to
    ``(K, ...)`` for the post-burn-in neighbor-count refresh and for
    re-init by ``run_ns_sharded`` (which re-shards via
    ``init_ns_sharded``).
    """
    from jaxrens.io.adaptation_log import AdaptationLogger
    from jaxrens.sampling.nested_sampling import (
        _choose_starting_bucket,
        init_ns_sharded,
    )

    ns = resolved.ns
    batcher = resolved.batcher  # ShardedSingleRun
    n_gpu = batcher.n_gpu
    n_live = ns.n_live

    # The resolver already applied the soft-core wrapper (if configured)
    # to ``base_backend`` before stashing it on ``resolved``, so no
    # additional wrap is needed here.
    base_backend = resolved.base_backend
    backend = EnsembleBackend(base_backend, pressure=0.0)
    init_fn, step_fn, per_move_fns = build_mwg(
        backend,
        list(resolved.move_descriptors),
        resolved.constraint_descriptors,
    )

    working_dir = resolved.output.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    positions = resolved.init.initial_positions  # (K, A, 3)
    types = resolved.init.initial_types
    cells = resolved.init.initial_cells
    energies = resolved.init.initial_energies

    if positions.shape[0] != n_live:
        raise RuntimeError(
            f"run_sharded_from_config: init positions axis 0 is "
            f"{positions.shape[0]} but n_live={n_live}."
        )

    key = jax.random.key(ns.seed)

    # --- Burn-in (sharded) -------------------------------------------------
    burn_in_cfg = resolved.initial_walk_config
    do_burn_in = (
        burn_in_cfg is not None
        and burn_in_cfg.n_walks > 0
        and resolved.init.restart_state is None
    )
    initial_max_neighbor_counts = resolved.init.initial_max_neighbor_counts
    if do_burn_in:
        from jaxrens.init.burn_in import initial_walk

        _ladder = tuple(int(x) for x in resolved.backend.max_neighbors_list)
        _offset = int(resolved.backend.max_neighbors_offset)
        starting_bucket = _choose_starting_bucket(
            initial_max_neighbor_counts,
            _ladder,
            _offset,
        )

        key, key_init, key_burn = jax.random.split(key, 3)
        step_sizes = jnp.full(
            len(resolved.move_descriptors),
            resolved.moves[0].step_size,
        )
        ns_state_burn = init_ns_sharded(
            init_fn,
            positions,
            types,
            energies,
            cells,
            key_init,
            n_gpu=n_gpu,
            step_sizes=step_sizes,
            ensemble_params=(
                resolved.ensemble_params_per_run[0]
                if resolved.ensemble_params_per_run
                else None
            ),
            max_neighbors=starting_bucket,
            max_neighbor_counts=initial_max_neighbor_counts,
        )

        n_atoms_burn = positions.shape[-2]
        adaptation_policies = resolved.adaptation_policies
        burn_per_move_fns = per_move_fns

        logger.info(
            "Starting sharded burn-in: n_walks=%d, walklength=%d, "
            "adjust_interval=%d, walker_batch_size=%s, n_gpu=%d, n_atoms=%d",
            burn_in_cfg.n_walks,
            burn_in_cfg.walklength,
            burn_in_cfg.adjust_interval,
            getattr(burn_in_cfg, "walker_batch_size", None),
            n_gpu,
            n_atoms_burn,
        )

        ns_state_burn = initial_walk(
            key=key_burn,
            ns_state=ns_state_burn,
            step_fn=step_fn,
            n_walks=burn_in_cfg.n_walks,
            walklength=burn_in_cfg.walklength,
            adjust_interval=burn_in_cfg.adjust_interval,
            emax_offset_per_atom=burn_in_cfg.emax_offset_per_atom,
            n_atoms=n_atoms_burn,
            batcher=batcher,
            walker_batch_size=getattr(burn_in_cfg, "walker_batch_size", None),
            per_move_fns=burn_per_move_fns,
            adaptation_policies=adaptation_policies,
            move_names=tuple(d.name for d in resolved.move_descriptors),
            adjust_n_samples=getattr(
                resolved.adaptation_cfg,
                "adjust_n_samples",
                50,
            ),
            adjust_max_rounds=getattr(
                resolved.adaptation_cfg,
                "adjust_max_rounds",
                15,
            ),
            max_neighbors_list=tuple(resolved.backend.max_neighbors_list),
            max_neighbors_offset=resolved.backend.max_neighbors_offset,
            max_neighbors_shrink_dwell=resolved.backend.max_neighbors_shrink_dwell,
        )

        # Flatten sharded (G, K/G, ...) population back to (K, ...) so the
        # neighbor-count refresh and ``run_ns_sharded``'s ``init_ns_sharded``
        # see the canonical flat input layout.
        pop_burn = ns_state_burn.population

        def _flatten_shard(arr):
            if arr is None:
                return None
            return arr.reshape((-1,) + arr.shape[2:])

        positions = _flatten_shard(pop_burn.positions)
        energies = _flatten_shard(pop_burn.energy)
        cells = _flatten_shard(pop_burn.cell)
        logger.info("Sharded burn-in complete")

        # Post-burn-in neighbor-count refresh — burn-in drifts
        # positions/cells; the resolver's pre-burn-in counts are stale.
        if hasattr(base_backend, "max_neighbors_for"):
            logger.info(
                "Recomputing post-burn-in max neighbor counts (n_walkers=%d)",
                positions.shape[0],
            )
            initial_max_neighbor_counts = _recompute_max_neighbor_counts(
                base_backend,
                positions,
                cells,
            )

    ensemble_params = (
        resolved.ensemble_params_per_run[0]
        if resolved.ensemble_params_per_run
        else None
    )

    callbacks: list[Any] = [
        ProgressCallback(info_interval=resolved.output.info_interval),
        EnergyCheckCallback(),
    ]

    if resolved.output.temperature_lag_interval is not None:
        callbacks.append(
            TemperatureCallback(
                n_live=ns.n_live,
                n_cull=ns.n_cull,
                lag=resolved.output.temperature_lag_interval,
                interval=resolved.output.temperature_interval,
                kB=resolved.output.temperature_kB,
            )
        )

    if resolved.output.collision_check_threshold is not None:
        callbacks.append(
            CollisionCheckCallback(
                threshold=resolved.output.collision_check_threshold,
                interval=resolved.output.collision_check_interval,
            )
        )

    symbol_map = resolved.init.symbol_map
    callbacks.append(
        CheckpointCallback(
            working_dir=working_dir,
            interval=resolved.output.checkpoint_interval,
            prefix=resolved.output.out_file_prefix,
            symbol_map=symbol_map,
        )
    )

    if memprof := os.environ.get("JAXRENS_MEMPROF"):
        callbacks.append(MemProfileCallback(working_dir / memprof))

    # Rewind output streams to the checkpoint on restart (0 = fresh).
    restart_iteration = _restart_iteration_from(resolved.init.restart_state)

    n_atoms = positions.shape[-2]
    traj_path = (
        working_dir
        / f"{resolved.output.out_file_prefix}.traj.{resolved.output.format}"
    )
    writer = create_trajectory_writer(
        resolved.output.format,
        traj_path,
        symbol_map,
        mode=writer_mode,
        restart_iteration=restart_iteration,
        wrap=resolved.output.wrap_atoms,
        clean_snapshots=resolved.output.snapshot_clean,
    )
    energy_path = working_dir / f"{resolved.output.out_file_prefix}.energies"
    energy_logger = EnergyLogger(
        energy_path,
        n_walkers=n_live,
        n_atoms=n_atoms,
        mode=writer_mode,
        restart_iteration=restart_iteration,
    )
    callbacks.append(
        TrajectoryCallback(
            writer=writer,
            energy_logger=energy_logger,
            traj_interval=resolved.output.traj_interval,
            snapshot_interval=resolved.output.snapshot_interval,
        )
    )

    if resolved.move_descriptors:
        adapt_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.adaptation.h5"
        )
        move_name_list = [d.name for d in resolved.move_descriptors]
        adaptation_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=move_name_list,
            n_runs=1,
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(AdaptationCallback(adaptation_logger))

    if getattr(resolved.output, "save_max_neighbors", False):
        from jaxrens.cli.monitor import MaxNeighborsCallback
        from jaxrens.io.max_neighbors_log import MaxNeighborsLogger

        mn_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.max_neighbors.h5"
        )
        mn_logger = MaxNeighborsLogger(
            path=mn_log_path,
            n_runs=1,
            n_walkers=n_live,
            flush_interval=int(
                getattr(resolved.output, "flush_interval", 1000)
            ),
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(
            MaxNeighborsCallback(
                mn_logger,
                interval=int(
                    getattr(resolved.output, "max_neighbors_interval", 1)
                ),
            )
        )

    if resolved.move_descriptors and getattr(
        resolved.output,
        "save_acc_rates",
        False,
    ):
        from jaxrens.cli.monitor import AccRatesCallback
        from jaxrens.io.acc_rates_log import AccRatesLogger

        acc_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.acc_rates.h5"
        )
        acc_logger = AccRatesLogger(
            path=acc_log_path,
            move_names=[d.name for d in resolved.move_descriptors],
            n_runs=1,
            flush_interval=int(
                getattr(resolved.output, "flush_interval", 1000)
            ),
            mode=writer_mode,
            restart_iteration=restart_iteration,
        )
        callbacks.append(
            AccRatesCallback(
                acc_logger,
                interval=int(
                    getattr(resolved.output, "acc_rates_interval", 1)
                ),
            )
        )

    first_mc = resolved.moves[0]
    full_auto_kwargs: dict[str, Any] = {}
    if (
        resolved.adaptation_cfg is not None
        and resolved.adaptation_cfg.full_auto
    ):
        adjust_factor = (
            resolved.adaptation_cfg.defaults.adjust_factor
            if resolved.adaptation_cfg.defaults.adjust_factor is not None
            else 1.5
        )
        full_auto_kwargs = dict(
            per_move_fns=per_move_fns,
            adjust_interval=resolved.adaptation_cfg.adjust_interval,
            adjust_n_samples=resolved.adaptation_cfg.adjust_n_samples,
            adjust_max_rounds=resolved.adaptation_cfg.adjust_max_rounds,
            adjust_factor=adjust_factor,
            trial_batch_size=resolved.adaptation_cfg.trial_batch_size,
        )

    logger.info(
        "Starting sharded-single NS: n_gpu=%d, n_live=%d "
        "(K_per_gpu=%d), n_mcmc=%d, max_iter=%s",
        n_gpu,
        n_live,
        n_live // n_gpu,
        ns.n_mcmc_steps,
        ns.max_iterations,
    )

    result = run_ns_sharded(
        positions=positions,
        types=types,
        energies=energies,
        cells=cells,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_key=key,
        n_gpu=n_gpu,
        n_walkers=n_live,
        max_iterations=ns.max_iterations,
        n_mcmc_steps=ns.n_mcmc_steps,
        n_extra=ns.n_extra,
        convergence_threshold=ns.convergence_threshold,
        initial_step_size=first_mc.step_size,
        target_acceptance=first_mc.target_acceptance,
        callbacks=callbacks,
        termination_criteria=list(resolved.termination),
        ensemble_params=ensemble_params,
        move_descriptors=list(resolved.move_descriptors),
        max_neighbors_list=tuple(resolved.backend.max_neighbors_list),
        max_neighbors_offset=resolved.backend.max_neighbors_offset,
        max_neighbors_shrink_dwell=resolved.backend.max_neighbors_shrink_dwell,
        initial_max_neighbor_counts=initial_max_neighbor_counts,
        batcher=batcher,
        restart_state=resolved.init.restart_state,
        **full_auto_kwargs,
    )

    return result
