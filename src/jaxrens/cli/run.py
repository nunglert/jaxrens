"""Main entry point for running a nested sampling calculation.

Wires together: config loading, backend creation, move kernel construction,
NS loop execution, and I/O callbacks.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.backends.loader import load_backend
from jaxrens.backends.ensemble import EnsembleBackend, make_ensemble_params
from jaxrens.cli.monitor import (
    AdaptationCallback,
    BatchedTrajectoryCallback,
    CheckpointCallback,
    EnergyCheckCallback,
    MemProfileCallback,
    ProgressCallback,
    TrajectoryCallback,
)
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.trajectory import create_trajectory_writer
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns_parallel,
    run_ns,
    run_ns_multi_gpu,
)
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig

logger = logging.getLogger(__name__)


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _configure_file_logging(
    *,
    working_dir: Path,
    prefix: str,
    level: str,
) -> None:
    """Attach file handlers to the ``jaxrens`` logger.

    Always writes INFO+ to ``<working_dir>/<prefix>.log``. If ``level`` is
    ``debug``, additionally writes DEBUG+ to ``<working_dir>/<prefix>.debug.log``.
    Idempotent: removes any prior handlers this function attached before
    re-attaching, so repeated calls (e.g. across cohort runs) stay clean.
    """
    root = logging.getLogger("jaxrens")
    root.setLevel(logging.DEBUG if level == "debug" else logging.INFO)

    for h in list(root.handlers):
        if getattr(h, "_jaxrens_managed", False):
            root.removeHandler(h)
            h.close()

    info_h = logging.FileHandler(working_dir / f"{prefix}.log", mode="w")
    info_h.setLevel(logging.INFO)
    info_h.setFormatter(logging.Formatter(_LOG_FORMAT))
    info_h._jaxrens_managed = True  # type: ignore[attr-defined]
    root.addHandler(info_h)

    if level == "debug":
        debug_h = logging.FileHandler(working_dir / f"{prefix}.debug.log", mode="w")
        debug_h.setLevel(logging.DEBUG)
        debug_h.setFormatter(logging.Formatter(_LOG_FORMAT))
        debug_h._jaxrens_managed = True  # type: ignore[attr-defined]
        root.addHandler(debug_h)


def _barrier(label: str, *arrays: Any) -> None:
    """Force materialisation on JAX arrays and log timing under ``label``.

    Use as a stage boundary in the multi-GPU dispatcher: any OOM during the
    just-completed stage's compile/execute will surface inside this call,
    not deferred to the next materialisation point.  No-op for non-array
    inputs and ``None``.  Only emits at DEBUG level — zero cost when the
    debug log handler is not attached.
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
            use ``build_mwg`` directly with pre-built ``MoveKernel``s.
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
    base_backend: Any = None,
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

        base_backend = load_backend(backend_config.backend_type, **backend_kwargs)

    # Wrap with ensemble corrections if needed
    ensemble_params = None
    if ns_config.pressure:
        backend = EnsembleBackend(base_backend, pressure=ns_config.pressure)
        ensemble_params = make_ensemble_params(pressure=ns_config.pressure)
    else:
        backend = base_backend

    if initial_energies is None:
        logger.debug(
            "initial_energies missing from ResolvedInit; computing in run_from_config. "
            "Prefer populating ResolvedInit.initial_energies via the resolver."
        )
        def eval_one(pos):
            e, _, _ = backend(pos, initial_types,
                             initial_cells[0] if initial_cells is not None else jnp.zeros((3, 3)),
                             0)
            return e
        initial_energies = jax.vmap(eval_one)(initial_positions)

    # Build MWG sampler. Prefer pre-built MoveKernels when available — they
    # carry n_atoms / n_species that MoveConfig can't express (volume, shear,
    # stretch, single_atom_sweep, alchemical_morph).
    if move_descriptors is not None:
        init_fn, step_fn, per_move_fns = build_mwg(backend, list(move_descriptors))
    else:
        init_fn, step_fn, per_move_fns = setup_mwg(move_config, backend)

    working_dir = output_config.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    _configure_file_logging(
        working_dir=working_dir,
        prefix=output_config.out_file_prefix,
        level=output_config.log_level,
    )

    # Set up callbacks
    callbacks = [
        ProgressCallback(info_interval=output_config.info_interval),
        EnergyCheckCallback(),
    ]

    callbacks.append(
        CheckpointCallback(
            working_dir=working_dir,
            interval=output_config.checkpoint_interval,
            prefix=output_config.out_file_prefix,
            symbol_map=symbol_map,
        )
    )

    if (memprof := os.environ.get("JAXRENS_MEMPROF")):
        callbacks.append(MemProfileCallback(working_dir / memprof))

    traj_path = working_dir / f"{output_config.out_file_prefix}.traj.{output_config.format}"
    writer = create_trajectory_writer(output_config.format, traj_path, symbol_map)
    energy_logger = EnergyLogger(
        working_dir / f"{output_config.out_file_prefix}.energies",
        n_walkers=ns_config.n_live,
        n_atoms=initial_positions.shape[-2],
    )
    callbacks.append(
        TrajectoryCallback(
            writer=writer,
            energy_logger=energy_logger,
            traj_interval=output_config.traj_interval,
            snapshot_interval=output_config.snapshot_interval,
        )
    )

    # Wire adaptation logger when full_auto is active
    if adaptation_config is not None and adaptation_config.full_auto and move_descriptors is not None:
        from jaxrens.io.adaptation_log import AdaptationLogger

        adapt_log_path = working_dir / f"{output_config.out_file_prefix}.adaptation.h5"
        move_name_list = [d.name for d in move_descriptors]
        adaptation_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=move_name_list,
            n_runs=1,
        )
        callbacks.append(AdaptationCallback(adaptation_logger))

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
            burn_in_cfg.n_walks, burn_in_cfg.walklength,
        )

        n_atoms = initial_positions.shape[1] if initial_positions.ndim >= 2 else 1

        _ladder = tuple(int(x) for x in backend_config.max_neighbors_list)
        _offset = int(backend_config.max_neighbors_offset)
        starting_bucket = _choose_starting_bucket(
            initial_max_neighbor_counts, _ladder, _offset,
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
                adaptation_config.resolve_for(d.name)
                for d in move_descriptors
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
            batched=False,
            walker_batch_size=getattr(burn_in_cfg, "walker_batch_size", None),
            run_batch_size=getattr(burn_in_cfg, "run_batch_size", None),
            per_move_fns=burn_per_move_fns,
            adaptation_policies=burn_adaptation_policies,
            adjust_n_samples=getattr(adaptation_config, "adjust_n_samples", 50) if adaptation_config is not None else 50,
            adjust_max_rounds=getattr(adaptation_config, "adjust_max_rounds", 15) if adaptation_config is not None else 15,
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
            backend, initial_positions, initial_cells,
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
            move_descriptors=list(move_descriptors) if move_descriptors is not None else None,
            adjust_interval=adaptation_config.full_auto_steps,
            adjust_n_samples=adaptation_config.adjust_n_samples,
            adjust_max_rounds=adaptation_config.adjust_max_rounds,
            adjust_factor=adjust_factor,
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
        initial_max_neighbor_counts=initial_max_neighbor_counts,
        **full_auto_kwargs,
    )

    return result


# ---------------------------------------------------------------------------
# Multi-run (multi-GPU) dispatch
# ---------------------------------------------------------------------------


def run_multi_gpu_from_config(resolved) -> dict:
    """Execute a multi-run NS dispatch from a ``ResolvedMultiRunConfig``.

    Mirrors :func:`run_from_config` but ends in ``run_ns_multi_gpu`` with
    per-replica ``ensemble_params_per_run``.  Wires all five callbacks with
    batched-aware variants where needed:

    * ``ProgressCallback``, ``AdaptationCallback``, ``EnergyCheckCallback`` —
      already (G, P)-safe (see WORKLOG 2026-04-18 Task A/B).
    * ``CheckpointCallback`` — saves HDF5 with batched shapes via the
      already-batched-safe ``io/checkpoint.py`` path.
    * ``BatchedTrajectoryCallback`` — one writer + energy logger per replica,
      file suffix ``.run{r:02d}``.

    Burn-in runs via ``initial_walk(batched=True)`` on the stacked
    ``(n_total, K, ...)``-shaped NSState, before entering
    ``run_ns_multi_gpu``.
    """
    from jaxrens.io.adaptation_log import AdaptationLogger
    from jaxrens.sampling.termination import IterationTermination, PriorMassTermination
    from jaxrens.state.ns import NSState as _NSState  # for isinstance

    ns = resolved.ns
    n_gpu = ns.n_gpu
    n_per_gpu = ns.n_per_gpu
    n_total = n_gpu * n_per_gpu

    # --- Backend: one base, wrapped once with EnsembleBackend --------------
    # Per-call ``ensemble_params`` dicts override the wrapper's closured
    # defaults (see backends/ensemble.py __call__).
    base_backend = resolved.base_backend
    backend = EnsembleBackend(base_backend, pressure=0.0)

    init_fn, step_fn, per_move_fns = build_mwg(backend, list(resolved.move_descriptors))

    working_dir = resolved.output.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    _configure_file_logging(
        working_dir=working_dir,
        prefix=resolved.output.out_file_prefix,
        level=resolved.output.log_level,
    )

    n_live = ns.n_live
    positions = resolved.init.initial_positions  # (n_total, K, A, 3)
    types = resolved.init.initial_types          # (A,)
    cells = resolved.init.initial_cells          # (n_total, K, 3, 3) | None
    energies = resolved.init.initial_energies    # (n_total, K)

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
        positions.shape, None if cells is None else cells.shape,
        energies.shape, starting_bucket, _ladder, _offset,
    )

    # --- Burn-in (batched=True) -------------------------------------------
    burn_in_cfg = resolved.initial_walk_config
    do_burn_in = burn_in_cfg is not None and burn_in_cfg.n_walks > 0
    if do_burn_in:
        from jaxrens.init.burn_in import initial_walk

        key, key_init, key_burn = jax.random.split(key, 3)
        rng_keys = jax.random.split(key_init, n_total)
        step_sizes = jnp.full(len(resolved.move_descriptors), resolved.moves[0].step_size)

        logger.debug("[stage] init_ns_parallel (burn-in NSState): starting")
        ns_state_burn = init_ns_parallel(
            init_fn,
            positions, types, energies, cells, rng_keys,
            step_sizes=step_sizes,
            ensemble_params_per_run=list(resolved.ensemble_params_per_run),
            max_neighbors=starting_bucket,
            max_neighbor_counts=_init_counts,
        )
        _barrier("init_ns_parallel", ns_state_burn.population.positions)

        n_atoms = positions.shape[-2]
        adaptation_policies = resolved.adaptation_policies
        burn_per_move_fns = per_move_fns

        logger.info(
            "Starting initial burn-in: n_walks=%d, walklength=%d, "
            "adjust_interval=%d, walker_batch_size=%s, run_batch_size=%s, "
            "n_atoms=%d",
            burn_in_cfg.n_walks, burn_in_cfg.walklength,
            burn_in_cfg.adjust_interval,
            getattr(burn_in_cfg, "walker_batch_size", None),
            getattr(burn_in_cfg, "run_batch_size", None),
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
            batched=True,
            walker_batch_size=getattr(burn_in_cfg, "walker_batch_size", None),
            run_batch_size=getattr(burn_in_cfg, "run_batch_size", None),
            per_move_fns=burn_per_move_fns,
            adaptation_policies=adaptation_policies,
            adjust_n_samples=getattr(resolved.adaptation_cfg, "adjust_n_samples", 50),
            adjust_max_rounds=getattr(resolved.adaptation_cfg, "adjust_max_rounds", 15),
        )
        pop = ns_state_burn.population
        positions = pop.positions
        energies = pop.energy
        cells = pop.cell
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
    post_burn_in_counts = _recompute_max_neighbor_counts(
        base_backend, positions, cells,
    ) if do_burn_in else _init_counts
    _barrier("_recompute_max_neighbor_counts", post_burn_in_counts)

    # --- PRNG keys for the multi-GPU dispatch -----------------------------
    key, key_run = jax.random.split(key)
    rng_keys_flat = jax.random.split(key_run, n_total)

    # --- Callbacks ---------------------------------------------------------
    callbacks: list[Any] = [
        ProgressCallback(info_interval=resolved.output.info_interval),
        EnergyCheckCallback(),
    ]

    symbol_map = resolved.init.symbol_map
    callbacks.append(
        CheckpointCallback(
            working_dir=working_dir,
            interval=resolved.output.checkpoint_interval,
            prefix=resolved.output.out_file_prefix,
            symbol_map=symbol_map,
        )
    )

    if (memprof := os.environ.get("JAXRENS_MEMPROF")):
        callbacks.append(MemProfileCallback(working_dir / memprof))

    n_atoms = positions.shape[-2]
    writers = []
    energy_loggers = []
    for r in range(n_total):
        traj_path = (
            working_dir
            / f"{resolved.output.out_file_prefix}.run{r:02d}.traj.{resolved.output.format}"
        )
        writers.append(
            create_trajectory_writer(resolved.output.format, traj_path, symbol_map)
        )
        energy_path = working_dir / f"{resolved.output.out_file_prefix}.run{r:02d}.energies"
        energy_loggers.append(
            EnergyLogger(energy_path, n_walkers=n_live, n_atoms=n_atoms)
        )
    callbacks.append(
        BatchedTrajectoryCallback(
            writers=writers,
            energy_loggers=energy_loggers,
            traj_interval=resolved.output.traj_interval,
            snapshot_interval=resolved.output.snapshot_interval,
        )
    )

    if resolved.adaptation_cfg is not None and resolved.adaptation_cfg.full_auto:
        adapt_log_path = (
            working_dir / f"{resolved.output.out_file_prefix}.adaptation.h5"
        )
        move_name_list = [d.name for d in resolved.move_descriptors]
        adaptation_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=move_name_list,
            n_runs=n_total,
        )
        callbacks.append(AdaptationCallback(adaptation_logger))

    first_mc = resolved.moves[0]
    full_auto_kwargs: dict[str, Any] = {}
    if resolved.adaptation_cfg is not None and resolved.adaptation_cfg.full_auto:
        adjust_factor = (
            resolved.adaptation_cfg.defaults.adjust_factor
            if resolved.adaptation_cfg.defaults.adjust_factor is not None
            else 1.5
        )
        full_auto_kwargs = dict(
            per_move_fns=per_move_fns,
            adjust_interval=resolved.adaptation_cfg.full_auto_steps,
            adjust_n_samples=resolved.adaptation_cfg.adjust_n_samples,
            adjust_max_rounds=resolved.adaptation_cfg.adjust_max_rounds,
            adjust_factor=adjust_factor,
        )

    logger.info(
        "Starting multi-GPU NS: n_gpu=%d, n_per_gpu=%d (n_total=%d), "
        "n_walkers=%d, n_mcmc=%d, max_iter=%d",
        n_gpu, n_per_gpu, n_total, n_live,
        ns.n_mcmc_steps, ns.max_iterations,
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
        initial_max_neighbor_counts=post_burn_in_counts,
        **full_auto_kwargs,
    )
    return result


