"""Main entry point for running a nested sampling calculation.

Wires together: config loading, backend creation, move kernel construction,
NS loop execution, and I/O callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.backends.loader import load_backend
from jaxrens.backends.ensemble import EnsembleBackend, make_ensemble_params
from jaxrens.cli.monitor import (
    AdaptationCallback,
    CheckpointCallback,
    EnergyCheckCallback,
    ProgressCallback,
    TrajectoryCallback,
)
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.trajectory import create_trajectory_writer
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns
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
        GalileanMoveSpec,
        GmcMoveSpec,
        HMCMoveSpec,
        RandomWalkMoveSpec,
        SingleAtomMoveSpec,
        SingleAtomSwapMoveSpec,
    )

    _SIMPLE_SPEC_MAP: dict[str, Any] = {
        "random_walk": RandomWalkMoveSpec,
        "galilean": GalileanMoveSpec,
        "gmc": GmcMoveSpec,
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
    symbol_map: dict[int, str] | None = None,
    termination_criteria: list | None = None,
    restart_state=None,
    initial_walk_config=None,
    adaptation_config=None,
    move_descriptors=None,
) -> dict:
    """Run NS from typed config objects."""
    if symbol_map is None:
        symbol_map = {0: "X"}

    # Load backend
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
        from jaxrens.sampling.nested_sampling import init_ns

        logger.info(
            "Running initial-walk burn-in: %d walks x %d steps/walk",
            burn_in_cfg.n_walks, burn_in_cfg.walklength,
        )

        n_atoms = initial_positions.shape[1] if initial_positions.ndim >= 2 else 1

        key, key_init, key_burn = jax.random.split(key, 3)
        ns_state_burn = init_ns(
            init_fn,
            initial_positions,
            initial_types,
            initial_energies,
            initial_cells,
            key_init,
            max_dead=ns_config.max_iterations,
            ensemble_params=ensemble_params,
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
        **full_auto_kwargs,
    )

    return result


