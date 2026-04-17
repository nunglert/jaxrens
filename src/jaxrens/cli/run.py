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
    CheckpointCallback,
    EnergyCheckCallback,
    ProgressCallback,
    TrajectoryCallback,
)
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.trajectory import create_trajectory_writer
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig

logger = logging.getLogger(__name__)


def _move_config_to_descriptor(mc: MoveConfig) -> MoveDescriptor:
    """Convert a ``MoveConfig`` dataclass to a ``MoveDescriptor``.

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
            use ``build_mwg`` directly with pre-built ``MoveDescriptor``s.
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

    # Compute initial energies if not provided
    if initial_energies is None:
        def eval_one(pos):
            e, _, _ = backend(pos, initial_types,
                             initial_cells[0] if initial_cells is not None else jnp.zeros((3, 3)),
                             0)
            return e
        initial_energies = jax.vmap(eval_one)(initial_positions)

    # Build MWG sampler
    init_fn, step_fn, per_move_fns = setup_mwg(move_config, backend)

    # Set up callbacks
    callbacks = [
        ProgressCallback(info_interval=output_config.info_interval),
        EnergyCheckCallback(),
    ]

    working_dir = output_config.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

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
        n_atoms=backend_config.n_atoms,
    )
    callbacks.append(
        TrajectoryCallback(
            writer=writer,
            energy_logger=energy_logger,
            traj_interval=output_config.traj_interval,
            snapshot_interval=output_config.snapshot_interval,
        )
    )

    first_mc = move_config[0] if isinstance(move_config, list) else move_config

    key = jax.random.key(ns_config.seed)
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
        convergence_threshold=ns_config.convergence_threshold,
        initial_step_size=first_mc.step_size,
        target_acceptance=first_mc.target_acceptance,
        adapt_warmup=first_mc.adaptation_warmup,
        callbacks=callbacks,
        ensemble_params=ensemble_params,
        termination_criteria=termination_criteria,
    )

    return result


