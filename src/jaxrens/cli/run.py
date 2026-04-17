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
from jaxrens.cli.parser import load_config
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.trajectory import create_trajectory_writer
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.moves import galilean, hmc, random_walk, single_atom, volume, shear, stretch
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig

logger = logging.getLogger(__name__)

# Map move type names to build_kernel functions
_MOVE_REGISTRY: dict[str, Any] = {
    "random_walk": random_walk.build_kernel,
    "galilean": galilean.build_kernel,
    "gmc": galilean.build_kernel,
    "hmc": hmc.build_kernel,
    "single_atom": single_atom.build_kernel,
    "single_atom_sweep": single_atom.build_sweep_kernel,
    "single_atom_swap": single_atom.build_swap_kernel,
    "volume": volume.build_kernel,
    "shear": shear.build_kernel,
    "stretch": stretch.build_kernel,
}


def _build_kernel_kwargs(move_config: MoveConfig) -> dict[str, Any]:
    """Extract kernel_kwargs from a MoveConfig based on move type."""
    kwargs: dict[str, Any] = {}
    match move_config.move_type:
        case "galilean" | "gmc":
            kwargs["n_reflect"] = move_config.n_steps
        case "hmc":
            kwargs["n_leapfrog"] = move_config.n_steps
        case "volume" | "shear" | "stretch":
            pass
    return kwargs


def _extra_state_fields(move_type: str) -> dict[str, tuple[type, Any]]:
    """Return extra MCState fields required by a move type."""
    if move_type in ("galilean", "gmc"):
        return {
            "direction": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros_like(positions),
            ),
        }
    return {}


def setup_mwg(
    move_configs: list[MoveConfig] | MoveConfig,
    backend: Any,
):
    """Create MWG init_fn and step_fn from move config(s).

    Args:
        move_configs: Single MoveConfig or list of MoveConfigs.
        backend: EnergyBackend instance.

    Returns:
        (init_fn, step_fn) from build_mwg.
    """
    if isinstance(move_configs, MoveConfig):
        move_configs = [move_configs]

    descriptors = []
    for mc in move_configs:
        build_fn = _MOVE_REGISTRY.get(mc.move_type)
        if build_fn is None:
            raise ValueError(
                f"Unknown move type: {mc.move_type!r}. "
                f"Available: {list(_MOVE_REGISTRY)}"
            )
        descriptors.append(
            MoveDescriptor(
                name=mc.move_type,
                build_kernel=build_fn,
                kernel_kwargs=_build_kernel_kwargs(mc),
                weight=mc.weight,
                step_size=mc.step_size,
                extra_state_fields=_extra_state_fields(mc.move_type),
            )
        )

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
        # Use backend to get initial energies (includes ensemble correction)
        def eval_one(pos):
            e, _, _ = backend(pos, initial_types,
                             initial_cells[0] if initial_cells is not None else jnp.zeros((3, 3)),
                             0)
            return e
        initial_energies = jax.vmap(eval_one)(initial_positions)

    # Build MWG sampler
    init_fn, step_fn = setup_mwg(move_config, backend)

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
    )

    return result


def run_from_file(
    config_path: Path | str,
    initial_positions: jnp.ndarray,
    initial_types: jnp.ndarray,
    **kwargs: Any,
) -> dict:
    """Run NS from an ns.inp config file."""
    ns_config, move_config, backend_config, output_config = load_config(config_path)
    return run_from_config(
        ns_config, move_config, backend_config, output_config,
        initial_positions, initial_types, **kwargs
    )
