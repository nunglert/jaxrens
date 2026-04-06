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
from jaxrens.cli.monitor import (
    CheckpointCallback,
    EnergyCheckCallback,
    ProgressCallback,
    TrajectoryCallback,
)
from jaxrens.cli.parser import load_config
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.trajectory import create_trajectory_writer
from jaxrens.sampling.moves.galilean import build_kernel as gmc_build_kernel
from jaxrens.sampling.moves.random_walk import build_kernel as rw_build_kernel
from jaxrens.sampling.nested_sampling import run_ns
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig

logger = logging.getLogger(__name__)


def setup_move_kernel(
    move_config: MoveConfig,
    energy_fn: Any,
    params: Any,
):
    """Create the MCMC step function based on move config."""
    match move_config.move_type:
        case "random_walk":
            return rw_build_kernel(energy_fn, params)
        case "galilean" | "gmc":
            return gmc_build_kernel(
                energy_fn, params,
                n_reflect=move_config.n_steps,
            )
        case _:
            raise ValueError(f"Unknown move type: {move_config.move_type!r}")


def run_from_config(
    ns_config: NSConfig,
    move_config: MoveConfig,
    backend_config: BackendConfig,
    output_config: OutputConfig,
    initial_positions: jnp.ndarray,
    initial_types: jnp.ndarray,
    initial_energies: jnp.ndarray | None = None,
    initial_boxes: jnp.ndarray | None = None,
    symbol_map: dict[int, str] | None = None,
) -> dict:
    """Run NS from typed config objects.

    This is the main programmatic entry point.

    Args:
        ns_config: NS run parameters.
        move_config: Move type and parameters.
        backend_config: Energy backend parameters.
        output_config: I/O and output parameters.
        initial_positions: Starting walker positions (n_walkers, n_atoms, 3).
        initial_types: Atom types (n_atoms,).
        initial_energies: Starting energies (n_walkers,). Computed if None.
        initial_boxes: Unit cells (n_walkers, 3, 3) or None.
        symbol_map: Atom type -> element symbol mapping.

    Returns:
        Final NS state dict.
    """
    if symbol_map is None:
        symbol_map = {0: "X"}

    # Load backend
    backend_kwargs = {}
    if backend_config.checkpoint_path:
        backend_kwargs["pickle_file"] = backend_config.checkpoint_path
    if backend_config.cutoff is not None:
        backend_kwargs["cutoff"] = backend_config.cutoff

    energy_fn, params = load_backend(backend_config.backend_type, **backend_kwargs)

    # Compute initial energies if not provided
    if initial_energies is None:
        initial_energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, initial_positions, initial_types
        )

    # Build move kernel
    step_fn = setup_move_kernel(move_config, energy_fn, params)

    # Set up callbacks
    callbacks = [
        ProgressCallback(info_interval=output_config.info_interval),
        EnergyCheckCallback(),
    ]

    working_dir = output_config.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback
    callbacks.append(
        CheckpointCallback(
            working_dir=working_dir,
            interval=output_config.checkpoint_interval,
            prefix=output_config.out_file_prefix,
            symbol_map=symbol_map,
        )
    )

    # Trajectory callback
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

    # Run
    key = jax.random.key(ns_config.seed)
    result = run_ns(
        positions=initial_positions,
        types=initial_types,
        energies=initial_energies,
        boxes=initial_boxes,
        step_fn=step_fn,
        rng_key=key,
        max_iterations=ns_config.max_iterations,
        n_mcmc_steps=ns_config.n_mcmc_steps,
        convergence_threshold=ns_config.convergence_threshold,
        initial_step_size=move_config.step_size,
        target_acceptance=move_config.target_acceptance,
        adapt_warmup=move_config.adaptation_warmup,
        callbacks=callbacks,
    )

    return result


def run_from_file(
    config_path: Path | str,
    initial_positions: jnp.ndarray,
    initial_types: jnp.ndarray,
    **kwargs: Any,
) -> dict:
    """Run NS from an ns.inp config file.

    Args:
        config_path: Path to configuration file.
        initial_positions: Starting walker positions.
        initial_types: Atom types.
        **kwargs: Additional overrides passed to run_from_config.

    Returns:
        Final NS state dict.
    """
    ns_config, move_config, backend_config, output_config = load_config(config_path)
    return run_from_config(
        ns_config, move_config, backend_config, output_config,
        initial_positions, initial_types, **kwargs
    )
