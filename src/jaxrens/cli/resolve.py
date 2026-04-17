"""Resolution layer: pydantic schemas -> library dataclasses.

``resolve`` is the single seam between the CLI config layer and the
library core.  Cohort expansion (pressure sweeps, seed sweeps) lives in
``expand_cohort``; the CLI iterates the resulting list sequentially.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxrens.backends.base import EnergyBackend
from jaxrens.cli.schema.adaptation import AdaptationConfig, ResolvedAdaptationPolicy
from jaxrens.cli.schema.cell import CellConfig
from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
from jaxrens.cli.schema.init import InitConfig
from jaxrens.cli.schema.root import RootConfig
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.termination import (
    IterationTermination,
    PriorMassTermination,
    TerminationCriterion,
)
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig

logger = logging.getLogger(__name__)

# Output fields that are accepted by OutputSchema but not yet consumed by the
# runtime callback layer.
_DEFERRED_OUTPUT_FIELDS: tuple[str, ...] = (
    "snapshot_time",
    "snapshot_clean",
    "wrap_atoms",
    "save_stepsizes",
    "write_traj_db",
    "write_walkers_db",
)


# ---------------------------------------------------------------------------
# ResolvedInit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedInit:
    """Holds the resolved initial state arrays for a single run.

    Arrays are None when not computable at resolution time (e.g., energies
    require the backend, which is evaluated separately).
    """

    initial_positions: Any  # shape: (n_live, n_atoms, 3) or None
    initial_types: Any       # shape: (n_atoms,) dtype int32 or None
    initial_cells: Any       # shape: (n_live, 3, 3) or None
    initial_energies: Any    # shape: (n_live,) or None — None until evaluated


def _resolve_init(init: InitConfig, n_live: int, seed: int) -> ResolvedInit:
    """Resolve an ``InitConfig`` into concrete initial-state arrays.

    Only ``start_species`` is fully supported today.  All other source types
    raise ``NotImplementedError`` with a clear message.

    Args:
        init: Validated ``InitConfig``.
        n_live: Number of live walkers (from ``NSConfig.n_live``).
        seed: PRNG seed.

    Returns:
        ``ResolvedInit`` with arrays for the ``start_species`` path, or raises.

    Raises:
        NotImplementedError: For ``start_config_file``, ``start_walker_set``,
            and ``restart_file`` — no structure reader / walker loader exists
            in jaxrens today.
    """
    if init.start_config_file is not None:
        raise NotImplementedError(
            "start_config_file is not yet supported: jaxrens has no structure "
            "file reader (e.g. ase.io.read).  Adding one is tracked as a "
            "separate task.  Use start_species instead."
        )

    if init.start_walker_set is not None:
        raise NotImplementedError(
            "start_walker_set is not yet supported: no walker-set loader "
            "exists in jaxrens today.  Use start_species instead."
        )

    if init.restart_file is not None:
        raise NotImplementedError(
            "restart_file is not yet supported: checkpoint-based restart is "
            "not yet wired into the CLI resolver.  Use start_species instead."
        )

    # start_species path — synthesize random initial positions
    assert init.start_species is not None
    species_counts = init.parsed_species()
    assert species_counts is not None

    # Derive n_atoms and types array from species_counts
    types_list: list[int] = []
    for z, count in sorted(species_counts.items()):
        types_list.extend([z] * count)
    n_atoms = len(types_list)

    # Map atomic numbers to contiguous 0-based integer indices
    unique_z = sorted(set(types_list))
    z_to_idx = {z: i for i, z in enumerate(unique_z)}
    type_indices = [z_to_idx[z] for z in types_list]
    initial_types = jnp.array(type_indices, dtype=jnp.int32)

    key = jax.random.key(seed)
    key, key_pos = jax.random.split(key)
    initial_positions = jax.random.uniform(
        key_pos,
        shape=(n_live, n_atoms, 3),
        minval=-3.0,
        maxval=3.0,
    )

    return ResolvedInit(
        initial_positions=initial_positions,
        initial_types=initial_types,
        initial_cells=None,
        initial_energies=None,
    )


# ---------------------------------------------------------------------------
# Warning helper for deferred output fields
# ---------------------------------------------------------------------------

def _warn_unused_output_fields(output_schema: Any) -> None:
    """Emit warnings for deferred ``OutputSchema`` fields that are non-default.

    Args:
        output_schema: An ``OutputSchema`` instance.
    """
    deferred_defaults: dict[str, Any] = {
        "snapshot_time": None,
        "snapshot_clean": False,
        "wrap_atoms": False,
        "save_stepsizes": False,
        "write_traj_db": False,
        "write_walkers_db": False,
    }
    for field_name, default in deferred_defaults.items():
        value = getattr(output_schema, field_name, default)
        if value != default:
            logger.warning(
                "output.%s=%r is set but not yet consumed by the runtime — "
                "this field is a deferred placeholder.  The value will be "
                "ignored until runtime support is added.",
                field_name,
                value,
            )


# ---------------------------------------------------------------------------
# ResolvedConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedConfig:
    """Holds all library dataclasses produced by ``resolve``."""

    ns: NSConfig
    moves: tuple[MoveConfig, ...]
    move_descriptors: tuple[MoveDescriptor, ...]
    backend: BackendConfig
    energy_backend: EnergyBackend
    output: OutputConfig
    termination: tuple[TerminationCriterion, ...]
    adaptation_policies: tuple[ResolvedAdaptationPolicy, ...]
    init: ResolvedInit
    cell: CellConfig
    cohort_index: int = 0
    ensemble_params: dict = field(default_factory=dict)


def _cohort_size(root: RootConfig) -> int:
    """Return the number of cohort elements implied by the ensemble spec."""
    from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
    if isinstance(root.ensemble, NPTEnsembleSpec):
        return root.ensemble.cohort_size()
    return 1


def _seed_list(root: RootConfig, n: int) -> list[int]:
    """Return a list of *n* seeds derived from ``root.run.seed``.

    If ``run.seed`` is a scalar, generate ``[seed + i for i in range(n)]``.
    """
    return [root.run.seed + i for i in range(n)]


def _resolve_one(root: RootConfig, cohort_index: int = 0) -> ResolvedConfig:
    """Resolve ``root`` into library dataclasses for a single cohort element."""
    ensemble_params = root.ensemble.to_ensemble_params(cohort_index=cohort_index)
    pressure = ensemble_params.get("pressure", None)

    seed = root.run.seed + cohort_index if cohort_index > 0 else root.run.seed

    ns = NSConfig(
        n_live=root.run.n_live,
        max_iterations=root.run.max_iterations,
        convergence_threshold=root.run.convergence_threshold,
        n_mcmc_steps=root.run.n_mcmc_steps,
        n_cull=root.run.n_cull,
        seed=seed,
        pressure=pressure,
    )

    moves = tuple(m.to_move_config() for m in root.moves)
    move_descriptors = tuple(m.to_descriptor() for m in root.moves)

    backend = root.backend.to_backend_config()
    energy_backend = root.backend.build_backend()

    output = OutputConfig(
        format=root.output.format,
        traj_interval=root.output.traj_interval,
        snapshot_interval=root.output.snapshot_interval,
        checkpoint_interval=root.output.checkpoint_interval,
        info_interval=root.output.info_interval,
        out_file_prefix=root.output.out_file_prefix,
        working_dir=Path(root.output.working_dir),
    )

    _warn_unused_output_fields(root.output)

    # CellConfig deferred-field warning: emit when non-default values are set
    cell_defaults = CellConfig()
    if root.cell != cell_defaults:
        logger.warning(
            "cell section contains non-default values %r but CellConfig fields "
            "are not yet automatically threaded into move kernels.  Per-move "
            "specs (VolumeMoveSpec, ShearMoveSpec, StretchMoveSpec) carry their "
            "own copies of these parameters.  Unified CellConfig propagation is "
            "planned for a future task.",
            root.cell.model_dump(),
        )

    if root.termination is not None:
        termination = tuple(spec.to_criterion() for spec in root.termination)
    else:
        termination = (
            IterationTermination(ns.max_iterations),
            PriorMassTermination(ns.n_live, ns.convergence_threshold),
        )

    adaptation_policies = tuple(
        root.adaptation.resolve_for(m._effective_name())
        for m in root.moves
    )

    resolved_init = _resolve_init(root.init, n_live=ns.n_live, seed=seed)

    return ResolvedConfig(
        ns=ns,
        moves=moves,
        move_descriptors=move_descriptors,
        backend=backend,
        energy_backend=energy_backend,
        output=output,
        termination=termination,
        adaptation_policies=adaptation_policies,
        init=resolved_init,
        cell=root.cell,
        cohort_index=cohort_index,
        ensemble_params=ensemble_params,
    )


def expand_cohort(root: RootConfig) -> list[ResolvedConfig]:
    """Expand a ``RootConfig`` into one ``ResolvedConfig`` per cohort element.

    Cohort axes are defined by the ensemble spec (e.g. a list of pressures in
    ``NPTEnsembleSpec``).  Scalar specs produce a single-element cohort.

    Alignment rule: all list-valued axes must have the same length, or be
    scalars (broadcast to the common length).  Mismatched lengths raise
    ``ValueError`` with a clear message.

    Seed handling: ``run.seed`` is a scalar; each cohort element receives
    ``seed + cohort_index`` so runs are distinct but deterministically
    reproducible from the base seed.

    Args:
        root: Fully validated ``RootConfig``.

    Returns:
        List of ``ResolvedConfig`` objects, one per cohort element.  The list
        is always non-empty; single-element cohorts are the common case.
    """
    n = _cohort_size(root)

    if n == 1:
        return [_resolve_one(root, cohort_index=0)]

    results = []
    for i in range(n):
        results.append(_resolve_one(root, cohort_index=i))
    return results


def resolve(root: RootConfig) -> ResolvedConfig:
    """Translate a validated ``RootConfig`` into library dataclasses.

    This is a thin wrapper around ``expand_cohort`` for single-element cohorts.
    For multi-element cohorts (pressure sweeps etc.) use ``expand_cohort``
    directly.

    Args:
        root: Fully validated pydantic config.

    Returns:
        ``ResolvedConfig`` for cohort index 0.

    Raises:
        AssertionError: If the config implies a multi-element cohort.  Callers
            that intentionally run sweeps should use ``expand_cohort`` instead.
    """
    cohort = expand_cohort(root)
    assert len(cohort) == 1, (
        f"resolve() called on a multi-element cohort (size {len(cohort)}). "
        "Use expand_cohort() instead."
    )
    return cohort[0]
