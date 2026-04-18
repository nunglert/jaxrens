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
from jaxrens.init.cells import cell_shape_walk, sample_initial_volume
from jaxrens.init.positions import grid_positions_in_cell, uniform_positions_in_cell
from jaxrens.init.rejection import rejection_sample_positions
from jaxrens.init.restart import RestartBundle, load_restart
from jaxrens.init.structure import load_structure
from jaxrens.init.walker_set import load_walker_set
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.termination import (
    IterationTermination,
    PriorMassTermination,
    TerminationCriterion,
)
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig
from jaxrens.utils.cell import check_cell_shape

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
    symbol_map: dict[int, str] | None = None
    restart_state: RestartBundle | None = None


def _build_cells(
    init: InitConfig,
    n_live: int,
    shape_key: jax.Array,
    base_cell: jnp.ndarray,
    n_atoms: int,
    cell_cfg: CellConfig,
) -> jnp.ndarray:
    """Produce (n_live, 3, 3) cells from a base cell.

    If ``init.random_initialise_cell``, run cell_shape_walk per walker;
    otherwise broadcast ``base_cell`` across all walkers.
    """
    _CELL_SHAPE_EQUIL_STEPS = 50

    if init.random_initialise_cell:
        walker_shape_keys = jax.random.split(shape_key, n_live)

        def _walk_one(k):
            cell, _ = cell_shape_walk(
                key=k,
                cell=base_cell,
                n_steps=_CELL_SHAPE_EQUIL_STEPS,
                step_size_shear=0.05,
                step_size_stretch=0.05,
                min_aspect_ratio_val=cell_cfg.min_aspect_ratio,
                n_atoms=n_atoms,
                max_volume_per_atom=cell_cfg.max_volume_per_atom,
                min_volume_per_atom=cell_cfg.min_volume_per_atom,
            )
            return cell

        return jax.vmap(_walk_one)(walker_shape_keys)
    else:
        return jnp.broadcast_to(base_cell[None], (n_live, 3, 3))


def _validate_cells(
    cells: jnp.ndarray,
    n_atoms: int,
    cell_cfg: CellConfig,
) -> None:
    """Raise ``RuntimeError`` if any walker cell fails check_cell_shape."""
    n_live = cells.shape[0]
    for wi in range(n_live):
        cell_valid = bool(
            check_cell_shape(
                cells[wi],
                n_atoms=n_atoms,
                max_vol_per_atom=cell_cfg.max_volume_per_atom,
                min_vol_per_atom=cell_cfg.min_volume_per_atom,
                min_aspect=cell_cfg.min_aspect_ratio,
            )
        )
        if not cell_valid:
            raise RuntimeError(
                f"Walker {wi} produced an invalid cell (failed check_cell_shape). "
                f"Cell:\n{cells[wi]}"
            )


def _sample_per_walker_positions(
    init: InitConfig,
    n_live: int,
    pos_key: jax.Array,
    initial_cells: jnp.ndarray,
    initial_types: jnp.ndarray,
    n_atoms: int,
    energy_backend: EnergyBackend | None,
) -> tuple[jnp.ndarray, list | None]:
    """Sample per-walker positions (and optionally energies) via grid or rejection.

    Returns:
        (initial_positions, energies_list):
          - initial_positions: (n_live, n_atoms, 3)
          - energies_list: list of n_live energy scalars if energy_backend is
            not None; otherwise None.
    """
    start_energy_ceiling = init.start_energy_ceiling_per_atom * n_atoms
    walker_pos_keys = jax.random.split(pos_key, n_live)
    positions_list = []
    energies_list: list | None = [] if energy_backend is not None else None

    for wi in range(n_live):
        w_cell = initial_cells[wi]

        if init.pos_randomization_mode == "grid":
            pos = grid_positions_in_cell(
                walker_pos_keys[wi], w_cell, n_atoms, init.grid_distance
            )
            positions_list.append(pos)
            if energy_backend is not None:
                e, _, _ = energy_backend(pos, initial_types, w_cell, 0)
                energies_list.append(e)
        else:
            pos, e = rejection_sample_positions(
                walker_pos_keys[wi],
                cell=w_cell,
                types=initial_types,
                n_atoms=n_atoms,
                energy_fn=energy_backend if energy_backend is not None else _null_energy_fn,
                start_energy_ceiling=start_energy_ceiling,
                min_distance=init.init_distance_criterion,
                max_tries=init.random_init_max_n_tries,
                mode="uniform",
                grid_distance=init.grid_distance,
            )
            positions_list.append(pos)
            if energy_backend is not None:
                energies_list.append(e)

    return jnp.stack(positions_list, axis=0), energies_list


def _resolve_init_walker_set(
    init: InitConfig,
    cell_cfg: CellConfig,
    n_live: int,
    energy_backend: EnergyBackend | None,
) -> ResolvedInit:
    """Mode C: load a pre-computed set of N walker configurations from disk."""
    if init.random_initialise_pos or init.random_initialise_cell:
        logger.warning(
            "start_walker_set: ignoring random_initialise_pos/cell=True. "
            "Walker-set files are taken verbatim; no randomization applied."
        )

    walker_set = load_walker_set(Path(init.start_walker_set), n_live)
    n_atoms = walker_set.types.shape[1]

    _validate_cells(walker_set.cells, n_atoms, cell_cfg)

    if energy_backend is not None:
        energies = jax.vmap(
            lambda pos, typs, cel: energy_backend(pos, typs, cel, 0)[0]
        )(walker_set.positions, walker_set.types, walker_set.cells)
    else:
        energies = None

    return ResolvedInit(
        initial_positions=walker_set.positions,
        initial_types=walker_set.types,
        initial_cells=walker_set.cells,
        initial_energies=energies,
        symbol_map=walker_set.symbol_map,
    )


def _resolve_init_restart(
    init: InitConfig,
    cell_cfg: CellConfig,
    n_live: int,
    energy_backend: EnergyBackend | None,
) -> ResolvedInit:
    """Mode D: resume an NS run from a checkpoint file (restart_file)."""
    if init.random_initialise_pos or init.random_initialise_cell:
        logger.warning(
            "restart_file: ignoring random_initialise_pos/cell=True. "
            "Checkpoint files are taken verbatim; no randomization applied."
        )

    walker_set, restart_bundle = load_restart(Path(init.restart_file))
    n_atoms = walker_set.types.shape[1]

    _validate_cells(walker_set.cells, n_atoms, cell_cfg)

    if energy_backend is not None:
        energies = jax.vmap(
            lambda pos, typs, cel: energy_backend(pos, typs, cel, 0)[0]
        )(walker_set.positions, walker_set.types, walker_set.cells)
    else:
        energies = None

    return ResolvedInit(
        initial_positions=walker_set.positions,
        initial_types=walker_set.types,
        initial_cells=walker_set.cells,
        initial_energies=energies,
        symbol_map=walker_set.symbol_map,
        restart_state=restart_bundle,
    )


def _resolve_init(
    init: InitConfig,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None = None,
    cell_cfg: CellConfig | None = None,
) -> ResolvedInit:
    """Resolve an ``InitConfig`` into concrete initial-state arrays.

    Supports ``start_species`` (Mode A), ``start_config_file`` (Mode B),
    ``start_walker_set`` (Mode C), and ``restart_file`` (Mode D).

    Args:
        init: Validated ``InitConfig``.
        n_live: Number of live walkers (from ``NSConfig.n_live``).
        seed: PRNG seed.
        energy_backend: Backend used to compute initial energies.  When
            ``None``, ``initial_energies`` is left as ``None``.
        cell_cfg: ``CellConfig`` carrying cell-geometry constraints.  When
            ``None`` a default ``CellConfig()`` is used.

    Returns:
        ``ResolvedInit`` with arrays populated for the chosen mode.
    """
    if cell_cfg is None:
        cell_cfg = CellConfig()

    if init.start_walker_set is not None:
        return _resolve_init_walker_set(init, cell_cfg, n_live, energy_backend)

    if init.restart_file is not None:
        return _resolve_init_restart(init, cell_cfg, n_live, energy_backend)

    if init.start_config_file is not None:
        return _resolve_init_config_file(init, n_live, seed, energy_backend, cell_cfg)

    assert init.start_species is not None
    return _resolve_init_species(init, n_live, seed, energy_backend, cell_cfg)


def _resolve_init_species(
    init: InitConfig,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None,
    cell_cfg: CellConfig,
) -> ResolvedInit:
    """Mode A: initialise from a species string (start_species)."""
    species_counts = init.parsed_species()
    assert species_counts is not None

    types_list: list[int] = []
    for z, count in sorted(species_counts.items()):
        types_list.extend([z] * count)
    n_atoms = len(types_list)

    unique_z = sorted(set(types_list))
    z_to_idx = {z: i for i, z in enumerate(unique_z)}
    type_indices = [z_to_idx[z] for z in types_list]
    initial_types = jnp.array(type_indices, dtype=jnp.int32)

    # Build a symbol_map from atomic numbers for Mode A.
    from ase.data import chemical_symbols
    symbol_map: dict[int, str] = {
        idx: chemical_symbols[z] for idx, z in enumerate(unique_z)
    }

    key = jax.random.key(seed)
    key, vol_key, shape_key, pos_key = jax.random.split(key, 4)

    lc = float(
        sample_initial_volume(
            vol_key,
            n_atoms=n_atoms,
            max_volume_per_atom=cell_cfg.max_volume_per_atom,
            flat_V_prior=cell_cfg.flat_V_prior,
        )
    )
    cubic_cell = jnp.eye(3, dtype=jnp.float32) * lc

    initial_cells = _build_cells(init, n_live, shape_key, cubic_cell, n_atoms, cell_cfg)
    _validate_cells(initial_cells, n_atoms, cell_cfg)

    if init.pos_autoscale_cells:
        logger.warning(
            "pos_autoscale_cells=True is set but not yet implemented in step 2. "
            "The cell will not be scaled to guarantee minimum atom distances. "
            "Rejection sampling may fail if the cell is too small."
        )

    if not init.random_initialise_pos:
        logger.warning(
            "random_initialise_pos=False: all %d walkers start from identical "
            "positions.  This introduces strong correlations across walkers and "
            "may degrade nested sampling evidence accuracy.",
            n_live,
        )
        single_key, _ = jax.random.split(pos_key)
        single_cell = initial_cells[0]
        if init.pos_randomization_mode == "grid":
            single_pos = grid_positions_in_cell(
                single_key, single_cell, n_atoms, init.grid_distance
            )
        else:
            single_pos = uniform_positions_in_cell(single_key, single_cell, n_atoms)
        initial_positions = jnp.broadcast_to(
            single_pos[None], (n_live, n_atoms, 3)
        )
        energies_list = None
    else:
        initial_positions, energies_list = _sample_per_walker_positions(
            init, n_live, pos_key, initial_cells, initial_types, n_atoms, energy_backend
        )

    if energies_list is not None:
        initial_energies = jnp.stack(energies_list, axis=0)
    elif energy_backend is not None and not init.random_initialise_pos:
        e_list = []
        for wi in range(n_live):
            e, _, _ = energy_backend(initial_positions[wi], initial_types, initial_cells[wi], 0)
            e_list.append(e)
        initial_energies = jnp.stack(e_list, axis=0)
    else:
        initial_energies = None

    return ResolvedInit(
        initial_positions=initial_positions,
        initial_types=initial_types,
        initial_cells=initial_cells,
        initial_energies=initial_energies,
        symbol_map=symbol_map,
    )


def _resolve_init_config_file(
    init: InitConfig,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None,
    cell_cfg: CellConfig,
) -> ResolvedInit:
    """Mode B: initialise from a founder structure file (start_config_file)."""
    positions_single, types_single, cell_single, symbol_map = load_structure(
        init.start_config_file
    )
    n_atoms = positions_single.shape[0]

    key = jax.random.key(seed)
    key, shape_key, pos_key = jax.random.split(key, 3)

    initial_cells = _build_cells(init, n_live, shape_key, cell_single, n_atoms, cell_cfg)
    _validate_cells(initial_cells, n_atoms, cell_cfg)

    if not init.random_initialise_pos:
        logger.warning(
            "start_config_file with random_initialise_pos=False: all %d walkers "
            "start with identical positions. They are fully correlated; enable "
            "burn-in (InitialWalkConfig.n_walks > 0) or set random_initialise_pos=True.",
            n_live,
        )
        initial_positions = jnp.broadcast_to(
            positions_single[None], (n_live, n_atoms, 3)
        )
        if energy_backend is not None:
            e_list = []
            for wi in range(n_live):
                e, _, _ = energy_backend(
                    initial_positions[wi], types_single, initial_cells[wi], 0
                )
                e_list.append(e)
            initial_energies: jnp.ndarray | None = jnp.stack(e_list, axis=0)
        else:
            initial_energies = None
    else:
        initial_positions, energies_list = _sample_per_walker_positions(
            init, n_live, pos_key, initial_cells, types_single, n_atoms, energy_backend
        )
        initial_energies = jnp.stack(energies_list, axis=0) if energies_list is not None else None

    return ResolvedInit(
        initial_positions=initial_positions,
        initial_types=types_single,
        initial_cells=initial_cells,
        initial_energies=initial_energies,
        symbol_map=symbol_map,
    )


def _null_energy_fn(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    cell: jnp.ndarray,
    max_neighbors: int,
) -> tuple[jnp.ndarray, int, bool]:
    """Placeholder energy function returning zero, for grid-mode without a backend."""
    return jnp.float32(0.0), 0, False


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
    move_descriptors: tuple[MoveKernel, ...]
    backend: BackendConfig
    energy_backend: EnergyBackend
    output: OutputConfig
    termination: tuple[TerminationCriterion, ...]
    adaptation_policies: tuple[ResolvedAdaptationPolicy, ...]
    init: ResolvedInit
    cell: CellConfig
    cohort_index: int = 0
    ensemble_params: dict = field(default_factory=dict)
    initial_walk_config: Any = None
    adaptation_cfg: Any = None


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
        n_extra=root.run.n_extra,
        n_cull=root.run.n_cull,
        seed=seed,
        pressure=pressure,
    )

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
        log_level=root.output.log_level,
    )

    _warn_unused_output_fields(root.output)

    if root.termination is not None:
        termination = tuple(
            spec.to_criterion(n_live=ns.n_live, n_cull=ns.n_cull)
            for spec in root.termination
        )
    else:
        termination = (
            IterationTermination(ns.max_iterations),
            PriorMassTermination(ns.n_live, ns.convergence_threshold),
        )

    adaptation_policies = tuple(
        root.adaptation.resolve_for(m._effective_name())
        for m in root.moves
    )

    resolved_init = _resolve_init(
        root.init,
        n_live=ns.n_live,
        seed=seed,
        energy_backend=energy_backend,
        cell_cfg=root.cell,
    )

    # Derive n_atoms from the resolved initial positions rather than from a
    # config field — this is the single canonical source of truth.
    n_atoms = int(resolved_init.initial_positions.shape[-2])

    moves = tuple(m.to_move_config() for m in root.moves)
    move_descriptors = tuple(
        m.to_descriptor(n_atoms=n_atoms, cell_cfg=root.cell)
        for m in root.moves
    )

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
        initial_walk_config=root.init.initial_walk,
        adaptation_cfg=root.adaptation,
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

    if root.init.restart_file is not None and n > 1:
        raise ValueError(
            "restart_file is only supported for single NS runs; cohort size is "
            f"{n}. Remove ensemble.pressure list / run.seed list, or "
            "remove restart_file."
        )

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
