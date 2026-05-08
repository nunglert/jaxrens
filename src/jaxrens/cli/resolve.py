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
from jaxrens.sampling.batch_descriptor import BatchDescriptor, PmapVmapRuns
from jaxrens.cli.schema.adaptation import AdaptationSpec, ResolvedAdaptationPolicy
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
from jaxrens.cli.schema.init import InitSpec
from jaxrens.cli.schema.root import RootSpec
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
from jaxrens.utils.cell import get_volume, min_aspect_ratio

logger = logging.getLogger(__name__)

# Output fields that are accepted by OutputSpec but not yet consumed by the
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
    initial_max_neighbor_counts: Any = None  # shape: (n_live,) int32 or None
    symbol_map: dict[int, str] | None = None
    restart_state: RestartBundle | None = None


def _build_cells(
    init: InitSpec,
    n_live: int,
    shape_key: jax.Array,
    base_cell: jnp.ndarray,
    n_atoms: int,
    cell_cfg: CellSpec,
) -> jnp.ndarray:
    """Produce (n_live, 3, 3) cells from a base cell.

    If ``init.random_initialise_cell``, run cell_shape_walk per walker;
    otherwise broadcast ``base_cell`` across all walkers.
    """
    _CELL_SHAPE_EQUIL_STEPS = 50

    if init.random_initialise_cell:
        logger.info(
            "[resolve] cell-shape random walk: n_walkers=%d, n_steps=%d",
            n_live, _CELL_SHAPE_EQUIL_STEPS,
        )
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


def _describe_cell_violation(
    cell: jnp.ndarray,
    n_atoms: int,
    cell_cfg: CellSpec,
) -> str | None:
    """Return a human-readable reason why ``cell`` fails ``check_cell_shape``.

    Returns ``None`` if the cell satisfies all constraints. The checks mirror
    ``check_cell_shape`` exactly: min/max volume per atom and min aspect ratio.
    """
    volume = float(get_volume(cell))
    vol_per_atom = volume / n_atoms
    if vol_per_atom < cell_cfg.min_volume_per_atom:
        return (
            f"volume/atom = {vol_per_atom:.4f} A^3 is below "
            f"cell.min_volume_per_atom = {cell_cfg.min_volume_per_atom}"
        )
    if vol_per_atom > cell_cfg.max_volume_per_atom:
        return (
            f"volume/atom = {vol_per_atom:.4f} A^3 exceeds "
            f"cell.max_volume_per_atom = {cell_cfg.max_volume_per_atom}"
        )
    aspect = float(min_aspect_ratio(cell, jnp.asarray(volume)))
    if aspect < cell_cfg.min_aspect_ratio:
        return (
            f"min aspect ratio = {aspect:.4f} is below "
            f"cell.min_aspect_ratio = {cell_cfg.min_aspect_ratio}"
        )
    return None


def _validate_input_cell(
    cell: jnp.ndarray,
    n_atoms: int,
    cell_cfg: CellSpec,
    source: str,
) -> None:
    """Reject a user-provided base cell that already violates ``cell_cfg``.

    ``cell_shape_walk`` is volume-preserving, so a volume-violating input
    cannot be rescued by init-time equilibration — the failure would only
    surface later in ``_validate_cells`` with a misleading "Walker produced
    an invalid cell" message. Fail fast with a message pointing at the input.
    """
    reason = _describe_cell_violation(cell, n_atoms, cell_cfg)
    if reason is None:
        return
    raise RuntimeError(
        f"Input structure {source!r} has a cell that violates the configured "
        f"cell bounds: {reason}. Init-time cell_shape_walk is volume-preserving "
        f"and cannot fix volume violations; adjust the cell.* bounds in the "
        f"config or provide a structure whose cell satisfies them.\n"
        f"Cell:\n{cell}"
    )


def _validate_cells(
    cells: jnp.ndarray,
    n_atoms: int,
    cell_cfg: CellSpec,
) -> None:
    """Raise ``RuntimeError`` if any walker cell fails check_cell_shape."""
    n_live = cells.shape[0]
    for wi in range(n_live):
        reason = _describe_cell_violation(cells[wi], n_atoms, cell_cfg)
        if reason is not None:
            raise RuntimeError(
                f"Walker {wi} produced an invalid cell: {reason}.\n"
                f"Cell:\n{cells[wi]}"
            )


def _finalise_initial_energies_and_counts(
    energy_backend: EnergyBackend | None,
    positions: jnp.ndarray,
    types: jnp.ndarray,
    cells: jnp.ndarray,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    """Compute per-walker initial ``(energies, max_neighbor_counts)``.

    For backends that expose ``max_neighbors_for`` (currently MACE), this
    computes each walker's true max neighbor count from geometry alone and
    then evaluates the backend once per walker with ``max_neighbors`` sized
    to the global max.  That avoids the degenerate case where the resolver
    previously passed ``max_neighbors=0`` — causing MACE to run on an
    edge-less graph and return isolated-atom energies — and gives the NS
    loop a correctly-sized starting bucket plus accurate per-walker
    ``max_neighbor_count`` without a separate config parameter.

    For backends without ``max_neighbors_for`` (LJ, toy), the bucket is
    irrelevant: ``max_neighbors=0`` is passed through and counts returned
    as ``None``.

    Returns ``(energies, counts)``; either or both may be ``None``.
    """
    if energy_backend is None:
        return None, None

    if types.ndim == 1:
        types_b = jnp.broadcast_to(types[None, :], (positions.shape[0],) + types.shape)
    else:
        types_b = types

    backend_label = type(energy_backend).__name__
    if hasattr(energy_backend, "max_neighbors_for"):
        logger.info(
            "[resolve] computing per-walker initial neighbor counts and energies "
            "(%s, n_walkers=%d, n_atoms=%d)",
            backend_label, positions.shape[0], positions.shape[1],
        )
        counts = jax.vmap(energy_backend.max_neighbors_for)(positions, cells)
        init_bucket = int(jnp.max(counts))
        logger.info(
            "[resolve] initial neighbor counts: max=%d → init bucket=%d; "
            "evaluating backend energies",
            init_bucket, init_bucket,
        )
        energies = jax.vmap(
            lambda p, t, c: energy_backend(p, t, c, init_bucket)[0]
        )(positions, types_b, cells)
        return energies, counts

    logger.info(
        "[resolve] computing initial energies (%s, n_walkers=%d, n_atoms=%d)",
        backend_label, positions.shape[0], positions.shape[1],
    )
    energies = jax.vmap(
        lambda p, t, c: energy_backend(p, t, c, 0)[0]
    )(positions, types_b, cells)
    return energies, None


def _sample_per_walker_positions(
    init: InitSpec,
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
    logger.info(
        "[resolve] sampling per-walker positions: n_walkers=%d, n_atoms=%d, "
        "mode=%s, max_tries=%d",
        n_live, n_atoms, init.pos_randomization_mode, init.random_init_max_n_tries,
    )
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
    init: InitSpec,
    cell_cfg: CellSpec,
    n_live: int,
    energy_backend: EnergyBackend | None,
) -> ResolvedInit:
    """Mode C: load a pre-computed set of N walker configurations from disk."""
    logger.info(
        "[resolve] init mode C: loading walker set from %s (n_live=%d)",
        init.start_walker_set, n_live,
    )
    if init.random_initialise_pos or init.random_initialise_cell:
        logger.warning(
            "start_walker_set: ignoring random_initialise_pos/cell=True. "
            "Walker-set files are taken verbatim; no randomization applied."
        )

    walker_set = load_walker_set(Path(init.start_walker_set), n_live)
    n_atoms = walker_set.types.shape[1]

    _validate_cells(walker_set.cells, n_atoms, cell_cfg)

    energies, counts = _finalise_initial_energies_and_counts(
        energy_backend, walker_set.positions, walker_set.types, walker_set.cells,
    )

    return ResolvedInit(
        initial_positions=walker_set.positions,
        initial_types=walker_set.types,
        initial_cells=walker_set.cells,
        initial_energies=energies,
        initial_max_neighbor_counts=counts,
        symbol_map=walker_set.symbol_map,
    )


def _resolve_init_restart(
    init: InitSpec,
    cell_cfg: CellSpec,
    n_live: int,
    energy_backend: EnergyBackend | None,
) -> ResolvedInit:
    """Mode D: resume an NS run from a checkpoint file (restart_file)."""
    logger.info(
        "[resolve] init mode D: restarting from checkpoint %s (n_live=%d)",
        init.restart_file, n_live,
    )
    if init.random_initialise_pos or init.random_initialise_cell:
        logger.warning(
            "restart_file: ignoring random_initialise_pos/cell=True. "
            "Checkpoint files are taken verbatim; no randomization applied."
        )

    walker_set, restart_bundle = load_restart(Path(init.restart_file))
    n_atoms = walker_set.types.shape[1]

    _validate_cells(walker_set.cells, n_atoms, cell_cfg)

    energies, counts = _finalise_initial_energies_and_counts(
        energy_backend, walker_set.positions, walker_set.types, walker_set.cells,
    )

    return ResolvedInit(
        initial_positions=walker_set.positions,
        initial_types=walker_set.types,
        initial_cells=walker_set.cells,
        initial_energies=energies,
        initial_max_neighbor_counts=counts,
        symbol_map=walker_set.symbol_map,
        restart_state=restart_bundle,
    )


def _resolve_init(
    init: InitSpec,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None = None,
    cell_cfg: CellSpec | None = None,
) -> ResolvedInit:
    """Resolve an ``InitSpec`` into concrete initial-state arrays.

    Supports ``start_species`` (Mode A), ``start_config_file`` (Mode B),
    ``start_walker_set`` (Mode C), and ``restart_file`` (Mode D).

    Args:
        init: Validated ``InitSpec``.
        n_live: Number of live walkers (from ``NSConfig.n_live``).
        seed: PRNG seed.
        energy_backend: Backend used to compute initial energies.  When
            ``None``, ``initial_energies`` is left as ``None``.
        cell_cfg: ``CellSpec`` carrying cell-geometry constraints.  When
            ``None`` a default ``CellSpec()`` is used.

    Returns:
        ``ResolvedInit`` with arrays populated for the chosen mode.
    """
    if cell_cfg is None:
        cell_cfg = CellSpec()

    if init.start_walker_set is not None:
        return _resolve_init_walker_set(init, cell_cfg, n_live, energy_backend)

    if init.restart_file is not None:
        return _resolve_init_restart(init, cell_cfg, n_live, energy_backend)

    if init.start_config_file is not None:
        return _resolve_init_config_file(init, n_live, seed, energy_backend, cell_cfg)

    assert init.start_species is not None
    return _resolve_init_species(init, n_live, seed, energy_backend, cell_cfg)


def _resolve_init_species(
    init: InitSpec,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None,
    cell_cfg: CellSpec,
) -> ResolvedInit:
    """Mode A: initialise from a species string (start_species)."""
    species_counts = init.parsed_species()
    assert species_counts is not None
    logger.info(
        "[resolve] init mode A: start_species=%r (n_live=%d, seed=%d)",
        init.start_species, n_live, seed,
    )

    types_list: list[int] = []
    for z, count in sorted(species_counts.items()):
        types_list.extend([z] * count)
    n_atoms = len(types_list)

    unique_z = sorted(set(types_list))
    from ase.data import chemical_symbols

    # Backend-aware species mapping. Model-based backends (MACE, and in the
    # future NeuralIL) expose an ``atomic_numbers`` attribute that defines the
    # order the model expects species indices in (one-hot over the z-table).
    # The default path (LJ, toy) uses contiguous 0-based indices over the
    # sorted unique Z-numbers.
    backend_table: list[int] | None = None
    if energy_backend is not None and hasattr(energy_backend, "atomic_numbers"):
        try:
            backend_table = [int(z) for z in energy_backend.atomic_numbers]
        except (TypeError, ValueError):
            backend_table = None

    if backend_table is not None:
        missing = [z for z in unique_z if z not in backend_table]
        if missing:
            raise ValueError(
                f"start_species references atomic numbers {missing} not in the "
                f"backend z-table (atomic_numbers length {len(backend_table)})."
            )
        z_to_idx = {z: backend_table.index(z) for z in unique_z}
    else:
        z_to_idx = {z: i for i, z in enumerate(unique_z)}

    type_indices = [z_to_idx[z] for z in types_list]
    initial_types = jnp.array(type_indices, dtype=jnp.int32)

    symbol_map: dict[int, str] = {
        z_to_idx[z]: chemical_symbols[z] for z in unique_z
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

    # Finalise initial energies with a correctly sized neighbor bucket.
    # The per-walker values collected during sampling (``energies_list``) were
    # computed with ``max_neighbors=0`` — fine for rejection-sampling ceiling
    # checks but wrong for the MCState's initial energy on GNN backends.
    initial_energies, initial_counts = _finalise_initial_energies_and_counts(
        energy_backend, initial_positions, initial_types, initial_cells,
    )

    return ResolvedInit(
        initial_positions=initial_positions,
        initial_types=initial_types,
        initial_cells=initial_cells,
        initial_energies=initial_energies,
        initial_max_neighbor_counts=initial_counts,
        symbol_map=symbol_map,
    )


def _resolve_init_config_file(
    init: InitSpec,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None,
    cell_cfg: CellSpec,
) -> ResolvedInit:
    """Mode B: initialise from a founder structure file (start_config_file)."""
    logger.info(
        "[resolve] init mode B: loading structure from %s (n_live=%d, seed=%d)",
        init.start_config_file, n_live, seed,
    )
    positions_single, types_single, cell_single, symbol_map = load_structure(
        init.start_config_file
    )
    n_atoms = positions_single.shape[0]

    _validate_input_cell(cell_single, n_atoms, cell_cfg, init.start_config_file)

    key = jax.random.key(seed)
    key, shape_key, pos_key = jax.random.split(key, 3)

    initial_cells = _build_cells(init, n_live, shape_key, cell_single, n_atoms, cell_cfg)
    _validate_cells(initial_cells, n_atoms, cell_cfg)

    if not init.random_initialise_pos:
        logger.warning(
            "start_config_file with random_initialise_pos=False: all %d walkers "
            "start with identical positions. They are fully correlated; enable "
            "burn-in (InitialWalkSpec.n_walks > 0) or set random_initialise_pos=True.",
            n_live,
        )
        initial_positions = jnp.broadcast_to(
            positions_single[None], (n_live, n_atoms, 3)
        )
    else:
        initial_positions, _ = _sample_per_walker_positions(
            init, n_live, pos_key, initial_cells, types_single, n_atoms, energy_backend
        )

    initial_energies, initial_counts = _finalise_initial_energies_and_counts(
        energy_backend, initial_positions, types_single, initial_cells,
    )

    return ResolvedInit(
        initial_positions=initial_positions,
        initial_types=types_single,
        initial_cells=initial_cells,
        initial_energies=initial_energies,
        initial_max_neighbor_counts=initial_counts,
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
    """Emit warnings for deferred ``OutputSpec`` fields that are non-default.

    Args:
        output_schema: An ``OutputSpec`` instance.
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
    """Holds all library dataclasses produced by ``resolve``.

    ``base_backend`` is the unwrapped backend produced by
    ``BackendSpec.build_backend()`` (e.g. the raw MACE / NeuralIL / LJ
    instance with no ensemble correction).  The runtime wraps it into an
    ``EnsembleBackend`` if needed; the resolver applies the same wrapping
    locally just to compute initial energies on the right scale, then
    discards the wrapper.
    """

    ns: NSConfig
    moves: tuple[MoveConfig, ...]
    move_descriptors: tuple[MoveKernel, ...]
    backend: BackendConfig
    base_backend: EnergyBackend
    output: OutputConfig
    termination: tuple[TerminationCriterion, ...]
    adaptation_policies: tuple[ResolvedAdaptationPolicy, ...]
    init: ResolvedInit
    cell: CellSpec
    cohort_index: int = 0
    ensemble_params: dict = field(default_factory=dict)
    initial_walk_config: Any = None
    adaptation_cfg: Any = None


def _cohort_size(root: RootSpec) -> int:
    """Return the number of cohort elements implied by the ensemble spec."""
    from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
    if isinstance(root.ensemble, NPTEnsembleSpec):
        return root.ensemble.cohort_size()
    return 1


def _seed_list(root: RootSpec, n: int) -> list[int]:
    """Return a list of *n* seeds derived from ``root.run.seed``.

    If ``run.seed`` is a scalar, generate ``[seed + i for i in range(n)]``.
    """
    return [root.run.seed + i for i in range(n)]


def _resolve_one(root: RootSpec, cohort_index: int = 0) -> ResolvedConfig:
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
    base_backend = root.backend.build_backend()

    # For initial-energy evaluation, locally wrap with EnsembleBackend so the
    # resolver's energies match the NS-loop scale (U + P*V for NPT) — without
    # this, walkers are initialized with bare LJ energies while the MWG
    # step_fn returns ensemble-corrected energies, causing systematic
    # emax < new_energy for all cell moves and 100% rejection from the first
    # adapt call.  Discarded after _resolve_init returns; only ``base_backend``
    # crosses the resolver boundary, leaving wrapping to the runtime.
    if pressure is not None:
        from jaxrens.backends.ensemble import EnsembleBackend
        init_energy_backend = EnsembleBackend(base_backend, pressure=float(pressure))
    else:
        init_energy_backend = base_backend

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

    # Build termination criteria.  ``IterationTermination`` is only added
    # when the user explicitly set ``run.max_iterations`` — when ``None``
    # the loop is bounded only by the user's other criteria (e.g.
    # prior_mass).  Dead-point history is host-side and unbounded by JAX
    # static-shape constraints, so there is no separate storage cap.
    if root.termination is not None:
        termination = tuple(
            spec.to_criterion(n_live=ns.n_live, n_cull=ns.n_cull)
            for spec in root.termination
        )
    else:
        termination = (
            PriorMassTermination(ns.n_live, ns.convergence_threshold),
        )
    if ns.max_iterations is not None:
        termination = termination + (IterationTermination(ns.max_iterations),)

    adaptation_policies = tuple(
        root.adaptation.resolve_for(m._effective_name())
        for m in root.moves
    )

    resolved_init = _resolve_init(
        root.init,
        n_live=ns.n_live,
        seed=seed,
        energy_backend=init_energy_backend,
        cell_cfg=root.cell,
    )

    # Derive n_atoms from the resolved initial positions rather than from a
    # config field — this is the single canonical source of truth.
    n_atoms = int(resolved_init.initial_positions.shape[-2])

    import dataclasses as _dc

    moves = tuple(m.to_move_config() for m in root.moves)
    move_descriptors = tuple(
        _dc.replace(
            m.to_descriptor(n_atoms=n_atoms, cell_cfg=root.cell),
            min_rate=policy.min_rate,
            max_rate=policy.max_rate,
            step_size_max=policy.step_size_max,
        )
        for m, policy in zip(root.moves, adaptation_policies)
    )

    return ResolvedConfig(
        ns=ns,
        moves=moves,
        move_descriptors=move_descriptors,
        backend=backend,
        base_backend=base_backend,
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


def expand_cohort(root: RootSpec) -> list[ResolvedConfig]:
    """Expand a ``RootSpec`` into one ``ResolvedConfig`` per cohort element.

    Cohort axes are defined by the ensemble spec (e.g. a list of pressures in
    ``NPTEnsembleSpec``).  Scalar specs produce a single-element cohort.

    Alignment rule: all list-valued axes must have the same length, or be
    scalars (broadcast to the common length).  Mismatched lengths raise
    ``ValueError`` with a clear message.

    Seed handling: ``run.seed`` is a scalar; each cohort element receives
    ``seed + cohort_index`` so runs are distinct but deterministically
    reproducible from the base seed.

    Args:
        root: Fully validated ``RootSpec``.

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


def resolve(root: RootSpec) -> ResolvedConfig:
    """Translate a validated ``RootSpec`` into library dataclasses.

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


# ---------------------------------------------------------------------------
# Multi-run resolution (multi-GPU pressure-RENS, XRENS, semi_grand)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedMultiRunConfig:
    """Resolved library dataclasses for a multi-run (multi-GPU) NS dispatch.

    Produced by :func:`_resolve_multi_run` when the YAML implies more than one
    replica (e.g. ``ensemble.pressure`` is a list or an inter-RE flavor has a
    replica-axis list).  Consumed by ``run_multi_gpu_from_config`` in
    ``cli/run.py``.

    Shape convention: ``n_total = n_gpu * n_per_gpu`` replicas, with arrays
    flat along the replica axis (``(n_total, n_walkers, ...)``).  The target
    dispatcher reshapes to ``(n_gpu, n_per_gpu, n_walkers, ...)``.
    """

    ns: NSConfig
    moves: tuple[MoveConfig, ...]
    move_descriptors: tuple[MoveKernel, ...]
    backend: BackendConfig
    # Unwrapped base backend; EnsembleBackend wrapping happens once inside
    # run_multi_gpu_from_config with a default pressure overridden per-call
    # via ensemble_params_per_run.
    base_backend: EnergyBackend
    output: OutputConfig
    termination: tuple[TerminationCriterion, ...]
    adaptation_policies: tuple[ResolvedAdaptationPolicy, ...]
    init: ResolvedInit
    cell: CellSpec
    # Per-replica ensemble_params, flat list of length n_total.  Ordering is
    # ``flat_idx = g * n_per_gpu + p`` — matches ``init_ns_multi_gpu``.
    ensemble_params_per_run: tuple[dict, ...]
    initial_walk_config: Any = None
    adaptation_cfg: Any = None
    inter_re_config: Any = None  # InterREConfig | None
    # Batcher describing the (n_gpu, n_per_gpu) topology.  Carries the same
    # information as ``ns.n_gpu``/``ns.n_per_gpu`` but in the canonical form
    # consumed by ``_run_loop``, ``AdaptationManager``, ``InterREManager``,
    # and (post-§C) ``initial_walk``.
    batcher: BatchDescriptor | None = None


def _local_device_count() -> int:
    """Read jax.local_devices() at resolve time (lazy: module-level jax import)."""
    return len(jax.local_devices())


def _derive_replica_axes(
    root: RootSpec,
) -> tuple[int, int, int, list[dict]]:
    """Compute (n_total, n_gpu, n_per_gpu, ensemble_params_per_run).

    Rules:
        * Single replica-axis list drives n_total: pressure list, composition
          targets, or chemical potentials.
        * If more than one list is present, lengths must agree.
        * ``n_gpu = len(jax.local_devices())``, clamped to ``n_total`` with a
          warning when the device count exceeds replica count.
        * ``n_total % n_gpu == 0`` must hold → ``n_per_gpu = n_total // n_gpu``.
        * Returns an empty list of per-run params if n_total == 1 (caller uses
          single-run path).

    Raises:
        ValueError: On any inconsistency described above.
    """
    # ---- Gather per-replica axis lengths ------------------------------------
    pressure_list: list[float] | None = None
    if isinstance(root.ensemble, NPTEnsembleSpec):
        plist = root.ensemble._pressure_list()
        if len(plist) > 1:
            pressure_list = plist

    comp_targets: list[list[int]] | None = None
    chem_pots: list[list[float]] | None = None
    if root.inter_re is not None:
        if root.inter_re.flavor == "xrens":
            comp_targets = list(root.inter_re.composition_targets or [])
        elif root.inter_re.flavor == "semi_grand":
            chem_pots = list(root.inter_re.chemical_potentials or [])
        elif root.inter_re.flavor == "pressure" and pressure_list is None:
            raise ValueError(
                "inter_re.flavor='pressure' requires a list-valued "
                "ensemble.pressure with at least 2 entries (one per replica)."
            )

    lengths: list[tuple[str, int]] = []
    if pressure_list is not None:
        lengths.append(("ensemble.pressure", len(pressure_list)))
    if comp_targets:
        lengths.append(("inter_re.composition_targets", len(comp_targets)))
    if chem_pots:
        lengths.append(("inter_re.chemical_potentials", len(chem_pots)))

    if not lengths:
        # Single replica.  No per-run params.
        return 1, 1, 1, []

    # All replica-axis lists must have the same length.
    n_total = lengths[0][1]
    for name, n in lengths[1:]:
        if n != n_total:
            raise ValueError(
                f"Multi-run axis length mismatch: {lengths[0][0]}={n_total} "
                f"vs {name}={n}."
            )

    # ---- Device topology ----------------------------------------------------
    n_detected = _local_device_count()
    if n_detected > n_total:
        logger.warning(
            "jax.local_devices() reports %d device(s) but only %d replicas "
            "requested; clamping n_gpu=%d (extra devices sit idle).",
            n_detected, n_total, n_total,
        )
        n_gpu = n_total
    else:
        n_gpu = max(1, n_detected)
    if n_total % n_gpu != 0:
        raise ValueError(
            f"Multi-run replica count ({n_total}) is not divisible by the "
            f"detected device count ({n_gpu}). Either adjust the replica "
            f"list length or the SLURM --gres=gpu:N allocation so that "
            f"n_total % n_gpu == 0."
        )
    n_per_gpu = n_total // n_gpu

    # ---- Build per-replica ensemble_params dicts ----------------------------
    # Pressures (scalar fallback: if pressure is scalar, broadcast).
    if pressure_list is not None:
        pressures = pressure_list
    elif isinstance(root.ensemble, NPTEnsembleSpec):
        # Scalar pressure broadcast to all replicas.
        pressures = root.ensemble._pressure_list() * n_total
    else:
        pressures = None

    # For list-valued pressure, honour pressure_units from the spec.
    if pressures is not None and isinstance(root.ensemble, NPTEnsembleSpec):
        if root.ensemble.pressure_units == "gpa":
            _GPA_TO_EVA3 = 0.006241509
            pressures = [p * _GPA_TO_EVA3 for p in pressures]

    params_per_run: list[dict] = []
    for r in range(n_total):
        params: dict = {}
        if pressures is not None:
            params["pressure"] = float(pressures[r])
        if comp_targets:
            params["target_composition"] = jnp.asarray(comp_targets[r], dtype=jnp.int32)
        if chem_pots:
            params["chemical_potentials"] = jnp.asarray(chem_pots[r], dtype=jnp.float32)
        params_per_run.append(params)

    return n_total, n_gpu, n_per_gpu, params_per_run


def _resolve_multi_run(root: RootSpec) -> ResolvedMultiRunConfig:
    """Resolve ``root`` into a ``ResolvedMultiRunConfig``.

    Builds per-replica initial positions / cells / energies by calling
    ``_resolve_init`` once per replica with its own seed and EnsembleBackend
    (so initial energies already include the replica's P·V term).  Stacks the
    per-replica arrays along axis 0 to produce ``(n_total, K, ...)`` pytrees.

    This intentionally mirrors the single-run :func:`_resolve_one` — the two
    paths diverge only in the init loop.
    """
    n_total, n_gpu, n_per_gpu, params_per_run = _derive_replica_axes(root)
    if n_total < 2:
        raise ValueError(
            "_resolve_multi_run called without a multi-replica axis — "
            "use expand_cohort() for single-run configs instead."
        )
    logger.info(
        "[resolve] multi-run topology: n_total=%d replicas across n_gpu=%d "
        "device(s), n_per_gpu=%d", n_total, n_gpu, n_per_gpu,
    )

    # Base (unwrapped) backend — the multi-GPU dispatch wraps it once.
    logger.info("[resolve] building base backend (%s)", root.backend.__class__.__name__)
    base_backend = root.backend.build_backend()
    backend_cfg = root.backend.to_backend_config()

    # Build a per-replica EnsembleBackend for initial-energy evaluation.
    from jaxrens.backends.ensemble import EnsembleBackend

    per_run_init: list[ResolvedInit] = []
    for r in range(n_total):
        p = params_per_run[r].get("pressure", None)
        per_run_backend = (
            EnsembleBackend(base_backend, pressure=float(p))
            if p is not None
            else base_backend
        )
        # Seed policy mirrors _resolve_one's `seed + cohort_index`.
        seed_r = root.run.seed + r
        logger.info(
            "[resolve] replica %d/%d: seed=%d%s",
            r + 1, n_total, seed_r,
            f", pressure={p:.4g}" if p is not None else "",
        )
        init_r = _resolve_init(
            root.init,
            n_live=root.run.n_live,
            seed=seed_r,
            energy_backend=per_run_backend,
            cell_cfg=root.cell,
        )
        per_run_init.append(init_r)

    # Validate shapes and stack along axis 0.
    ref = per_run_init[0]
    for r, init_r in enumerate(per_run_init):
        for field_name in ("initial_positions", "initial_types", "initial_cells", "initial_energies"):
            a = getattr(ref, field_name)
            b = getattr(init_r, field_name)
            if a is None or b is None:
                if (a is None) != (b is None):
                    raise RuntimeError(
                        f"Per-replica init disagreement at r={r}, field {field_name}: "
                        f"one is None, the other isn't."
                    )
            elif a.shape != b.shape:
                raise RuntimeError(
                    f"Per-replica init shape mismatch at r={r}, field "
                    f"{field_name}: ref={a.shape} vs r={b.shape}."
                )

    initial_positions = jnp.stack([x.initial_positions for x in per_run_init], axis=0)
    initial_cells = (
        jnp.stack([x.initial_cells for x in per_run_init], axis=0)
        if per_run_init[0].initial_cells is not None else None
    )
    initial_energies = (
        jnp.stack([x.initial_energies for x in per_run_init], axis=0)
        if per_run_init[0].initial_energies is not None else None
    )
    initial_max_neighbor_counts = (
        jnp.stack([x.initial_max_neighbor_counts for x in per_run_init], axis=0)
        if per_run_init[0].initial_max_neighbor_counts is not None else None
    )
    # Types are identical across replicas (same start_species).
    initial_types = per_run_init[0].initial_types

    # symbol_map, restart_state from ref (must be identical across replicas).
    symbol_map = per_run_init[0].symbol_map
    stacked_init = ResolvedInit(
        initial_positions=initial_positions,
        initial_types=initial_types,
        initial_cells=initial_cells,
        initial_energies=initial_energies,
        initial_max_neighbor_counts=initial_max_neighbor_counts,
        symbol_map=symbol_map,
        restart_state=None,  # multi-run restart is a follow-up; see plan.
    )

    # ns_config with derived topology fields.
    ns = NSConfig(
        n_live=root.run.n_live,
        max_iterations=root.run.max_iterations,
        convergence_threshold=root.run.convergence_threshold,
        n_mcmc_steps=root.run.n_mcmc_steps,
        n_extra=root.run.n_extra,
        n_cull=root.run.n_cull,
        seed=root.run.seed,
        pressure=None,  # per-replica pressure lives in ensemble_params_per_run.
        inter_re=(
            root.inter_re.to_inter_re_config() if root.inter_re is not None else None
        ),
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
    )

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

    # Same termination logic as the single-run path: default to PriorMass
    # only; append IterationTermination only when max_iterations is set.
    if root.termination is not None:
        termination = tuple(
            spec.to_criterion(n_live=ns.n_live, n_cull=ns.n_cull)
            for spec in root.termination
        )
    else:
        termination = (
            PriorMassTermination(ns.n_live, ns.convergence_threshold),
        )
    if ns.max_iterations is not None:
        termination = termination + (IterationTermination(ns.max_iterations),)

    adaptation_policies = tuple(
        root.adaptation.resolve_for(m._effective_name())
        for m in root.moves
    )

    n_atoms = int(stacked_init.initial_positions.shape[-2])
    import dataclasses as _dc

    moves = tuple(m.to_move_config() for m in root.moves)
    move_descriptors = tuple(
        _dc.replace(
            m.to_descriptor(n_atoms=n_atoms, cell_cfg=root.cell),
            min_rate=policy.min_rate,
            max_rate=policy.max_rate,
            step_size_max=policy.step_size_max,
        )
        for m, policy in zip(root.moves, adaptation_policies)
    )

    return ResolvedMultiRunConfig(
        ns=ns,
        moves=moves,
        move_descriptors=move_descriptors,
        backend=backend_cfg,
        base_backend=base_backend,
        output=output,
        termination=termination,
        adaptation_policies=adaptation_policies,
        init=stacked_init,
        cell=root.cell,
        ensemble_params_per_run=tuple(params_per_run),
        initial_walk_config=root.init.initial_walk,
        adaptation_cfg=root.adaptation,
        inter_re_config=(
            root.inter_re.to_inter_re_config() if root.inter_re is not None else None
        ),
        batcher=PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu),
    )


def expand_multi_run_or_cohort(
    root: RootSpec,
) -> list[ResolvedConfig] | ResolvedMultiRunConfig:
    """Dispatch between multi-run and single-run (cohort) resolution.

    Returns a :class:`ResolvedMultiRunConfig` when the YAML implies more than
    one replica via a replica-axis list (pressure list, composition_targets,
    chemical_potentials).  Otherwise returns a list of :class:`ResolvedConfig`
    from :func:`expand_cohort` — same output shape as before (single-element
    list for scalar configs, multi-element for cohort sweeps).
    """
    # Cheap pre-check to avoid building per-run init when we don't need it.
    has_multi_axis = False
    if isinstance(root.ensemble, NPTEnsembleSpec):
        if len(root.ensemble._pressure_list()) > 1:
            has_multi_axis = True
    if root.inter_re is not None:
        if (root.inter_re.composition_targets and len(root.inter_re.composition_targets) > 1) or (
            root.inter_re.chemical_potentials and len(root.inter_re.chemical_potentials) > 1
        ):
            has_multi_axis = True
        # Pressure-RENS with scalar pressure gets caught by _derive_replica_axes.

    if has_multi_axis:
        return _resolve_multi_run(root)
    return expand_cohort(root)
