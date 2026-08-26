"""Resolution layer: pydantic schemas -> library dataclasses.

``resolve`` is the single seam between the CLI config layer and the
library core.  It always returns one :class:`ResolvedConfig` carrying a
``batcher`` field that picks the execution topology:

* ``SingleRun()`` when the YAML implies a single NS run (scalar
  ``ensemble.pressure``, no inter-RE replica-axis list).
* ``PmapVmapRuns(n_gpu, n_per_gpu)`` when the YAML implies multiple
  replicas (list-valued pressure, ``inter_re.composition_targets``,
  ``inter_re.chemical_potentials``).

There is no longer a separate "cohort" (independent sequential) path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op
from jaxrens.backends.base import BackendResult, EnergyBackend
from jaxrens.cli.schema.adaptation import (
    AdaptationSpec,
    ResolvedAdaptationPolicy,
)
from jaxrens.cli.schema.backend import LJBackendSpec
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.ensemble import SemiGrandEnsembleSpec
from jaxrens.cli.schema.init import InitSpec
from jaxrens.cli.schema.root import RootSpec
from jaxrens.cli.schema.termination import IterationTerminationSpec
from jaxrens.constraints.base import ConstraintDescriptor
from jaxrens.init.cells import cell_shape_walk, sample_initial_volume
from jaxrens.init.positions import (
    grid_positions_in_cell,
    uniform_positions_in_cell,
)
from jaxrens.init.rejection import rejection_sample_positions
from jaxrens.init.restart import (
    BatchedRestart,
    RestartBundle,
    load_restart,
    load_restart_batched,
)
from jaxrens.init.structure import load_structure
from jaxrens.init.walker_set import load_walker_set
from jaxrens.sampling.batch_descriptor import (
    BatchDescriptor,
    PmapVmapRuns,
    SingleRun,
)
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.nested_sampling import _choose_starting_bucket
from jaxrens.sampling.termination import (
    IterationTermination,
    PriorMassTermination,
    TerminationCriterion,
)
from jaxrens.state.config import (
    BackendConfig,
    MoveConfig,
    NSConfig,
    OutputConfig,
)
from jaxrens.utils.cell import get_volume, min_aspect_ratio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval-unit scaling (RootSpec.interval_units = "absolute" | "per_walker")
# ---------------------------------------------------------------------------


def _scale_interval(v: int | float | None, *, factor: int) -> int | None:
    """Scale one iteration-counted field for the resolver.

    * ``None`` (e.g. unset ``run.max_iterations``) passes through unchanged.
    * Numeric values are multiplied by ``factor`` and cast to int via
      ``round``; the result is clamped to ``>= 1`` so that a per-walker
      ``snapshot_interval: 0.001`` does not collapse to zero.
    """
    if v is None:
        return None
    scaled = round(float(v) * factor)
    return max(1, int(scaled))


def _apply_interval_units(root: RootSpec) -> RootSpec:
    """Return a new RootSpec with the 8 interval fields scaled to absolute iters.

    When ``root.interval_units == "absolute"`` this is a no-op apart from
    rounding any float values down to int (so the downstream runtime
    dataclasses always see ints).  When ``"per_walker"`` every affected field
    is multiplied by ``root.run.n_live`` first.

    The scaled fields:
        output.{info,traj,snapshot,checkpoint}_interval
        output.{temperature_lag_interval,temperature_interval}
        output.{acc_rates_interval,max_neighbors_interval}
        output.collision_check_interval
        output.flush_interval
        run.max_iterations  (None preserved)
        termination[iteration].max_iterations
        inter_re.re_interval
        adaptation.adjust_interval
    """
    factor = root.run.n_live if root.interval_units == "per_walker" else 1

    output_upd = {
        name: _scale_interval(getattr(root.output, name), factor=factor)
        for name in (
            "info_interval",
            "traj_interval",
            "snapshot_interval",
            "checkpoint_interval",
            "temperature_lag_interval",
            "temperature_interval",
            "acc_rates_interval",
            "max_neighbors_interval",
            "collision_check_interval",
            "flush_interval",
        )
    }
    run_upd = {
        "max_iterations": _scale_interval(
            root.run.max_iterations, factor=factor
        ),
    }
    adaptation_upd = {
        "adjust_interval": _scale_interval(
            root.adaptation.adjust_interval,
            factor=factor,
        ),
    }

    update: dict[str, Any] = {
        "output": root.output.model_copy(update=output_upd),
        "run": root.run.model_copy(update=run_upd),
        "adaptation": root.adaptation.model_copy(update=adaptation_upd),
    }

    if root.inter_re is not None:
        update["inter_re"] = root.inter_re.model_copy(
            update={
                "re_interval": _scale_interval(
                    root.inter_re.re_interval, factor=factor
                )
            },
        )

    if root.termination is not None:
        update["termination"] = [
            (
                t.model_copy(
                    update={
                        "max_iterations": _scale_interval(
                            t.max_iterations,
                            factor=factor,
                        ),
                    },
                )
                if isinstance(t, IterationTerminationSpec)
                else t
            )
            for t in root.termination
        ]

    return root.model_copy(update=update)


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
    initial_types: Any  # shape: (n_live, n_atoms) dtype int32 or None
    initial_cells: Any  # shape: (n_live, 3, 3) or None
    initial_energies: Any  # shape: (n_live,) or None — None until evaluated
    initial_max_neighbor_counts: Any = None  # shape: (n_live,) int32 or None
    symbol_map: dict[int, str] | None = None
    # ``RestartBundle`` for single-replica restart; ``list[list[RestartBundle]]``
    # of shape ``(n_gpu, n_per_gpu)`` for multi-replica restart; ``None`` when
    # starting fresh.  Consumer (``cli/run.py``) dispatches by isinstance.
    restart_state: RestartBundle | list[list[RestartBundle]] | None = None


def _check_initial_constraints(
    descriptors: tuple[ConstraintDescriptor, ...],
    resolved_init: ResolvedInit,
) -> None:
    """Fail fast if any initial walker violates a configuration constraint.

    The MWG constraint gate assumes every walker entering a step already
    satisfies the constraints (a move can then only be *blamed* for a new
    violation). That invariant is established here: a starting configuration
    that is already illegal is a user error, reported before the run begins.
    """
    positions = jnp.asarray(resolved_init.initial_positions)
    types = jnp.asarray(resolved_init.initial_types)
    # The multi-replica path stacks walkers as ``(n_total, K, n_atoms, 3)``
    # while the single-replica path is already ``(n_live, n_atoms, 3)``.
    # Collapse any leading replica/live axes into one flat walker axis so a
    # single vmap feeds the predicate per-walker ``(n_atoms, 3)`` / ``(3, 3)``
    # arrays in both layouts (otherwise the cell stays batched and
    # ``pairwise_distances`` fails to broadcast).  ``types`` carries the same
    # leading axes (batched ``(n_total, K, n_atoms)`` / ``(n_live, n_atoms)``
    # since the ResolvedInit types contract became per-walker) and must be
    # flattened the same way, or vmap sees mismatched mapped-axis sizes.
    n_atoms = positions.shape[-2]
    positions = positions.reshape(-1, n_atoms, 3)
    types = types.reshape(-1, n_atoms)
    n_walkers = int(positions.shape[0])
    cells = resolved_init.initial_cells
    cells = (
        jnp.zeros((n_walkers, 3, 3))
        if cells is None
        else jnp.asarray(cells).reshape(n_walkers, 3, 3)
    )

    for desc in descriptors:
        predicate = desc.build(**desc.build_kwargs)
        valid = jax.vmap(lambda p, t, c: predicate(p, t, c))(
            positions, types, cells
        )
        valid = np.asarray(valid)
        if not valid.all():
            bad = int(np.argmax(~valid))
            raise ValueError(
                f"Initial walker {bad} of {n_walkers} violates the "
                f"{desc.name!r} configuration constraint. Adjust the initial "
                f"configuration (e.g. start_species / start_config_file) or "
                f"relax the constraint before running."
            )


def _resolve_constraints(
    root: RootSpec, resolved_init: ResolvedInit
) -> tuple[ConstraintDescriptor, ...]:
    """Build constraint descriptors from the config and validate initial state.

    Returns an empty tuple when no constraints are configured (the common
    case — zero overhead downstream).
    """
    if not root.constraints:
        return ()
    if resolved_init.symbol_map is None:
        raise ValueError(
            "Configuration constraints require a resolved species map, but "
            "none is available for this init mode. This is a jaxrens "
            "limitation — please report it."
        )
    descriptors = tuple(
        c.to_descriptor(symbol_map=resolved_init.symbol_map)
        for c in root.constraints
    )
    _check_initial_constraints(descriptors, resolved_init)
    return descriptors


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

    if init.cell_randomization_mode == "linear_1d":
        # 1-D embedding: the cell is diag(a, 1, 1), so det(cell) == a and the
        # standard NPT term P*V is the paper's P*a.  A 3-D shape walk would
        # destroy that structure on the first shear.  Draw one box length per
        # walker so the population starts spread out in a.
        a_lo = cell_cfg.effective_initial_min_volume_per_atom * n_atoms
        a_hi = cell_cfg.max_volume_per_atom * n_atoms
        logger.info(
            "[resolve] 1-D cell init: n_walkers=%d, a in [%.4g, %.4g]",
            n_live,
            a_lo,
            a_hi,
        )
        a = jax.random.uniform(
            shape_key, shape=(n_live,), minval=a_lo, maxval=a_hi
        )
        cells = jnp.broadcast_to(jnp.eye(3), (n_live, 3, 3))
        return cells.at[:, 0, 0].set(a)

    if init.random_initialise_cell:
        logger.info(
            "[resolve] cell-shape random walk: n_walkers=%d, n_steps=%d",
            n_live,
            _CELL_SHAPE_EQUIL_STEPS,
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


def _warn_if_lj_cutoff_unsafe(
    backend_spec: Any,
    cell_cfg: CellSpec,
    n_atoms: int,
) -> None:
    """Warn if the LJ cutoff cannot be honoured by the smallest legal cell.

    Triggers only for the LJ backend with a finite cutoff *and* periodic
    images in play. The smallest cell permitted by the prior is the one at
    ``min_volume_per_atom`` with the worst-case aspect ratio. For an
    isotropic cubic cell at that volume, the perpendicular distance equals
    the cube side; allowing the aspect ratio to drop to ``min_aspect_ratio``
    shrinks this proportionally. The MIC + image sum is correct iff
    ``perp · min(supercell_trafo) >= 2 · cutoff``.
    """
    if not isinstance(backend_spec, LJBackendSpec):
        return
    if not backend_spec.periodic:
        # No periodic images -- the cell only bounds where atoms are drawn
        # from, so the cutoff/cell-prior/supercell_trafo interaction this
        # check is about does not exist.  Without this the warning fired
        # (misleadingly) even for a cluster in vacuum.
        return
    if backend_spec.cutoff is None:
        return

    sc_min = min(backend_spec.supercell_trafo)
    if sc_min <= 0:
        return

    vmin = cell_cfg.min_volume_per_atom * n_atoms
    cubic_side = vmin ** (1.0 / 3.0)
    # Worst-case perpendicular distance under the prior: shrinks linearly
    # with min_aspect_ratio relative to the equal-axis cubic shape.
    worst_perp = cubic_side * cell_cfg.min_aspect_ratio
    required = 2.0 * backend_spec.cutoff

    if worst_perp * sc_min < required:
        logger.warning(
            "LJ cutoff vs cell-prior bounds: smallest legal cell has worst-case "
            "perpendicular distance %.4f A (n_atoms=%d, min_volume_per_atom=%.4f, "
            "min_aspect_ratio=%.4f); with supercell_trafo=%s the effective span "
            "is %.4f A, below the required 2 * cutoff = %.4f A. LJ energies will "
            "undercount neighbours on the tight end of the cell-prior range. "
            "Mitigate by raising cell.min_volume_per_atom, raising "
            "cell.min_aspect_ratio, lowering backend.cutoff, or bumping "
            "backend.supercell_trafo.",
            worst_perp,
            n_atoms,
            cell_cfg.min_volume_per_atom,
            cell_cfg.min_aspect_ratio,
            backend_spec.supercell_trafo,
            worst_perp * sc_min,
            required,
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


def _compute_initial_energies(
    energy_backend: EnergyBackend,
    types: jnp.ndarray,
    batcher: BatchDescriptor,
    positions: jnp.ndarray,
    cells: jnp.ndarray,
    bucket: int,
    ensemble_params_batched: Any | None,
) -> jnp.ndarray:
    """Batched initial-energy evaluation at a fixed neighbor ``bucket``.

    Routes the per-replica energy compute through ``batcher.wrap_for_batch``
    (SingleRun / PmapVmapRuns) exactly like the surrounding finalize.

    ``ensemble_params_batched`` is an ensemble-agnostic pytree (e.g.
    ``{"pressure": (*prefix,), "chemical_potentials": (*prefix, n_species)}``)
    whose leaves carry one entry per replica along the batcher's
    ``shape_prefix`` axes.  ``jax.vmap``/``pmap`` map those leaves natively, so
    each replica's per-call ``ensemble_params`` dict flows through to the
    backend and its initial energy reflects that replica's own P·V and −μ·N
    terms.  When ``None`` (no ensemble corrections, or the SingleRun path
    where the backend already closes over its ensemble params) no
    ``ensemble_params`` kwarg is passed.

    ``bucket`` is the static neighbor-list size to compile at — ``0`` for
    backends without ``max_neighbors_for``, else the chosen starting bucket.
    """
    if ensemble_params_batched is None:

        def per_replica(pos_K, types_K, cells_K):
            return jax.vmap(
                lambda p, t, c: energy_backend(p, t, c, bucket).energy
            )(pos_K, types_K, cells_K)

        return batcher.wrap_for_batch(per_replica)(positions, types, cells)

    def per_replica_ep(pos_K, types_K, cells_K, ep):
        return jax.vmap(
            lambda p, t, c: energy_backend(
                p,
                t,
                c,
                bucket,
                ensemble_params=ep,
            )[0]
        )(pos_K, types_K, cells_K)

    return batcher.wrap_for_batch(per_replica_ep)(
        positions, types, cells, ensemble_params_batched
    )


def _stack_ensemble_params(
    params_per_run: tuple[dict, ...] | list[dict],
    keys: tuple[str, ...],
    shape_prefix: tuple[int, ...],
) -> dict | None:
    """Stack per-replica ensemble-param dicts into one batched pytree.

    For each key in ``keys`` present in *every* replica dict, stacks the
    per-replica values along a new leading axis and reshapes to
    ``shape_prefix + leaf_shape`` so the leaves align with the batcher's
    replica axes.  Returns ``None`` if no key qualifies (no ensemble
    corrections to thread).

    Only energy-relevant keys should be passed in ``keys`` — these are the
    same per-replica dicts the runtime NS loop hands to its ``EnsembleBackend``
    via ``ensemble_params_per_run``, so resolved initial energies match the
    runtime by construction.
    """
    batched: dict = {}
    for k in keys:
        if all(k in p for p in params_per_run):
            stacked = jnp.stack(
                [jnp.asarray(p[k]) for p in params_per_run], axis=0
            )
            batched[k] = stacked.reshape(shape_prefix + stacked.shape[1:])
    return batched or None


def _finalise_initial_energies_and_counts(
    energy_backend: EnergyBackend | None,
    positions: jnp.ndarray,
    types: jnp.ndarray,
    cells: jnp.ndarray,
    batcher: BatchDescriptor | None = None,
    ladder: tuple[int, ...] | None = None,
    offset: int = 0,
    ensemble_params_batched: Any | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    """Compute per-walker initial ``(energies, max_neighbor_counts)``.

    Dispatch is routed through ``batcher.wrap_for_batch`` so the
    compute uses ``jax.jit`` / ``jax.jit(vmap)`` / ``pmap(vmap)``
    appropriate for the call site:

    * SingleRun path (``_resolve_single_replica``) → ``SingleRun()``
      (default).  ``positions`` shape ``(K, N, 3)``, ``cells``
      ``(K, 3, 3)``.
    * Multi-replica path (``_resolve_multi_replica``) →
      ``PmapVmapRuns(G, P)`` (the same ``batcher`` instance stored on
      ``ResolvedConfig.batcher``).  ``positions`` shape
      ``(G, P, K, N, 3)``, ``cells`` ``(G, P, K, 3, 3)``.  Energy
      compile happens in parallel across G GPUs and at the same shape
      burn-in + NS step use, so all three stages share one JIT cache
      slot.

    For backends that expose ``max_neighbors_for`` (MACE, NeuralIL),
    per-walker neighbor counts are computed geometry-only.  The energy
    compile is then sized to the bucket
    ``_choose_starting_bucket(counts, ladder, offset)`` so the resolver
    and ``cli/run.py``'s starting-bucket choice agree.  When
    ``ladder``/``offset`` are not supplied, the legacy
    ``int(jnp.max(counts))`` fallback is used.

    For backends without ``max_neighbors_for`` (LJ, toy, jax-md), the
    bucket is irrelevant and ``max_neighbors=0`` is passed through;
    ``counts`` is returned as ``None``.

    ``types`` (shape ``(K, N)`` per replica) is vmap-axis aligned with
    ``positions`` / ``cells``: the per-replica function maps it over the
    walker axis alongside them. Fixed-composition runs carry the same
    row for every walker (the broadcast in ``_resolve_init_*``), but the
    per-walker axis is still present so composition-changing ensembles
    (semi-grand / alchemical) fit the same contract.

    ``ensemble_params_batched`` is an ensemble-agnostic pytree (leaves
    shaped ``batcher.shape_prefix + leaf_shape``) carrying per-replica
    ensemble params — ``pressure`` for NPT, ``chemical_potentials`` for
    semi-grand μPT, or both.  When supplied with an EnsembleBackend, each
    replica's initial energy reflects its own P·V and −μ·N terms, matching
    the runtime NS loop by construction — even though a single
    EnsembleBackend instance handles all replicas in the consolidated
    finalize.  When ``None``, no ensemble correction is threaded (the
    backend's own closured params are used, as on the SingleRun path).

    Returns ``(energies, counts)``; either or both may be ``None``.
    """
    if energy_backend is None:
        return None, None

    if batcher is None:
        batcher = SingleRun()

    backend_label = type(energy_backend).__name__
    shape_prefix = batcher.shape_prefix
    n_walkers_total = (
        int(np.prod(positions.shape[: len(shape_prefix) + 1]))
        if len(shape_prefix) > 0
        else positions.shape[0]
    )
    n_atoms = positions.shape[-2]

    if hasattr(energy_backend, "max_neighbors_for"):
        logger.info(
            "[resolve] computing per-walker initial neighbor counts and energies "
            "(%s, n_walkers*n_runs=%d, n_atoms=%d, batcher=%s)",
            backend_label,
            n_walkers_total,
            n_atoms,
            type(batcher).__name__,
        )

        def per_replica_counts(pos_K, cells_K):
            return jax.vmap(energy_backend.max_neighbors_for)(pos_K, cells_K)

        batched_counts = batcher.wrap_for_batch(per_replica_counts)
        counts = batched_counts(positions, cells)

        if ladder is not None:
            init_bucket = _choose_starting_bucket(
                counts, tuple(ladder), int(offset)
            )
        else:
            init_bucket = int(jnp.max(counts))

        logger.info(
            "[resolve] initial neighbor counts: max=%d → init bucket=%d; "
            "evaluating backend energies",
            int(jnp.max(counts)),
            init_bucket,
        )

        energies = _compute_initial_energies(
            energy_backend,
            types,
            batcher,
            positions,
            cells,
            init_bucket,
            ensemble_params_batched,
        )
        return energies, counts

    logger.info(
        "[resolve] computing initial energies (%s, n_walkers*n_runs=%d, n_atoms=%d, "
        "batcher=%s)",
        backend_label,
        n_walkers_total,
        n_atoms,
        type(batcher).__name__,
    )

    energies = _compute_initial_energies(
        energy_backend,
        types,
        batcher,
        positions,
        cells,
        0,
        ensemble_params_batched,
    )
    return energies, None


def _sample_per_walker_positions(
    init: InitSpec,
    n_live: int,
    pos_key: jax.Array,
    initial_cells: jnp.ndarray,
    initial_types: jnp.ndarray,
    n_atoms: int,
    energy_backend: EnergyBackend | None,
) -> jnp.ndarray:
    """Sample per-walker positions via grid or rejection.

    Returns only positions of shape ``(n_live, n_atoms, 3)``.  Energies
    are recomputed downstream in
    ``_finalise_initial_energies_and_counts`` at the right bucket size,
    so this function does not return them.  For rejection mode the
    energy is consumed internally by
    ``rejection_sample_positions``'s ceiling check; for grid mode no
    energy evaluation is needed at all.
    """
    logger.info(
        "[resolve] sampling per-walker positions: n_walkers=%d, n_atoms=%d, "
        "mode=%s, max_tries=%d",
        n_live,
        n_atoms,
        init.pos_randomization_mode,
        init.random_init_max_n_tries,
    )
    start_energy_ceiling = init.start_energy_ceiling_per_atom * n_atoms
    walker_pos_keys = jax.random.split(pos_key, n_live)
    positions_list = []

    for wi in range(n_live):
        w_cell = initial_cells[wi]

        if init.pos_randomization_mode == "grid":
            pos = grid_positions_in_cell(
                walker_pos_keys[wi], w_cell, n_atoms, init.grid_distance
            )
            positions_list.append(pos)
        else:
            pos, _ = rejection_sample_positions(
                walker_pos_keys[wi],
                cell=w_cell,
                types=initial_types,
                n_atoms=n_atoms,
                energy_fn=energy_backend
                if energy_backend is not None
                else _null_energy_fn,
                start_energy_ceiling=start_energy_ceiling,
                min_distance=init.init_distance_criterion,
                max_tries=init.random_init_max_n_tries,
                mode="uniform",
                grid_distance=init.grid_distance,
            )
            positions_list.append(pos)

    return jnp.stack(positions_list, axis=0)


def _resolve_init_walker_set(
    init: InitSpec,
    cell_cfg: CellSpec,
    n_live: int,
) -> ResolvedInit:
    """Mode C: load a pre-computed set of N walker configurations from disk.

    Returns structural-init only.  ``initial_energies`` and
    ``initial_max_neighbor_counts`` are left as ``None``; the caller
    (``_resolve_single_replica`` / ``_resolve_multi_replica``) finalises them once.
    """
    logger.info(
        "[resolve] init mode C: loading walker set from %s (n_live=%d)",
        init.start_walker_set,
        n_live,
    )
    if init.random_initialise_pos or init.random_initialise_cell:
        logger.warning(
            "start_walker_set: ignoring random_initialise_pos/cell=True. "
            "Walker-set files are taken verbatim; no randomization applied."
        )

    walker_set = load_walker_set(Path(init.start_walker_set), n_live)
    n_atoms = walker_set.types.shape[1]

    _validate_cells(walker_set.cells, n_atoms, cell_cfg)

    return ResolvedInit(
        initial_positions=walker_set.positions,
        initial_types=walker_set.types,
        initial_cells=walker_set.cells,
        initial_energies=None,
        initial_max_neighbor_counts=None,
        symbol_map=walker_set.symbol_map,
    )


def _resolve_init_restart(
    init: InitSpec,
    cell_cfg: CellSpec,
    n_live: int,
) -> ResolvedInit:
    """Mode D: resume an NS run from a checkpoint file (restart_file).

    Returns structural-init only.  ``initial_energies`` and
    ``initial_max_neighbor_counts`` are left as ``None``; the caller
    (``_resolve_single_replica`` / ``_resolve_multi_replica``) finalises them once.
    """
    logger.info(
        "[resolve] init mode D: restarting from checkpoint %s (n_live=%d)",
        init.restart_file,
        n_live,
    )
    if init.random_initialise_pos or init.random_initialise_cell:
        logger.warning(
            "restart_file: ignoring random_initialise_pos/cell=True. "
            "Checkpoint files are taken verbatim; no randomization applied."
        )

    walker_set, restart_bundle = load_restart(Path(init.restart_file))
    n_atoms = walker_set.types.shape[1]

    _validate_cells(walker_set.cells, n_atoms, cell_cfg)

    return ResolvedInit(
        initial_positions=walker_set.positions,
        initial_types=walker_set.types,
        initial_cells=walker_set.cells,
        initial_energies=None,
        initial_max_neighbor_counts=None,
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

    Returns structural-init only — ``initial_energies`` and
    ``initial_max_neighbor_counts`` are left as ``None`` on the returned
    ``ResolvedInit``.  The caller (``_resolve_single_replica`` or
    ``_resolve_multi_replica``) performs a single consolidated finalize via
    ``_finalise_initial_energies_and_counts`` after this returns.

    Args:
        init: Validated ``InitSpec``.
        n_live: Number of live walkers (from ``NSConfig.n_live``).
        seed: PRNG seed.
        energy_backend: Backend used by rejection-mode position placement
            (mode A / mode B) for the internal ceiling check.  Not used
            for energy finalize — that is the caller's responsibility.
        cell_cfg: ``CellSpec`` carrying cell-geometry constraints.  When
            ``None`` a default ``CellSpec()`` is used.

    Returns:
        ``ResolvedInit`` with positions / types / cells / symbol_map /
        (optionally) ``restart_state`` populated; energies and neighbor
        counts left as ``None``.
    """
    if cell_cfg is None:
        cell_cfg = CellSpec()

    if init.restart_file is not None:
        return _resolve_init_restart(init, cell_cfg, n_live)

    if init.start_walker_set is not None:
        return _resolve_init_walker_set(init, cell_cfg, n_live)

    if init.start_config_file is not None:
        return _resolve_init_config_file(
            init,
            n_live,
            seed,
            energy_backend,
            cell_cfg,
        )

    assert init.start_species is not None
    return _resolve_init_species(
        init,
        n_live,
        seed,
        energy_backend,
        cell_cfg,
    )


def _resolve_init_species(
    init: InitSpec,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None,
    cell_cfg: CellSpec,
) -> ResolvedInit:
    """Mode A: initialise from a species string (start_species).

    Returns structural-init only.  ``initial_energies`` and
    ``initial_max_neighbor_counts`` are left as ``None``; the caller
    finalises them once.  ``energy_backend`` is still required for the
    rejection-mode ceiling check inside ``_sample_per_walker_positions``.
    """
    species_counts = init.parsed_species()
    assert species_counts is not None
    logger.info(
        "[resolve] init mode A: start_species=%r (n_live=%d, seed=%d)",
        init.start_species,
        n_live,
        seed,
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
    if energy_backend is not None and hasattr(
        energy_backend, "atomic_numbers"
    ):
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
            min_volume_per_atom=cell_cfg.effective_initial_min_volume_per_atom,
        )
    )
    cubic_cell = jnp.eye(3, dtype=jnp.float32) * lc

    initial_cells = _build_cells(
        init, n_live, shape_key, cubic_cell, n_atoms, cell_cfg
    )
    _validate_cells(initial_cells, n_atoms, cell_cfg)

    if init.pos_autoscale_cells:
        logger.warning(
            "pos_autoscale_cells=True is set but not yet implemented. "
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
            single_pos = uniform_positions_in_cell(
                single_key, single_cell, n_atoms
            )
        initial_positions = jnp.broadcast_to(
            single_pos[None], (n_live, n_atoms, 3)
        )
    else:
        initial_positions = _sample_per_walker_positions(
            init,
            n_live,
            pos_key,
            initial_cells,
            initial_types,
            n_atoms,
            energy_backend,
        )

    initial_types_broadcast = jnp.broadcast_to(
        initial_types[None], (n_live, n_atoms)
    )

    # Energies and neighbor counts are computed once at the correct
    # bucket size by the caller via ``_finalise_initial_energies_and_counts``
    # — this helper returns structural-init only.
    return ResolvedInit(
        initial_positions=initial_positions,
        initial_types=initial_types_broadcast,
        initial_cells=initial_cells,
        initial_energies=None,
        initial_max_neighbor_counts=None,
        symbol_map=symbol_map,
    )


def _resolve_init_config_file(
    init: InitSpec,
    n_live: int,
    seed: int,
    energy_backend: EnergyBackend | None,
    cell_cfg: CellSpec,
) -> ResolvedInit:
    """Mode B: initialise from a founder structure file (start_config_file).

    Returns structural-init only.  ``initial_energies`` and
    ``initial_max_neighbor_counts`` are left as ``None``; the caller
    finalises them once.  ``energy_backend`` is still required for the
    rejection-mode ceiling check inside ``_sample_per_walker_positions``.
    """
    logger.info(
        "[resolve] init mode B: loading structure from %s (n_live=%d, seed=%d)",
        init.start_config_file,
        n_live,
        seed,
    )
    positions_single, types_single, cell_single, symbol_map = load_structure(
        init.start_config_file
    )
    n_atoms = positions_single.shape[0]

    _validate_input_cell(
        cell_single, n_atoms, cell_cfg, init.start_config_file
    )

    key = jax.random.key(seed)
    key, shape_key, pos_key = jax.random.split(key, 3)

    initial_cells = _build_cells(
        init, n_live, shape_key, cell_single, n_atoms, cell_cfg
    )
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
        initial_positions = _sample_per_walker_positions(
            init,
            n_live,
            pos_key,
            initial_cells,
            types_single,
            n_atoms,
            energy_backend,
        )

    initial_types_broadcast = jnp.broadcast_to(
        types_single[None], (n_live, n_atoms)
    )

    return ResolvedInit(
        initial_positions=initial_positions,
        initial_types=initial_types_broadcast,
        initial_cells=initial_cells,
        initial_energies=None,
        initial_max_neighbor_counts=None,
        symbol_map=symbol_map,
    )


def _null_energy_fn(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    cell: jnp.ndarray,
    max_neighbors: int,
) -> BackendResult:
    """Placeholder energy function returning zero, for grid-mode without a backend."""
    return BackendResult(energy=jnp.float32(0.0))


# ---------------------------------------------------------------------------
# ResolvedConfig (unified single-/multi-replica)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """All library dataclasses produced by :func:`resolve`.

    Single-/multi-replica unified type.  Shape conventions:

    * ``batcher == SingleRun()``: arrays in ``init`` carry shape ``(K, ...)``.
      ``ensemble_params_per_run`` is a length-1 tuple.
    * ``batcher == PmapVmapRuns(G, P)``: arrays in ``init`` carry shape
      ``(n_total, K, ...)`` with ``n_total == G * P``.
      ``ensemble_params_per_run`` is a length-``n_total`` tuple in
      ``flat_idx = g * P + p`` order (matches ``init_ns_multi_gpu``).

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
    # Per-replica ensemble params.  Length 1 for SingleRun; length n_total
    # for PmapVmapRuns.  Ordering is flat_idx = g * n_per_gpu + p.
    ensemble_params_per_run: tuple[dict, ...] = ()
    initial_walk_config: Any = None
    adaptation_cfg: Any = None
    inter_re_config: Any = None  # InterREConfig | None
    # Configuration constraints to enforce via the MWG gate. Empty when none
    # are configured (the common case). See jaxrens.constraints.
    constraint_descriptors: tuple[ConstraintDescriptor, ...] = ()
    # Batcher describing the (G, P) topology.  ``SingleRun`` when
    # n_total == 1, ``PmapVmapRuns(n_gpu, n_per_gpu)`` otherwise.
    # Consumed uniformly by ``_run_loop``, ``build_adapt_step``,
    # ``InterREManager`` and the dispatcher in ``cli/run.py``.
    batcher: BatchDescriptor | None = None


def _resolve_single_replica(
    root: RootSpec,
    *,
    ensemble_params: dict,
    shard_n_gpu: int = 1,
) -> ResolvedConfig:
    """Resolve ``root`` into a single-replica ``ResolvedConfig`` (n_total = 1).

    ``shard_n_gpu`` (default 1) selects the batcher:

    * ``1`` → :class:`SingleRun` (the historical behaviour).
    * ``> 1`` → :class:`ShardedSingleRun(n_gpu=shard_n_gpu)`.  The
      walker population is split across ``shard_n_gpu`` devices at
      ``init_ns_sharded`` time; ``run_sharded_from_config`` is the
      runtime dispatcher.

    The structural init layout is identical between the two — initial
    positions / energies / counts come out of the resolver as flat
    ``(K, ...)`` arrays.  The reshape to ``(G, K // G, ...)`` happens
    later in ``init_ns_sharded`` so this resolver doesn't need to know
    about the physical sharding layout.

    Validates ``n_live % shard_n_gpu == 0`` and rejects the
    ``inter_re ∧ shard_n_gpu > 1`` combination (sharded single run is
    one population — there's nothing to swap with).
    """
    if shard_n_gpu > 1:
        if root.run.n_live % shard_n_gpu != 0:
            raise ValueError(
                f"run.n_live ({root.run.n_live}) is not divisible by "
                f"run.shard_n_gpu ({shard_n_gpu}).  Adjust n_live or "
                f"shard_n_gpu so that n_live % shard_n_gpu == 0."
            )
        if root.inter_re is not None:
            raise ValueError(
                "run.shard_n_gpu > 1 is incompatible with inter_re.  "
                "Sharded single run holds one population spread across "
                "GPUs — there's no second replica to swap with.  "
                "Either remove inter_re or set shard_n_gpu = 1 (and "
                "supply a multi-replica axis like ensemble.pressure: "
                "[...] for inter-RE)."
            )
    # Scale iteration-counted fields once at the top so every downstream read
    # of root.{output,run,adaptation,inter_re,termination} sees absolute-iter
    # values (see ``_apply_interval_units`` for the field list).
    #
    # ``ensemble_params`` is passed in by ``resolve`` (the caller already
    # built it via ``root.ensemble.to_ensemble_params(cohort_index=0)``).
    pressure = ensemble_params.get("pressure", None)
    chemical_potentials = ensemble_params.get("chemical_potentials", None)

    seed = root.run.seed

    ns = NSConfig(
        n_live=root.run.n_live,
        max_iterations=root.run.max_iterations,
        convergence_threshold=root.run.convergence_threshold,
        n_mcmc_steps=root.run.n_mcmc_steps,
        n_extra=root.run.n_extra,
        n_cull=root.run.n_cull,
        seed=seed,
    )

    backend = root.backend.to_backend_config()
    base_backend = root.backend.build_backend()

    # Soft-core repulsion wrapper: applied closest to the bare backend so
    # the ensemble PV term sits outside it.  Mutates ``base_backend`` so
    # both single-replica and downstream multi-replica / rejection-mode
    # init paths inherit the wrap automatically.
    if backend.softcore_repulsion is not None:
        from jaxrens.backends.softcore import SoftCoreBackend

        base_backend = SoftCoreBackend(
            base_backend, **backend.softcore_repulsion
        )

    # For initial-energy evaluation, locally wrap with EnsembleBackend so the
    # resolver's energies match the NS-loop scale (U + P*V - μ·N) — without
    # this, walkers are initialized with bare LJ energies while the MWG
    # step_fn returns ensemble-corrected energies, causing systematic
    # emax < new_energy for all cell moves and 100% rejection from the first
    # adapt call.  Discarded after _resolve_init returns; only ``base_backend``
    # crosses the resolver boundary, leaving wrapping to the runtime.
    if pressure is not None or chemical_potentials is not None:
        import jax.numpy as jnp

        from jaxrens.backends.ensemble import EnsembleBackend

        init_energy_backend = EnsembleBackend(
            base_backend,
            pressure=float(pressure) if pressure is not None else 0.0,
            chemical_potentials=(
                jnp.asarray(chemical_potentials, dtype=jnp.float32)
                if chemical_potentials is not None
                else None
            ),
        )
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
        flush_interval=int(root.output.flush_interval),
        save_acc_rates=root.output.save_acc_rates,
        acc_rates_interval=int(root.output.acc_rates_interval),
        save_max_neighbors=root.output.save_max_neighbors,
        max_neighbors_interval=int(root.output.max_neighbors_interval),
        save_re_stats=root.output.save_re_stats,
        temperature_lag_interval=root.output.temperature_lag_interval,
        temperature_interval=int(root.output.temperature_interval),
        temperature_kB=float(root.output.temperature_kB),
        collision_check_threshold=root.output.collision_check_threshold,
        collision_check_interval=int(root.output.collision_check_interval),
        wrap_atoms=bool(root.output.wrap_atoms),
        snapshot_clean=bool(root.output.snapshot_clean),
    )

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
        root.adaptation.resolve_for(m._effective_name()) for m in root.moves
    )

    resolved_init = _resolve_init(
        root.init,
        n_live=ns.n_live,
        seed=seed,
        energy_backend=init_energy_backend,
        cell_cfg=root.cell,
    )

    # Single consolidated finalize for the single-run path.  Multi-run has
    # its own equivalent seam in ``_resolve_multi_replica`` after stacking.
    # Passes ``ladder``/``offset`` from the backend config so the chosen
    # initial bucket matches ``cli/run.py``'s starting-bucket pick
    # (previously this path used the legacy ``int(max(counts))`` fallback).
    initial_energies, initial_counts = _finalise_initial_energies_and_counts(
        init_energy_backend,
        resolved_init.initial_positions,
        resolved_init.initial_types,
        resolved_init.initial_cells,
        batcher=SingleRun(),
        ladder=tuple(backend.max_neighbors_list),
        offset=int(backend.max_neighbors_offset),
    )
    resolved_init = replace(
        resolved_init,
        initial_energies=initial_energies,
        initial_max_neighbor_counts=initial_counts,
    )

    # Derive n_atoms from the resolved initial positions rather than from a
    # config field — this is the single canonical source of truth.
    n_atoms = int(resolved_init.initial_positions.shape[-2])

    _warn_if_lj_cutoff_unsafe(root.backend, root.cell, n_atoms)

    import dataclasses as _dc

    moves = tuple(m.to_move_config() for m in root.moves)
    move_descriptors = tuple(
        _dc.replace(
            m.to_descriptor(
                n_atoms=n_atoms,
                cell_cfg=root.cell,
                symbol_map=resolved_init.symbol_map,
            ),
            min_rate=policy.min_rate,
            max_rate=policy.max_rate,
            step_size_max=policy.step_size_max,
        )
        for m, policy in zip(root.moves, adaptation_policies)
    )

    constraint_descriptors = _resolve_constraints(root, resolved_init)

    if shard_n_gpu > 1:
        from jaxrens.sampling.batch_descriptor import ShardedSingleRun

        chosen_batcher: BatchDescriptor = ShardedSingleRun(n_gpu=shard_n_gpu)
    else:
        chosen_batcher = SingleRun()

    return ResolvedConfig(
        ns=ns,
        moves=moves,
        move_descriptors=move_descriptors,
        constraint_descriptors=constraint_descriptors,
        backend=backend,
        base_backend=base_backend,
        output=output,
        termination=termination,
        adaptation_policies=adaptation_policies,
        init=resolved_init,
        cell=root.cell,
        ensemble_params_per_run=(ensemble_params,),
        initial_walk_config=root.init.initial_walk,
        adaptation_cfg=root.adaptation,
        inter_re_config=(
            root.inter_re.to_inter_re_config()
            if root.inter_re is not None
            else None
        ),
        batcher=chosen_batcher,
    )


# ---------------------------------------------------------------------------
# Multi-replica resolution helpers (PmapVmapRuns(G, P))
# ---------------------------------------------------------------------------


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
    # The ensemble spec owns its own replica axis (NPT pressure list, semi_grand
    # μ / pressure list); ask it generically rather than special-casing keys.
    ensemble_cohort = root.ensemble.cohort_size()

    comp_targets: list[list[int]] | None = None
    chem_pots: list[list[float]] | None = None
    if root.inter_re is not None:
        if root.inter_re.flavor == "xrens":
            comp_targets = list(root.inter_re.composition_targets or [])
        elif root.inter_re.flavor == "semi_grand":
            # μ may come from the ensemble spec OR inter_re, never both.
            if isinstance(root.ensemble, SemiGrandEnsembleSpec):
                raise ValueError(
                    "Conflicting chemical potentials: both ensemble "
                    "(type=semi_grand) and inter_re (flavor=semi_grand) specify "
                    "chemical_potentials. Set them in exactly one place."
                )
            chem_pots = list(root.inter_re.chemical_potentials or [])
        elif root.inter_re.flavor == "pressure" and ensemble_cohort <= 1:
            raise ValueError(
                "inter_re.flavor='pressure' requires a list-valued "
                "ensemble.pressure with at least 2 entries (one per replica)."
            )

    lengths: list[tuple[str, int]] = []
    if ensemble_cohort > 1:
        lengths.append(("ensemble", ensemble_cohort))
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
            n_detected,
            n_total,
            n_total,
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
    # Start from the ensemble spec's own params (handles pressure-unit
    # conversion, μ vectors, and scalar→broadcast), then layer on the
    # inter_re-specific axes.  ``cohort_index`` broadcasts when the ensemble is
    # scalar (cohort 1) and indexes per-replica when it's a list.
    params_per_run: list[dict] = []
    for r in range(n_total):
        cohort_index = r if ensemble_cohort > 1 else 0
        params: dict = dict(
            root.ensemble.to_ensemble_params(cohort_index=cohort_index)
        )
        # Normalise vector leaves (e.g. chemical_potentials) to device arrays.
        if "chemical_potentials" in params:
            params["chemical_potentials"] = jnp.asarray(
                params["chemical_potentials"], dtype=jnp.float32
            )
        if comp_targets:
            params["target_composition"] = jnp.asarray(
                comp_targets[r], dtype=jnp.int32
            )
        if chem_pots:
            params["chemical_potentials"] = jnp.asarray(
                chem_pots[r], dtype=jnp.float32
            )
        params_per_run.append(params)

    return n_total, n_gpu, n_per_gpu, params_per_run


def _resolve_multi_replica(
    root: RootSpec,
    *,
    n_total: int,
    n_gpu: int,
    n_per_gpu: int,
    params_per_run: list[dict],
) -> ResolvedConfig:
    """Resolve ``root`` into a multi-replica (PmapVmapRuns) ``ResolvedConfig``.

    Builds per-replica initial positions / cells / energies by calling
    ``_resolve_init`` once per replica with its own seed and EnsembleBackend
    (so initial energies already include the replica's P·V term).  Stacks the
    per-replica arrays along axis 0 to produce ``(n_total, K, ...)`` pytrees,
    then runs a single consolidated PmapVmapRuns finalize so that resolver,
    burn-in, and NS step all dispatch through the same ``PmapVmapRuns(G, P)``
    batcher.

    Topology (``n_total``, ``n_gpu``, ``n_per_gpu``, ``params_per_run``) is
    pre-computed by :func:`resolve` via :func:`_derive_replica_axes` and
    passed through, so this helper does not re-derive it.
    """
    logger.info(
        "[resolve] multi-run topology: n_total=%d replicas across n_gpu=%d "
        "device(s), n_per_gpu=%d",
        n_total,
        n_gpu,
        n_per_gpu,
    )

    # Single batcher instance shared by the consolidated initial-energy
    # finalize below and the ResolvedConfig dataclass we construct at the
    # end of this function — so resolver, burn-in, and NS step all
    # dispatch through the *same* PmapVmapRuns(G, P) instance.  Burn-in
    # and NS step pick it up from ``ResolvedConfig.batcher`` (see
    # ``run_multi_gpu_from_config``).
    batcher = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)

    # Base (unwrapped) backend — the multi-GPU dispatch wraps it once.
    logger.info(
        "[resolve] building base backend (%s)", root.backend.__class__.__name__
    )
    base_backend = root.backend.build_backend()
    backend_cfg = root.backend.to_backend_config()

    # Soft-core repulsion wrapper applied closest to the bare backend so
    # the per-replica EnsembleBackend (PV term) sits outside it.  Mutates
    # ``base_backend`` so the per-replica EnsembleBackend wrap below and
    # the runtime in ``run_multi_gpu_from_config`` both inherit it.
    if backend_cfg.softcore_repulsion is not None:
        from jaxrens.backends.softcore import SoftCoreBackend

        base_backend = SoftCoreBackend(
            base_backend,
            **backend_cfg.softcore_repulsion,
        )

    # Build a per-replica EnsembleBackend for the *rejection-mode*
    # ceiling check inside ``_sample_per_walker_positions`` (grid mode
    # doesn't use it).  The actual initial-energy compute is deferred
    # to a single consolidated call below — per-replica pressure flows
    # through ``ensemble_params`` rather than separate backend objects.
    from jaxrens.backends.ensemble import EnsembleBackend

    # --- Multi-replica restart: load once, slice per replica --------------
    # When ``init.restart_file`` is set, every replica's positions / cells /
    # types come from the checkpoint instead of from the fresh species path.
    # We load the checkpoint exactly once and verify its shape matches the
    # YAML's (n_gpu, n_per_gpu) topology before slicing into per-replica
    # ResolvedInits.  This bypasses ``_resolve_init`` entirely on the restart
    # branch — that helper only knows the scalar single-run case.
    batched_restart: BatchedRestart | None = None
    if root.init.restart_file is not None:
        try:
            batched_restart = load_restart_batched(
                Path(root.init.restart_file)
            )
        except ValueError as exc:
            # ``load_restart_batched`` raises when the checkpoint is a scalar
            # single-run snapshot.  Re-raise with a multi-replica-specific
            # message that names both ``restart_file`` and the expected
            # ``n_total`` so the user can see which side of the mismatch to
            # fix without parsing the lower-level loader's diagnostics.
            if "single-run checkpoint" in str(exc):
                raise ValueError(
                    f"restart_file at {root.init.restart_file!r} is a scalar "
                    f"single-run checkpoint, but the current YAML implies "
                    f"n_total={n_total} replicas (n_gpu={n_gpu}, "
                    f"n_per_gpu={n_per_gpu}). Multi-replica resume requires a "
                    f"batched checkpoint with matching n_total."
                ) from exc
            raise
        if batched_restart.n_total != n_total:
            raise ValueError(
                f"restart_file checkpoint has n_total={batched_restart.n_total} "
                f"replicas (n_gpu={batched_restart.n_gpu}, "
                f"n_per_gpu={batched_restart.n_per_gpu}) but the current "
                f"YAML implies n_total={n_total} "
                f"(n_gpu={n_gpu}, n_per_gpu={n_per_gpu}). "
                f"Match the replica-list length (ensemble.pressure / "
                f"inter_re.*) to the checkpoint, or restart into a different "
                f"output dir."
            )
        # Cross-topology check (G mismatch) when both sides have >1 GPU is
        # raised inside ``load_restart_batched`` already.  The other case —
        # saved n_gpu=1 (VmapRuns) but current n_gpu>1 — we accept: the data
        # is the same n_total replicas, just re-tiled across devices.
        if batched_restart.n_gpu != n_gpu and batched_restart.n_gpu != 1:
            raise ValueError(
                f"restart_file checkpoint topology (n_gpu="
                f"{batched_restart.n_gpu}) does not match the current host "
                f"(n_gpu={n_gpu}). Cross-topology restart is only supported "
                f"when the saved checkpoint has n_gpu=1."
            )

    per_run_init: list[ResolvedInit] = []
    for r in range(n_total):
        p = params_per_run[r].get("pressure", None)
        # Seed policy: distinct per-replica seed = root.run.seed + r.
        seed_r = root.run.seed + r
        logger.info(
            "[resolve] replica %d/%d: seed=%d%s",
            r + 1,
            n_total,
            seed_r,
            f", pressure={p:.4g}" if p is not None else "",
        )
        if batched_restart is not None:
            # Mode D (multi-replica): slice the pre-loaded checkpoint.
            # Energies / neighbor counts left as None — the consolidated
            # finalize below recomputes them from the loaded positions.
            init_r = ResolvedInit(
                initial_positions=batched_restart.positions[r],
                initial_types=batched_restart.types[r],
                initial_cells=batched_restart.cells[r],
                initial_energies=None,
                initial_max_neighbor_counts=None,
                symbol_map=batched_restart.symbol_map,
                restart_state=None,  # 2-D list attached at the stacked level
            )
        else:
            per_run_backend = (
                EnsembleBackend(base_backend, pressure=float(p))
                if p is not None
                else base_backend
            )
            init_r = _resolve_init(
                root.init,
                n_live=root.run.n_live,
                seed=seed_r,
                energy_backend=per_run_backend,
                cell_cfg=root.cell,
            )
        per_run_init.append(init_r)

    # Validate structural shapes (energies/counts are None at this
    # point — the consolidated finalize fills them in below).
    ref = per_run_init[0]
    for r, init_r in enumerate(per_run_init):
        for field_name in (
            "initial_positions",
            "initial_types",
            "initial_cells",
        ):
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

    initial_positions = jnp.stack(
        [x.initial_positions for x in per_run_init], axis=0
    )
    initial_cells = (
        jnp.stack([x.initial_cells for x in per_run_init], axis=0)
        if per_run_init[0].initial_cells is not None
        else None
    )
    # This allows for varying types across replicas.
    initial_types = jnp.stack([x.initial_types for x in per_run_init], axis=0)

    # --- Consolidated finalize on stacked (G, P, K, ...) arrays -----------
    # Reshape (n_total, K, ...) → (G, P, K, ...) and run a single
    # PmapVmapRuns finalize.  This parallel-compiles the energy
    # function across G GPUs at the same shape burn-in / NS step use,
    # so all three stages share one JIT cache slot.
    K_axis = initial_positions.shape[1]
    n_atoms = initial_positions.shape[2]
    reshaped_positions = initial_positions.reshape(
        n_gpu, n_per_gpu, K_axis, n_atoms, 3
    )
    reshaped_cells = (
        initial_cells.reshape(n_gpu, n_per_gpu, K_axis, 3, 3)
        if initial_cells is not None
        else None
    )

    reshaped_types = (
        initial_types.reshape(n_gpu, n_per_gpu, K_axis, n_atoms)
        if initial_types is not None
        else None
    )

    # Thread the per-replica ensemble params the runtime NS loop uses
    # (``ensemble_params_per_run``) into the consolidated initial-energy
    # compute, so resolved initial energies include the same P·V and −μ·N
    # terms the EnsembleBackend applies at runtime — matching by construction.
    # ``EnsembleBackend`` reads exactly these keys; others (e.g.
    # ``target_composition``) are not energy terms and are not threaded.
    _ENERGY_KEYS = ("pressure", "chemical_potentials")
    ensemble_params_batched = _stack_ensemble_params(
        params_per_run, _ENERGY_KEYS, (n_gpu, n_per_gpu)
    )

    # If any replica carries an energy-relevant ensemble param, all replicas
    # share one EnsembleBackend wrapper and their per-replica params flow
    # through ``ensemble_params``; otherwise use the raw base backend.
    if ensemble_params_batched is not None:
        finalize_backend = EnsembleBackend(base_backend, pressure=0.0)
    else:
        finalize_backend = base_backend

    initial_energies, initial_counts = _finalise_initial_energies_and_counts(
        finalize_backend,
        reshaped_positions,
        reshaped_types,
        reshaped_cells,
        batcher=batcher,
        ladder=tuple(backend_cfg.max_neighbors_list),
        offset=int(backend_cfg.max_neighbors_offset),
        ensemble_params_batched=ensemble_params_batched,
    )

    # Collapse the (G, P, K, ...) shape back to (n_total, K, ...) so
    # downstream code (e.g. the dispatcher) sees the layout it
    # already handles.  ``init_ns_multi_gpu`` accepts both shapes.
    initial_positions = reshaped_positions.reshape(n_total, K_axis, n_atoms, 3)
    initial_cells = (
        reshaped_cells.reshape(n_total, K_axis, 3, 3)
        if reshaped_cells is not None
        else None
    )
    initial_types = (
        reshaped_types.reshape(n_total, K_axis, n_atoms)
        if reshaped_types is not None
        else None
    )

    if initial_energies is not None:
        initial_energies = initial_energies.reshape(n_total, K_axis)
    if initial_counts is not None:
        initial_max_neighbor_counts = initial_counts.reshape(n_total, K_axis)
    else:
        initial_max_neighbor_counts = None

    # symbol_map, restart_state from ref (must be identical across replicas).
    symbol_map = per_run_init[0].symbol_map
    # For the multi-replica restart branch, the 2-D bundle list was loaded
    # once at the top of this function; attach it here so the dispatcher in
    # ``run_multi_gpu_from_config`` can forward it as ``restart_states=`` to
    # ``run_ns_multi_gpu``.  Reshape from saved n_gpu=1 to (n_gpu_current,
    # n_per_gpu_current) when the host re-tiles the replicas across more
    # devices than the checkpoint was saved on (saved n_gpu=1 → current n_gpu>1).
    restart_state_2d: list[list[RestartBundle]] | None = None
    if batched_restart is not None:
        if batched_restart.n_gpu == n_gpu:
            restart_state_2d = batched_restart.bundles_2d
        else:
            # saved n_gpu=1 retile case: flatten then re-nest as (n_gpu, n_per_gpu).
            flat = batched_restart.bundles_flat
            restart_state_2d = [
                flat[g * n_per_gpu : (g + 1) * n_per_gpu] for g in range(n_gpu)
            ]
    stacked_init = ResolvedInit(
        initial_positions=initial_positions,
        initial_types=initial_types,
        initial_cells=initial_cells,
        initial_energies=initial_energies,
        initial_max_neighbor_counts=initial_max_neighbor_counts,
        symbol_map=symbol_map,
        restart_state=restart_state_2d,
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
        inter_re=(
            root.inter_re.to_inter_re_config()
            if root.inter_re is not None
            else None
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
        flush_interval=int(root.output.flush_interval),
        save_acc_rates=root.output.save_acc_rates,
        acc_rates_interval=int(root.output.acc_rates_interval),
        save_max_neighbors=root.output.save_max_neighbors,
        max_neighbors_interval=int(root.output.max_neighbors_interval),
        save_re_stats=root.output.save_re_stats,
        temperature_lag_interval=root.output.temperature_lag_interval,
        temperature_interval=int(root.output.temperature_interval),
        temperature_kB=float(root.output.temperature_kB),
        collision_check_threshold=root.output.collision_check_threshold,
        collision_check_interval=int(root.output.collision_check_interval),
        wrap_atoms=bool(root.output.wrap_atoms),
        snapshot_clean=bool(root.output.snapshot_clean),
    )

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
        root.adaptation.resolve_for(m._effective_name()) for m in root.moves
    )

    n_atoms = int(stacked_init.initial_positions.shape[-2])
    _warn_if_lj_cutoff_unsafe(root.backend, root.cell, n_atoms)

    import dataclasses as _dc

    moves = tuple(m.to_move_config() for m in root.moves)
    move_descriptors = tuple(
        _dc.replace(
            m.to_descriptor(
                n_atoms=n_atoms,
                cell_cfg=root.cell,
                symbol_map=stacked_init.symbol_map,
            ),
            min_rate=policy.min_rate,
            max_rate=policy.max_rate,
            step_size_max=policy.step_size_max,
        )
        for m, policy in zip(root.moves, adaptation_policies)
    )

    constraint_descriptors = _resolve_constraints(root, stacked_init)

    return ResolvedConfig(
        ns=ns,
        moves=moves,
        move_descriptors=move_descriptors,
        constraint_descriptors=constraint_descriptors,
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
            root.inter_re.to_inter_re_config()
            if root.inter_re is not None
            else None
        ),
        batcher=batcher,
    )


# ---------------------------------------------------------------------------
# Plan phase — decisions the resolver can make without building or running
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPlan:
    """What the resolver can decide without materialising anything.

    Splitting the resolver in two lets ``jaxrens validate`` answer the
    questions people actually ask of it — is the topology legal, do the
    referenced files exist, does the cell prior fit the cutoff — without
    loading an energy model, placing walkers, or compiling a kernel.  On the
    Lennard-Jones example that is ~0.4 s of work instead of ~15 s, and the
    gap is far wider for an MLIP backend.

    A plan is *not* enough to run: it deliberately holds no backend, no
    positions and no energies.  :func:`resolve` builds on top of it.

    Attributes:
        root: The config after interval-unit scaling — every iteration-counted
            field is in absolute iterations.
        n_total: Total replica count implied by the config.
        n_gpu: Devices the replicas are spread over.
        n_per_gpu: Replicas per device (``n_total // n_gpu``).
        ensemble_params_per_run: Per-replica ensemble params; empty when
            ``n_total == 1`` (see ``_derive_replica_axes``).
        n_atoms: Atom count when it is derivable from the config alone, else
            ``None`` — a walker-set or restart file carries it in the data,
            and reading that is materialisation.
        topology: Human-readable one-line description of the dispatch shape.
    """

    root: RootSpec
    n_total: int
    n_gpu: int
    n_per_gpu: int
    ensemble_params_per_run: list[dict]
    n_atoms: int | None
    topology: str


def _plan_n_atoms(root: RootSpec) -> int | None:
    """Atom count from the config alone, or ``None`` if it needs the data.

    ``start_species`` carries the counts outright; a structure file gives them
    up for a single cheap read.  A walker set or a restart bundle stores the
    count inside arrays whose loading is exactly the work the plan phase
    exists to avoid, so those report ``None``.
    """
    init = root.init
    if init.start_species is not None:
        return sum(init.parsed_species().values())
    if init.start_config_file is not None:
        try:
            from ase.io import read as ase_read

            return len(ase_read(str(init.start_config_file), index=0))
        except Exception:  # unreadable / bad format -> reported by _plan_paths
            return None
    return None


def _plan_paths(root: RootSpec) -> None:
    """Check that every path the config names exists and is readable.

    Cheap stand-in for the confidence the old full-resolve default gave: it
    catches the mistyped checkpoint or structure path — by far the most
    common way a config fails minutes into a job — without paying to load
    what is behind it.  It cannot tell you the file is *valid*; only
    ``resolve`` can.

    Raises:
        FileNotFoundError: If a referenced path does not exist.
        PermissionError: If it exists but cannot be read.
    """
    candidates: list[tuple[str, object]] = [
        ("init.start_config_file", root.init.start_config_file),
        ("init.start_walker_set", root.init.start_walker_set),
        ("init.restart_file", root.init.restart_file),
    ]
    checkpoint = getattr(root.backend, "checkpoint_path", None)
    # A Nequix ``checkpoint_path`` may name a bundled model rather than a
    # file on disk, so only check it when it looks like a path.
    if isinstance(checkpoint, str) and (
        os.sep in checkpoint or checkpoint.endswith((".pkl", ".model", ".nqx"))
    ):
        candidates.append(("backend.checkpoint_path", Path(checkpoint)))
    for key, value in candidates:
        if value is None:
            continue
        path = Path(str(value))
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path} does not exist.")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"{key}: {path} exists but is not readable.")


def _describe_topology(
    root: RootSpec, n_total: int, n_gpu: int, n_per_gpu: int
) -> str:
    """One-line description of the dispatch shape this config implies."""
    if n_total == 1:
        if root.run.shard_n_gpu > 1:
            return (
                f"ShardedSingleRun (1 replica, sharded across "
                f"{root.run.shard_n_gpu} GPUs)"
            )
        return "SingleRun (1 replica, 1 GPU)"
    return f"n_gpu={n_gpu} × n_per_gpu={n_per_gpu} = {n_total} replica(s)"


def resolve_plan(
    root: RootSpec, *, geometry_checks: bool = True
) -> ResolvedPlan:
    """Make every resolver decision that needs no backend and no walkers.

    Args:
        root: Fully validated pydantic config.
        geometry_checks: Run the cell-prior geometry warnings here.  Set
            ``False`` from :func:`resolve`, which runs them further down with
            an exact ``n_atoms`` and would otherwise emit each twice.

    Returns:
        The :class:`ResolvedPlan` for this config.

    Raises:
        ValueError: On an illegal topology, an incompatible ``shard_n_gpu``,
            or a referenced path that is missing or unreadable.
    """
    root = _apply_interval_units(root)
    n_total, n_gpu, n_per_gpu, params_per_run = _derive_replica_axes(root)

    if n_total > 1 and root.run.shard_n_gpu > 1:
        raise ValueError(
            f"run.shard_n_gpu ({root.run.shard_n_gpu}) is incompatible "
            f"with the multi-replica topology implied by this config "
            f"(n_total = {n_total} > 1).  Sharded single-run holds one "
            f"population spread across GPUs; multi-replica runs hold "
            f"n_total *independent* populations.  Pick one — remove the "
            f"replica-axis list (ensemble.pressure / inter_re.*) or set "
            f"shard_n_gpu = 1."
        )

    _plan_paths(root)
    n_atoms = _plan_n_atoms(root)
    if geometry_checks and n_atoms is not None:
        _warn_if_lj_cutoff_unsafe(root.backend, root.cell, n_atoms)

    return ResolvedPlan(
        root=root,
        n_total=n_total,
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
        ensemble_params_per_run=params_per_run,
        n_atoms=n_atoms,
        topology=_describe_topology(root, n_total, n_gpu, n_per_gpu),
    )


def resolve(root: RootSpec) -> ResolvedConfig:
    """Translate a validated ``RootSpec`` into one unified :class:`ResolvedConfig`.

    Topology resolution decides the batcher:

    * ``n_total == 1`` (scalar pressure, no inter-RE replica-axis list) →
      :class:`SingleRun()`.  Init arrays carry shape ``(K, ...)``.
    * ``n_total >= 2`` → :class:`PmapVmapRuns(n_gpu, n_per_gpu)` with
      ``n_gpu = max(1, len(jax.local_devices()))`` clamped to ``n_total``.
      Init arrays carry shape ``(n_total, K, ...)``; the dispatcher
      reshapes to ``(n_gpu, n_per_gpu, K, ...)`` at consume time.

    Future case "single NS sharded across multiple GPUs" will be a third
    branch keyed on a new batcher type — orthogonal to this dispatcher.

    Args:
        root: Fully validated pydantic config.

    Returns:
        Single :class:`ResolvedConfig` carrying the resolved batcher.

    Raises:
        ValueError: Replica-axis list lengths disagree, ``n_total`` is not
            divisible by the detected device count, or a ``pressure``-flavor
            inter-RE has a scalar ``ensemble.pressure``.
    """
    # Every decision that needs neither a backend nor walkers is made by the
    # plan phase; this function is the materialisation on top of it.  Geometry
    # checks are deferred: the plan may not know ``n_atoms`` (walker-set and
    # restart configs carry it in the data), whereas the branches below always
    # do, and running them in both places would double every warning.
    plan = resolve_plan(root, geometry_checks=False)
    root = plan.root
    n_total = plan.n_total
    n_gpu, n_per_gpu = plan.n_gpu, plan.n_per_gpu
    params_per_run = plan.ensemble_params_per_run

    if n_total == 1:
        # SingleRun (or sharded-single) path.  ``params_per_run`` is
        # empty here per the contract of ``_derive_replica_axes``;
        # reconstruct the scalar dict from the ensemble spec directly
        # so ``ensemble_params_per_run`` always has length 1 in the
        # resolved config.
        ensemble_params = root.ensemble.to_ensemble_params(cohort_index=0)
        return _resolve_single_replica(
            root,
            ensemble_params=ensemble_params,
            shard_n_gpu=root.run.shard_n_gpu,
        )

    return _resolve_multi_replica(
        root,
        n_total=n_total,
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
        params_per_run=params_per_run,
    )
