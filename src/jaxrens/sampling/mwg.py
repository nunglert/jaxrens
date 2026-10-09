"""Metropolis-within-Gibbs (MWG) sampler factory.

Assembles multiple move types into a single step function that dispatches
via lax.switch. The MCState class is built dynamically from the move
descriptors — only fields needed by the active moves are included.

Usage:
    backend = HarmonicBackend(k=1.0)
    init_fn, step_fn, per_move_fns = build_mwg(backend, [
        MoveKernel("random_walk", random_walk.build_kernel, weight=7),
        MoveKernel("volume", volume.build_kernel,
                   kernel_kwargs={"n_atoms": 64}, weight=2),
        MoveKernel("galilean", galilean.build_kernel,
                   kernel_kwargs={"n_reflect": 5}, weight=1,
                   extra_state_fields={"direction": (jnp.ndarray,
                       lambda pos, types: jnp.zeros_like(pos))}),
    ])
    state = init_fn(positions, types, energy, cell)
    state, info = step_fn(rng_key, state, likelihood_constraint)
    # per_move_fns[i](state, key, constraint) runs move i directly
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jaxrens.backends.locality import probe_local_capability
from jaxrens.base import MoveInfo
from jaxrens.constraints.base import ConstraintDescriptor, make_move_gate
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.state.mc_state import make_mc_state_class

logger = logging.getLogger(__name__)

_LOCAL_KERNEL_KWARGS = ("max_affected",)


def _downgrade_to_full(d: MoveKernel) -> MoveKernel:
    """Strip whichever kernel_kwargs would select ``d``'s local/incremental
    path and reclassify it as ``affects="all"``.

    Every local-capable move kernel (``build_swap_kernel``,
    ``build_morph_kernel``, ``single_atom.build_kernel``,
    ``build_sweep_kernel``) already has a full-recompute branch it falls
    back to when its local-path kwarg (``max_affected``) is ``None`` —
    that's the only thing removing it needs to trigger. ``extra_state_fields``
    is also cleared: if some other move in the set is still genuinely
    local, that move's own declaration already contributes the same shared
    cache fields; if none is, this avoids seeding an
    ``atomic_energies``/``raw_energy`` cache nothing will read.
    """
    stripped_kwargs = {
        k: v
        for k, v in d.kernel_kwargs.items()
        if k not in _LOCAL_KERNEL_KWARGS
    }
    return dataclasses.replace(
        d, affects="all", kernel_kwargs=stripped_kwargs, extra_state_fields={}
    )


def _resolve_local_affects(d: MoveKernel, backend: Any) -> MoveKernel:
    """Return ``d`` unchanged if its ``affects="local"`` declaration (if
    any) is safely usable given ``backend``; otherwise log a warning and
    return a downgraded (``affects="all"``) copy. A no-op for moves that
    never declared ``"local"``.

    Every local move recomputes its affected-atom set fresh from the
    CURRENT ``state.positions``/``state.cell`` on every proposal (see
    ``sampling/neighbor_list.py`` module docstring) — there is no cached
    structure left that another move in the set could invalidate, so the
    only remaining reason a ``"local"`` declaration can't be honored is a
    genuine backend limitation (not a combination of moves).
    """
    if d.affects != "local":
        return d

    cap = probe_local_capability(backend)
    if cap.kind != "single_cutoff":
        logger.warning(
            "move %r declared affects='local', but the backend's local "
            "energy capability is %r (needs 'single_cutoff') — falling "
            "back to full energy recomputation for this move (see "
            "jaxrens.backends.locality).",
            d.name,
            cap.kind,
        )
        return _downgrade_to_full(d)

    return d


def build_mwg(
    backend: Any,
    move_descriptors: list[MoveKernel],
    constraint_descriptors: tuple[ConstraintDescriptor, ...] = (),
) -> tuple[Callable, Callable, list[Callable]]:
    """Build a Metropolis-within-Gibbs sampler from move descriptors.

    Dynamically builds an MCState class containing only the fields
    needed by the active moves. Step sizes are stored on the state
    as a per-move array for independent per-replica adaptation.

    Args:
        backend: EnergyBackend instance. Captured in move closures.
        move_descriptors: List of MoveKernel, each specifying a move
            type with its build_kernel, kwargs, weight, step_size, and
            optional extra_state_fields.
        constraint_descriptors: Configuration constraints to enforce. Each is
            paired statically with the moves that can violate it (via
            ``MoveKernel.mutates`` vs ``ConstraintDescriptor.depends_on``);
            only intersecting moves get a constraint gate, and moves with no
            relevant constraint keep the unmodified fast path. Empty by
            default, so existing callers are unaffected.

    Returns:
        (init_fn, step_fn) where:
        - init_fn(positions, types, energy, cell, step_sizes) -> MCState
        - step_fn(rng_key, state, likelihood_constraint) -> (MCState, MoveInfo)
    """
    n_moves = len(move_descriptors)

    # --- Local/incremental energy update: capability check ---
    # A move declares affects="local" as a structural fact about itself
    # (see MoveKernel docstring) — it recomputes its affected-atom set
    # fresh from the current state on every proposal, so it is safe in ANY
    # combination with other moves, cell-mutating ones included. The only
    # thing that can still prevent it from running is the backend itself
    # not supporting the subset-energy query it needs; when that happens,
    # the declaration is downgraded to "all" (full recompute for that move
    # only) with a logged warning — every combination of moves is accepted.
    move_descriptors = [
        _resolve_local_affects(d, backend) for d in move_descriptors
    ]

    has_local_move = any(d.affects == "local" for d in move_descriptors)

    # The unwrapped base backend (EnsembleBackend.base if wrapped, else
    # backend itself) — needed to recompute the atomic-energy cache from
    # scratch after an affects="all" move, in the *raw* (pre-ensemble-
    # correction) convention the cache is kept in.
    _base_backend = getattr(backend, "base", backend)

    # --- Collect extra state fields from all descriptors ---
    all_extra_fields: dict[str, tuple[type, Callable]] = {}
    for desc in move_descriptors:
        for name, (typ, initializer) in desc.extra_state_fields.items():
            if name in all_extra_fields:
                existing_typ = all_extra_fields[name][0]
                if existing_typ != typ:
                    raise ValueError(
                        f"Conflicting types for extra field '{name}': "
                        f"{existing_typ} vs {typ}"
                    )
            all_extra_fields[name] = (typ, initializer)

    # --- Build MCState class ---
    extra_types = {name: typ for name, (typ, _) in all_extra_fields.items()}
    MCStateClass = make_mc_state_class(extra_types)

    # --- Normalize weights to probabilities ---
    weights = jnp.array([d.weight for d in move_descriptors])
    move_probs = weights / weights.sum()

    # --- Build per-move step functions ---
    raw_step_fns = [
        desc.build_kernel(backend, **desc.kernel_kwargs)
        for desc in move_descriptors
    ]

    # --- Static constraint-gate per move ---
    # ``gate`` is None for moves that cannot violate any registered
    # constraint (mutated aspects disjoint from every constraint's
    # depends_on); those keep the unmodified fast path with zero added graph.
    move_gates = [
        make_move_gate(constraint_descriptors, desc.mutates)
        for desc in move_descriptors
    ]

    # --- Wrap each step_fn ---
    def _wrap(raw_fn, move_idx, gate, affects):
        def wrapped(state, key: jax.Array, constraint: float | jnp.ndarray):
            # Inject this move's step_size from the per-move array
            state_with_ss = state.set(step_size=state.step_sizes[move_idx])
            new_state, info = raw_fn(key, state_with_ss, constraint)

            if has_local_move and affects == "all":
                # This move doesn't know about the atomic-energy cache and
                # may have changed any atom's contribution (or the cell) —
                # recompute the cache from scratch so later "local" moves
                # patch it against a correct baseline. Costs one extra full
                # evaluation per *accepted* affects="all" step; only paid
                # when local and non-local moves are mixed in the same
                # move-set (the primary intended workflow — swap/morph
                # only — never hits this branch).
                recomputed_atomic = _base_backend.atomic_energies(
                    new_state.positions,
                    new_state.types,
                    new_state.cell,
                    new_state.max_neighbors,
                )
                shift = getattr(_base_backend, "energy_shift_per_atom", 0.0)
                n_real = jnp.sum(new_state.types >= 0).astype(jnp.float32)
                recomputed_raw = recomputed_atomic.sum() + shift * n_real
                new_state = new_state.set(
                    atomic_energies=jnp.where(
                        info.accepted, recomputed_atomic, state.atomic_energies
                    ),
                    raw_energy=jnp.where(
                        info.accepted, recomputed_raw, state.raw_energy
                    ),
                )

            if gate is not None:
                # Enforce configuration constraints on the proposed config.
                # The incoming ``state`` is valid by invariant (initial
                # walkers are checked at setup), so a constraint violation can
                # only come from this move; revert the physical config to the
                # pre-move state and flip the acceptance + reject_reason.
                ok, reason = gate(
                    new_state.positions, new_state.types, new_state.cell
                )
                new_state = new_state.set(
                    positions=jnp.where(
                        ok, new_state.positions, state.positions
                    ),
                    cell=jnp.where(ok, new_state.cell, state.cell),
                    types=jnp.where(ok, new_state.types, state.types),
                    energy=jnp.where(ok, new_state.energy, state.energy),
                )
                violated = info.accepted & ~ok
                info = info._replace(
                    accepted=info.accepted & ok,
                    reject_reason=jnp.where(
                        violated, reason, info.reject_reason
                    ),
                )

            # Track per-move acceptance (uses the post-gate acceptance)
            new_state = new_state.set(
                n_proposed=new_state.n_proposed.at[move_idx].add(1),
                n_accepted=new_state.n_accepted.at[move_idx].add(
                    info.accepted.astype(jnp.int32)
                ),
            )
            return new_state, info

        return wrapped

    wrapped_fns = [
        _wrap(fn, i, move_gates[i], move_descriptors[i].affects)
        for i, fn in enumerate(raw_step_fns)
    ]

    # --- Default step sizes from descriptors ---
    default_step_sizes = jnp.array([d.step_size for d in move_descriptors])

    # --- init_fn ---

    def init_fn(
        positions: jnp.ndarray,
        types: jnp.ndarray,
        energy: float | jnp.ndarray,
        cell: jnp.ndarray | None = None,
        step_sizes: jnp.ndarray | None = None,
        step_size: float | jnp.ndarray | None = None,
        ensemble_params: dict | None = None,
        max_neighbors: int = 0,
        max_neighbor_count_init: int | jnp.ndarray = 0,
        image_bucket: int = 1,
        image_count_needed_init: int | jnp.ndarray = 0,
    ) -> Any:  # returns MCStateClass instance
        """Create initial MCState from walker data.

        Args:
            step_sizes: Per-move step size array, shape (n_move_types,).
                If None, uses defaults from descriptors.
            step_size: Scalar step size — broadcast to all moves.
            ensemble_params: Ensemble parameters dict (e.g. {"pressure": 0.01}).
                Stored on the MCState for use by EnsembleBackend.
            max_neighbors: Initial neighbor-bucket size for GNN-style
                backends.  0 is the legacy default and causes the first
                ns_step to overflow immediately (wasteful first retry);
                pass a value from ``BackendConfig.max_neighbors_list[0]``
                to avoid that.  Ignored by backends that don't use buckets.
            max_neighbor_count_init: Observed per-walker max neighbor
                count at init time (from ``backend.max_neighbors_for``).
                Seeds the dynamic ``max_neighbor_count`` field so the
                outer-loop overflow retry sees accurate counts from iter
                0 instead of zeros that falsely suggest "nothing observed
                yet".  Default 0 preserves legacy behaviour.
            image_bucket: Initial periodic-image half-width for local-
                update move kernels (see
                ``sampling/neighbor_list.py::build_symmetric_image_offsets``).
                1 is the legacy/inert default; pass a value chosen from
                the reference geometry (see
                ``sampling/neighbor_list.py::initial_image_bucket_for_cell``)
                to avoid an immediate first-step overflow retry. Ignored
                when no local move is active.
            image_count_needed_init: Observed per-walker true image count
                needed at init time (see
                ``sampling/neighbor_list.py::initial_image_bucket_for_cell``).
                Seeds the dynamic ``image_count_needed`` field, mirroring
                ``max_neighbor_count_init`` above. Default 0 preserves
                legacy behaviour.

        Note: when an ``affects="local"`` move is active, the
        ``atomic_energies``/``raw_energy`` extra fields are created here at
        their zero-valued placeholder default (the generic
        ``extra_state_fields`` initializer signature has no backend
        access). Callers must seed them with real values immediately after
        calling ``init_fn`` via
        ``sampling/local_energy.py::seed_local_energy_cache`` — see that
        function's docstring for why this is a required, not optional,
        follow-up step.
        """
        if cell is None:
            cell = jnp.zeros((3, 3))
        if step_sizes is None:
            if step_size is not None:
                step_sizes = jnp.full(n_moves, step_size)
            else:
                step_sizes = default_step_sizes
        if ensemble_params is None:
            ensemble_params = {}

        kwargs = dict(
            positions=jnp.asarray(positions),
            types=jnp.asarray(types),
            energy=jnp.asarray(energy),
            cell=jnp.asarray(cell),
            step_size=jnp.asarray(0.0),  # ephemeral — set by wrapper
            step_sizes=jnp.asarray(step_sizes),
            n_accepted=jnp.zeros(n_moves, dtype=jnp.int32),
            n_proposed=jnp.zeros(n_moves, dtype=jnp.int32),
            max_neighbor_count=jnp.asarray(
                max_neighbor_count_init, dtype=jnp.int32
            ),
            overflow=jnp.asarray(False),
            image_count_needed=jnp.asarray(
                image_count_needed_init, dtype=jnp.int32
            ),
            image_overflow=jnp.asarray(False),
            ensemble_params=ensemble_params,
            max_neighbors=int(max_neighbors),
            image_bucket=int(image_bucket),
        )

        # Initialize move-specific fields. Note: when an affects="local"
        # move is active, this seeds atomic_energies/raw_energy at a
        # zero-valued placeholder (this initializer has no backend access)
        # — see this function's docstring: callers MUST follow up with
        # sampling/local_energy.py::seed_local_energy_cache before running
        # any step.
        for name, (_, initializer) in all_extra_fields.items():
            kwargs[name] = initializer(positions, types)

        return MCStateClass(**kwargs)

    # --- step_fn ---

    def step_fn(
        rng_key: jax.Array,
        state: Any,
        likelihood_constraint: float | jnp.ndarray,
    ) -> tuple[Any, MoveInfo]:
        """One MWG step: randomly select a move and execute it."""
        key_select, key_move = jax.random.split(rng_key)
        move_idx = jax.random.choice(key_select, n_moves, p=move_probs)

        new_state, info = jax.lax.switch(
            move_idx,
            wrapped_fns,
            state,
            key_move,
            likelihood_constraint,
        )

        # Inject the chosen move_idx so downstream consumers (ns_step scan)
        # can attribute accepted/rejected counts to the correct move.
        info = info._replace(move_idx=jnp.asarray(move_idx, dtype=jnp.int32))

        return new_state, info

    return init_fn, step_fn, wrapped_fns
