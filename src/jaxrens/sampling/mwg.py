"""Metropolis-within-Gibbs (MWG) sampler factory.

Assembles multiple move types into a single step function that dispatches
via lax.switch. The MCState class is built dynamically from the move
descriptors — only fields needed by the active moves are included.

Usage::

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

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key

from jaxrens.base import MoveInfo
from jaxrens.constraints.base import ConstraintDescriptor, make_move_gate
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.state.mc_state import make_mc_state_class


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
    def _wrap(raw_fn, move_idx, gate):
        def wrapped(
            state, key: Key[Array, ""], constraint: float | Float[Array, ""]
        ):
            # Inject this move's step_size from the per-move array
            state_with_ss = state.set(step_size=state.step_sizes[move_idx])
            new_state, info = raw_fn(key, state_with_ss, constraint)

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
        _wrap(fn, i, move_gates[i]) for i, fn in enumerate(raw_step_fns)
    ]

    # --- Default step sizes from descriptors ---
    default_step_sizes = jnp.array([d.step_size for d in move_descriptors])

    # --- init_fn ---

    def init_fn(
        positions: Float[Array, "*B N 3"],
        types: Int[Array, "*B N"],
        energy: float | Float[Array, "*B"],
        cell: Float[Array, "*B 3 3"] | None = None,
        step_sizes: Float[Array, "n_moves"] | None = None,
        step_size: float | Float[Array, ""] | None = None,
        ensemble_params: dict | None = None,
        max_neighbors: int = 0,
        max_neighbor_count_init: int | Int[Array, "*B"] = 0,
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
            ensemble_params=ensemble_params,
            max_neighbors=int(max_neighbors),
        )

        # Initialize move-specific fields
        for name, (_, initializer) in all_extra_fields.items():
            kwargs[name] = initializer(positions, types)

        return MCStateClass(**kwargs)

    # --- step_fn ---

    def step_fn(
        rng_key: Key[Array, ""],
        state: Any,
        likelihood_constraint: float | Float[Array, ""],
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
