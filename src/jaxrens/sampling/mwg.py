"""Metropolis-within-Gibbs (MWG) sampler factory.

Assembles multiple move types into a single step function that dispatches
via lax.switch. The MCState class is built dynamically from the move
descriptors — only fields needed by the active moves are included.

Usage:
    init_fn, step_fn = build_mwg(energy_fn, params, [
        MoveDescriptor("random_walk", random_walk.build_kernel, weight=7),
        MoveDescriptor("volume", volume.build_kernel,
                       kernel_kwargs={"n_atoms": 64}, weight=2),
        MoveDescriptor("galilean", galilean.build_kernel,
                       kernel_kwargs={"n_reflect": 5}, weight=1,
                       extra_state_fields={"direction": (jnp.ndarray,
                           lambda pos, types: jnp.zeros_like(pos))}),
    ])
    state = init_fn(positions, types, energy, box)
    state, info = step_fn(rng_key, state, likelihood_constraint)
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.state.mc_state import make_mc_state_class
from jaxrens.types import Params


def build_mwg(
    energy_fn: Any,
    params: Params,
    move_descriptors: list[MoveDescriptor],
) -> tuple[Callable, Callable]:
    """Build a Metropolis-within-Gibbs sampler from move descriptors.

    Dynamically builds an MCState class containing only the fields
    needed by the active moves. Step sizes are stored on the state
    as a per-move array for independent per-replica adaptation.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.
        move_descriptors: List of MoveDescriptor, each specifying a move
            type with its build_kernel, kwargs, weight, step_size, and
            optional extra_state_fields.

    Returns:
        (init_fn, step_fn) where:
        - init_fn(positions, types, energy, box, step_sizes) -> MCState
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
        desc.build_kernel(energy_fn, params, **desc.kernel_kwargs)
        for desc in move_descriptors
    ]

    # --- Wrap each step_fn ---
    def _wrap(raw_fn, move_idx):
        def wrapped(state, key: jax.Array, constraint: float):
            # Inject this move's step_size from the per-move array
            state_with_ss = state.set(step_size=state.step_sizes[move_idx])
            new_state, info = raw_fn(key, state_with_ss, constraint)
            # Track per-move acceptance
            new_state = new_state.set(
                n_proposed=new_state.n_proposed.at[move_idx].add(1),
                n_accepted=new_state.n_accepted.at[move_idx].add(
                    info.accepted.astype(jnp.int32)
                ),
            )
            return new_state, info
        return wrapped

    wrapped_fns = [_wrap(fn, i) for i, fn in enumerate(raw_step_fns)]

    # --- Default step sizes from descriptors ---
    default_step_sizes = jnp.array([d.step_size for d in move_descriptors])

    # --- init_fn ---

    def init_fn(
        positions: jnp.ndarray,
        types: jnp.ndarray,
        energy: float,
        box: jnp.ndarray | None = None,
        step_sizes: jnp.ndarray | None = None,
        step_size: float | None = None,
    ) -> Any:  # returns MCStateClass instance
        """Create initial MCState from walker data.

        Args:
            step_sizes: Per-move step size array, shape (n_move_types,).
                If None, uses defaults from descriptors.
            step_size: Scalar step size — broadcast to all moves.
                Convenience for single-move setups and adaptation.
                Overridden by step_sizes if both are given.
        """
        if box is None:
            box = jnp.zeros((3, 3))
        if step_sizes is None:
            if step_size is not None:
                step_sizes = jnp.full(n_moves, step_size)
            else:
                step_sizes = default_step_sizes

        kwargs = dict(
            positions=jnp.asarray(positions),
            types=jnp.asarray(types),
            energy=jnp.asarray(energy),
            box=jnp.asarray(box),
            step_size=jnp.asarray(0.0),  # ephemeral — set by wrapper
            step_sizes=jnp.asarray(step_sizes),
            n_accepted=jnp.zeros(n_moves, dtype=jnp.int32),
            n_proposed=jnp.zeros(n_moves, dtype=jnp.int32),
            max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
            overflow=jnp.asarray(False),
        )

        # Initialize move-specific fields
        for name, (_, initializer) in all_extra_fields.items():
            kwargs[name] = initializer(positions, types)

        return MCStateClass(**kwargs)

    # --- step_fn ---

    def step_fn(
        rng_key: jax.Array,
        state: Any,
        likelihood_constraint: float,
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

        return new_state, info

    return init_fn, step_fn
