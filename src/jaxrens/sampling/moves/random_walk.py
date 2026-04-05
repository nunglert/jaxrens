"""Random walk move: Gaussian perturbation with accept/reject.

The simplest NS move. Proposes a random displacement of all atom positions,
accepts if the new energy is below the likelihood constraint (Emax).

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types


class RandomWalkState(NamedTuple):
    """State for the random walk move."""

    positions: jnp.ndarray  # (n_atoms, 3)
    types: jnp.ndarray  # (n_atoms,)
    energy: jnp.ndarray  # scalar
    box: jnp.ndarray | None  # (3, 3) or None
    step_size: jnp.ndarray  # scalar


def init(
    positions: Positions,
    types: Types,
    energy: float,
    box: Box | None = None,
    step_size: float = 0.1,
) -> RandomWalkState:
    """Create initial random walk state."""
    return RandomWalkState(
        positions=positions,
        types=types,
        energy=jnp.asarray(energy),
        box=box,
        step_size=jnp.asarray(step_size),
    )


def build_kernel(
    energy_fn: Any,
    params: Params,
):
    """Build a random walk move kernel.

    The returned step function operates on a SINGLE walker.
    Energy function and params are captured via closure.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(
        rng_key: jax.Array,
        state: RandomWalkState,
        likelihood_constraint: float,
    ) -> tuple[RandomWalkState, MoveInfo]:
        # Propose: random Gaussian displacement
        dpos = state.step_size * jax.random.normal(
            rng_key, shape=state.positions.shape
        )
        new_positions = state.positions + dpos

        # Evaluate energy at proposed position
        new_energy = energy_fn(
            params, new_positions, state.types, box=state.box
        )

        # Accept if energy below constraint (NS rejection sampling)
        accepted = new_energy < likelihood_constraint

        # Conditional update: keep old state if rejected
        out_positions = jnp.where(accepted, new_positions, state.positions)
        out_energy = jnp.where(accepted, new_energy, state.energy)

        new_state = RandomWalkState(
            positions=out_positions,
            types=state.types,
            energy=out_energy,
            box=state.box,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-out_energy,
            n_evaluations=1,
        )

        return new_state, info

    return step


def as_top_level_api(
    energy_fn: Any,
    params: Params,
    step_size: float = 0.1,
) -> MoveKernel:
    """Convenient top-level API."""
    kernel = build_kernel(energy_fn, params)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)
