"""Random walk move: Gaussian perturbation with accept/reject.

The simplest NS move. Proposes a random displacement of all atom positions,
accepts if the new energy is below the likelihood constraint (Emax).

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.state.mc_state import MCState
from jaxrens.types import Params


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
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
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
        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
        )

        return new_state, info

    return step
