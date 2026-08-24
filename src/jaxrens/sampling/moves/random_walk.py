"""Random walk move: Gaussian perturbation with accept/reject.

The simplest NS move. Proposes a random displacement of all atom positions,
accepts if the new energy is below the likelihood constraint (Emax).

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.sampling.base import MoveInfo


def build_kernel(backend: Any):
    """Build a random walk move kernel.

    The returned step function operates on a SINGLE walker.
    Backend is captured via closure.

    Args:
        backend: EnergyBackend instance.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        # Propose: random Gaussian displacement
        dpos = state.step_size * jax.random.normal(
            rng_key, shape=state.positions.shape
        )
        new_positions = state.positions + dpos

        # Evaluate energy at proposed position
        result = backend(
            new_positions,
            state.types,
            state.cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        new_energy, count, overflow = (
            result.energy,
            result.max_neighbor_count,
            result.overflow,
        )

        # Accept if energy below constraint (NS rejection sampling)
        accepted = new_energy < likelihood_constraint

        # Conditional update: keep old state if rejected
        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            max_neighbor_count=jnp.maximum(state.max_neighbor_count, count),
            overflow=state.overflow | overflow,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
        )

        return new_state, info

    return step
