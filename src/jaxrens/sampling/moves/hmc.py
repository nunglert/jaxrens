"""Hamiltonian Monte Carlo (HMC) move for nested sampling.

Uses leapfrog integration to propose moves along Hamiltonian trajectories,
with NS accept/reject based on the likelihood constraint (E < Emax).

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo


def build_kernel(
    backend: Any,
    n_leapfrog: int = 10,
):
    """Build an HMC move kernel.

    Args:
        backend: EnergyBackend instance. Must be differentiable w.r.t. positions.
        n_leapfrog: Number of leapfrog integration steps.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        # Sample random momentum
        momentum = jax.random.normal(rng_key, shape=state.positions.shape)
        kinetic_init = 0.5 * jnp.sum(momentum**2)

        max_neighbors = state.max_neighbors
        ensemble_params = state.ensemble_params

        def energy_with_aux(pos):
            e, count, overflow = backend(
                pos, state.types, state.cell, max_neighbors,
                ensemble_params=ensemble_params,
            )
            return e, (count, overflow)

        # Leapfrog integration via lax.scan
        def leapfrog_step(carry, _):
            pos, mom, acc_count, acc_overflow = carry

            # Half-step momentum (using forces from autodiff)
            (_, (count, overflow)), grad = jax.value_and_grad(
                energy_with_aux, has_aux=True
            )(pos)
            acc_count = jnp.maximum(acc_count, count)
            acc_overflow = acc_overflow | overflow
            mom = mom - 0.5 * state.step_size * grad

            # Full-step position
            pos = pos + state.step_size * mom

            # Half-step momentum
            (_, (count, overflow)), grad = jax.value_and_grad(
                energy_with_aux, has_aux=True
            )(pos)
            acc_count = jnp.maximum(acc_count, count)
            acc_overflow = acc_overflow | overflow
            mom = mom - 0.5 * state.step_size * grad

            return (pos, mom, acc_count, acc_overflow), None

        init_carry = (
            state.positions, momentum,
            state.max_neighbor_count, state.overflow,
        )
        (new_positions, new_momentum, acc_count, acc_overflow), _ = jax.lax.scan(
            leapfrog_step, init_carry, None, length=n_leapfrog,
        )

        # Evaluate energy at proposed position
        new_energy, count, overflow = backend(
            new_positions, state.types, state.cell, max_neighbors,
            ensemble_params=ensemble_params,
        )
        acc_count = jnp.maximum(acc_count, count)
        acc_overflow = acc_overflow | overflow

        # Kinetic energy for Hamiltonian check
        kinetic_final = 0.5 * jnp.sum(new_momentum**2)
        delta_H = (new_energy + kinetic_final) - (state.energy + kinetic_init)
        hamiltonian_ok = delta_H < 1.0
        ns_ok = new_energy < likelihood_constraint
        accepted = ns_ok & hamiltonian_ok

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            max_neighbor_count=acc_count,
            overflow=acc_overflow,
        )

        # leapfrog_step makes 2 value_and_grad calls per step (half-step each side).
        # The final energy evaluation (line 84) is a plain backend call, not value_and_grad.
        # Total: 2*n_leapfrog evaluations (all value_and_grad) + 1 plain eval = 2*n_leapfrog+1.
        # n_grad_evaluations = 2*n_leapfrog (the value_and_grad subset only).
        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=2 * n_leapfrog + 1,
            n_grad_evaluations=jnp.int32(2 * n_leapfrog),
        )

        return new_state, info

    return step
