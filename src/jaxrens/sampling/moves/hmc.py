"""Hamiltonian Monte Carlo (HMC) move for nested sampling.

Uses leapfrog integration to propose moves along Hamiltonian trajectories,
with NS accept/reject based on the likelihood constraint (E < Emax).

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.backends.base import eval_energy_and_forces
from jaxrens.base import MoveInfo
from jaxrens.unvalidated import unvalidated


@unvalidated(
    concern=(
        "no production NS run has used this move.  Two specifics: the "
        "acceptance test is a hard `delta_H < 1.0` cut -- an absolute, "
        "unit-dependent energy tolerance rather than a Metropolis "
        "`exp(-delta_H)` test -- so it is not a detailed-balance-preserving "
        "accept; and the leapfrog timestep reuses `state.step_size`, which "
        "the adaptation tunes as a random-walk displacement, not as an "
        "integrator timestep"
    ),
    since="0.2.2",
    clears_when=(
        "a multi-walker NS run whose HMC accept rate and energy trace track "
        "the random-walk move on the same system"
    ),
)
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

        def eval_forces(pos):
            # Backend uses its native force path when available, else autodiff.
            return eval_energy_and_forces(
                backend,
                pos,
                state.types,
                state.cell,
                max_neighbors,
                ensemble_params=ensemble_params,
            )

        # Leapfrog integration via lax.scan. The potential gradient is
        # ``-forces``, so each half-step momentum kick ``p -= step * dU/dq``
        # is written as ``p += step * forces``.
        def leapfrog_step(carry, _):
            pos, mom, acc_count, acc_overflow = carry

            # Half-step momentum
            res = eval_forces(pos)
            acc_count = jnp.maximum(acc_count, res.max_neighbor_count)
            acc_overflow = acc_overflow | res.overflow
            mom = mom + 0.5 * state.step_size * res.forces

            # Full-step position
            pos = pos + state.step_size * mom

            # Half-step momentum
            res = eval_forces(pos)
            acc_count = jnp.maximum(acc_count, res.max_neighbor_count)
            acc_overflow = acc_overflow | res.overflow
            mom = mom + 0.5 * state.step_size * res.forces

            return (pos, mom, acc_count, acc_overflow), None

        init_carry = (
            state.positions,
            momentum,
            state.max_neighbor_count,
            state.overflow,
        )
        (
            new_positions,
            new_momentum,
            acc_count,
            acc_overflow,
        ), _ = jax.lax.scan(
            leapfrog_step,
            init_carry,
            None,
            length=n_leapfrog,
        )

        # Evaluate energy at proposed position
        result = backend(
            new_positions,
            state.types,
            state.cell,
            max_neighbors,
            ensemble_params=ensemble_params,
        )
        new_energy, count, overflow = (
            result.energy,
            result.max_neighbor_count,
            result.overflow,
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
