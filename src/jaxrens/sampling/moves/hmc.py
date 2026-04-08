"""Hamiltonian Monte Carlo (HMC) move for nested sampling.

Uses leapfrog integration to propose moves along Hamiltonian trajectories,
with NS accept/reject based on the likelihood constraint (E < Emax).

HMC is particularly efficient for smooth, differentiable potentials where
forces (energy gradients) are available. The leapfrog integrator preserves
the symplectic structure, giving high acceptance rates.

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types


class HMCState(NamedTuple):
    """State for the HMC move."""

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
    step_size: float = 0.01,
) -> HMCState:
    """Create initial HMC state."""
    return HMCState(
        positions=positions,
        types=types,
        energy=jnp.asarray(energy),
        box=box,
        step_size=jnp.asarray(step_size),
    )


def build_kernel(
    energy_fn: Any,
    params: Params,
    n_leapfrog: int = 10,
):
    """Build an HMC move kernel.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol. Must be
            differentiable w.r.t. positions (argnums=1).
        params: Opaque pytree of backend parameters.
        n_leapfrog: Number of leapfrog integration steps.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    grad_energy = jax.grad(energy_fn, argnums=1)

    def step(
        rng_key: jax.Array,
        state: HMCState,
        likelihood_constraint: float,
    ) -> tuple[HMCState, MoveInfo]:
        # Sample random momentum
        momentum = jax.random.normal(rng_key, shape=state.positions.shape)

        # Initial kinetic energy
        kinetic_init = 0.5 * jnp.sum(momentum**2)

        # Leapfrog integration via lax.scan
        def leapfrog_step(carry, _):
            pos, mom = carry
            # Half-step momentum
            grad = grad_energy(params, pos, state.types, box=state.box)
            mom = mom - 0.5 * state.step_size * grad
            # Full-step position
            pos = pos + state.step_size * mom
            # Half-step momentum
            grad = grad_energy(params, pos, state.types, box=state.box)
            mom = mom - 0.5 * state.step_size * grad
            return (pos, mom), None

        (new_positions, new_momentum), _ = jax.lax.scan(
            leapfrog_step,
            (state.positions, momentum),
            None,
            length=n_leapfrog,
        )

        # Evaluate energy at proposed position
        new_energy = energy_fn(params, new_positions, state.types, box=state.box)

        # Final kinetic energy
        kinetic_final = 0.5 * jnp.sum(new_momentum**2)

        # Hamiltonian-based acceptance: total energy should be conserved
        # But in NS, the primary constraint is E < Emax
        # We also require Hamiltonian is roughly preserved (prevents drift)
        delta_H = (new_energy + kinetic_final) - (state.energy + kinetic_init)
        hamiltonian_ok = delta_H < 1.0  # generous tolerance
        ns_ok = new_energy < likelihood_constraint
        accepted = ns_ok & hamiltonian_ok

        out_positions = jnp.where(accepted, new_positions, state.positions)
        out_energy = jnp.where(accepted, new_energy, state.energy)

        new_state = HMCState(
            positions=out_positions,
            types=state.types,
            energy=out_energy,
            box=state.box,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-out_energy,
            n_evaluations=2 * n_leapfrog,  # 2 grad evals per leapfrog step
        )

        return new_state, info

    return step


def as_top_level_api(
    energy_fn: Any,
    params: Params,
    step_size: float = 0.01,
    n_leapfrog: int = 10,
) -> MoveKernel:
    """Convenient top-level API."""
    kernel = build_kernel(energy_fn, params, n_leapfrog=n_leapfrog)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)
