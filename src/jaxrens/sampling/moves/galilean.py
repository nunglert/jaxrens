"""Galilean Monte Carlo (GMC) move with elastic reflection.

A persistent-direction move that follows a straight-line trajectory,
reflecting off the energy constraint surface when violated. Much more
efficient than random walk for exploring constrained regions.

Algorithm per NS step:
1. Initialize or perturb the velocity direction
2. For each of n_reflect iterations:
   a. Step: new_pos = pos + step_size * direction
   b. Evaluate energy at new_pos
   c. If energy >= Emax: reflect direction off constraint surface
      using forces: direction -= 2 * (F_hat . direction) * F_hat
3. Accept if final energy < Emax, else reject and flip direction

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo


def _random_direction(key: jax.Array, shape: tuple) -> jnp.ndarray:
    """Generate a random unit direction vector."""
    d = jax.random.normal(key, shape)
    norm = jnp.sqrt(jnp.sum(d**2))
    return d / jnp.maximum(norm, 1e-10)


def _perturb_direction(
    key: jax.Array,
    direction: jnp.ndarray,
    perturb_angle: float = 0.1,
) -> jnp.ndarray:
    """Perturb a direction vector by a small random angle."""
    noise = perturb_angle * jax.random.normal(key, direction.shape)
    new_dir = direction + noise
    norm = jnp.sqrt(jnp.sum(new_dir**2))
    return new_dir / jnp.maximum(norm, 1e-10)


def build_kernel(
    backend: Any,
    n_reflect: int = 5,
    perturb_angle: float = 0.1,
    use_forces: bool = True,
):
    """Build a Galilean Monte Carlo move kernel.

    Args:
        backend: EnergyBackend instance.
        n_reflect: Number of trajectory steps (with reflections).
        perturb_angle: Angle (radians) to perturb direction between moves.
        use_forces: If True, reflect using gradient (forces). If False,
            use random reflection (cheaper but less efficient).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        key_init, key_perturb, key_reflect = jax.random.split(rng_key, 3)

        # Initialize direction if zero (first call), otherwise perturb
        dir_norm = jnp.sqrt(jnp.sum(state.direction**2))
        is_zero = dir_norm < 1e-8
        direction = jnp.where(
            is_zero,
            _random_direction(key_init, state.positions.shape),
            _perturb_direction(key_perturb, state.direction, perturb_angle),
        )

        max_neighbors = state.max_neighbors
        ensemble_params = state.ensemble_params

        # Reflection loop via lax.scan
        def reflect_step(carry, step_key):
            pos, direction, energy, acc_count, acc_overflow = carry

            # Propose step
            new_pos = pos + state.step_size * direction

            if use_forces:
                # Energy + forces via autodiff through backend
                def energy_fn(p):
                    e, count, overflow = backend(
                        p, state.types, state.cell, max_neighbors,
                        ensemble_params=ensemble_params,
                    )
                    return e, (count, overflow)

                (new_energy, (count, overflow)), grad = jax.value_and_grad(
                    energy_fn, has_aux=True
                )(new_pos)
            else:
                new_energy, count, overflow = backend(
                    new_pos, state.types, state.cell, max_neighbors,
                    ensemble_params=ensemble_params,
                )
                grad = None

            # Accumulate overflow tracking
            acc_count = jnp.maximum(acc_count, count)
            acc_overflow = acc_overflow | overflow

            # Check if constraint is violated
            violated = new_energy >= likelihood_constraint

            if use_forces:
                grad_norm = jnp.sqrt(jnp.sum(grad**2))
                f_hat = grad / jnp.maximum(grad_norm, 1e-10)
                reflected_dir = direction - 2.0 * jnp.sum(f_hat * direction) * f_hat
            else:
                reflected_dir = _random_direction(step_key, direction.shape)

            direction_out = jnp.where(violated, reflected_dir, direction)
            pos_out = jnp.where(violated, pos, new_pos)
            energy_out = jnp.where(violated, energy, new_energy)

            return (pos_out, direction_out, energy_out, acc_count, acc_overflow), None

        reflect_keys = jax.random.split(key_reflect, n_reflect)
        init_carry = (
            state.positions, direction, state.energy,
            state.max_neighbor_count, state.overflow,
        )
        (final_pos, final_dir, final_energy, acc_count, acc_overflow), _ = (
            jax.lax.scan(reflect_step, init_carry, reflect_keys)
        )

        # Accept if final energy < Emax
        accepted = final_energy < likelihood_constraint

        # If rejected: revert to initial positions, flip direction
        new_state = state.set(
            positions=jnp.where(accepted, final_pos, state.positions),
            energy=jnp.where(accepted, final_energy, state.energy),
            direction=jnp.where(accepted, final_dir, -state.direction),
            max_neighbor_count=acc_count,
            overflow=acc_overflow,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=n_reflect,
        )

        return new_state, info

    return step
