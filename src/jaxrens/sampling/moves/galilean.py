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

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types


class GalileanState(NamedTuple):
    """State for the Galilean Monte Carlo move."""

    positions: jnp.ndarray  # (n_atoms, 3)
    types: jnp.ndarray  # (n_atoms,)
    energy: jnp.ndarray  # scalar
    box: jnp.ndarray | None  # (3, 3) or None
    direction: jnp.ndarray  # (n_atoms, 3) — unit velocity direction
    step_size: jnp.ndarray  # scalar


def init(
    positions: Positions,
    types: Types,
    energy: float,
    box: Box | None = None,
    step_size: float = 0.1,
    direction: jnp.ndarray | None = None,
) -> GalileanState:
    """Create initial Galilean state.

    If direction is None, it will be initialized to zeros and randomized
    on the first step.
    """
    if direction is None:
        direction = jnp.zeros_like(positions)
    return GalileanState(
        positions=positions,
        types=types,
        energy=jnp.asarray(energy),
        box=box,
        direction=direction,
        step_size=jnp.asarray(step_size),
    )


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
    """Perturb a direction vector by a small random angle.

    Adds a random component and renormalizes. The perturb_angle controls
    the magnitude of the random perturbation relative to the original direction.
    """
    noise = perturb_angle * jax.random.normal(key, direction.shape)
    new_dir = direction + noise
    norm = jnp.sqrt(jnp.sum(new_dir**2))
    return new_dir / jnp.maximum(norm, 1e-10)


def build_kernel(
    energy_fn: Any,
    params: Params,
    n_reflect: int = 5,
    perturb_angle: float = 0.1,
    use_forces: bool = True,
):
    """Build a Galilean Monte Carlo move kernel.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.
        n_reflect: Number of trajectory steps (with reflections).
        perturb_angle: Angle (radians) to perturb direction between moves.
        use_forces: If True, reflect using gradient (forces). If False,
            use random reflection (cheaper but less efficient).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    if use_forces:
        grad_energy = jax.grad(energy_fn, argnums=1)

    def step(
        rng_key: jax.Array,
        state: GalileanState,
        likelihood_constraint: float,
    ) -> tuple[GalileanState, MoveInfo]:
        key_init, key_perturb, key_reflect = jax.random.split(rng_key, 3)

        # Initialize direction if zero (first call), otherwise perturb
        dir_norm = jnp.sqrt(jnp.sum(state.direction**2))
        is_zero = dir_norm < 1e-8
        direction = jnp.where(
            is_zero,
            _random_direction(key_init, state.positions.shape),
            _perturb_direction(key_perturb, state.direction, perturb_angle),
        )

        # Reflection loop via lax.scan
        def reflect_step(carry, step_key):
            pos, direction, energy, n_evals = carry

            # Propose step
            new_pos = pos + state.step_size * direction

            # Evaluate energy
            new_energy = energy_fn(params, new_pos, state.types, box=state.box)

            # Check if constraint is violated
            violated = new_energy >= likelihood_constraint

            if use_forces:
                # Reflect using gradient of energy (force direction)
                grad = grad_energy(params, new_pos, state.types, box=state.box)
                grad_norm = jnp.sqrt(jnp.sum(grad**2))
                f_hat = grad / jnp.maximum(grad_norm, 1e-10)
                # Elastic reflection: d_new = d - 2*(F.d)*F
                reflected_dir = direction - 2.0 * jnp.sum(f_hat * direction) * f_hat
            else:
                # Random reflection (fallback)
                reflected_dir = _random_direction(step_key, direction.shape)

            # Apply reflection only if violated
            direction_out = jnp.where(violated, reflected_dir, direction)

            # Keep position: if violated, stay at old pos; if not, advance
            pos_out = jnp.where(violated, pos, new_pos)
            energy_out = jnp.where(violated, energy, new_energy)

            return (pos_out, direction_out, energy_out, n_evals + 1), None

        reflect_keys = jax.random.split(key_reflect, n_reflect)
        init_carry = (state.positions, direction, state.energy, 0)
        (final_pos, final_dir, final_energy, n_evals), _ = jax.lax.scan(
            reflect_step, init_carry, reflect_keys
        )

        # Accept if final energy < Emax
        accepted = final_energy < likelihood_constraint

        # If rejected: revert to initial positions, flip direction
        out_positions = jnp.where(accepted, final_pos, state.positions)
        out_energy = jnp.where(accepted, final_energy, state.energy)
        out_direction = jnp.where(accepted, final_dir, -state.direction)

        new_state = GalileanState(
            positions=out_positions,
            types=state.types,
            energy=out_energy,
            box=state.box,
            direction=out_direction,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-out_energy,
            n_evaluations=n_evals,
        )

        return new_state, info

    return step


def as_top_level_api(
    energy_fn: Any,
    params: Params,
    step_size: float = 0.1,
    n_reflect: int = 5,
    **kwargs: Any,
) -> MoveKernel:
    """Convenient top-level API."""
    kernel = build_kernel(energy_fn, params, n_reflect=n_reflect, **kwargs)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)
