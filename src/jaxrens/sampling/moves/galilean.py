"""Galilean Monte Carlo (GMC) move with elastic reflection.

A persistent-direction move that follows a straight-line trajectory,
reflecting off the energy constraint surface when violated. Much more
efficient than random walk for exploring constrained regions.

Algorithm per NS step (Baldock semantics — mirrors
``jaxnest_dev/src/jaxnest/mcmc.py::create_GMC_atom_walk``):

1. Initialize or perturb the velocity direction.
2. For each of ``n_reflect`` iterations:

   a. Step: ``new_pos = pos + step_size * direction``.  Position always
      advances — even into the violated region.
   b. Evaluate energy at ``new_pos``.  NaN energies are coerced to
      ``+inf`` so they always trigger reflection (matches legacy
      ``mcmc.py:788-792``).
   c. If ``new_energy >= Emax``: reflect direction off the constraint
      surface using forces: ``direction -= 2 * (F_hat . direction) * F_hat``.

3. Accept iff the trajectory's *final* energy is below ``Emax`` — i.e.
   the walker has exited the violated region by the last step.  Reject
   reverts to the initial position and flips the direction.

The non-obvious bit: position drifts through violated regions inside
the scan; the final-state gate is what enforces in-region acceptance.
This is the standard Baldock construction.  An earlier variant of this
kernel reverted ``pos``/``energy`` on every violation, which made the
final accept-gate trivially True (carry always held the last *good*
state) and silently NaN-corrupted under NaN-producing backends.

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from jaxrens.backends.base import eval_energy_and_forces
from jaxrens.base import MoveInfo


def _random_direction(
    key: Key[Array, ""], shape: tuple
) -> Float[Array, "N 3"]:
    """Generate a random unit direction vector."""
    d = jax.random.normal(key, shape)
    norm = jnp.sqrt(jnp.sum(d**2))
    return d / jnp.maximum(norm, 1e-10)


def _perturb_direction(
    key: Key[Array, ""],
    direction: Float[Array, "N 3"],
    perturb_angle: float = 0.1,
) -> Float[Array, "N 3"]:
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
                # Energy + forces; backend uses its native force path when
                # available, otherwise eval_energy_and_forces autodiffs.
                res = eval_energy_and_forces(
                    backend,
                    new_pos,
                    state.types,
                    state.cell,
                    max_neighbors,
                    ensemble_params=ensemble_params,
                )
                force = res.forces
            else:
                res = backend(
                    new_pos,
                    state.types,
                    state.cell,
                    max_neighbors,
                    ensemble_params=ensemble_params,
                )
                force = None
            new_energy, count, overflow = (
                res.energy,
                res.max_neighbor_count,
                res.overflow,
            )

            # Accumulate overflow tracking
            acc_count = jnp.maximum(acc_count, count)
            acc_overflow = acc_overflow | overflow

            # NaN trap: a NaN energy (e.g. from a degenerate proposed
            # geometry under a model with r^-N singularities) must
            # trigger reflection just like an over-Emax energy.  Without
            # this, ``NaN >= Emax`` evaluates to False, the carry
            # advances with NaN energy, and the final accept gate
            # silently reports rejected.  Mirrors mcmc.py:788-792.
            new_energy = jnp.where(jnp.isnan(new_energy), jnp.inf, new_energy)
            violated = new_energy >= likelihood_constraint

            if use_forces:
                # Reflect off the constraint surface normal. The force is
                # antiparallel to the energy gradient, and the reflection
                # d - 2(n·d)n is invariant under n -> -n, so using the force
                # direction is identical to using the gradient.
                f_norm = jnp.sqrt(jnp.sum(force**2))
                f_hat = force / jnp.maximum(f_norm, 1e-10)
                reflected_dir = (
                    direction - 2.0 * jnp.sum(f_hat * direction) * f_hat
                )
            else:
                reflected_dir = _random_direction(step_key, direction.shape)

            # Position and energy advance unconditionally — the walker
            # drifts through violated regions.  Only the direction
            # reflects on violation.  The trajectory's terminal energy
            # is what gates acceptance, not the per-step energies.
            direction_out = jnp.where(violated, reflected_dir, direction)
            pos_out = new_pos
            energy_out = new_energy

            return (
                pos_out,
                direction_out,
                energy_out,
                acc_count,
                acc_overflow,
            ), None

        reflect_keys = jax.random.split(key_reflect, n_reflect)
        init_carry = (
            state.positions,
            direction,
            state.energy,
            state.max_neighbor_count,
            state.overflow,
        )
        (
            final_pos,
            final_dir,
            final_energy,
            acc_count,
            acc_overflow,
        ), _ = jax.lax.scan(reflect_step, init_carry, reflect_keys)

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

        # n_reflect calls to value_and_grad (when use_forces=True),
        # or n_reflect plain energy calls (when use_forces=False).
        # n_grad_evaluations tracks the value_and_grad subset.
        n_grad = n_reflect if use_forces else 0
        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=n_reflect,
            n_grad_evaluations=jnp.int32(n_grad),
        )

        return new_state, info

    return step
