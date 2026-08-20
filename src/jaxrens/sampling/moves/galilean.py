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

Species scoping
---------------
Passing ``species=(code, ...)`` restricts the move to the atoms of those
type codes and holds every other atom fixed — a GMC trajectory confined
to one element's sublattice.  This is what makes independent step sizes
per sublattice possible: register one scoped kernel per element and the
MWG per-move step-size array (plus the per-move bisection in
``adaptation/manager.py``) tunes each one on its own acceptance rate.
Useful when one sublattice melts well before the other, where a single
joint move is throttled by whichever sublattice is stiffer.

The move stays a valid NS kernel: confining it to a linear subspace is
just GMC with the frozen coordinates as fixed parameters.  Reflection
off the *projected* gradient is still volume-preserving, the
flip-on-reject still gives reversibility, and the ``Emax`` gate is still
on the total energy.  Ergodicity comes from composing the per-species
moves, exactly as it does for ``single_atom``.

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from jaxrens.backends.base import eval_energy_and_forces
from jaxrens.base import MoveInfo


def _normalize(
    v: Float[Array, "N 3"], mask: Float[Array, "N 1"] | None
) -> Float[Array, "N 3"]:
    """Mask ``v`` to the moving subspace, then normalize to unit length.

    Masking *before* normalizing is what keeps the result a unit vector of
    the restricted subspace; the frozen rows are exactly zero, so they
    contribute nothing to the norm and never move.
    """
    if mask is not None:
        v = v * mask
    norm = jnp.sqrt(jnp.sum(v**2))
    return v / jnp.maximum(norm, 1e-10)


def _random_direction(
    key: Key[Array, ""], shape: tuple, mask: Float[Array, "N 1"] | None = None
) -> Float[Array, "N 3"]:
    """Generate a random unit direction vector."""
    return _normalize(jax.random.normal(key, shape), mask)


def _perturb_direction(
    key: Key[Array, ""],
    direction: Float[Array, "N 3"],
    perturb_angle: float = 0.1,
    mask: Float[Array, "N 1"] | None = None,
) -> Float[Array, "N 3"]:
    """Perturb a direction vector by a small random angle."""
    noise = perturb_angle * jax.random.normal(key, direction.shape)
    return _normalize(direction + noise, mask)


def build_kernel(
    backend: Any,
    n_reflect: int = 5,
    perturb_angle: float = 0.1,
    use_forces: bool = True,
    species: tuple[int, ...] | None = None,
    direction_field: str = "direction",
):
    """Build a Galilean Monte Carlo move kernel.

    Args:
        backend: EnergyBackend instance.
        n_reflect: Number of trajectory steps (with reflections).
        perturb_angle: Angle (radians) to perturb direction between moves.
        use_forces: If True, reflect using gradient (forces). If False,
            use random reflection (cheaper but less efficient).
        species: Type codes this move may displace; every other atom is
            held fixed for the whole trajectory.  ``None`` (the default)
            moves all atoms.  See the module docstring.
        direction_field: Name of the MCState field holding this move's
            persistent direction.  Species-scoped moves **must** each use a
            distinct field: ``build_mwg`` unions ``extra_state_fields`` by
            name, so two scoped moves sharing ``"direction"`` would zero out
            each other's persistence on every call — destroying the very
            thing GMC relies on.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)

    Note:
        A scoped move whose species is absent from a given walker degenerates
        to a no-op that always "accepts" (the identity move trivially satisfies
        detailed balance, but its acceptance rate carries no information, so
        its adapted step size is meaningless).  The resolver rejects species
        absent from the system; this can still arise per-walker under
        composition-changing ensembles (semi-grand, alchemical moves).
    """

    def step(rng_key, state, likelihood_constraint):
        key_init, key_perturb, key_reflect = jax.random.split(rng_key, 3)

        prev_direction = getattr(state, direction_field)

        # Moving-subspace mask, derived from ``state.types`` at trace time
        # rather than baked in at build time: swap/alchemical moves mutate
        # types, so a positional mask would silently go stale.
        if species is None:
            mask = None
        else:
            codes = jnp.asarray(species, dtype=state.types.dtype)
            mask = jnp.isin(state.types, codes)[:, None].astype(
                state.positions.dtype
            )

        # Initialize direction if zero (first call), otherwise perturb
        dir_norm = jnp.sqrt(jnp.sum(prev_direction**2))
        is_zero = dir_norm < 1e-8
        direction = jnp.where(
            is_zero,
            _random_direction(key_init, state.positions.shape, mask),
            _perturb_direction(
                key_perturb, prev_direction, perturb_angle, mask
            ),
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
                #
                # Under species scoping the normal is the *projection* of the
                # gradient onto the moving subspace (``_normalize`` masks
                # before normalizing) — that is the correct constraint-surface
                # normal for the restricted problem, not merely bookkeeping to
                # keep the frozen rows at zero.
                f_hat = _normalize(force, mask)
                reflected_dir = (
                    direction - 2.0 * jnp.sum(f_hat * direction) * f_hat
                )
            else:
                reflected_dir = _random_direction(
                    step_key, direction.shape, mask
                )

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
            max_neighbor_count=acc_count,
            overflow=acc_overflow,
            **{
                direction_field: jnp.where(
                    accepted, final_dir, -prev_direction
                )
            },
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
