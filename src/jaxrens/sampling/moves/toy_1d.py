"""The two step types of the RENS-paper 1-D toy model: lattice and distance.

The toy model of Unglert, Pártay and Madsen (J. Chem. Theory Comput. **21**,
7304, 2025) is two particles in a periodic 1-D box, driven by two moves in a
1:1 ratio: one that changes the box length ``a`` and one that changes the
interparticle distance.

Neither of the general 3-D cell moves can play the first role.
``moves.volume`` scales the cell *isotropically*, so a cell of
``diag(a, 1, 1)`` — the embedding :class:`~jaxrens.backends.toy.RENSToyBackend`
relies on, which makes ``V = a`` and turns the standard NPT enthalpy into the
paper's eq 18 — would immediately stop being diagonal-with-unit-yz. These
kernels keep that structure exact.

Both are single-walker functions, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.sampling.base import MoveInfo


def build_lattice_kernel(
    backend: Any,
    n_atoms: int = 2,
    max_vol_per_atom: float = 5.0,
    min_vol_per_atom: float = 0.25,
):
    """Build the 1-D lattice move: change the box length ``a``.

    Proposes ``a' = a + step_size * N(0, 1)`` and rescales the particles'
    fractional coordinates, so the move is the 1-D analogue of a volume move.
    The paper samples ``a`` under a *uniform* prior (its eq 3), so there is no
    Jacobian factor here — acceptance is the hard likelihood constraint plus
    the box-length bounds.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Particle count, used to convert the per-atom volume bounds.
        max_vol_per_atom: Upper bound on ``a / n_atoms``.
        min_vol_per_atom: Lower bound on ``a / n_atoms``.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    a_max = max_vol_per_atom * n_atoms
    a_min = min_vol_per_atom * n_atoms

    def step(rng_key, state, likelihood_constraint):
        a = state.cell[0, 0]
        da = state.step_size * jax.random.normal(rng_key)
        new_a = jnp.abs(a + da)

        in_bounds = (new_a >= a_min) & (new_a <= a_max)

        # Rescale fractional coordinates; y/z do not enter the toy energy and
        # are left untouched so the particles stay on the axis.
        scale = new_a / a
        new_positions = state.positions.at[:, 0].multiply(scale)
        new_cell = state.cell.at[0, 0].set(new_a)

        result = backend(
            new_positions,
            state.types,
            new_cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        new_energy = result.energy

        accepted = in_bounds & (new_energy < likelihood_constraint)
        # 2 = cell-geometry rejection, 1 = likelihood; see MoveInfo.
        reject_reason = jnp.where(
            accepted, 0, jnp.where(in_bounds, 1, 2)
        ).astype(jnp.int32)

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            cell=jnp.where(accepted, new_cell, state.cell),
            energy=jnp.where(accepted, new_energy, state.energy),
            max_neighbor_count=jnp.maximum(
                state.max_neighbor_count, result.max_neighbor_count
            ),
            overflow=state.overflow | result.overflow,
        )
        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
            reject_reason=reject_reason,
        )
        return new_state, info

    return step


def build_distance_kernel(backend: Any):
    """Build the 1-D distance move: displace one particle along the axis.

    Picks a particle at random and shifts its ``x`` by a Gaussian step,
    wrapping back into ``[0, a)``. Only ``x`` moves: ``y`` and ``z`` are not
    arguments of the toy energy, and letting them drift would add dimensions
    the model does not have.

    Args:
        backend: EnergyBackend instance.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        k_atom, k_step = jax.random.split(rng_key)
        a = state.cell[0, 0]

        idx = jax.random.randint(
            k_atom, shape=(), minval=0, maxval=state.positions.shape[0]
        )
        dx = state.step_size * jax.random.normal(k_step)
        new_x = jnp.mod(state.positions[idx, 0] + dx, a)
        new_positions = state.positions.at[idx, 0].set(new_x)

        result = backend(
            new_positions,
            state.types,
            state.cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        new_energy = result.energy
        accepted = new_energy < likelihood_constraint

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            max_neighbor_count=jnp.maximum(
                state.max_neighbor_count, result.max_neighbor_count
            ),
            overflow=state.overflow | result.overflow,
        )
        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
        )
        return new_state, info

    return step
