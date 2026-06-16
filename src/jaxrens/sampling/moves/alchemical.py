"""Alchemical move kernels for nested sampling.

- build_morph_kernel: change one atom's species (semi-grand-canonical)
- build_shift_kernel: random translation of all atoms (rigid shift)

Single-walker functions, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo


def build_morph_kernel(backend: Any, n_species: int):
    """Build an atom morph kernel.

    Selects a random atom, changes its type to a random different species,
    and accepts if the new energy is below the NS constraint.

    Args:
        backend: EnergyBackend instance.
        n_species: Total number of distinct species.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        key_atom, key_species = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        atom_idx = jax.random.randint(key_atom, (), 0, n_atoms)

        current_type = state.types[atom_idx]
        candidate = jax.random.randint(key_species, (), 0, n_species - 1)
        new_type = candidate + (candidate >= current_type).astype(jnp.int32)

        new_types = state.types.at[atom_idx].set(new_type)
        new_energy, count, overflow = backend(
            state.positions, new_types, state.cell, state.max_neighbors,
            ensemble_params=state.ensemble_params,
        ).legacy()

        accepted = new_energy < likelihood_constraint

        new_state = state.set(
            types=jnp.where(accepted, new_types, state.types),
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


def build_shift_kernel(backend: Any):
    """Build a random shift kernel.

    Applies a uniform random translation to all atoms simultaneously.

    Args:
        backend: EnergyBackend instance.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        shift = state.step_size * jax.random.normal(rng_key, (3,))
        new_positions = state.positions + shift[None, :]

        new_energy, count, overflow = backend(
            new_positions, state.types, state.cell, state.max_neighbors,
            ensemble_params=state.ensemble_params,
        ).legacy()
        accepted = new_energy < likelihood_constraint

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
