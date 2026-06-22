"""Single-atom move kernels for nested sampling.

Moves that perturb one atom at a time, efficient for large systems where
displacing all atoms simultaneously has low acceptance.

- build_kernel: move one random atom per step
- build_sweep_kernel: sweep through all atoms sequentially via lax.scan
- build_swap_kernel: swap species of two random atoms (multi-component)

Single-walker functions, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo


def build_kernel(backend: Any):
    """Build a single-atom random displacement kernel.

    Args:
        backend: EnergyBackend instance.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        key_atom, key_disp = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        atom_idx = jax.random.randint(key_atom, (), 0, n_atoms)

        displacement = state.step_size * jax.random.normal(key_disp, (3,))
        new_positions = state.positions.at[atom_idx].add(displacement)

        result = backend(
            new_positions,
            state.types,
            state.cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        accepted = result.energy < likelihood_constraint

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, result.energy, state.energy),
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


def build_sweep_kernel(backend: Any, n_atoms: int):
    """Build a single-atom sweep kernel.

    Sweeps through all atoms sequentially (via lax.scan), displacing each.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms (needed for scan length).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        keys = jax.random.split(rng_key, n_atoms)
        max_neighbors = state.max_neighbors
        ensemble_params = state.ensemble_params

        def sweep_one(carry, key_and_idx):
            positions, energy, n_accepted, acc_count, acc_overflow = carry
            key, idx = key_and_idx

            displacement = state.step_size * jax.random.normal(key, (3,))
            new_positions = positions.at[idx].add(displacement)

            result = backend(
                new_positions,
                state.types,
                state.cell,
                max_neighbors,
                ensemble_params=ensemble_params,
            )
            accepted = result.energy < likelihood_constraint

            out_positions = jnp.where(accepted, new_positions, positions)
            out_energy = jnp.where(accepted, result.energy, energy)
            acc_count = jnp.maximum(acc_count, result.max_neighbor_count)
            acc_overflow = acc_overflow | result.overflow

            return (
                out_positions,
                out_energy,
                n_accepted + accepted.astype(jnp.int32),
                acc_count,
                acc_overflow,
            ), None

        atom_indices = jnp.arange(n_atoms)
        init_carry = (
            state.positions,
            state.energy,
            jnp.int32(0),
            state.max_neighbor_count,
            state.overflow,
        )
        (
            final_positions,
            final_energy,
            n_accepted,
            acc_count,
            acc_overflow,
        ), _ = jax.lax.scan(sweep_one, init_carry, (keys, atom_indices))

        new_state = state.set(
            positions=final_positions,
            energy=final_energy,
            max_neighbor_count=acc_count,
            overflow=acc_overflow,
        )

        info = MoveInfo(
            accepted=n_accepted > 0,
            log_likelihood=-new_state.energy,
            n_evaluations=n_atoms,
        )

        return new_state, info

    return step


def build_swap_kernel(backend: Any):
    """Build a single-atom species swap kernel.

    Selects two atoms of different species and swaps their types.

    Args:
        backend: EnergyBackend instance.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        key_a, key_b = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        idx_a = jax.random.randint(key_a, (), 0, n_atoms)
        idx_b = jax.random.randint(key_b, (), 0, n_atoms)

        type_a = state.types[idx_a]
        type_b = state.types[idx_b]
        new_types = state.types.at[idx_a].set(type_b).at[idx_b].set(type_a)

        result = backend(
            state.positions,
            new_types,
            state.cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )

        different_species = type_a != type_b
        accepted = (result.energy < likelihood_constraint) & different_species

        new_state = state.set(
            types=jnp.where(accepted, new_types, state.types),
            energy=jnp.where(accepted, result.energy, state.energy),
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
