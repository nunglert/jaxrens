"""Alchemical move kernels for nested sampling.

- build_morph_kernel: gradually morph an atom's type between species
  (interpolate between species for semi-grand-canonical sampling)
- build_shift_kernel: random translation of all atoms (rigid shift)

Single-walker functions, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.state.mc_state import MCState
from jaxrens.types import Params


# ---------------------------------------------------------------------------
# AtomMorphKernel: change one atom's species
# ---------------------------------------------------------------------------


def build_morph_kernel(
    energy_fn: Any,
    params: Params,
    n_species: int,
):
    """Build an atom morph kernel.

    Selects a random atom, changes its type to a random different species,
    and accepts if the new energy is below the NS constraint. This is
    the discrete version appropriate for integer-typed species.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.
        n_species: Total number of distinct species.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(
        rng_key: jax.Array,
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
        key_atom, key_species = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        atom_idx = jax.random.randint(key_atom, (), 0, n_atoms)

        # Pick a new species different from current
        current_type = state.types[atom_idx]
        # Sample from [0, n_species-1) and shift past current
        candidate = jax.random.randint(key_species, (), 0, n_species - 1)
        new_type = candidate + (candidate >= current_type).astype(jnp.int32)

        new_types = state.types.at[atom_idx].set(new_type)
        new_energy = energy_fn(params, state.positions, new_types, box=state.box)

        accepted = new_energy < likelihood_constraint

        new_state = state.set(
            types=jnp.where(accepted, new_types, state.types),
            energy=jnp.where(accepted, new_energy, state.energy),
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
        )

        return new_state, info

    return step


# ---------------------------------------------------------------------------
# RandomShiftKernel: rigid translation of all atoms
# ---------------------------------------------------------------------------


def build_shift_kernel(
    energy_fn: Any,
    params: Params,
):
    """Build a random shift kernel.

    Applies a uniform random translation to all atoms simultaneously.
    Useful for systems with PBC where the center-of-mass position matters
    (e.g., slab geometries, interface systems).

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(
        rng_key: jax.Array,
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
        # Random 3D translation, same for all atoms
        shift = state.step_size * jax.random.normal(rng_key, (3,))
        new_positions = state.positions + shift[None, :]

        new_energy = energy_fn(params, new_positions, state.types, box=state.box)
        accepted = new_energy < likelihood_constraint

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
        )

        return new_state, info

    return step
