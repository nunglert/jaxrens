"""Alchemical move kernels for nested sampling.

- AtomMorphKernel: gradually morph an atom's type between species
  (interpolate between species for semi-grand-canonical sampling)
- RandomShiftKernel: random translation of all atoms (rigid shift)

Single-walker functions, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types


class AlchemicalState(NamedTuple):
    """State for alchemical moves."""

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
    step_size: float = 0.1,
) -> AlchemicalState:
    """Create initial alchemical state."""
    return AlchemicalState(
        positions=positions,
        types=types,
        energy=jnp.asarray(energy),
        box=box,
        step_size=jnp.asarray(step_size),
    )


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
        state: AlchemicalState,
        likelihood_constraint: float,
    ) -> tuple[AlchemicalState, MoveInfo]:
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

        out_types = jnp.where(accepted, new_types, state.types)
        out_energy = jnp.where(accepted, new_energy, state.energy)

        new_state = AlchemicalState(
            positions=state.positions,
            types=out_types,
            energy=out_energy,
            box=state.box,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-out_energy,
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
        state: AlchemicalState,
        likelihood_constraint: float,
    ) -> tuple[AlchemicalState, MoveInfo]:
        # Random 3D translation, same for all atoms
        shift = state.step_size * jax.random.normal(rng_key, (3,))
        new_positions = state.positions + shift[None, :]

        new_energy = energy_fn(params, new_positions, state.types, box=state.box)
        accepted = new_energy < likelihood_constraint

        out_positions = jnp.where(accepted, new_positions, state.positions)
        out_energy = jnp.where(accepted, new_energy, state.energy)

        new_state = AlchemicalState(
            positions=out_positions,
            types=state.types,
            energy=out_energy,
            box=state.box,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-out_energy,
            n_evaluations=1,
        )

        return new_state, info

    return step


# ---------------------------------------------------------------------------
# Top-level APIs
# ---------------------------------------------------------------------------


def as_morph_api(
    energy_fn: Any,
    params: Params,
    n_species: int,
    step_size: float = 0.1,
) -> MoveKernel:
    """Top-level API for atom morph move."""
    kernel = build_morph_kernel(energy_fn, params, n_species)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)


def as_shift_api(
    energy_fn: Any,
    params: Params,
    step_size: float = 0.1,
) -> MoveKernel:
    """Top-level API for random shift move."""
    kernel = build_shift_kernel(energy_fn, params)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)
