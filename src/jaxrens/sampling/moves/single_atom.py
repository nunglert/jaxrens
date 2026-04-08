"""Single-atom move kernels for nested sampling.

Moves that perturb one atom at a time, efficient for large systems where
displacing all atoms simultaneously has low acceptance.

- SingleAtomMoveKernel: move one random atom per step
- SingleAtomSweepKernel: sweep through all atoms sequentially via lax.scan
- SingleAtomSwapKernel: swap species of two random atoms (multi-component)

Single-walker functions, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types


class SingleAtomState(NamedTuple):
    """State for single-atom moves."""

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
) -> SingleAtomState:
    """Create initial single-atom move state."""
    return SingleAtomState(
        positions=positions,
        types=types,
        energy=jnp.asarray(energy),
        box=box,
        step_size=jnp.asarray(step_size),
    )


# ---------------------------------------------------------------------------
# SingleAtomMoveKernel: perturb one random atom
# ---------------------------------------------------------------------------


def build_kernel(
    energy_fn: Any,
    params: Params,
):
    """Build a single-atom random displacement kernel.

    Each step selects one atom at random and applies a Gaussian
    displacement. Accepts if the new energy is below the NS constraint.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(
        rng_key: jax.Array,
        state: SingleAtomState,
        likelihood_constraint: float,
    ) -> tuple[SingleAtomState, MoveInfo]:
        key_atom, key_disp = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        atom_idx = jax.random.randint(key_atom, (), 0, n_atoms)

        # Displace selected atom
        displacement = state.step_size * jax.random.normal(key_disp, (3,))
        new_positions = state.positions.at[atom_idx].add(displacement)

        new_energy = energy_fn(params, new_positions, state.types, box=state.box)
        accepted = new_energy < likelihood_constraint

        out_positions = jnp.where(accepted, new_positions, state.positions)
        out_energy = jnp.where(accepted, new_energy, state.energy)

        new_state = SingleAtomState(
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
# SingleAtomSweepKernel: sweep through all atoms sequentially
# ---------------------------------------------------------------------------


def build_sweep_kernel(
    energy_fn: Any,
    params: Params,
    n_atoms: int,
):
    """Build a single-atom sweep kernel.

    Sweeps through all atoms sequentially (via lax.scan), displacing each.
    One full sweep = one "step" for adaptation purposes.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.
        n_atoms: Number of atoms (needed for scan length).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(
        rng_key: jax.Array,
        state: SingleAtomState,
        likelihood_constraint: float,
    ) -> tuple[SingleAtomState, MoveInfo]:
        keys = jax.random.split(rng_key, n_atoms)

        def sweep_one(carry, key_and_idx):
            positions, energy, n_accepted = carry
            key, idx = key_and_idx

            displacement = state.step_size * jax.random.normal(key, (3,))
            new_positions = positions.at[idx].add(displacement)

            new_energy = energy_fn(params, new_positions, state.types, box=state.box)
            accepted = new_energy < likelihood_constraint

            out_positions = jnp.where(accepted, new_positions, positions)
            out_energy = jnp.where(accepted, new_energy, energy)

            return (out_positions, out_energy, n_accepted + accepted.astype(jnp.int32)), None

        atom_indices = jnp.arange(n_atoms)
        init_carry = (state.positions, state.energy, jnp.int32(0))
        (final_positions, final_energy, n_accepted), _ = jax.lax.scan(
            sweep_one, init_carry, (keys, atom_indices)
        )

        new_state = SingleAtomState(
            positions=final_positions,
            types=state.types,
            energy=final_energy,
            box=state.box,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=n_accepted > 0,  # at least one atom accepted
            log_likelihood=-final_energy,
            n_evaluations=n_atoms,
        )

        return new_state, info

    return step


# ---------------------------------------------------------------------------
# SingleAtomSwapKernel: swap species of two random atoms
# ---------------------------------------------------------------------------


def build_swap_kernel(
    energy_fn: Any,
    params: Params,
):
    """Build a single-atom species swap kernel.

    Selects two atoms of different species and swaps their types.
    Useful for multi-component systems (e.g., alloys, mixtures).

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(
        rng_key: jax.Array,
        state: SingleAtomState,
        likelihood_constraint: float,
    ) -> tuple[SingleAtomState, MoveInfo]:
        key_a, key_b = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        idx_a = jax.random.randint(key_a, (), 0, n_atoms)
        idx_b = jax.random.randint(key_b, (), 0, n_atoms)

        # Swap types of atoms a and b
        type_a = state.types[idx_a]
        type_b = state.types[idx_b]
        new_types = state.types.at[idx_a].set(type_b).at[idx_b].set(type_a)

        new_energy = energy_fn(params, state.positions, new_types, box=state.box)

        # Accept if different species and energy below constraint
        different_species = type_a != type_b
        accepted = (new_energy < likelihood_constraint) & different_species

        out_types = jnp.where(accepted, new_types, state.types)
        out_energy = jnp.where(accepted, new_energy, state.energy)

        new_state = SingleAtomState(
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
# Top-level APIs
# ---------------------------------------------------------------------------


def as_top_level_api(
    energy_fn: Any,
    params: Params,
    step_size: float = 0.1,
) -> MoveKernel:
    """Top-level API for single-atom random displacement."""
    kernel = build_kernel(energy_fn, params)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)


def as_sweep_api(
    energy_fn: Any,
    params: Params,
    n_atoms: int,
    step_size: float = 0.1,
) -> MoveKernel:
    """Top-level API for single-atom sweep."""
    kernel = build_sweep_kernel(energy_fn, params, n_atoms)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)


def as_swap_api(
    energy_fn: Any,
    params: Params,
    step_size: float = 0.1,
) -> MoveKernel:
    """Top-level API for single-atom species swap."""
    kernel = build_swap_kernel(energy_fn, params)
    init_fn = lambda pos, types, energy, box=None: init(
        pos, types, energy, box, step_size
    )
    return MoveKernel(init=init_fn, step=kernel)
