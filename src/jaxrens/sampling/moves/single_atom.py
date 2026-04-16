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
from jaxrens.state.mc_state import MCState
from jaxrens.types import Params


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
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
        key_atom, key_disp = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        atom_idx = jax.random.randint(key_atom, (), 0, n_atoms)

        # Displace selected atom
        displacement = state.step_size * jax.random.normal(key_disp, (3,))
        new_positions = state.positions.at[atom_idx].add(displacement)

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
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
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

        new_state = state.set(
            positions=final_positions,
            energy=final_energy,
        )

        info = MoveInfo(
            accepted=n_accepted > 0,  # at least one atom accepted
            log_likelihood=-new_state.energy,
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
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
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
