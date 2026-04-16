"""Stretch move: volume-preserving anisotropic cell deformation.

Proposes a stretch along two random axes that preserves the cell volume.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.state.mc_state import MCState
from jaxrens.types import Params
from jaxrens.utils.cell import check_cell_shape


def build_kernel(
    energy_fn: Any,
    params: Params,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
):
    """Build a stretch move kernel.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    # Pre-define the 3 axis pairs: (0,1), (0,2), (1,2)
    axis_pairs = jnp.array([[0, 1], [0, 2], [1, 2]])

    def step(
        rng_key: jax.Array,
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
        k1, k2 = jax.random.split(rng_key)

        # Pick random axis pair
        pair_idx = jax.random.randint(k1, (), 0, 3)
        axes = axis_pairs[pair_idx]
        i, j = axes[0], axes[1]

        # Random stretch magnitude
        rv = state.step_size * jax.random.normal(k2)

        # Build volume-preserving transform
        # transform[i,i] = exp(rv), transform[j,j] = exp(-rv), rest = 1
        diag = jnp.ones(3)
        diag = diag.at[i].set(jnp.exp(rv))
        diag = diag.at[j].set(jnp.exp(-rv))
        transform = jnp.diag(diag)

        new_box = state.box @ transform
        new_positions = state.positions @ transform

        # Evaluate energy
        new_energy = energy_fn(params, new_positions, state.types, box=new_box)

        # Check cell shape validity
        cell_valid = check_cell_shape(
            new_box, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        # Accept/reject (no volume prior since volume is preserved)
        accepted = (new_energy < likelihood_constraint) & cell_valid

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            box=jnp.where(accepted, new_box, state.box),
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
        )

        return new_state, info

    return step
