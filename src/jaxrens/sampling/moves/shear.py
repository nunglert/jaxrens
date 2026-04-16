"""Shear move: volume-preserving shear deformation of a cell vector.

Proposes a shear of one cell vector within the plane spanned by the other two.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.state.mc_state import MCState
from jaxrens.types import Params
from jaxrens.utils.cell import (
    check_cell_shape,
    get_cell_transformation,
    transform_positions,
)


def _build_shear_box(box, shear_idx, rv1, rv2):
    """Build a sheared box by displacing one cell vector.

    Uses lax.switch to handle the 3 possible shear indices in a
    jit-compatible way.
    """

    def shear_0(args):
        box, rv1, rv2 = args
        v1 = box[1]
        v2 = box[2]
        v1_norm = v1 / jnp.linalg.norm(v1)
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return box.at[0].set(box[0] + rv1 * v1_norm + rv2 * v2_norm)

    def shear_1(args):
        box, rv1, rv2 = args
        v1 = box[0]
        v2 = box[2]
        v1_norm = v1 / jnp.linalg.norm(v1)
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return box.at[1].set(box[1] + rv1 * v1_norm + rv2 * v2_norm)

    def shear_2(args):
        box, rv1, rv2 = args
        v1 = box[0]
        v2 = box[1]
        v1_norm = v1 / jnp.linalg.norm(v1)
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return box.at[2].set(box[2] + rv1 * v1_norm + rv2 * v2_norm)

    return jax.lax.switch(shear_idx, [shear_0, shear_1, shear_2], (box, rv1, rv2))


def build_kernel(
    energy_fn: Any,
    params: Params,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
):
    """Build a shear move kernel.

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

    def step(
        rng_key: jax.Array,
        state: MCState,
        likelihood_constraint: float,
    ) -> tuple[MCState, MoveInfo]:
        k1, k2 = jax.random.split(rng_key)

        # Pick random cell vector to shear
        shear_idx = jax.random.randint(k1, (), 0, 3)

        # Random displacements
        rvs = state.step_size * jax.random.normal(k2, (2,))
        rv1, rv2 = rvs[0], rvs[1]

        # Build new box with shear
        new_box = _build_shear_box(state.box, shear_idx, rv1, rv2)

        # Transform positions
        new_positions = transform_positions(state.positions, state.box, new_box)

        # Evaluate energy
        new_energy = energy_fn(params, new_positions, state.types, box=new_box)

        # Check cell shape validity
        cell_valid = check_cell_shape(
            new_box, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        # Accept/reject (no volume prior since volume is preserved by shear)
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
