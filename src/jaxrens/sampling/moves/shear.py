"""Shear move: volume-preserving shear deformation of a cell vector.

Proposes a shear of one cell vector within the plane spanned by the other two.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types
from jaxrens.utils.cell import (
    check_cell_shape,
    get_cell_transformation,
    transform_positions,
)


class ShearMoveState(NamedTuple):
    """State for the shear move."""

    positions: jnp.ndarray  # (n_atoms, 3)
    types: jnp.ndarray  # (n_atoms,)
    energy: jnp.ndarray  # scalar
    box: jnp.ndarray  # (3, 3)
    step_size: jnp.ndarray  # scalar


def init(
    positions: Positions,
    types: Types,
    energy: float,
    box: Box,
    step_size: float = 0.1,
) -> ShearMoveState:
    """Create initial shear move state."""
    return ShearMoveState(
        positions=positions,
        types=types,
        energy=jnp.asarray(energy),
        box=jnp.asarray(box),
        step_size=jnp.asarray(step_size),
    )


def _build_shear_box(box, shear_idx, rv1, rv2):
    """Build a sheared box by displacing one cell vector.

    Uses lax.switch to handle the 3 possible shear indices in a
    jit-compatible way.
    """

    def shear_0(args):
        box, rv1, rv2 = args
        # Shear vector 0, basis from vectors 1 and 2
        v1 = box[1]
        v2 = box[2]
        v1_norm = v1 / jnp.linalg.norm(v1)
        # Orthogonalize v2 against v1
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return box.at[0].set(box[0] + rv1 * v1_norm + rv2 * v2_norm)

    def shear_1(args):
        box, rv1, rv2 = args
        # Shear vector 1, basis from vectors 0 and 2
        v1 = box[0]
        v2 = box[2]
        v1_norm = v1 / jnp.linalg.norm(v1)
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return box.at[1].set(box[1] + rv1 * v1_norm + rv2 * v2_norm)

    def shear_2(args):
        box, rv1, rv2 = args
        # Shear vector 2, basis from vectors 0 and 1
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
        state: ShearMoveState,
        likelihood_constraint: float,
    ) -> tuple[ShearMoveState, MoveInfo]:
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

        out_positions = jnp.where(accepted, new_positions, state.positions)
        out_energy = jnp.where(accepted, new_energy, state.energy)
        out_box = jnp.where(accepted, new_box, state.box)

        new_state = ShearMoveState(
            positions=out_positions,
            types=state.types,
            energy=out_energy,
            box=out_box,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-out_energy,
            n_evaluations=1,
        )

        return new_state, info

    return step


def as_top_level_api(
    energy_fn: Any,
    params: Params,
    n_atoms: int,
    step_size: float = 0.1,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
) -> MoveKernel:
    """Convenient top-level API."""
    kernel = build_kernel(
        energy_fn, params, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect,
    )
    init_fn = lambda pos, types, energy, box: init(pos, types, energy, box, step_size)
    return MoveKernel(init=init_fn, step=kernel)
