"""Shear move: volume-preserving shear deformation of a cell vector.

Proposes a shear of one cell vector within the plane spanned by the other two.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.constraints.cell_geometry import build_cell_geometry
from jaxrens.sampling.moves._common import finalize_cell_move
from jaxrens.utils.cell import transform_positions


def _build_shear_cell(cell, shear_idx, rv1, rv2):
    """Build a sheared cell by displacing one cell vector."""

    def shear_0(args):
        cell, rv1, rv2 = args
        v1 = cell[1]
        v2 = cell[2]
        v1_norm = v1 / jnp.linalg.norm(v1)
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return cell.at[0].set(cell[0] + rv1 * v1_norm + rv2 * v2_norm)

    def shear_1(args):
        cell, rv1, rv2 = args
        v1 = cell[0]
        v2 = cell[2]
        v1_norm = v1 / jnp.linalg.norm(v1)
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return cell.at[1].set(cell[1] + rv1 * v1_norm + rv2 * v2_norm)

    def shear_2(args):
        cell, rv1, rv2 = args
        v1 = cell[0]
        v2 = cell[1]
        v1_norm = v1 / jnp.linalg.norm(v1)
        v2_perp = v2 - jnp.dot(v2, v1_norm) * v1_norm
        v2_norm = v2_perp / jnp.linalg.norm(v2_perp)
        return cell.at[2].set(cell[2] + rv1 * v1_norm + rv2 * v2_norm)

    return jax.lax.switch(
        shear_idx, [shear_0, shear_1, shear_2], (cell, rv1, rv2)
    )


def build_kernel(
    backend: Any,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
):
    """Build a shear move kernel.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    # Cell-geometry guard (kernel-claimed); see
    # jaxrens.constraints.cell_geometry and the volume kernel for rationale.
    cell_geometry = build_cell_geometry(
        n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
    )

    def step(rng_key, state, likelihood_constraint):
        k1, k2 = jax.random.split(rng_key)

        shear_idx = jax.random.randint(k1, (), 0, 3)
        rvs = state.step_size * jax.random.normal(k2, (2,))
        rv1, rv2 = rvs[0], rvs[1]

        new_cell = _build_shear_cell(state.cell, shear_idx, rv1, rv2)
        new_positions = transform_positions(
            state.positions, state.cell, new_cell
        )

        result = backend(
            new_positions,
            state.types,
            new_cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )
        new_energy, count, overflow = (
            result.energy,
            result.max_neighbor_count,
            result.overflow,
        )

        cell_valid = cell_geometry(new_positions, state.types, new_cell)

        energy_ok = new_energy < likelihood_constraint

        return finalize_cell_move(
            state,
            new_positions,
            new_cell,
            new_energy,
            count,
            overflow,
            cell_valid,
            energy_ok,
        )

    return step
