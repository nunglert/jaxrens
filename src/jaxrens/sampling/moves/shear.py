"""Shear move: volume-preserving shear deformation of a cell vector.

Proposes a shear of one cell vector within the plane spanned by the other two.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.utils.cell import (
    check_cell_shape,
    transform_positions,
)


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

    return jax.lax.switch(shear_idx, [shear_0, shear_1, shear_2], (cell, rv1, rv2))


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

    def step(rng_key, state, likelihood_constraint):
        k1, k2 = jax.random.split(rng_key)

        shear_idx = jax.random.randint(k1, (), 0, 3)
        rvs = state.step_size * jax.random.normal(k2, (2,))
        rv1, rv2 = rvs[0], rvs[1]

        new_cell = _build_shear_cell(state.cell, shear_idx, rv1, rv2)
        new_positions = transform_positions(state.positions, state.cell, new_cell)

        new_energy, count, overflow = backend(
            new_positions, state.types, new_cell, state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )

        cell_valid = check_cell_shape(
            new_cell, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        energy_ok = new_energy < likelihood_constraint
        accepted = energy_ok & cell_valid

        reject_reason = jnp.where(
            accepted, jnp.int32(0),
            jnp.where(~energy_ok, jnp.int32(1), jnp.int32(2)),
        )

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            cell=jnp.where(accepted, new_cell, state.cell),
            max_neighbor_count=jnp.maximum(state.max_neighbor_count, count),
            overflow=state.overflow | overflow,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
            reject_reason=reject_reason,
        )

        return new_state, info

    return step
