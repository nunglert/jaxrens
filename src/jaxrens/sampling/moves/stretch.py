"""Stretch move: volume-preserving anisotropic cell deformation.

Proposes a stretch along two random axes that preserves the cell volume.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.utils.cell import check_cell_shape


def build_kernel(
    backend: Any,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
):
    """Build a stretch move kernel.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    axis_pairs = jnp.array([[0, 1], [0, 2], [1, 2]])

    def step(rng_key, state, likelihood_constraint):
        k1, k2 = jax.random.split(rng_key)

        pair_idx = jax.random.randint(k1, (), 0, 3)
        axes = axis_pairs[pair_idx]
        i, j = axes[0], axes[1]

        rv = state.step_size * jax.random.normal(k2)

        diag = jnp.ones(3)
        diag = diag.at[i].set(jnp.exp(rv))
        diag = diag.at[j].set(jnp.exp(-rv))
        transform = jnp.diag(diag)

        new_cell = state.cell @ transform
        new_positions = state.positions @ transform

        new_energy, count, overflow = backend(
            new_positions, state.types, new_cell, state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )

        cell_valid = check_cell_shape(
            new_cell, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        accepted = (new_energy < likelihood_constraint) & cell_valid

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
        )

        return new_state, info

    return step
