"""Stretch move: volume-preserving anisotropic cell deformation.

Proposes a stretch along two random axes that preserves the cell volume.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.constraints.cell_geometry import build_cell_geometry
from jaxrens.sampling.moves._common import finalize_cell_move


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

    # Cell-geometry guard (kernel-claimed); see
    # jaxrens.constraints.cell_geometry and the volume kernel for rationale.
    cell_geometry = build_cell_geometry(
        n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
    )

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
        # HIGHEST precision: TF32 (10-bit mantissa on GPU) corrupts positions@T
        # by ~3.7e-3 even for identity T, spiking LJ energy at dense packing.
        new_positions = jnp.einsum(
            "ij,jk->ik",
            state.positions,
            transform,
            precision=jax.lax.Precision.HIGHEST,
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
