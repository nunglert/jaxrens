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

        cell_valid = check_cell_shape(
            new_cell, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        energy_ok = new_energy < likelihood_constraint
        accepted = energy_ok & cell_valid

        reject_reason = jnp.where(
            accepted,
            jnp.int32(0),
            jnp.where(~energy_ok, jnp.int32(1), jnp.int32(2)),
        )

        # See ``volume.py`` for the rationale: bucket-sizing signals are
        # gated on ``cell_valid`` so hard cell-shape rejections (which the
        # chain will never live at) don't permanently inflate the neighbor
        # bucket.
        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            cell=jnp.where(accepted, new_cell, state.cell),
            max_neighbor_count=jnp.maximum(
                state.max_neighbor_count,
                jnp.where(cell_valid, count, 0),
            ),
            overflow=state.overflow | (overflow & cell_valid),
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
            reject_reason=reject_reason,
        )

        return new_state, info

    return step
