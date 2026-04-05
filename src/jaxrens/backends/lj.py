"""Lennard-Jones pair potential backend.

E = sum_{i<j} 4 * epsilon * [(sigma/r_ij)^12 - (sigma/r_ij)^6]

Supports non-periodic and periodic (minimum image convention) systems.
"""

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.types import Box, Params, Positions, Types


def create_lj(
    epsilon: float = 1.0,
    sigma: float = 1.0,
    cutoff: float | None = None,
) -> tuple:
    """Create a Lennard-Jones pair potential.

    Args:
        epsilon: Energy scale.
        sigma: Length scale.
        cutoff: Cutoff distance. None = no cutoff.

    Returns:
        (energy_fn, params) tuple.
    """
    params = {
        "epsilon": jnp.array(epsilon),
        "sigma": jnp.array(sigma),
        "cutoff": jnp.array(cutoff) if cutoff is not None else None,
    }

    def energy_fn(
        params: Params,
        positions: Positions,
        types: Types,
        box: Box | None = None,
        **unused_kwargs: Any,
    ) -> jnp.ndarray:
        eps = params["epsilon"]
        sig = params["sigma"]
        cut = params["cutoff"]
        n_atoms = positions.shape[0]

        # All-pairs distance computation
        # dr_ij = positions[i] - positions[j] for all i < j
        dr = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)

        # Minimum image convention for periodic systems
        if box is not None:
            # Assumes orthorhombic box for simplicity
            box_diag = jnp.diag(box)
            dr = dr - jnp.round(dr / box_diag[None, None, :]) * box_diag[None, None, :]

        r2 = jnp.sum(dr**2, axis=-1)  # (N, N)

        # Mask out self-interactions and double counting
        mask = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)

        # Avoid division by zero on diagonal
        r2_safe = jnp.where(mask, r2, jnp.ones_like(r2))

        sig_r2 = sig**2 / r2_safe
        sig_r6 = sig_r2**3
        sig_r12 = sig_r6**2

        pair_energy = 4.0 * eps * (sig_r12 - sig_r6)

        # Apply cutoff if specified
        if cut is not None:
            cutoff_mask = r2_safe < cut**2
            pair_energy = jnp.where(cutoff_mask, pair_energy, 0.0)

        # Sum over upper triangle only
        return jnp.sum(jnp.where(mask, pair_energy, 0.0))

    return energy_fn, params
