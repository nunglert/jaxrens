"""Lennard-Jones pair potential backend.

E = sum_{i<j} 4 * epsilon * [(sigma/r_ij)^12 - (sigma/r_ij)^6]

Supports non-periodic and periodic (minimum image convention) systems.
All-pairs computation — no neighbor list needed.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


class LJBackend:
    """Lennard-Jones pair potential.

    All-pairs computation with minimum image convention.
    Ignores max_neighbors (no neighbor list needed).
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        cutoff: float | None = None,
    ):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.r_cutoff = cutoff if cutoff is not None else 0.0

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        eps = self.epsilon
        sig = self.sigma
        n_atoms = positions.shape[0]

        dr = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)

        # Minimum image convention for periodic systems
        # For non-periodic (cell=zeros), replace zeros with large value to avoid div-by-zero
        box_diag = jnp.diag(cell)
        safe_diag = jnp.where(box_diag == 0, 1e10, box_diag)
        dr = dr - jnp.round(dr / safe_diag[None, None, :]) * box_diag[None, None, :]

        r2 = jnp.sum(dr**2, axis=-1)  # (N, N)

        mask = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
        r2_safe = jnp.where(mask, r2, jnp.ones_like(r2))

        sig_r2 = sig**2 / r2_safe
        sig_r6 = sig_r2**3
        sig_r12 = sig_r6**2

        pair_energy = 4.0 * eps * (sig_r12 - sig_r6)

        if self.cutoff is not None:
            cutoff_mask = r2_safe < self.cutoff**2
            pair_energy = jnp.where(cutoff_mask, pair_energy, 0.0)

        energy = jnp.sum(jnp.where(mask, pair_energy, 0.0))
        return energy, 0, False


def create_lj(
    epsilon: float = 1.0,
    sigma: float = 1.0,
    cutoff: float | None = None,
) -> LJBackend:
    """Create a Lennard-Jones backend."""
    return LJBackend(epsilon=epsilon, sigma=sigma, cutoff=cutoff)
