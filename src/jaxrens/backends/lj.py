"""Lennard-Jones pair potential backend.

E = sum_{i<j} 4 * eps_ij * [(sig_ij/r_ij)^12 - (sig_ij/r_ij)^6]

Single-species mode (default):
    eps_ij = epsilon, sig_ij = sigma  (scalars, ignore species)

Multi-species mode (``epsilon``/``sigma`` given as 1-D sequences):
    eps_ij = sqrt(eps[s_i] * eps[s_j])     (geometric Lorentz-Berthelot)
    sig_ij = (sig[s_i] + sig[s_j]) / 2     (arithmetic Lorentz-Berthelot)

Supports non-periodic and periodic (minimum image convention) systems.
All-pairs computation — no neighbor list needed.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp


_ScalarOrSeq = float | Sequence[float] | jnp.ndarray


class LJBackend:
    """Lennard-Jones pair potential.

    All-pairs computation with minimum image convention.
    Ignores ``max_neighbors`` (no neighbor list needed).

    Args:
        epsilon: Energy well depth.  Scalar (single-species) or 1-D
            sequence of length ``n_species`` (per-species table; pair
            interactions use geometric Lorentz-Berthelot mixing).
        sigma: Length scale.  Same shape rules as ``epsilon``;
            arithmetic Lorentz-Berthelot mixing for cross-species pairs.
        cutoff: Distance cutoff (scalar; applies uniformly across species).
    """

    def __init__(
        self,
        epsilon: _ScalarOrSeq = 1.0,
        sigma: _ScalarOrSeq = 1.0,
        cutoff: float | None = None,
    ):
        eps_arr = jnp.asarray(epsilon, dtype=jnp.float32)
        sig_arr = jnp.asarray(sigma, dtype=jnp.float32)
        if eps_arr.ndim != sig_arr.ndim:
            raise ValueError(
                f"epsilon and sigma must have the same rank, got "
                f"epsilon.ndim={eps_arr.ndim}, sigma.ndim={sig_arr.ndim}"
            )
        if eps_arr.ndim > 1:
            raise ValueError(
                f"epsilon/sigma must be scalar or 1-D (per-species), got "
                f"rank {eps_arr.ndim}"
            )
        if eps_arr.ndim == 1:
            if eps_arr.shape != sig_arr.shape:
                raise ValueError(
                    f"per-species epsilon and sigma must have matching length, "
                    f"got {eps_arr.shape} vs {sig_arr.shape}"
                )
            if eps_arr.shape[0] == 0:
                raise ValueError("per-species epsilon/sigma must be non-empty")

        self._per_species = bool(eps_arr.ndim == 1)
        if self._per_species:
            self.epsilon = None
            self.sigma = None
            self._eps_table = eps_arr
            self._sig_table = sig_arr
            self.n_species = int(eps_arr.shape[0])
        else:
            self.epsilon = float(eps_arr)
            self.sigma = float(sig_arr)
            self._eps_table = None
            self._sig_table = None
            self.n_species = 1

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

        if self._per_species:
            eps_per_atom = self._eps_table[species]                          # (N,)
            sig_per_atom = self._sig_table[species]                          # (N,)
            eps_ij = jnp.sqrt(eps_per_atom[:, None] * eps_per_atom[None, :]) # (N, N)
            sig_ij = 0.5 * (sig_per_atom[:, None] + sig_per_atom[None, :])   # (N, N)
            sig_r2 = sig_ij**2 / r2_safe
        else:
            eps_ij = self.epsilon
            sig_r2 = self.sigma**2 / r2_safe

        sig_r6 = sig_r2**3
        sig_r12 = sig_r6**2

        pair_energy = 4.0 * eps_ij * (sig_r12 - sig_r6)

        if self.cutoff is not None:
            cutoff_mask = r2_safe < self.cutoff**2
            pair_energy = jnp.where(cutoff_mask, pair_energy, 0.0)

        energy = jnp.sum(jnp.where(mask, pair_energy, 0.0))
        return energy, 0, False


def create_lj(
    epsilon: _ScalarOrSeq = 1.0,
    sigma: _ScalarOrSeq = 1.0,
    cutoff: float | None = None,
) -> LJBackend:
    """Create a Lennard-Jones backend.

    Pass scalar ``epsilon``/``sigma`` for the classic single-species
    potential; pass 1-D sequences (length ``n_species``) for a per-species
    table with Lorentz-Berthelot mixing on cross-species pairs.
    """
    return LJBackend(epsilon=epsilon, sigma=sigma, cutoff=cutoff)
