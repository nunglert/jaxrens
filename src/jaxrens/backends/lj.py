"""Lennard-Jones pair potential backend.

E = sum_{i<j, k} 4 * eps_ij * [(sig_ij/r_ijk)^12 - (sig_ij/r_ijk)^6]

where ``k`` ranges over the periodic-image offsets enumerated by
``supercell_trafo``. The summation uses **triclinic minimum-image** for
the central image (correct for sheared cells from shear/stretch moves),
then explicitly tiles ``supercell_trafo`` additional images on top of
that so cells smaller than ``2 · r_cut`` can be handled when
``supercell_trafo`` is bumped above ``(1, 1, 1)``.

Single-species mode (default):
    eps_ij = epsilon, sig_ij = sigma  (scalars, ignore species)

Multi-species mode (``epsilon``/``sigma`` given as 1-D sequences):
    eps_ij = sqrt(eps[s_i] * eps[s_j])     (geometric Lorentz-Berthelot)
    sig_ij = (sig[s_i] + sig[s_j]) / 2     (arithmetic Lorentz-Berthelot)

The ``supercell_trafo`` convention matches MACE / Nequix: it must satisfy
``min(perp_distance · sc) >= 2 · r_cut`` for the energy to capture every
true neighbor. The resolver emits a startup warning when the cell prior
permits cells that violate this bound.

All-pairs computation — no neighbor list needed.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp

from jaxrens.backends._graph_neighbors import _make_image_offsets
from jaxrens.backends.base import BackendResult


_ScalarOrSeq = float | Sequence[float] | jnp.ndarray


class LJBackend:
    """Lennard-Jones pair potential.

    All-pairs computation with triclinic minimum-image convention and
    explicit supercell-image enumeration. Ignores ``max_neighbors`` (no
    neighbor list needed).

    Args:
        epsilon: Energy well depth.  Scalar (single-species) or 1-D
            sequence of length ``n_species`` (per-species table; pair
            interactions use geometric Lorentz-Berthelot mixing).
        sigma: Length scale.  Same shape rules as ``epsilon``;
            arithmetic Lorentz-Berthelot mixing for cross-species pairs.
        cutoff: Distance cutoff (scalar; applies uniformly across species).
        supercell_trafo: ``(sc_a, sc_b, sc_c)`` integer expansion for
            periodic-image enumeration. Must satisfy
            ``min(perp_distance · sc) >= 2 · r_cut`` to capture every
            true neighbor. Default ``(1, 1, 1)`` is MIC-only (correct
            iff the actual cell already has min perpendicular distance
            ≥ ``2 · r_cut``); bump to ``(2, 2, 2)`` (27 images) for
            cells that may shrink below that threshold.
    """

    def __init__(
        self,
        epsilon: _ScalarOrSeq = 1.0,
        sigma: _ScalarOrSeq = 1.0,
        cutoff: float | None = None,
        supercell_trafo: tuple[int, int, int] = (1, 1, 1),
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

        sc_a, sc_b, sc_c = (int(x) for x in supercell_trafo)
        if min(sc_a, sc_b, sc_c) < 1:
            raise ValueError(
                f"supercell_trafo entries must be >= 1, got {supercell_trafo}"
            )
        self.supercell_trafo = (sc_a, sc_b, sc_c)
        # _make_image_offsets returns centered integer offsets in fractional
        # coordinates: e.g. (2,2,2) → [-1, 0, +1]^3 → 27 images.
        self._image_offsets = jnp.asarray(
            _make_image_offsets(sc_a, sc_b, sc_c), dtype=jnp.float32
        )

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        n_atoms = positions.shape[0]

        # Non-periodic systems: cell is all zeros. Substitute a huge cube so
        # (a) inv(cell) is well-defined, (b) MIC rounds to zero (no wrap),
        # (c) supercell image translations land far outside any cutoff.
        det = jnp.linalg.det(cell)
        safe_cell = jnp.where(det == 0.0, 1.0e10 * jnp.eye(3), cell)

        # (N, N, 3) pair displacements: r_ij = r_j - r_i.
        dr = positions[None, :, :] - positions[:, None, :]

        # Triclinic minimum-image: project to fractional coords, round, project
        # back. Replaces the previous diag-only MIC, which was wrong for any
        # sheared cell produced by shear / stretch moves.
        inv_cell = jnp.linalg.inv(safe_cell)
        dr_frac = dr @ inv_cell
        dr_mic = dr - jnp.round(dr_frac) @ safe_cell  # (N, N, 3)

        # Add explicit supercell image translations on top of the MIC pair.
        # image_translations: (K, 3), broadcast to (K, N, N, 3).
        image_translations = self._image_offsets @ safe_cell
        dr_kij = dr_mic[None, :, :, :] + image_translations[:, None, None, :]
        r2 = jnp.sum(dr_kij**2, axis=-1)  # (K, N, N)

        # Exclude only the central self-pair (i = j and k = (0,0,0)) — every
        # other (i, j, k) entry is a real interaction. For i ≠ j and any k,
        # the (j, i, -k) mirror entry duplicates it: hence the 0.5 prefactor
        # below counts each interaction exactly once.
        is_center = jnp.all(self._image_offsets == 0.0, axis=-1)  # (K,)
        self_pair = jnp.eye(n_atoms, dtype=bool)                  # (N, N)
        exclude = is_center[:, None, None] & self_pair[None, :, :]  # (K, N, N)

        # Avoid 0/0 at the excluded entries.
        r2_safe = jnp.where(exclude, 1.0, r2)

        if self._per_species:
            eps_per_atom = self._eps_table[species]                          # (N,)
            sig_per_atom = self._sig_table[species]                          # (N,)
            eps_ij = jnp.sqrt(eps_per_atom[:, None] * eps_per_atom[None, :]) # (N, N)
            sig_ij = 0.5 * (sig_per_atom[:, None] + sig_per_atom[None, :])   # (N, N)
            sig_r2 = sig_ij[None, :, :] ** 2 / r2_safe                       # (K, N, N)
            eps_ij_bcast = eps_ij[None, :, :]
        else:
            eps_ij_bcast = self.epsilon
            sig_r2 = self.sigma ** 2 / r2_safe

        sig_r6 = sig_r2 ** 3
        sig_r12 = sig_r6 ** 2

        pair_energy = 4.0 * eps_ij_bcast * (sig_r12 - sig_r6)

        if self.cutoff is not None:
            cutoff_mask = r2 < self.cutoff ** 2
            pair_energy = jnp.where(cutoff_mask, pair_energy, 0.0)

        # Zero out excluded (central self) entries before summing.
        pair_energy = jnp.where(exclude, 0.0, pair_energy)

        # Each (i, j, k) and its mirror (j, i, -k) both contribute — divide by 2.
        energy = 0.5 * jnp.sum(pair_energy)
        return BackendResult(energy=energy)


def create_lj(
    epsilon: _ScalarOrSeq = 1.0,
    sigma: _ScalarOrSeq = 1.0,
    cutoff: float | None = None,
    supercell_trafo: tuple[int, int, int] = (1, 1, 1),
) -> LJBackend:
    """Create a Lennard-Jones backend.

    Pass scalar ``epsilon``/``sigma`` for the classic single-species
    potential; pass 1-D sequences (length ``n_species``) for a per-species
    table with Lorentz-Berthelot mixing on cross-species pairs.

    ``supercell_trafo`` enumerates explicit periodic images on top of the
    triclinic MIC pair distance — bump above ``(1, 1, 1)`` when the cell
    prior permits cells smaller than ``2 · cutoff`` along any direction.
    """
    return LJBackend(
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        supercell_trafo=supercell_trafo,
    )
