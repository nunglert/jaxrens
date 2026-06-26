"""Volume move: isotropic volume change with accept/reject.

Proposes an isotropic scaling of the simulation cell and atom positions.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.constraints.cell_geometry import build_cell_geometry
from jaxrens.sampling.moves._common import finalize_cell_move
from jaxrens.utils.cell import get_volume


def build_kernel(
    backend: Any,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
    flat_v_prior: bool = False,
):
    """Build a volume move kernel.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.
        flat_v_prior: If True, use flat volume prior (p_accept=1).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    # Cell-geometry guard, defined once in the constraints framework. The
    # kernel claims this constraint (rather than the central MWG gate) because
    # its result also gates the neighbor-bucket bookkeeping below and the
    # reject-reason ordering; see jaxrens.constraints.cell_geometry.
    cell_geometry = build_cell_geometry(
        n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
    )

    def step(rng_key, state, likelihood_constraint):
        k1, k2 = jax.random.split(rng_key)

        old_V = get_volume(state.cell)

        # Propose volume change
        dV = state.step_size * n_atoms * jax.random.normal(k1)
        new_V = jnp.abs(old_V + dV)
        vol_ratio = new_V / old_V

        # Isotropic scaling
        scale = vol_ratio ** (1.0 / 3.0)
        transform = jnp.eye(3) * scale

        new_cell = state.cell @ transform
        # HIGHEST precision: TF32 (10-bit mantissa on GPU) corrupts positions@T
        # by ~3.7e-3 even for identity T, spiking LJ energy at dense packing.
        new_positions = jnp.einsum(
            "ij,jk->ik",
            state.positions,
            transform,
            precision=jax.lax.Precision.HIGHEST,
        )

        # Evaluate energy
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

        # Check cell shape validity
        cell_valid = cell_geometry(new_positions, state.types, new_cell)

        # Volume prior acceptance
        p_accept = jnp.where(
            flat_v_prior, 1.0, jnp.minimum(1.0, vol_ratio**n_atoms)
        )

        # Accept/reject — order matters for reject_reason attribution
        energy_ok = new_energy < likelihood_constraint
        prior_ok = jax.random.uniform(k2) < p_accept

        return finalize_cell_move(
            state,
            new_positions,
            new_cell,
            new_energy,
            count,
            overflow,
            cell_valid,
            energy_ok,
            prior_ok=prior_ok,
        )

    return step
