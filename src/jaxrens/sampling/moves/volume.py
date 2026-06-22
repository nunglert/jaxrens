"""Volume move: isotropic volume change with accept/reject.

Proposes an isotropic scaling of the simulation cell and atom positions.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo
from jaxrens.utils.cell import check_cell_shape, get_volume


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
        cell_valid = check_cell_shape(
            new_cell, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        # Volume prior acceptance
        p_accept = jnp.where(
            flat_v_prior, 1.0, jnp.minimum(1.0, vol_ratio**n_atoms)
        )

        # Accept/reject — order matters for reject_reason attribution
        energy_ok = new_energy < likelihood_constraint
        prior_ok = jax.random.uniform(k2) < p_accept
        accepted = energy_ok & cell_valid & prior_ok

        # Reject priority: energy > cell > prior (so energy reason is reported
        # when multiple reasons apply — usually the most actionable signal)
        reject_reason = jnp.where(
            accepted,
            jnp.int32(0),
            jnp.where(
                ~energy_ok,
                jnp.int32(1),
                jnp.where(~cell_valid, jnp.int32(2), jnp.int32(3)),
            ),
        )

        # Gate the bucket-sizing signals on ``cell_valid``: hard cell-shape
        # rejections (max/min volume per atom, min aspect ratio) describe
        # configurations the chain will *never* live at.  Letting their
        # overflow / max_neighbor_count leak into state would force the
        # outer loop to escalate the neighbor bucket permanently to support
        # proposals that get rejected on the spot — pure waste, and
        # min-volume violations can blow the neighbor count up by ~10×.
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
