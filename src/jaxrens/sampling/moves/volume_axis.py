"""Volume-axis move: volume change by lengthening one lattice vector.

Same proposal and acceptance as the isotropic ``volume`` move, but the new
volume is reached by scaling, along the normal ``n`` of the plane spanned by
the *other two* lattice vectors, the component of the chosen vector normal to
that plane.  The other two vectors and the in-plane part of the chosen one
are left exactly as they are, and fractional coordinates are held fixed.

For a layered cell with the stacking vector as ``axis`` this changes the
interlayer spacing ``V / |a x b|`` without straining or sliding the layers,
which the isotropic ``volume`` and the axis-pair ``stretch`` cannot do (they
deform the stiff plane too, so their adapted step size is capped by it).

Single-walker function, designed for shard_map(vmap(vmap(...))) wrapping.
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
    axis: int = 2,
):
    """Build a volume-axis move kernel.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.
        flat_v_prior: If True, use flat volume prior (p_accept=1).
        axis: Index (0, 1, 2) of the lattice vector whose normal component
            is scaled; the other two span the reference plane.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)

    Note:
        ``step_size`` has the units of the ``volume`` move (volume per atom).
        The plane normal ``n`` depends only on the two vectors this move
        leaves unchanged, so it is the same for the forward and the reverse
        proposal.  The scaling ``T = I + (r - 1) n n^T`` has
        ``det T = r = V'/V``, the proposal is symmetric in V and fractional
        coordinates are preserved, so the acceptance is the ``volume`` one:
        ``min(1, (V'/V)**n_atoms)``.  This assumes the cell-shape prior is
        flat in the log of the axis lengths, as implied by ``stretch``
        (symmetric log steps, no prior factor).
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    other_1, other_2 = [i for i in range(3) if i != axis]

    cell_geometry = build_cell_geometry(
        n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
    )

    def step(rng_key, state, likelihood_constraint):
        k1, k2 = jax.random.split(rng_key)

        old_V = get_volume(state.cell)

        dV = state.step_size * n_atoms * jax.random.normal(k1)
        new_V = jnp.abs(old_V + dV)
        vol_ratio = new_V / old_V

        # Unit normal of the plane of the other two vectors (rows of ``cell``).
        normal = jnp.cross(state.cell[other_1], state.cell[other_2])
        normal = normal / jnp.linalg.norm(normal)

        # x' = x @ T with T = I + (r - 1) n n^T, det T = r.  The chosen
        # vector gets its normal component scaled; building the cell row
        # directly keeps the other two rows bit-identical.
        transform = jnp.eye(3) + (vol_ratio - 1.0) * jnp.outer(normal, normal)
        new_cell = state.cell.at[axis].add(
            (vol_ratio - 1.0) * (state.cell[axis] @ normal) * normal
        )
        # HIGHEST precision: TF32 (10-bit mantissa on GPU) corrupts positions@T
        # by ~3.7e-3 even for identity T, spiking energies at dense packing.
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

        p_accept = jnp.where(
            flat_v_prior, 1.0, jnp.minimum(1.0, vol_ratio**n_atoms)
        )

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
