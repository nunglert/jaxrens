"""Volume moves: isotropic and normal-strain volume changes.

- build_kernel: isotropic scaling of the cell and atom positions
- build_normal_strain_kernel: scaling along the normal to the (a, b) plane,
  with ``a`` and ``b`` fixed

Single-walker functions, designed for shard_map(vmap(vmap(...))) wrapping.
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
    restrict_c_in_plane: bool = False,
):
    """Build a volume move kernel.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.
        flat_v_prior: If True, use flat volume prior (p_accept=1).
        restrict_c_in_plane: Passed to the cell-geometry guard; see
            :func:`jaxrens.utils.cell.check_cell_shape`.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    # Cell-geometry guard, defined once in the constraints framework. The
    # kernel claims this constraint (rather than the central MWG gate) because
    # its result also gates the neighbor-bucket bookkeeping below and the
    # reject-reason ordering; see jaxrens.constraints.cell_geometry.
    cell_geometry = build_cell_geometry(
        n_atoms,
        max_vol_per_atom,
        min_vol_per_atom,
        min_aspect,
        restrict_c_in_plane,
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


def build_normal_strain_kernel(
    backend: Any,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
    flat_v_prior: bool = False,
    restrict_c_in_plane: bool = False,
):
    """Build a kernel straining the cell along the normal to the (a, b) plane.

    With ``n = a x b / |a x b|`` and ``h = |c . n|`` the cell height, propose
    ``h' = h + step_size * N(0, 1)`` (Angstrom) and apply

        T = I + (lambda - 1) n n^T,   lambda = h' / h

    to the cell and to every position.  ``a`` and ``b`` are unchanged
    (``a . n = b . n = 0``), the in-plane part of ``c`` is unchanged, and so
    are the fractional coordinates of every atom.  In a layered host every
    gallery widens or narrows by the same factor.

    At fixed ``a`` and ``b`` the volume is proportional to ``h``, so the
    volume prior ``V^N dV`` becomes ``h^N dh``.  The proposal is symmetric
    in ``h``, so the prior acceptance is ``min(1, lambda^N)``, or 1 with
    ``flat_v_prior``.  A proposal with ``h' <= 0`` is rejected as an
    invalid cell (reason 2).

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.
        flat_v_prior: If True, use flat volume prior (p_accept=1).
        restrict_c_in_plane: Passed to the cell-geometry guard; see
            :func:`jaxrens.utils.cell.check_cell_shape`.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    cell_geometry = build_cell_geometry(
        n_atoms,
        max_vol_per_atom,
        min_vol_per_atom,
        min_aspect,
        restrict_c_in_plane,
    )

    def step(rng_key, state, likelihood_constraint):
        k1, k2 = jax.random.split(rng_key)
        cell = state.cell

        n = jnp.cross(cell[0], cell[1])
        n = n / jnp.linalg.norm(n)
        h = jnp.abs(jnp.dot(cell[2], n))

        new_h = h + state.step_size * jax.random.normal(k1)
        positive = new_h > 0.0
        # A non-positive height is rejected below; evaluate the identity
        # instead so the backend never sees a degenerate cell.
        lam = jnp.where(positive, new_h / h, 1.0)
        transform = jnp.eye(3) + (lam - 1.0) * jnp.outer(n, n)

        # HIGHEST precision: see build_kernel.
        new_cell = jnp.einsum(
            "ij,jk->ik", cell, transform, precision=jax.lax.Precision.HIGHEST
        )
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

        cell_valid = positive & cell_geometry(
            new_positions, state.types, new_cell
        )

        p_accept = jnp.where(
            flat_v_prior, 1.0, jnp.minimum(1.0, lam**n_atoms)
        )
        energy_ok = result.energy < likelihood_constraint
        prior_ok = jax.random.uniform(k2) < p_accept

        return finalize_cell_move(
            state,
            new_positions,
            new_cell,
            result.energy,
            result.max_neighbor_count,
            result.overflow,
            cell_valid,
            energy_ok,
            prior_ok=prior_ok,
        )

    return step
