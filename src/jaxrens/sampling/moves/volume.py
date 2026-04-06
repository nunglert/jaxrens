"""Volume move: isotropic volume change with accept/reject.

Proposes an isotropic scaling of the simulation cell and atom positions.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types
from jaxrens.utils.cell import check_cell_shape, get_volume


class VolumeMoveState(NamedTuple):
    """State for the volume move."""

    positions: jnp.ndarray  # (n_atoms, 3)
    types: jnp.ndarray  # (n_atoms,)
    energy: jnp.ndarray  # scalar
    box: jnp.ndarray  # (3, 3)
    step_size: jnp.ndarray  # scalar


def init(
    positions: Positions,
    types: Types,
    energy: float,
    box: Box,
    step_size: float = 0.1,
) -> VolumeMoveState:
    """Create initial volume move state."""
    return VolumeMoveState(
        positions=positions,
        types=types,
        energy=jnp.asarray(energy),
        box=jnp.asarray(box),
        step_size=jnp.asarray(step_size),
    )


def build_kernel(
    energy_fn: Any,
    params: Params,
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
    flat_v_prior: bool = False,
):
    """Build a volume move kernel.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.
        flat_v_prior: If True, use flat volume prior (p_accept=1).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(
        rng_key: jax.Array,
        state: VolumeMoveState,
        likelihood_constraint: float,
    ) -> tuple[VolumeMoveState, MoveInfo]:
        k1, k2 = jax.random.split(rng_key)

        old_V = get_volume(state.box)

        # Propose volume change
        dV = state.step_size * n_atoms * jax.random.normal(k1)
        new_V = jnp.abs(old_V + dV)
        vol_ratio = new_V / old_V

        # Isotropic scaling
        scale = vol_ratio ** (1.0 / 3.0)
        transform = jnp.eye(3) * scale

        new_box = state.box @ transform
        new_positions = state.positions @ transform

        # Evaluate energy
        new_energy = energy_fn(params, new_positions, state.types, box=new_box)

        # Check cell shape validity
        cell_valid = check_cell_shape(
            new_box, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        # Volume prior acceptance
        p_accept = jnp.where(
            flat_v_prior, 1.0, jnp.minimum(1.0, vol_ratio ** n_atoms)
        )

        # Accept/reject
        accepted = (
            (new_energy < likelihood_constraint)
            & cell_valid
            & (jax.random.uniform(k2) < p_accept)
        )

        out_positions = jnp.where(accepted, new_positions, state.positions)
        out_energy = jnp.where(accepted, new_energy, state.energy)
        out_box = jnp.where(accepted, new_box, state.box)

        new_state = VolumeMoveState(
            positions=out_positions,
            types=state.types,
            energy=out_energy,
            box=out_box,
            step_size=state.step_size,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-out_energy,
            n_evaluations=1,
        )

        return new_state, info

    return step


def as_top_level_api(
    energy_fn: Any,
    params: Params,
    n_atoms: int,
    step_size: float = 0.1,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
    flat_v_prior: bool = False,
) -> MoveKernel:
    """Convenient top-level API."""
    kernel = build_kernel(
        energy_fn, params, n_atoms, max_vol_per_atom, min_vol_per_atom,
        min_aspect, flat_v_prior,
    )
    init_fn = lambda pos, types, energy, box: init(pos, types, energy, box, step_size)
    return MoveKernel(init=init_fn, step=kernel)
