"""Stretch move: volume-preserving anisotropic cell deformation.

Proposes a stretch along two random axes that preserves the cell volume.
Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxrens.base import MoveInfo, MoveKernel
from jaxrens.types import Box, Params, Positions, Types
from jaxrens.utils.cell import check_cell_shape


class StretchMoveState(NamedTuple):
    """State for the stretch move."""

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
) -> StretchMoveState:
    """Create initial stretch move state."""
    return StretchMoveState(
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
):
    """Build a stretch move kernel.

    Args:
        energy_fn: Callable conforming to EnergyFn protocol.
        params: Opaque pytree of backend parameters.
        n_atoms: Number of atoms.
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    # Pre-define the 3 axis pairs: (0,1), (0,2), (1,2)
    axis_pairs = jnp.array([[0, 1], [0, 2], [1, 2]])

    def step(
        rng_key: jax.Array,
        state: StretchMoveState,
        likelihood_constraint: float,
    ) -> tuple[StretchMoveState, MoveInfo]:
        k1, k2 = jax.random.split(rng_key)

        # Pick random axis pair
        pair_idx = jax.random.randint(k1, (), 0, 3)
        axes = axis_pairs[pair_idx]
        i, j = axes[0], axes[1]

        # Random stretch magnitude
        rv = state.step_size * jax.random.normal(k2)

        # Build volume-preserving transform
        # transform[i,i] = exp(rv), transform[j,j] = exp(-rv), rest = 1
        diag = jnp.ones(3)
        diag = diag.at[i].set(jnp.exp(rv))
        diag = diag.at[j].set(jnp.exp(-rv))
        transform = jnp.diag(diag)

        new_box = state.box @ transform
        new_positions = state.positions @ transform

        # Evaluate energy
        new_energy = energy_fn(params, new_positions, state.types, box=new_box)

        # Check cell shape validity
        cell_valid = check_cell_shape(
            new_box, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

        # Accept/reject (no volume prior since volume is preserved)
        accepted = (new_energy < likelihood_constraint) & cell_valid

        out_positions = jnp.where(accepted, new_positions, state.positions)
        out_energy = jnp.where(accepted, new_energy, state.energy)
        out_box = jnp.where(accepted, new_box, state.box)

        new_state = StretchMoveState(
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
) -> MoveKernel:
    """Convenient top-level API."""
    kernel = build_kernel(
        energy_fn, params, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect,
    )
    init_fn = lambda pos, types, energy, box: init(pos, types, energy, box, step_size)
    return MoveKernel(init=init_fn, step=kernel)
