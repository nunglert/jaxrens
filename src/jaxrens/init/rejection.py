"""Rejection-sampling loop for initial position generation.

The public function ``rejection_sample_positions`` wraps a JIT-compiled
inner check (energy + pair-distance) in a Python while-loop.  On exhaustion
it raises ``RuntimeError`` with per-reason counters.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp

from jaxrens.init.positions import (
    grid_positions_in_cell,
    uniform_positions_in_cell,
)

# ---------------------------------------------------------------------------
# Minimum-image pair-distance check
# ---------------------------------------------------------------------------


@jax.jit
def _check_min_distance(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    min_distance: float,
) -> jnp.ndarray:
    """Return True if any pair of atoms is closer than min_distance under PBC.

    Minimum-image convention: fractional displacements are wrapped to [-0.5, 0.5]
    before converting back to Cartesian.

    Args:
        positions: (n_atoms, 3) Cartesian positions.
        cell: (3, 3) cell matrix, rows are lattice vectors.
        min_distance: Distance threshold.

    Returns:
        Scalar bool: True if any pair is too close.
    """
    cell_inv = jnp.linalg.inv(cell)
    frac = positions @ cell_inv

    diff_frac = frac[:, None, :] - frac[None, :, :]
    diff_frac = diff_frac - jnp.round(diff_frac)
    diff_cart = diff_frac @ cell

    dist_sq = jnp.sum(diff_cart**2, axis=-1)

    n = positions.shape[0]
    i_idx, j_idx = jnp.tril_indices(n, k=-1)
    pair_dist_sq = dist_sq[i_idx, j_idx]

    return jnp.any(pair_dist_sq < min_distance**2)


# ---------------------------------------------------------------------------
# JIT-compiled inner check
# ---------------------------------------------------------------------------

_REJECT_OK = jnp.int32(0)
_REJECT_ENERGY = jnp.int32(1)
_REJECT_NAN = jnp.int32(2)
_REJECT_CLOSE = jnp.int32(3)


@partial(jax.jit, static_argnames=("energy_fn",))
def _inner_check(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    cell: jnp.ndarray,
    start_energy_ceiling: jnp.ndarray,
    min_distance: jnp.ndarray,
    energy_fn: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate one candidate configuration.

    `energy_fn` is a static arg, so distinct backend instances cache separately
    but repeated calls with the same backend reuse the compiled trace.
    Scalars are dynamic to avoid retracing on minor parameter changes.
    """
    energy = energy_fn(positions, types, cell, 0).energy

    too_close = _check_min_distance(positions, cell, min_distance)

    is_nan = jnp.isnan(energy)
    over_ceiling = energy > start_energy_ceiling

    reject_code = jnp.where(
        is_nan,
        _REJECT_NAN,
        jnp.where(
            over_ceiling,
            _REJECT_ENERGY,
            jnp.where(too_close, _REJECT_CLOSE, _REJECT_OK),
        ),
    )
    return energy, reject_code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rejection_sample_positions(
    key: jax.Array,
    *,
    cell: jnp.ndarray,
    types: jnp.ndarray,
    n_atoms: int,
    energy_fn: Callable,
    start_energy_ceiling: float,
    min_distance: float,
    max_tries: int,
    mode: Literal["uniform", "grid"],
    grid_distance: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw and reject positions until a valid configuration is found.

    Loop: draw positions (uniform or grid), evaluate energy + pair distances
    via a JIT-compiled inner check, accept if all criteria pass.

    Args:
        key: JAX PRNG key.
        cell: (3, 3) cell matrix.
        types: (n_atoms,) int32 type indices.
        n_atoms: Number of atoms.
        energy_fn: Callable (positions, types, cell, max_neighbors) ->
            (energy, count, overflow).  Should already have backend kwargs bound.
        start_energy_ceiling: Absolute energy ceiling for the whole configuration.
        min_distance: Minimum allowed interatomic distance under PBC.
        max_tries: Maximum number of draw attempts before raising RuntimeError.
        mode: "uniform" or "grid".
        grid_distance: Grid spacing for grid mode (used by grid_positions_in_cell).

    Returns:
        (positions, energy): (n_atoms, 3) and scalar jnp.ndarray.

    Raises:
        RuntimeError: If max_tries attempts all fail, with per-reason counters.
        ValueError: (re-raised) if grid is too coarse to fit n_atoms.
    """
    ceiling = jnp.asarray(start_energy_ceiling)
    min_dist = jnp.asarray(min_distance)

    n_energy_over_ceiling = 0
    n_nan_energy = 0
    n_atoms_too_close = 0

    for _attempt in range(max_tries):
        key, subkey = jax.random.split(key)

        if mode == "uniform":
            positions = uniform_positions_in_cell(subkey, cell, n_atoms)
        else:
            try:
                positions = grid_positions_in_cell(
                    subkey, cell, n_atoms, grid_distance
                )
            except ValueError as exc:
                raise ValueError(
                    f"grid_positions_in_cell failed during rejection sampling: {exc}"
                ) from exc

        energy, reject_code = _inner_check(
            positions, types, cell, ceiling, min_dist, energy_fn
        )
        code = int(reject_code)

        if code == 0:
            return positions, energy

        if code == 1:
            n_energy_over_ceiling += 1
        elif code == 2:
            n_nan_energy += 1
        elif code == 3:
            n_atoms_too_close += 1

    counters = {
        "n_energy_over_ceiling": n_energy_over_ceiling,
        "n_nan_energy": n_nan_energy,
        "n_atoms_too_close": n_atoms_too_close,
    }
    raise RuntimeError(
        f"Rejection sampling failed after {max_tries} attempts: {counters}. "
        f"Tune one of: init.random_initialise_pos, init.pos_randomization_mode, "
        f"init.init_distance_criterion, init.start_energy_ceiling_per_atom, "
        f"init.random_init_max_n_tries."
    )
