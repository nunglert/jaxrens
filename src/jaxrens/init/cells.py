"""JAX-native cell utilities for walker initialization.

Two public functions:
  - sample_initial_volume: sample a cubic cell edge length from the NS volume prior.
  - cell_shape_walk: constant-volume random walk over cell shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxrens.sampling.moves.shear import _build_shear_cell
from jaxrens.utils.cell import get_volume, min_aspect_ratio


# ---------------------------------------------------------------------------
# Pure stretch proposal (mirrors stretch.py kernel logic, without MCState)
# ---------------------------------------------------------------------------

_AXIS_PAIRS = jnp.array([[0, 1], [0, 2], [1, 2]])


def _propose_stretch(cell: jnp.ndarray, key: jax.Array, step_size: float) -> jnp.ndarray:
    """Propose a volume-preserving stretch of `cell`.

    Selects a random axis pair (i, j), scales axis i by exp(rv) and axis j by
    exp(-rv) where rv ~ Normal(0, step_size).  The resulting cell has the same
    volume as the input cell.

    Args:
        cell: (3, 3) cell matrix, rows are lattice vectors.
        key: JAX PRNG key.
        step_size: Standard deviation of the normal displacement.

    Returns:
        new_cell: (3, 3) stretched cell with |det| == |det(cell)|.
    """
    k1, k2 = jax.random.split(key)
    pair_idx = jax.random.randint(k1, (), 0, 3)
    axes = _AXIS_PAIRS[pair_idx]
    i, j = axes[0], axes[1]
    rv = step_size * jax.random.normal(k2)
    diag = jnp.ones(3).at[i].set(jnp.exp(rv)).at[j].set(jnp.exp(-rv))
    return cell @ jnp.diag(diag)


def _propose_shear(cell: jnp.ndarray, key: jax.Array, step_size: float) -> jnp.ndarray:
    """Propose a volume-preserving shear of `cell`.

    Selects a random cell vector index (0, 1, or 2) and displaces it within the
    plane spanned by the other two vectors.  Shear of a column-lattice does not
    change the determinant.

    Args:
        cell: (3, 3) cell matrix, rows are lattice vectors.
        key: JAX PRNG key.
        step_size: Scale of the normal displacements for rv1 and rv2.

    Returns:
        new_cell: (3, 3) sheared cell with |det| == |det(cell)|.
    """
    k1, k2 = jax.random.split(key)
    shear_idx = jax.random.randint(k1, (), 0, 3)
    rvs = step_size * jax.random.normal(k2, (2,))
    return _build_shear_cell(cell, shear_idx, rvs[0], rvs[1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_initial_volume(
    key: jax.Array,
    n_atoms: int,
    max_volume_per_atom: float,
    flat_V_prior: bool = False,
) -> jnp.ndarray:
    """Sample an initial cell edge length (cubic cell) from the NS volume prior.

    Returns the scalar edge length `lc` such that cell = lc * I_3.

    Two prior modes:

    flat_V_prior=True:
        V ~ Uniform(0, n_atoms * max_volume_per_atom).  The edge length is
        lc = V ** (1/3).

    flat_V_prior=False (legacy default, pymatnest convention):
        lc = (n_atoms * max_volume_per_atom * U ** (1 / (n_atoms + 1))) ** (1/3)
        where U ~ Uniform(0, 1).  The exponent 1/(N+1) biases the prior toward
        smaller volumes; this matches the pymatnest / jaxnest convention used in
        randomization.py::create_random_initialise_cell (lines 218-225).

    Args:
        key: JAX PRNG key.
        n_atoms: Number of atoms in the cell.
        max_volume_per_atom: Upper bound on volume per atom (Angstrom^3 or similar).
        flat_V_prior: If True, draw from a flat volume prior; otherwise use the
            V^N biased prior.

    Returns:
        Scalar (0-d) jnp.ndarray: the cubic cell edge length lc.
    """
    u = jax.random.uniform(key, shape=(), dtype=jnp.float32)
    v_max = float(n_atoms) * max_volume_per_atom

    lc_flat = (v_max * u) ** (1.0 / 3.0)
    # 1/(n_atoms+1) exponent: biases toward smaller volumes per pymatnest convention
    lc_vn = (v_max * u ** (1.0 / float(n_atoms + 1))) ** (1.0 / 3.0)

    return jnp.where(flat_V_prior, lc_flat, lc_vn)


def cell_shape_walk(
    key: jax.Array,
    cell: jnp.ndarray,
    n_steps: int,
    step_size_shear: float,
    step_size_stretch: float,
    min_aspect_ratio_val: float,
    n_atoms: int,
    max_volume_per_atom: float,
    min_volume_per_atom: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run n_steps shear+stretch moves on a cell, keeping volume constant.

    At each step a random key selects either a shear or stretch proposal
    (matching the legacy random-choice convention in randomization.py lines
    154-163).  A proposal is accepted only if the resulting cell satisfies
    min_aspect_ratio_val; volume is held constant by construction of both
    proposal kernels.

    max_volume_per_atom and min_volume_per_atom are passed as parameters for
    interface symmetry with later init pipeline stages but are NOT enforced
    here -- volume is held constant throughout the walk and was chosen before
    this function is called.

    Args:
        key: JAX PRNG key.
        cell: (3, 3) starting cell, rows are lattice vectors.
        n_steps: Number of shear/stretch proposals.
        step_size_shear: Scale parameter for shear proposals.
        step_size_stretch: Scale parameter for stretch proposals.
        min_aspect_ratio_val: Minimum acceptable aspect ratio; proposals that
            produce an aspect ratio below this value are rejected.
        n_atoms: Number of atoms (used for aspect-ratio volume denominator).
        max_volume_per_atom: Not enforced; passed for interface completeness.
        min_volume_per_atom: Not enforced; passed for interface completeness.

    Returns:
        (final_cell, n_accepted): final (3, 3) cell and scalar int32 acceptance
        count over n_steps proposals.
    """
    target_volume = get_volume(cell)

    def _body(i, state):
        current_cell, acc, loop_key = state
        loop_key, propose_key, choose_key = jax.random.split(loop_key, 3)

        step_type = jax.random.randint(choose_key, (), 0, 2)

        new_cell_shear = _propose_shear(current_cell, propose_key, step_size_shear)
        new_cell_stretch = _propose_stretch(current_cell, propose_key, step_size_stretch)

        proposed_cell = jax.lax.cond(
            step_type == 0,
            lambda _: new_cell_stretch,
            lambda _: new_cell_shear,
            None,
        )

        volume = get_volume(proposed_cell)
        # Restore volume exactly: scale the proposed cell so |det| == target_volume.
        # This is necessary because floating-point arithmetic in shear/stretch
        # accumulates tiny errors over many steps.
        scale = (target_volume / volume) ** (1.0 / 3.0)
        proposed_cell = proposed_cell * scale

        aspect = min_aspect_ratio(proposed_cell, target_volume)
        valid = aspect >= min_aspect_ratio_val

        accepted_cell = jnp.where(valid, proposed_cell, current_cell)
        new_acc = acc + jnp.where(valid, jnp.int32(1), jnp.int32(0))

        return accepted_cell, new_acc, loop_key

    init_state = (cell, jnp.int32(0), key)
    final_cell, n_accepted, _ = jax.lax.fori_loop(0, n_steps, _body, init_state)
    return final_cell, n_accepted
