"""Minimum inter-atomic distance constraint.

Rejects any configuration in which a pair of atoms is closer than a
per-species-pair threshold ``d_min[type_i, type_j]`` (Angstrom). Distances
use the minimum-image convention under a periodic cell, so contacts across a
boundary are caught (see :func:`jaxrens.backends.geometry.pairwise_distances`).

A *scalar* threshold is the special case of a uniform ``d_min`` matrix; a
*per-species-pair* threshold lets different element pairs carry different
floors (e.g. a tighter contact for a small-atom pair). When the matrix is
non-uniform the predicate's value depends on the atom *types*, so a move that
relabels species (alchemical morph / single-atom swap) can change validity
and must be gated — this is expressed by the descriptor's ``depends_on``
(see :func:`min_distance_descriptor`).
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxrens.backends.geometry import pairwise_distances
from jaxrens.constraints.base import ConstraintDescriptor


def build_min_distance(d_min_matrix) -> object:
    """Build a minimum-distance predicate from a per-pair threshold matrix.

    Args:
        d_min_matrix: ``(n_types, n_types)`` symmetric matrix of minimum
            allowed distances, indexed by the run's integer type labels. A
            zero entry disables the constraint for that pair.

    Returns:
        ``is_valid(positions, types, cell) -> bool`` scalar predicate.
    """
    d_min = jnp.asarray(d_min_matrix)

    def is_valid(positions, types, cell):
        r = pairwise_distances(positions, cell)  # (N, N), MIC-aware
        thresh = d_min[types[:, None], types[None, :]]  # (N, N)
        n = r.shape[0]
        off_diagonal = ~jnp.eye(n, dtype=bool)
        violated = off_diagonal & (r < thresh)
        return ~jnp.any(violated)

    return is_valid


def min_distance_descriptor(
    d_min_matrix, *, type_dependent: bool
) -> ConstraintDescriptor:
    """Wrap a minimum-distance matrix in a :class:`ConstraintDescriptor`.

    Args:
        d_min_matrix: ``(n_types, n_types)`` threshold matrix.
        type_dependent: Whether thresholds vary by species pair. When
            ``True`` the predicate reads ``types`` (so type-changing moves are
            gated); when ``False`` (uniform matrix) only geometry-moving moves
            are gated. The resolver sets this from the parsed config.

    Returns:
        A descriptor registering this constraint with the sampler.
    """
    depends_on = {"positions", "cell"}
    if type_dependent:
        depends_on = depends_on | {"types"}
    return ConstraintDescriptor(
        name="minimum_distance",
        depends_on=frozenset(depends_on),
        build=build_min_distance,
        build_kwargs={"d_min_matrix": d_min_matrix},
        reject_reason=4,
    )
