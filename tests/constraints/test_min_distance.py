"""Tests for the minimum inter-atomic distance constraint."""

from __future__ import annotations

import jax.numpy as jnp

from jaxrens.constraints.min_distance import (
    build_min_distance,
    min_distance_descriptor,
)

OPEN = jnp.zeros((3, 3))


def _uniform(d: float, n_types: int = 1):
    return jnp.full((n_types, n_types), d)


def test_scalar_threshold_accept_and_reject():
    positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    types = jnp.array([0, 0])

    assert bool(build_min_distance(_uniform(1.0))(positions, types, OPEN))
    assert not bool(build_min_distance(_uniform(2.0))(positions, types, OPEN))


def test_single_atom_always_valid():
    positions = jnp.array([[0.0, 0.0, 0.0]])
    types = jnp.array([0])
    # Self-pair (the diagonal) must never trigger a violation.
    assert bool(build_min_distance(_uniform(5.0))(positions, types, OPEN))


def test_per_pair_thresholds():
    # Types 0 and 1 at distance 1.5. d_min[0,1] = 1.0 (ok) but a tighter
    # 0-0 floor of 2.0 must not apply to the 0-1 pair.
    positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    types = jnp.array([0, 1])
    d_min = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    assert bool(build_min_distance(d_min)(positions, types, OPEN))

    # Raise the cross-pair floor above 1.5 -> now violated.
    d_min_tight = jnp.array([[2.0, 1.8], [1.8, 2.0]])
    assert not bool(build_min_distance(d_min_tight)(positions, types, OPEN))


def test_swap_changes_validity_under_per_pair():
    # Same geometry; swapping which atom is which species flips validity
    # when the cross-pair floor differs from the like-pair floor.
    positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    d_min = jnp.array([[1.0, 1.8], [1.8, 1.0]])  # like: 1.0, cross: 1.8
    pred = build_min_distance(d_min)
    assert bool(pred(positions, jnp.array([0, 0]), OPEN))  # like-pair ok
    assert not bool(
        pred(positions, jnp.array([0, 1]), OPEN)
    )  # cross-pair fails


def test_minimum_image_periodic_violation():
    cell = jnp.eye(3) * 10.0
    positions = jnp.array([[0.5, 0.5, 0.5], [9.5, 0.5, 0.5]])  # 1 A across PBC
    types = jnp.array([0, 0])
    pred = build_min_distance(_uniform(1.5))
    assert not bool(pred(positions, types, cell))  # MIC catches it
    assert bool(pred(positions, types, OPEN))  # raw 9 A is fine


def test_descriptor_type_dependence_flag():
    desc_uniform = min_distance_descriptor(_uniform(1.0), type_dependent=False)
    assert desc_uniform.depends_on == frozenset({"positions", "cell"})
    assert desc_uniform.reject_reason == 4

    desc_pair = min_distance_descriptor(
        jnp.array([[1.0, 1.8], [1.8, 1.0]]), type_dependent=True
    )
    assert desc_pair.depends_on == frozenset({"positions", "cell", "types"})
