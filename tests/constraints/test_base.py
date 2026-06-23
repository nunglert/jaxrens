"""Tests for the constraint descriptor + per-move gate machinery."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxrens.constraints.base import ConstraintDescriptor, make_move_gate
from jaxrens.constraints.min_distance import (
    build_min_distance,
    min_distance_descriptor,
)

OPEN = jnp.zeros((3, 3))


def _min_dist_desc(d: float, type_dependent: bool = False):
    return min_distance_descriptor(
        jnp.full((1, 1), d), type_dependent=type_dependent
    )


def test_descriptor_rejects_unknown_aspect():
    with pytest.raises(ValueError, match="unknown depends_on"):
        ConstraintDescriptor(
            name="bogus",
            depends_on=frozenset({"velocities"}),
            build=lambda: (lambda p, t, c: jnp.asarray(True)),
        )


def test_gate_none_when_move_cannot_violate():
    # Constraint depends on positions/cell; a type-only move cannot violate it.
    desc = (_min_dist_desc(1.0),)
    assert make_move_gate(desc, frozenset({"types"})) is None


def test_gate_present_when_move_can_violate():
    desc = (_min_dist_desc(2.0),)
    gate = make_move_gate(desc, frozenset({"positions"}))
    assert gate is not None

    positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    types = jnp.array([0, 0])
    ok, reason = gate(positions, types, OPEN)
    assert not bool(ok)
    assert int(reason) == 4  # configuration-constraint bucket


def test_gate_reports_zero_reason_when_satisfied():
    gate = make_move_gate((_min_dist_desc(1.0),), frozenset({"positions"}))
    positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    ok, reason = gate(positions, jnp.array([0, 0]), OPEN)
    assert bool(ok)
    assert int(reason) == 0


def test_per_pair_constraint_gates_type_moves():
    # A type-dependent min-distance must gate a type-mutating move.
    desc = (
        min_distance_descriptor(
            jnp.array([[1.0, 1.8], [1.8, 1.0]]), type_dependent=True
        ),
    )
    assert make_move_gate(desc, frozenset({"types"})) is not None
