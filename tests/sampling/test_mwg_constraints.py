"""Tests for the configuration-constraint gate in build_mwg.

An unreachable minimum distance makes every proposal violate the constraint,
so the outcome does not depend on the random proposal.  Runs without a
constraint, or with a satisfiable one, are the controls.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.moves import RandomWalkMoveSpec
from jaxrens.constraints.min_distance import min_distance_descriptor
from jaxrens.sampling.mwg import build_mwg

# Two atoms 0.5 apart.  With the huge EMAX every proposal passes the energy
# test, so only the constraint gate decides acceptance.
POSITIONS = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
TYPES = jnp.array([0, 0], dtype=jnp.int32)
CELL = jnp.zeros((3, 3))
EMAX = 1.0e9


def _make(constraint_descriptors, mutates=None):
    """Build step_fn and the initial state with the given constraints.

    ``mutates`` overrides the aspects the move declares it writes.
    """
    backend = create_harmonic()
    move = RandomWalkMoveSpec(step_size=0.3).to_descriptor()
    if mutates is not None:
        move = dataclasses.replace(move, mutates=frozenset(mutates))
    init_fn, step_fn, _ = build_mwg(
        backend,
        [move],
        constraint_descriptors=constraint_descriptors,
    )
    energy = backend(POSITIONS, TYPES, CELL, 0).energy
    state = init_fn(POSITIONS, TYPES, energy, CELL)
    return step_fn, state


def _run(step_fn, state, n_steps=25, seed=0):
    """Run n_steps and return the state, reject reasons and accepts."""
    key = jax.random.key(seed)
    reasons, accepts = [], []
    for _ in range(n_steps):
        key, sub, move_key = jax.random.split(key, 3)
        state, info = step_fn(sub, state, EMAX, move_key)
        reasons.append(int(info.reject_reason))
        accepts.append(bool(info.accepted))
    return state, reasons, accepts


def test_unreachable_constraint_rejects_every_move():
    """Every proposal violates the constraint and is rejected with reason 4."""
    desc = (
        min_distance_descriptor(jnp.full((1, 1), 100.0), type_dependent=False),
    )
    step_fn, state0 = _make(desc)
    state, reasons, accepts = _run(step_fn, state0)

    assert not any(accepts)
    assert all(r == 4 for r in reasons)
    assert int(state.n_accepted.sum()) == 0
    assert int(state.n_proposed.sum()) == 25
    assert jnp.allclose(state.positions, POSITIONS)


def test_satisfiable_constraint_matches_unconstrained():
    """A gate that always passes (d_min=0) does not change the accepts."""
    step_fn_free, s0_free = _make(())
    state_free, _, accepts_free = _run(step_fn_free, s0_free)

    desc = (
        min_distance_descriptor(jnp.full((1, 1), 0.0), type_dependent=False),
    )
    step_fn_gated, s0_gated = _make(desc)
    _, reasons_gated, accepts_gated = _run(step_fn_gated, s0_gated)

    assert accepts_gated == accepts_free
    assert any(accepts_free)
    assert not jnp.allclose(state_free.positions, POSITIONS)
    assert all(r != 4 for r in reasons_gated)


def test_constraint_does_not_gate_a_move_that_leaves_its_aspect_alone():
    """A move that only mutates types is not gated by a distance constraint.

    The move still displaces the atoms here, but build_mwg pairs constraints
    with moves by their declared ``mutates``, so no gate is attached and
    reason 4 never appears.
    """
    desc = (
        min_distance_descriptor(jnp.full((1, 1), 100.0), type_dependent=False),
    )
    step_fn, state0 = _make(desc, mutates={"types"})

    _, reasons, accepts = _run(step_fn, state0)

    assert any(accepts)
    assert all(r != 4 for r in reasons)
