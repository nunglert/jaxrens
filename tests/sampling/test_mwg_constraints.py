"""Constraint-gate behavior in build_mwg.

Uses an unreachable minimum-distance threshold so the gate's effect is
deterministic regardless of the random proposal: every proposed move is
constraint-violated, hence every step must be rejected with reject_reason 4
and the walker must never move. The no-constraint / satisfiable-constraint
runs are the controls.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.moves import RandomWalkMoveSpec
from jaxrens.constraints.min_distance import min_distance_descriptor
from jaxrens.sampling.mwg import build_mwg

# Two atoms 0.5 A apart; harmonic energy keeps every proposal below the
# (huge) likelihood threshold, so acceptance is governed purely by the
# constraint gate.
POSITIONS = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
TYPES = jnp.array([0, 0], dtype=jnp.int32)
CELL = jnp.zeros((3, 3))
EMAX = 1.0e9


def _make(constraint_descriptors):
    backend = create_harmonic()
    init_fn, step_fn, _ = build_mwg(
        backend,
        [RandomWalkMoveSpec(step_size=0.3).to_descriptor()],
        constraint_descriptors=constraint_descriptors,
    )
    energy = backend(POSITIONS, TYPES, CELL, 0).energy
    state = init_fn(POSITIONS, TYPES, energy, CELL)
    return step_fn, state


def _run(step_fn, state, n_steps=25, seed=0):
    key = jax.random.key(seed)
    reasons, accepts = [], []
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        state, info = step_fn(sub, state, EMAX)
        reasons.append(int(info.reject_reason))
        accepts.append(bool(info.accepted))
    return state, reasons, accepts


def test_unreachable_constraint_rejects_every_move():
    desc = (
        min_distance_descriptor(jnp.full((1, 1), 100.0), type_dependent=False),
    )
    step_fn, state0 = _make(desc)
    state, reasons, accepts = _run(step_fn, state0)

    assert not any(accepts)  # nothing accepted
    assert all(r == 4 for r in reasons)  # all blamed on the constraint
    assert int(state.n_accepted.sum()) == 0
    # Walker never moved off its initial configuration.
    assert jnp.allclose(state.positions, POSITIONS)


def test_satisfiable_constraint_matches_unconstrained():
    # A passing gate (d_min=0) must not change which moves are accepted
    # relative to no constraint at all, on the same seed.
    _, base_state = _make(())
    step_fn_free, s0_free = _make(())
    _, _, accepts_free = _run(step_fn_free, s0_free)

    desc = (
        min_distance_descriptor(jnp.full((1, 1), 0.0), type_dependent=False),
    )
    step_fn_gated, s0_gated = _make(desc)
    _, reasons_gated, accepts_gated = _run(step_fn_gated, s0_gated)

    assert accepts_gated == accepts_free  # identical accept pattern
    assert any(accepts_free)  # control actually moves
    assert all(r != 4 for r in reasons_gated)  # gate never fires


def test_unconstrained_walker_moves():
    # Sanity: with no constraint the walker does move (so the block in the
    # first test is really the gate, not a frozen setup).
    step_fn, state0 = _make(())
    state, _, accepts = _run(step_fn, state0)
    assert any(accepts)
    assert not jnp.allclose(state.positions, POSITIONS)
