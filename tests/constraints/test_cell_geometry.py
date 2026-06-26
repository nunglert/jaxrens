"""Tests for the cell-geometry constraint (framework wrapper of check_cell_shape)."""

from __future__ import annotations

import jax.numpy as jnp

from jaxrens.constraints.base import make_move_gate
from jaxrens.constraints.cell_geometry import (
    build_cell_geometry,
    cell_geometry_descriptor,
)
from jaxrens.utils.cell import check_cell_shape

DUMMY_POS = jnp.zeros((4, 3))
DUMMY_TYPES = jnp.zeros((4,), dtype=jnp.int32)


def _valid(cell, n_atoms, **kw):
    return bool(
        build_cell_geometry(n_atoms, **kw)(DUMMY_POS, DUMMY_TYPES, cell)
    )


def test_predicate_matches_check_cell_shape():
    cell = jnp.eye(3) * 2.0
    for n_atoms in (1, 4):
        for kw in (
            {},
            {
                "max_vol_per_atom": 10.0,
                "min_vol_per_atom": 1.0,
                "min_aspect": 0.5,
            },
        ):
            expected = bool(check_cell_shape(cell, n_atoms, **kw))
            assert _valid(cell, n_atoms, **kw) == expected


def test_accepts_well_formed_cell():
    assert _valid(jnp.eye(3) * 2.0, n_atoms=4)  # vol/atom = 2, aspect 1


def test_rejects_too_large_and_too_small():
    assert not _valid(jnp.eye(3) * 10.0, n_atoms=1)  # vol/atom = 1000 > 100
    assert not _valid(jnp.eye(3) * 1.0, n_atoms=4)  # vol/atom = 0.25 < 1


def test_rejects_bad_aspect():
    skewed = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.05]])
    assert not _valid(skewed, n_atoms=1, max_vol_per_atom=1e6, min_aspect=0.5)


def test_descriptor_aspects_and_bucket():
    desc = cell_geometry_descriptor(n_atoms=8)
    assert desc.name == "cell_geometry"
    assert desc.depends_on == frozenset({"cell"})
    assert desc.reject_reason == 2


def test_descriptor_is_gate_capable_for_cell_moves():
    # Not registered by the resolver (kernel-claimed), but it *could* gate a
    # cell-mutating move and must be inert for a types-only move.
    desc = (cell_geometry_descriptor(n_atoms=4, max_vol_per_atom=10.0),)
    assert make_move_gate(desc, frozenset({"positions", "cell"})) is not None
    assert make_move_gate(desc, frozenset({"types"})) is None

    gate = make_move_gate(desc, frozenset({"cell"}))
    ok, reason = gate(
        DUMMY_POS, DUMMY_TYPES, jnp.eye(3) * 10.0
    )  # vol/atom=250>10
    assert not bool(ok)
    assert int(reason) == 2
