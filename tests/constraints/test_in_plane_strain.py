"""Tests for the in-plane strain constraint of layered cells."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.constraints.base import make_move_gate
from jaxrens.constraints.in_plane_strain import (
    build_in_plane_strain,
    in_plane_strain_descriptor,
    principal_stretches,
    reference_lattice_2d,
)

# 3x3 graphene cell with c along z, as in the LiC walker sets.
A = 7.386
REF_2D = reference_lattice_2d(A, A, 120.0)
REF_INV_T = jnp.asarray(np.linalg.inv(REF_2D).T)
HEX = np.array(
    [[A, 0.0, 0.0], [-A / 2, A * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 6.8]]
)

DUMMY_POS = jnp.zeros((4, 3))
DUMMY_TYPES = jnp.zeros((4,), dtype=jnp.int32)


def _lam(cell):
    return np.asarray(principal_stretches(jnp.asarray(cell), REF_INV_T))


def _valid(cell, lam_min=0.94, lam_max=1.06):
    pred = build_in_plane_strain(REF_2D, lam_min, lam_max)
    return bool(pred(DUMMY_POS, DUMMY_TYPES, jnp.asarray(cell)))


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_reference_cell_is_unstrained():
    assert _lam(HEX) == pytest.approx([1.0, 1.0], abs=1e-6)
    assert _valid(HEX)


@pytest.mark.parametrize("s, ok", [(1.05, True), (1.07, False), (0.93, False)])
def test_isotropic_scaling(s, ok):
    assert _lam(HEX * s) == pytest.approx([s, s], rel=1e-5)
    assert _valid(HEX * s) is ok


def test_rigid_rotation_is_not_strain():
    # Includes a tilt of the (a, b) plane out of xy.
    rot = _rotation([1.0, 2.0, 0.5], 0.7)
    cell = HEX @ rot.T
    assert _lam(cell) == pytest.approx([1.0, 1.0], abs=1e-5)
    assert _valid(cell)


def test_changes_of_c_are_ignored():
    cell = HEX.copy()
    cell[2] = [3.1, -1.7, 9.5]  # tilted and longer c
    assert _lam(cell) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_stretch_move_is_measured_as_its_factors():
    # jaxrens stretch: cell @ diag(e^r, e^-r, 1) on the Cartesian (x, y).
    for r in (0.03, -0.08):
        cell = HEX @ np.diag([np.exp(r), np.exp(-r), 1.0])
        lam = _lam(cell)
        assert lam == pytest.approx(sorted([np.exp(r), np.exp(-r)]), rel=1e-5)
    assert _valid(HEX @ np.diag([np.exp(0.03), np.exp(-0.03), 1.0]))
    assert not _valid(HEX @ np.diag([np.exp(0.08), np.exp(-0.08), 1.0]))


def test_sqrt3_deformation_is_rejected():
    # The 4-coordinated net found by NS: stretches 1/sqrt(3) and sqrt(3).
    s = np.sqrt(3.0)
    cell = HEX @ np.diag([1 / s, s, 1.0])
    assert _lam(cell) == pytest.approx([1 / s, s], rel=1e-5)
    assert not _valid(cell)


def test_shear_of_b_along_a():
    # Shear changes angle and length of b: rejected beyond a small shear.
    cell = HEX.copy()
    cell[1] = cell[1] + 0.1 * cell[0] / np.linalg.norm(cell[0])
    assert _valid(cell)
    cell[1] = cell[1] + 1.0 * cell[0] / np.linalg.norm(cell[0])
    assert not _valid(cell)


def test_descriptor_and_gate_pairing():
    desc = in_plane_strain_descriptor(REF_2D, 0.94, 1.06)
    assert desc.name == "in_plane_strain"
    assert desc.depends_on == frozenset({"cell"})
    assert desc.reject_reason == 4
    # Position-only moves are not gated; cell moves are.
    assert make_move_gate((desc,), frozenset({"positions"})) is None
    gate = make_move_gate((desc,), frozenset({"positions", "cell"}))
    ok, reason = gate(DUMMY_POS, DUMMY_TYPES, jnp.asarray(HEX))
    assert bool(ok) and int(reason) == 0
    s = np.sqrt(3.0)
    ok, reason = gate(
        DUMMY_POS, DUMMY_TYPES, jnp.asarray(HEX @ np.diag([1 / s, s, 1.0]))
    )
    assert not bool(ok) and int(reason) == 4
