"""Visual check of volume_axis: write one frame sequence per test structure.

The move is applied with no energy model (constant energy, flat prior, wide
cell bounds), so every proposal is accepted and the sequence shows the move
and nothing else.  Each test also asserts, frame by frame, that only the
interlayer distance changes.

Output: one extxyz per structure in ``$JAXRENS_VISUAL_OUT`` (default
``tests/sampling/moves/_visual_out``).  Open them with ``ase gui`` or OVITO
and play the frames; ``*.extxyz`` is git-ignored.  Per-frame info: ``step``,
``volume``, ``d_gallery`` (distance between neighbouring layers, i.e. the
cell height along the stacking normal divided by the number of layers).
"""

from __future__ import annotations

import os
from pathlib import Path

import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms

from jaxrens.backends.base import BackendResult
from jaxrens.sampling.moves.volume_axis import build_kernel
from jaxrens.state.mc_state import MCState
from jaxrens.utils.cell import get_volume

A_LATTICE = 2.46  # graphene lattice constant (Angstrom)
D_START = 3.35  # starting interlayer distance (Angstrom)
N_FRAMES = 40
N_LAYERS = 2
STEP_SIZE = 0.25  # ~3% volume change per proposal for these cells


class _ZeroEnergyBackend:
    """No energy model: every proposal passes the likelihood check."""

    r_cutoff = 0.0

    def __call__(
        self, positions, species, cell, max_neighbors=0, ensemble_params=None
    ):
        return BackendResult(energy=jnp.zeros(()))


# -- test structures ---------------------------------------------------------

# Fractional in-plane coordinates in the 3x3 supercell: carbon sublattices of
# one layer, and the hollow sites above hexagon centres.
_C_SITES = [(0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0)]
_LI_FILL_AA = [(i, j) for i in range(3) for j in range(3) if (i - j) % 3 == 0]


def _carbon_layer(shift=(0.0, 0.0)):
    """18 C atoms of one layer: (frac_x, frac_y) in the 3x3 cell."""
    out = []
    for i in range(3):
        for j in range(3):
            for bx, by in _C_SITES:
                out.append(
                    (
                        ((i + bx) / 3.0 + shift[0]) % 1.0,
                        ((j + by) / 3.0 + shift[1]) % 1.0,
                    )
                )
    return out


def _lithium(sites_filled, offset=(0, 0)):
    """Li on hollow sites (sqrt3 x sqrt3 pattern) -> (frac_x, frac_y)."""
    return [
        ((i + 2.0 / 3.0) / 3.0 + offset[0], (j + 2.0 / 3.0) / 3.0 + offset[1])
        for i, j in sites_filled
    ]


def _build(stacking, galleries, c_tilt=(0.0, 0.0), rotation=None):
    """Two-layer 3x3 graphite with Li in the listed galleries.

    stacking: "AA" or "AB".  galleries: subset of {0, 1} that hold Li (0 sits
    between layer 0 and layer 1, 1 between layer 1 and the next image).
    c_tilt: in-plane Cartesian part of the c vector (tilted stacking).
    rotation: optional 3x3 matrix rotating the whole structure.
    """
    a = A_LATTICE * 3.0
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0])
    c = np.array([c_tilt[0], c_tilt[1], N_LAYERS * D_START])
    cell = np.array([a1, a2, c])

    shift_b = (1.0 / 9.0, 1.0 / 9.0) if stacking == "AB" else (0.0, 0.0)
    frac, symbols = [], []
    for fx, fy in _carbon_layer():
        frac.append((fx, fy, 0.0))
        symbols.append("C")
    for fx, fy in _carbon_layer(shift_b):
        frac.append((fx, fy, 0.5))
        symbols.append("C")
    for g in galleries:
        for fx, fy in _lithium(_LI_FILL_AA, offset=(g / 9.0, 0.0)):
            frac.append((fx % 1.0, fy % 1.0, 0.25 + 0.5 * g))
            symbols.append("Li")

    positions = np.asarray(frac) @ cell
    if rotation is not None:
        cell, positions = cell @ rotation.T, positions @ rotation.T
    return symbols, positions, cell


def _rot(axis, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


STRUCTURES = {
    # Li6C36: both galleries filled, aligned layers
    "stage1_AA_Li6C36": lambda: _build("AA", [0, 1]),
    # Li3C36: one gallery filled, one empty, Bernal stacking
    "stage2_AB_Li3C36": lambda: _build("AB", [0]),
    # c has an in-plane part (as after shear moves): no lateral sliding
    "stage1_AB_tilted_c": lambda: _build("AB", [0, 1], c_tilt=(1.6, 0.9)),
    # whole structure rotated: the stacking normal is not along z
    "stage2_AB_rotated": lambda: _build(
        "AB", [0], rotation=_rot("x", 30.0) @ _rot("z", 40.0)
    ),
}


# -- helpers -----------------------------------------------------------------


def _make_state(positions, cell):
    return MCState(
        positions=jnp.asarray(positions),
        types=jnp.zeros(len(positions), dtype=jnp.int32),
        energy=jnp.zeros(()),
        cell=jnp.asarray(cell),
        step_size=jnp.asarray(STEP_SIZE),
        step_sizes=jnp.array([STEP_SIZE]),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
    )


def _normal(cell):
    n = np.cross(cell[0], cell[1])
    return n / np.linalg.norm(n)


def _d_gallery(cell):
    cell = np.asarray(cell, dtype=float)
    return (
        abs(np.linalg.det(cell))
        / np.linalg.norm(np.cross(cell[0], cell[1]))
        / N_LAYERS
    )


def _frac(positions, cell):
    return np.linalg.solve(
        np.asarray(cell, dtype=float).T, np.asarray(positions, dtype=float).T
    ).T


@pytest.fixture(scope="module")
def out_dir():
    default = Path(__file__).parent / "_visual_out"
    path = Path(os.environ.get("JAXRENS_VISUAL_OUT", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_sequence_changes_only_interlayer_distance(name, out_dir):
    symbols, positions, cell = STRUCTURES[name]()
    n_atoms = len(symbols)
    step = jax.jit(
        build_kernel(
            _ZeroEnergyBackend(),
            n_atoms,
            max_vol_per_atom=1.0e4,
            min_vol_per_atom=1.0e-3,
            min_aspect=1.0e-3,
            flat_v_prior=True,
            axis=2,
        )
    )

    state = _make_state(positions, cell)
    cell0 = np.asarray(state.cell, dtype=float)
    frac0 = _frac(state.positions, state.cell)
    n0 = _normal(cell0)
    # In-plane (parallel to the layers) part of every atom and of c.
    plane_proj0 = np.asarray(state.positions) - np.outer(
        np.asarray(state.positions) @ n0, n0
    )
    c_par0 = cell0[2] - (cell0[2] @ n0) * n0

    frames = []

    def record(st, i):
        at = Atoms(
            symbols=symbols,
            positions=np.asarray(st.positions, dtype=float),
            cell=np.asarray(st.cell, dtype=float),
            pbc=True,
        )
        at.info["step"] = i
        at.info["volume"] = float(get_volume(st.cell))
        at.info["d_gallery"] = float(_d_gallery(st.cell))
        frames.append(at)

    record(state, 0)
    keys = jax.random.split(jax.random.key(11), N_FRAMES)
    for i, key in enumerate(keys, start=1):
        state, info = step(key, state, 1e9)
        assert bool(info.accepted)
        record(state, i)

        c_now = np.asarray(state.cell, dtype=float)
        pos_now = np.asarray(state.positions, dtype=float)
        # a and b never change, and neither does the plane normal.
        np.testing.assert_array_equal(c_now[:2], cell0[:2])
        np.testing.assert_allclose(_normal(c_now), n0, atol=1e-6)
        # c: only its normal component moves, the in-plane part is fixed.
        np.testing.assert_allclose(
            c_now[2] - (c_now[2] @ n0) * n0, c_par0, atol=1e-4
        )
        # atoms: fractional coordinates are kept, and so is the position
        # projected on the layer plane (no lateral displacement or sliding).
        np.testing.assert_allclose(_frac(pos_now, c_now), frac0, atol=1e-4)
        np.testing.assert_allclose(
            pos_now - np.outer(pos_now @ n0, n0), plane_proj0, atol=1e-3
        )

    d = np.array([f.info["d_gallery"] for f in frames])
    assert (
        d.max() / d.min() > 1.05
    ), "the sequence should show a visible change"
    assert d.min() > 1.0

    ase.io.write(out_dir / f"{name}.extxyz", frames)
