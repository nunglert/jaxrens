"""Tests for the volume_axis move: one lattice vector lengthens, the rest stay."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.moves import VolumeAxisMoveSpec
from jaxrens.sampling.moves.volume_axis import build_kernel
from jaxrens.state.mc_state import MCState
from jaxrens.utils.cell import get_volume

N_ATOMS = 2


class _ZeroEnergyBackend:
    """Constant energy: the likelihood never rejects, only cell/prior do."""

    r_cutoff = 0.0

    def __call__(
        self, positions, species, cell, max_neighbors=0, ensemble_params=None
    ):
        return BackendResult(energy=jnp.zeros(()))


def _state(positions, cell, step_size=0.5):
    return MCState(
        positions=jnp.asarray(positions),
        types=jnp.zeros(len(positions), dtype=jnp.int32),
        energy=jnp.zeros(()),
        cell=jnp.asarray(cell),
        step_size=jnp.asarray(step_size),
        step_sizes=jnp.array([step_size]),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
    )


def _frac(positions, cell):
    return jnp.linalg.solve(cell.T, positions.T).T


def _kernel(axis=2):
    return jax.jit(
        build_kernel(
            _ZeroEnergyBackend(),
            N_ATOMS,
            max_vol_per_atom=100.0,
            min_vol_per_atom=0.1,
            min_aspect=0.05,
            flat_v_prior=True,
            axis=axis,
        )
    )


_POSITIONS = jnp.array([[0.3, 0.2, 0.4], [1.1, 0.7, 1.0]])
# Triclinic: in-plane pair not orthogonal, c tilted so it has an in-plane part.
_TRICLINIC = jnp.array([[2.0, 0.0, 0.0], [-1.0, 1.8, 0.0], [0.3, 0.2, 2.2]])


def _normal(cell, axis):
    o1, o2 = [i for i in range(3) if i != axis]
    n = jnp.cross(cell[o1], cell[o2])
    return n / jnp.linalg.norm(n)


def _check_only_axis_changes(state, new_state, axis):
    ratio = get_volume(new_state.cell) / get_volume(state.cell)
    assert not jnp.allclose(ratio, 1.0)
    for i in range(3):
        if i != axis:
            np.testing.assert_array_equal(new_state.cell[i], state.cell[i])
    n = _normal(state.cell, axis)
    delta = new_state.cell[axis] - state.cell[axis]
    # The change is purely along the plane normal ...
    np.testing.assert_allclose(delta, (delta @ n) * n, atol=1e-5, rtol=1e-5)
    # ... it scales the normal component by the volume ratio, and the
    # in-plane part of the vector is untouched.
    np.testing.assert_allclose(
        new_state.cell[axis] @ n, ratio * (state.cell[axis] @ n), rtol=1e-5
    )
    np.testing.assert_allclose(
        new_state.cell[axis] - (new_state.cell[axis] @ n) * n,
        state.cell[axis] - (state.cell[axis] @ n) * n,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        _frac(new_state.positions, new_state.cell),
        _frac(state.positions, state.cell),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "cell", [jnp.diag(jnp.array([2.0, 2.0, 2.0])), _TRICLINIC]
)
def test_default_axis_c_changes_only_c(cell):
    state = _state(_POSITIONS, cell)
    new_state, info = _kernel()(jax.random.key(0), state, 1e9)
    assert bool(info.accepted)
    _check_only_axis_changes(state, new_state, axis=2)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_axis_changes_only_that_vector(axis):
    state = _state(_POSITIONS, _TRICLINIC)
    new_state, info = _kernel(axis=axis)(jax.random.key(3), state, 1e9)
    assert bool(info.accepted)
    _check_only_axis_changes(state, new_state, axis=axis)


def test_normal_follows_the_current_cell():
    """The plane normal is a x b of the current cell, not a fixed z.

    The in-plane pair is tilted out of the xy plane, so a x b is not along z;
    still only c changes.
    """
    cell = jnp.array([[2.0, 0.0, 0.3], [0.0, 2.0, -0.2], [0.4, 0.1, 2.5]])
    state = _state(_POSITIONS, cell)
    new_state, info = _kernel()(jax.random.key(5), state, 1e9)
    assert bool(info.accepted)
    _check_only_axis_changes(state, new_state, axis=2)


def test_bad_axis_rejected():
    with pytest.raises(ValueError, match="axis"):
        build_kernel(_ZeroEnergyBackend(), N_ATOMS, axis=3)


def test_jit_vmap_scan():
    cell = jnp.diag(jnp.array([2.0, 2.0, 2.0]))
    state = _state(jnp.array([[0.3, 0.2, 0.4], [1.1, 0.7, 1.0]]), cell)
    step = jax.jit(build_kernel(_ZeroEnergyBackend(), N_ATOMS))

    def body(st, key):
        ns, info = step(key, st, 1e9)
        return ns, info.accepted

    final, _ = jax.lax.scan(
        body, state, jax.random.split(jax.random.key(1), 5)
    )
    assert final.cell.shape == (3, 3)


def _sample_volumes(flat_v_prior, n_walkers=2000, n_steps=400):
    """Chain on a constant-energy surface; only cell bounds and prior act."""
    # a = b = 2 so V = 4c; V in [6, 12] <=> c in [1.5, 3].
    cell0 = jnp.diag(jnp.array([2.0, 2.0, 1.7]))
    positions = jnp.array([[0.3, 0.2, 0.4], [1.1, 0.7, 0.9]])
    step = build_kernel(
        _ZeroEnergyBackend(),
        N_ATOMS,
        max_vol_per_atom=6.0,
        min_vol_per_atom=3.0,
        min_aspect=0.3,
        flat_v_prior=flat_v_prior,
    )

    def run(key):
        def body(st, k):
            ns, _ = step(k, st, 1e9)
            return ns, None

        final, _ = jax.lax.scan(
            body, _state(positions, cell0), jax.random.split(key, n_steps)
        )
        return get_volume(final.cell)

    keys = jax.random.split(jax.random.key(7), n_walkers)
    return np.asarray(jax.jit(jax.vmap(run))(keys))


def _mean_density(power, lo, hi):
    """Mean of V under a density proportional to V**power on [lo, hi]."""
    num = (hi ** (power + 2) - lo ** (power + 2)) / (power + 2)
    den = (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)
    return num / den


@pytest.mark.parametrize("flat_v_prior, power", [(True, 0), (False, N_ATOMS)])
def test_stationary_volume_distribution(flat_v_prior, power):
    """Constant energy: V is distributed as the prior, within the bounds.

    flat_V_prior -> uniform in V; default -> density ~ V**N (the Metropolis
    factor min(1, (V'/V)**N) of ``volume``).  The cell-shape prior is not
    tested here (c is the only free axis).
    """
    volumes = _sample_volumes(flat_v_prior)
    assert volumes.min() >= 6.0 - 1e-4 and volumes.max() <= 12.0 + 1e-4
    assert abs(volumes.mean() - _mean_density(power, 6.0, 12.0)) < 0.15


def test_spec_descriptor():
    cell_cfg = CellSpec(
        max_volume_per_atom=50.0,
        min_volume_per_atom=1.0,
        min_aspect_ratio=0.5,
        flat_V_prior=True,
    )
    desc = VolumeAxisMoveSpec().to_descriptor(n_atoms=10, cell_cfg=cell_cfg)
    assert desc.name == "volume_axis"
    assert desc.kernel_kwargs["n_atoms"] == 10
    assert desc.kernel_kwargs["flat_v_prior"] is True
    assert desc.kernel_kwargs["axis"] == 2
    b_desc = VolumeAxisMoveSpec(axis="a").to_descriptor(
        n_atoms=10, cell_cfg=cell_cfg
    )
    assert b_desc.kernel_kwargs["axis"] == 0
    assert desc.reject_reasons == frozenset({"energy", "cell", "prior"})
    assert desc.mutates == frozenset({"positions", "cell"})
    with pytest.raises(ValueError, match="n_atoms"):
        VolumeAxisMoveSpec().to_descriptor()
    with pytest.raises(ValueError):
        VolumeAxisMoveSpec(axis="d")
