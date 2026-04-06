"""Test cell move kernels: volume, shear, and stretch.

For each move: (a) init returns valid state, (b) step produces expected output,
(c) JIT compilation succeeds, (d) vmap over walkers works, (e) lax.scan
compatible, (f) cell shape constraints enforced.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.moves.volume import (
    VolumeMoveState,
    init as vol_init,
    build_kernel as vol_build_kernel,
    as_top_level_api as vol_api,
)
from jaxrens.sampling.moves.shear import (
    ShearMoveState,
    init as shear_init,
    build_kernel as shear_build_kernel,
    as_top_level_api as shear_api,
)
from jaxrens.sampling.moves.stretch import (
    StretchMoveState,
    init as stretch_init,
    build_kernel as stretch_build_kernel,
    as_top_level_api as stretch_api,
)
from jaxrens.utils.cell import (
    get_volume,
    min_aspect_ratio,
    check_cell_shape,
    transform_positions,
)
from jaxrens.backends.lj import create_lj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def positions():
    return jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])


@pytest.fixture
def types():
    return jnp.array([0, 0])


@pytest.fixture
def box():
    return 5.0 * jnp.eye(3)


def cell_energy_fn(params, positions, types, box=None):
    """Simple harmonic energy that depends on positions only."""
    return jnp.sum(positions**2)


N_ATOMS = 2


# ---------------------------------------------------------------------------
# Cell Utility Tests
# ---------------------------------------------------------------------------


class TestCellUtils:
    def test_get_volume_cubic(self, box):
        vol = get_volume(box)
        assert jnp.allclose(vol, 125.0)

    def test_min_aspect_ratio_cubic(self, box):
        vol = get_volume(box)
        ratio = min_aspect_ratio(box, vol)
        assert jnp.allclose(ratio, 1.0, atol=1e-5)

    def test_check_cell_shape_valid(self, box):
        valid = check_cell_shape(box, N_ATOMS, max_vol_per_atom=100.0,
                                 min_vol_per_atom=1.0, min_aspect=0.5)
        assert valid

    def test_check_cell_shape_too_large(self, box):
        # 125 / 2 = 62.5 > 10.0
        valid = check_cell_shape(box, N_ATOMS, max_vol_per_atom=10.0,
                                 min_vol_per_atom=1.0, min_aspect=0.5)
        assert not valid

    def test_check_cell_shape_too_small(self, box):
        # 125 / 2 = 62.5 < 100.0
        valid = check_cell_shape(box, N_ATOMS, max_vol_per_atom=200.0,
                                 min_vol_per_atom=100.0, min_aspect=0.5)
        assert not valid

    def test_check_cell_shape_bad_aspect(self):
        # Degenerate cell with very poor aspect ratio
        bad_cell = jnp.array([[10.0, 0.0, 0.0],
                              [0.0, 0.01, 0.0],
                              [0.0, 0.0, 10.0]])
        valid = check_cell_shape(bad_cell, 1, max_vol_per_atom=10000.0,
                                 min_vol_per_atom=0.001, min_aspect=0.5)
        assert not valid

    def test_transform_positions_identity(self, positions, box):
        new_pos = transform_positions(positions, box, box)
        assert jnp.allclose(new_pos, positions, atol=1e-6)

    def test_transform_positions_scaling(self, positions, box):
        new_box = 2.0 * box
        new_pos = transform_positions(positions, box, new_box)
        assert jnp.allclose(new_pos, 2.0 * positions, atol=1e-6)


# ---------------------------------------------------------------------------
# Volume Move Tests
# ---------------------------------------------------------------------------


class TestVolumeMoveInit:
    def test_returns_named_tuple(self, positions, types, box):
        state = vol_init(positions, types, energy=-1.0, box=box, step_size=0.1)
        assert isinstance(state, VolumeMoveState)
        assert jnp.array_equal(state.positions, positions)
        assert jnp.allclose(state.energy, -1.0)
        assert jnp.allclose(state.step_size, 0.1)
        assert state.box is not None


class TestVolumeMoveStep:
    def test_step_returns_state_and_info(self, positions, types, box):
        state = vol_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = vol_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=100.0)

        assert isinstance(new_state, VolumeMoveState)
        assert hasattr(info, "accepted")
        assert hasattr(info, "n_evaluations")

    def test_accepts_below_constraint(self, positions, types, box):
        state = vol_init(positions, types, energy=0.5, box=box, step_size=0.01)
        step = vol_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        assert info.accepted

    def test_rejects_above_constraint(self, positions, types, box):
        state = vol_init(positions, types, energy=0.5, box=box, step_size=0.01)
        step = vol_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(0)
        # Very tight constraint
        new_state, info = step(key, state, likelihood_constraint=0.0)
        assert not info.accepted
        assert jnp.array_equal(new_state.positions, state.positions)

    def test_volume_changes_on_accept(self, positions, types, box):
        state = vol_init(positions, types, energy=0.5, box=box, step_size=0.5)
        step = vol_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        if info.accepted:
            old_vol = get_volume(state.box)
            new_vol = get_volume(new_state.box)
            assert not jnp.allclose(old_vol, new_vol, atol=1e-8)

    def test_jit(self, positions, types, box):
        state = vol_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = vol_build_kernel(cell_energy_fn, {}, N_ATOMS)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 100.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, positions, types, box):
        step = vol_build_kernel(cell_energy_fn, {}, N_ATOMS)

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.5, 1.0, 1.5, 2.0])
        batch_step_size = jnp.full(4, 0.1)
        batch_box = jnp.stack([box] * 4)

        batch_state = VolumeMoveState(
            positions=batch_pos,
            types=batch_types,
            energy=batch_energy,
            box=batch_box,
            step_size=batch_step_size,
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 100.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_scan_compatible(self, positions, types, box):
        state = vol_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = vol_build_kernel(cell_energy_fn, {}, N_ATOMS)

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_cell_shape_constraint_enforced(self, positions, types, box):
        # min_aspect=0.99 is very strict; most volume changes on cubic box
        # should still pass since isotropic scaling preserves aspect ratio.
        # Use extreme volume bounds instead to force rejection.
        state = vol_init(positions, types, energy=0.5, box=box, step_size=10.0)
        step = vol_build_kernel(
            cell_energy_fn, {}, N_ATOMS,
            max_vol_per_atom=63.0,  # current is 62.5, tight bound
            min_vol_per_atom=62.0,
            min_aspect=0.99,
        )

        n_rejected = 0
        key = jax.random.key(0)
        for i in range(20):
            key, subkey = jax.random.split(key)
            new_state, info = step(subkey, state, likelihood_constraint=1e6)
            if not info.accepted:
                n_rejected += 1
        # With such tight constraints and large step_size, most should reject
        assert n_rejected > 5

    def test_top_level_api(self, positions, types, box):
        kernel = vol_api(cell_energy_fn, {}, N_ATOMS, step_size=0.1)
        state = kernel.init(positions, types, 0.5, box)
        new_state, info = kernel.step(jax.random.key(0), state, 100.0)
        assert isinstance(new_state, VolumeMoveState)


# ---------------------------------------------------------------------------
# Shear Move Tests
# ---------------------------------------------------------------------------


class TestShearMoveInit:
    def test_returns_named_tuple(self, positions, types, box):
        state = shear_init(positions, types, energy=-1.0, box=box, step_size=0.1)
        assert isinstance(state, ShearMoveState)
        assert jnp.array_equal(state.positions, positions)
        assert jnp.allclose(state.energy, -1.0)
        assert jnp.allclose(state.step_size, 0.1)


class TestShearMoveStep:
    def test_step_returns_state_and_info(self, positions, types, box):
        state = shear_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = shear_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=100.0)

        assert isinstance(new_state, ShearMoveState)
        assert hasattr(info, "accepted")
        assert hasattr(info, "n_evaluations")

    def test_accepts_below_constraint(self, positions, types, box):
        state = shear_init(positions, types, energy=0.5, box=box, step_size=0.01)
        step = shear_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        assert info.accepted

    def test_rejects_above_constraint(self, positions, types, box):
        state = shear_init(positions, types, energy=0.5, box=box, step_size=0.01)
        step = shear_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.0)
        assert not info.accepted
        assert jnp.array_equal(new_state.positions, state.positions)

    def test_volume_preserved(self, positions, types, box):
        """Volume before and after shear should be equal."""
        state = shear_init(positions, types, energy=0.5, box=box, step_size=0.5)
        step = shear_build_kernel(cell_energy_fn, {}, N_ATOMS)

        old_vol = get_volume(state.box)
        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        new_vol = get_volume(new_state.box)
        assert jnp.allclose(old_vol, new_vol, atol=1e-4)

    def test_jit(self, positions, types, box):
        state = shear_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = shear_build_kernel(cell_energy_fn, {}, N_ATOMS)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 100.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, positions, types, box):
        step = shear_build_kernel(cell_energy_fn, {}, N_ATOMS)

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.5, 1.0, 1.5, 2.0])
        batch_step_size = jnp.full(4, 0.1)
        batch_box = jnp.stack([box] * 4)

        batch_state = ShearMoveState(
            positions=batch_pos,
            types=batch_types,
            energy=batch_energy,
            box=batch_box,
            step_size=batch_step_size,
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 100.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_scan_compatible(self, positions, types, box):
        state = shear_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = shear_build_kernel(cell_energy_fn, {}, N_ATOMS)

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_cell_shape_constraint_enforced(self, positions, types, box):
        state = shear_init(positions, types, energy=0.5, box=box, step_size=10.0)
        step = shear_build_kernel(
            cell_energy_fn, {}, N_ATOMS, min_aspect=0.99,
        )

        n_rejected = 0
        key = jax.random.key(0)
        for i in range(20):
            key, subkey = jax.random.split(key)
            new_state, info = step(subkey, state, likelihood_constraint=1e6)
            if not info.accepted:
                n_rejected += 1
        # Large shear + strict aspect ratio -> many rejections
        assert n_rejected > 5

    def test_top_level_api(self, positions, types, box):
        kernel = shear_api(cell_energy_fn, {}, N_ATOMS, step_size=0.1)
        state = kernel.init(positions, types, 0.5, box)
        new_state, info = kernel.step(jax.random.key(0), state, 100.0)
        assert isinstance(new_state, ShearMoveState)


# ---------------------------------------------------------------------------
# Stretch Move Tests
# ---------------------------------------------------------------------------


class TestStretchMoveInit:
    def test_returns_named_tuple(self, positions, types, box):
        state = stretch_init(positions, types, energy=-1.0, box=box, step_size=0.1)
        assert isinstance(state, StretchMoveState)
        assert jnp.array_equal(state.positions, positions)
        assert jnp.allclose(state.energy, -1.0)
        assert jnp.allclose(state.step_size, 0.1)


class TestStretchMoveStep:
    def test_step_returns_state_and_info(self, positions, types, box):
        state = stretch_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = stretch_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=100.0)

        assert isinstance(new_state, StretchMoveState)
        assert hasattr(info, "accepted")
        assert hasattr(info, "n_evaluations")

    def test_accepts_below_constraint(self, positions, types, box):
        state = stretch_init(positions, types, energy=0.5, box=box, step_size=0.01)
        step = stretch_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        assert info.accepted

    def test_rejects_above_constraint(self, positions, types, box):
        state = stretch_init(positions, types, energy=0.5, box=box, step_size=0.01)
        step = stretch_build_kernel(cell_energy_fn, {}, N_ATOMS)

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.0)
        assert not info.accepted
        assert jnp.array_equal(new_state.positions, state.positions)

    def test_volume_preserved(self, positions, types, box):
        """Volume before and after stretch should be equal."""
        state = stretch_init(positions, types, energy=0.5, box=box, step_size=0.5)
        step = stretch_build_kernel(cell_energy_fn, {}, N_ATOMS)

        old_vol = get_volume(state.box)
        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        new_vol = get_volume(new_state.box)
        assert jnp.allclose(old_vol, new_vol, atol=1e-4)

    def test_jit(self, positions, types, box):
        state = stretch_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = stretch_build_kernel(cell_energy_fn, {}, N_ATOMS)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 100.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, positions, types, box):
        step = stretch_build_kernel(cell_energy_fn, {}, N_ATOMS)

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.5, 1.0, 1.5, 2.0])
        batch_step_size = jnp.full(4, 0.1)
        batch_box = jnp.stack([box] * 4)

        batch_state = StretchMoveState(
            positions=batch_pos,
            types=batch_types,
            energy=batch_energy,
            box=batch_box,
            step_size=batch_step_size,
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 100.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_scan_compatible(self, positions, types, box):
        state = stretch_init(positions, types, energy=0.5, box=box, step_size=0.1)
        step = stretch_build_kernel(cell_energy_fn, {}, N_ATOMS)

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_cell_shape_constraint_enforced(self, positions, types, box):
        state = stretch_init(positions, types, energy=0.5, box=box, step_size=10.0)
        step = stretch_build_kernel(
            cell_energy_fn, {}, N_ATOMS, min_aspect=0.99,
        )

        n_rejected = 0
        key = jax.random.key(0)
        for i in range(20):
            key, subkey = jax.random.split(key)
            new_state, info = step(subkey, state, likelihood_constraint=1e6)
            if not info.accepted:
                n_rejected += 1
        # Large stretch + strict aspect ratio -> many rejections
        assert n_rejected > 5

    def test_top_level_api(self, positions, types, box):
        kernel = stretch_api(cell_energy_fn, {}, N_ATOMS, step_size=0.1)
        state = kernel.init(positions, types, 0.5, box)
        new_state, info = kernel.step(jax.random.key(0), state, 100.0)
        assert isinstance(new_state, StretchMoveState)


# ---------------------------------------------------------------------------
# Integration Test: Volume Move on LJ Cluster with PBC
# ---------------------------------------------------------------------------


class TestLJIntegration:
    def test_volume_move_on_lj_cluster(self):
        """Run 200 volume move steps on a 4-atom LJ cluster with PBC."""
        energy_fn, params = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        n_atoms = 4

        box = 5.0 * jnp.eye(3)
        positions = jnp.array([
            [0.5, 0.5, 0.5],
            [2.0, 0.5, 0.5],
            [0.5, 2.0, 0.5],
            [0.5, 0.5, 2.0],
        ])
        types = jnp.zeros(n_atoms, dtype=jnp.int32)

        init_energy = energy_fn(params, positions, types, box=box)
        state = vol_init(positions, types, energy=init_energy, box=box, step_size=0.5)
        step = vol_build_kernel(
            energy_fn, params, n_atoms,
            max_vol_per_atom=100.0,
            min_vol_per_atom=1.0,
            min_aspect=0.5,
        )

        n_steps = 200
        n_accepted = 0
        volumes = []
        key = jax.random.key(123)

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=1e6)
            if info.accepted:
                n_accepted += 1
            volumes.append(float(get_volume(state.box)))

        volumes = jnp.array(volumes)

        # Some moves should have been accepted
        assert n_accepted > 10, f"Only {n_accepted} accepted out of {n_steps}"

        # Accepted moves should produce different volumes
        unique_vols = jnp.unique(jnp.round(volumes, decimals=2))
        assert len(unique_vols) > 1, "Expected different volumes from accepted moves"

        # Cell constraints should be respected throughout
        assert jnp.all(volumes > 0.0), "All volumes should be positive"
        assert jnp.all(volumes / n_atoms <= 100.0), "Volume per atom constraint"
        assert jnp.all(volumes / n_atoms >= 1.0), "Min volume per atom constraint"
