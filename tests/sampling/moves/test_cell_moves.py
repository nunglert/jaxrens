"""Tests for the volume, shear and stretch cell move kernels."""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.backends.lj import create_lj
from jaxrens.sampling.moves.shear import build_kernel as shear_build_kernel
from jaxrens.sampling.moves.stretch import build_kernel as stretch_build_kernel
from jaxrens.sampling.moves.volume import build_kernel as vol_build_kernel
from jaxrens.state.mc_state import MCState
from jaxrens.utils.cell import (
    check_cell_shape,
    get_volume,
    min_aspect_ratio,
    transform_positions,
)


def _make_cell_state(positions, types, energy, cell, step_size=0.1):
    """Build an MCState for one walker with the given cell."""
    return MCState(
        positions=jnp.asarray(positions),
        types=jnp.asarray(types),
        energy=jnp.asarray(energy),
        cell=jnp.asarray(cell),
        step_size=jnp.asarray(step_size),
        step_sizes=jnp.array([step_size]),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
    )


@pytest.fixture
def positions():
    return jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])


@pytest.fixture
def types():
    return jnp.array([0, 0])


@pytest.fixture
def cell():
    return 5.0 * jnp.eye(3)


class _CellEnergyBackend:
    """Minimal backend for cell-move tests: E = sum(positions^2)."""

    r_cutoff = 0.0

    def __call__(
        self, positions, species, cell, max_neighbors=0, ensemble_params=None
    ):
        energy = jnp.sum(positions**2)
        return BackendResult(energy=energy)


cell_energy_backend = _CellEnergyBackend()


N_ATOMS = 2


class TestCellUtils:
    """Check the cell helpers the moves rely on.

    The tests cover volume, aspect ratio, the shape check and the position
    transform.
    """

    def test_get_volume_cubic(self, cell):
        vol = get_volume(cell)
        assert jnp.allclose(vol, 125.0)

    def test_min_aspect_ratio_cubic(self, cell):
        vol = get_volume(cell)
        ratio = min_aspect_ratio(cell, vol)
        assert jnp.allclose(ratio, 1.0, atol=1e-5)

    def test_check_cell_shape_valid(self, cell):
        valid = check_cell_shape(
            cell,
            N_ATOMS,
            max_vol_per_atom=100.0,
            min_vol_per_atom=1.0,
            min_aspect=0.5,
        )
        assert valid

    def test_check_cell_shape_too_large(self, cell):
        valid = check_cell_shape(
            cell,
            N_ATOMS,
            max_vol_per_atom=10.0,
            min_vol_per_atom=1.0,
            min_aspect=0.5,
        )
        assert not valid

    def test_check_cell_shape_too_small(self, cell):
        valid = check_cell_shape(
            cell,
            N_ATOMS,
            max_vol_per_atom=200.0,
            min_vol_per_atom=100.0,
            min_aspect=0.5,
        )
        assert not valid

    def test_check_cell_shape_bad_aspect(self):
        bad_cell = jnp.array(
            [[10.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 10.0]]
        )
        valid = check_cell_shape(
            bad_cell,
            1,
            max_vol_per_atom=10000.0,
            min_vol_per_atom=0.001,
            min_aspect=0.5,
        )
        assert not valid

    def test_transform_positions_identity(self, positions, cell):
        new_pos = transform_positions(positions, cell, cell)
        assert jnp.allclose(new_pos, positions, atol=1e-6)

    def test_transform_positions_scaling(self, positions, cell):
        new_cell = 2.0 * cell
        new_pos = transform_positions(positions, cell, new_cell)
        assert jnp.allclose(new_pos, 2.0 * positions, atol=1e-6)


class _CellMoveChecks:
    """Checks that every cell move must pass.

    Subclasses set `build` to the kernel builder and `shape_kwargs` to cell
    bounds that make most proposals invalid.
    """

    build = None
    shape_kwargs = {}

    def test_step_basic(self, positions, types, cell):
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=0.1
        )
        step = jax.jit(self.build(cell_energy_backend, N_ATOMS))

        new_state, info = step(
            jax.random.key(0), state, likelihood_constraint=100.0
        )

        assert new_state.positions.shape == positions.shape
        assert new_state.cell.shape == (3, 3)
        assert new_state.energy.shape == ()
        assert info.n_evaluations == 1

    def test_accepts_below_constraint(self, positions, types, cell):
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=0.01
        )
        step = jax.jit(self.build(cell_energy_backend, N_ATOMS))

        new_state, info = step(
            jax.random.key(42), state, likelihood_constraint=1000.0
        )
        assert info.accepted

    def test_rejects_above_constraint(self, positions, types, cell):
        """A rejected move leaves positions, cell and energy unchanged."""
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=0.01
        )
        step = jax.jit(self.build(cell_energy_backend, N_ATOMS))

        new_state, info = step(
            jax.random.key(0), state, likelihood_constraint=0.0
        )

        assert not info.accepted
        assert jnp.array_equal(new_state.positions, state.positions)
        assert jnp.array_equal(new_state.cell, state.cell)
        assert jnp.array_equal(new_state.energy, state.energy)

    def test_vmap(self, positions, types, cell):
        step = jax.jit(self.build(cell_energy_backend, N_ATOMS))

        batch_state = MCState(
            positions=jnp.stack([positions] * 4),
            types=jnp.stack([types] * 4),
            energy=jnp.array([0.5, 1.0, 1.5, 2.0]),
            cell=jnp.stack([cell] * 4),
            step_size=jnp.full(4, 0.1),
            step_sizes=jnp.full((4, 1), 0.1),
            n_accepted=jnp.zeros((4, 1), dtype=jnp.int32),
            n_proposed=jnp.zeros((4, 1), dtype=jnp.int32),
            max_neighbor_count=jnp.zeros(4, dtype=jnp.int32),
            overflow=jnp.full(4, False),
            ensemble_params={},
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 100.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_lax_scan(self, positions, types, cell):
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=0.1
        )
        step = jax.jit(self.build(cell_energy_backend, N_ATOMS))

        def scan_step(carry, key):
            new_state, info = step(key, carry, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_cell_shape_constraint_enforced(self, positions, types, cell):
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=10.0
        )
        step = jax.jit(
            self.build(cell_energy_backend, N_ATOMS, **self.shape_kwargs)
        )

        n_rejected = 0
        key = jax.random.key(0)
        for i in range(20):
            key, subkey = jax.random.split(key)
            new_state, info = step(subkey, state, likelihood_constraint=1e6)
            if not info.accepted:
                n_rejected += 1
        assert n_rejected > 5


class _VolumePreservingChecks(_CellMoveChecks):
    """Adds the volume check shared by the shear and stretch moves."""

    def test_volume_preserved(self, positions, types, cell):
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=0.5
        )
        step = jax.jit(self.build(cell_energy_backend, N_ATOMS))

        old_vol = get_volume(state.cell)
        new_state, info = step(
            jax.random.key(42), state, likelihood_constraint=1000.0
        )
        new_vol = get_volume(new_state.cell)
        assert jnp.allclose(old_vol, new_vol, atol=1e-4)


class TestVolumeMoveStep(_CellMoveChecks):
    """Check the volume move on a quadratic-energy backend.

    Besides the shared checks, the tests cover the volume change.
    """

    build = staticmethod(vol_build_kernel)
    shape_kwargs = {
        "max_vol_per_atom": 63.0,
        "min_vol_per_atom": 62.0,
        "min_aspect": 0.99,
    }

    def test_volume_changes_on_accept(self, positions, types, cell):
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=0.5
        )
        step = jax.jit(self.build(cell_energy_backend, N_ATOMS))

        new_state, info = step(
            jax.random.key(42), state, likelihood_constraint=1000.0
        )

        assert info.accepted
        old_vol = get_volume(state.cell)
        new_vol = get_volume(new_state.cell)
        assert not jnp.allclose(old_vol, new_vol, atol=1e-8)


class TestShearMoveStep(_VolumePreservingChecks):
    """Check the shear move on a quadratic-energy backend.

    Besides the shared checks, the tests cover volume preservation.
    """

    build = staticmethod(shear_build_kernel)
    shape_kwargs = {"min_aspect": 0.99}


class TestStretchMoveStep(_VolumePreservingChecks):
    """Check the stretch move on a quadratic-energy backend.

    Besides the shared checks, the tests cover volume preservation.
    """

    build = staticmethod(stretch_build_kernel)
    shape_kwargs = {"min_aspect": 0.99}


class TestLJIntegration:
    """Check the volume move on a Lennard-Jones cluster with PBC."""

    def test_volume_move_on_lj_cluster(self):
        """200 volume steps keep the cell volume within the bounds."""
        backend = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        n_atoms = 4

        cell = 5.0 * jnp.eye(3)
        positions = jnp.array(
            [
                [0.5, 0.5, 0.5],
                [2.0, 0.5, 0.5],
                [0.5, 2.0, 0.5],
                [0.5, 0.5, 2.0],
            ]
        )
        types = jnp.zeros(n_atoms, dtype=jnp.int32)

        init_energy = backend(positions, types, cell, 0)[0]
        state = _make_cell_state(
            positions, types, energy=init_energy, cell=cell, step_size=0.5
        )
        step = jax.jit(
            vol_build_kernel(
                backend,
                n_atoms,
                max_vol_per_atom=100.0,
                min_vol_per_atom=1.0,
                min_aspect=0.5,
            )
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
            volumes.append(float(get_volume(state.cell)))

        volumes = jnp.array(volumes)

        assert n_accepted > 10
        unique_vols = jnp.unique(jnp.round(volumes, decimals=2))
        assert len(unique_vols) > 1
        assert jnp.all(volumes > 0.0)
        assert jnp.all(volumes / n_atoms <= 100.0)
        assert jnp.all(volumes / n_atoms >= 1.0)


class TestTF32Precision:
    """Check that position transforms keep full matmul precision.

    On CUDA, JAX defaults to TF32 (10-bit mantissa).  Even for an identity
    transform, `positions @ eye(3)` then adds about 3.7e-3 of noise per
    element, enough to spike the LJ energy at dense packing and reject every
    volume move.  transform_positions uses jnp.einsum with Precision.HIGHEST
    to avoid this.  The tests pass on CPU and fail on GPU if TF32 leaks in.
    """

    def test_transform_positions_identity_is_bit_exact(self):
        """An identity transform returns the positions bit-exactly."""
        # Dense-packing scenario: 64 atoms in a cell of side 11.
        key = jax.random.key(0)
        n_atoms = 64
        cell_side = 11.0
        cell = cell_side * jnp.eye(3)
        positions = jax.random.uniform(
            key, (n_atoms, 3), minval=0.0, maxval=cell_side
        )

        new_positions = transform_positions(
            positions, cell, cell
        )  # T = identity

        max_err = float(jnp.max(jnp.abs(new_positions - positions)))
        assert max_err == 0.0

    def test_transform_positions_identity_bit_exact_under_jit(self):
        """The identity transform is also bit-exact under jax.jit."""
        key = jax.random.key(1)
        n_atoms = 64
        cell_side = 11.0
        cell = cell_side * jnp.eye(3)
        positions = jax.random.uniform(
            key, (n_atoms, 3), minval=0.0, maxval=cell_side
        )

        jit_transform = jax.jit(transform_positions)
        new_positions = jit_transform(positions, cell, cell)

        max_err = float(jnp.max(jnp.abs(new_positions - positions)))
        assert max_err == 0.0

    def test_volume_move_tiny_step_size_preserves_energy(self):
        """A volume move with step size 1e-20 leaves the energy unchanged.

        The proposed cell is indistinguishable from the old one, so the
        energy must agree to within 1e-10.  With TF32 the position matmul
        would add noise of about 3.7e-3 and spike the energy.
        """
        backend = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        n_atoms = 4

        cell = 5.0 * jnp.eye(3)
        positions = jnp.array(
            [
                [0.5, 0.5, 0.5],
                [2.0, 0.5, 0.5],
                [0.5, 2.0, 0.5],
                [0.5, 0.5, 2.0],
            ]
        )
        types = jnp.zeros(n_atoms, dtype=jnp.int32)

        init_energy = backend(positions, types, cell, 0)[0]
        state = _make_cell_state(
            positions, types, energy=init_energy, cell=cell, step_size=1e-20
        )

        step = jax.jit(
            vol_build_kernel(
                backend,
                n_atoms,
                max_vol_per_atom=100.0,
                min_vol_per_atom=1.0,
                min_aspect=0.5,
            )
        )

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1e10)

        energy_diff = float(jnp.abs(new_state.energy - init_energy))
        assert energy_diff < 1e-10


class _OverflowingBackend:
    """Backend that always reports overflow and a large neighbor count."""

    r_cutoff = 0.0

    def __init__(self, count: int = 999):
        self._count = count

    def __call__(
        self, positions, species, cell, max_neighbors=0, ensemble_params=None
    ):
        energy = jnp.sum(positions**2)
        return BackendResult(
            energy=energy,
            max_neighbor_count=jnp.int32(self._count),
            overflow=jnp.bool_(True),
        )


_OVERFLOWING_BACKEND = _OverflowingBackend(count=999)


_MOVE_BUILDERS = {
    "volume": vol_build_kernel,
    "shear": shear_build_kernel,
    "stretch": stretch_build_kernel,
}


class TestCellInvalidOverflowGated:
    """Check that overflow signals from cell-invalid proposals are dropped.

    `state.overflow` and `state.max_neighbor_count` must not be set by
    proposals that violate the cell-shape constraint.  Otherwise the outer
    loop would enlarge the neighbor bucket for configurations the chain
    never samples.  Signals from cell-valid proposals must still pass.
    """

    @pytest.mark.parametrize("move", ["volume", "shear", "stretch"])
    def test_cell_invalid_does_not_set_overflow_or_count(
        self,
        move,
        positions,
        types,
        cell,
    ):
        # Big step and tight cell bounds: every proposal is cell-invalid.
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=10.0
        )
        step = jax.jit(
            _MOVE_BUILDERS[move](
                _OVERFLOWING_BACKEND,
                N_ATOMS,
                max_vol_per_atom=63.0,
                min_vol_per_atom=62.0,
                min_aspect=0.999,
            )
        )

        for i in range(20):
            key = jax.random.key(i)
            state, info = step(key, state, likelihood_constraint=1e10)
            assert not bool(info.accepted)

        assert bool(state.overflow) is False
        assert int(state.max_neighbor_count) == 0

    @pytest.mark.parametrize("move", ["volume", "shear", "stretch"])
    def test_cell_valid_still_propagates_overflow(
        self,
        move,
        positions,
        types,
        cell,
    ):
        # Loose bounds and a tiny step: the proposal is cell-valid.
        state = _make_cell_state(
            positions, types, energy=0.5, cell=cell, step_size=1e-4
        )
        step = jax.jit(
            _MOVE_BUILDERS[move](
                _OVERFLOWING_BACKEND,
                N_ATOMS,
                max_vol_per_atom=1000.0,
                min_vol_per_atom=1.0,
                min_aspect=0.5,
            )
        )
        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=1e10)

        assert bool(new_state.overflow) is True
        assert int(new_state.max_neighbor_count) == 999
