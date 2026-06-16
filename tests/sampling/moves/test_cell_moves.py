"""Test cell move kernels: volume, shear, and stretch.

For each move: (a) step produces expected output, (b) JIT compilation
succeeds, (c) vmap over walkers works, (d) lax.scan compatible,
(e) cell shape constraints enforced.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.moves.volume import build_kernel as vol_build_kernel
from jaxrens.sampling.moves.shear import build_kernel as shear_build_kernel
from jaxrens.sampling.moves.stretch import build_kernel as stretch_build_kernel
from jaxrens.state.mc_state import MCState
from jaxrens.utils.cell import (
    get_volume,
    min_aspect_ratio,
    check_cell_shape,
    transform_positions,
)
from jaxrens.backends.base import BackendResult
from jaxrens.backends.lj import create_lj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cell_state(positions, types, energy, cell, step_size=0.1):
    """Helper: create MCState for cell move tests."""
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
def cell():
    return 5.0 * jnp.eye(3)


class _CellEnergyBackend:
    """Minimal backend for cell-move tests: E = sum(positions^2)."""

    r_cutoff = 0.0

    def __call__(self, positions, species, cell, max_neighbors=0,
                 ensemble_params=None):
        energy = jnp.sum(positions**2)
        return BackendResult(energy=energy)


cell_energy_backend = _CellEnergyBackend()


N_ATOMS = 2


# ---------------------------------------------------------------------------
# Cell Utility Tests
# ---------------------------------------------------------------------------


class TestCellUtils:
    def test_get_volume_cubic(self, cell):
        vol = get_volume(cell)
        assert jnp.allclose(vol, 125.0)

    def test_min_aspect_ratio_cubic(self, cell):
        vol = get_volume(cell)
        ratio = min_aspect_ratio(cell, vol)
        assert jnp.allclose(ratio, 1.0, atol=1e-5)

    def test_check_cell_shape_valid(self, cell):
        valid = check_cell_shape(cell, N_ATOMS, max_vol_per_atom=100.0,
                                 min_vol_per_atom=1.0, min_aspect=0.5)
        assert valid

    def test_check_cell_shape_too_large(self, cell):
        valid = check_cell_shape(cell, N_ATOMS, max_vol_per_atom=10.0,
                                 min_vol_per_atom=1.0, min_aspect=0.5)
        assert not valid

    def test_check_cell_shape_too_small(self, cell):
        valid = check_cell_shape(cell, N_ATOMS, max_vol_per_atom=200.0,
                                 min_vol_per_atom=100.0, min_aspect=0.5)
        assert not valid

    def test_check_cell_shape_bad_aspect(self):
        bad_cell = jnp.array([[10.0, 0.0, 0.0],
                              [0.0, 0.01, 0.0],
                              [0.0, 0.0, 10.0]])
        valid = check_cell_shape(bad_cell, 1, max_vol_per_atom=10000.0,
                                 min_vol_per_atom=0.001, min_aspect=0.5)
        assert not valid

    def test_transform_positions_identity(self, positions, cell):
        new_pos = transform_positions(positions, cell, cell)
        assert jnp.allclose(new_pos, positions, atol=1e-6)

    def test_transform_positions_scaling(self, positions, cell):
        new_cell = 2.0 * cell
        new_pos = transform_positions(positions, cell, new_cell)
        assert jnp.allclose(new_pos, 2.0 * positions, atol=1e-6)


# ---------------------------------------------------------------------------
# Volume Move Tests
# ---------------------------------------------------------------------------


class TestVolumeMoveStep:
    def test_step_returns_state_and_info(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = jax.jit(vol_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=100.0)

        assert isinstance(new_state, MCState)
        assert hasattr(info, "accepted")
        assert hasattr(info, "n_evaluations")

    def test_accepts_below_constraint(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.01)
        step = jax.jit(vol_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        assert info.accepted

    def test_rejects_above_constraint(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.01)
        step = jax.jit(vol_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.0)
        assert not info.accepted
        assert jnp.array_equal(new_state.positions, state.positions)

    def test_volume_changes_on_accept(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.5)
        step = jax.jit(vol_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        if info.accepted:
            old_vol = get_volume(state.cell)
            new_vol = get_volume(new_state.cell)
            assert not jnp.allclose(old_vol, new_vol, atol=1e-8)

    def test_jit(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = vol_build_kernel(cell_energy_backend, N_ATOMS)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 100.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, positions, types, cell):
        step = jax.jit(vol_build_kernel(cell_energy_backend, N_ATOMS))

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.5, 1.0, 1.5, 2.0])
        batch_step_size = jnp.full(4, 0.1)
        batch_cell = jnp.stack([cell] * 4)

        batch_state = MCState(
            positions=batch_pos, types=batch_types,
            energy=batch_energy, cell=batch_cell,
            step_size=batch_step_size,
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

    def test_scan_compatible(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = jax.jit(vol_build_kernel(cell_energy_backend, N_ATOMS))

        def scan_step(carry, key):
            new_state, info = step(key, carry, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_cell_shape_constraint_enforced(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=10.0)
        step = jax.jit(vol_build_kernel(
            cell_energy_backend, N_ATOMS,
            max_vol_per_atom=63.0,
            min_vol_per_atom=62.0,
            min_aspect=0.99,
        ))

        n_rejected = 0
        key = jax.random.key(0)
        for i in range(20):
            key, subkey = jax.random.split(key)
            new_state, info = step(subkey, state, likelihood_constraint=1e6)
            if not info.accepted:
                n_rejected += 1
        assert n_rejected > 5


# ---------------------------------------------------------------------------
# Shear Move Tests
# ---------------------------------------------------------------------------


class TestShearMoveStep:
    def test_step_returns_state_and_info(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = jax.jit(shear_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=100.0)

        assert isinstance(new_state, MCState)
        assert hasattr(info, "accepted")

    def test_accepts_below_constraint(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.01)
        step = jax.jit(shear_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        assert info.accepted

    def test_rejects_above_constraint(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.01)
        step = jax.jit(shear_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.0)
        assert not info.accepted

    def test_volume_preserved(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.5)
        step = jax.jit(shear_build_kernel(cell_energy_backend, N_ATOMS))

        old_vol = get_volume(state.cell)
        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        new_vol = get_volume(new_state.cell)
        assert jnp.allclose(old_vol, new_vol, atol=1e-4)

    def test_jit(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = shear_build_kernel(cell_energy_backend, N_ATOMS)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 100.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, positions, types, cell):
        step = jax.jit(shear_build_kernel(cell_energy_backend, N_ATOMS))

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.5, 1.0, 1.5, 2.0])
        batch_step_size = jnp.full(4, 0.1)
        batch_cell = jnp.stack([cell] * 4)

        batch_state = MCState(
            positions=batch_pos, types=batch_types,
            energy=batch_energy, cell=batch_cell,
            step_size=batch_step_size,
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

    def test_scan_compatible(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = jax.jit(shear_build_kernel(cell_energy_backend, N_ATOMS))

        def scan_step(carry, key):
            new_state, info = step(key, carry, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_cell_shape_constraint_enforced(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=10.0)
        step = jax.jit(shear_build_kernel(
            cell_energy_backend, N_ATOMS, min_aspect=0.99,
        ))

        n_rejected = 0
        key = jax.random.key(0)
        for i in range(20):
            key, subkey = jax.random.split(key)
            new_state, info = step(subkey, state, likelihood_constraint=1e6)
            if not info.accepted:
                n_rejected += 1
        assert n_rejected > 5


# ---------------------------------------------------------------------------
# Stretch Move Tests
# ---------------------------------------------------------------------------


class TestStretchMoveStep:
    def test_step_returns_state_and_info(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = jax.jit(stretch_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=100.0)

        assert isinstance(new_state, MCState)
        assert hasattr(info, "accepted")

    def test_accepts_below_constraint(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.01)
        step = jax.jit(stretch_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        assert info.accepted

    def test_rejects_above_constraint(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.01)
        step = jax.jit(stretch_build_kernel(cell_energy_backend, N_ATOMS))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.0)
        assert not info.accepted

    def test_volume_preserved(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.5)
        step = jax.jit(stretch_build_kernel(cell_energy_backend, N_ATOMS))

        old_vol = get_volume(state.cell)
        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1000.0)
        new_vol = get_volume(new_state.cell)
        assert jnp.allclose(old_vol, new_vol, atol=1e-4)

    def test_jit(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = stretch_build_kernel(cell_energy_backend, N_ATOMS)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 100.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, positions, types, cell):
        step = jax.jit(stretch_build_kernel(cell_energy_backend, N_ATOMS))

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.5, 1.0, 1.5, 2.0])
        batch_step_size = jnp.full(4, 0.1)
        batch_cell = jnp.stack([cell] * 4)

        batch_state = MCState(
            positions=batch_pos, types=batch_types,
            energy=batch_energy, cell=batch_cell,
            step_size=batch_step_size,
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

    def test_scan_compatible(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=0.1)
        step = jax.jit(stretch_build_kernel(cell_energy_backend, N_ATOMS))

        def scan_step(carry, key):
            new_state, info = step(key, carry, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_cell_shape_constraint_enforced(self, positions, types, cell):
        state = _make_cell_state(positions, types, energy=0.5, cell=cell, step_size=10.0)
        step = jax.jit(stretch_build_kernel(
            cell_energy_backend, N_ATOMS, min_aspect=0.99,
        ))

        n_rejected = 0
        key = jax.random.key(0)
        for i in range(20):
            key, subkey = jax.random.split(key)
            new_state, info = step(subkey, state, likelihood_constraint=1e6)
            if not info.accepted:
                n_rejected += 1
        assert n_rejected > 5


# ---------------------------------------------------------------------------
# Integration Test: Volume Move on LJ Cluster with PBC
# ---------------------------------------------------------------------------


class TestLJIntegration:
    def test_volume_move_on_lj_cluster(self):
        """Run 200 volume move steps on a 4-atom LJ cluster with PBC."""
        backend = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        n_atoms = 4

        cell = 5.0 * jnp.eye(3)
        positions = jnp.array([
            [0.5, 0.5, 0.5],
            [2.0, 0.5, 0.5],
            [0.5, 2.0, 0.5],
            [0.5, 0.5, 2.0],
        ])
        types = jnp.zeros(n_atoms, dtype=jnp.int32)

        init_energy = backend(positions, types, cell, 0)[0]
        state = _make_cell_state(positions, types, energy=init_energy, cell=cell, step_size=0.5)
        step = jax.jit(vol_build_kernel(
            backend, n_atoms,
            max_vol_per_atom=100.0,
            min_vol_per_atom=1.0,
            min_aspect=0.5,
        ))

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

        assert n_accepted > 10, f"Only {n_accepted} accepted out of {n_steps}"
        unique_vols = jnp.unique(jnp.round(volumes, decimals=2))
        assert len(unique_vols) > 1, "Expected different volumes from accepted moves"
        assert jnp.all(volumes > 0.0), "All volumes should be positive"
        assert jnp.all(volumes / n_atoms <= 100.0), "Volume per atom constraint"
        assert jnp.all(volumes / n_atoms >= 1.0), "Min volume per atom constraint"


# ---------------------------------------------------------------------------
# Regression Tests for TF32 precision fix (Fix 1)
# ---------------------------------------------------------------------------


class TestTF32PrecisionFix:
    """Regression tests for TF32 matmul precision bug.

    On CUDA, JAX defaults to TF32 (10-bit mantissa).  Even for identity T,
    `positions @ eye(3)` introduces ~3.7e-3 noise per element, which is
    enough to spike LJ energy by 10^2-10^3 units at dense packing, causing
    100% rejection regardless of step size.

    The fix (jnp.einsum with Precision.HIGHEST) must produce bit-exact
    results for an identity transform and keep energy stable at tiny step size.
    Both tests pass on CPU (no TF32) and catch regressions on GPU (TF32 active).
    """

    def test_transform_positions_identity_is_bit_exact(self):
        """positions @ eye(3) must be bit-exact after the HIGHEST precision fix.

        This catches any reintroduction of TF32 matmul in transform_positions
        (cell.py), which is shared by the shear move path.
        """
        # Use a realistic dense-packing scenario: 64 atoms, cell side ~11 Å
        key = jax.random.key(0)
        n_atoms = 64
        cell_side = 11.0
        cell = cell_side * jnp.eye(3)
        positions = jax.random.uniform(key, (n_atoms, 3), minval=0.0, maxval=cell_side)

        new_positions = transform_positions(positions, cell, cell)  # T = identity

        max_err = float(jnp.max(jnp.abs(new_positions - positions)))
        assert max_err == 0.0, (
            f"transform_positions(identity) is not bit-exact: max |err| = {max_err:.3e}. "
            "TF32 precision leaking through — check Precision.HIGHEST in cell.py."
        )

    def test_transform_positions_identity_bit_exact_under_jit(self):
        """Same bit-exactness check, compiled under jax.jit."""
        key = jax.random.key(1)
        n_atoms = 64
        cell_side = 11.0
        cell = cell_side * jnp.eye(3)
        positions = jax.random.uniform(key, (n_atoms, 3), minval=0.0, maxval=cell_side)

        jit_transform = jax.jit(transform_positions)
        new_positions = jit_transform(positions, cell, cell)

        max_err = float(jnp.max(jnp.abs(new_positions - positions)))
        assert max_err == 0.0, (
            f"JIT transform_positions(identity) is not bit-exact: max |err| = {max_err:.3e}. "
            "TF32 precision leaking through — check Precision.HIGHEST in cell.py."
        )

    def test_volume_move_tiny_step_size_preserves_energy(self):
        """A volume move at ss=1e-20 must return new_energy == pre-move energy.

        At an infinitesimally small step, the proposed new cell is
        indistinguishable from the old one, so the energy should be
        identical to within 1e-10.  On GPU with TF32, the position matmul
        would introduce ~3.7e-3 noise, causing the energy to spike and this
        assertion to fail.
        """
        backend = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        n_atoms = 4

        cell = 5.0 * jnp.eye(3)
        positions = jnp.array([
            [0.5, 0.5, 0.5],
            [2.0, 0.5, 0.5],
            [0.5, 2.0, 0.5],
            [0.5, 0.5, 2.0],
        ])
        types = jnp.zeros(n_atoms, dtype=jnp.int32)

        init_energy = backend(positions, types, cell, 0)[0]
        # Tiny step size: proposed volume change is ~1e-20 * n_atoms * N(0,1)
        state = _make_cell_state(positions, types, energy=init_energy,
                                 cell=cell, step_size=1e-20)

        step = jax.jit(vol_build_kernel(
            backend, n_atoms,
            max_vol_per_atom=100.0,
            min_vol_per_atom=1.0,
            min_aspect=0.5,
        ))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=1e10)

        # The proposed energy at an infinitesimal step must equal original energy
        # (tolerance 1e-10; TF32 noise would give ~1e2 discrepancy at dense packing)
        energy_diff = float(jnp.abs(new_state.energy - init_energy))
        assert energy_diff < 1e-10, (
            f"Energy changed by {energy_diff:.3e} at ss=1e-20.  "
            "Expected bit-stable result — TF32 may be leaking through the matmul."
        )


# ---------------------------------------------------------------------------
# Regression: bucket-sizing signals are gated on ``cell_valid``
# ---------------------------------------------------------------------------


class _OverflowingBackend:
    """Stub backend that always reports overflow + a large neighbor count.

    Used to verify that cell moves do not let bucket-sizing signals leak
    out of proposals that violated the cell-shape constraint.
    """

    r_cutoff = 0.0

    def __init__(self, count: int = 999):
        self._count = count

    def __call__(self, positions, species, cell, max_neighbors=0,
                 ensemble_params=None):
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
    """``state.overflow`` and ``state.max_neighbor_count`` must NOT be
    poisoned by proposals that the chain cannot live at (cell-shape
    rejections from max/min volume per atom or min aspect).

    Without the ``cell_valid`` gate, hard-rejected proposals would force
    the outer Python loop to permanently escalate the neighbor bucket
    despite the chain never sampling near those configurations.
    """

    @pytest.mark.parametrize("move", ["volume", "shear", "stretch"])
    def test_cell_invalid_does_not_set_overflow_or_count(
        self, move, positions, types, cell,
    ):
        # Big step + extremely tight cell bounds → every proposal is
        # cell-invalid, the backend reports overflow=True and count=999,
        # but state must stay clean.
        state = _make_cell_state(positions, types, energy=0.5,
                                 cell=cell, step_size=10.0)
        step = jax.jit(_MOVE_BUILDERS[move](
            _OVERFLOWING_BACKEND, N_ATOMS,
            max_vol_per_atom=63.0,
            min_vol_per_atom=62.0,
            min_aspect=0.999,
        ))

        for i in range(20):
            key = jax.random.key(i)
            state, info = step(key, state, likelihood_constraint=1e10)
            # Cell rejection (reason 2 for shear/stretch, 2 for volume:
            # priority is energy=1 > cell=2 > prior=3, and we set Emax
            # to 1e10 so energy never trips).
            assert not bool(info.accepted)

        assert bool(state.overflow) is False, (
            f"{move}: overflow leaked from a cell-invalid proposal"
        )
        assert int(state.max_neighbor_count) == 0, (
            f"{move}: max_neighbor_count leaked from a cell-invalid proposal"
        )

    @pytest.mark.parametrize("move", ["volume", "shear", "stretch"])
    def test_cell_valid_still_propagates_overflow(
        self, move, positions, types, cell,
    ):
        # Loose bounds + tiny step → proposal cell is valid; the gate
        # must NOT swallow the genuine overflow signal.
        state = _make_cell_state(positions, types, energy=0.5,
                                 cell=cell, step_size=1e-4)
        step = jax.jit(_MOVE_BUILDERS[move](
            _OVERFLOWING_BACKEND, N_ATOMS,
            max_vol_per_atom=1000.0,
            min_vol_per_atom=1.0,
            min_aspect=0.5,
        ))
        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=1e10)

        assert bool(new_state.overflow) is True, (
            f"{move}: overflow gate swallowed a legitimate cell-valid signal"
        )
        assert int(new_state.max_neighbor_count) == 999, (
            f"{move}: max_neighbor_count gate dropped a cell-valid count"
        )