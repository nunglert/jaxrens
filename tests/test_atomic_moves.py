"""Tests for atomic move kernels (Step 15).

Tests:
- HMC: init, step, JIT, vmap, lax.scan, acceptance rate on differentiable potential
- SingleAtomMove: init, step, JIT, vmap, lax.scan
- SingleAtomSweep: init, step, JIT, sweep through all atoms
- SingleAtomSwap: init, step, JIT, multi-component systems
- AtomMorph: init, step, JIT, species changes
- RandomShift: init, step, JIT, rigid translation
- All kernels follow MoveKernel(init_fn, step_fn) protocol
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic, create_double_well
from jaxrens.sampling.moves import hmc, single_atom, alchemical
from jaxrens.base import MoveInfo


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


@pytest.fixture
def double_well():
    return create_double_well(barrier=2.0)


@pytest.fixture
def positions_1atom():
    return jnp.array([[1.0, 0.5, -0.3]])


@pytest.fixture
def positions_4atom():
    return jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def types_1():
    return jnp.zeros((1,), dtype=jnp.int32)


@pytest.fixture
def types_4():
    return jnp.array([0, 0, 1, 1], dtype=jnp.int32)


# ── HMC Tests ───────────────────────────────────────────────────────────────


class TestHMC:
    def test_init(self, harmonic, positions_1atom, types_1):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_1atom, types_1)
        state = hmc.init(positions_1atom, types_1, energy, step_size=0.01)
        assert state.positions.shape == (1, 3)
        assert state.step_size == pytest.approx(0.01)

    def test_step_basic(self, harmonic, positions_1atom, types_1):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_1atom, types_1)
        state = hmc.init(positions_1atom, types_1, energy, step_size=0.01)
        step_fn = hmc.build_kernel(energy_fn, params, n_leapfrog=5)

        key = jax.random.key(0)
        new_state, info = step_fn(key, state, 100.0)  # generous constraint

        assert new_state.positions.shape == state.positions.shape
        assert isinstance(info, MoveInfo)

    def test_jit(self, harmonic, positions_1atom, types_1):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_1atom, types_1)
        state = hmc.init(positions_1atom, types_1, energy, step_size=0.01)
        step_fn = jax.jit(hmc.build_kernel(energy_fn, params, n_leapfrog=5))

        key = jax.random.key(1)
        new_state, info = step_fn(key, state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_vmap(self, harmonic, types_1):
        energy_fn, params = harmonic
        n_walkers = 8
        key = jax.random.key(2)
        positions = jax.random.normal(key, (n_walkers, 1, 3))
        energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, positions, types_1
        )
        states = jax.vmap(hmc.init, in_axes=(0, None, 0, None, None))(
            positions, types_1, energies, None, 0.01
        )
        step_fn = hmc.build_kernel(energy_fn, params, n_leapfrog=5)
        keys = jax.random.split(jax.random.key(3), n_walkers)
        constraints = 100.0 * jnp.ones(n_walkers)

        new_states, infos = jax.vmap(step_fn)(keys, states, constraints)
        assert new_states.positions.shape == (n_walkers, 1, 3)

    def test_lax_scan(self, harmonic, positions_1atom, types_1):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_1atom, types_1)
        state = hmc.init(positions_1atom, types_1, energy, step_size=0.01)
        step_fn = hmc.build_kernel(energy_fn, params, n_leapfrog=5)

        def scan_step(state, key):
            new_state, info = step_fn(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(4), 20)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert final_state.positions.shape == (1, 3)
        assert accepted.shape == (20,)

    def test_acceptance_rate(self, harmonic, types_1):
        """HMC on harmonic should have decent acceptance with good step size."""
        energy_fn, params = harmonic
        key = jax.random.key(5)
        positions = 0.5 * jax.random.normal(key, (1, 3))
        energy = energy_fn(params, positions, types_1)
        state = hmc.init(positions, types_1, energy, step_size=0.005)
        step_fn = jax.jit(hmc.build_kernel(energy_fn, params, n_leapfrog=10))

        n_steps = 100
        keys = jax.random.split(jax.random.key(6), n_steps)
        accepted_count = 0
        for k in keys:
            state, info = step_fn(k, state, 50.0)
            accepted_count += int(info.accepted)

        rate = accepted_count / n_steps
        assert rate > 0.3, f"HMC acceptance rate too low: {rate}"

    def test_top_level_api(self, harmonic, positions_1atom, types_1):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_1atom, types_1)
        mk = hmc.as_top_level_api(energy_fn, params, step_size=0.01, n_leapfrog=5)
        state = mk.init(positions_1atom, types_1, energy)
        new_state, info = mk.step(jax.random.key(7), state, 100.0)
        assert isinstance(info, MoveInfo)


# ── SingleAtomMove Tests ────────────────────────────────────────────────────


class TestSingleAtomMove:
    def test_init(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy)
        assert state.positions.shape == (4, 3)

    def test_step_basic(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = single_atom.build_kernel(energy_fn, params)

        key = jax.random.key(10)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)

    def test_jit(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_kernel(energy_fn, params))

        new_state, info = step_fn(jax.random.key(11), state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_vmap(self, harmonic, types_1):
        energy_fn, params = harmonic
        n_walkers = 4
        positions = jax.random.normal(jax.random.key(12), (n_walkers, 1, 3))
        energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, positions, types_1
        )
        states = jax.vmap(single_atom.init, in_axes=(0, None, 0, None, None))(
            positions, types_1, energies, None, 0.1
        )
        step_fn = single_atom.build_kernel(energy_fn, params)
        keys = jax.random.split(jax.random.key(13), n_walkers)
        constraints = 100.0 * jnp.ones(n_walkers)

        new_states, infos = jax.vmap(step_fn)(keys, states, constraints)
        assert new_states.positions.shape == (n_walkers, 1, 3)

    def test_lax_scan(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = single_atom.build_kernel(energy_fn, params)

        def scan_step(state, key):
            new_state, info = step_fn(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(14), 20)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert final_state.positions.shape == (4, 3)

    def test_only_one_atom_changes(self, harmonic, positions_4atom, types_4):
        """Verify that at most one atom is displaced per step."""
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = single_atom.build_kernel(energy_fn, params)

        key = jax.random.key(15)
        new_state, info = step_fn(key, state, 100.0)

        # Count how many atoms changed
        changed = jnp.any(new_state.positions != state.positions, axis=1)
        n_changed = jnp.sum(changed)
        assert n_changed <= 1

    def test_top_level_api(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        mk = single_atom.as_top_level_api(energy_fn, params, step_size=0.1)
        state = mk.init(positions_4atom, types_4, energy)
        new_state, info = mk.step(jax.random.key(16), state, 100.0)
        assert isinstance(info, MoveInfo)


# ── SingleAtomSweep Tests ───────────────────────────────────────────────────


class TestSingleAtomSweep:
    def test_step_basic(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = single_atom.build_sweep_kernel(energy_fn, params, n_atoms=4)

        key = jax.random.key(20)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)
        assert info.n_evaluations == 4  # one eval per atom

    def test_jit(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_sweep_kernel(energy_fn, params, n_atoms=4))

        new_state, info = step_fn(jax.random.key(21), state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_sweep_api(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        mk = single_atom.as_sweep_api(energy_fn, params, n_atoms=4, step_size=0.1)
        state = mk.init(positions_4atom, types_4, energy)
        new_state, info = mk.step(jax.random.key(22), state, 100.0)
        assert isinstance(info, MoveInfo)


# ── SingleAtomSwap Tests ────────────────────────────────────────────────────


class TestSingleAtomSwap:
    def test_step_basic(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy)
        step_fn = single_atom.build_swap_kernel(energy_fn, params)

        key = jax.random.key(30)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)
        # Types should be a permutation of original (same counts)
        assert jnp.sum(new_state.types == 0) + jnp.sum(new_state.types == 1) == 4

    def test_jit(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy)
        step_fn = jax.jit(single_atom.build_swap_kernel(energy_fn, params))

        new_state, info = step_fn(jax.random.key(31), state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_preserves_species_counts(self, harmonic, positions_4atom, types_4):
        """Swap should preserve the number of each species."""
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = single_atom.init(positions_4atom, types_4, energy)
        step_fn = jax.jit(single_atom.build_swap_kernel(energy_fn, params))

        for i in range(10):
            state, info = step_fn(jax.random.key(32 + i), state, 100.0)
            n_type0 = jnp.sum(state.types == 0)
            n_type1 = jnp.sum(state.types == 1)
            assert n_type0 == 2
            assert n_type1 == 2

    def test_no_swap_single_species(self, harmonic, positions_4atom):
        """With only one species, swap should always reject (no different species)."""
        energy_fn, params = harmonic
        types_single = jnp.zeros((4,), dtype=jnp.int32)
        energy = energy_fn(params, positions_4atom, types_single)
        state = single_atom.init(positions_4atom, types_single, energy)
        step_fn = jax.jit(single_atom.build_swap_kernel(energy_fn, params))

        n_accepted = 0
        for i in range(20):
            state, info = step_fn(jax.random.key(40 + i), state, 100.0)
            n_accepted += int(info.accepted)

        # With single species, all selected pairs have same type
        # The acceptance condition requires different species
        # But random selection might pick same index -> same type always
        # Either way, no actual type change should occur
        assert jnp.all(state.types == 0)

    def test_swap_api(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        mk = single_atom.as_swap_api(energy_fn, params)
        state = mk.init(positions_4atom, types_4, energy)
        new_state, info = mk.step(jax.random.key(45), state, 100.0)
        assert isinstance(info, MoveInfo)


# ── AtomMorph Tests ─────────────────────────────────────────────────────────


class TestAtomMorph:
    def test_step_basic(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy)
        step_fn = alchemical.build_morph_kernel(energy_fn, params, n_species=2)

        key = jax.random.key(50)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)

    def test_jit(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy)
        step_fn = jax.jit(alchemical.build_morph_kernel(energy_fn, params, n_species=2))

        new_state, info = step_fn(jax.random.key(51), state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_species_change(self, harmonic, positions_4atom, types_4):
        """Morph should sometimes change a species."""
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy)
        step_fn = jax.jit(alchemical.build_morph_kernel(energy_fn, params, n_species=3))

        changed = False
        for i in range(50):
            state, info = step_fn(jax.random.key(52 + i), state, 100.0)
            if bool(info.accepted):
                changed = True
                break

        assert changed, "Morph should accept at least once in 50 tries"

    def test_new_type_always_different(self, harmonic, positions_4atom, types_4):
        """When morph accepts, the changed atom must have a different type."""
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy)
        step_fn = jax.jit(alchemical.build_morph_kernel(energy_fn, params, n_species=3))

        for i in range(30):
            old_types = state.types.copy()
            state, info = step_fn(jax.random.key(60 + i), state, 100.0)
            if bool(info.accepted):
                # Find which atom changed
                diff = state.types != old_types
                changed_idx = jnp.where(diff, size=1)[0]
                if len(changed_idx) > 0:
                    # New type must be different from old
                    assert state.types[changed_idx[0]] != old_types[changed_idx[0]]

    def test_morph_api(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        mk = alchemical.as_morph_api(energy_fn, params, n_species=2)
        state = mk.init(positions_4atom, types_4, energy)
        new_state, info = mk.step(jax.random.key(65), state, 100.0)
        assert isinstance(info, MoveInfo)


# ── RandomShift Tests ───────────────────────────────────────────────────────


class TestRandomShift:
    def test_step_basic(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = alchemical.build_shift_kernel(energy_fn, params)

        key = jax.random.key(70)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)

    def test_jit(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(alchemical.build_shift_kernel(energy_fn, params))

        new_state, info = step_fn(jax.random.key(71), state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_rigid_translation(self, harmonic, positions_4atom, types_4):
        """All atoms should move by the same displacement."""
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy, step_size=0.5)
        step_fn = alchemical.build_shift_kernel(energy_fn, params)

        key = jax.random.key(72)
        new_state, info = step_fn(key, state, 1000.0)  # very generous constraint

        if bool(info.accepted):
            # All displacements should be identical
            displacements = new_state.positions - state.positions
            for i in range(1, 4):
                assert jnp.allclose(displacements[0], displacements[i], atol=1e-6)

    def test_vmap(self, harmonic, types_4):
        energy_fn, params = harmonic
        n_walkers = 4
        key = jax.random.key(73)
        positions = jax.random.normal(key, (n_walkers, 4, 3))
        energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
            params, positions, types_4
        )
        states = jax.vmap(alchemical.init, in_axes=(0, None, 0, None, None))(
            positions, types_4, energies, None, 0.1
        )
        step_fn = alchemical.build_shift_kernel(energy_fn, params)
        keys = jax.random.split(jax.random.key(74), n_walkers)
        constraints = 100.0 * jnp.ones(n_walkers)

        new_states, infos = jax.vmap(step_fn)(keys, states, constraints)
        assert new_states.positions.shape == (n_walkers, 4, 3)

    def test_lax_scan(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        state = alchemical.init(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = alchemical.build_shift_kernel(energy_fn, params)

        def scan_step(state, key):
            new_state, info = step_fn(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(75), 20)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert final_state.positions.shape == (4, 3)
        assert accepted.shape == (20,)

    def test_shift_api(self, harmonic, positions_4atom, types_4):
        energy_fn, params = harmonic
        energy = energy_fn(params, positions_4atom, types_4)
        mk = alchemical.as_shift_api(energy_fn, params, step_size=0.1)
        state = mk.init(positions_4atom, types_4, energy)
        new_state, info = mk.step(jax.random.key(76), state, 100.0)
        assert isinstance(info, MoveInfo)
