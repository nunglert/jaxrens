"""Tests for atomic move kernels.

Tests:
- HMC: step, JIT, vmap, lax.scan, acceptance rate on differentiable potential
- SingleAtomMove: step, JIT, vmap, lax.scan
- SingleAtomSweep: step, JIT, sweep through all atoms
- AtomMorph: step, JIT, species changes
- All kernels operate on MCState
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_double_well, create_harmonic
from jaxrens.base import MoveInfo
from jaxrens.sampling.moves import alchemical, hmc, single_atom
from jaxrens.state.mc_state import MCState


def _make_state(positions, types, energy, cell=None, step_size=0.1):
    """Helper: create MCState for tests."""
    if cell is None:
        cell = jnp.zeros((3, 3))
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


def _make_batch_state(positions, types, energies, step_size=0.1):
    """Helper: create batched MCState for vmap tests."""
    n = positions.shape[0]
    return MCState(
        positions=positions,
        types=jnp.broadcast_to(types, (n, *types.shape))
        if types.ndim == 1
        else types,
        energy=energies,
        cell=jnp.zeros((n, 3, 3)),
        step_size=jnp.full(n, step_size),
        step_sizes=jnp.full((n, 1), step_size),
        n_accepted=jnp.zeros((n, 1), dtype=jnp.int32),
        n_proposed=jnp.zeros((n, 1), dtype=jnp.int32),
        max_neighbor_count=jnp.zeros(n, dtype=jnp.int32),
        overflow=jnp.full(n, False),
        ensemble_params={},
    )


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
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def types_1():
    return jnp.zeros((1,), dtype=jnp.int32)


@pytest.fixture
def types_4():
    return jnp.array([0, 0, 1, 1], dtype=jnp.int32)


# ── HMC Tests ───────────────────────────────────────────────────────────────


class TestHMC:
    def test_step_basic(self, harmonic, positions_1atom, types_1):
        backend = harmonic
        energy = backend(positions_1atom, types_1, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_1atom, types_1, energy, step_size=0.01)
        step_fn = jax.jit(hmc.build_kernel(backend, n_leapfrog=5))

        key = jax.random.key(0)
        new_state, info = step_fn(key, state, 100.0)

        assert new_state.positions.shape == state.positions.shape
        assert isinstance(info, MoveInfo)

    def test_jit(self, harmonic, positions_1atom, types_1):
        backend = harmonic
        energy = backend(positions_1atom, types_1, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_1atom, types_1, energy, step_size=0.01)
        step_fn = jax.jit(hmc.build_kernel(backend, n_leapfrog=5))

        key = jax.random.key(1)
        new_state, info = step_fn(key, state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_vmap(self, harmonic, types_1):
        backend = harmonic
        n_walkers = 8
        key = jax.random.key(2)
        positions = jax.random.normal(key, (n_walkers, 1, 3))
        cell = jnp.zeros((3, 3))
        energies = jax.vmap(lambda pos: backend(pos, types_1, cell, 0)[0])(
            positions
        )
        states = _make_batch_state(
            positions, types_1, energies, step_size=0.01
        )
        step_fn = jax.jit(hmc.build_kernel(backend, n_leapfrog=5))
        keys = jax.random.split(jax.random.key(3), n_walkers)
        constraints = 100.0 * jnp.ones(n_walkers)

        new_states, infos = jax.vmap(step_fn)(keys, states, constraints)
        assert new_states.positions.shape == (n_walkers, 1, 3)

    def test_lax_scan(self, harmonic, positions_1atom, types_1):
        backend = harmonic
        energy = backend(positions_1atom, types_1, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_1atom, types_1, energy, step_size=0.01)
        step_fn = jax.jit(hmc.build_kernel(backend, n_leapfrog=5))

        def scan_step(state, key):
            new_state, info = step_fn(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(4), 20)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert final_state.positions.shape == (1, 3)
        assert accepted.shape == (20,)

    def test_acceptance_rate(self, harmonic, types_1):
        backend = harmonic
        key = jax.random.key(5)
        positions = 0.5 * jax.random.normal(key, (1, 3))
        energy = backend(positions, types_1, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions, types_1, energy, step_size=0.005)
        step_fn = jax.jit(hmc.build_kernel(backend, n_leapfrog=10))

        n_steps = 100
        keys = jax.random.split(jax.random.key(6), n_steps)
        accepted_count = 0
        for k in keys:
            state, info = step_fn(k, state, 50.0)
            accepted_count += int(info.accepted)

        rate = accepted_count / n_steps
        assert rate > 0.3, f"HMC acceptance rate too low: {rate}"


# ── SingleAtomMove Tests ────────────────────────────────────────────────────


class TestSingleAtomMove:
    def test_step_basic(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_kernel(backend))

        key = jax.random.key(10)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)

    def test_jit(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_kernel(backend))

        new_state, info = step_fn(jax.random.key(11), state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_vmap(self, harmonic, types_1):
        backend = harmonic
        n_walkers = 4
        positions = jax.random.normal(jax.random.key(12), (n_walkers, 1, 3))
        cell = jnp.zeros((3, 3))
        energies = jax.vmap(lambda pos: backend(pos, types_1, cell, 0)[0])(
            positions
        )
        states = _make_batch_state(positions, types_1, energies, step_size=0.1)
        step_fn = jax.jit(single_atom.build_kernel(backend))
        keys = jax.random.split(jax.random.key(13), n_walkers)
        constraints = 100.0 * jnp.ones(n_walkers)

        new_states, infos = jax.vmap(step_fn)(keys, states, constraints)
        assert new_states.positions.shape == (n_walkers, 1, 3)

    def test_lax_scan(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_kernel(backend))

        def scan_step(state, key):
            new_state, info = step_fn(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(14), 20)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert final_state.positions.shape == (4, 3)

    def test_only_one_atom_changes(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_kernel(backend))

        key = jax.random.key(15)
        new_state, info = step_fn(key, state, 100.0)

        changed = jnp.any(new_state.positions != state.positions, axis=1)
        n_changed = jnp.sum(changed)
        assert n_changed <= 1


# ── SingleAtomSweep Tests ───────────────────────────────────────────────────


class TestSingleAtomSweep:
    def test_step_basic(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_sweep_kernel(backend, n_atoms=4))

        key = jax.random.key(20)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)
        assert info.n_evaluations == 4

    def test_jit(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy, step_size=0.1)
        step_fn = jax.jit(single_atom.build_sweep_kernel(backend, n_atoms=4))

        new_state, info = step_fn(jax.random.key(21), state, 100.0)
        assert jnp.isfinite(new_state.energy)


# ── AtomMorph Tests ─────────────────────────────────────────────────────────


class TestAtomMorph:
    def test_step_basic(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy)
        step_fn = jax.jit(alchemical.build_morph_kernel(backend, n_species=2))

        key = jax.random.key(50)
        new_state, info = step_fn(key, state, 100.0)
        assert new_state.positions.shape == (4, 3)

    def test_jit(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy)
        step_fn = jax.jit(alchemical.build_morph_kernel(backend, n_species=2))

        new_state, info = step_fn(jax.random.key(51), state, 100.0)
        assert jnp.isfinite(new_state.energy)

    def test_species_change(self, harmonic, positions_4atom, types_4):
        backend = harmonic
        energy = backend(positions_4atom, types_4, jnp.zeros((3, 3)), 0)[0]
        state = _make_state(positions_4atom, types_4, energy)
        step_fn = jax.jit(alchemical.build_morph_kernel(backend, n_species=3))

        changed = False
        for i in range(50):
            state, info = step_fn(jax.random.key(52 + i), state, 100.0)
            if bool(info.accepted):
                changed = True
                break

        assert changed, "Morph should accept at least once in 50 tries"
