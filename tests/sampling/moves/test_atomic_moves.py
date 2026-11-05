"""Tests for the atomic and species-swap kernels on the harmonic backend.

Each kernel has its own test class, so one move can be run alone with
`pytest -k TestHMC`.  The checks shared by all kernels live in _KernelChecks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec
from jaxrens.sampling.base import MoveInfo
from jaxrens.sampling.moves import alchemical, hmc, single_atom, swap
from jaxrens.state.mc_state import MCState


def _make_state(positions, types, energy, cell=None, step_size=0.1):
    """Build an MCState for one walker."""
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
    """Build an MCState with a leading walker axis, for vmap."""
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


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


@pytest.fixture
def positions():
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def types():
    return jnp.array([0, 0, 1, 1], dtype=jnp.int32)


def _initial_state(backend, positions, types, step_size=0.01):
    energy = backend(positions, types, jnp.zeros((3, 3)), 0)[0]
    return _make_state(positions, types, energy, step_size=step_size)


class _KernelChecks:
    """Checks that every move kernel must pass.

    Subclasses define build(backend), which returns the kernel under test.
    """

    @staticmethod
    def build(backend):
        raise NotImplementedError

    def test_step_basic(self, harmonic, positions, types):
        state = _initial_state(harmonic, positions, types)
        step_fn = jax.jit(self.build(harmonic))

        new_state, info = step_fn(jax.random.key(0), state, 100.0)

        assert new_state.positions.shape == state.positions.shape
        assert isinstance(info, MoveInfo)
        assert jnp.isfinite(new_state.energy)

    def test_vmap(self, harmonic, positions, types):
        n_walkers = 4
        batch = positions + 0.1 * jax.random.normal(
            jax.random.key(1), (n_walkers, 4, 3)
        )
        cell = jnp.zeros((3, 3))
        energies = jax.vmap(lambda pos: harmonic(pos, types, cell, 0)[0])(
            batch
        )
        states = _make_batch_state(batch, types, energies, step_size=0.01)
        step_fn = jax.jit(self.build(harmonic))
        keys = jax.random.split(jax.random.key(2), n_walkers)
        constraints = 100.0 * jnp.ones(n_walkers)

        new_states, infos = jax.vmap(step_fn)(keys, states, constraints)

        assert new_states.positions.shape == (n_walkers, 4, 3)
        assert infos.accepted.shape == (n_walkers,)

    def test_lax_scan(self, harmonic, positions, types):
        state = _initial_state(harmonic, positions, types)
        step_fn = jax.jit(self.build(harmonic))

        def scan_step(state, key):
            new_state, info = step_fn(key, state, 100.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(3), 20)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)

        assert final_state.positions.shape == (4, 3)
        assert accepted.shape == (20,)

    def test_rejected_move_leaves_state_unchanged(
        self, harmonic, positions, types
    ):
        """A constraint below the current energy rejects every proposal."""
        state = _initial_state(harmonic, positions, types)
        step_fn = jax.jit(self.build(harmonic))

        new_state, info = step_fn(jax.random.key(4), state, state.energy - 1.0)

        assert not bool(info.accepted)
        assert jnp.array_equal(new_state.positions, state.positions)
        assert jnp.array_equal(new_state.types, state.types)
        assert jnp.array_equal(new_state.energy, state.energy)


class TestHMC(_KernelChecks):
    """Check the HMC kernel on a harmonic potential.

    Besides the shared checks, the tests cover the acceptance rate.
    """

    @staticmethod
    def build(backend):
        return hmc.build_kernel(backend, n_leapfrog=5)

    def test_acceptance_rate(self, harmonic, positions, types):
        state = _initial_state(
            harmonic, 0.5 * positions, types, step_size=0.005
        )
        step_fn = jax.jit(hmc.build_kernel(harmonic, n_leapfrog=10))

        n_steps = 100
        keys = jax.random.split(jax.random.key(5), n_steps)
        accepted_count = 0
        for k in keys:
            state, info = step_fn(k, state, 50.0)
            accepted_count += int(info.accepted)

        assert accepted_count / n_steps > 0.3


class TestSingleAtomMove(_KernelChecks):
    """Check the kernel that displaces one random atom per step.

    Besides the shared checks, the tests cover that exactly one atom moves.
    """

    @staticmethod
    def build(backend):
        return single_atom.build_kernel(backend)

    def test_only_one_atom_changes(self, harmonic, positions, types):
        state = _initial_state(harmonic, positions, types, step_size=0.1)
        step_fn = jax.jit(self.build(harmonic))

        new_state, info = step_fn(jax.random.key(6), state, 100.0)

        changed = jnp.any(new_state.positions != state.positions, axis=1)
        assert jnp.sum(changed) == 1


class TestSingleAtomSweep(_KernelChecks):
    """Check the kernel that sweeps through all atoms in one step.

    Besides the shared checks, the tests cover the number of evaluations.
    """

    @staticmethod
    def build(backend):
        return single_atom.build_sweep_kernel(backend, n_atoms=4)

    def test_n_evaluations(self, harmonic, positions, types):
        state = _initial_state(harmonic, positions, types, step_size=0.1)
        step_fn = jax.jit(self.build(harmonic))

        new_state, info = step_fn(jax.random.key(7), state, 100.0)

        assert info.n_evaluations == 4


class TestAtomMorph(_KernelChecks):
    """Check the kernel that changes the species of one atom.

    Besides the shared checks, the tests cover the type change itself.
    """

    @staticmethod
    def build(backend):
        return alchemical.build_morph_kernel(backend, n_species=3)

    def test_species_change(self, harmonic, positions, types):
        state = _initial_state(harmonic, positions, types)
        step_fn = jax.jit(self.build(harmonic))

        new_state, info = step_fn(jax.random.key(52), state, 100.0)

        assert bool(info.accepted)
        assert int(jnp.sum(new_state.types != state.types)) == 1
        assert jnp.all((new_state.types >= 0) & (new_state.types < 3))


class _ConstantBackend:
    """Backend that returns a fixed energy for any configuration."""

    def __init__(self, energy=0.0):
        self.r_cutoff = 0.0
        self.energy = energy

    def __call__(
        self, positions, species, cell, max_neighbors=0, ensemble_params=None
    ):
        return BackendResult(energy=jnp.asarray(self.energy))


def _proposals(step_fn, state, n_draws, seed=0):
    """Return the proposed types for ``n_draws`` independent draws.

    The proposal is read off the accepted state, so the constraint is loose
    enough to accept every draw.  Recording it from inside the backend would
    only fire once, at trace time.
    """
    keys = jax.random.split(jax.random.key(seed), n_draws)
    new_states, info = jax.jit(jax.vmap(step_fn, in_axes=(0, None, None)))(
        keys, state, 1e6
    )
    assert bool(jnp.all(info.accepted))
    return new_states.types


def _run(step_fn, state, seeds, emax=1e6):
    """Apply the kernel once per seed and return the final state and infos."""
    out = []
    for seed in seeds:
        state, info = step_fn(jax.random.key(seed), state, emax)
        out.append(info)
    return state, out


class TestSpeciesSwap(_KernelChecks):
    """Check the kernel that swaps the species of two unlike atoms.

    Besides the shared checks, the tests cover argument validation, the pair
    proposal (unlike, uniform, scoped), accept and reject edge cases, and the
    SpeciesSwapMoveSpec that builds the kernel.
    """

    @staticmethod
    def build(backend):
        return swap.build_kernel(backend, n_species=2)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_species": 1}, "n_species >= 2"),
            ({"n_species": 3, "species": (1,)}, "at least two"),
            ({"n_species": 3, "species": (2, 2)}, "at least two"),
            ({"n_species": 2, "species": (0, 5)}, "outside"),
        ],
        ids=[
            "one_species",
            "one_code",
            "duplicate_codes",
            "code_out_of_range",
        ],
    )
    def test_build_kernel_rejects_invalid_species(
        self, harmonic, kwargs, match
    ):
        with pytest.raises(ValueError, match=match):
            swap.build_kernel(harmonic, **kwargs)

    def test_every_proposal_is_an_unlike_pair(
        self, harmonic, positions, types
    ):
        """The whole point: no draw is spent on a same-species pair."""
        state = _make_state(positions, types, 0.0)
        step_fn = self.build(harmonic)

        proposed = _proposals(step_fn, state, 500)

        # A real swap changes types at exactly two positions.
        assert bool(jnp.all(jnp.sum(proposed != types, axis=1) == 2))

    def test_pair_selection_is_uniform_ternary(self, harmonic):
        """Uniform over unlike *pairs*, not merely over unlike species.

        A ternary system with unequal abundances is what separates this
        kernel from the first-atom-uniform alternative, which would
        over-weight pairs involving the rare species.
        """
        positions = jnp.zeros((6, 3))
        types = jnp.array(
            [0, 0, 0, 1, 1, 2], dtype=jnp.int32
        )  # 3x A, 2x B, 1x C
        state = _make_state(positions, types, 0.0)
        step_fn = swap.build_kernel(harmonic, n_species=3)

        n_draws = 22000
        proposed = _proposals(step_fn, state, n_draws)

        # Encode the pair of changed positions in each draw as a bitmask.
        changed = proposed != types
        pair_ids = changed @ (2 ** jnp.arange(types.size))
        _, counts = jnp.unique(pair_ids, return_counts=True)
        # 3*2 + 3*1 + 2*1 = 11 unlike pairs, each expected n_draws / 11.
        assert len(counts) == 11
        expected = n_draws / 11
        assert bool(jnp.all(jnp.abs(counts - expected) <= 0.15 * expected))

    def test_scoping_restricts_to_named_codes(self, harmonic):
        positions = jnp.zeros((6, 3))
        types = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
        state = _make_state(positions, types, 0.0)
        step_fn = swap.build_kernel(harmonic, n_species=3, species=(0, 2))

        proposed = _proposals(step_fn, state, 500)

        changed = proposed != types
        assert bool(jnp.all(jnp.sum(changed, axis=1) == 2))
        # Species 1 sits at indices 2 and 3 and must never be touched.
        assert not bool(jnp.any(changed[:, 2:4]))

    def test_preserves_composition(self, harmonic, positions, types):
        state = _initial_state(harmonic, positions, types)
        step_fn = jax.jit(self.build(harmonic))

        new_state, infos = _run(step_fn, state, range(50))

        assert all(bool(i.accepted) for i in infos)
        assert all(int(i.reject_reason) == 0 for i in infos)
        assert int(jnp.sum(new_state.types == 0)) == 2
        assert int(jnp.sum(new_state.types == 1)) == 2
        assert jnp.array_equal(new_state.positions, state.positions)
        assert jnp.array_equal(new_state.cell, state.cell)

    def test_nan_energy_is_rejected(self, positions, types):
        """A NaN energy is rejected and leaves the state unchanged.

        Written as ``E >= Emax -> reject``, the test would accept NaN.
        """
        backend = _ConstantBackend(energy=jnp.nan)
        state = _make_state(positions, types, 1.0)
        step_fn = jax.jit(self.build(backend))

        new_state, info = step_fn(jax.random.key(0), state, 1e6)

        assert not bool(info.accepted)
        assert bool(jnp.all(new_state.types == types))
        assert bool(jnp.isfinite(new_state.energy))

    def test_scope_with_absent_species_rejects(
        self, harmonic, positions, types
    ):
        """A scoped species missing from the walker leaves no pair to swap."""
        state = _initial_state(harmonic, positions, types)
        step_fn = jax.jit(
            swap.build_kernel(harmonic, n_species=3, species=(0, 2))
        )

        new_state, infos = _run(step_fn, state, range(20))

        assert all(not bool(i.accepted) for i in infos)
        assert bool(jnp.all(new_state.types == types))

    def test_no_unlike_pair_present_rejects_cleanly(self, harmonic, positions):
        """A walker that lost its second species must not NaN or mutate."""
        types = jnp.zeros((4,), dtype=jnp.int32)
        state = _initial_state(harmonic, positions, types)
        step_fn = jax.jit(self.build(harmonic))

        new_state, infos = _run(step_fn, state, range(20))

        assert all(not bool(i.accepted) for i in infos)
        assert bool(jnp.all(new_state.types == 0))
        assert not bool(new_state.overflow)

    def test_spec_builds_working_kernel(self, harmonic):
        """A scoped spec resolves to a kernel that leaves other species."""
        symbols = {0: "Ge", 1: "Si", 2: "Sn"}
        desc = SpeciesSwapMoveSpec(species=("Ge", "Sn")).to_descriptor(
            n_atoms=6, symbol_map=symbols
        )
        step_fn = desc.build_kernel(harmonic, **desc.kernel_kwargs)
        types = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
        state = _make_state(jnp.zeros((6, 3)), types, 0.0)

        proposed = _proposals(step_fn, state, 200)

        changed = proposed != types
        assert bool(jnp.all(jnp.sum(changed, axis=1) == 2))
        assert not bool(jnp.any(changed[:, 2:4]))  # Si untouched
