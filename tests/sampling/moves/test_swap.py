"""Tests for the species-swap move kernel (``moves/swap.py``).

Covers what distinguishes it from ``single_atom.build_swap_kernel``:
- every proposal is an unlike pair (no wasted evaluation)
- the atom pair is *uniform* over unlike pairs, including ternary systems
- composition is preserved, rejection restores the original types
- NaN energies reject rather than poisoning the walker
- species scoping restricts the exchange to the named codes
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.moves import swap
from jaxrens.state.mc_state import MCState


def _make_state(positions, types, energy, step_size=0.1):
    return MCState(
        positions=jnp.asarray(positions),
        types=jnp.asarray(types),
        energy=jnp.asarray(energy),
        cell=jnp.zeros((3, 3)),
        step_size=jnp.asarray(step_size),
        step_sizes=jnp.array([step_size]),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
    )


class _ConstantBackend:
    """Backend returning a fixed energy, ignoring the configuration.

    Used for the NaN-rejection guard, where the energy must be NaN
    regardless of which pair the kernel happened to pick.
    """

    def __init__(self, energy=0.0):
        self.r_cutoff = 0.0
        self.energy = energy

    def __call__(
        self, positions, species, cell, max_neighbors=0, ensemble_params=None
    ):
        return BackendResult(energy=jnp.asarray(self.energy))


def _proposals(step_fn, state, n_draws, seed=0):
    """Return the proposed ``types`` for ``n_draws`` independent draws.

    Reading the proposal off the *accepted* state (rather than recording it
    from inside the backend) keeps this jit- and vmap-safe: a Python-side
    recorder would only ever fire once, at trace time.  The caller must
    therefore use an unconditionally-accepting constraint.
    """
    keys = jax.random.split(jax.random.key(seed), n_draws)
    new_states, info = jax.jit(jax.vmap(step_fn, in_axes=(0, None, None)))(
        keys, state, 1e6
    )
    assert bool(
        jnp.all(info.accepted)
    ), "helper assumes every draw is accepted"
    return new_states.types


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


def _run(step_fn, state, seeds, emax=1e6):
    out = []
    for s in seeds:
        state, info = step_fn(jax.random.key(s), state, emax)
        out.append(info)
    return state, out


# ── Construction-time validation ────────────────────────────────────────────


class TestBuildValidation:
    def test_rejects_single_species_system(
        self,
    ):
        with pytest.raises(ValueError, match="n_species >= 2"):
            swap.build_kernel(create_harmonic(), n_species=1)

    def test_rejects_single_species_scope(self):
        with pytest.raises(ValueError, match="at least two"):
            swap.build_kernel(create_harmonic(), n_species=3, species=(1,))

    def test_rejects_duplicate_only_scope(self):
        with pytest.raises(ValueError, match="at least two"):
            swap.build_kernel(create_harmonic(), n_species=3, species=(2, 2))

    def test_rejects_out_of_range_code(self):
        with pytest.raises(ValueError, match="outside"):
            swap.build_kernel(create_harmonic(), n_species=2, species=(0, 5))


# ── Proposal correctness ────────────────────────────────────────────────────


class TestProposal:
    def test_every_proposal_is_an_unlike_pair(self, positions_4atom):
        """The whole point: no draw is spent on a same-species pair."""
        types = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        state = _make_state(positions_4atom, types, 0.0)
        step_fn = swap.build_kernel(create_harmonic(), n_species=2)

        proposed = _proposals(step_fn, state, 500)
        # A real swap changes types at exactly two positions.
        assert bool(jnp.all(jnp.sum(proposed != types, axis=1) == 2))

    def test_pair_selection_is_uniform_ternary(self):
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
        step_fn = swap.build_kernel(create_harmonic(), n_species=3)

        n_draws = 22000
        proposed = _proposals(step_fn, state, n_draws)

        pairs = Counter(
            tuple(int(x) for x in jnp.where(row != types)[0])
            for row in proposed
        )
        # 3*2 + 3*1 + 2*1 = 11 unlike pairs, each expected n_draws / 11.
        assert len(pairs) == 11
        expected = n_draws / 11
        for pair, count in pairs.items():
            assert count == pytest.approx(expected, rel=0.15), (
                pair,
                count,
                expected,
            )

    def test_scoping_restricts_to_named_codes(self):
        positions = jnp.zeros((6, 3))
        types = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
        state = _make_state(positions, types, 0.0)
        step_fn = swap.build_kernel(
            create_harmonic(), n_species=3, species=(0, 2)
        )

        proposed = _proposals(step_fn, state, 500)
        changed = proposed != types
        assert bool(jnp.all(jnp.sum(changed, axis=1) == 2))
        # Species 1 sits at indices 2 and 3 and must never be touched.
        assert not bool(jnp.any(changed[:, 2:4]))


# ── Accept / reject semantics ───────────────────────────────────────────────


class TestAcceptReject:
    def test_preserves_composition(self, positions_4atom):
        backend = create_harmonic()
        types = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        energy = backend(positions_4atom, types, jnp.zeros((3, 3)), 0).energy
        state = _make_state(positions_4atom, types, energy)
        step_fn = jax.jit(swap.build_kernel(backend, n_species=2))

        state, _ = _run(step_fn, state, range(50))
        assert int(jnp.sum(state.types == 0)) == 2
        assert int(jnp.sum(state.types == 1)) == 2

    def test_accepts_below_constraint(self, positions_4atom):
        backend = create_harmonic()
        types = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        energy = backend(positions_4atom, types, jnp.zeros((3, 3)), 0).energy
        state = _make_state(positions_4atom, types, energy)
        step_fn = jax.jit(swap.build_kernel(backend, n_species=2))

        _, infos = _run(step_fn, state, range(20), emax=1e6)
        assert all(bool(i.accepted) for i in infos)
        assert all(int(i.reject_reason) == 0 for i in infos)

    def test_rejects_above_constraint_and_restores_types(
        self, positions_4atom
    ):
        backend = create_harmonic()
        types = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        energy = backend(positions_4atom, types, jnp.zeros((3, 3)), 0).energy
        state = _make_state(positions_4atom, types, energy)
        step_fn = jax.jit(swap.build_kernel(backend, n_species=2))

        new_state, info = step_fn(jax.random.key(0), state, -1e6)
        assert not bool(info.accepted)
        assert int(info.reject_reason) == 1
        assert bool(jnp.all(new_state.types == types))
        assert new_state.energy == state.energy

    def test_nan_energy_is_rejected(self, positions_4atom):
        """Regression guard: ``E >= Emax -> reject`` would ACCEPT NaN here."""
        backend = _ConstantBackend(energy=jnp.nan)
        types = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        state = _make_state(positions_4atom, types, 1.0)
        step_fn = jax.jit(swap.build_kernel(backend, n_species=2))

        new_state, info = step_fn(jax.random.key(0), state, 1e6)
        assert not bool(info.accepted)
        assert bool(jnp.all(new_state.types == types))
        assert bool(jnp.isfinite(new_state.energy))

    def test_no_unlike_pair_present_rejects_cleanly(self, positions_4atom):
        """A walker that lost its second species must not NaN or mutate."""
        backend = create_harmonic()
        types = jnp.zeros((4,), dtype=jnp.int32)
        energy = backend(positions_4atom, types, jnp.zeros((3, 3)), 0).energy
        state = _make_state(positions_4atom, types, energy)
        step_fn = jax.jit(swap.build_kernel(backend, n_species=2))

        new_state, infos = _run(step_fn, state, range(20))
        assert all(not bool(i.accepted) for i in infos)
        assert bool(jnp.all(new_state.types == 0))
        assert not bool(new_state.overflow)


# ── Transform compatibility ─────────────────────────────────────────────────


class TestTransforms:
    def test_vmap_over_walkers(self):
        backend = create_harmonic()
        n_walkers = 8
        positions = jax.random.normal(jax.random.key(0), (n_walkers, 4, 3))
        types = jnp.broadcast_to(
            jnp.array([0, 0, 1, 1], dtype=jnp.int32), (n_walkers, 4)
        )
        batch = MCState(
            positions=positions,
            types=types,
            energy=jnp.zeros(n_walkers),
            cell=jnp.zeros((n_walkers, 3, 3)),
            step_size=jnp.full(n_walkers, 0.1),
            step_sizes=jnp.full((n_walkers, 1), 0.1),
            n_accepted=jnp.zeros((n_walkers, 1), dtype=jnp.int32),
            n_proposed=jnp.zeros((n_walkers, 1), dtype=jnp.int32),
            max_neighbor_count=jnp.zeros(n_walkers, dtype=jnp.int32),
            overflow=jnp.full(n_walkers, False),
            ensemble_params={},
        )
        step_fn = swap.build_kernel(backend, n_species=2)
        keys = jax.random.split(jax.random.key(1), n_walkers)
        new_batch, info = jax.jit(jax.vmap(step_fn, in_axes=(0, 0, None)))(
            keys, batch, 1e6
        )

        assert new_batch.types.shape == (n_walkers, 4)
        assert bool(jnp.all(jnp.sum(new_batch.types == 1, axis=1) == 2))
        assert info.accepted.shape == (n_walkers,)

    def test_scan(self, positions_4atom):
        backend = create_harmonic()
        types = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        energy = backend(positions_4atom, types, jnp.zeros((3, 3)), 0).energy
        state = _make_state(positions_4atom, types, energy)
        step_fn = swap.build_kernel(backend, n_species=2)

        def body(carry, key):
            new_state, info = step_fn(key, carry, 1e6)
            return new_state, info.accepted

        final, accepted = jax.lax.scan(
            body, state, jax.random.split(jax.random.key(3), 25)
        )
        assert accepted.shape == (25,)
        assert int(jnp.sum(final.types == 1)) == 2


# ── CLI spec wiring ─────────────────────────────────────────────────────────


class TestSpec:
    """``SpeciesSwapMoveSpec`` -> ``MoveKernel`` plumbing.

    Kept next to the kernel because the spec's job is to turn element
    symbols into the type codes this kernel consumes; a mismatch here is a
    kernel bug wearing a config hat.
    """

    SYMBOLS = {0: "Ge", 1: "Si", 2: "Sn"}

    def test_unscoped_descriptor(self):
        from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec

        desc = SpeciesSwapMoveSpec().to_descriptor(
            n_atoms=6, symbol_map=self.SYMBOLS
        )
        assert desc.name == "species_swap"
        assert desc.kernel_kwargs == {"n_species": 3}
        assert desc.mutates == frozenset({"types"})

    def test_scoped_descriptor_resolves_symbols_to_codes(self):
        from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec

        desc = SpeciesSwapMoveSpec(species=("Si", "Sn")).to_descriptor(
            n_atoms=6, symbol_map=self.SYMBOLS
        )
        assert desc.name == "species_swap_Si_Sn"
        assert desc.kernel_kwargs == {"n_species": 3, "species": (1, 2)}

    def test_descriptor_builds_a_working_kernel(self):
        from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec

        desc = SpeciesSwapMoveSpec(species=("Ge", "Sn")).to_descriptor(
            n_atoms=6, symbol_map=self.SYMBOLS
        )
        step_fn = desc.build_kernel(create_harmonic(), **desc.kernel_kwargs)
        types = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
        state = _make_state(jnp.zeros((6, 3)), types, 0.0)

        proposed = _proposals(step_fn, state, 200)
        changed = proposed != types
        assert bool(jnp.all(jnp.sum(changed, axis=1) == 2))
        assert not bool(jnp.any(changed[:, 2:4]))  # Si untouched

    def test_bare_symbol_shorthand_rejected_at_parse_time(self):
        from pydantic import ValidationError

        from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec

        with pytest.raises(ValidationError, match="at least two elements"):
            SpeciesSwapMoveSpec(species="Ge")

    def test_single_distinct_element_rejected_at_parse_time(self):
        from pydantic import ValidationError

        from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec

        with pytest.raises(ValidationError, match="distinct element"):
            SpeciesSwapMoveSpec(species=("Ge", "Ge"))

    def test_absent_element_rejected(self):
        from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec

        with pytest.raises(ValueError, match="not present in the system"):
            SpeciesSwapMoveSpec(species=("Ge", "Pu")).to_descriptor(
                n_atoms=6, symbol_map=self.SYMBOLS
            )

    def test_missing_symbol_map_rejected(self):
        from jaxrens.cli.schema.moves import SpeciesSwapMoveSpec

        with pytest.raises(ValueError, match="requires symbol_map"):
            SpeciesSwapMoveSpec().to_descriptor(n_atoms=6, symbol_map=None)
