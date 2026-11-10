"""Species-scoped single-atom move: kernel selection and schema wiring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.adaptation import AdaptationSpec
from jaxrens.cli.schema.moves import GMCMoveSpec, SingleAtomMoveSpec
from jaxrens.sampling.moves import single_atom
from jaxrens.state.mc_state import MCState

# Atoms 0, 2, 4 are species 0 ("C"); atoms 1, 3 are species 1 ("Li").
_TYPES = jnp.array([0, 1, 0, 1, 0], dtype=jnp.int32)
_POSITIONS = jnp.array(
    [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.3, 0.3, 0.0],
        [0.0, 0.3, 0.3],
    ]
)
_SYMBOL_MAP = {0: "C", 1: "Li"}
_LI = (1,)


def _make_state(types=_TYPES, step_size=0.1):
    return MCState(
        positions=_POSITIONS,
        types=types,
        energy=jnp.asarray(0.5 * jnp.sum(_POSITIONS**2)),
        cell=jnp.zeros((3, 3)),
        step_size=jnp.asarray(step_size),
        step_sizes=jnp.array([step_size]),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
    )


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


def _moved_atoms(step_fn, state, n_draws, seed=0):
    """Index of the displaced atom in each of ``n_draws`` accepted steps."""
    keys = jax.random.split(jax.random.key(seed), n_draws)
    new_states, infos = jax.vmap(lambda k: step_fn(k, state, 1e6))(keys)
    assert bool(jnp.all(infos.accepted))
    changed = jnp.any(new_states.positions != state.positions, axis=-1)
    assert bool(jnp.all(changed.sum(axis=-1) == 1))
    return np.asarray(jnp.argmax(changed, axis=-1))


class TestKernel:
    def test_only_scoped_atoms_move_uniformly(self, harmonic):
        step_fn = jax.jit(single_atom.build_kernel(harmonic, species=_LI))
        moved = _moved_atoms(step_fn, _make_state(), 4000)
        counts = np.bincount(moved, minlength=5)
        assert counts[0] == counts[2] == counts[4] == 0
        # Both Li picked with equal probability.
        assert stats.chisquare(counts[[1, 3]]).pvalue > 1e-3

    def test_multiple_species_in_one_scope(self, harmonic):
        step_fn = jax.jit(single_atom.build_kernel(harmonic, species=(0, 1)))
        moved = _moved_atoms(step_fn, _make_state(), 5000, seed=1)
        counts = np.bincount(moved, minlength=5)
        assert stats.chisquare(counts).pvalue > 1e-3

    def test_selection_follows_current_types(self, harmonic):
        types = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        step_fn = jax.jit(single_atom.build_kernel(harmonic, species=_LI))
        moved = _moved_atoms(step_fn, _make_state(types=types), 50)
        assert np.all(moved == 0)

    def test_walker_without_species_rejects(self, harmonic):
        state = _make_state(types=jnp.zeros(5, dtype=jnp.int32))
        step_fn = jax.jit(single_atom.build_kernel(harmonic, species=_LI))
        new_state, info = step_fn(jax.random.key(3), state, 1e6)
        assert not bool(info.accepted)
        np.testing.assert_array_equal(new_state.positions, state.positions)
        assert new_state.energy == state.energy

    def test_unscoped_matches_default_kernel(self, harmonic):
        state = _make_state()
        legacy = jax.jit(single_atom.build_kernel(harmonic))
        unscoped = jax.jit(single_atom.build_kernel(harmonic, species=None))
        for seed in range(5):
            key = jax.random.key(seed)
            a, info_a = legacy(key, state, 0.3)
            b, info_b = unscoped(key, state, 0.3)
            np.testing.assert_array_equal(a.positions, b.positions)
            assert bool(info_a.accepted) == bool(info_b.accepted)


class TestSchema:
    def test_bare_string_species_is_wrapped(self):
        assert SingleAtomMoveSpec(species="Li").species == ("Li",)
        assert SingleAtomMoveSpec(species=["Li"]).species == ("Li",)

    def test_symbols_resolve_to_type_codes(self):
        spec = SingleAtomMoveSpec(species="Li")
        assert spec._kernel_kwargs(symbol_map=_SYMBOL_MAP) == {"species": (1,)}

    def test_auto_name_and_explicit_name(self):
        assert (
            SingleAtomMoveSpec(species="Li")._effective_name()
            == "single_atom_Li"
        )
        assert (
            SingleAtomMoveSpec(species=["C", "Li"])._effective_name()
            == "single_atom_C_Li"
        )
        assert (
            SingleAtomMoveSpec(species="Li", name="li_local")._effective_name()
            == "li_local"
        )

    def test_unscoped_spec_is_unchanged(self):
        desc = SingleAtomMoveSpec().to_descriptor()
        assert desc.name == "single_atom"
        assert desc.kernel_kwargs == {}

    def test_auto_name_keys_per_move_adaptation(self):
        desc = SingleAtomMoveSpec(species="Li").to_descriptor(
            symbol_map=_SYMBOL_MAP
        )
        adaptation = AdaptationSpec(
            per_move={"single_atom_Li": {"step_size_max": 10.0}}
        )
        assert adaptation.resolve_for(desc.name).step_size_max == 10.0

    def test_unknown_species_raises(self):
        spec = SingleAtomMoveSpec(species="Xe")
        with pytest.raises(ValueError, match="not present in the system"):
            spec.to_descriptor(symbol_map=_SYMBOL_MAP)

    def test_missing_symbol_map_raises(self):
        spec = SingleAtomMoveSpec(species="Li")
        with pytest.raises(ValueError, match="SingleAtomMoveSpec.*symbol_map"):
            spec.to_descriptor()

    def test_gmc_still_uses_shared_resolution(self):
        spec = GMCMoveSpec(species="Li")
        assert spec._species_codes(_SYMBOL_MAP) == (1,)
        with pytest.raises(ValueError, match="gmc species"):
            GMCMoveSpec(species="Xe").to_descriptor(symbol_map=_SYMBOL_MAP)
