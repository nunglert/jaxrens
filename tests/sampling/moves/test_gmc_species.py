"""Species-scoped Galilean Monte Carlo: sublattice masking and wiring.

Covers the three things that make per-sublattice GMC work:

1. the kernel mask (frozen atoms never move, in *any* code path — accept,
   reject, and reflection),
2. the per-move ``direction`` field (two scoped moves must not clobber each
   other's persistent direction),
3. the schema/resolver seam that turns ``species: [Ge]`` into type codes and
   distinct move names.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.moves import GMCMoveSpec
from jaxrens.sampling.moves.galilean import build_kernel as gmc_build_kernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.state.mc_state import make_mc_state_class

# Two-species toy system: atoms 0,1 are species 0; atoms 2,3 are species 1.
_TYPES = jnp.array([0, 0, 1, 1])
_POSITIONS = jnp.array(
    [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.3, 0.3, 0.0],
    ]
)
_SYMBOL_MAP = {0: "Ge", 1: "Si"}


def _make_state(
    cls, *, step_size=0.1, n_moves=1, direction_fields=("direction",)
):
    kwargs = dict(
        positions=_POSITIONS,
        types=_TYPES,
        energy=jnp.asarray(0.5 * jnp.sum(_POSITIONS**2)),
        cell=jnp.zeros((3, 3)),
        step_size=jnp.asarray(step_size),
        step_sizes=jnp.full(n_moves, step_size),
        n_accepted=jnp.zeros(n_moves, dtype=jnp.int32),
        n_proposed=jnp.zeros(n_moves, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
    )
    for name in direction_fields:
        kwargs[name] = jnp.zeros_like(_POSITIONS)
    return cls(**kwargs)


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


# ---------------------------------------------------------------------------
# Kernel-level masking
# ---------------------------------------------------------------------------


class TestSpeciesMask:
    @pytest.mark.parametrize(
        "constraint",
        [
            10.0,  # loose: trajectory stays in-region, move accepts
            0.55,  # tight: forces reflections and rejections
        ],
    )
    @pytest.mark.parametrize("use_forces", [True, False])
    def test_frozen_species_never_moves(
        self, harmonic, constraint, use_forces
    ):
        """Species-1 atoms must be bit-identical after many steps.

        Both the accept and the reject path are exercised by the two
        constraints; ``use_forces=False`` covers the random-reflection branch,
        which has its own mask application.
        """
        cls = make_mc_state_class({"direction": jnp.ndarray})
        state = _make_state(cls, step_size=0.2)
        step = jax.jit(
            gmc_build_kernel(
                harmonic,
                n_reflect=3,
                species=(0,),
                use_forces=use_forces,
            )
        )

        initial = state.positions
        for i in range(20):
            state, _ = step(jax.random.key(i), state, constraint)

        # Frozen sublattice: exactly unchanged, not merely close.
        assert jnp.array_equal(state.positions[2:], initial[2:])

    def test_moving_species_actually_moves(self, harmonic):
        cls = make_mc_state_class({"direction": jnp.ndarray})
        state = _make_state(cls, step_size=0.1)
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=3, species=(0,)))

        initial = state.positions
        for i in range(10):
            state, _ = step(jax.random.key(i), state, 10.0)

        assert not jnp.allclose(state.positions[:2], initial[:2])

    def test_direction_is_confined_to_the_subspace(self, harmonic):
        """Frozen rows of the direction stay exactly zero; moving rows are unit."""
        cls = make_mc_state_class({"direction": jnp.ndarray})
        state = _make_state(cls, step_size=0.1)
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=3, species=(0,)))

        for i in range(5):
            state, _ = step(jax.random.key(i), state, 10.0)
            assert jnp.array_equal(state.direction[2:], jnp.zeros((2, 3)))
            norm = jnp.sqrt(jnp.sum(state.direction**2))
            assert jnp.isclose(norm, 1.0, atol=1e-5)

    def test_multiple_species_in_one_scope(self, harmonic):
        """species=(0, 1) on a 2-species system is equivalent to unscoped."""
        cls = make_mc_state_class({"direction": jnp.ndarray})
        scoped = jax.jit(
            gmc_build_kernel(harmonic, n_reflect=3, species=(0, 1))
        )
        unscoped = jax.jit(gmc_build_kernel(harmonic, n_reflect=3))

        key = jax.random.key(7)
        a, _ = scoped(key, _make_state(cls, step_size=0.1), 10.0)
        b, _ = unscoped(key, _make_state(cls, step_size=0.1), 10.0)
        assert jnp.allclose(a.positions, b.positions)

    def test_unscoped_matches_legacy_behaviour(self, harmonic):
        """species=None must leave the original code path bit-identical."""
        cls = make_mc_state_class({"direction": jnp.ndarray})
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=3, species=None))
        state, info = step(jax.random.key(0), _make_state(cls), 10.0)
        assert state.positions.shape == _POSITIONS.shape
        assert info.n_evaluations == 3

    def test_custom_direction_field(self, harmonic):
        cls = make_mc_state_class({"direction_gmc_Ge": jnp.ndarray})
        state = _make_state(cls, direction_fields=("direction_gmc_Ge",))
        step = jax.jit(
            gmc_build_kernel(
                harmonic,
                n_reflect=3,
                species=(0,),
                direction_field="direction_gmc_Ge",
            )
        )
        new_state, _ = step(jax.random.key(0), state, 10.0)

        norm = jnp.sqrt(jnp.sum(new_state.direction_gmc_Ge**2))
        assert norm > 0.1

    def test_mask_follows_types_not_positions(self, harmonic):
        """The mask is read from ``state.types``, so a swap retargets the move.

        Swap/alchemical moves mutate types mid-run; a mask baked in at build
        time would keep displacing the original atom indices.
        """
        cls = make_mc_state_class({"direction": jnp.ndarray})
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=3, species=(0,)))

        # Reverse the species assignment: now atoms 2,3 are the movers.
        state = _make_state(cls, step_size=0.2).set(
            types=jnp.array([1, 1, 0, 0])
        )
        initial = state.positions
        for i in range(10):
            state, _ = step(jax.random.key(i), state, 10.0)

        assert jnp.array_equal(state.positions[:2], initial[:2])
        assert not jnp.allclose(state.positions[2:], initial[2:])


# ---------------------------------------------------------------------------
# MWG integration: two scoped moves side by side
# ---------------------------------------------------------------------------


class TestScopedMovesInMWG:
    def _descriptors(self):
        ge = GMCMoveSpec(species=["Ge"], step_size=0.3, n_reflect=3)
        si = GMCMoveSpec(species=["Si"], step_size=0.05, n_reflect=3)
        return [
            ge.to_descriptor(symbol_map=_SYMBOL_MAP),
            si.to_descriptor(symbol_map=_SYMBOL_MAP),
        ]

    def test_separate_direction_fields_and_step_sizes(self, harmonic):
        descs = self._descriptors()
        assert [d.name for d in descs] == ["gmc_Ge", "gmc_Si"]

        init_fn, step_fn, _ = build_mwg(harmonic, descs)
        state = init_fn(
            _POSITIONS,
            _TYPES,
            energy=0.5 * jnp.sum(_POSITIONS**2),
            cell=jnp.zeros((3, 3)),
        )

        # Distinct persistent-direction fields — the union in build_mwg must
        # not collapse them into one.
        assert hasattr(state, "direction_gmc_Ge")
        assert hasattr(state, "direction_gmc_Si")
        # Per-move step sizes, seeded from each spec.
        assert state.step_sizes.shape == (2,)
        assert jnp.allclose(state.step_sizes, jnp.array([0.3, 0.05]))

    def test_each_move_only_touches_its_own_sublattice(self, harmonic):
        init_fn, _, per_move_fns = build_mwg(harmonic, self._descriptors())
        state = init_fn(
            _POSITIONS,
            _TYPES,
            energy=0.5 * jnp.sum(_POSITIONS**2),
            cell=jnp.zeros((3, 3)),
        )
        ge_move, si_move = per_move_fns

        after_ge = ge_move(state, jax.random.key(1), 10.0)[0]
        assert jnp.array_equal(after_ge.positions[2:], state.positions[2:])
        # The Si move's direction is untouched by the Ge move.
        assert jnp.array_equal(
            after_ge.direction_gmc_Si, state.direction_gmc_Si
        )

        after_si = si_move(after_ge, jax.random.key(2), 10.0)[0]
        assert jnp.array_equal(after_si.positions[:2], after_ge.positions[:2])
        # ...and vice versa: Ge's accumulated direction survives the Si move.
        assert jnp.array_equal(
            after_si.direction_gmc_Ge, after_ge.direction_gmc_Ge
        )

    def test_both_sublattices_move_under_the_full_mwg_step(self, harmonic):
        init_fn, step_fn, _ = build_mwg(harmonic, self._descriptors())
        state = init_fn(
            _POSITIONS,
            _TYPES,
            energy=0.5 * jnp.sum(_POSITIONS**2),
            cell=jnp.zeros((3, 3)),
        )
        initial = state.positions

        jitted = jax.jit(step_fn)
        for i in range(40):
            state, _ = jitted(jax.random.key(i), state, 10.0)

        assert not jnp.allclose(state.positions[:2], initial[:2])
        assert not jnp.allclose(state.positions[2:], initial[2:])
        assert int(jnp.sum(state.n_proposed)) == 40


# ---------------------------------------------------------------------------
# Schema / resolver seam
# ---------------------------------------------------------------------------


class TestGMCSpeciesSchema:
    def test_bare_string_species_is_wrapped(self):
        assert GMCMoveSpec(species="Ge").species == ("Ge",)

    def test_symbols_resolve_to_type_codes(self):
        spec = GMCMoveSpec(species=["Si"])
        kwargs = spec._kernel_kwargs(symbol_map=_SYMBOL_MAP)
        assert kwargs["species"] == (1,)
        assert kwargs["direction_field"] == "direction_gmc_Si"
        assert kwargs["n_reflect"] == 5

    def test_auto_name_and_explicit_name(self):
        assert GMCMoveSpec(species=["Ge"])._effective_name() == "gmc_Ge"
        assert (
            GMCMoveSpec(species=["Ge", "Si"])._effective_name() == "gmc_Ge_Si"
        )
        assert (
            GMCMoveSpec(species=["Ge"], name="fast")._effective_name()
            == "fast"
        )
        # An explicit name still gets its own direction field when scoped.
        assert (
            GMCMoveSpec(species=["Ge"], name="fast")._direction_field()
            == "direction_fast"
        )

    def test_unscoped_spec_is_unchanged(self):
        spec = GMCMoveSpec()
        assert spec._effective_name() == "gmc"
        assert spec._direction_field() == "direction"
        # Unscoped kernel_kwargs stay exactly as they were before species
        # scoping existed — the kernel defaults supply the rest.
        assert spec._kernel_kwargs(symbol_map=_SYMBOL_MAP) == {"n_reflect": 5}
        assert "direction" in spec._extra_state_fields()

    def test_unknown_species_raises(self):
        spec = GMCMoveSpec(species=["Xe"])
        with pytest.raises(ValueError, match="not present in the system"):
            spec.to_descriptor(symbol_map=_SYMBOL_MAP)

    def test_missing_symbol_map_raises(self):
        spec = GMCMoveSpec(species=["Ge"])
        with pytest.raises(ValueError, match="requires symbol_map"):
            spec.to_descriptor()

    def test_unscoped_spec_needs_no_symbol_map(self):
        """The legacy setup_mwg()/MoveConfig path passes no symbol_map."""
        desc = GMCMoveSpec(n_reflect=4).to_descriptor()
        assert desc.name == "gmc"
        assert desc.kernel_kwargs == {"n_reflect": 4}
        assert "direction" in desc.extra_state_fields
