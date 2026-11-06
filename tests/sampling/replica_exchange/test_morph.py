"""Tests for jaxrens.sampling.morph.morph_types_to_composition.

The function relabels atom types so that their counts match a target
composition. The tests check the composition and shape of the output,
determinism, JIT and vmap compatibility, and edge cases such as an identity
target or a target with a single species.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.morph import morph_types_to_composition


def make_types(key, n_atoms, n_species):
    """Random types array in [0, n_species-1] of shape (n_atoms,)."""
    return jax.random.randint(key, (n_atoms,), 0, n_species)


def make_target(key, n_atoms, n_species):
    """Random target composition summing to n_atoms."""
    # Sorted random cut points split n_atoms into n_species non-negative
    # parts. Boundaries are 0, cuts[0], ..., cuts[n_species-2], n_atoms.
    cuts = jnp.sort(jax.random.randint(key, (n_species - 1,), 0, n_atoms + 1))
    boundaries = jnp.concatenate([jnp.array([0]), cuts, jnp.array([n_atoms])])
    return jnp.diff(boundaries).astype(jnp.int32)


class TestDeterminism:
    """Check that the output depends only on the key and the inputs."""

    def test_same_key_same_output(self):
        key = jax.random.key(7)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms, n_species = 10, 3
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        r1 = morph_types_to_composition(k3, types, target, n_species)
        r2 = morph_types_to_composition(k3, types, target, n_species)
        assert jnp.array_equal(r1, r2)

    def test_different_keys_differ(self):
        key = jax.random.key(13)
        k1, _, k3a, k3b = jax.random.split(key, 4)
        n_atoms, n_species = 20, 3
        # Going from [5, 8, 7] to [10, 5, 5] picks 3 of 8 species-1 atoms
        # and 2 of 7 species-2 atoms, so two keys almost never agree.
        types = make_types(k1, n_atoms, n_species)
        target = jnp.array([10, 5, 5], dtype=jnp.int32)
        r1 = morph_types_to_composition(k3a, types, target, n_species)
        r2 = morph_types_to_composition(k3b, types, target, n_species)
        for r in (r1, r2):
            got = jnp.bincount(r, length=n_species)
            assert jnp.array_equal(got, target)
        assert not jnp.array_equal(r1, r2)


def test_jit_output_matches_eager():
    key = jax.random.key(5)
    k1, k2, k3 = jax.random.split(key, 3)
    n_atoms, n_species = 18, 3
    types = make_types(k1, n_atoms, n_species)
    target = make_target(k2, n_atoms, n_species)

    eager = morph_types_to_composition(k3, types, target, n_species)
    jit_fn = jax.jit(morph_types_to_composition, static_argnums=(3,))
    jitted = jit_fn(k3, types, target, n_species)
    assert jnp.array_equal(eager, jitted)


class TestVmap:
    """Check that a vmapped morph gives each replica its own target."""

    def test_vmap_composition_invariant(self):
        key = jax.random.key(10)
        n_replicas, n_atoms, n_species = 4, 12, 3

        keys = jax.random.split(key, n_replicas * 3).reshape(n_replicas, 3)
        types_batch = jax.vmap(lambda k: make_types(k, n_atoms, n_species))(
            keys[:, 0]
        )
        target_batch = jax.vmap(lambda k: make_target(k, n_atoms, n_species))(
            keys[:, 1]
        )
        morph_keys = keys[:, 2]

        vmapped_morph = jax.vmap(
            morph_types_to_composition, in_axes=(0, 0, 0, None)
        )
        results = vmapped_morph(
            morph_keys, types_batch, target_batch, n_species
        )

        assert results.shape == (n_replicas, n_atoms)
        for i in range(n_replicas):
            got = jnp.bincount(results[i], length=n_species)
            assert jnp.array_equal(got, target_batch[i]), f"replica {i}"

    def test_vmap_jit_combined(self):
        """Same as above, with the vmapped function wrapped in jax.jit."""
        key = jax.random.key(11)
        n_replicas, n_atoms, n_species = 3, 10, 2

        keys = jax.random.split(key, n_replicas * 3).reshape(n_replicas, 3)
        types_batch = jax.vmap(lambda k: make_types(k, n_atoms, n_species))(
            keys[:, 0]
        )
        target_batch = jax.vmap(lambda k: make_target(k, n_atoms, n_species))(
            keys[:, 1]
        )
        morph_keys = keys[:, 2]

        vmapped_morph = jax.vmap(
            morph_types_to_composition, in_axes=(0, 0, 0, None)
        )
        jit_vmapped = jax.jit(vmapped_morph, static_argnums=(3,))
        results = jit_vmapped(morph_keys, types_batch, target_batch, n_species)

        assert results.shape == (n_replicas, n_atoms)
        for i in range(n_replicas):
            got = jnp.bincount(results[i], length=n_species)
            assert jnp.array_equal(got, target_batch[i]), f"replica {i}"


class TestEdgeCases:
    """Check identity, single-species and minimal-size targets."""

    def test_identity_target_leaves_types_unchanged(self):
        """A target equal to the current composition leaves types unchanged."""
        key = jax.random.key(20)
        k1, k2 = jax.random.split(key)
        n_atoms, n_species = 12, 3
        types = make_types(k1, n_atoms, n_species)
        target = jnp.bincount(types, length=n_species)
        result = morph_types_to_composition(k2, types, target, n_species)
        assert jnp.array_equal(result, types)

    def test_single_species_target(self):
        """All atoms converted to species 0."""
        key = jax.random.key(21)
        n_atoms, n_species = 8, 3
        types = jnp.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=jnp.int32)
        target = jnp.array([n_atoms, 0, 0], dtype=jnp.int32)
        result = morph_types_to_composition(key, types, target, n_species)
        assert jnp.all(result == 0)
        assert result.shape == types.shape

    def test_extreme_skew_all_to_one_species(self):
        """All atoms converted to species 2, a non-zero index."""
        key = jax.random.key(22)
        n_atoms, n_species = 10, 4
        types = jax.random.randint(key, (n_atoms,), 0, n_species)
        target = jnp.array([0, 0, n_atoms, 0], dtype=jnp.int32)
        k2 = jax.random.fold_in(key, 1)
        result = morph_types_to_composition(k2, types, target, n_species)
        assert jnp.all(result == 2)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)

    def test_single_atom_single_species(self):
        key = jax.random.key(24)
        types = jnp.array([0], dtype=jnp.int32)
        target = jnp.array([1], dtype=jnp.int32)
        n_species = 1
        result = morph_types_to_composition(key, types, target, n_species)
        assert result.shape == (1,)
        assert int(result[0]) == 0

    def test_two_species_partial_relabel(self):
        """Going from 4/2 to 2/4 relabels two species-0 atoms as species 1."""
        key = jax.random.key(25)
        n_species = 2
        types = jnp.array([0, 0, 0, 0, 1, 1], dtype=jnp.int32)
        target = jnp.array([2, 4], dtype=jnp.int32)
        result = morph_types_to_composition(key, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)


class TestPropertyBased:
    """Check shape and composition for a grid of sizes and random seeds."""

    @pytest.mark.parametrize(
        "n_atoms,n_species",
        [
            (1, 1),
            (4, 2),
            (9, 3),
            (16, 4),
            (25, 5),
            (30, 6),
            (50, 3),
            (64, 2),
            (100, 5),
        ],
    )
    def test_composition_invariant_parametrized(self, n_atoms, n_species):
        key = jax.random.key(n_atoms + n_species * 1000)
        k1, k2, k3 = jax.random.split(key, 3)
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        result = morph_types_to_composition(k3, types, target, n_species)
        assert result.shape == (n_atoms,)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)

    @pytest.mark.parametrize("seed", range(8))
    def test_composition_invariant_random_targets(self, seed):
        """Random targets for a fixed 20-atom, 4-species system."""
        n_atoms, n_species = 20, 4
        key = jax.random.key(seed * 997)
        k1, k2, k3 = jax.random.split(key, 3)
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        result = morph_types_to_composition(k3, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)
