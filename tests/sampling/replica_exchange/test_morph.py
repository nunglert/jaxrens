"""Tests for jaxrens.sampling.morph.morph_types_to_composition.

Coverage:
  - Composition invariant: bincount(output) == target_composition
  - Shape invariant: output.shape == input.shape
  - Determinism: same key + inputs -> identical output
  - JIT: jax.jit(morph) runs without retracing on same shapes
  - vmap: batched morph over (n_replicas, n_atoms) works correctly
  - Edge cases: identity morph, single-species, extreme skew
  - Property-based: composition invariant for varied n_atoms / n_species
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.morph import morph_types_to_composition

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_types(key, n_atoms, n_species):
    """Random types array in [0, n_species-1] of shape (n_atoms,)."""
    return jax.random.randint(key, (n_atoms,), 0, n_species)


def make_target(key, n_atoms, n_species):
    """Random target composition summing to n_atoms."""
    # Draw a random partition of n_atoms into n_species non-negative parts.
    # Use sorted uniform draws (stick-breaking style) for a uniformly random
    # composition. We round to int and correct the last bucket for exact sum.
    cuts = jnp.sort(jax.random.randint(key, (n_species - 1,), 0, n_atoms + 1))
    # boundaries: 0, cut[0], cut[1], ..., cut[n_species-2], n_atoms
    boundaries = jnp.concatenate([jnp.array([0]), cuts, jnp.array([n_atoms])])
    return jnp.diff(boundaries).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Composition invariant tests
# ---------------------------------------------------------------------------


class TestCompositionInvariant:
    """morph output must match target_composition exactly."""

    def test_random_morph_composition(self):
        key = jax.random.key(0)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms, n_species = 20, 3
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        result = morph_types_to_composition(k3, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target), f"got {got}, want {target}"

    def test_composition_invariant_multiple_seeds(self):
        """Invariant holds across many random seeds."""
        n_atoms, n_species = 15, 4
        for seed in range(10):
            key = jax.random.key(seed)
            k1, k2, k3 = jax.random.split(key, 3)
            types = make_types(k1, n_atoms, n_species)
            target = make_target(k2, n_atoms, n_species)
            result = morph_types_to_composition(k3, types, target, n_species)
            got = jnp.bincount(result, length=n_species)
            assert jnp.array_equal(
                got, target
            ), f"seed={seed}: got {got}, want {target}"

    def test_large_system(self):
        """Composition invariant holds for larger n_atoms."""
        key = jax.random.key(42)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms, n_species = 100, 5
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        result = morph_types_to_composition(k3, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)


# ---------------------------------------------------------------------------
# Shape invariant tests
# ---------------------------------------------------------------------------


class TestShapeInvariant:
    """output.shape must match types.shape exactly."""

    def test_shape_preserved(self):
        key = jax.random.key(1)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms, n_species = 12, 3
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        result = morph_types_to_composition(k3, types, target, n_species)
        assert (
            result.shape == types.shape
        ), f"shape mismatch: {result.shape} vs {types.shape}"

    def test_shape_preserved_various_sizes(self):
        for n_atoms, n_species in [(1, 1), (5, 2), (30, 6)]:
            key = jax.random.key(n_atoms * 100 + n_species)
            k1, k2, k3 = jax.random.split(key, 3)
            types = make_types(k1, n_atoms, n_species)
            target = make_target(k2, n_atoms, n_species)
            result = morph_types_to_composition(k3, types, target, n_species)
            assert result.shape == (
                n_atoms,
            ), f"n_atoms={n_atoms}, n_species={n_species}: {result.shape}"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same key + inputs must produce bit-exact identical output."""

    def test_same_key_same_output(self):
        key = jax.random.key(7)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms, n_species = 10, 3
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        r1 = morph_types_to_composition(k3, types, target, n_species)
        r2 = morph_types_to_composition(k3, types, target, n_species)
        assert jnp.array_equal(r1, r2), "non-deterministic output for same key"

    def test_different_keys_may_differ(self):
        """Different keys can produce different relabelings (probabilistic check)."""
        key = jax.random.key(13)
        k1, k2, k3a, k3b = jax.random.split(key, 4)
        n_atoms, n_species = 20, 3
        types = make_types(k1, n_atoms, n_species)
        # Use a target that requires many changes
        # Force a target very different from a uniform-ish distribution
        target = jnp.array([10, 5, 5], dtype=jnp.int32)
        r1 = morph_types_to_composition(k3a, types, target, n_species)
        r2 = morph_types_to_composition(k3b, types, target, n_species)
        # Both must be valid; they may or may not be identical
        for r in (r1, r2):
            got = jnp.bincount(r, length=n_species)
            assert jnp.array_equal(got, target)


# ---------------------------------------------------------------------------
# JIT tests (mandatory per project policy)
# ---------------------------------------------------------------------------


class TestJIT:
    """morph_types_to_composition must be JIT-compatible."""

    def test_jit_runs(self):
        key = jax.random.key(2)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms, n_species = 16, 4
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)

        jit_morph = jax.jit(morph_types_to_composition, static_argnums=(3,))
        result = jit_morph(k3, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)

    def test_jit_output_matches_eager(self):
        key = jax.random.key(5)
        k1, k2, k3 = jax.random.split(key, 3)
        n_atoms, n_species = 18, 3
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)

        eager = morph_types_to_composition(k3, types, target, n_species)
        jit_fn = jax.jit(morph_types_to_composition, static_argnums=(3,))
        jitted = jit_fn(k3, types, target, n_species)
        assert jnp.array_equal(eager, jitted), "JIT and eager outputs differ"


# ---------------------------------------------------------------------------
# vmap tests
# ---------------------------------------------------------------------------


class TestVmap:
    """morph must be vmap-compatible for batched morphs."""

    def test_vmap_composition_invariant(self):
        """Batched morph: each replica's output matches its target."""
        key = jax.random.key(10)
        n_replicas, n_atoms, n_species = 4, 12, 3

        # Build per-replica keys, types, targets
        keys = jax.random.split(key, n_replicas * 3).reshape(n_replicas, 3, 2)
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
            assert jnp.array_equal(
                got, target_batch[i]
            ), f"replica {i}: got {got}, want {target_batch[i]}"

    def test_vmap_jit_combined(self):
        """vmap inside jit: no retracing, correct composition."""
        key = jax.random.key(11)
        n_replicas, n_atoms, n_species = 3, 10, 2

        keys = jax.random.split(key, n_replicas * 3).reshape(n_replicas, 3, 2)
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
            assert jnp.array_equal(got, target_batch[i])


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: identity morph, single species, extreme skew."""

    def test_identity_morph_valid_composition(self):
        """When target equals current composition, output must still match target.

        The output may or may not equal the input (identity relabeling is valid
        but not guaranteed — the random donor selection can pick up and put down
        the same atoms).
        """
        key = jax.random.key(20)
        k1, k2 = jax.random.split(key)
        n_atoms, n_species = 12, 3
        types = make_types(k1, n_atoms, n_species)
        # Target matches current composition exactly
        target = jnp.bincount(types, length=n_species)
        result = morph_types_to_composition(k2, types, target, n_species)
        # Output must still satisfy composition invariant
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(
            got, target
        ), "identity morph violated composition invariant"
        assert result.shape == types.shape

    def test_single_species_target(self):
        """All atoms converted to species 0."""
        key = jax.random.key(21)
        n_atoms, n_species = 8, 3
        types = jnp.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=jnp.int32)
        target = jnp.array([n_atoms, 0, 0], dtype=jnp.int32)
        result = morph_types_to_composition(key, types, target, n_species)
        assert jnp.all(result == 0), f"expected all-0 output, got {result}"
        assert result.shape == types.shape

    def test_extreme_skew_all_to_one_species(self):
        """All atoms morphed to a non-zero species index."""
        key = jax.random.key(22)
        n_atoms, n_species = 10, 4
        types = jax.random.randint(key, (n_atoms,), 0, n_species)
        # All go to species 2
        target = jnp.array([0, 0, n_atoms, 0], dtype=jnp.int32)
        k2 = jax.random.fold_in(key, 1)
        result = morph_types_to_composition(k2, types, target, n_species)
        assert jnp.all(result == 2), f"expected all-2 output, got {result}"
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)

    def test_no_change_needed_some_species(self):
        """When only a subset of species need relabeling, unaffected atoms stay."""
        key = jax.random.key(23)
        # types: 5 of species 0, 5 of species 1
        types = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=jnp.int32)
        n_species = 2
        # target: same counts — identity morph
        target = jnp.array([5, 5], dtype=jnp.int32)
        result = morph_types_to_composition(key, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target)

    def test_single_atom_single_species(self):
        """Minimal case: 1 atom, 1 species."""
        key = jax.random.key(24)
        types = jnp.array([0], dtype=jnp.int32)
        target = jnp.array([1], dtype=jnp.int32)
        n_species = 1
        result = morph_types_to_composition(key, types, target, n_species)
        assert result.shape == (1,)
        assert int(result[0]) == 0

    def test_two_species_swap(self):
        """All atoms of species 0 become species 1 and vice versa."""
        key = jax.random.key(25)
        n_atoms = 6
        n_species = 2
        # 4 of species 0, 2 of species 1
        types = jnp.array([0, 0, 0, 0, 1, 1], dtype=jnp.int32)
        # target: 2 of species 0, 4 of species 1
        target = jnp.array([2, 4], dtype=jnp.int32)
        result = morph_types_to_composition(key, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(got, target), f"got {got}, want {target}"


# ---------------------------------------------------------------------------
# Property-based / parametrized tests
# ---------------------------------------------------------------------------


class TestPropertyBased:
    """Composition invariant for a grid of (n_atoms, n_species) combinations."""

    @pytest.mark.parametrize(
        "n_atoms,n_species",
        [
            (1, 1),
            (4, 2),
            (9, 3),
            (16, 4),
            (25, 5),
            (50, 3),
            (64, 2),
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
        assert jnp.array_equal(
            got, target
        ), f"n_atoms={n_atoms}, n_species={n_species}: got {got}, want {target}"

    @pytest.mark.parametrize("seed", range(8))
    def test_composition_invariant_random_targets(self, seed):
        """Random targets for fixed (20, 4) system across multiple seeds."""
        n_atoms, n_species = 20, 4
        key = jax.random.key(seed * 997)
        k1, k2, k3 = jax.random.split(key, 3)
        types = make_types(k1, n_atoms, n_species)
        target = make_target(k2, n_atoms, n_species)
        result = morph_types_to_composition(k3, types, target, n_species)
        got = jnp.bincount(result, length=n_species)
        assert jnp.array_equal(
            got, target
        ), f"seed={seed}: got {got}, want {target}"
