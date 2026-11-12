"""Tests for ``pad_to_multiple`` in ``jaxrens.utils.padding``.

It pads the leading axis of a pytree up to a multiple of the ``lax.map``
batch size; ``init/burn_in.py`` and ``adaptation/stepsize_handler.py`` call it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.utils.padding import pad_to_multiple


class TestPadArithmetic:
    """Check that ``n_pad`` and the padded length are right, including an
    empty axis and a chunk larger than the axis."""

    @pytest.mark.parametrize(
        "axis_len,chunk,expected_pad",
        [
            (10, 4, 2),  # 10 -> 12
            (7, 3, 2),
            (5, 8, 3),  # chunk larger than the axis
            (1, 4, 3),
            (8, 4, 0),  # already divisible
            (8, 1, 0),  # chunk of 1 never pads
            (0, 4, 0),  # empty leading axis: nothing to repeat, nothing to pad
        ],
    )
    def test_n_pad_and_resulting_length(self, axis_len, chunk, expected_pad):
        x = jnp.arange(axis_len, dtype=jnp.float32).reshape(axis_len, 1)
        padded, n_pad = pad_to_multiple(x, axis_len, chunk)

        assert n_pad == expected_pad
        assert padded.shape[0] == axis_len + expected_pad
        assert padded.shape[0] % chunk == 0

    def test_trailing_dims_preserved(self):
        x = jnp.zeros((5, 3, 7))
        padded, n_pad = pad_to_multiple(x, 5, 4)
        assert n_pad == 3
        assert padded.shape == (8, 3, 7)


class TestPadContents:
    """Check that padded slots copy the last real entry, not zeros."""

    def test_padding_repeats_the_last_entry_not_zeros(self):
        # Padded slots go through the energy backend; a zeroed walker puts
        # every atom at the origin and gives a NaN/inf energy.
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        padded, n_pad = pad_to_multiple(x, 3, 4)

        assert n_pad == 1
        np.testing.assert_allclose(np.asarray(padded[:3]), np.asarray(x))
        np.testing.assert_allclose(np.asarray(padded[3]), np.asarray(x[-1]))

    def test_multiple_pad_slots_all_repeat_the_last_entry(self):
        x = jnp.arange(5, dtype=jnp.float32)
        padded, n_pad = pad_to_multiple(x, 5, 8)

        assert n_pad == 3
        np.testing.assert_allclose(
            np.asarray(padded[5:]), np.full(3, 4.0, dtype=np.float32)
        )

    def test_slicing_off_n_pad_recovers_the_input(self):
        x = jnp.arange(7, dtype=jnp.float32)
        padded, n_pad = pad_to_multiple(x, 7, 4)
        np.testing.assert_allclose(np.asarray(padded[:-n_pad]), np.asarray(x))


class TestPytrees:
    """Check that every leaf is padded alike and that dtypes and tree
    structure survive."""

    def test_every_leaf_padded_consistently(self):
        tree = {
            "positions": jnp.ones((3, 4, 3)),
            "energy": jnp.arange(3, dtype=jnp.float32),
            "nested": {"types": jnp.zeros((3, 4), dtype=jnp.int32)},
        }
        padded, n_pad = pad_to_multiple(tree, 3, 2)

        assert n_pad == 1
        leaves = jax.tree.leaves(padded)
        assert all(leaf.shape[0] == 4 for leaf in leaves)
        assert jax.tree.structure(padded) == jax.tree.structure(tree)

    def test_leaf_dtypes_preserved(self):
        tree = {
            "f32": jnp.zeros((3, 2), dtype=jnp.float32),
            "i32": jnp.zeros((3,), dtype=jnp.int32),
            "bool": jnp.zeros((3,), dtype=bool),
        }
        padded, _ = pad_to_multiple(tree, 3, 4)
        assert padded["f32"].dtype == jnp.float32
        assert padded["i32"].dtype == jnp.int32
        assert padded["bool"].dtype == bool

    def test_already_divisible_returns_the_input_untouched(self):
        # No copy: the same object comes back.
        tree = {"a": jnp.ones((4, 2)), "b": jnp.zeros((4,))}
        padded, n_pad = pad_to_multiple(tree, 4, 4)

        assert n_pad == 0
        assert padded is tree


class TestFeedsLaxMap:
    """Check that padded arrays map through ``lax.map`` as at call sites."""

    def test_padded_output_maps_in_whole_chunks(self):
        # jax 0.10 already accepts a non-divisible axis in lax.map, so this
        # only checks that padding doesn't change the real entries' result.
        chunk = 4
        n = 6
        x = jnp.arange(n, dtype=jnp.float32)

        padded, n_pad = pad_to_multiple(x, n, chunk)
        assert padded.shape[0] % chunk == 0
        out = jax.lax.map(lambda v: v * 2.0, padded, batch_size=chunk)

        np.testing.assert_allclose(
            np.asarray(out[:-n_pad]), np.asarray(x) * 2.0
        )

    def test_works_on_a_key_array_like_the_call_sites(self):
        # burn_in.py pads the per-walker key array alongside the population.
        keys = jax.random.split(jax.random.key(0), 5)
        padded, n_pad = pad_to_multiple(keys, 5, 4)

        assert n_pad == 3
        assert padded.shape[0] == 8
        assert jnp.array_equal(
            jax.random.key_data(padded[-1]), jax.random.key_data(keys[-1])
        )
