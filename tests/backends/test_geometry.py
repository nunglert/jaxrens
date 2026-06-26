"""Tests for the shared minimum-image pairwise-distance helper."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxrens.backends.geometry import pairwise_distances


def test_open_boundary_raw_distances():
    positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    cell = jnp.zeros((3, 3))  # non-periodic
    r = pairwise_distances(positions, cell)
    assert r.shape == (2, 2)
    np.testing.assert_allclose(np.diag(r), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(float(r[0, 1]), 5.0, atol=1e-5)
    np.testing.assert_allclose(float(r[1, 0]), 5.0, atol=1e-5)


def test_minimum_image_across_boundary():
    # Two atoms 1 Angstrom apart across a periodic boundary of a 10 A cube.
    cell = jnp.eye(3) * 10.0
    positions = jnp.array([[0.5, 0.5, 0.5], [9.5, 0.5, 0.5]])
    r_periodic = pairwise_distances(positions, cell)
    np.testing.assert_allclose(float(r_periodic[0, 1]), 1.0, atol=1e-4)

    # Same coordinates, no cell -> raw distance is the full 9 A.
    r_open = pairwise_distances(positions, jnp.zeros((3, 3)))
    np.testing.assert_allclose(float(r_open[0, 1]), 9.0, atol=1e-4)
