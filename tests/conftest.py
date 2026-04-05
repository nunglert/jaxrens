"""Shared test fixtures for jaxrens.

Provides dummy data, known-answer toy problems, and helper functions
used across test modules.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.state.walker import WalkerState
from jaxrens.state.ns import NSState


# ---------------------------------------------------------------------------
# Dummy data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    """Fresh JAX PRNG key."""
    return jax.random.key(42)


@pytest.fixture
def dummy_positions_3():
    """3-atom positions for minimal tests."""
    return jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.fixture
def dummy_types_3():
    """3-atom type codes."""
    return jnp.array([0, 0, 1])


@pytest.fixture
def dummy_box():
    """Cubic box with side length 5.0."""
    return 5.0 * jnp.eye(3)


@pytest.fixture
def dummy_walker(dummy_positions_3, dummy_types_3, dummy_box):
    """Single WalkerState with 3 atoms in a box."""
    return WalkerState(
        positions=dummy_positions_3,
        types=dummy_types_3,
        energy=jnp.array(-1.5),
        box=dummy_box,
        n_atoms=3,
    )


@pytest.fixture
def dummy_walker_nonperiodic(dummy_positions_3, dummy_types_3):
    """Single WalkerState without periodic boundaries."""
    return WalkerState(
        positions=dummy_positions_3,
        types=dummy_types_3,
        energy=jnp.array(-1.5),
        box=None,
        n_atoms=3,
    )


# ---------------------------------------------------------------------------
# Toy energy functions (known-answer problems)
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_energy_fn():
    """Harmonic potential: E = 0.5 * sum(positions^2).

    Known log-evidence for 3D harmonic with N atoms in a box of side L:
    log Z = N * 3/2 * log(2*pi) - 3*N*log(L)  (approximate, large L)
    """

    def energy_fn(params, positions, types, box=None, **kwargs):
        return 0.5 * jnp.sum(positions**2)

    return energy_fn


@pytest.fixture
def lj_pair_energy_fn():
    """Simple Lennard-Jones pair potential for testing.

    E = sum_{i<j} 4 * epsilon * [(sigma/r_ij)^12 - (sigma/r_ij)^6]
    with epsilon=1, sigma=1.
    """

    def energy_fn(params, positions, types, box=None, **kwargs):
        n_atoms = positions.shape[0]
        energy = jnp.array(0.0)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dr = positions[i] - positions[j]
                r2 = jnp.sum(dr**2)
                r6 = r2**3
                r12 = r6**2
                energy = energy + 4.0 * (1.0 / r12 - 1.0 / r6)
        return energy

    return energy_fn


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def n_devices():
    """Number of available JAX devices."""
    return jax.local_device_count()


@pytest.fixture
def is_multi_gpu(n_devices):
    """Whether multiple GPUs are available."""
    return n_devices > 1
