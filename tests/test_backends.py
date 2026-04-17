"""Test energy backends: toy potentials, LJ, loader, kernel dispatch.

Part of Step 2: backend protocol and implementations.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic, create_double_well, create_gaussian_mixture
from jaxrens.backends.lj import create_lj
from jaxrens.backends.loader import load_backend
from jaxrens.backends.kernel_dispatch import (
    find_closest_higher_number,
    CompiledKernelSet,
    SingleKernelSet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def positions_3():
    return jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.fixture
def types_3():
    return jnp.array([0, 0, 1])


@pytest.fixture
def box_5():
    return 5.0 * jnp.eye(3)


# ---------------------------------------------------------------------------
# Toy backends
# ---------------------------------------------------------------------------


class TestHarmonic:
    def test_energy_at_origin(self, types_3):
        backend = create_harmonic(k=1.0)
        pos = jnp.zeros((3, 3))
        e = backend(pos, types_3, jnp.zeros((3, 3)), 0)[0]
        assert jnp.allclose(e, 0.0)

    def test_energy_known_value(self, positions_3, types_3):
        backend = create_harmonic(k=1.0)
        e = backend(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        expected = 0.5 * jnp.sum(positions_3**2)
        assert jnp.allclose(e, expected)

    def test_energy_scales_with_k(self, positions_3, types_3):
        backend1 = create_harmonic(k=1.0)
        backend2 = create_harmonic(k=2.0)
        e1 = backend1(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        e2 = backend2(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        assert jnp.allclose(e2, 2.0 * e1)

    def test_jit(self, positions_3, types_3):
        backend = create_harmonic()
        jitted = jax.jit(backend)
        e = jitted(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        assert e.shape == ()

    def test_grad(self, positions_3, types_3):
        backend = create_harmonic(k=1.0)
        def energy_fn(pos):
            return backend(pos, types_3, jnp.zeros((3, 3)), 0)[0]
        grad_fn = jax.grad(energy_fn)
        forces = grad_fn(positions_3)
        assert jnp.allclose(forces, positions_3)  # dE/dx = k*x = x for k=1

    def test_vmap(self, positions_3, types_3):
        backend = create_harmonic()
        batch_pos = jnp.stack([positions_3] * 5)
        batch_types = jnp.stack([types_3] * 5)
        cell = jnp.zeros((3, 3))
        vmapped = jax.vmap(backend, in_axes=(0, 0, None, None))
        result = vmapped(batch_pos, batch_types, cell, 0)
        assert result[0].shape == (5,)


class TestDoubleWell:
    def test_minima_at_sqrt_b(self, types_3):
        backend = create_double_well(a=1.0, b=1.0)
        # Minimum at x = +-1, y=z=0
        pos_min = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        e = backend(pos_min, types_3, jnp.zeros((3, 3)), 0)[0]
        assert jnp.allclose(e, 0.0, atol=1e-6)

    def test_barrier_at_origin(self, types_3):
        backend = create_double_well(a=1.0, b=1.0)
        pos_origin = jnp.zeros((3, 3))
        e = backend(pos_origin, types_3, jnp.zeros((3, 3)), 0)[0]
        # E = 3 * a * b^2 = 3.0 (three atoms at origin, each contributes b^2)
        assert jnp.allclose(e, 3.0)

    def test_jit(self, positions_3, types_3):
        backend = create_double_well()
        e = jax.jit(backend)(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        assert e.shape == ()


class TestGaussianMixture:
    def test_energy_at_center(self, types_3):
        backend = create_gaussian_mixture(
            centers=[[0.0, 0.0, 0.0]], sigma=1.0
        )
        # Single atom at center of single Gaussian
        pos = jnp.zeros((1, 3))
        types = jnp.array([0])
        e = backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        # E = -log(N(0|0,1)) = 0.5 * 3 * log(2*pi)
        expected = 1.5 * jnp.log(2.0 * jnp.pi)
        assert jnp.allclose(e, expected, atol=1e-5)

    def test_jit(self, positions_3, types_3):
        backend = create_gaussian_mixture()
        e = jax.jit(backend)(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        assert e.shape == ()

    def test_two_modes_lower_than_one(self, types_3):
        backend1 = create_gaussian_mixture(centers=[[0.0, 0.0, 0.0]], sigma=1.0)
        backend2 = create_gaussian_mixture(
            centers=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], sigma=1.0
        )
        pos = jnp.zeros((1, 3))
        types = jnp.array([0])
        e1 = backend1(pos, types, jnp.zeros((3, 3)), 0)[0]
        e2 = backend2(pos, types, jnp.zeros((3, 3)), 0)[0]
        # Two overlapping modes -> lower energy (higher probability)
        assert e2 < e1


# ---------------------------------------------------------------------------
# Lennard-Jones
# ---------------------------------------------------------------------------


class TestLJ:
    def test_two_atoms_known_value(self):
        backend = create_lj(epsilon=1.0, sigma=1.0)
        # Two atoms at distance 1.0: E = 4*(1 - 1) = 0
        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        types = jnp.array([0, 0])
        e = backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        assert jnp.allclose(e, 0.0, atol=1e-5)

    def test_two_atoms_equilibrium(self):
        backend = create_lj(epsilon=1.0, sigma=1.0)
        # Equilibrium distance r_eq = 2^(1/6) * sigma
        r_eq = 2.0 ** (1.0 / 6.0)
        pos = jnp.array([[0.0, 0.0, 0.0], [r_eq, 0.0, 0.0]])
        types = jnp.array([0, 0])
        e = backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        assert jnp.allclose(e, -1.0, atol=1e-5)  # E_min = -epsilon

    def test_repulsive_at_short_range(self):
        backend = create_lj(epsilon=1.0, sigma=1.0)
        pos = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        types = jnp.array([0, 0])
        e = backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        assert e > 0  # Repulsive at r < sigma

    def test_periodic_minimum_image(self):
        backend = create_lj(epsilon=1.0, sigma=1.0)
        box = 3.0 * jnp.eye(3)
        # Two atoms: one at (0,0,0), one at (2.8,0,0) in a 3.0 box
        # Minimum image distance = 0.2 (wraps around)
        pos = jnp.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        types = jnp.array([0, 0])
        e_pbc = backend(pos, types, box, 0)[0]
        e_no_pbc = backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        # With PBC, distance is 0.2 -> very repulsive
        # Without PBC, distance is 2.8 -> weakly attractive
        assert e_pbc > e_no_pbc

    def test_jit(self, positions_3, types_3):
        backend = create_lj()
        e = jax.jit(backend)(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        assert e.shape == ()

    def test_grad(self, positions_3, types_3):
        backend = create_lj()
        def energy_fn(pos):
            return backend(pos, types_3, jnp.zeros((3, 3)), 0)[0]
        grad_fn = jax.grad(energy_fn)
        forces = grad_fn(positions_3)
        assert forces.shape == positions_3.shape

    def test_vmap(self, positions_3, types_3):
        backend = create_lj()
        batch_pos = jnp.stack([positions_3] * 4)
        batch_types = jnp.stack([types_3] * 4)
        cell = jnp.zeros((3, 3))
        vmapped = jax.vmap(backend, in_axes=(0, 0, None, None))
        result = vmapped(batch_pos, batch_types, cell, 0)
        assert result[0].shape == (4,)

    def test_cutoff(self, positions_3, types_3):
        backend_no_cut = create_lj(cutoff=None)
        backend_cut = create_lj(cutoff=0.5)
        e_no_cut = backend_no_cut(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        e_cut = backend_cut(positions_3, types_3, jnp.zeros((3, 3)), 0)[0]
        # With cutoff=0.5, all pairs (distance >= 1.0) are beyond cutoff
        assert jnp.allclose(e_cut, 0.0, atol=1e-5)
        assert not jnp.allclose(e_no_cut, 0.0)


# ---------------------------------------------------------------------------
# Backend loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_harmonic(self):
        backend = load_backend("harmonic", k=2.0)
        pos = jnp.zeros((3, 3))
        types = jnp.array([0, 0, 0])
        e = backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        assert jnp.allclose(e, 0.0)

    def test_load_lj(self):
        backend = load_backend("lj", epsilon=1.0, sigma=1.0)
        assert backend is not None

    def test_load_double_well(self):
        backend = load_backend("double_well")
        assert backend is not None

    def test_load_gaussian_mixture(self):
        backend = load_backend("gaussian_mixture")
        assert backend is not None

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            load_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# Kernel dispatch
# ---------------------------------------------------------------------------


class TestFindClosestHigherNumber:
    def test_exact_match(self):
        assert find_closest_higher_number(15, [4, 6, 8, 10, 15, 20]) == 15

    def test_between_values(self):
        assert find_closest_higher_number(12, [4, 6, 8, 10, 15, 20]) == 15

    def test_extra_offset(self):
        assert find_closest_higher_number(12, [4, 6, 8, 10, 15, 20], extra_offset=1) == 20

    def test_at_minimum(self):
        assert find_closest_higher_number(1, [10, 20, 30]) == 10

    def test_at_maximum(self):
        assert find_closest_higher_number(30, [10, 20, 30]) == 30

    def test_exceeds_all(self):
        with pytest.raises(ValueError, match="No bucket large enough"):
            find_closest_higher_number(100, [10, 20, 30])

    def test_extra_offset_clamps_to_end(self):
        # extra_offset=10 but only 3 elements after match
        assert find_closest_higher_number(5, [10, 20, 30], extra_offset=10) == 30


class TestCompiledKernelSet:
    def test_select_basic(self):
        factory = lambda n: f"kernel_{n}"
        ks = CompiledKernelSet(factory, [10, 20, 30, 40, 50], max_neighbors_offset=5)

        # current=12 + offset=5 = 17 -> bucket 20
        kernel = ks.select(12)
        assert kernel == "kernel_20"

    def test_select_with_escalation(self):
        factory = lambda n: f"kernel_{n}"
        ks = CompiledKernelSet(factory, [10, 20, 30, 40, 50], max_neighbors_offset=5)

        # current=12 + offset=5 = 17 -> bucket 20, then escalate by 1 -> 30
        kernel = ks.select(12, extra_offset=1)
        assert kernel == "kernel_30"

    def test_adjust_and_run_no_violation(self):
        factory = lambda n: f"kernel_{n}"
        ks = CompiledKernelSet(factory, [10, 20, 30], max_neighbors_offset=0)

        result, info = ks.adjust_and_run(
            walkers="walkers",
            get_neighbor_count=lambda w: 15,
            run_fn=lambda kernel, w: (f"result_{kernel}", "info"),
            check_violation=lambda r: False,
        )
        assert result == "result_kernel_20"

    def test_adjust_and_run_with_retry(self):
        call_count = 0

        def run_fn(kernel, walkers):
            nonlocal call_count
            call_count += 1
            return f"result_{kernel}_{call_count}", "info"

        def check_violation(result):
            # First call violates, second succeeds
            return "1" in result

        factory = lambda n: f"kernel_{n}"
        ks = CompiledKernelSet(factory, [10, 20, 30], max_neighbors_offset=0)

        result, info = ks.adjust_and_run(
            walkers="walkers",
            get_neighbor_count=lambda w: 15,
            run_fn=run_fn,
            check_violation=check_violation,
        )
        assert call_count == 2
        assert "kernel_30" in result  # Escalated to next bucket

    def test_adjust_and_run_max_retries_exceeded(self):
        factory = lambda n: f"kernel_{n}"
        ks = CompiledKernelSet(factory, [10, 20, 30], max_neighbors_offset=0)

        with pytest.raises(ValueError, match="violations persisted"):
            ks.adjust_and_run(
                walkers="walkers",
                get_neighbor_count=lambda w: 15,
                run_fn=lambda k, w: ("result", "info"),
                check_violation=lambda r: True,  # Always violates
                max_retries=2,
            )


class TestSingleKernelSet:
    def test_select_returns_kernel(self):
        ks = SingleKernelSet("my_kernel")
        assert ks.select() == "my_kernel"
        assert ks.select(42, extra_offset=3) == "my_kernel"
