"""Test MACE-JAX energy backend.

Tests supercell edge finding (no model needed) and full backend
integration (requires test fixture from save_mace_test_fixture.py).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.mace import (
    _build_mace_data,
    _make_image_offsets,
    _supercell_edges,
    is_available,
)

mace_required = pytest.mark.skipif(
    not is_available(),
    reason="mace-jax not installed",
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mace_mp_small"


# ---------------------------------------------------------------------------
# Supercell edge finding (no model needed)
# ---------------------------------------------------------------------------


class TestImageOffsets:
    def test_shape(self):
        # sc=2 -> offsets [-1, 0, 1] per axis -> 3^3 = 27
        offsets = _make_image_offsets(2, 2, 2)
        assert offsets.shape == (27, 3)

    def test_contains_origin(self):
        offsets = _make_image_offsets(2, 2, 2)
        assert any(np.all(o == 0) for o in offsets)

    def test_symmetric(self):
        # Should contain both positive and negative offsets
        offsets = _make_image_offsets(2, 2, 2)
        assert any(np.all(o == [1, 0, 0]) for o in offsets)
        assert any(np.all(o == [-1, 0, 0]) for o in offsets)

    def test_count_sc3(self):
        # sc=3 -> offsets [-1, 0, 1] per axis -> 3^3 = 27
        offsets = _make_image_offsets(3, 3, 3)
        assert offsets.shape[0] == 27


class TestSupercellEdges:
    def test_two_atoms_cubic(self):
        """Two atoms in a cubic box — known edge count."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        cell = jnp.eye(3) * 5.0
        r_cutoff = 2.0
        image_offsets = jnp.array(_make_image_offsets(2, 2, 2), dtype=jnp.int32)
        max_edges = 100

        senders, receivers, shifts, n_actual, overflow, true_max = (
            _supercell_edges(positions, cell, r_cutoff, max_edges, image_offsets)
        )

        # In a 5A box with cutoff 2A, atom 0 and atom 1 are 1.5A apart
        # With (2,2,2) supercell, each atom sees the other in the (0,0,0) image
        # plus images where the wrapped distance < 2A
        # (0,0,0) image: d=1.5 -> 2 edges (0->1, 1->0)
        # (1,0,0) image: atom 1 at 1.5+5.0=6.5 from atom 0 -> d=6.5 > 2 -> no
        # etc.
        # So we expect exactly 2 edges
        assert int(n_actual) == 2
        assert not overflow
        # Each atom has exactly one in-cutoff neighbor (the other atom).
        assert int(true_max) == 1

    def test_overflow_detected(self):
        """Too few max_edges -> overflow."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cell = jnp.eye(3) * 3.0
        r_cutoff = 2.0
        image_offsets = jnp.array(_make_image_offsets(2, 2, 2), dtype=jnp.int32)

        _, _, _, n_actual, overflow, true_max = _supercell_edges(
            positions, cell, r_cutoff, max_edges=1, image_offsets=image_offsets,
        )
        # At least 2 edges exist, but max_edges=1
        assert overflow
        # true_max is computed from the full mask before truncation, so it
        # reflects the real neighbor count even when the edge buffer overflows.
        assert int(true_max) >= 1

    def test_no_overflow_with_budget(self):
        """Sufficient max_edges -> no overflow."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cell = jnp.eye(3) * 5.0
        r_cutoff = 2.0
        image_offsets = jnp.array(_make_image_offsets(2, 2, 2), dtype=jnp.int32)

        _, _, _, _, overflow, _ = _supercell_edges(
            positions, cell, r_cutoff, max_edges=100, image_offsets=image_offsets,
        )
        assert not overflow

    def test_jit_compatible(self):
        """Edge finding works under JIT."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        cell = jnp.eye(3) * 5.0
        image_offsets = jnp.array(_make_image_offsets(2, 2, 2), dtype=jnp.int32)

        jit_fn = jax.jit(_supercell_edges, static_argnums=(2, 3))
        senders, receivers, shifts, n_actual, overflow, true_max = jit_fn(
            positions, cell, 2.0, 100, image_offsets,
        )
        assert int(n_actual) == 2
        assert not overflow
        assert int(true_max) == 1

    def test_shifts_are_correct(self):
        """Edge shift vectors should reconstruct correct distances."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        cell = jnp.eye(3) * 5.0
        r_cutoff = 2.0
        image_offsets = jnp.array(_make_image_offsets(2, 2, 2), dtype=jnp.int32)

        senders, receivers, shifts, n_actual, _, _ = _supercell_edges(
            positions, cell, r_cutoff, 100, image_offsets,
        )

        # Reconstruct edge vectors: pos[receiver] - pos[sender] + shift
        n = int(n_actual)
        for i in range(n):
            s, r = int(senders[i]), int(receivers[i])
            vec = positions[r] - positions[s] + shifts[i]
            dist = float(jnp.linalg.norm(vec))
            assert dist < r_cutoff, f"Edge {i}: dist={dist} >= r_cutoff={r_cutoff}"
            assert dist > 1e-10, f"Edge {i}: self-interaction"


# ---------------------------------------------------------------------------
# Full MACE backend (requires fixture)
# ---------------------------------------------------------------------------


@mace_required
class TestMACEBackend:
    @pytest.fixture
    def backend(self):
        from jaxrens.backends.mace import create_mace

        return create_mace(
            model_path=str(FIXTURE_DIR),
            supercell_trafo=(4, 4, 4),  # SrTiO3 a=3.9A, r_cutoff=6A -> need sc>=4
        )

    @pytest.fixture
    def reference(self):
        return np.load(FIXTURE_DIR / "reference.npz")

    def test_protocol_compliance(self, backend):
        assert hasattr(backend, "r_cutoff")
        assert isinstance(backend.r_cutoff, float)
        assert backend.r_cutoff > 0

    def test_energy_matches_reference(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["species"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])
        ref_energy = float(reference["jax_energy"])

        energy, count, overflow = backend(positions, species, cell, max_neighbors=100)

        assert not overflow, "Overflow with max_neighbors=100"
        assert jnp.isfinite(energy), "Energy is not finite"
        assert abs(float(energy) - ref_energy) < 1e-2, (
            f"Energy mismatch: {float(energy):.6f} vs ref {ref_energy:.6f}"
        )

    def test_jit_compatible(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["species"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])

        jit_backend = jax.jit(backend, static_argnums=(3,))
        energy, count, overflow = jit_backend(positions, species, cell, 100)

        assert jnp.isfinite(energy)
        assert not overflow

    def test_grad_gives_finite_forces(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["species"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])

        def energy_fn(pos):
            e, _, _ = backend(pos, species, cell, 100)
            return e

        forces = -jax.grad(energy_fn)(positions)
        assert jnp.all(jnp.isfinite(forces)), "Forces contain NaN/Inf"
        assert forces.shape == positions.shape

    def test_overflow_with_tiny_budget(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["species"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])

        _, _, overflow = backend(positions, species, cell, max_neighbors=1)
        assert overflow, "Should overflow with max_neighbors=1"


@mace_required
class TestMACENSStep:
    """Test ns_step with the MACE backend under JIT."""

    @pytest.fixture
    def mace_ns_setup(self):
        from jaxrens.backends.mace import create_mace
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.moves import random_walk
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        ref = np.load(FIXTURE_DIR / "reference.npz")
        backend = create_mace(
            model_path=str(FIXTURE_DIR),
            supercell_trafo=(4, 4, 4),
        )

        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("random_walk", random_walk.build_kernel, step_size=0.01),
        ])

        n_walkers = 4
        key = jax.random.key(0)
        positions = jnp.tile(
            jnp.array(ref["positions"])[None, :, :], (n_walkers, 1, 1)
        )
        # Add small random noise per walker
        key, noise_key = jax.random.split(key)
        positions = positions + 0.01 * jax.random.normal(noise_key, positions.shape)

        species = jnp.array(ref["species"], dtype=jnp.int32)
        cell = jnp.array(ref["cell"])
        cells = jnp.tile(cell[None, :, :], (n_walkers, 1, 1))

        energies = jax.vmap(
            lambda pos: backend(pos, species, cell, 100)[0]
        )(positions)

        state = init_ns(
            init_fn, positions, species, energies,
            cells=cells, rng_key=key,
        )
        # Set max_neighbors for MACE (static field, controls edge buffer size)
        state = state.set(
            population=state.population.set(max_neighbors=100),
        )

        return {
            "state": state,
            "step_fn": step_fn,
            "ns_step": ns_step,
        }

    def test_jit_ns_step(self, mace_ns_setup):
        s = mace_ns_setup
        jit_step = jax.jit(s["ns_step"], static_argnums=(1, 2, 3))

        new_state, info = jit_step(s["state"], s["step_fn"], 5, 0)

        assert new_state.iteration == 1
        assert jnp.isfinite(info["emax"])
        assert 0 <= info["acceptance_rate"] <= 1.0
