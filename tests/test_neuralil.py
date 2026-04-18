"""Test NeuralIL backend wrapper.

Tests import guards, model loading, energy evaluation, JIT compatibility,
and ns_step integration. Skipped if NeuralIL is not installed.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.neuralil import is_available, _NEURALIL_IMPORT_ERROR

neuralil_required = pytest.mark.skipif(
    not is_available(),
    reason=f"NeuralIL not installed: {_NEURALIL_IMPORT_ERROR}",
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "neuralil_tiny"


def fixture_available():
    return FIXTURE_DIR.exists() and (FIXTURE_DIR / "model.pkl").exists()


# ---------------------------------------------------------------------------
# Always-run tests (not skipped)
# ---------------------------------------------------------------------------


class TestNeuralILAvailability:
    def test_import_does_not_crash(self):
        import jaxrens.backends.neuralil  # noqa: F401

    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)


# ---------------------------------------------------------------------------
# Import guard tests
# ---------------------------------------------------------------------------


@neuralil_required
class TestNeuralILImport:
    def test_create_neuralil_without_pickle_raises(self):
        from jaxrens.backends.neuralil import create_neuralil

        with pytest.raises(ValueError, match="pickle_file is required"):
            create_neuralil(pickle_file=None)


# ---------------------------------------------------------------------------
# Backend integration tests (require fixture)
# ---------------------------------------------------------------------------


@neuralil_required
class TestNeuralILBackend:
    @pytest.fixture
    def backend(self):
        if not fixture_available():
            pytest.skip("NeuralIL fixture not found. Run train_neuralil_fixture.py.")
        from jaxrens.backends.neuralil import create_neuralil

        return create_neuralil(
            pickle_file=str(FIXTURE_DIR / "model.pkl"),
            supercell_trafo=(1, 1, 1),
        )

    @pytest.fixture
    def reference(self):
        if not fixture_available():
            pytest.skip("NeuralIL fixture not found.")
        return np.load(FIXTURE_DIR / "reference.npz", allow_pickle=True)

    def test_protocol_compliance(self, backend):
        assert hasattr(backend, "r_cutoff")
        assert isinstance(backend.r_cutoff, float)
        assert backend.r_cutoff > 0

    def test_energy_is_finite(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["types"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])
        max_neighbors = int(reference["max_neighbors"])

        energy, count, overflow = backend(positions, species, cell, max_neighbors)

        assert jnp.isfinite(energy), f"Energy is not finite: {energy}"
        assert not overflow, f"Overflow with max_neighbors={max_neighbors}"

    def test_energy_matches_reference(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["types"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])
        max_neighbors = int(reference["max_neighbors"])
        ref_energy = float(reference["energy"])

        energy, _, _ = backend(positions, species, cell, max_neighbors)

        assert abs(float(energy) - ref_energy) < 1e-3, (
            f"Energy mismatch: {float(energy):.6f} vs ref {ref_energy:.6f}"
        )

    def test_overflow_with_tiny_budget(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["types"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])

        _, _, overflow = backend(positions, species, cell, max_neighbors=1)
        assert overflow, "Should overflow with max_neighbors=1"


@neuralil_required
class TestNeuralILNSStep:
    """Test ns_step with the NeuralIL backend under JIT."""

    @pytest.fixture
    def neuralil_ns_setup(self):
        if not fixture_available():
            pytest.skip("NeuralIL fixture not found.")
        from jaxrens.backends.neuralil import create_neuralil
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.moves import random_walk
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        ref = np.load(FIXTURE_DIR / "reference.npz", allow_pickle=True)
        max_neighbors = int(ref["max_neighbors"])

        backend = create_neuralil(
            pickle_file=str(FIXTURE_DIR / "model.pkl"),
            supercell_trafo=(1, 1, 1),
        )

        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("random_walk", random_walk.build_kernel, step_size=0.01),
        ])

        # Use 3 walkers with slightly different positions
        n_walkers = 3
        base_pos = jnp.array(ref["positions"])
        species = jnp.array(ref["types"], dtype=jnp.int32)
        cell = jnp.array(ref["cell"])

        key = jax.random.key(0)
        positions = jnp.tile(base_pos[None, :, :], (n_walkers, 1, 1))
        key, noise_key = jax.random.split(key)
        positions = positions + 0.01 * jax.random.normal(noise_key, positions.shape)

        cells = jnp.tile(cell[None, :, :], (n_walkers, 1, 1))

        energies = jax.vmap(
            lambda pos: backend(pos, species, cell, max_neighbors)[0]
        )(positions)

        state = init_ns(
            init_fn, positions, species, energies,
            cells=cells, rng_key=key,
        )
        state = state.set(
            population=state.population.set(max_neighbors=max_neighbors),
        )

        return {
            "state": state,
            "step_fn": step_fn,
            "ns_step": ns_step,
        }

    def test_jit_ns_step(self, neuralil_ns_setup):
        s = neuralil_ns_setup
        jit_step = jax.jit(s["ns_step"], static_argnums=(1, 2, 3))

        new_state, info = jit_step(s["state"], s["step_fn"], 3, 0)

        assert new_state.iteration == 1
        assert new_state.n_dead == 1
        assert jnp.isfinite(info["emax"])
        assert 0 <= info["acceptance_rate"] <= 1.0
