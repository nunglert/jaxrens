"""Test NeuralIL backend wrapper.

Tests import guards, model loading, energy evaluation, JIT compatibility,
and ns_step integration. Skipped if NeuralIL is not installed.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.neuralil import _NEURALIL_IMPORT_ERROR, is_available

neuralil_required = pytest.mark.skipif(
    not is_available(),
    reason=f"NeuralIL not installed: {_NEURALIL_IMPORT_ERROR}",
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "neuralil_tiny"


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
        from jaxrens.backends.neuralil import create_neuralil

        return create_neuralil(
            pickle_file=str(FIXTURE_DIR / "model.pkl"),
            supercell_trafo=(1, 1, 1),
        )

    @pytest.fixture
    def reference(self):
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

        _r = backend(positions, species, cell, max_neighbors)
        energy, count, overflow = _r.energy, _r.max_neighbor_count, _r.overflow

        assert jnp.isfinite(energy), f"Energy is not finite: {energy}"
        assert not overflow, f"Overflow with max_neighbors={max_neighbors}"

    def test_energy_matches_reference(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["types"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])
        max_neighbors = int(reference["max_neighbors"])
        ref_energy = float(reference["energy"])

        energy = backend(positions, species, cell, max_neighbors).energy

        # Relative tolerance: the energy is computed in float32 (x64 is pinned
        # off), so the result drifts ~1e-4 relative across GPU architectures
        # (TF32 matmuls + reduction order). An absolute bound silently coupled
        # this to the runner family; rtol keeps it hardware-portable while still
        # catching real regressions, which shift the energy far more than meV.
        assert abs(float(energy) - ref_energy) < 1e-3 * abs(
            ref_energy
        ), f"Energy mismatch: {float(energy):.6f} vs ref {ref_energy:.6f}"

    def test_overflow_with_tiny_budget(self, backend, reference):
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["types"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])

        overflow = backend(positions, species, cell, max_neighbors=1).overflow
        assert overflow, "Should overflow with max_neighbors=1"

    def test_max_neighbors_for(self, backend, reference):
        """Geometry-only neighbor count agrees with the value the
        backend itself reports during a real energy call.

        Mirrors ``tests/test_mace.py::test_max_neighbors_for``.  The
        resolver's ``_finalise_initial_energies_and_counts`` relies on
        this method existing so it can size the initial energy compile
        to the bucket burn-in / NS step will use, rather than wasting
        a heavy compile at ``max_neighbors=0``.
        """
        positions = jnp.array(reference["positions"])
        species = jnp.array(reference["types"], dtype=jnp.int32)
        cell = jnp.array(reference["cell"])

        # max_neighbors_for is geometry-only — no species mask, no
        # energy compile — so passing the real cell/positions should
        # produce the same count the backend reports internally.
        n_from_method = int(backend.max_neighbors_for(positions, cell))
        n_from_call = backend(
            positions, species, cell, max_neighbors=64
        ).max_neighbor_count
        assert n_from_method == int(n_from_call), (
            f"max_neighbors_for={n_from_method} disagrees with the count "
            f"the backend reports during __call__={int(n_from_call)}"
        )

    def test_max_neighbors_for_vmap(self, backend, reference):
        """``max_neighbors_for`` must be jax.vmap-compatible.

        The resolver vmaps it over walkers (and pmaps over GPUs);
        confirm a 2-walker stack works.
        """
        positions = jnp.array(reference["positions"])
        cell = jnp.array(reference["cell"])
        stacked_positions = jnp.stack([positions, positions], axis=0)
        stacked_cells = jnp.stack([cell, cell], axis=0)

        counts = jax.vmap(backend.max_neighbors_for)(
            stacked_positions,
            stacked_cells,
        )
        assert counts.shape == (2,)
        assert int(counts[0]) == int(counts[1])  # identical inputs
        assert int(counts[0]) > 0


@neuralil_required
class TestNeuralILForces:
    """Native ``energy_and_forces`` agrees with the autodiff fallback.

    NeuralIL forces are the exact gradient of the energy, so the native
    path (``calc_potential_energy_and_forces``) must reproduce ``-dE/dx``
    obtained by differentiating ``__call__``.
    """

    @pytest.fixture
    def backend(self):
        from jaxrens.backends.neuralil import create_neuralil

        return create_neuralil(
            pickle_file=str(FIXTURE_DIR / "model.pkl"),
            supercell_trafo=(1, 1, 1),
        )

    @pytest.fixture
    def config(self):
        ref = np.load(FIXTURE_DIR / "reference.npz", allow_pickle=True)
        return (
            jnp.array(ref["positions"]),
            jnp.array(ref["types"], dtype=jnp.int32),
            jnp.array(ref["cell"]),
            int(ref["max_neighbors"]),
        )

    def test_native_forces_shape_and_finite(self, backend, config):
        positions, species, cell, mn = config
        res = backend.energy_and_forces(positions, species, cell, mn)
        assert res.forces is not None
        assert res.forces.shape == positions.shape
        assert jnp.all(jnp.isfinite(res.forces))

    def test_energy_matches_call(self, backend, config):
        positions, species, cell, mn = config
        e_call = backend(positions, species, cell, mn).energy
        e_force = backend.energy_and_forces(
            positions, species, cell, mn
        ).energy
        assert abs(float(e_call) - float(e_force)) < 1e-5

    def test_native_forces_match_autodiff(self, backend, config):
        positions, species, cell, mn = config

        def energy_of(pos):
            return backend(pos, species, cell, mn).energy

        autodiff_forces = -jax.grad(energy_of)(positions)
        native_forces = backend.energy_and_forces(
            positions, species, cell, mn
        ).forces

        # Mathematically identical (-dE/dx); the gap is float32 + GPU
        # non-determinism through the deep descriptor model (~1e-4 here).
        np.testing.assert_allclose(
            np.asarray(native_forces),
            np.asarray(autodiff_forces),
            atol=2e-3,
            rtol=0.0,
        )

    def test_eval_energy_and_forces_dispatches_native(self, backend, config):
        """``eval_energy_and_forces`` must route to the backend's native
        method, not the autodiff fallback.

        Native-vs-native differs only by GPU non-determinism (~1e-7), while
        the autodiff fallback differs by ~1e-4, so a 1e-5 tolerance confirms
        the dispatch took the native path.
        """
        from jaxrens.backends.base import eval_energy_and_forces

        positions, species, cell, mn = config
        res = eval_energy_and_forces(backend, positions, species, cell, mn)
        native = backend.energy_and_forces(positions, species, cell, mn)
        np.testing.assert_allclose(
            np.asarray(res.forces),
            np.asarray(native.forces),
            atol=1e-5,
            rtol=0.0,
        )

    def test_members_keeps_per_member_axis(self, backend, config):
        """``members()`` keeps the ``(M, …)`` axis; reduced fields are the
        committee means; ``committee_uncertainty`` returns finite σ."""
        from jaxrens.backends.base import committee_uncertainty

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]
        m = backend.n_ensemble
        res = backend.members(positions, species, cell, mn)

        assert res.energy_members is not None
        assert res.forces_members is not None
        assert res.energy_members.shape == (m,)
        assert res.forces_members.shape == (m, n_atoms, 3)
        # Reduced fields are the member means.
        assert abs(float(res.energy) - float(res.energy_members.mean())) < 1e-5
        np.testing.assert_allclose(
            np.asarray(res.forces),
            np.asarray(res.forces_members.mean(axis=0)),
            atol=1e-5,
        )

        e_std, f_std = committee_uncertainty(res)
        assert float(e_std) >= 0.0
        assert f_std.shape == (n_atoms,)
        assert float(jnp.min(f_std)) >= 0.0


@neuralil_required
class TestNeuralILNSStep:
    """Test ns_step with the NeuralIL backend under JIT."""

    @pytest.fixture
    def neuralil_ns_setup(self):
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

        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel(
                    "random_walk", random_walk.build_kernel, step_size=0.01
                ),
            ],
        )

        # Use 3 walkers with slightly different positions
        n_walkers = 3
        base_pos = jnp.array(ref["positions"])
        species = jnp.array(ref["types"], dtype=jnp.int32)
        cell = jnp.array(ref["cell"])

        key = jax.random.key(0)
        positions = jnp.tile(base_pos[None, :, :], (n_walkers, 1, 1))
        key, noise_key = jax.random.split(key)
        positions = positions + 0.01 * jax.random.normal(
            noise_key, positions.shape
        )

        cells = jnp.tile(cell[None, :, :], (n_walkers, 1, 1))

        energies = jax.vmap(
            lambda pos: backend(pos, species, cell, max_neighbors)[0]
        )(positions)

        state = init_ns(
            init_fn,
            positions,
            species,
            energies,
            cells=cells,
            rng_key=key,
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
        assert jnp.isfinite(info["emax"])
        assert 0 <= info["acceptance_rate"] <= 1.0
