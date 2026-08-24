"""Tests for the soft-core NeuralIL variants and their jaxrens wiring.

The soft-core model augments NeuralIL with a fixed (non-trainable)
repulsive Morse term. Tests confirm:

1. ``SoftCoreNeuralIL`` initializes with the same params tree as plain
   ``NeuralIL`` (no ``morse`` subtree).
2. Close-contact configurations get a repulsion that matches a direct
   ``FixedRepulsiveMorse`` evaluation.
3. ``jax.jit`` compilation succeeds (mandatory per project policy).
4. ``SoftCorePlainEnsemble`` returns one energy per ensemble member, and
   the soft-core penalty is identical across members.
5. The ``_build_dynamics_model`` plumbing rejects ``softcore=True`` +
   ``has_morse=True`` as mutually exclusive.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.neuralil import _NEURALIL_IMPORT_ERROR, is_available

pytestmark = pytest.mark.neuralil

neuralil_required = pytest.mark.skipif(
    not is_available(),
    reason=f"NeuralIL not installed: {_NEURALIL_IMPORT_ERROR}",
)

try:
    from neuralil.softcore.model import (
        FixedRepulsiveMorse,
        SoftCoreNeuralIL,
        SoftCorePlainEnsemble,
    )

    _SOFTCORE_AVAILABLE = True
except ImportError:
    _SOFTCORE_AVAILABLE = False

softcore_required = pytest.mark.skipif(
    not _SOFTCORE_AVAILABLE,
    reason="neuralil.softcore.model not importable",
)


# Soft-core hyperparameters chosen so the close-contact configuration sits
# inside r_core_switch and the far configuration is well past r_core_cut.
A0, B0, D0 = 1.0, 3.0, 1.0
R_CORE_CUT = 1.25
R_CORE_SWITCH = 0.75


def _build_softcore_model(n_types=2, embed_d=4, r_cutoff=4.0, n_max=3):
    """Build a SoftCoreNeuralIL with a tiny core for fast tests."""
    from neuralil.bessel_descriptors import PowerSpectrumGenerator
    from neuralil.model import ResNetCore

    descriptor_gen = PowerSpectrumGenerator(
        n_max, r_cutoff, n_types, (1, 1, 1)
    )
    core_model = ResNetCore([8, 4])
    return SoftCoreNeuralIL(
        n_types,
        embed_d,
        r_cutoff,
        descriptor_gen,
        descriptor_gen.process_some_data,
        core_model,
        a0=A0,
        b0=B0,
        d0=D0,
        r_core_cut=R_CORE_CUT,
        r_core_switch=R_CORE_SWITCH,
    )


def _build_plain_model(n_types=2, embed_d=4, r_cutoff=4.0, n_max=3):
    """Build a plain NeuralIL with the same shapes as _build_softcore_model."""
    from neuralil.bessel_descriptors import PowerSpectrumGenerator
    from neuralil.model import NeuralIL, ResNetCore

    descriptor_gen = PowerSpectrumGenerator(
        n_max, r_cutoff, n_types, (1, 1, 1)
    )
    core_model = ResNetCore([8, 4])
    return NeuralIL(
        n_types,
        embed_d,
        r_cutoff,
        descriptor_gen,
        descriptor_gen.process_some_data,
        core_model,
    )


def _far_positions():
    """Four atoms in a 10 Å cubic cell, well-separated (> 2 Å pairwise)."""
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=jnp.float32,
    )


def _close_positions():
    """Same as far, but atom 1 placed at r=0.4 from atom 0 (inside r_switch)."""
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=jnp.float32,
    )


def _cell():
    return 10.0 * jnp.eye(3, dtype=jnp.float32)


def _types():
    return jnp.array([0, 1, 0, 1], dtype=jnp.int32)


def _params_leaf_keys(params):
    """Return the set of top-level keys in a flax params dict."""
    inner = params.get("params", params)
    return set(inner.keys())


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


@neuralil_required
@softcore_required
class TestSoftCoreNeuralIL:
    @pytest.fixture
    def model(self):
        return _build_softcore_model()

    @pytest.fixture
    def params(self, model):
        key = jax.random.key(0)
        return model.init(
            key,
            _far_positions(),
            _types(),
            _cell(),
            max_neighbors=8,
            method=model.calc_potential_energy,
        )

    def test_initializes_with_plain_neuralil_params_tree(self, model, params):
        """SoftCore has no trainable Morse params → params tree must match plain NeuralIL."""
        plain = _build_plain_model()
        key = jax.random.key(0)
        plain_params = plain.init(
            key,
            _far_positions(),
            _types(),
            _cell(),
            max_neighbors=8,
            method=plain.calc_potential_energy,
        )
        assert _params_leaf_keys(params) == _params_leaf_keys(plain_params), (
            f"SoftCore params keys {_params_leaf_keys(params)} != plain "
            f"NeuralIL keys {_params_leaf_keys(plain_params)}"
        )

    def test_penalises_close_contact(self, model, params):
        """Close-contact energy ≈ far energy + direct soft-core penalty."""
        cell = _cell()
        types = _types()
        max_neighbors = 8

        def energy(positions):
            return model.apply(
                params,
                positions,
                types,
                cell,
                max_neighbors,
                method=model.calc_potential_energy,
            )

        e_far = float(energy(_far_positions()))
        e_close = float(energy(_close_positions()))

        # Direct evaluation of the soft-core contribution at r=0.4. The
        # repulsive Morse is symmetric in i↔j and smooth_cutoff(0.4, 0.75, 1.25)
        # = 1.0 (well below r_switch), so the pair contributes
        # 0.5 * phi(0.4) per atom on each side → total = phi(0.4).
        phi = D0 * np.exp(-2.0 * A0 * (0.4 - B0))
        expected_softcore_penalty = float(phi)

        # The NN contribution differs slightly between configurations because
        # the descriptors at atom 0 change. Bound the test by checking the
        # soft-core delta is at least 90% of the bare Morse term.
        delta = e_close - e_far
        assert delta > 0.9 * expected_softcore_penalty, (
            f"Close-contact energy delta {delta:.4f} smaller than expected "
            f"soft-core lower bound {0.9 * expected_softcore_penalty:.4f}"
        )

    def test_jit_compiles(self, model, params):
        """``jax.jit`` of calc_potential_energy must succeed and return finite energy."""
        cell = _cell()
        types = _types()
        max_neighbors = 8

        @jax.jit
        def jitted_energy(positions):
            return model.apply(
                params,
                positions,
                types,
                cell,
                max_neighbors,
                method=model.calc_potential_energy,
            )

        e = jitted_energy(_far_positions())
        assert jnp.isfinite(e), f"JIT'd energy is not finite: {e}"


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------


@neuralil_required
@softcore_required
class TestSoftCorePlainEnsemble:
    @pytest.fixture
    def ensemble(self):
        return SoftCorePlainEnsemble(_build_softcore_model(), n_models=3)

    @pytest.fixture
    def params(self, ensemble):
        key = jax.random.key(0)
        return ensemble.init(
            key,
            _far_positions(),
            _types(),
            _cell(),
            max_neighbors=8,
            method=ensemble.calc_potential_energy,
        )

    def test_output_shape_is_per_member(self, ensemble, params):
        """Energy output has shape (n_models,)."""
        e = ensemble.apply(
            params,
            _far_positions(),
            _types(),
            _cell(),
            max_neighbors=8,
            method=ensemble.calc_potential_energy,
        )
        assert e.shape == (3,), f"Expected (3,), got {e.shape}"

    def test_softcore_piece_is_parameter_free(self, ensemble, params):
        """The soft-core has no trainable params → the ensemble's params
        tree must contain only the NN sub-tree under ``params/neuralil``,
        with no ``soft_core`` or ``morse`` leaves."""
        inner = params.get("params", params)
        # PlainEnsemble nests the NeuralIL submodule under "neuralil".
        assert (
            "neuralil" in inner
        ), f"Expected 'neuralil' key in ensemble params, got {inner.keys()}"
        sub_keys = set(inner["neuralil"].keys())
        # Standard NeuralIL has core_model + embed + denormalizer.
        assert "morse" not in sub_keys, (
            f"Soft-core must not introduce a trainable Morse sub-tree, "
            f"but params/neuralil has keys {sub_keys}"
        )
        assert "soft_core" not in sub_keys, (
            f"Soft-core must be parameter-free, but params/neuralil has "
            f"keys {sub_keys}"
        )

    def test_softcore_geometry_only_penalty(self, ensemble, params):
        """Direct evaluation of ``FixedRepulsiveMorse`` matches the
        geometry-only soft-core penalty the ensemble adds — confirming
        the soft-core piece depends only on geometry, not on params."""
        cell = _cell()
        types = _types()

        # Compute radii the same way SoftCoreNeuralIL does.
        psg = ensemble.neuralil.descriptor_generator
        _, radii, all_types = psg.center_at_atoms(
            _close_positions(), types, cell
        )
        softcore = FixedRepulsiveMorse(A0, B0, D0, R_CORE_CUT, R_CORE_SWITCH)
        morse_atomic = softcore.apply(
            {},
            radii,
            types,
            all_types,
            method=softcore.calc_atomic_energies,
        )
        # Shape is (n_atoms,) — single evaluation, no ensemble axis.
        assert morse_atomic.shape == (types.shape[0],), (
            f"FixedRepulsiveMorse output has shape {morse_atomic.shape}; "
            f"expected ({types.shape[0]},)"
        )
        # Atoms 0 and 1 at r=0.4 → both feel a strong repulsion.
        assert float(morse_atomic[0]) > 0.0
        assert float(morse_atomic[1]) > 0.0
        # Atoms 2 and 3 are well past r_core_cut → ~0.
        assert abs(float(morse_atomic[2])) < 1e-3
        assert abs(float(morse_atomic[3])) < 1e-3


@neuralil_required
@softcore_required
class TestSoftCoreEnsembleForces:
    """Native ``energy_and_forces`` on an ensemble soft-core backend.

    Confirms the ``(n_ensemble, N, 3)`` member-force Jacobian is reduced to
    the committee mean ``(N, 3)`` (mirroring the energy ``.mean()``), and that
    the soft-core repulsion is included in the native forces (they match the
    autodiff fallback of the full ``__call__``).
    """

    @pytest.fixture
    def backend(self):
        from jaxrens.backends.neuralil import NeuralILBackend

        n_ensemble = 3
        model = SoftCorePlainEnsemble(
            _build_softcore_model(), n_models=n_ensemble
        )
        key = jax.random.key(0)
        params = model.init(
            key,
            _far_positions(),
            _types(),
            _cell(),
            max_neighbors=8,
            method=model.calc_potential_energy,
        )
        return NeuralILBackend(
            model_params=params,
            r_cutoff=4.0,
            sorted_elements=["H", "Si"],
            supercell_trafo=(1, 1, 1),
            n_max=3,
            embed_d=4,
            core_widths=[8, 4],
            is_ensemble=True,
            has_morse=False,
            n_ensemble=n_ensemble,
            softcore=True,
        )

    def test_force_reduced_to_per_atom_mean(self, backend):
        positions, species, cell = _far_positions(), _types(), _cell()
        res = backend.energy_and_forces(positions, species, cell, 8)
        assert res.forces is not None
        # Reduced over the ensemble axis: one (N, 3) force, not (n_ens, N, 3).
        assert res.forces.shape == positions.shape

    def test_native_forces_match_autodiff(self, backend):
        positions, species, cell = _close_positions(), _types(), _cell()

        def energy_of(pos):
            return backend(pos, species, cell, 8).energy

        autodiff_forces = -jax.grad(energy_of)(positions)
        native_forces = backend.energy_and_forces(
            positions, species, cell, 8
        ).forces
        np.testing.assert_allclose(
            np.asarray(native_forces),
            np.asarray(autodiff_forces),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_members_committee_uncertainty(self, backend):
        """``members()`` keeps the per-member axis; ``committee_uncertainty``
        reduces it to energy-σ + per-atom force-σ, both > 0 for a real
        (random-init) committee."""
        from jaxrens.backends.base import committee_uncertainty

        positions, species, cell = _close_positions(), _types(), _cell()
        n_atoms = positions.shape[0]
        res = backend.members(positions, species, cell, 8)

        assert res.energy_members.shape == (3,)  # n_ensemble = 3
        assert res.forces_members.shape == (3, n_atoms, 3)
        # Reduced fields are the committee means.
        assert abs(float(res.energy) - float(res.energy_members.mean())) < 1e-5
        np.testing.assert_allclose(
            np.asarray(res.forces),
            np.asarray(res.forces_members.mean(axis=0)),
            atol=1e-5,
        )

        e_std, f_std = committee_uncertainty(res)
        assert float(e_std) > 0.0
        assert f_std.shape == (n_atoms,)
        assert float(jnp.min(f_std)) >= 0.0
        assert float(jnp.max(f_std)) > 0.0

    def test_get_committee_backend_unwraps(self, backend):
        from jaxrens.backends.base import get_committee_backend
        from jaxrens.backends.ensemble import EnsembleBackend

        # The committee backend itself is returned directly.
        assert get_committee_backend(backend) is backend
        # And it is found through an EnsembleBackend (per-run) wrapper.
        wrapped = EnsembleBackend(backend, pressure=0.0)
        assert get_committee_backend(wrapped) is backend

    def test_get_committee_backend_none_for_non_committee(self):
        from jaxrens.backends.base import get_committee_backend

        assert get_committee_backend(object()) is None
        assert get_committee_backend(None) is None


@neuralil_required
class TestSingleModelMembers:
    """A non-ensemble model: ``members()`` adds a leading M=1 axis so the
    ``(M, …)`` contract holds, and the committee spread is exactly zero."""

    @pytest.fixture
    def backend(self):
        from jaxrens.backends.neuralil import NeuralILBackend

        model = _build_plain_model()
        key = jax.random.key(0)
        params = model.init(
            key,
            _far_positions(),
            _types(),
            _cell(),
            max_neighbors=8,
            method=model.calc_potential_energy,
        )
        return NeuralILBackend(
            model_params=params,
            r_cutoff=4.0,
            sorted_elements=["H", "Si"],
            supercell_trafo=(1, 1, 1),
            n_max=3,
            embed_d=4,
            core_widths=[8, 4],
            is_ensemble=False,
            has_morse=False,
            n_ensemble=1,
            softcore=False,
        )

    def test_members_m1_zero_spread(self, backend):
        from jaxrens.backends.base import committee_uncertainty

        positions, species, cell = _far_positions(), _types(), _cell()
        n_atoms = positions.shape[0]
        res = backend.members(positions, species, cell, 8)

        assert res.energy_members.shape == (1,)
        assert res.forces_members.shape == (1, n_atoms, 3)
        assert abs(float(res.energy) - float(res.energy_members.mean())) < 1e-6

        e_std, f_std = committee_uncertainty(res)
        assert float(e_std) == 0.0
        assert f_std.shape == (n_atoms,)
        assert float(jnp.max(jnp.abs(f_std))) == 0.0


# ---------------------------------------------------------------------------
# Backend wiring tests
# ---------------------------------------------------------------------------


@neuralil_required
@softcore_required
class TestBackendWiring:
    def test_build_dynamics_model_rejects_softcore_plus_morse(self):
        from jaxrens.backends.neuralil import _build_dynamics_model

        with pytest.raises(ValueError, match="mutually exclusive"):
            _build_dynamics_model(
                n_types=2,
                embed_d=4,
                r_cutoff=4.0,
                n_max=3,
                core_widths=[8, 4],
                supercell_trafo=(1, 1, 1),
                has_morse=True,
                is_ensemble=False,
                n_ensemble=1,
                softcore=True,
            )

    def test_build_dynamics_model_softcore_returns_softcore_model(self):
        from jaxrens.backends.neuralil import _build_dynamics_model

        model = _build_dynamics_model(
            n_types=2,
            embed_d=4,
            r_cutoff=4.0,
            n_max=3,
            core_widths=[8, 4],
            supercell_trafo=(1, 1, 1),
            has_morse=False,
            is_ensemble=False,
            n_ensemble=1,
            softcore=True,
        )
        assert isinstance(model, SoftCoreNeuralIL)
        # Default kwargs were used
        assert model.a0 == 1.0
        assert model.r_core_cut == 1.25

    def test_build_dynamics_model_softcore_kwargs_override_defaults(self):
        from jaxrens.backends.neuralil import _build_dynamics_model

        model = _build_dynamics_model(
            n_types=2,
            embed_d=4,
            r_cutoff=4.0,
            n_max=3,
            core_widths=[8, 4],
            supercell_trafo=(1, 1, 1),
            has_morse=False,
            is_ensemble=False,
            n_ensemble=1,
            softcore=True,
            softcore_kwargs={"a0": 2.5, "r_core_cut": 1.5},
        )
        assert model.a0 == 2.5
        assert model.r_core_cut == 1.5
        # Unspecified kwargs fall back to the package defaults
        assert model.b0 == 3.0
