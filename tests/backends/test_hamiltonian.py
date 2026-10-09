"""Tests for the flat Hamiltonian = model + additive EnergyTerms composition.

Regression target: wrapper backends used to forward attribute access to
their base via ``__getattr__``, so ``eval_energy_and_forces`` found the
*model's* native ``energy_and_forces`` on the wrapper and silently dropped
the wrapper's own terms (soft core, ``P·V − μ·N``) from both energy and
forces.  The Hamiltonian composes per-layer energies and forces instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult, eval_energy_and_forces
from jaxrens.backends.ensemble import EnsembleBackend, EnsembleTerm
from jaxrens.backends.hamiltonian import EnergyTerm, Hamiltonian, unwrap_model
from jaxrens.backends.softcore import SoftCoreBackend, SoftCoreTerm
from jaxrens.backends.toy import HarmonicBackend
from jaxrens.utils.cell import get_volume


class _NativeForcesHarmonic(HarmonicBackend):
    """Harmonic model with a native ``energy_and_forces``.

    ``force_offset`` lets a test make the native forces deliberately differ
    from autodiff of the energy, to prove the native path is the one used.
    """

    def __init__(self, k: float = 1.0, force_offset: float = 0.0):
        super().__init__(k=k)
        self.force_offset = force_offset
        self.native_calls = 0

    def energy_and_forces(
        self, positions, species, cell, max_neighbors, ensemble_params=None
    ) -> BackendResult:
        self.native_calls += 1
        energy = 0.5 * self.k * jnp.sum(positions**2)
        forces = -self.k * positions + self.force_offset
        return BackendResult(energy=energy, forces=forces)


def _config():
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.6, 0.1, 0.0], [2.5, 0.3, 0.4]],
        dtype=jnp.float32,
    )
    species = jnp.array([0, 1, 1], dtype=jnp.int32)
    cell = 6.0 * jnp.eye(3, dtype=jnp.float32)
    return positions, species, cell


def _autodiff_reference(backend, positions, species, cell, ensemble_params):
    def energy_of(pos):
        return backend(
            pos, species, cell, 0, ensemble_params=ensemble_params
        ).energy

    e, g = jax.value_and_grad(energy_of)(positions)
    return e, -g


class TestComposedForces:
    @pytest.mark.parametrize(
        "ensemble_params",
        [
            None,
            {"pressure": 0.3, "chemical_potentials": jnp.array([0.2, -0.5])},
        ],
    )
    def test_stack_matches_full_autodiff(self, ensemble_params):
        """Model native forces + per-term forces == autodiff of the full H."""
        model = _NativeForcesHarmonic(k=1.5)
        H = Hamiltonian(model, [SoftCoreTerm(), EnsembleTerm(pressure=0.1)])
        positions, species, cell = _config()

        res = eval_energy_and_forces(
            H, positions, species, cell, 0, ensemble_params=ensemble_params
        )
        e_ref, f_ref = _autodiff_reference(
            H, positions, species, cell, ensemble_params
        )

        assert model.native_calls == 1
        assert float(res.energy) == pytest.approx(float(e_ref), rel=1e-5)
        assert jnp.allclose(res.forces, f_ref, rtol=1e-5, atol=1e-5)

    def test_energy_includes_every_term(self):
        """Energy on the force path = U + E_core + P·V − μ·N (the old bug
        returned bare U here)."""
        model = _NativeForcesHarmonic(k=1.0)
        mu = jnp.array([0.2, -0.5])
        H = EnsembleBackend(
            SoftCoreBackend(model), pressure=0.25, chemical_potentials=mu
        )
        positions, species, cell = _config()

        res = eval_energy_and_forces(H, positions, species, cell, 0)

        U = 0.5 * jnp.sum(positions**2)
        E_core = SoftCoreTerm().energy(positions, species, cell)
        assert float(E_core) > 0.0  # the 0.6 Å pair is inside the core
        expected = U + E_core + 0.25 * get_volume(cell) - (0.2 * 1 - 0.5 * 2)
        assert float(res.energy) == pytest.approx(float(expected), rel=1e-5)
        assert float(res.energy) == pytest.approx(
            float(H(positions, species, cell, 0).energy), rel=1e-6
        )

    def test_model_native_forces_not_differentiated(self):
        """The model's own forces are used as-is (no autodiff through it)."""
        model = _NativeForcesHarmonic(k=1.0, force_offset=7.0)
        term = SoftCoreTerm()
        H = Hamiltonian(model, [term, EnsembleTerm(pressure=0.1)])
        positions, species, cell = _config()

        res = eval_energy_and_forces(H, positions, species, cell, 0)

        _, f_core = term.energy_and_forces(positions, species, cell)
        expected = -positions + 7.0 + f_core
        assert jnp.allclose(res.forces, expected, atol=1e-5)

    def test_model_without_native_forces_autodiffs(self):
        model = HarmonicBackend(k=2.0)
        H = Hamiltonian(model, [SoftCoreTerm()])
        positions, species, cell = _config()

        res = eval_energy_and_forces(H, positions, species, cell, 0)
        e_ref, f_ref = _autodiff_reference(H, positions, species, cell, None)

        assert float(res.energy) == pytest.approx(float(e_ref), rel=1e-5)
        assert jnp.allclose(res.forces, f_ref, rtol=1e-5, atol=1e-5)

    def test_softcore_forces_finite_at_close_contact(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=jnp.float32
        )
        species = jnp.zeros(2, dtype=jnp.int32)
        cell = 6.0 * jnp.eye(3, dtype=jnp.float32)

        _, f = SoftCoreTerm().energy_and_forces(positions, species, cell)

        assert bool(jnp.all(jnp.isfinite(f)))
        # Repulsive: atom 0 pushed to -x, atom 1 to +x, equal and opposite.
        assert float(f[0, 0]) < 0.0 < float(f[1, 0])
        assert jnp.allclose(f[0], -f[1], atol=1e-5)

    def test_ensemble_term_forces_are_zero(self):
        positions, species, cell = _config()
        term = EnsembleTerm(pressure=0.5, chemical_potentials=jnp.ones(2))
        e, f = term.energy_and_forces(positions, species, cell)
        assert jnp.array_equal(f, jnp.zeros_like(positions))
        assert float(e) == pytest.approx(
            float(0.5 * get_volume(cell) - 3.0), rel=1e-6
        )

    def test_jit(self):
        model = _NativeForcesHarmonic(k=1.0)
        H = Hamiltonian(model, [SoftCoreTerm(), EnsembleTerm(pressure=0.1)])
        positions, species, cell = _config()
        ep = {"pressure": jnp.asarray(0.2)}

        eager = eval_energy_and_forces(
            H, positions, species, cell, 0, ensemble_params=ep
        )
        jitted = jax.jit(
            lambda p, e: eval_energy_and_forces(
                H, p, species, cell, 0, ensemble_params=e
            )
        )(positions, ep)

        assert float(jitted.energy) == pytest.approx(
            float(eager.energy), rel=1e-6
        )
        assert jnp.allclose(jitted.forces, eager.forces, atol=1e-6)


class TestComposition:
    def test_nesting_is_flattened(self):
        model = HarmonicBackend()
        a, b = SoftCoreTerm(), EnsembleTerm(pressure=0.1)
        H = Hamiltonian(Hamiltonian(model, [a]), [b])
        assert H.model is model
        assert H.terms == (a, b)

    def test_convenience_factories_compose_flat(self):
        model = HarmonicBackend()
        H = EnsembleBackend(SoftCoreBackend(model), pressure=0.1)
        assert isinstance(H, Hamiltonian)
        assert H.model is model
        assert [type(t) for t in H.terms] == [SoftCoreTerm, EnsembleTerm]

    def test_with_terms_does_not_mutate(self):
        H = Hamiltonian(HarmonicBackend())
        H2 = H.with_terms(SoftCoreTerm())
        assert H.terms == ()
        assert len(H2.terms) == 1

    def test_no_attribute_forwarding(self):
        """Model attributes are not impersonated; reach them via ``.model``."""
        model = _NativeForcesHarmonic(k=3.0)
        H = Hamiltonian(model, [SoftCoreTerm()])
        assert H.r_cutoff == model.r_cutoff
        with pytest.raises(AttributeError):
            H.k  # noqa: B018
        # energy_and_forces is the Hamiltonian's own, not the model's.
        assert H.energy_and_forces.__func__ is Hamiltonian.energy_and_forces
        assert unwrap_model(H) is model
        assert unwrap_model(model) is model

    def test_custom_term_default_autodiff(self):
        class Quadratic(EnergyTerm):
            def energy(self, positions, species, cell, ensemble_params=None):
                return jnp.sum(positions**2)

        positions, species, cell = _config()
        e, f = Quadratic().energy_and_forces(positions, species, cell)
        assert float(e) == pytest.approx(float(jnp.sum(positions**2)))
        assert jnp.allclose(f, -2.0 * positions)
