"""Ensemble corrections (PV, μN) as an additive Hamiltonian term.

Adds thermodynamic potential terms to the raw model energy::

    NVT:  H = U                    (no term)
    NPT:  H = U + P*V
    μPT:  H = U + P*V - μ·N

Usage::

    model = HarmonicBackend(k=1.0)
    backend = Hamiltonian(model, [EnsembleTerm(pressure=0.01)])
    result = backend(positions, species, cell, max_neighbors)
    # result.energy = U + P*V

For per-run vmap with different pressures or chemical potentials, pass
``ensemble_params`` (keys ``"pressure"`` and ``"chemical_potentials"``)::

    backend(pos, species, cell, mn, ensemble_params={"pressure": 0.02})
    backend(pos, species, cell, mn,
            ensemble_params={"chemical_potentials": mu})  # (n_species,)
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jaxrens.backends.hamiltonian import EnergyTerm, Hamiltonian
from jaxrens.utils.cell import get_volume


class EnsembleTerm(EnergyTerm):
    """``P·V − μ·N`` ensemble correction.

    The closured ``pressure`` / ``chemical_potentials`` are defaults; a
    per-call ``ensemble_params`` dict overrides them, which lets different
    runs carry different pressures / chemical potentials under one term.

    Both contributions are independent of atomic positions (``V`` depends on
    the cell, ``N`` on the species), so the forces are exactly zero.
    """

    def __init__(
        self,
        pressure: float = 0.0,
        chemical_potentials: jnp.ndarray | None = None,
    ):
        self.pressure = pressure
        self.chemical_potentials = chemical_potentials

    def energy(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        ensemble_params: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        # Use per-run params if provided, else closured defaults
        pressure = self.pressure
        mu = self.chemical_potentials
        if ensemble_params is not None:
            pressure = ensemble_params.get("pressure", pressure)
            mu = ensemble_params.get("chemical_potentials", mu)

        E = pressure * get_volume(cell)

        if mu is not None:
            n_species = mu.shape[0]
            if n_species > 0:
                counts = jnp.zeros(n_species, dtype=species.dtype)
                counts = counts.at[species].add(1)
                E = E - jnp.dot(mu, counts.astype(jnp.float32))

        return E

    def energy_and_forces(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Analytic: position-independent, so forces are zero."""
        E = self.energy(positions, species, cell, ensemble_params)
        return E, jnp.zeros_like(positions)


def EnsembleBackend(
    base: Any,
    pressure: float = 0.0,
    chemical_potentials: jnp.ndarray | None = None,
) -> Hamiltonian:
    """Convenience: ``base`` plus an :class:`EnsembleTerm`.

    Returns a flat :class:`Hamiltonian` (``base`` may itself be one; its
    terms are kept and the ensemble term appended).
    """
    return Hamiltonian(base, [EnsembleTerm(pressure, chemical_potentials)])


def make_ensemble_params(
    pressure: float = 0.0,
    chemical_potentials: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Create ensemble_params dict for MCState.

    Returns a dict suitable for storing on MCState.ensemble_params
    and passing to an :class:`EnsembleTerm` via the ensemble_params kwarg.
    """
    params: dict[str, jnp.ndarray] = {"pressure": jnp.asarray(pressure)}
    if chemical_potentials is not None:
        params["chemical_potentials"] = jnp.asarray(chemical_potentials)
    return params
