"""Hamiltonian — one energy model plus a flat tuple of additive energy terms.

The sampled Hamiltonian is composed, not nested::

    H = U_model + sum_k E_term_k

where ``U_model`` is a bare :class:`~jaxrens.backends.base.EnergyBackend`
(MACE, NeuralIL, LJ, …) and each term is a model-independent
:class:`EnergyTerm` (soft-core repulsion, ensemble ``P·V − μ·N``, …).

Every layer contributes its **own** energy and forces, which are summed:

- the model uses its native ``energy_and_forces`` when it has one (e.g.
  NeuralIL's ``calc_potential_energy_and_forces``), and autodiff of its
  energy otherwise — via :func:`~jaxrens.backends.base.eval_energy_and_forces`;
- each term returns ``(energy, forces)`` from :meth:`EnergyTerm.energy_and_forces`,
  which defaults to autodiff of that term alone and can be overridden with
  analytic forces (e.g. zeros for the position-independent ensemble terms).

Nothing is differentiated through a layer it does not own.

There is deliberately **no** ``__getattr__`` forwarding to the model.
Forwarding made earlier wrapper backends impersonate the model for every
attribute — including optional capabilities such as ``energy_and_forces``,
whose meaning changes once a term is added on top — so a native-force
lookup on the wrapper silently skipped the wrapper's own terms.  Model
metadata (``atomic_numbers``, ``max_neighbors_for``, ``is_ensemble``, …)
is reached explicitly via ``hamiltonian.model`` or :func:`unwrap_model`.

Usage::

    from jaxrens.backends.ensemble import EnsembleTerm
    from jaxrens.backends.softcore import SoftCoreTerm

    H = Hamiltonian(model, [SoftCoreTerm(), EnsembleTerm(pressure=P)])
    H(positions, species, cell, max_neighbors).energy   # U + E_core + P·V
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.backends.base import (
    BackendResult,
    EnergyBackend,
    eval_energy_and_forces,
)


class EnergyTerm:
    """Additive, model-independent contribution to the Hamiltonian.

    Subclasses implement :meth:`energy`.  :meth:`energy_and_forces` defaults
    to reverse-mode autodiff of :meth:`energy` w.r.t. ``positions`` only;
    override it to supply analytic forces.

    ``ensemble_params`` is the per-call (per-run) parameter dict threaded
    from ``MCState.ensemble_params``; terms that do not use it ignore it.
    """

    def energy(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        ensemble_params: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        """This term's scalar energy contribution."""
        raise NotImplementedError

    def energy_and_forces(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """This term's ``(energy, forces)``, ``forces = -dE/dx`` of this term only."""
        e, g = jax.value_and_grad(self.energy)(
            positions, species, cell, ensemble_params
        )
        return e, -g


class Hamiltonian:
    """Energy model plus additive terms; satisfies the EnergyBackend protocol.

    Args:
        model: The bare energy backend.  If a :class:`Hamiltonian` is passed,
            it is flattened: its model is used and its terms are prepended to
            ``terms``, so composition never nests.
        terms: Additive :class:`EnergyTerm` instances, summed in order after
            the model energy.
    """

    def __init__(self, model: EnergyBackend, terms: Sequence[EnergyTerm] = ()):
        if isinstance(model, Hamiltonian):
            terms = (*model.terms, *terms)
            model = model.model
        self.model: EnergyBackend = model
        self.terms: tuple[EnergyTerm, ...] = tuple(terms)
        self.r_cutoff = model.r_cutoff

    def with_terms(self, *terms: EnergyTerm) -> Hamiltonian:
        """Return a new Hamiltonian with ``terms`` appended."""
        return Hamiltonian(self.model, (*self.terms, *terms))

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Model energy plus every term's energy. Control fields pass through."""
        res = self.model(positions, species, cell, max_neighbors)
        energy = res.energy
        for term in self.terms:
            energy = energy + term.energy(
                positions, species, cell, ensemble_params
            )
        return res._replace(energy=energy)

    def energy_and_forces(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Per-layer energies and forces, summed.

        The model goes through :func:`eval_energy_and_forces` (native force
        path when available, autodiff of the model alone otherwise); each term
        contributes its own :meth:`EnergyTerm.energy_and_forces`.
        """
        res = eval_energy_and_forces(
            self.model, positions, species, cell, max_neighbors
        )
        energy, forces = res.energy, res.forces
        for term in self.terms:
            e, f = term.energy_and_forces(
                positions, species, cell, ensemble_params
            )
            energy = energy + e
            forces = forces + f
        return res._replace(energy=energy, forces=forces)


def unwrap_model(backend: EnergyBackend) -> EnergyBackend:
    """The bare energy model behind ``backend`` (identity for a bare backend)."""
    if isinstance(backend, Hamiltonian):
        return backend.model
    return backend
