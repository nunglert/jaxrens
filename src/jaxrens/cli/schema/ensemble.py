"""Pydantic schema for the [ensemble] section of a jaxrens YAML config.

Each concrete spec class maps to a thermodynamic ensemble.  ``to_ensemble_params``
returns the per-cohort scalar/vector ensemble parameters (pressure, chemical
potentials, ...) consumed by ``EnsembleBackend`` at run time.

Supported ensembles
-------------------
NVT         — canonical; no additional parameters.
NPT         — isothermal-isobaric; requires ``pressure``.
semi_grand  — semi-grand μPT; requires a per-species ``chemical_potentials``
              vector and an optional ``pressure``.  The backend adds
              ``- μ·N`` (and ``+ P·V`` when a pressure is given) to the raw
              potential, i.e. ``H = U + P·V - μ·N``.

Each spec may carry a *list* on its driving parameter (pressures for NPT,
μ-vectors for semi_grand) to fan a run out across replicas — the resolver
turns that list into the replica axis.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 1 GPa = 1e9 Pa * 0.6241509e-11 eV/Å³/Pa = 0.006241509 eV/Å³
_GPA_TO_EVA3: float = 0.006241509


class BaseEnsembleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def cohort_size(self) -> int:
        """Number of replicas this ensemble drives (1 = single run).

        Overridden by specs whose driving parameter (pressure, μ) can be a
        list.  The default covers parameterless ensembles like NVT.
        """
        return 1

    def to_ensemble_params(self, *, cohort_index: int = 0) -> dict:
        """Return per-run scalar ensemble parameters for ``cohort_index``.

        Returns:
            Dict of keyword arguments passed to the ensemble backend.  Empty
            for NVT; ``{"pressure": float}`` for NPT.
        """
        raise NotImplementedError


class NVTEnsembleSpec(BaseEnsembleSpec):
    type: Literal["nvt"] = "nvt"

    def to_ensemble_params(self, *, cohort_index: int = 0) -> dict:
        return {}


class NPTEnsembleSpec(BaseEnsembleSpec):
    type: Literal["npt"] = "npt"
    pressure: float | list[float]
    pressure_units: Literal["gpa", "eva3"] = "eva3"

    def _pressure_list(self) -> list[float]:
        p = self.pressure
        return [p] if isinstance(p, (int, float)) else list(p)

    def cohort_size(self) -> int:
        return len(self._pressure_list())

    def to_ensemble_params(self, *, cohort_index: int = 0) -> dict:
        """Return ``{"pressure": float}`` in eV/Å³ for ``cohort_index``.

        Pressure is converted from GPa to eV/Å³ when ``pressure_units == "gpa"``.

        Args:
            cohort_index: Index into the pressure list; must be < cohort_size().

        Returns:
            Dict with a single ``"pressure"`` key in eV/Å³.
        """
        pressures = self._pressure_list()
        raw = pressures[cohort_index]
        if self.pressure_units == "gpa":
            raw = raw * _GPA_TO_EVA3
        return {"pressure": raw}


class SemiGrandEnsembleSpec(BaseEnsembleSpec):
    """Semi-grand μPT: ``H = U + P·V - μ·N``.

    ``chemical_potentials`` is a per-species vector ``[μ_0, μ_1, ...]``; pass a
    *list of vectors* to fan out across replicas (one μ-vector per replica).
    ``pressure`` is optional (default 0 → no ``P·V`` term) and may itself be a
    list for a pressure replica axis.  When both are lists their lengths must
    match; a scalar/single-vector broadcasts across the other's replicas.
    """

    type: Literal["semi_grand"] = "semi_grand"
    chemical_potentials: list[float] | list[list[float]]
    pressure: float | list[float] = 0.0
    pressure_units: Literal["gpa", "eva3"] = "eva3"

    def _mu_list(self) -> list[list[float]]:
        """Normalise ``chemical_potentials`` to a list of μ-vectors."""
        mu = self.chemical_potentials
        # Empty or vector-of-scalars → a single μ-vector cohort.
        if not mu or isinstance(mu[0], (int, float)):
            return [[float(v) for v in mu]]  # type: ignore[arg-type]
        return [[float(v) for v in row] for row in mu]  # type: ignore[union-attr]

    def _pressure_list(self) -> list[float]:
        p = self.pressure
        return (
            [float(p)]
            if isinstance(p, (int, float))
            else [float(x) for x in p]
        )

    @model_validator(mode="after")
    def _check_shapes(self) -> "SemiGrandEnsembleSpec":
        mus = self._mu_list()
        n_species = {len(row) for row in mus}
        if len(n_species) != 1:
            raise ValueError(
                "ensemble.chemical_potentials: every μ-vector must have the "
                f"same length (n_species); got lengths {sorted(n_species)}."
            )
        if n_species == {0}:
            raise ValueError(
                "ensemble.chemical_potentials: n_species must be >= 1."
            )
        n_mu, n_p = len(mus), len(self._pressure_list())
        if n_mu > 1 and n_p > 1 and n_mu != n_p:
            raise ValueError(
                "ensemble: chemical_potentials and pressure are both "
                f"list-valued but disagree in length ({n_mu} vs {n_p}); a "
                "scalar broadcasts, otherwise the lengths must match."
            )
        return self

    def cohort_size(self) -> int:
        """Number of replicas this spec drives (1 = single run)."""
        return max(len(self._mu_list()), len(self._pressure_list()))

    def to_ensemble_params(self, *, cohort_index: int = 0) -> dict:
        """Return ``{"chemical_potentials": [...], "pressure": <eV/Å³>}``.

        ``pressure`` is converted from GPa to eV/Å³ when
        ``pressure_units == "gpa"``.  Scalar/single-vector parameters broadcast
        across ``cohort_index``.
        """
        mus = self._mu_list()
        pressures = self._pressure_list()
        mu = mus[cohort_index] if len(mus) > 1 else mus[0]
        raw_p = pressures[cohort_index] if len(pressures) > 1 else pressures[0]
        if self.pressure_units == "gpa":
            raw_p = raw_p * _GPA_TO_EVA3
        return {"chemical_potentials": list(mu), "pressure": raw_p}


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

EnsembleSpec = Annotated[
    Union[NVTEnsembleSpec, NPTEnsembleSpec, SemiGrandEnsembleSpec],
    Field(discriminator="type"),
]
