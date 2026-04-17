"""Pydantic schema for the [ensemble] section of a jaxrens YAML config.

Each concrete spec class maps to a thermodynamic ensemble.  ``to_ensemble_params``
returns the per-cohort scalar ensemble parameters (pressure, etc.) needed at run
time.

Supported ensembles
-------------------
NVT  — canonical; no additional parameters.
NPT  — isothermal-isobaric; requires ``pressure``.

Deferred (runtime does not yet have chemical-potential machinery):
    muVT  — grand canonical.
    semi-grand — composition-preserving grand canonical.
    These will be added once the sampling layer implements mu-based acceptance.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# 1 GPa = 1e9 Pa * 0.6241509e-11 eV/Å³/Pa = 0.006241509 eV/Å³
_GPA_TO_EVA3: float = 0.006241509


class BaseEnsembleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

EnsembleSpec = Annotated[
    Union[NVTEnsembleSpec, NPTEnsembleSpec],
    Field(discriminator="type"),
]
