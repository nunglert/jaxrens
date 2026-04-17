"""Pydantic schema for the [termination] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields its criterion constructor
accepts.  ``to_criterion()`` is the seam between the CLI config layer and the
library core.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from jaxrens.sampling.termination import (
    EnergyTermination,
    IterationTermination,
    PriorMassTermination,
    TerminationCriterion,
    TempTermination,
)


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------

class BaseTerminationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_criterion(self) -> TerminationCriterion:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------------

class IterationTerminationSpec(BaseTerminationSpec):
    type: Literal["iteration"] = "iteration"
    max_iterations: int

    def to_criterion(self) -> TerminationCriterion:
        return IterationTermination(max_iterations=self.max_iterations)


class PriorMassTerminationSpec(BaseTerminationSpec):
    type: Literal["prior_mass"] = "prior_mass"
    n_live: int
    threshold: float = 0.1

    def to_criterion(self) -> TerminationCriterion:
        return PriorMassTermination(n_live=self.n_live, threshold=self.threshold)


class TemperatureTerminationSpec(BaseTerminationSpec):
    type: Literal["temperature"] = "temperature"
    n_walkers: int
    target_temp: float
    n_cull: int = 1
    threshold: float = 10.0

    def to_criterion(self) -> TerminationCriterion:
        return TempTermination(
            n_walkers=self.n_walkers,
            target_temp=self.target_temp,
            n_cull=self.n_cull,
            threshold=self.threshold,
        )


class EnergyTerminationSpec(BaseTerminationSpec):
    type: Literal["energy"] = "energy"
    min_energy: float

    def to_criterion(self) -> TerminationCriterion:
        return EnergyTermination(min_energy=self.min_energy)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

TerminationSpec = Annotated[
    Union[
        IterationTerminationSpec,
        PriorMassTerminationSpec,
        TemperatureTerminationSpec,
        EnergyTerminationSpec,
    ],
    Field(discriminator="type"),
]
