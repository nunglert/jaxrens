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
    TempTermination,
    TerminationCriterion,
)

# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------


class BaseTerminationSpec(BaseModel):
    """Fields shared by every termination criterion.

    ``to_criterion`` takes ``n_live`` and ``n_cull`` from the resolver so
    criteria that need walker-count information (e.g. prior-mass,
    temperature) don't redundantly re-declare it in the YAML.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_criterion(
        self,
        *,
        n_live: int | None = None,
        n_cull: int | None = None,
    ) -> TerminationCriterion:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------------


class IterationTerminationSpec(BaseTerminationSpec):
    type: Literal["iteration"] = Field(
        default="iteration",
        description="Discriminator selecting this criterion.",
    )
    # Widened to int|float so RootSpec.interval_units='per_walker' can express
    # the cap in walker-sweeps; resolver scales+casts before to_criterion.
    max_iterations: int | float = Field(
        description=(
            "Stop once the NS iteration counter reaches this value.  "
            "Honours ``interval_units``, so in ``per_walker`` mode it is a "
            "number of walker-sweeps."
        ),
    )

    def to_criterion(
        self,
        *,
        n_live: int | None = None,
        n_cull: int | None = None,
    ) -> TerminationCriterion:
        return IterationTermination(max_iterations=int(self.max_iterations))


class PriorMassTerminationSpec(BaseTerminationSpec):
    type: Literal["prior_mass"] = Field(
        default="prior_mass",
        description="Discriminator selecting this criterion.",
    )
    threshold: float = Field(
        default=0.1,
        description=(
            "Stop once the remaining prior mass ``X_i = exp(-i / n_live)`` "
            "falls below this fraction.  The usual way to terminate: it "
            "adapts the iteration count to ``n_live`` instead of pinning "
            "it.  Smaller values sample further into the low-energy tail."
        ),
    )

    def to_criterion(
        self,
        *,
        n_live: int | None = None,
        n_cull: int | None = None,
    ) -> TerminationCriterion:
        if n_live is None:
            raise ValueError(
                "PriorMassTerminationSpec.to_criterion() requires n_live. "
                "Call to_criterion(n_live=...) from the resolver with run.n_live."
            )
        return PriorMassTermination(n_live=n_live, threshold=self.threshold)


class TemperatureTerminationSpec(BaseTerminationSpec):
    type: Literal["temperature"] = Field(
        default="temperature",
        description="Discriminator selecting this criterion.",
    )
    target_temp: float = Field(
        description=(
            "Stop once the finite-difference NS temperature estimate has "
            "cooled to this value (Kelvin).  Useful when you care about "
            "reaching a physical temperature rather than an iteration "
            "count."
        ),
    )
    threshold: float = Field(
        default=10.0,
        description=(
            "Tolerance band around ``target_temp`` (Kelvin).  The run "
            "stops when the estimate first falls within it, absorbing the "
            "noise in the finite-difference estimator."
        ),
    )

    def to_criterion(
        self,
        *,
        n_live: int | None = None,
        n_cull: int | None = None,
    ) -> TerminationCriterion:
        if n_live is None or n_cull is None:
            raise ValueError(
                "TemperatureTerminationSpec.to_criterion() requires n_live and n_cull. "
                "Call to_criterion(n_live=..., n_cull=...) from the resolver."
            )
        return TempTermination(
            n_walkers=n_live,
            target_temp=self.target_temp,
            n_cull=n_cull,
            threshold=self.threshold,
        )


class EnergyTerminationSpec(BaseTerminationSpec):
    type: Literal["energy"] = Field(
        default="energy",
        description="Discriminator selecting this criterion.",
    )
    min_energy: float = Field(
        description=(
            "Stop once the highest live-walker energy (the NS ``E_max`` "
            "threshold) drops below this value, in the backend's own "
            "energy units.  Backend-specific — a hard floor, not a "
            "portable one."
        ),
    )

    def to_criterion(
        self,
        *,
        n_live: int | None = None,
        n_cull: int | None = None,
    ) -> TerminationCriterion:
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
