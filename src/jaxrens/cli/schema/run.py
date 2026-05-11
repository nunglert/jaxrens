"""Pydantic schema for the [run] section of a jaxrens YAML config."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class RunSpec(BaseModel):
    """Mirrors NSConfig; YAML key is ``run:``.

    ``max_iterations`` is optional — when omitted the resolver auto-derives a
    sensible upper bound from the termination criteria (chiefly
    ``prior_mass.threshold`` and ``n_live``), so users who rely on
    ``prior_mass`` for actual termination don't have to pick a number.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_live: int = 500
    # Widened to int|float so RootSpec.interval_units='per_walker' can accept
    # fractional sweeps (e.g. ``max_iterations: 2.5``).  Resolver rounds + casts.
    max_iterations: Optional[int | float] = None
    convergence_threshold: float = 0.1
    n_mcmc_steps: int = 20
    n_extra: int = 0
    n_cull: int = 1
    seed: int = 42
    pressure: Optional[float] = None
