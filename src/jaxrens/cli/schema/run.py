"""Pydantic schema for the [run] section of a jaxrens YAML config."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class RunSchema(BaseModel):
    """Mirrors NSConfig; YAML key is ``run:``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_live: int = 500
    max_iterations: int = 50_000
    convergence_threshold: float = 0.1
    n_mcmc_steps: int = 20
    n_cull: int = 1
    seed: int = 42
    pressure: Optional[float] = None
