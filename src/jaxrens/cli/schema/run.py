"""Pydantic schema for the [run] section of a jaxrens YAML config."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RunSpec(BaseModel):
    """Mirrors NSConfig; YAML key is ``run:``.

    ``max_iterations`` is optional — when omitted the resolver auto-derives a
    sensible upper bound from the termination criteria (chiefly
    ``prior_mass.threshold`` and ``n_live``), so users who rely on
    ``prior_mass`` for actual termination don't have to pick a number.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_live: int = Field(
        default=500,
        description=(
            "Number of live walkers in the nested-sampling population. "
            "Controls resolution: the prior mass contracts by "
            "``exp(-n_cull / n_live)`` per iteration, so larger values "
            "resolve sharper phase transitions at proportionally higher "
            "cost."
        ),
    )
    # Widened to int|float so RootSpec.interval_units='per_walker' can accept
    # fractional sweeps (e.g. ``max_iterations: 2.5``).  Resolver rounds + casts.
    max_iterations: Optional[int | float] = Field(
        default=None,
        description=(
            "Hard cap on NS iterations.  ``null`` (default) lets the "
            "resolver derive a bound from the termination criteria "
            "(chiefly ``termination[prior_mass].threshold`` and "
            "``n_live``), which is what you want when ``prior_mass`` is "
            "doing the actual stopping.  Honours ``interval_units``."
        ),
    )
    convergence_threshold: float = Field(
        default=0.1,
        description=(
            "Relative log-evidence increment below which the run is "
            "considered converged.  Only consulted when no explicit "
            "``termination`` criterion fires first."
        ),
    )
    n_mcmc_steps: int = Field(
        default=20,
        description=(
            "MCMC steps taken inside the JIT-compiled inner scan to "
            "decorrelate each replacement walker from the live point it "
            "was cloned from.  Too few leaves the new walker correlated "
            "with its parent and biases the evidence; this is the main "
            "accuracy/cost dial after ``n_live``."
        ),
    )
    n_extra: int = Field(
        default=0,
        description=(
            "Additional walkers, drawn at random from the existing "
            "population, that are MCMC-walked alongside the replacement "
            "each iteration.  This is the **parallel batch width**: "
            "``1 + n_extra`` chains run under one ``vmap``, so ``0`` (the "
            "default) walks a single chain and leaves a GPU almost idle.  "
            "It does not enlarge the population — that stays ``n_live`` — "
            "and the extras are not dead points; they decorrelate the "
            "live set at the cost of more energy evaluations per iteration."
        ),
    )
    n_cull: int = Field(
        default=1,
        description=(
            "Number of highest-energy walkers removed per NS iteration.  "
            "Values > 1 contract the prior mass faster per iteration and "
            "amortise fixed per-iteration overhead, at the cost of a "
            "coarser energy grid."
        ),
    )
    seed: int = Field(
        default=42,
        description=(
            "Seed for the master PRNG key.  Fixes walker initialisation, "
            "every move proposal, and the burn-in walk, so two runs with "
            "the same config and the same device topology are identical."
        ),
    )
    shard_n_gpu: int = Field(
        default=1,
        ge=1,
        description=(
            "Shard the n_live walker population across this many GPUs.  "
            "When > 1 and the config implies a single logical run (no "
            "pressure / composition / chemical-potential list, no inter_re), "
            "the resolver builds a ShardedSingleRun batcher and the CLI "
            "dispatches to run_sharded_from_config.  "
            "n_live must be divisible by shard_n_gpu.  Default 1 → "
            "behaviour identical to today's single-GPU single run."
        ),
    )
