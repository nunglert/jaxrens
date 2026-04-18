"""Pydantic schema for the [init] section of a jaxrens YAML config.

Source-of-atoms rules:
  Exactly one of {start_species, start_config_file, start_walker_set, restart_file}
  must be set.  Setting zero or more than one raises ``ValidationError``.

Multi-composition species strings (e.g. ``"1 3: 0 16, 8 8"``) are deferred.
Only the single-composition form (e.g. ``"1 3"``) is parsed here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# InitialWalkConfig (deferred: n_walks > 0 is not yet consumed by runtime)
# ---------------------------------------------------------------------------

class InitialWalkConfig(BaseModel):
    """Parameters for optional fixed-Emax burn-in walks before nested sampling.

    When ``n_walks > 0``, ``run_from_config`` runs ``initial_walk`` between
    ``init_ns`` and ``run_ns`` to decorrelate the live-walker population from
    initialization artifacts.  Burn-in is skipped for Mode D (restart) runs.

    Memory-control knobs:
        ``walker_batch_size``: chunk the per-walker vmap via ``lax.map``.
            ``None`` = full vmap over all walkers (default; fastest).
            Must divide ``n_walkers`` evenly or ``initial_walk`` raises.
        ``run_batch_size``: chunk the per-run vmap when ``batched=True``.
            ``None`` = full vmap over all runs (default).
            Must divide ``n_runs`` evenly. Ignored for single-run configs.

    Deferred fields:
        ``write_initial_walkers``: accepted but not yet consumed by the runtime.
        ``only``: raises ``NotImplementedError`` at run time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_walks: int = 0
    walklength: int = 100
    adjust_interval: int = 10
    emax_offset_per_atom: float = 0.0
    only: str | None = None
    write_initial_walkers: bool = False
    walker_batch_size: int | None = Field(
        default=None,
        description=(
            "Chunk walkers during burn-in vmap for memory control. "
            "None = full vmap over all walkers (max speed). "
            "Must divide n_walkers evenly."
        ),
    )
    run_batch_size: int | None = Field(
        default=None,
        description=(
            "Chunk runs during batched burn-in. "
            "None = full vmap over all runs. "
            "Must divide n_runs evenly."
        ),
    )


# ---------------------------------------------------------------------------
# InitConfig
# ---------------------------------------------------------------------------

def _parse_species_string(s: str) -> dict[int, int]:
    """Parse a single-composition species string into element counts.

    Accepts the form ``"N_A N_B ..."`` where each token is an atomic number
    and duplicate tokens accumulate (e.g. ``"1 3"`` -> ``{1: 1, 3: 1}``).
    Multi-composition form (with ``:`` separator) is rejected here; it is
    deferred to a future cohort-expansion step.

    Args:
        s: Species string, e.g. ``"1 3"`` or ``"14 14 8 8 8"``.

    Returns:
        Mapping from atomic number to count.

    Raises:
        ValueError: If the string contains ``:`` (multi-composition) or
            contains non-integer tokens.
    """
    if ":" in s:
        raise ValueError(
            f"Multi-composition species strings (containing ':') are not yet "
            f"supported in step 6.  Received: {s!r}.  Defer to a future "
            f"cohort-expansion step."
        )
    tokens = s.strip().split()
    counts: dict[int, int] = {}
    for tok in tokens:
        if not re.fullmatch(r"\d+", tok):
            raise ValueError(
                f"Species string token {tok!r} is not a non-negative integer "
                f"atomic number.  Full string: {s!r}"
            )
        z = int(tok)
        counts[z] = counts.get(z, 0) + 1
    if not counts:
        raise ValueError(f"Species string {s!r} produced no atoms.")
    return counts


class InitConfig(BaseModel):
    """System-initialization parameters.

    Exactly one of {start_species, start_config_file, start_walker_set,
    restart_file} must be set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- Source of atoms (mutually exclusive) --
    start_species: str | None = None
    start_config_file: Path | None = None
    start_walker_set: Path | None = None
    restart_file: Path | None = None

    # -- Energy ceiling --
    start_energy_ceiling_per_atom: float = 1e9

    # -- Randomization --
    random_initialise_pos: bool = True
    random_initialise_cell: bool = True
    pos_randomization_mode: Literal["grid", "uniform"] = "grid"
    grid_distance: float = 1.5
    init_distance_criterion: float = 1.0
    random_init_max_n_tries: int = 100
    pos_autoscale_cells: bool = False

    # -- Burn-in (active when n_walks > 0; skipped for restart_file) --
    initial_walk: InitialWalkConfig = Field(default_factory=InitialWalkConfig)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "InitConfig":
        """Enforce that exactly one source-of-atoms field is set."""
        sources = {
            "start_species": self.start_species,
            "start_config_file": self.start_config_file,
            "start_walker_set": self.start_walker_set,
            "restart_file": self.restart_file,
        }
        set_sources = [k for k, v in sources.items() if v is not None]
        if len(set_sources) == 0:
            raise ValueError(
                "Exactly one of {start_species, start_config_file, "
                "start_walker_set, restart_file} must be set; none were set."
            )
        if len(set_sources) > 1:
            raise ValueError(
                f"Exactly one of {{start_species, start_config_file, "
                f"start_walker_set, restart_file}} must be set; got: "
                f"{set_sources!r}"
            )
        return self

    def parsed_species(self) -> dict[int, int] | None:
        """Return element-count mapping when ``start_species`` is set.

        Returns:
            ``{atomic_number: count}`` or ``None`` if ``start_species`` is
            not set.

        Raises:
            ValueError: If the species string cannot be parsed.
        """
        if self.start_species is None:
            return None
        return _parse_species_string(self.start_species)
