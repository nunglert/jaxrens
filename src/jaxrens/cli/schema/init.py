"""Pydantic schema for the [init] section of a jaxrens YAML config.

Source-of-atoms rules:
  Exactly one of {start_species, start_config_file, start_walker_set, restart_file}
  must be set.  Setting zero or more than one raises ``ValidationError``.

Species strings use the form ``"Z N[, Z N]*"`` where ``Z`` is the atomic
number and ``N`` is the count (e.g. ``"18 8"`` for 8 Ar atoms, or
``"1 6, 3 6"`` for 6 H + 6 Li).  Multi-composition cohort strings
(``"Z1 Z2: N1 N2, ..."``) are deferred.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# InitialWalkSpec (deferred: n_walks > 0 is not yet consumed by runtime)
# ---------------------------------------------------------------------------

class InitialWalkSpec(BaseModel):
    """Parameters for optional fixed-Emax burn-in walks before nested sampling.

    When ``n_walks > 0``, ``run_from_config`` runs ``initial_walk`` between
    ``init_ns`` and ``run_ns`` to decorrelate the live-walker population from
    initialization artifacts.  Burn-in is skipped for Mode D (restart) runs.

    Memory-control knob:
        ``walker_batch_size``: chunk the per-walker vmap via ``lax.map``.
            ``None`` = full vmap over all walkers (default; fastest).
            Any positive int; non-divisors of ``n_walkers`` are handled
            internally by padding.

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
            "Any positive int; non-divisors of n_walkers are padded "
            "internally."
        ),
    )


# ---------------------------------------------------------------------------
# InitSpec
# ---------------------------------------------------------------------------

def _parse_species_string(s: str) -> dict[int, int]:
    """Parse a species string into element counts.

    Accepts the form ``"Z N[, Z N]*"`` where ``Z`` is an atomic number and
    ``N`` is the count of that element.  Multiple species are separated by
    commas.

    Examples::

        "18 8"       -> {18: 8}          (8 argon atoms)
        "1 6, 3 6"   -> {1: 6, 3: 6}    (6 hydrogen + 6 lithium)

    The multi-composition cohort form ``"Z1 Z2: N1 N2, ..."`` is deferred to
    a future cohort-expansion step and raises ``ValueError``.

    Mass tokens (three-token groups ``"Z N mass"``) are not yet wired into any
    move kernel and raise ``NotImplementedError``.

    Args:
        s: Species string as described above.

    Returns:
        Mapping from atomic number to count, e.g. ``{18: 8}``.

    Raises:
        ValueError: If the string is empty, contains ``:``, contains
            non-integer tokens, or has an odd number of tokens in any
            comma-separated group.
        NotImplementedError: If a mass token is detected (three tokens in a
            group).
    """
    if not s or not s.strip():
        raise ValueError(
            f"Species string must not be empty.  Received: {s!r}"
        )

    if ":" in s:
        raise ValueError(
            f"Multi-composition species strings (containing ':') are not yet "
            f"supported.  Received: {s!r}.  Multi-composition cohort support "
            f"is deferred to a future step."
        )

    counts: dict[int, int] = {}

    for group_raw in s.split(","):
        group = group_raw.strip()
        if not group:
            raise ValueError(
                f"Empty group in species string {s!r} — check for trailing "
                f"commas or double commas."
            )
        tokens = group.split()

        if len(tokens) == 3:
            raise NotImplementedError(
                f"masses in start_species are not yet wired into any move "
                f"kernel; drop the mass tokens for now.  Full string: {s!r}"
            )

        if len(tokens) % 2 != 0 or len(tokens) == 0:
            raise ValueError(
                f"Each comma-separated group must contain an even number of "
                f"tokens (atomic_number count pairs).  Group {group!r} has "
                f"{len(tokens)} token(s).  Full string: {s!r}"
            )

        for i in range(0, len(tokens), 2):
            z_tok, n_tok = tokens[i], tokens[i + 1]
            for label, tok in (("atomic number", z_tok), ("count", n_tok)):
                if not re.fullmatch(r"\d+", tok):
                    raise ValueError(
                        f"Species string {label} token {tok!r} is not a "
                        f"non-negative integer.  Full string: {s!r}"
                    )
            z = int(z_tok)
            n = int(n_tok)
            if n == 0:
                raise ValueError(
                    f"Count for atomic number {z} is zero in species string "
                    f"{s!r}.  Every specified element must have count >= 1."
                )
            counts[z] = counts.get(z, 0) + n

    if not counts:
        raise ValueError(f"Species string {s!r} produced no atoms.")
    return counts


class InitSpec(BaseModel):
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
    initial_walk: InitialWalkSpec = Field(default_factory=InitialWalkSpec)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "InitSpec":
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

        Parses the ``"Z N[, Z N]*"`` format where ``Z`` is an atomic number
        and ``N`` is the count.  For example ``"18 8"`` yields ``{18: 8}``
        (eight argon atoms).

        Returns:
            ``{atomic_number: count}`` or ``None`` if ``start_species`` is
            not set.

        Raises:
            ValueError: If the species string cannot be parsed.
            NotImplementedError: If mass tokens are detected.
        """
        if self.start_species is None:
            return None
        return _parse_species_string(self.start_species)
