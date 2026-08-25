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

    n_walks: int = Field(
        default=0,
        description=(
            "Number of fixed-``E_max`` burn-in walks run between "
            "initialisation and nested sampling, to decorrelate the live "
            "population from grid or rejection-sampling artefacts.  ``0`` "
            "(default) skips burn-in; it is always skipped on restart."
        ),
    )
    walklength: int = Field(
        default=100,
        description=(
            "MCMC steps per burn-in walk.  Together with ``n_walks`` this "
            "sets the total burn-in cost."
        ),
    )
    adjust_interval: int = Field(
        default=10,
        description=(
            "Step-size adjustment cadence *within* burn-in, in walk "
            "steps.  Independent of ``adaptation.adjust_interval``, which "
            "governs the sampling run."
        ),
    )
    emax_offset_per_atom: float = Field(
        default=0.0,
        description=(
            "Raise the fixed burn-in energy ceiling by this much per atom "
            "above the initial maximum walker energy.  A small positive "
            "offset gives the burn-in room to move when the initial "
            "population sits right at the ceiling."
        ),
    )
    only: str | None = Field(
        default=None,
        description=(
            "Restrict burn-in to a single named move.  **Deferred** — "
            "raises ``NotImplementedError`` at run time."
        ),
    )
    write_initial_walkers: bool = Field(
        default=False,
        description=(
            "Dump the post-burn-in walker population before sampling "
            "starts.  **Deferred** — accepted but not yet consumed by the "
            "runtime."
        ),
    )
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
        raise ValueError(f"Species string must not be empty.  Received: {s!r}")

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
        raise ValueError(
            f"Species string {s!r} produced no atoms. It must name at least "
            f"one element with a positive count, e.g. 'Si16' or 'Ge8Si8'."
        )
    return counts


class InitSpec(BaseModel):
    """System-initialization parameters.

    Exactly one of {start_species, start_config_file, start_walker_set,
    restart_file} must be set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- Source of atoms (mutually exclusive) --
    start_species: str | None = Field(
        default=None,
        description=(
            "Build the system from a composition string of the form "
            '``"Z N[, Z N]*"``, where ``Z`` is an atomic number and '
            '``N`` a count — e.g. ``"18 8"`` for 8 argon atoms, '
            '``"1 6, 3 6"`` for 6 H plus 6 Li.  Positions and cell are '
            "then generated per the randomisation settings below."
        ),
    )
    start_config_file: Path | None = Field(
        default=None,
        description=(
            "Read one structure from this file (any ASE-readable format) "
            "and replicate it across all walkers."
        ),
    )
    start_walker_set: Path | None = Field(
        default=None,
        description=(
            "Read a full, pre-built walker population from this file — one "
            "frame per walker.  The frame count must match ``run.n_live`` "
            "(plus ``run.n_extra``)."
        ),
    )
    restart_file: Path | None = Field(
        default=None,
        description=(
            "Resume from this checkpoint.  Restores walkers, iteration "
            "counter, and log-evidence; burn-in is skipped and the "
            "compatibility validator checks the config against what the "
            "checkpoint was written with."
        ),
    )

    # -- Energy ceiling --
    start_energy_ceiling_per_atom: float = Field(
        default=1e9,
        description=(
            "Reject initial configurations whose energy per atom exceeds "
            "this value, in the backend's energy units.  Guards against "
            "seeding the population with overlapping atoms.  The default "
            "is effectively no ceiling; lower it when initialisation keeps "
            "producing unphysical starts."
        ),
    )

    # -- Randomization --
    random_initialise_pos: bool = Field(
        default=True,
        description=(
            "Draw fresh random positions for every walker.  Set ``false`` "
            "to keep the positions exactly as read from "
            "``start_config_file`` / ``start_walker_set``."
        ),
    )
    random_initialise_cell: bool = Field(
        default=True,
        description=(
            "Draw a fresh random cell for every walker, within the "
            "``cell:`` volume and aspect-ratio bounds.  Set ``false`` to "
            "keep the input cell."
        ),
    )
    cell_randomization_mode: Literal["shape_walk", "linear_1d"] = Field(
        default="shape_walk",
        description=(
            "How random cells are drawn.  ``shape_walk`` (default) runs a "
            "shear/stretch random walk per walker, giving general triclinic "
            "cells.  ``linear_1d`` instead draws a box length ``a`` per "
            "walker and builds ``diag(a, 1, 1)``, the embedding the "
            "``rens_toy`` backend needs so that ``det(cell) == a``; the draw "
            "spans ``cell.initial_min_volume_per_atom`` (falling back to "
            "``min_volume_per_atom``) up to ``cell.max_volume_per_atom``, "
            "which is how the RENS paper biases its initial population "
            "toward the high-enthalpy end."
        ),
    )
    pos_randomization_mode: Literal["grid", "uniform"] = Field(
        default="grid",
        description=(
            "How random positions are drawn.  ``grid`` (default) places "
            "atoms on a jittered lattice with spacing ``grid_distance``, "
            "which reliably avoids overlaps; ``uniform`` draws uniformly "
            "in the cell and relies on rejection against "
            "``init_distance_criterion``."
        ),
    )
    grid_distance: float = Field(
        default=1.5,
        description=(
            "Lattice spacing for ``pos_randomization_mode: grid`` "
            "(Angstrom).  Must be large enough that the grid fits the "
            "atoms inside the initial cell."
        ),
    )
    init_distance_criterion: float = Field(
        default=1.0,
        description=(
            "Minimum interatomic distance a drawn configuration must "
            "satisfy to be accepted (Angstrom).  Distinct from the "
            "``constraints:`` section, which is enforced for the whole "
            "run rather than only at initialisation."
        ),
    )
    random_init_max_n_tries: int = Field(
        default=100,
        description=(
            "Rejection-sampling attempts per walker before initialisation "
            "gives up and raises.  Raise it, loosen "
            "``init_distance_criterion``, or start at a larger volume if "
            "you hit the limit."
        ),
    )
    pos_autoscale_cells: bool = Field(
        default=False,
        description=(
            "Rescale each walker's cell so the drawn positions fit inside "
            "it, instead of rejecting draws that do not fit.  Helpful for "
            "dense compositions where the grid barely fits."
        ),
    )

    # -- Burn-in (active when n_walks > 0; skipped for restart_file) --
    initial_walk: InitialWalkSpec = Field(
        default_factory=InitialWalkSpec,
        description=(
            "Fixed-``E_max`` burn-in settings, active when "
            "``initial_walk.n_walks > 0``."
        ),
    )

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
