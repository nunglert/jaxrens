"""Pydantic schema for the optional [inter_re] section of a jaxrens YAML config."""

from __future__ import annotations

from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from jaxrens.state.config import InterREConfig

_IMPLEMENTED_FLAVORS = frozenset({"pressure", "xrens", "semi_grand"})


class InterRESpec(BaseModel):
    """YAML → pydantic → InterREConfig for the ``inter_re:`` section.

    The ``"pressure"``, ``"xrens"``, and ``"semi_grand"`` flavors are
    implemented.

    For ``flavor: xrens``, ``composition_targets`` is required: a list of
    per-run target compositions (one list-of-ints per run).  Each row must
    sum to the same ``n_atoms`` value (validated at runtime when ``n_atoms``
    and ``n_runs`` are known).

    For ``flavor: semi_grand``, ``chemical_potentials`` is required: a list
    of per-run per-species chemical potential vectors (one list-of-floats per
    run).  All rows must have the same length (= n_species).

    Example YAML (pressure)::

        inter_re:
          flavor: pressure
          re_interval: 1
          n_swap_cycles: 1

    Example YAML (XRENS with 2 runs, 8 atoms each, 2 species)::

        inter_re:
          flavor: xrens
          re_interval: 1
          n_swap_cycles: 1
          composition_targets:
            - [8, 0]   # run 0: 8 atoms of species 0, 0 of species 1
            - [4, 4]   # run 1: 50/50 mix

    Example YAML (semi-grand with 2 runs, 2 species)::

        inter_re:
          flavor: semi_grand
          re_interval: 1
          n_swap_cycles: 1
          chemical_potentials:
            - [0.0, 0.0]   # run 0: μ for each species
            - [0.5, 1.0]   # run 1
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    flavor: str = Field(
        default="pressure",
        description=(
            "Which exchange protocol to run.  ``pressure`` swaps walkers "
            "between replicas held at different pressures (pressure-RENS); "
            "``xrens`` swaps between different target compositions and "
            "requires ``composition_targets``; ``semi_grand`` swaps "
            "between chemical-potential vectors and requires "
            "``chemical_potentials``."
        ),
    )
    # Widened to int|float for RootSpec.interval_units='per_walker'; resolver
    # scales+casts to int before constructing InterREConfig.
    re_interval: int | float = Field(
        default=1,
        description=(
            "Fire a swap attempt every N NS iterations.  Honours "
            "``interval_units``; in ``per_walker`` mode values below ~0.05 "
            "sweeps are almost always a mistake and are warned about."
        ),
    )
    n_swap_cycles: int = Field(
        default=1,
        description=(
            "Swap cycles performed per fire.  Each cycle offers every "
            "adjacent replica pair once, so more cycles move walkers "
            "further along the ladder per fire."
        ),
    )
    composition_targets: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "Required for ``flavor: xrens``: one target composition per "
            "replica, as per-species atom counts (e.g. "
            "``[[8, 0], [4, 4]]`` for two replicas of an 8-atom binary "
            "system).  Every row must have the same length and sum.  "
            "Ignored — with a warning — by the other flavors."
        ),
    )
    chemical_potentials: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "Required for ``flavor: semi_grand``: one per-species "
            "chemical-potential vector per replica (eV), e.g. "
            "``[[0.0, 0.0], [0.5, 1.0]]``.  All rows must have the same "
            "length.  Ignored — with a warning — by the other flavors."
        ),
    )

    @field_validator("flavor")
    @classmethod
    def _check_flavor(cls, v: str) -> str:
        if v not in _IMPLEMENTED_FLAVORS:
            raise NotImplementedError(
                f"inter_re flavor {v!r} is not yet implemented. "
                f"Implemented flavors: {sorted(_IMPLEMENTED_FLAVORS)}."
            )
        return v

    @model_validator(mode="after")
    def _check_flavor_fields(self) -> "InterRESpec":
        """Validate flavor-specific required fields."""
        if self.flavor == "xrens":
            if self.composition_targets is None:
                raise ValueError(
                    "inter_re flavor 'xrens' requires 'composition_targets': "
                    "a list of per-run target compositions, one list-of-ints "
                    "per run (e.g. [[8, 0], [4, 4]] for 2 runs, 2 species, "
                    "8 atoms each)."
                )
            # Check all rows have the same length (n_species).
            n_species_set = {len(row) for row in self.composition_targets}
            if len(n_species_set) > 1:
                raise ValueError(
                    f"inter_re.composition_targets: all rows must have the same "
                    f"length (n_species), got lengths {sorted(n_species_set)}."
                )
            # Check all rows sum to the same value (n_atoms).
            row_sums = [sum(row) for row in self.composition_targets]
            sum_set = set(row_sums)
            if len(sum_set) > 1:
                raise ValueError(
                    f"inter_re.composition_targets: all rows must sum to the same "
                    f"n_atoms, got row sums {row_sums}."
                )
            # Warn if chemical_potentials is also set.
            if self.chemical_potentials is not None:
                import warnings

                warnings.warn(
                    "inter_re.chemical_potentials is set but flavor='xrens'. "
                    "chemical_potentials will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        elif self.flavor == "semi_grand":
            if self.chemical_potentials is None:
                raise ValueError(
                    "inter_re flavor 'semi_grand' requires 'chemical_potentials': "
                    "a list of per-run per-species chemical potentials, one "
                    "list-of-floats per run (e.g. [[0.0, 0.0], [0.5, 1.0]] "
                    "for 2 runs, 2 species)."
                )
            # Check all rows have the same length (n_species).
            n_species_set = {len(row) for row in self.chemical_potentials}
            if len(n_species_set) > 1:
                raise ValueError(
                    f"inter_re.chemical_potentials: all rows must have the same "
                    f"length (n_species), got lengths {sorted(n_species_set)}."
                )
            if len(n_species_set) == 1 and next(iter(n_species_set)) < 1:
                raise ValueError(
                    "inter_re.chemical_potentials: n_species must be >= 1, "
                    f"got {next(iter(n_species_set))}."
                )
            # Warn if composition_targets is also set.
            if self.composition_targets is not None:
                import warnings

                warnings.warn(
                    "inter_re.composition_targets is set but flavor='semi_grand'. "
                    "composition_targets will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        else:
            # pressure flavor
            if self.composition_targets is not None:
                import warnings

                warnings.warn(
                    f"inter_re.composition_targets is set but flavor={self.flavor!r} "
                    f"(not 'xrens'). composition_targets will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.chemical_potentials is not None:
                import warnings

                warnings.warn(
                    f"inter_re.chemical_potentials is set but flavor={self.flavor!r} "
                    f"(not 'semi_grand'). chemical_potentials will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    def to_inter_re_config(self) -> InterREConfig:
        """Convert to the frozen :class:`InterREConfig` dataclass."""
        comp_targets = None
        if self.composition_targets is not None:
            comp_targets = tuple(
                tuple(row) for row in self.composition_targets
            )

        chem_pots = None
        if self.chemical_potentials is not None:
            chem_pots = tuple(
                tuple(float(v) for v in row)
                for row in self.chemical_potentials
            )

        return InterREConfig(
            flavor=self.flavor,
            re_interval=self.re_interval,
            n_swap_cycles=self.n_swap_cycles,
            composition_targets=comp_targets,
            chemical_potentials=chem_pots,
        )
