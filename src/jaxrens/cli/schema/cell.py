"""Pydantic schema for the [cell] section of a jaxrens YAML config.

``CellSpec`` exposes cell-geometry constraints that the volume, shear, and
stretch move kernels accept.  The resolver threads these values into
``MoveKernel.kernel_kwargs`` automatically at resolution time via
``BaseMoveSpec.to_descriptor(cell_cfg=...)``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CellSpec(BaseModel):
    """Cell-geometry constraint parameters.

    These values are threaded automatically into the ``kernel_kwargs`` of
    volume, shear, and stretch move descriptors by the resolver.  They are
    the single source of truth for cell-geometry bounds; the individual move
    specs no longer carry copies of these fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_volume_per_atom: float = Field(
        default=1e4,
        description=(
            "Upper bound on cell volume per atom (Angstrom^3).  Caps the "
            "cell prior: volume, shear, and stretch proposals above it are "
            "rejected.  The default is deliberately loose — tighten it to "
            "stop a gas-phase run from expanding without limit."
        ),
    )
    min_volume_per_atom: float = Field(
        default=1.0,
        description=(
            "Lower bound on cell volume per atom (Angstrom^3) enforced "
            "throughout sampling.  Stops the cell collapsing into a "
            "region where the energy model is unphysical."
        ),
    )
    initial_min_volume_per_atom: float | None = Field(
        default=None,
        description=(
            "Lower floor on the *initial* cell-volume draw (Angstrom^3 per "
            "atom), decoupled from the sampling constraint "
            "``min_volume_per_atom``.  Lets walkers be initialised on a "
            "low-density grid (large volume/atom, so atoms start well "
            "separated) while the sampler stays free to compress down to "
            "``min_volume_per_atom`` during the run.  ``null`` (default) "
            "falls back to ``min_volume_per_atom``, so initial draws never "
            "violate the sampling constraint."
        ),
    )
    min_aspect_ratio: float = Field(
        default=0.8,
        description=(
            "Minimum ratio of the shortest to the longest cell-vector "
            "perpendicular distance.  Rejects proposals that would flatten "
            "the cell into a sliver, which breaks the minimum-image "
            "convention and starves neighbour lists."
        ),
    )
    flat_V_prior: bool = Field(
        default=False,
        description=(
            "Sample volume under a flat prior in ``V`` instead of the "
            "default flat-in-``log V`` prior.  Changes the volume-move "
            "acceptance ratio; pick the one your reference implementation "
            "uses when comparing equations of state."
        ),
    )

    @property
    def effective_initial_min_volume_per_atom(self) -> float:
        """Resolved lower floor for the initial volume draw.

        Returns ``initial_min_volume_per_atom`` when set, otherwise falls back
        to ``min_volume_per_atom``.
        """
        if self.initial_min_volume_per_atom is None:
            return self.min_volume_per_atom
        return self.initial_min_volume_per_atom

    @model_validator(mode="after")
    def _check_initial_min_volume(self) -> CellSpec:
        floor = self.initial_min_volume_per_atom
        if floor is None:
            return self
        if floor < 0.0:
            raise ValueError(
                f"cell.initial_min_volume_per_atom must be >= 0, got {floor}."
            )
        if floor > self.max_volume_per_atom:
            raise ValueError(
                f"cell.initial_min_volume_per_atom ({floor}) exceeds "
                f"cell.max_volume_per_atom ({self.max_volume_per_atom}); the "
                f"initial volume draw would have an empty range."
            )
        return self
