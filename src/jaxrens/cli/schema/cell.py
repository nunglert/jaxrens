"""Pydantic schema for the [cell] section of a jaxrens YAML config.

``CellSpec`` exposes cell-geometry constraints that the volume, shear, and
stretch move kernels accept.  The resolver threads these values into
``MoveKernel.kernel_kwargs`` automatically at resolution time via
``BaseMoveSpec.to_descriptor(cell_cfg=...)``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CellSpec(BaseModel):
    """Cell-geometry constraint parameters.

    These values are threaded automatically into the ``kernel_kwargs`` of
    volume, shear, and stretch move descriptors by the resolver.  They are
    the single source of truth for cell-geometry bounds; the individual move
    specs no longer carry copies of these fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_volume_per_atom: float = 1e4
    min_volume_per_atom: float = 1.0
    min_aspect_ratio: float = 0.8
    flat_V_prior: bool = False
