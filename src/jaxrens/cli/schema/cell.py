"""Pydantic schema for the [cell] section of a jaxrens YAML config.

``CellConfig`` exposes cell-geometry constraints that the existing volume,
shear, and stretch move kernels accept individually (``max_vol_per_atom``,
``min_vol_per_atom``, ``min_aspect``).

DEFERRED: ``CellConfig`` fields are accepted and validated here but are NOT
yet automatically threaded into move kernels.  Individual move specs
(``VolumeMoveSpec``, ``ShearMoveSpec``, ``StretchMoveSpec``) carry their own
per-move copies of these fields; unifying them through ``CellConfig`` is
planned for a future task.  A resolver warning is emitted when any non-default
value is set in ``CellConfig`` to ensure users are not silently ignored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CellConfig(BaseModel):
    """Cell-geometry constraint parameters.

    DEFERRED: these fields are validated and stored but not yet consumed by
    move kernels automatically.  See module docstring for details.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_volume_per_atom: float = 1e4
    min_volume_per_atom: float = 1.0
    min_aspect_ratio: float = 0.8
    flat_V_prior: bool = False
