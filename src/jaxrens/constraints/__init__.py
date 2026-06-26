"""Composable configuration constraints for nested sampling.

See :mod:`jaxrens.constraints.base` for the core abstraction.
"""

from __future__ import annotations

from jaxrens.constraints.base import (
    ASPECTS,
    Constraint,
    ConstraintDescriptor,
    make_move_gate,
)
from jaxrens.constraints.cell_geometry import (
    build_cell_geometry,
    cell_geometry_descriptor,
)
from jaxrens.constraints.min_distance import (
    build_min_distance,
    min_distance_descriptor,
)

__all__ = [
    "ASPECTS",
    "Constraint",
    "ConstraintDescriptor",
    "make_move_gate",
    "build_min_distance",
    "min_distance_descriptor",
    "build_cell_geometry",
    "cell_geometry_descriptor",
]
