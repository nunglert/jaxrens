"""Shared utilities: units, structure manipulation, logging setup."""

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op

from jaxrens.utils.cell import (
    check_cell_shape,
    get_cell_transformation,
    get_volume,
    min_aspect_ratio,
    transform_positions,
)
