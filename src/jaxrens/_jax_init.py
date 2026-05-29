"""Idempotent JAX float32 pin.

Imported by every jaxrens module that uses JAX, BEFORE any other JAX use.
Python caches modules in ``sys.modules`` so the second-and-onwards imports
are a single dict lookup.

The pin must run before any third-party backend (notably mace-jax) gets a
chance to flip ``jax_enable_x64`` at construction time — jaxrens represents
positions, cells and energies in float32 throughout and ``lax.cond`` branch
dtype invariants (e.g. in ``init/cells.py::cell_shape_walk``) depend on it.
"""

import jax

jax.config.update("jax_enable_x64", False)
