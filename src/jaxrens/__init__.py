"""jaxrens: JAX-based nested sampling for atomistic systems.

JAX is NOT imported at package import time, and ``import jaxrens`` exposes
only ``__version__``.  Public symbols are imported from their subpackages,
e.g. ``from jaxrens.sampling.nested_sampling import run_ns`` or
``from jaxrens.postprocess.thermodynamics import free_energy``.

(A curated top-level API — ``from jaxrens import run_ns`` — used to be
re-exported lazily via PEP 562 ``__getattr__``; it is currently disabled,
see the commented block below for how to reinstate it.)

The float32 precision config still runs before any JAX op: every
JAX-using subpackage / module imports :mod:`jaxrens._jax_init` at the
top of its ``__init__.py`` (float32 by default; opt into 64-bit with
``JAX_ENABLE_X64``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Used only by the disabled top-level-API block below; uncomment to reinstate.
# from typing import TYPE_CHECKING


try:
    # Single source of truth is the git tag (via setuptools-scm); report the
    # installed package version.
    __version__ = version("jaxrens")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"
