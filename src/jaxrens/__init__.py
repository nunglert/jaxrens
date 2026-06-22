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

# ===========================================================================
# Curated top-level public API (lazy PEP 562 re-exports) — DISABLED.
#
# This block re-exported a curated set of symbols at the top level
# (``from jaxrens import run_ns``, ``jaxrens.free_energy``, …) via a lazy
# ``__getattr__``.  It has been commented out: those symbols are now reached
# only through their real subpackages (``from jaxrens.sampling.nested_sampling
# import run_ns`` etc.), which is what all internal code, the CLI, and the
# tutorials already do.  ``import jaxrens`` still works and exposes
# ``__version__``.
#
# To REINSTATE: uncomment this whole block, and re-enable the matching pieces
# in ``docs/reference/index.rst`` (the thematic ``autosummary`` sections under
# ``.. currentmodule:: jaxrens``) and ``tests/test_lazy_jax_import.py``.
# ===========================================================================
#
# # Static analysers (pyright/pylance, mypy) can't introspect a module-level
# # ``__getattr__``.  Re-export the same symbols here under ``TYPE_CHECKING``
# # so editors see the public API; at runtime this block is skipped and the
# # lazy ``__getattr__`` below remains the single source of truth.
# if TYPE_CHECKING:
#     from jaxrens.backends.base import EnergyBackend
#     from jaxrens.backends.ensemble import EnsembleBackend, make_ensemble_params
#     from jaxrens.backends.loader import load_backend
#     from jaxrens.cli.run import run_from_config
#     from jaxrens.postprocess.thermodynamics import (
#         calc_log_weights,
#         calc_log_weights_live,
#         expectation,
#         free_energy,
#         heat_capacity,
#         log_evidence,
#         partition_function,
#     )
#     from jaxrens.sampling.move_kernel import MoveKernel
#     from jaxrens.sampling.mwg import build_mwg
#     from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns
#     from jaxrens.state.config import (
#         BackendConfig,
#         MoveConfig,
#         NSConfig,
#         OutputConfig,
#     )
#     from jaxrens.state.mc_state import MCState, make_mc_state_class
#
# # (name -> (module_path, attr, needs_jax_pin)).  ``needs_jax_pin`` flags
# # the JAX-using exports — accessing one of those names triggers an
# # idempotent x64 pin before the lazy import.  Pure-Python re-exports
# # (pydantic config + NumPy thermodynamics) skip the pin entirely.
# _LAZY_EXPORTS: dict[str, tuple[str, str, bool]] = {
#     # JAX-free re-exports
#     "BackendConfig":          ("jaxrens.state.config", "BackendConfig", False),
#     "MoveConfig":             ("jaxrens.state.config", "MoveConfig", False),
#     "NSConfig":               ("jaxrens.state.config", "NSConfig", False),
#     "OutputConfig":           ("jaxrens.state.config", "OutputConfig", False),
#     "calc_log_weights":       ("jaxrens.postprocess.thermodynamics", "calc_log_weights", False),
#     "calc_log_weights_live":  ("jaxrens.postprocess.thermodynamics", "calc_log_weights_live", False),
#     "expectation":            ("jaxrens.postprocess.thermodynamics", "expectation", False),
#     "free_energy":            ("jaxrens.postprocess.thermodynamics", "free_energy", False),
#     "heat_capacity":          ("jaxrens.postprocess.thermodynamics", "heat_capacity", False),
#     "log_evidence":           ("jaxrens.postprocess.thermodynamics", "log_evidence", False),
#     "partition_function":     ("jaxrens.postprocess.thermodynamics", "partition_function", False),
#     # JAX-using re-exports
#     "EnergyBackend":          ("jaxrens.backends.base", "EnergyBackend", True),
#     "EnsembleBackend":        ("jaxrens.backends.ensemble", "EnsembleBackend", True),
#     "make_ensemble_params":   ("jaxrens.backends.ensemble", "make_ensemble_params", True),
#     "load_backend":           ("jaxrens.backends.loader", "load_backend", True),
#     "run_from_config":        ("jaxrens.cli.run", "run_from_config", True),
#     "MoveKernel":             ("jaxrens.sampling.move_kernel", "MoveKernel", True),
#     "build_mwg":              ("jaxrens.sampling.mwg", "build_mwg", True),
#     "init_ns":                ("jaxrens.sampling.nested_sampling", "init_ns", True),
#     "ns_step":                ("jaxrens.sampling.nested_sampling", "ns_step", True),
#     "run_ns":                 ("jaxrens.sampling.nested_sampling", "run_ns", True),
#     "MCState":                ("jaxrens.state.mc_state", "MCState", True),
#     "make_mc_state_class":    ("jaxrens.state.mc_state", "make_mc_state_class", True),
# }
#
#
# def __getattr__(name: str):
#     try:
#         mod_path, attr, needs_pin = _LAZY_EXPORTS[name]
#     except KeyError:
#         raise AttributeError(f"module 'jaxrens' has no attribute {name!r}")
#     if needs_pin:
#         from jaxrens import _jax_init  # noqa: F401 -- idempotent x64 pin
#     import importlib
#     return getattr(importlib.import_module(mod_path), attr)
#
#
# def __dir__() -> list[str]:
#     return sorted({*_LAZY_EXPORTS, "__version__"})
#
#
# # Static ``__all__`` so pyright/pylance and other linters can evaluate the
# # public API without running code.  Must stay in sync with ``_LAZY_EXPORTS``.
# __all__ = [
#     "BackendConfig",
#     "EnergyBackend",
#     "EnsembleBackend",
#     "MCState",
#     "MoveConfig",
#     "MoveKernel",
#     "NSConfig",
#     "OutputConfig",
#     "build_mwg",
#     "calc_log_weights",
#     "calc_log_weights_live",
#     "expectation",
#     "free_energy",
#     "heat_capacity",
#     "init_ns",
#     "load_backend",
#     "log_evidence",
#     "make_ensemble_params",
#     "make_mc_state_class",
#     "ns_step",
#     "partition_function",
#     "run_from_config",
#     "run_ns",
# ]
#
# assert set(__all__) == set(_LAZY_EXPORTS), (
#     "jaxrens/__init__.py: __all__ drifted from _LAZY_EXPORTS"
# )
