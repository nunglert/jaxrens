"""jaxrens: JAX-based nested sampling for atomistic systems."""

__version__ = "0.1.0"

import os as _os
import tempfile as _tempfile

_jax_tmpdir = _os.environ.get("TMPDIR", "/tmp")
try:
    with _tempfile.NamedTemporaryFile(dir=_jax_tmpdir, delete=True):
        pass
except OSError as _e:
    raise RuntimeError(
        f"jaxrens: {_jax_tmpdir!r} is not writable ({_e}). "
        f"JAX writes compilation artifacts there and will fail with a "
        f"confusing JaxRuntimeError later. "
        f"Set $TMPDIR to a writable directory before importing jaxrens."
    ) from _e

# Pin float32 precision globally.  jaxrens represents positions, cells and
# energies in float32 throughout; enabling x64 would silently promote some
# operations and break dtype invariants (e.g. ``lax.cond`` branches in
# ``init/cells.py::cell_shape_walk`` mismatching).  Some third-party backends
# (notably mace-jax) toggle this flag at construction time — running this
# update at jaxrens import makes sure the default is off, and ``create_mace``
# is pinned to ``dtype="float32"`` so no backend can flip it back on through
# the CLI path.
import jax as _jax
_jax.config.update("jax_enable_x64", False)

# High-level API
from jaxrens.backends.base import EnergyBackend
from jaxrens.backends.ensemble import EnsembleBackend, make_ensemble_params
from jaxrens.backends.loader import load_backend
from jaxrens.cli.run import run_from_config
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig
from jaxrens.state.mc_state import MCState, make_mc_state_class
from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    calc_log_weights_live,
    expectation,
    free_energy,
    heat_capacity,
    log_evidence,
    partition_function,
)

__all__ = [
    "EnergyBackend",
    "EnsembleBackend",
    "make_ensemble_params",
    "load_backend",
    "build_mwg",
    "MCState",
    "make_mc_state_class",
    "MoveKernel",
    "run_ns",
    "init_ns",
    "ns_step",
    "run_from_config",
    "NSConfig",
    "MoveConfig",
    "BackendConfig",
    "OutputConfig",
    "calc_log_weights",
    "calc_log_weights_live",
    "expectation",
    "free_energy",
    "heat_capacity",
    "log_evidence",
    "partition_function",
]
