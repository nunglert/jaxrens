"""jaxrens: JAX-based nested sampling for atomistic systems."""

__version__ = "0.1.0"

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
