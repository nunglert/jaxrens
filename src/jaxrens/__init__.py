"""jaxrens: JAX-based nested sampling for atomistic systems."""

__version__ = "0.1.0"

# High-level API
from jaxrens.backends.loader import load_backend
from jaxrens.cli.run import run_from_config, run_from_file
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig
from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    expectation,
    free_energy,
    heat_capacity,
    log_evidence,
    partition_function,
)

__all__ = [
    "load_backend",
    "run_ns",
    "init_ns",
    "ns_step",
    "run_from_config",
    "run_from_file",
    "NSConfig",
    "MoveConfig",
    "BackendConfig",
    "OutputConfig",
    "calc_log_weights",
    "expectation",
    "free_energy",
    "heat_capacity",
    "log_evidence",
    "partition_function",
]
