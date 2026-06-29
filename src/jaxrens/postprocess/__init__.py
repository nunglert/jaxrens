"""Post-processing of nested sampling results."""

from jaxrens.postprocess.collection import MonitorCollection
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.plotting import (
    plot_acceptance_rates,
    plot_energy_trace,
    plot_free_energy,
    plot_heat_capacity,
    plot_heatmap,
    plot_log_evidence_trace,
    plot_max_neighbors,
    plot_partition_function,
    plot_re_acceptance,
    plot_re_acceptance_stacked,
    plot_step_sizes,
)
from jaxrens.postprocess.steinhardt import calc_qw, qw_from_edges
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
    # Monitor
    "Monitor",
    "MonitorCollection",
    # Plotting helpers
    "plot_acceptance_rates",
    "plot_energy_trace",
    "plot_free_energy",
    "plot_heat_capacity",
    "plot_heatmap",
    "plot_log_evidence_trace",
    "plot_max_neighbors",
    "plot_partition_function",
    "plot_re_acceptance",
    "plot_re_acceptance_stacked",
    "plot_step_sizes",
    # Steinhardt order parameters
    "calc_qw",
    "qw_from_edges",
    # Thermodynamics
    "calc_log_weights",
    "calc_log_weights_live",
    "expectation",
    "free_energy",
    "heat_capacity",
    "log_evidence",
    "partition_function",
]
