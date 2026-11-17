"""Post-processing of nested sampling results."""

from jaxrens.postprocess.collection import MonitorCollection
from jaxrens.postprocess.layered_run import (
    LayeredParams,
    layered_observables,
    layered_params,
    load_layered_observables,
)
from jaxrens.postprocess.layers import LayerGeometry, layer_geometry
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.observables import cell_observables, layer_observables
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
from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    calc_log_weights_live,
    class_probability,
    conditional_mean,
    expectation,
    free_energy,
    heat_capacity,
    log_evidence,
    log_point_weights,
    partition_function,
    thermal_weights,
    weighted_covariance,
    weighted_mean,
    weighted_variance,
)

__all__ = [
    # Monitor
    "Monitor",
    "MonitorCollection",
    # Layered geometry
    "LayerGeometry",
    "layer_geometry",
    # Structural observables
    "LayeredParams",
    "layered_observables",
    "layered_params",
    "load_layered_observables",
    "cell_observables",
    "layer_observables",
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
    # Thermodynamics
    "calc_log_weights",
    "calc_log_weights_live",
    "class_probability",
    "conditional_mean",
    "expectation",
    "free_energy",
    "heat_capacity",
    "log_evidence",
    "log_point_weights",
    "partition_function",
    "thermal_weights",
    "weighted_covariance",
    "weighted_mean",
    "weighted_variance",
]
