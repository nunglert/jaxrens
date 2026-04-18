"""Matplotlib plotting helpers for NS run analysis.

Each function accepts a Monitor (or array data) and an optional ``ax``
argument, and returns the Axes object for further customisation.
Keyword arguments flow through to the underlying matplotlib call.

Matplotlib is imported lazily inside each function so that
``jaxrens.postprocess`` can be imported in environments without
matplotlib installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from jaxrens.postprocess.monitor import Monitor


def _get_ax(ax):
    """Return ax if provided, else create a new figure and axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_energy_trace(
    monitor: "Monitor",
    *,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot the live-walker culled energy vs iteration.

    Uses ``monitor.energy_trace`` and ``monitor.iteration_trace``.  Raises
    ``ValueError`` if the monitor has no energy trace (i.e., the .energies
    file was not loaded).

    Args:
        monitor: Populated Monitor.
        ax: Existing axes to plot into.  Created if None.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    if monitor.energy_trace is None:
        raise ValueError(
            "Monitor has no energy_trace.  Load from a directory that contains "
            "a .energies file, or pass energy_trace when constructing Monitor."
        )

    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "energy trace")

    x = (
        monitor.iteration_trace
        if monitor.iteration_trace is not None
        else np.arange(len(monitor.energy_trace))
    )
    ax.plot(x, monitor.energy_trace, **kwargs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    return ax


def plot_log_evidence_trace(
    monitor: "Monitor",
    *,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot cumulative log evidence vs dead-point index.

    Computes a running log Z using the prior-mass weights from
    ``thermodynamics.calc_log_weights``.

    Args:
        monitor: Populated Monitor.
        ax: Existing axes to plot into.  Created if None.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    from jax.scipy.special import logsumexp
    import jax.numpy as jnp

    from jaxrens.postprocess.thermodynamics import calc_log_weights

    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "log Z trace")

    n_dead = monitor.n_dead
    dead_e = jnp.asarray(monitor.dead_energies)
    log_w = calc_log_weights(n_dead, monitor.n_live, monitor.n_cull)
    log_L = -dead_e

    # Cumulative log Z: log sum_{i<=k} w_i * L_i
    cumulative = np.zeros(n_dead)
    for k in range(1, n_dead + 1):
        cumulative[k - 1] = float(logsumexp(log_w[:k] + log_L[:k]))

    ax.plot(np.arange(1, n_dead + 1), cumulative, **kwargs)
    ax.set_xlabel("Dead-point index")
    ax.set_ylabel("log Z (cumulative)")
    return ax


def plot_heat_capacity(
    monitor: "Monitor",
    T: np.ndarray,
    *,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot heat capacity C_v vs temperature.

    Args:
        monitor: Populated Monitor.
        T: Temperature array, shape (n_T,).
        ax: Existing axes to plot into.  Created if None.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "Cv")

    Cv = monitor.heat_capacity(T)
    ax.plot(T, Cv, **kwargs)
    ax.set_xlabel("T")
    ax.set_ylabel("Cv")
    return ax


def plot_partition_function(
    monitor: "Monitor",
    T: np.ndarray,
    *,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot log partition function log Z vs temperature.

    Args:
        monitor: Populated Monitor.
        T: Temperature array, shape (n_T,).
        ax: Existing axes to plot into.  Created if None.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "log Z")

    log_Z = monitor.partition_function(T)
    ax.plot(T, log_Z, **kwargs)
    ax.set_xlabel("T")
    ax.set_ylabel("log Z")
    return ax


def plot_free_energy(
    monitor: "Monitor",
    T: np.ndarray,
    *,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Plot Helmholtz free energy F vs temperature.

    Args:
        monitor: Populated Monitor.
        T: Temperature array, shape (n_T,).
        ax: Existing axes to plot into.  Created if None.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "F")

    F = monitor.free_energy(T)
    ax.plot(T, F, **kwargs)
    ax.set_xlabel("T")
    ax.set_ylabel("F")
    return ax
