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
    from jaxrens.postprocess.thermodynamics import calc_log_weights

    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "log Z trace")

    n_dead = monitor.n_dead
    log_w = np.asarray(calc_log_weights(n_dead, monitor.n_live, monitor.n_cull))
    log_L = -np.asarray(monitor.dead_energies)
    log_terms = log_w + log_L

    # Cumulative log Z via running logaddexp — pure numpy, O(n_dead).
    cumulative = np.empty(n_dead)
    running = -np.inf
    for i in range(n_dead):
        running = np.logaddexp(running, log_terms[i])
        cumulative[i] = running

    ax.plot(np.arange(1, n_dead + 1), cumulative, **kwargs)
    ax.set_xlabel("Dead-point index")
    ax.set_ylabel("log Z (cumulative)")
    return ax


def plot_heat_capacity(
    monitor: "Monitor",
    T: np.ndarray,
    *,
    ax: "Axes | None" = None,
    k_B: float = 1.0,
    **kwargs,
) -> "Axes":
    """Plot heat capacity C_v vs temperature.

    Args:
        monitor: Populated Monitor.
        T: Temperature array, shape (n_T,).
        ax: Existing axes to plot into.  Created if None.
        k_B: Boltzmann constant in the same energy units as the stored
            ``dead_energies``.  Defaults to ``1.0``.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "Cv")

    Cv = monitor.heat_capacity(T, k_B=k_B)
    ax.plot(T, Cv, **kwargs)
    ax.set_xlabel("T")
    ax.set_ylabel("Cv")
    return ax


def plot_partition_function(
    monitor: "Monitor",
    T: np.ndarray,
    *,
    ax: "Axes | None" = None,
    k_B: float = 1.0,
    **kwargs,
) -> "Axes":
    """Plot log partition function log Z vs temperature.

    Args:
        monitor: Populated Monitor.
        T: Temperature array, shape (n_T,).
        ax: Existing axes to plot into.  Created if None.
        k_B: Boltzmann constant in the same energy units as the stored
            ``dead_energies``.  Defaults to ``1.0``.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "log Z")

    log_Z = monitor.partition_function(T, k_B=k_B)
    ax.plot(T, log_Z, **kwargs)
    ax.set_xlabel("T")
    ax.set_ylabel("log Z")
    return ax


def plot_free_energy(
    monitor: "Monitor",
    T: np.ndarray,
    *,
    ax: "Axes | None" = None,
    k_B: float = 1.0,
    **kwargs,
) -> "Axes":
    """Plot Helmholtz free energy F vs temperature.

    Args:
        monitor: Populated Monitor.
        T: Temperature array, shape (n_T,).
        ax: Existing axes to plot into.  Created if None.
        k_B: Boltzmann constant in the same energy units as the stored
            ``dead_energies``.  Defaults to ``1.0``.
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.
    """
    ax = _get_ax(ax)
    kwargs.setdefault("label", monitor.label or "F")

    F = monitor.free_energy(T, k_B=k_B)
    ax.plot(T, F, **kwargs)
    ax.set_xlabel("T")
    ax.set_ylabel("F")
    return ax


def plot_step_sizes(
    monitor: "Monitor",
    *,
    ax: "Axes | None" = None,
    per_run: bool = False,
    **kwargs,
) -> "Axes":
    """Plot per-move step size vs iteration from the adaptation trace.

    When the trace is batched (``n_runs > 1``) and ``per_run=True``, draws one
    line per (move, run) combination.  When ``per_run=False`` (default), plots
    the mean across runs with a shaded ``fill_between`` for ±1 std.

    Args:
        monitor:  Populated Monitor with a non-None ``adaptation_trace``.
        ax:       Existing axes.  Created if None.
        per_run:  If True, draw individual run lines instead of mean±std.
        **kwargs: Forwarded to ``ax.plot`` (not to ``fill_between``).

    Returns:
        The Axes object.

    Raises:
        ValueError: If ``monitor.adaptation_trace`` is None.
    """
    import matplotlib.pyplot as plt

    if monitor.adaptation_trace is None:
        raise ValueError(
            "Monitor has no adaptation_trace.  Load from a directory that "
            "contains a .adaptation.h5 file."
        )

    ax = _get_ax(ax)
    trace = monitor.adaptation_trace

    iters = trace.iterations                     # (n_entries,)
    ss = trace.step_sizes                        # (n_entries, n_runs, n_moves)
    move_names = trace.move_names
    n_moves = trace.n_moves
    n_runs = trace.n_runs

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, name in enumerate(move_names):
        color = colors[k % len(colors)]
        ss_k = ss[:, :, k]  # (n_entries, n_runs)

        if per_run or n_runs == 1:
            for r in range(n_runs):
                lbl = f"{name}" if n_runs == 1 else f"{name} run{r}"
                ax.plot(iters, ss_k[:, r], label=lbl, color=color, **kwargs)
        else:
            mean_k = ss_k.mean(axis=1)
            std_k = ss_k.std(axis=1)
            ax.plot(iters, mean_k, label=name, color=color, **kwargs)
            ax.fill_between(iters, mean_k - std_k, mean_k + std_k,
                            alpha=0.2, color=color)

    ax.set_xlabel("iteration")
    ax.set_ylabel("step size")
    ax.legend()
    return ax


def plot_acceptance_rates(
    monitor: "Monitor",
    *,
    ax: "Axes | None" = None,
    per_run: bool = False,
    **kwargs,
) -> "Axes":
    """Plot per-move acceptance rate vs iteration from the adaptation trace.

    When the trace is batched (``n_runs > 1``) and ``per_run=True``, draws one
    line per (move, run) combination.  When ``per_run=False`` (default), plots
    the mean across runs with a shaded ``fill_between`` for ±1 std.

    Args:
        monitor:  Populated Monitor with a non-None ``adaptation_trace``.
        ax:       Existing axes.  Created if None.
        per_run:  If True, draw individual run lines instead of mean±std.
        **kwargs: Forwarded to ``ax.plot`` (not to ``fill_between``).

    Returns:
        The Axes object.

    Raises:
        ValueError: If ``monitor.adaptation_trace`` is None.
    """
    import matplotlib.pyplot as plt

    if monitor.adaptation_trace is None:
        raise ValueError(
            "Monitor has no adaptation_trace.  Load from a directory that "
            "contains a .adaptation.h5 file."
        )

    ax = _get_ax(ax)
    trace = monitor.adaptation_trace

    iters = trace.iterations                      # (n_entries,)
    acc = trace.acceptance_rates                  # (n_entries, n_runs, n_moves)
    move_names = trace.move_names
    n_moves = trace.n_moves
    n_runs = trace.n_runs

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, name in enumerate(move_names):
        color = colors[k % len(colors)]
        acc_k = acc[:, :, k]  # (n_entries, n_runs)

        if per_run or n_runs == 1:
            for r in range(n_runs):
                lbl = f"{name}" if n_runs == 1 else f"{name} run{r}"
                ax.plot(iters, acc_k[:, r], label=lbl, color=color, **kwargs)
        else:
            mean_k = acc_k.mean(axis=1)
            std_k = acc_k.std(axis=1)
            ax.plot(iters, mean_k, label=name, color=color, **kwargs)
            ax.fill_between(iters, mean_k - std_k, mean_k + std_k,
                            alpha=0.2, color=color)

    ax.set_xlabel("iteration")
    ax.set_ylabel("acceptance rate")
    ax.legend()
    return ax
