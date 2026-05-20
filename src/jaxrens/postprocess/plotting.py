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


def plot_heatmap(
    T: np.ndarray,
    P: np.ndarray,
    Z: np.ndarray,
    *,
    ax: "Axes | None" = None,
    cmap: str = "viridis",
    fmt: str = "PT",
    vmin: float | None = None,
    vmax: float | None = None,
    do_colorbar: bool = True,
    cbar_label: str | None = None,
    T_label: str = "T",
    P_label: str = "P",
    **pcolormesh_kwargs,
) -> "Axes":
    """Draw a 2D pressure-temperature heatmap of an observable.

    Args:
        T: Temperature grid, shape ``(n_T,)``.
        P: Pressure grid, shape ``(n_P,)``.  Must be sorted by the caller —
            ``pcolormesh`` does not reorder cells.
        Z: Observable values, shape ``(n_P, n_T)`` (rows index pressure).
        ax: Existing axes to plot into.  Created if None.
        cmap: Matplotlib colormap name.
        fmt: ``"PT"`` puts pressure on the x-axis, temperature on y.
            ``"TP"`` swaps them.
        vmin, vmax: Colour-scale clipping bounds; ``None`` uses the data range.
        do_colorbar: Attach a colorbar to the figure when ``True``.
        cbar_label: Label drawn next to the colorbar.
        T_label, P_label: Axis labels for the temperature / pressure axes.
        **pcolormesh_kwargs: Forwarded to ``ax.pcolormesh``.

    Returns:
        The Axes object.
    """
    import matplotlib.pyplot as plt

    ax = _get_ax(ax)
    Z = np.asarray(Z)
    T = np.asarray(T)
    P = np.asarray(P)
    if Z.shape != (P.shape[0], T.shape[0]):
        raise ValueError(
            f"Z.shape={Z.shape} must be (n_P, n_T)="
            f"({P.shape[0]}, {T.shape[0]})"
        )

    if fmt == "PT":
        X, Y = np.meshgrid(P, T)
        Z_plot = Z.T
        xlabel, ylabel = P_label, T_label
    elif fmt == "TP":
        X, Y = np.meshgrid(T, P)
        Z_plot = Z
        xlabel, ylabel = T_label, P_label
    else:
        raise ValueError(f"fmt must be 'PT' or 'TP', got {fmt!r}")

    pcolormesh_kwargs.setdefault("shading", "auto")
    pcolormesh_kwargs.setdefault("rasterized", True)
    im = ax.pcolormesh(
        X, Y, Z_plot, vmin=vmin, vmax=vmax, cmap=cmap, **pcolormesh_kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if do_colorbar:
        plt.colorbar(im, ax=ax, label=cbar_label)
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


def _resolve_adaptation_trace(monitor_or_trace):
    """Accept either a Monitor (legacy) or an AdaptationLog (new) and
    return the underlying ``AdaptationLog``.  Raises if neither has one.
    """
    from jaxrens.io.adaptation_log import AdaptationLog

    if isinstance(monitor_or_trace, AdaptationLog):
        return monitor_or_trace
    trace = getattr(monitor_or_trace, "adaptation_trace", None)
    if trace is None:
        raise ValueError(
            "No adaptation_trace available.  Pass an AdaptationLog "
            "directly, or load a Monitor / MonitorCollection from a "
            "directory that contains a .adaptation.h5 file."
        )
    return trace


def plot_step_sizes(
    monitor_or_trace,
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
        monitor_or_trace: A populated Monitor with a non-None
            ``adaptation_trace`` *or* an ``AdaptationLog`` directly (the
            latter is what ``MonitorCollection.plot_step_sizes`` passes
            when the cohort-wide log is loaded by
            ``from_multi_run_directory``).
        ax:       Existing axes.  Created if None.
        per_run:  If True, draw individual run lines instead of mean±std.
        **kwargs: Forwarded to ``ax.plot`` (not to ``fill_between``).

    Returns:
        The Axes object.

    Raises:
        ValueError: If no adaptation trace can be resolved.
    """
    import matplotlib.pyplot as plt

    trace = _resolve_adaptation_trace(monitor_or_trace)
    ax = _get_ax(ax)

    iters = trace.iterations                     # (n_entries,)
    ss = trace.step_sizes                        # (n_entries, n_runs, n_moves)
    move_names = trace.move_names
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
    monitor_or_trace,
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
        monitor_or_trace: A populated Monitor with a non-None
            ``adaptation_trace`` *or* an ``AdaptationLog`` directly.
        ax:       Existing axes.  Created if None.
        per_run:  If True, draw individual run lines instead of mean±std.
        **kwargs: Forwarded to ``ax.plot`` (not to ``fill_between``).

    Returns:
        The Axes object.

    Raises:
        ValueError: If no adaptation trace can be resolved.
    """
    import matplotlib.pyplot as plt

    trace = _resolve_adaptation_trace(monitor_or_trace)
    ax = _get_ax(ax)

    iters = trace.iterations                      # (n_entries,)
    acc = trace.acceptance_rates                  # (n_entries, n_runs, n_moves)
    move_names = trace.move_names
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


def _resolve_re_trace(monitor_or_trace):
    """Accept either a Monitor (legacy) or an RELog (new); return the RELog."""
    from jaxrens.io.re_stats_log import RELog

    if isinstance(monitor_or_trace, RELog):
        return monitor_or_trace
    trace = getattr(monitor_or_trace, "re_trace", None)
    if trace is None:
        raise ValueError(
            "No re_trace available.  Pass an RELog directly, or load a "
            "Monitor from a directory that contains a .re_stats.h5 file."
        )
    return trace


def plot_re_acceptance(
    monitor_or_trace,
    *,
    ax: "Axes | None" = None,
    per_pair: bool = True,
    window: int | None = None,
    label_prefix: str = "",
    **kwargs,
) -> "Axes":
    """Plot inter-replica swap acceptance rate vs iteration.

    Per-fire per-pair acceptance comes from the RE trace (loaded from
    ``<prefix>.re_stats.h5``).  When ``per_pair=True`` (default), draws one
    line per adjacent replica pair.  When ``per_pair=False``, plots the mean
    across pairs with a shaded ``fill_between`` for ±1 std.

    Per-fire values are 0/1 noisy for ``n_swap_cycles=1`` runs.  Pass
    ``window=W`` to apply a centred boxcar average of width ``W`` to each
    pair's trace before plotting.

    Args:
        monitor_or_trace: A populated Monitor with a non-None ``re_trace``
            *or* an ``RELog`` instance directly.
        ax: Existing axes to plot into.  Created if None.
        per_pair: If True, draw one line per adjacent replica pair.  Otherwise
            plot mean ± std across pairs.
        window: Optional boxcar smoothing window length (in fire entries).
        label_prefix: Prefix prepended to the legend label when
            ``per_pair=False`` (used by the CLI plot dispatcher).
        **kwargs: Forwarded to ``ax.plot``.

    Returns:
        The Axes object.

    Raises:
        ValueError: If no RE trace can be resolved or it has zero pairs.
    """
    import matplotlib.pyplot as plt

    re_log = _resolve_re_trace(monitor_or_trace)
    if re_log.n_pairs == 0:
        raise ValueError(
            "re_trace has zero pairs — nothing to plot (single-replica run)."
        )

    ax = _get_ax(ax)
    iters = re_log.iterations                              # (n_entries,)
    acc = np.asarray(re_log.acceptance_rates, dtype=np.float64)

    if window is not None and window > 1:
        kernel = np.ones(int(window), dtype=np.float64) / float(window)
        acc = np.stack(
            [
                np.convolve(acc[:, p], kernel, mode="same")
                for p in range(acc.shape[1])
            ],
            axis=1,
        )

    if per_pair:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for p in range(re_log.n_pairs):
            color = colors[p % len(colors)]
            ax.plot(
                iters, acc[:, p],
                label=f"pair {p}-{p + 1}", color=color, **kwargs,
            )
    else:
        mean = acc.mean(axis=1)
        std = acc.std(axis=1)
        # Label: monitor's label takes precedence when available; otherwise
        # fall back to label_prefix + "mean swap acc".
        mon_label = getattr(monitor_or_trace, "label", None)
        lbl = mon_label or f"{label_prefix}mean swap acc".strip()
        ax.plot(iters, mean, label=lbl, **kwargs)
        ax.fill_between(iters, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("iteration")
    ax.set_ylabel(f"swap acceptance rate ({re_log.flavor})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    return ax


def plot_re_acceptance_stacked(
    monitor_or_trace,
    *,
    ax: "Axes | None" = None,
    window: int | None = None,
    pair_labels: list[str] | None = None,
    cmap: str = "viridis",
    raw_alpha: float = 0.3,
    raw_lw: float = 0.5,
    smooth_lw: float = 1.5,
    offset_scale: float = 1.0,
) -> "Axes":
    """Plot per-pair RE swap acceptance with vertical stacking.

    Each adjacent-pair acceptance trace is shifted vertically by
    ``p * offset_scale`` so overlapping curves do not crowd a single
    [0, 1] band.  A faint raw line is overlaid by a boxcar-smoothed line
    in the same color; black baselines mark each pair's zero.  A right-hand
    twin axis carries the pair labels.

    Mirrors the legacy ``jaxnest.plot.plot_output.plot_acc_rates`` style.

    Args:
        monitor_or_trace: Populated Monitor with a non-None ``re_trace``
            or an ``RELog`` directly.
        ax: Existing axes to plot into.  Created if None.
        window: Optional boxcar smoothing window length (in fire entries).
            Defaults to ``max(1, n_entries // 50)``.
        pair_labels: Optional per-pair labels for the right-hand twin axis.
            Length must equal ``n_pairs``.  Defaults to ``"p↔p+1"``.
        cmap: Colormap name used to color the per-pair curves.
        raw_alpha: Alpha for the raw (unsmoothed) curves.
        raw_lw: Linewidth for the raw curves.
        smooth_lw: Linewidth for the smoothed overlay.
        offset_scale: Vertical separation between pairs in [0, 1] units.

    Returns:
        The left Axes object (the right twin can be retrieved via
        ``ax.figure.axes[-1]``).
    """
    import matplotlib.pyplot as plt

    re_log = _resolve_re_trace(monitor_or_trace)
    if re_log.n_pairs == 0:
        raise ValueError(
            "re_trace has zero pairs — nothing to plot (single-replica run)."
        )

    iters = np.asarray(re_log.iterations, dtype=np.float64)
    acc = np.asarray(re_log.acceptance_rates, dtype=np.float64)
    n_entries, n_pairs = acc.shape

    if pair_labels is not None and len(pair_labels) != n_pairs:
        raise ValueError(
            f"len(pair_labels)={len(pair_labels)} does not match "
            f"n_pairs={n_pairs}"
        )

    if window is None:
        window = max(1, n_entries // 50)

    if window > 1:
        kernel = np.ones(int(window), dtype=np.float64) / float(window)
        acc_smooth = np.stack(
            [
                np.convolve(acc[:, p], kernel, mode="same")
                for p in range(n_pairs)
            ],
            axis=1,
        )
    else:
        acc_smooth = acc

    ax = _get_ax(ax)
    colors = plt.get_cmap(cmap)(np.linspace(0.0, 0.95, n_pairs))

    for p in range(n_pairs):
        offset = p * offset_scale
        ax.axhline(offset, lw=0.5, alpha=0.5, color="black")
        ax.plot(
            iters, acc[:, p] + offset,
            color=colors[p], alpha=raw_alpha, lw=raw_lw,
        )
        ax.plot(
            iters, acc_smooth[:, p] + offset,
            color=colors[p], lw=smooth_lw,
        )

    ax.set_ylim(-0.1 * offset_scale, (n_pairs - 1) * offset_scale + 1.1)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["0%", "100%"])
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"swap acceptance rate ({re_log.flavor})")

    twin = ax.twinx()
    twin.set_ylim(ax.get_ylim())
    twin.set_yticks(np.arange(n_pairs) * offset_scale)
    if pair_labels is None:
        pair_labels = [f"{p}↔{p + 1}" for p in range(n_pairs)]
    twin.set_yticklabels(pair_labels, ha="left", fontfamily="monospace")
    twin.tick_params(axis="y", length=0)

    return ax


def _resolve_max_neighbors_trace(monitor_or_trace):
    """Accept either a Monitor or a MaxNeighborsLog; return the log."""
    from jaxrens.io.max_neighbors_log import MaxNeighborsLog

    if isinstance(monitor_or_trace, MaxNeighborsLog):
        return monitor_or_trace
    trace = getattr(monitor_or_trace, "max_neighbors_trace", None)
    if trace is None:
        raise ValueError(
            "No max_neighbors_trace available.  Pass a MaxNeighborsLog "
            "directly, or load a Monitor from a directory that contains a "
            ".max_neighbors.h5 file (enable via output.save_max_neighbors=True)."
        )
    return trace


def plot_max_neighbors(
    monitor_or_trace,
    *,
    ax: "Axes | None" = None,
    kind: str = "percentiles",
    run: int = 0,
    percentiles: tuple[float, ...] = (50.0, 90.0, 100.0),
    show_bucket: bool = True,
    **kwargs,
) -> "Axes":
    """Plot per-iteration neighbor-bucket diagnostics.

    Two display modes:

    * ``kind="percentiles"`` (default): line plot showing the requested
      ``percentiles`` of the per-walker max_neighbor_count distribution
      vs iteration.  Useful for "is the bucket too generous / too tight"
      at a glance.
    * ``kind="heatmap"``: 2-D histogram of (iteration, max_neighbor_count)
      densities — closest to "the distribution per iteration" requested
      by the user.  Cell color = log-count of walkers at that
      (iteration, count) bin.

    When ``show_bucket=True`` overlays the current bucket size as a
    step line — handy for spotting the gap between observed peak and
    the configured bucket.

    Args:
        monitor_or_trace: A Monitor with a non-None ``max_neighbors_trace``
            *or* a ``MaxNeighborsLog`` instance.
        ax: Existing axes to plot into.  Created if None.
        kind: ``"percentiles"`` or ``"heatmap"``.
        run: Run index for multi-run logs.  Default 0 (the only entry
            for SingleRun).
        percentiles: Percentiles to plot when ``kind="percentiles"``.
            Defaults to (50, 90, 100) — median, p90, max.
        show_bucket: Overlay the bucket-size step line.
        **kwargs: Forwarded to ``ax.plot`` (percentiles) or
            ``ax.pcolormesh`` (heatmap).

    Returns:
        The Axes object.
    """
    log = _resolve_max_neighbors_trace(monitor_or_trace)
    if not (0 <= run < log.n_runs):
        raise ValueError(
            f"run={run} out of range for max_neighbors_trace with "
            f"n_runs={log.n_runs}."
        )

    ax = _get_ax(ax)
    iters = log.iterations  # (n_entries,)
    counts = log.max_neighbor_count[:, run, :]  # (n_entries, n_walkers)
    buckets = log.bucket_size[:, run]            # (n_entries,)

    if kind == "percentiles":
        for pct in percentiles:
            vals = np.percentile(counts, pct, axis=-1)
            ax.plot(iters, vals, label=f"p{int(pct)}", **kwargs)
        ylabel = "max_neighbor_count (per-walker percentiles)"
    elif kind == "heatmap":
        # Build a 2-D histogram: x = iteration index in the log, y = count.
        # Each (iter, walker) pair contributes one entry at (iter, count).
        cmax = int(counts.max()) + 1
        cmin = int(counts.min())
        n_iter, n_walk = counts.shape
        H = np.zeros((cmax - cmin + 1, n_iter), dtype=np.int64)
        for i in range(n_iter):
            vals, freq = np.unique(counts[i], return_counts=True)
            H[vals - cmin, i] = freq
        # log scale to make sparse bins visible.
        with np.errstate(divide="ignore"):
            H_log = np.log10(H, where=H > 0, out=np.full_like(H, np.nan, dtype=np.float64))
        y_edges = np.arange(cmin, cmax + 2) - 0.5
        x_edges = np.concatenate([iters, [iters[-1] + 1]]).astype(np.float64) - 0.5
        ax.pcolormesh(x_edges, y_edges, H_log, **kwargs)
        ylabel = "max_neighbor_count"
    else:
        raise ValueError(
            f"kind must be 'percentiles' or 'heatmap', got {kind!r}."
        )

    if show_bucket:
        ax.step(
            iters, buckets, where="post",
            label="bucket", color="k", linestyle="--", linewidth=1.0,
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if kind == "percentiles" or show_bucket:
        ax.legend()
    return ax
