"""``jaxrens plot <file>`` — produce a PNG from a single artefact.

Dispatches by filename suffix::

    *.adaptation.h5    → 2-panel step-size + acceptance-rate plot
                         (mean ± std across replicas).
    *.re_stats.h5      → swap acceptance per adjacent pair vs iteration.
    *.max_neighbors.h5 → 2-panel: per-walker count percentiles (top) +
                         distribution heatmap (bottom), with bucket overlay.
    *.energies         → dead-point energy trail (and volume if present).

This is the "quick look at one file" utility — for full multi-run cohort
analysis use ``MonitorCollection.from_multi_run_directory`` and the
methods on the collection.
"""

from __future__ import annotations

import re
from pathlib import Path


def _suffix_kind(path: Path) -> str:
    """Return the artefact kind from the filename.

    Recognises ``.adaptation.h5``, ``.re_stats.h5``, and ``.energies``.
    Raises ``ValueError`` for unknown suffixes.
    """
    name = path.name
    if name.endswith(".adaptation.h5"):
        return "adaptation"
    if name.endswith(".re_stats.h5"):
        return "re_stats"
    if name.endswith(".max_neighbors.h5"):
        return "max_neighbors"
    if name.endswith(".energies"):
        return "energies"
    raise ValueError(
        f"Unrecognised file kind for {path.name!r}.  Supported suffixes: "
        ".adaptation.h5, .re_stats.h5, .max_neighbors.h5, .energies"
    )


def _default_output(input_path: Path, kind: str) -> Path:
    """Pick an output PNG path next to the input file."""
    # Strip the recognised suffix.  ``.adaptation.h5`` and ``.re_stats.h5``
    # share a double extension; ``.energies`` is single.
    suffixes = {
        "adaptation": ".adaptation.h5",
        "re_stats": ".re_stats.h5",
        "max_neighbors": ".max_neighbors.h5",
        "energies": ".energies",
    }
    stem_name = input_path.name[: -len(suffixes[kind])]
    return input_path.parent / f"{stem_name}.{kind}.png"


def plot_adaptation(input_path: Path, output_path: Path) -> Path:
    """Render a 2-panel step-size + acceptance-rate figure from an
    ``.adaptation.h5`` file.  Both panels show mean ± std across replicas.
    """
    import matplotlib.pyplot as plt

    from jaxrens.io.adaptation_log import AdaptationLogger
    from jaxrens.postprocess.plotting import (
        plot_acceptance_rates,
        plot_step_sizes,
    )

    trace = AdaptationLogger.read(input_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    plot_step_sizes(trace, ax=axes[0], per_run=False)
    axes[0].set_title("Step size — mean ± std across replicas")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3, which="both")
    plot_acceptance_rates(trace, ax=axes[1], per_run=False)
    axes[1].set_title("Acceptance rate — mean ± std across replicas")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"adaptation trace · {input_path.name}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_re_stats(input_path: Path, output_path: Path) -> Path:
    """Render a per-pair swap-acceptance plot from a ``.re_stats.h5`` file."""
    import matplotlib.pyplot as plt

    from jaxrens.io.re_stats_log import RELogger
    from jaxrens.postprocess.plotting import plot_re_acceptance_stacked

    trace = RELogger.read(input_path)
    # Stacked, not overlaid: every adjacent pair gets its own lane, so a rung
    # whose acceptance dies is visible instead of being hidden under the
    # others in a single [0, 1] band.  Tuning a pressure ladder means reading
    # the pairs individually.
    n_pairs = int(trace.n_pairs)
    fig, ax = plt.subplots(figsize=(9, 2.0 + 1.1 * max(n_pairs, 1)))
    plot_re_acceptance_stacked(trace, ax=ax)
    ax.set_title(
        f"RE swap acceptance per pair ({trace.flavor}) · {input_path.name}",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_max_neighbors_file(input_path: Path, output_path: Path) -> Path:
    """Render a 2-panel neighbor-bucket diagnostic figure from a
    ``.max_neighbors.h5`` file.

    Top panel: per-walker count percentiles (p50/p90/p100) vs iteration,
    with the configured bucket size overlaid as a step line.  Bottom panel:
    log-density heatmap of the per-walker count distribution at each
    iteration (also with bucket overlay).  Both views use ``run=0`` for
    multi-run logs — for cohort-wide views use the library API directly.
    """
    import matplotlib.pyplot as plt

    from jaxrens.io.max_neighbors_log import MaxNeighborsLogger
    from jaxrens.postprocess.plotting import plot_max_neighbors

    trace = MaxNeighborsLogger.read(input_path)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    plot_max_neighbors(trace, ax=axes[0], kind="percentiles", show_bucket=True)
    axes[0].set_title("per-walker percentiles + bucket", fontsize=10)
    axes[0].grid(alpha=0.3)
    plot_max_neighbors(trace, ax=axes[1], kind="heatmap", show_bucket=True)
    axes[1].set_title("density heatmap (log-count)", fontsize=10)
    fig.suptitle(
        f"neighbor-bucket diagnostics · {input_path.name}", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


_REPLICA_RE = re.compile(r"^(?P<prefix>.+)\.run(?P<idx>\d+)\.energies$")


def _replica_label(path: Path) -> str:
    """``toy.run02.energies`` -> ``run02``."""
    m = _REPLICA_RE.match(path.name)
    return f"run{m.group('idx')}" if m else path.stem


def _replica_siblings(input_path: Path) -> list[Path]:
    """Every per-replica energies log belonging to the same run.

    A multi-replica run writes one ``<prefix>.runNN.energies`` per replica.
    Plotting a single one hides exactly the comparison you opened the plot
    for, so pointing at any of them picks up all of them, in replica order.
    A single-run log (no ``.runNN.`` infix) is returned on its own.
    """
    m = _REPLICA_RE.match(input_path.name)
    if m is None:
        return [input_path]
    siblings = sorted(
        input_path.parent.glob(f"{m.group('prefix')}.run*.energies"),
        key=lambda p: int(_REPLICA_RE.match(p.name).group("idx")),
    )
    return siblings or [input_path]


def plot_energies(input_path: Path, output_path: Path) -> Path:
    """Render the dead-point energy (and volume if NPT) trail from a
    ``.energies`` log."""
    import matplotlib.pyplot as plt
    import numpy as np

    from jaxrens.io.energy_log import EnergyLogger

    paths = _replica_siblings(input_path)
    logs = [(p, EnergyLogger.read(p)) for p in paths]
    has_volume = any(
        bool(np.any(np.asarray(e.volumes) != 0.0)) for _, e in logs
    )

    n_panels = 2 if has_volume else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for i, (path, elog) in enumerate(logs):
        iters = np.asarray(elog.iterations)
        label = _replica_label(path) if len(logs) > 1 else None
        axes[0].plot(
            iters,
            np.asarray(elog.energies),
            lw=0.8,
            color=f"C{i % 10}",
            label=label,
        )
        if has_volume:
            axes[1].plot(
                iters,
                np.asarray(elog.volumes),
                lw=0.8,
                color=f"C{i % 10}",
                label=label,
            )

    ref = logs[0][1]
    title = (
        f"dead-point energy trail · {input_path.name}"
        if len(logs) == 1
        else f"dead-point energy trail · {len(logs)} replicas"
    )
    axes[0].set_xlabel("NS iteration")
    axes[0].set_ylabel("dead-point energy  [model units]")
    axes[0].set_title(
        f"{title}\n"
        f"n_walkers={ref.n_walkers}, n_cull={ref.n_cull}, "
        f"n_atoms={ref.n_atoms}",
        fontsize=10,
    )
    axes[0].grid(alpha=0.3)
    if len(logs) > 1:
        axes[0].legend(fontsize=8, ncol=min(len(logs), 4))

    if has_volume:
        axes[1].set_xlabel("NS iteration")
        axes[1].set_ylabel("dead-point cell volume  [Å³]")
        axes[1].set_title("dead-point volume trail (NPT)", fontsize=10)
        axes[1].grid(alpha=0.3)
        if len(logs) > 1:
            axes[1].legend(fontsize=8, ncol=min(len(logs), 4))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


_DISPATCH = {
    "adaptation": plot_adaptation,
    "re_stats": plot_re_stats,
    "max_neighbors": plot_max_neighbors_file,
    "energies": plot_energies,
}


def plot_file(input_path: Path, output_path: Path | None = None) -> Path:
    """Auto-detect file kind and render the corresponding plot.

    Args:
        input_path: Path to ``.adaptation.h5`` / ``.re_stats.h5`` /
            ``.max_neighbors.h5`` / ``.energies`` file.
        output_path: Destination PNG.  Defaults to a sibling file
            ``<stem>.<kind>.png``.

    Returns:
        The path the figure was written to.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    kind = _suffix_kind(input_path)
    if output_path is None:
        output_path = _default_output(input_path, kind)
    else:
        output_path = Path(output_path)
    return _DISPATCH[kind](input_path, output_path)
