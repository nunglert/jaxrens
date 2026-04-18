"""Multi-run cohort aggregation: MonitorCollection.

Holds a list of Monitor objects and provides overlay plotting and
tabular summary.  All plotting delegates to postprocess.plotting
functions so there is no duplicated matplotlib logic here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from jaxrens.postprocess.monitor import Monitor


class MonitorCollection:
    """Aggregate multiple Monitor objects (e.g. a pressure sweep cohort).

    Args:
        monitors: Initial list of Monitor objects.
    """

    def __init__(self, monitors: list[Monitor]) -> None:
        self._monitors: list[Monitor] = list(monitors)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_directories(
        cls,
        paths: list[Path | str],
        labels: list[str] | None = None,
        *,
        prefix: str = "ns",
    ) -> "MonitorCollection":
        """Build a collection by loading one Monitor per directory.

        Args:
            paths: Per-run directories.
            labels: Display labels, one per path.  Defaults to directory names.
            prefix: Checkpoint file prefix.

        Returns:
            Populated MonitorCollection.
        """
        if labels is not None and len(labels) != len(paths):
            raise ValueError(
                f"len(labels)={len(labels)} must match len(paths)={len(paths)}"
            )
        monitors = []
        for i, p in enumerate(paths):
            label = labels[i] if labels is not None else None
            monitors.append(Monitor.from_directory(p, label=label, prefix=prefix))
        return cls(monitors)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, monitor: Monitor) -> None:
        """Append a Monitor to the collection.

        Args:
            monitor: Monitor to add.
        """
        self._monitors.append(monitor)

    def remove(self, label: str) -> None:
        """Remove the first Monitor whose label matches.

        Args:
            label: Label to match.

        Raises:
            KeyError: If no monitor with that label exists.
        """
        for i, m in enumerate(self._monitors):
            if m.label == label:
                del self._monitors[i]
                return
        raise KeyError(f"No Monitor with label {label!r}")

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._monitors)

    def __iter__(self) -> Iterator[Monitor]:
        return iter(self._monitors)

    def __getitem__(self, key: int | str) -> Monitor:
        """Index by integer position or by label string.

        Args:
            key: Integer index or label string.

        Raises:
            KeyError: If a string key is not found.
            IndexError: If an integer index is out of range.
        """
        if isinstance(key, str):
            for m in self._monitors:
                if m.label == key:
                    return m
            raise KeyError(f"No Monitor with label {key!r}")
        return self._monitors[key]

    # ------------------------------------------------------------------
    # Overlay plots
    # ------------------------------------------------------------------

    def plot_energy_trace(self, *, ax=None, **kwargs):
        """Overlay energy traces for all monitors on a single axes.

        Args:
            ax: Existing axes.  Created if None.
            **kwargs: Forwarded to each ``plot_energy_trace`` call.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_energy_trace

        for monitor in self._monitors:
            ax = plot_energy_trace(monitor, ax=ax, **kwargs)
        return ax

    def plot_heat_capacity(self, T: np.ndarray, *, ax=None, **kwargs):
        """Overlay C_v curves for all monitors.

        Args:
            T: Temperature array.
            ax: Existing axes.  Created if None.
            **kwargs: Forwarded to each ``plot_heat_capacity`` call.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_heat_capacity

        for monitor in self._monitors:
            ax = plot_heat_capacity(monitor, T, ax=ax, **kwargs)
        return ax

    def plot_partition_function(self, T: np.ndarray, *, ax=None, **kwargs):
        """Overlay log Z curves for all monitors.

        Args:
            T: Temperature array.
            ax: Existing axes.  Created if None.
            **kwargs: Forwarded to each ``plot_partition_function`` call.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_partition_function

        for monitor in self._monitors:
            ax = plot_partition_function(monitor, T, ax=ax, **kwargs)
        return ax

    def plot_free_energy(self, T: np.ndarray, *, ax=None, **kwargs):
        """Overlay free energy curves for all monitors.

        Args:
            T: Temperature array.
            ax: Existing axes.  Created if None.
            **kwargs: Forwarded to each ``plot_free_energy`` call.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_free_energy

        for monitor in self._monitors:
            ax = plot_free_energy(monitor, T, ax=ax, **kwargs)
        return ax

    def plot_log_evidence_trace(self, *, ax=None, **kwargs):
        """Overlay cumulative log Z traces for all monitors.

        Args:
            ax: Existing axes.  Created if None.
            **kwargs: Forwarded to each ``plot_log_evidence_trace`` call.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_log_evidence_trace

        for monitor in self._monitors:
            ax = plot_log_evidence_trace(monitor, ax=ax, **kwargs)
        return ax

    # ------------------------------------------------------------------
    # Tabular summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict summarising all monitors.

        Returns:
            Dict with keys ``'labels'``, ``'n_dead'``, ``'log_evidence'``,
            ``'is_npt'``, each mapping to a list of length ``len(self)``.
        """
        return {
            "labels": [m.label for m in self._monitors],
            "n_dead": [m.n_dead for m in self._monitors],
            "log_evidence": [m.log_evidence for m in self._monitors],
            "is_npt": [m.is_npt for m in self._monitors],
        }

    def __repr__(self) -> str:
        labels = [m.label for m in self._monitors]
        return f"<MonitorCollection n={len(self)} labels={labels!r}>"
