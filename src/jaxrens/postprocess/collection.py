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

    @classmethod
    def from_multi_run_directory(
        cls,
        path: Path | str,
        *,
        prefix: str = "ns",
        prefer_final: bool = True,
        labels: list[str] | None = None,
    ) -> "MonitorCollection":
        """Build a collection from a single multi-run output directory.

        Multi-run NS (``run_ns_parallel`` / ``run_ns_multi_gpu``) writes one
        combined checkpoint with leading batch shape ``(n_runs,)`` or
        ``(G, P)`` plus per-replica ``<prefix>.runNN.energies`` files. This
        constructor flattens the leading batch axes, reads each per-replica
        energy log, and produces one ``Monitor`` per replica.

        Convention: stored energies are treated as the full thermal variable
        (``dead_volumes = live_volumes = None``).  When the run used
        ``EnsembleBackend``, the stored energy is the NPT enthalpy
        ``H = U + P*V``, so leaving volumes ``None`` yields
        ``log Z_NPT`` / ``Cp`` / Gibbs ``G`` directly from the standard
        thermodynamics functions (passing the bare cell volume column would
        double-count the ``P*V`` term).  The bare cell-volume column from
        each ``.energies`` log is stashed on each Monitor as
        ``volume_trace`` so callers can compute ``<V>(T)`` separately.

        For single-run checkpoints (no batch axis) this delegates to
        ``Monitor.from_directory`` and returns a one-element collection.

        Args:
            path: Directory produced by ``jaxrens run`` on a multi-run config.
            prefix: File prefix matching ``output.out_file_prefix``.
            prefer_final: Prefer ``<prefix>.final.checkpoint.h5`` over the
                periodic ``<prefix>.checkpoint.h5``; falls back further to
                ``<prefix>.initial.checkpoint.h5`` when the run is still in
                flight.  Zero-byte files (mid-write) are skipped.
            labels: Per-replica display labels, length ``n_total``.  Defaults
                to ``f"run{i:02d}"``.

        Returns:
            Populated ``MonitorCollection``.

        Raises:
            FileNotFoundError: If no usable checkpoint is found, or if any
                ``<prefix>.runNN.energies`` file is missing.
            ValueError: If ``labels`` is supplied with the wrong length, or
                if the checkpoint's leading batch shape is unrecognised.
        """
        import json

        import h5py

        from jaxrens.io.checkpoint import load_checkpoint
        from jaxrens.io.energy_log import EnergyLogger

        path = Path(path)

        # Resolve checkpoint: skip 0-byte stubs from mid-write periodic saves.
        candidates: list[Path] = []
        if prefer_final:
            candidates.extend([
                path / f"{prefix}.final.checkpoint.h5",
                path / f"{prefix}.checkpoint.h5",
                path / f"{prefix}.initial.checkpoint.h5",
            ])
        else:
            candidates.extend([
                path / f"{prefix}.checkpoint.h5",
                path / f"{prefix}.final.checkpoint.h5",
                path / f"{prefix}.initial.checkpoint.h5",
            ])
        ckpt_path = next(
            (p for p in candidates if p.exists() and p.stat().st_size > 0),
            None,
        )
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No non-empty checkpoint in {path} (looked for "
                f"{', '.join(c.name for c in candidates)})"
            )

        state = load_checkpoint(ckpt_path)
        live_E = np.asarray(state["energies"])
        iteration = np.asarray(state["iteration"])
        log_Z = np.asarray(state["log_evidence"])
        n_live = int(state["n_walkers"])

        with h5py.File(ckpt_path, "r") as f:
            raw_symmap = f.attrs.get("symbol_map")
        symbol_map: dict[int, str] | None = (
            {int(k): v for k, v in json.loads(raw_symmap).items()}
            if raw_symmap is not None
            else None
        )

        # Single-run checkpoint (no batch axis): delegate.
        if live_E.ndim == 1:
            label = labels[0] if labels else None
            return cls(
                [Monitor.from_directory(
                    path, label=label, prefix=prefix, prefer_final=prefer_final,
                )]
            )

        # Multi-run: flatten leading batch axes (P,) or (G, P) → n_total.
        if live_E.ndim == 2:
            n_total = live_E.shape[0]
            live_E_flat = live_E
            log_Z_flat = log_Z
            iter_flat = iteration
        elif live_E.ndim == 3:
            G, P, K = live_E.shape
            n_total = G * P
            live_E_flat = live_E.reshape(n_total, K)
            log_Z_flat = log_Z.reshape(n_total)
            iter_flat = iteration.reshape(n_total)
        else:
            raise ValueError(
                f"Unexpected `energies` shape {live_E.shape} in "
                f"{ckpt_path.name}; expected 1, 2, or 3 dims."
            )

        if labels is not None and len(labels) != n_total:
            raise ValueError(
                f"len(labels)={len(labels)} does not match n_total={n_total} "
                f"derived from checkpoint shape {live_E.shape}"
            )

        monitors: list[Monitor] = []
        for i in range(n_total):
            elog_path = path / f"{prefix}.run{i:02d}.energies"
            if not elog_path.exists():
                raise FileNotFoundError(
                    f"Per-replica energies file missing: {elog_path}"
                )
            elog = EnergyLogger.read(elog_path)

            label = labels[i] if labels is not None else f"run{i:02d}"
            m = Monitor(
                dead_energies=np.asarray(elog.energies, dtype=np.float64),
                dead_volumes=None,
                live_energies=np.asarray(live_E_flat[i], dtype=np.float64),
                live_volumes=None,
                log_evidence=float(log_Z_flat[i]),
                iteration=int(iter_flat[i]),
                n_live=n_live,
                n_cull=1,
                symbol_map=symbol_map,
                energy_trace=np.asarray(elog.energies, dtype=np.float64),
                iteration_trace=np.asarray(elog.iterations, dtype=np.int64),
                adaptation_trace=None,
                label=label,
                path=path,
            )
            # Bare cell volume series, stashed for separate <V>(T) analysis.
            m.volume_trace = np.asarray(elog.volumes, dtype=np.float64)
            monitors.append(m)

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

    def plot_step_sizes(self, *, ax=None, per_run: bool = False, **kwargs):
        """Overlay step-size traces for all monitors that have adaptation data.

        Monitors without ``adaptation_trace`` are silently skipped.

        Args:
            ax:      Existing axes.  Created if None.
            per_run: Passed through to :func:`~jaxrens.postprocess.plotting.plot_step_sizes`.
            **kwargs: Forwarded to each ``plot_step_sizes`` call.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_step_sizes

        for monitor in self._monitors:
            if monitor.adaptation_trace is not None:
                ax = plot_step_sizes(monitor, ax=ax, per_run=per_run, **kwargs)
        return ax

    def plot_acceptance_rates(self, *, ax=None, per_run: bool = False, **kwargs):
        """Overlay acceptance-rate traces for all monitors that have adaptation data.

        Monitors without ``adaptation_trace`` are silently skipped.

        Args:
            ax:      Existing axes.  Created if None.
            per_run: Passed through to :func:`~jaxrens.postprocess.plotting.plot_acceptance_rates`.
            **kwargs: Forwarded to each ``plot_acceptance_rates`` call.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_acceptance_rates

        for monitor in self._monitors:
            if monitor.adaptation_trace is not None:
                ax = plot_acceptance_rates(monitor, ax=ax, per_run=per_run, **kwargs)
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
