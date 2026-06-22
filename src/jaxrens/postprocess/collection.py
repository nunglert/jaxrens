"""Multi-run cohort aggregation: MonitorCollection.

Holds a list of Monitor objects and provides overlay plotting and
tabular summary.  All plotting delegates to postprocess.plotting
functions so there is no duplicated matplotlib logic here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from jaxrens.postprocess.monitor import Monitor


def _observable_volume(monitor: Monitor, T: np.ndarray, **kwargs) -> np.ndarray:
    if not hasattr(monitor, "volume_trace"):
        raise AttributeError(
            f"Monitor {monitor.label!r} has no volume_trace; "
            "the 'volume' heatmap needs a multi-run NPT collection built via "
            "MonitorCollection.from_multi_run_directory."
        )
    return monitor.expectation(monitor.volume_trace, T, **kwargs)


_OBSERVABLE_DISPATCH: dict[str, Callable[..., np.ndarray]] = {
    "heat_capacity": lambda m, T, **k: m.heat_capacity(T, **k),
    "free_energy": lambda m, T, **k: m.free_energy(T, **k),
    "log_partition_function": lambda m, T, **k: m.partition_function(T, **k),
    "log_Z": lambda m, T, **k: m.partition_function(T, **k),
    "volume": _observable_volume,
}


def _discover_config(path: Path) -> Path | None:
    """Find a ``config.yaml`` next to the run output.

    Looks first in ``path`` itself (rare — the working dir), then in
    ``path.parent`` (the standard layout where ``output/`` is a sibling
    of ``config.yaml``).
    """
    for candidate in (path / "config.yaml", path.parent / "config.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _load_yaml(config_path: Path) -> dict[str, Any]:
    import yaml
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def _prefix_from_config(cfg: dict[str, Any]) -> str | None:
    """Return ``output.out_file_prefix`` from a parsed config, or None."""
    output = cfg.get("output") or {}
    prefix = output.get("out_file_prefix")
    return str(prefix) if prefix is not None else None


def _labels_and_metadata_from_config(
    cfg: dict[str, Any], n_total: int,
) -> tuple[list[str] | None, list[dict[str, Any]]]:
    """Derive overlay labels and per-replica metadata from a config dict.

    Currently understands the ``ensemble`` section: for an NPT cohort
    with a pressure list whose length matches ``n_total`` we emit
    ``"P=… GPa"`` (or eV/Å³) labels and attach ``pressure_gpa`` /
    ``pressure_eva3`` to each monitor.  Returns ``(None, [{}, …])`` when
    nothing useful can be derived — callers fall back to ``runNN``.
    """
    empty_meta: list[dict[str, Any]] = [{} for _ in range(n_total)]
    ensemble = cfg.get("ensemble") or {}
    if ensemble.get("type") != "npt":
        return None, empty_meta

    pressure = ensemble.get("pressure")
    if pressure is None:
        return None, empty_meta
    if isinstance(pressure, (int, float)):
        pressure = [pressure]
    pressure = list(pressure)
    if len(pressure) != n_total:
        return None, empty_meta

    units = ensemble.get("pressure_units", "eva3")
    if units == "gpa":
        labels = [f"P={p:>5.2f} GPa" for p in pressure]
        metadata = [{"pressure_gpa": float(p)} for p in pressure]
    else:
        labels = [f"P={p:.4g} eV/Å³" for p in pressure]
        metadata = [{"pressure_eva3": float(p)} for p in pressure]
    return labels, metadata


class MonitorCollection:
    """Aggregate multiple Monitor objects (e.g. a pressure sweep cohort).

    Args:
        monitors: Initial list of Monitor objects.
    """

    def __init__(self, monitors: list[Monitor]) -> None:
        self._monitors: list[Monitor] = list(monitors)
        # Shared adaptation trace across the cohort (populated by
        # ``from_multi_run_directory`` when the ``.adaptation.h5`` file is
        # present).  Stays ``None`` for collections built by hand or by
        # ``from_directories``; the per-replica ``Monitor.adaptation_trace``
        # remains the per-replica record in those cases.
        self.adaptation_trace = None

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
        prefix: str | None = None,
        prefer_final: bool = True,
        labels: list[str] | None = None,
        config: Path | str | None = None,
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

        For single-run checkpoints (no batch axis) this builds a one-element
        collection using the same convention as the multi-run path (reading
        the ``<prefix>.energies`` log, ``dead_volumes=None``, bare volume
        stashed as ``volume_trace``) rather than delegating to
        ``Monitor.from_directory``, so single- and multi-replica collections
        behave identically.

        Config auto-discovery: when ``config`` is ``None`` the constructor
        looks for ``config.yaml`` in ``path`` and then in ``path.parent``
        (the standard ``experiment_dir/output/`` layout).  A found config
        supplies the run prefix (``output.out_file_prefix``) and — for an
        NPT pressure sweep whose pressure-list length matches the replica
        count — per-replica labels and ``pressure_gpa`` / ``pressure_eva3``
        attributes attached to each Monitor.  Pass ``prefix=`` / ``labels=``
        explicitly to override the inferred values, or hand a specific
        path via ``config=`` to skip auto-discovery.  When no config is
        found we fall back to ``prefix='ns'`` and ``runNN`` labels.

        Args:
            path: Directory produced by ``jaxrens run`` on a multi-run config.
            prefix: File prefix matching ``output.out_file_prefix``.  If
                ``None`` (default), inferred from the config when present;
                otherwise falls back to ``"ns"``.
            prefer_final: Prefer ``<prefix>.final.checkpoint.h5`` over the
                periodic ``<prefix>.checkpoint.h5``; falls back further to
                ``<prefix>.initial.checkpoint.h5`` when the run is still in
                flight.  Zero-byte files (mid-write) are skipped.
            labels: Per-replica display labels, length ``n_total``.  If
                ``None``, inferred from the config (e.g. NPT pressures);
                otherwise falls back to ``f"run{i:02d}"``.
            config: Explicit path to the run's YAML config.  When ``None``,
                auto-discovery looks in ``path`` then ``path.parent``.

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

        # Resolve config: explicit path > auto-discovery > none.
        cfg: dict[str, Any] = {}
        if config is not None:
            cfg_path = Path(config)
            if not cfg_path.is_file():
                raise FileNotFoundError(f"config not found: {cfg_path}")
            cfg = _load_yaml(cfg_path)
        else:
            auto = _discover_config(path)
            if auto is not None:
                cfg = _load_yaml(auto)

        if prefix is None:
            prefix = _prefix_from_config(cfg) or "ns"

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

        # Resolve leading batch axes → n_total replicas, funnelling all cases
        # through the same per-replica construction loop below.  A 1-D
        # ``energies`` is a single-run checkpoint (one replica, no batch
        # axis); 2-D is a flat ``(P,)`` sweep; 3-D a ``(G, P)`` grid.  The
        # single-run case is built here rather than delegated to
        # ``Monitor.from_directory`` so single- and multi-replica collections
        # carry identical conventions: ``dead_volumes=None`` (energy column is
        # the full thermal variable) with the bare cell volume stashed as
        # ``volume_trace``.  Delegating instead yielded a Monitor that lacked
        # ``volume_trace`` and folded the volume column into log Z.
        if live_E.ndim == 1:
            n_total = 1
            single_run = True
            live_E_flat = live_E[None, :]
            log_Z_flat = np.atleast_1d(log_Z)
            iter_flat = np.atleast_1d(iteration)
        elif live_E.ndim == 2:
            n_total = live_E.shape[0]
            single_run = False
            live_E_flat = live_E
            log_Z_flat = log_Z
            iter_flat = iteration
        elif live_E.ndim == 3:
            G, P, K = live_E.shape
            n_total = G * P
            single_run = False
            live_E_flat = live_E.reshape(n_total, K)
            log_Z_flat = log_Z.reshape(n_total)
            iter_flat = iteration.reshape(n_total)
        else:
            raise ValueError(
                f"Unexpected `energies` shape {live_E.shape} in "
                f"{ckpt_path.name}; expected 1, 2, or 3 dims."
            )

        # Derive labels + per-replica metadata from config when caller did
        # not supply explicit labels.  ``metadata`` is always n_total long.
        inferred_labels, metadata = _labels_and_metadata_from_config(cfg, n_total)
        if labels is None and inferred_labels is not None:
            labels = inferred_labels

        if labels is not None and len(labels) != n_total:
            raise ValueError(
                f"len(labels)={len(labels)} does not match n_total={n_total} "
                f"derived from checkpoint shape {live_E.shape}"
            )

        monitors: list[Monitor] = []
        for i in range(n_total):
            elog_path = (
                path / f"{prefix}.energies"
                if single_run
                else path / f"{prefix}.run{i:02d}.energies"
            )
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
            # Attach per-replica metadata from the config (e.g. pressure_gpa).
            for k, v in metadata[i].items():
                setattr(m, k, v)
            monitors.append(m)

        coll = cls(monitors)

        # Pick up the cohort-wide adaptation log if the run wrote one.
        # Multi-run NS produces a single ``<prefix>.adaptation.h5`` with
        # arrays of shape ``(n_entries, n_runs, n_moves)`` — one row per
        # replica along the second axis.  We stash the full trace on the
        # collection so ``plot_step_sizes`` / ``plot_acceptance_rates``
        # can render mean±std across replicas or per-replica overlays.
        adaptation_path = path / f"{prefix}.adaptation.h5"
        if adaptation_path.exists():
            from jaxrens.io.adaptation_log import AdaptationLogger
            coll.adaptation_trace = AdaptationLogger.read(adaptation_path)

        return coll

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
        """Plot per-move step-size adaptation trace for the cohort.

        Uses the collection-level ``self.adaptation_trace`` populated by
        ``from_multi_run_directory`` (one shared ``.adaptation.h5`` for the
        whole cohort, indexed along an ``n_runs`` axis).  Falls back to
        per-monitor traces when the cohort-level trace is unavailable
        (e.g. collections built via ``from_directories``).

        Args:
            ax:      Existing axes.  Created if None.
            per_run: When True, draws one line per (move, replica)
                combination; when False, draws mean ± std across replicas.
            **kwargs: Forwarded to the underlying plotting function.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_step_sizes

        if self.adaptation_trace is not None:
            return plot_step_sizes(
                self.adaptation_trace, ax=ax, per_run=per_run, **kwargs,
            )
        for monitor in self._monitors:
            if monitor.adaptation_trace is not None:
                ax = plot_step_sizes(monitor, ax=ax, per_run=per_run, **kwargs)
        return ax

    def plot_heatmap(
        self,
        T: np.ndarray,
        observable: str | Callable[..., np.ndarray] = "heat_capacity",
        *,
        ax=None,
        cmap: str = "viridis",
        fmt: str = "PT",
        vmin: float | None = None,
        vmax: float | None = None,
        do_colorbar: bool = True,
        cbar_label: str | None = None,
        T_label: str = "T",
        P_label: str | None = None,
        pressure_attr: str = "auto",
        **obs_kwargs,
    ):
        """Render a pressure-temperature heatmap of an observable.

        Each monitor in the collection contributes one row of the
        ``(n_P, n_T)`` grid.  Pressure values are taken from the
        ``pressure_gpa`` or ``pressure_eva3`` attribute set by
        ``from_multi_run_directory`` when a config is found; when neither is
        present we fall back to ordinal replica indices (the heatmap then
        functions purely as a 2D overlay).  Monitors are sorted by pressure
        before plotting so ``pcolormesh`` sees a monotonic axis.

        Args:
            T: Temperature grid, shape ``(n_T,)``.  Passed to each monitor's
                observable method.
            observable: One of ``"heat_capacity"``, ``"free_energy"``,
                ``"log_partition_function"``/``"log_Z"``, ``"volume"``, or a
                callable ``(monitor, T, **obs_kwargs) -> array (n_T,)``.
            ax: Existing axes to plot into.  Created if None.
            cmap, fmt, vmin, vmax, do_colorbar, cbar_label, T_label, P_label:
                Forwarded to :func:`~jaxrens.postprocess.plotting.plot_heatmap`.
                ``P_label`` defaults to a unit-aware string when
                ``pressure_attr`` is resolved to a named field.
            pressure_attr: ``"auto"`` (default) prefers ``pressure_gpa`` then
                ``pressure_eva3`` then ordinal indices; pass an explicit
                attribute name to force a choice, or ``"index"`` for replica
                indices.
            **obs_kwargs: Forwarded to the observable function
                (e.g. ``k_B=8.617e-5``).

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_heatmap

        if not self._monitors:
            raise ValueError("Cannot plot a heatmap from an empty collection")

        # Resolve pressure-axis source.
        if pressure_attr == "auto":
            if hasattr(self._monitors[0], "pressure_gpa"):
                pressure_attr = "pressure_gpa"
            elif hasattr(self._monitors[0], "pressure_eva3"):
                pressure_attr = "pressure_eva3"
            else:
                pressure_attr = "index"

        if pressure_attr == "index":
            pressures = np.arange(len(self._monitors), dtype=float)
            P_label_resolved = P_label or "replica index"
        else:
            try:
                pressures = np.array(
                    [getattr(m, pressure_attr) for m in self._monitors],
                    dtype=float,
                )
            except AttributeError as exc:
                raise AttributeError(
                    f"Not every monitor carries {pressure_attr!r}; "
                    "pass pressure_attr='index' or build the collection via "
                    "from_multi_run_directory with a config that declares an "
                    "NPT pressure list."
                ) from exc
            P_label_resolved = P_label or (
                "P [GPa]" if pressure_attr == "pressure_gpa"
                else r"P [eV/Å³]" if pressure_attr == "pressure_eva3"
                else pressure_attr
            )

        # Resolve observable function.
        if isinstance(observable, str):
            try:
                fn = _OBSERVABLE_DISPATCH[observable]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown observable {observable!r}; valid choices: "
                    f"{sorted(_OBSERVABLE_DISPATCH)}"
                ) from exc
            cbar_label_resolved = cbar_label or observable
        else:
            fn = observable
            cbar_label_resolved = cbar_label

        # Compute observable per monitor, in pressure-sorted order.
        T_arr = np.asarray(T, dtype=np.float64)
        order = np.argsort(pressures)
        pressures_sorted = pressures[order]
        Z = np.empty((len(self._monitors), T_arr.shape[0]), dtype=np.float64)
        for row, idx in enumerate(order):
            Z[row] = np.asarray(fn(self._monitors[idx], T_arr, **obs_kwargs))

        return plot_heatmap(
            T_arr, pressures_sorted, Z,
            ax=ax, cmap=cmap, fmt=fmt, vmin=vmin, vmax=vmax,
            do_colorbar=do_colorbar, cbar_label=cbar_label_resolved,
            T_label=T_label, P_label=P_label_resolved,
        )

    def plot_acceptance_rates(self, *, ax=None, per_run: bool = False, **kwargs):
        """Plot per-move acceptance-rate trace for the cohort.

        Uses the cohort-wide ``self.adaptation_trace`` (set by
        ``from_multi_run_directory``) when present, falling back to
        per-monitor traces otherwise.

        Args:
            ax:      Existing axes.  Created if None.
            per_run: When True, one line per (move, replica); when False,
                mean ± std across replicas.
            **kwargs: Forwarded to the underlying plotting function.

        Returns:
            The Axes object.
        """
        from jaxrens.postprocess.plotting import plot_acceptance_rates

        if self.adaptation_trace is not None:
            return plot_acceptance_rates(
                self.adaptation_trace, ax=ax, per_run=per_run, **kwargs,
            )
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
