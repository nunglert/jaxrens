"""Single-run NS analysis: Monitor class.

Loads a jaxrens run from disk (checkpoint + energy log) and provides
thin wrappers over postprocess.thermodynamics for computing observables.
All scientific computation delegates to thermodynamics.py; this module
is pure data-loading and numpy-level orchestration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from jaxrens.postprocess.thermodynamics import (
    class_probability,
    conditional_mean,
)
from jaxrens.postprocess.thermodynamics import free_energy as _free_energy
from jaxrens.postprocess.thermodynamics import heat_capacity as _heat_capacity
from jaxrens.postprocess.thermodynamics import log_point_weights
from jaxrens.postprocess.thermodynamics import (
    partition_function as _partition_function,
)
from jaxrens.postprocess.thermodynamics import (
    thermal_weights,
    weighted_covariance,
    weighted_mean,
    weighted_variance,
)


def _optional_array(x, dtype) -> np.ndarray | None:
    return None if x is None else np.asarray(x, dtype=dtype)


def per_walker(x, shape: tuple[int, ...]) -> np.ndarray | None:
    """``x`` broadcast to ``shape``, or None if it is missing or does not fit.

    Checkpoints may store the types (and cells) once for all walkers instead
    of once per walker.  The live configurations are auxiliary data, so an
    array that cannot be read per walker leaves them unset rather than
    failing the load.
    """
    if x is None:
        return None
    try:
        return np.broadcast_to(np.asarray(x), shape)
    except ValueError:
        return None


class Monitor:
    """Load and analyze a single jaxrens NS run.

    Reads the checkpoint HDF5 (preferred) plus the .energies file for the
    iteration trace.  All scientific computations delegate to
    postprocess.thermodynamics.

    In NPT the stored energies are already enthalpies ``H = U + P V``
    (``EnsembleTerm`` adds ``P V`` during sampling), so the observables use
    them as they are.  The cell volumes are kept as data only: they set
    ``is_npt`` and feed ``<V>(T)`` via ``expectation(volume_trace, T)``.

    Attributes:
        dead_energies: Dead-point energies, shape (n_dead,); enthalpies in
            NPT.
        dead_volumes: Dead-point cell volumes for NPT runs, shape (n_dead,),
            or None.
        live_energies: Live-walker energies at run end, shape (n_live_actual,).
        live_volumes: Live-walker cell volumes for NPT, shape
            (n_live_actual,), or None.
        live_positions: Live-walker positions at run end, shape
            (n_live_actual, n_atoms, 3), or None.
        live_cells: Live-walker cells (rows are lattice vectors), shape
            (n_live_actual, 3, 3), or None for a non-periodic run.
        live_types: Live-walker type codes, shape (n_live_actual, n_atoms),
            or None.  With ``symbol_map`` and the positions and cells, the
            live walkers can be turned into structures, so per-point
            observables cover them exactly.
        log_evidence: Scalar log Z stored in the checkpoint.
        iteration: Iteration number at checkpoint.
        n_live: Number of live walkers used in the run.
        n_cull: Number of walkers culled per iteration.
        symbol_map: Mapping from integer type codes to element symbols, or None.
        energy_trace: Per-iteration culled energy from .energies file, or None.
        iteration_trace: Iteration indices matching energy_trace, or None.
        adaptation_trace: Per-move step sizes and acceptance rates, or None.
        re_trace: Per-fire inter-replica swap counts and acceptance rates, or
            None.
        max_neighbors_trace: Per-iteration neighbor-bucket diagnostics
            (per-walker max_neighbor_count + bucket size + overflow), or
            None.  Loaded from ``<prefix>.max_neighbors.h5`` when the run
            opted in via ``output.save_max_neighbors``.
        label: Human-readable label for plots.
        path: Source directory, or None.
        volume_trace: Cell volume of each dead point from the .energies
            log, shape (n_dead,).  Set by ``from_directory`` (and by
            ``MonitorCollection.from_multi_run_directory``) only.
    """

    @classmethod
    def from_directory(
        cls,
        path: Path | str,
        label: str | None = None,
        *,
        prefix: str = "ns",
        prefer_final: bool = True,
    ) -> "Monitor":
        """Load a run from a directory.

        Picks ``<prefix>.final.checkpoint.h5`` if it exists and
        ``prefer_final=True``, otherwise falls back to
        ``<prefix>.checkpoint.h5``.  Raises ``FileNotFoundError`` if neither
        exists.

        Args:
            path: Directory produced by ``jaxrens run``.
            label: Display label.  Defaults to the directory name.
            prefix: File prefix used by the run (matches ``output.out_file_prefix``
                in the config, default ``"ns"``).
            prefer_final: Prefer the ``.final.checkpoint.h5`` over the periodic
                checkpoint.

        Returns:
            Populated Monitor.
        """
        # Lazy import to avoid circular deps and allow headless use.
        from jaxrens.io.checkpoint import load_checkpoint
        from jaxrens.io.energy_log import EnergyLogger

        path = Path(path)
        if label is None:
            label = path.name

        final_ckpt = path / f"{prefix}.final.checkpoint.h5"
        periodic_ckpt = path / f"{prefix}.checkpoint.h5"

        if prefer_final and final_ckpt.exists():
            ckpt_path = final_ckpt
        elif periodic_ckpt.exists():
            ckpt_path = periodic_ckpt
        elif final_ckpt.exists():
            ckpt_path = final_ckpt
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {path!r} with prefix {prefix!r}. "
                f"Looked for: {final_ckpt} and {periodic_ckpt}"
            )

        # Load checkpoint via existing IO layer.
        state = load_checkpoint(ckpt_path)

        n_dead: int = int(state["n_dead"])
        n_live: int = int(state["n_walkers"])

        live_energies = np.asarray(state["energies"], dtype=np.float64)

        live_volumes = None
        if state.get("live_volumes") is not None:
            live_volumes = np.asarray(state["live_volumes"], dtype=np.float64)

        # Live-walker configurations.  Types and cells may be stored once
        # for all walkers; broadcast them to one per walker.
        live_positions = np.asarray(state["positions"], dtype=np.float64)
        n_walkers_live, n_atoms = live_positions.shape[:2]
        live_types = per_walker(state["types"], (n_walkers_live, n_atoms))
        live_cells = per_walker(state.get("cells"), (n_walkers_live, 3, 3))

        energies_path = path / f"{prefix}.energies"
        # Pre-load the energies log when present — used for both the dead-
        # array fallback and the iteration trace below.
        energies_log = (
            EnergyLogger.read(energies_path)
            if energies_path.exists()
            else None
        )

        if energies_log is not None:
            # New canonical path: read dead_energies (and dead_volumes when
            # NPT) from the per-iteration ``.energies`` log written by
            # ``EnergyLogger``.  The volume column is zero for NVT runs;
            # drop it when ``live_volumes`` is also absent (NVT signal).
            dead_energies = np.asarray(energies_log.energies, dtype=np.float64)
            if live_volumes is not None:
                dead_volumes = np.asarray(
                    energies_log.volumes, dtype=np.float64
                )
            else:
                dead_volumes = None
        else:
            raise FileNotFoundError(
                f"Checkpoint {ckpt_path} has no dead_* arrays and "
                f"{energies_path} is missing — cannot reconstruct dead-point "
                f"trace.  Re-run with EnergyLogger enabled or restore an "
                f"older checkpoint that included dead arrays in HDF5."
            )

        # symbol_map stored as JSON string in HDF5 attrs; we need to re-read it
        # because load_checkpoint does not expose it.
        import h5py

        symbol_map: dict[int, str] | None = None
        with h5py.File(ckpt_path, "r") as f:
            if "symbol_map" in f.attrs:
                raw = f.attrs["symbol_map"]
                symbol_map = {int(k): v for k, v in json.loads(raw).items()}

        # Iteration trace from the same energies log we used above (if any).
        if energies_log is not None:
            energy_trace = energies_log.energies
            iteration_trace = energies_log.iterations
        else:
            energy_trace = None
            iteration_trace = None

        # Optional adaptation trace.
        adaptation_trace = None
        adaptation_path = path / f"{prefix}.adaptation.h5"
        if adaptation_path.exists():
            from jaxrens.io.adaptation_log import AdaptationLogger

            adaptation_trace = AdaptationLogger.read(adaptation_path)

        # Optional inter-RE swap trace.
        re_trace = None
        re_path = path / f"{prefix}.re_stats.h5"
        if re_path.exists():
            from jaxrens.io.re_stats_log import RELogger

            re_trace = RELogger.read(re_path)

        # Optional max-neighbors diagnostic trace.
        max_neighbors_trace = None
        mn_path = path / f"{prefix}.max_neighbors.h5"
        if mn_path.exists():
            from jaxrens.io.max_neighbors_log import MaxNeighborsLogger

            max_neighbors_trace = MaxNeighborsLogger.read(mn_path)

        monitor = cls(
            dead_energies=dead_energies,
            dead_volumes=dead_volumes,
            live_energies=live_energies,
            live_volumes=live_volumes,
            log_evidence=float(state["log_evidence"]),
            iteration=int(state["iteration"]),
            n_live=n_live,
            # The ``.energies`` header records the run's cull width (and every
            # file written before multi-cull existed says 1, which is what
            # those runs actually did).
            n_cull=energies_log.n_cull if energies_log is not None else 1,
            symbol_map=symbol_map,
            energy_trace=energy_trace,
            iteration_trace=iteration_trace,
            adaptation_trace=adaptation_trace,
            re_trace=re_trace,
            max_neighbors_trace=max_neighbors_trace,
            live_positions=live_positions,
            live_cells=live_cells,
            live_types=live_types,
            label=label,
            path=path,
        )
        # Bare cell volume per dead point, for <V>(T); same attribute as
        # MonitorCollection.from_multi_run_directory sets.
        monitor.volume_trace = np.asarray(
            energies_log.volumes, dtype=np.float64
        )
        return monitor

    def __init__(
        self,
        *,
        dead_energies: np.ndarray,
        dead_volumes: np.ndarray | None,
        live_energies: np.ndarray,
        live_volumes: np.ndarray | None,
        log_evidence: float,
        iteration: int,
        n_live: int,
        n_cull: int = 1,
        symbol_map: dict[int, str] | None = None,
        energy_trace: np.ndarray | None = None,
        iteration_trace: np.ndarray | None = None,
        adaptation_trace=None,
        re_trace=None,
        max_neighbors_trace=None,
        live_positions: np.ndarray | None = None,
        live_cells: np.ndarray | None = None,
        live_types: np.ndarray | None = None,
        label: str = "",
        path: Path | None = None,
    ) -> None:
        self.dead_energies = np.asarray(dead_energies, dtype=np.float64)
        self.dead_volumes = (
            np.asarray(dead_volumes, dtype=np.float64)
            if dead_volumes is not None
            else None
        )
        self.live_energies = np.asarray(live_energies, dtype=np.float64)
        self.live_volumes = (
            np.asarray(live_volumes, dtype=np.float64)
            if live_volumes is not None
            else None
        )
        self.log_evidence = float(log_evidence)
        self.iteration = int(iteration)
        self.n_live = int(n_live)
        self.n_cull = int(n_cull)
        self.symbol_map = symbol_map
        self.energy_trace = (
            np.asarray(energy_trace, dtype=np.float64)
            if energy_trace is not None
            else None
        )
        self.iteration_trace = (
            np.asarray(iteration_trace, dtype=np.int64)
            if iteration_trace is not None
            else None
        )
        # adaptation_trace is an AdaptationLog dataclass or None
        self.adaptation_trace = adaptation_trace
        # re_trace is an RELog dataclass or None
        self.re_trace = re_trace
        # max_neighbors_trace is a MaxNeighborsLog dataclass or None
        self.max_neighbors_trace = max_neighbors_trace
        self.live_positions = _optional_array(live_positions, np.float64)
        self.live_cells = _optional_array(live_cells, np.float64)
        self.live_types = _optional_array(live_types, np.int64)
        self.label = label
        self.path = Path(path) if path is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_dead(self) -> int:
        """Number of dead points."""
        return int(self.dead_energies.shape[0])

    @property
    def is_npt(self) -> bool:
        """True if volume information is present (NPT ensemble)."""
        return self.dead_volumes is not None

    @property
    def n_points(self) -> int:
        """Number of points the thermal averages run over: dead + live."""
        return self.n_dead + int(self.live_energies.shape[0])

    @property
    def point_energies(self) -> np.ndarray:
        """Energies of the dead points followed by the live walkers,
        shape ``(n_points,)``; enthalpies ``H = U + P V`` in NPT."""
        arrs = self._arrays()
        return np.concatenate([arrs["dead_e"], arrs["live_e"]])

    @property
    def point_volumes(self) -> np.ndarray | None:
        """Cell volumes of the dead points followed by the live walkers,
        shape ``(n_points,)``, or None when the run has no volumes (NVT)
        or the live volumes are missing."""
        if self.dead_volumes is None or self.live_volumes is None:
            return None
        return np.concatenate([self.dead_volumes, self.live_volumes])

    # ------------------------------------------------------------------
    # Observable methods — thin wrappers over thermodynamics.py
    # ------------------------------------------------------------------
    #
    # Every method takes the temperature ``T`` as a scalar or a 1-D array
    # and computes the thermal weights once for the whole grid.  A scalar
    # ``T`` gives a 0-d result (``(k,)`` for a vector observable), an array
    # a leading ``n_T`` axis.  Per-point observables have one value per
    # point, dead points first then live walkers: shape ``(n_points,)`` or
    # ``(n_points, k)``.  Values for the dead points only, ``(n_dead,)`` or
    # ``(n_dead, k)``, are averaged over the dead points alone: their
    # weights are renormalised and the live walkers are left out.
    # :meth:`live_weight` tells how much weight that leaves out.

    def _beta(self, T: np.ndarray | float, k_B: float = 1.0) -> np.ndarray:
        """Convert temperature to inverse temperature array."""
        T = np.asarray(T, dtype=np.float64)
        if T.ndim > 1:
            raise ValueError(f"T must be a scalar or 1-D, got shape {T.shape}")
        return 1.0 / (k_B * T)

    def _arrays(self):
        """Cache the energies as float64 numpy arrays.

        The volumes are deliberately not used: in NPT the energies already
        contain ``P V``, and adding the volume again double-counts it.
        """
        if not hasattr(self, "_cached_np"):
            self._cached_np = {
                "dead_e": np.asarray(self.dead_energies, dtype=np.float64),
                "live_e": np.asarray(self.live_energies, dtype=np.float64),
            }
        return self._cached_np

    def _per_point(self, values, name: str, dtype=np.float64):
        """Per-point values and whether they include the live walkers.

        Returns ``(values, with_live)``: ``with_live`` is True for
        ``(n_points,)`` / ``(n_points, k)`` values and False for values on
        the dead points only, ``(n_dead,)`` / ``(n_dead, k)``.
        """
        values = np.asarray(values, dtype=dtype)
        if values.ndim in (1, 2):
            if values.shape[0] == self.n_points:
                return values, True
            if values.shape[0] == self.n_dead:
                return values, False
        raise ValueError(
            f"{name} must have shape (n_points,) or (n_points, k) with "
            f"n_points={self.n_points} (dead + live), or (n_dead,) / "
            f"(n_dead, k) with n_dead={self.n_dead}; got {values.shape}"
        )

    def _dead_weights(self, T: np.ndarray | float, k_B: float) -> np.ndarray:
        """Thermal weights of the dead points alone, renormalised to one.

        Computed in log space, so they stay finite where the live walkers
        carry nearly all the weight.
        """
        beta = self._beta(T, k_B)
        dead_e = self._arrays()["dead_e"]
        log_w = log_point_weights(self.n_dead, self.n_live, self.n_cull)[
            : self.n_dead
        ]
        if beta.ndim == 0:
            log_t = log_w - beta * dead_e
        else:
            log_t = log_w - beta[:, None] * dead_e
        return np.exp(log_t - logsumexp(log_t, axis=-1, keepdims=True))

    def _weights_for(self, T, k_B, *items):
        """Weights and values on a common set of points.

        ``items`` are ``(values, with_live)`` pairs from :meth:`_per_point`.
        If all of them include the live walkers, every point is used;
        otherwise only the dead points, with renormalised weights.
        """
        if all(with_live for _, with_live in items):
            return self.weights(T, k_B), [v for v, _ in items]
        w = self._dead_weights(T, k_B)
        return w, [v[: self.n_dead] for v, _ in items]

    def weights(self, T: np.ndarray | float, k_B: float = 1.0) -> np.ndarray:
        """Normalised thermal weights of the dead points then live walkers.

        Shape ``(n_points,)`` for a scalar ``T`` and ``(n_T, n_points)``
        for an array.  ``weights(T) @ obs`` is ``<O>(T)``; compute it once
        to evaluate many observables on the same grid with the estimators
        of :mod:`jaxrens.postprocess.thermodynamics`.
        """
        arrs = self._arrays()
        return thermal_weights(
            self._beta(T, k_B),
            arrs["dead_e"],
            arrs["live_e"],
            n_live=self.n_live,
            n_cull=self.n_cull,
        )

    def live_weight(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Total thermal weight of the live walkers, ``Σ_live w_i(T)``.

        The share of every average that an observable given on the dead
        points only leaves out.  Close to 0 at high ``T``; it grows towards
        1 near and below the temperature the run ended at.
        """
        return self.weights(T, k_B)[..., self.n_dead :].sum(axis=-1)[()]

    def partition_function(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Log partition function ``log Z(T)``, relative to the prior."""
        arrs = self._arrays()
        return _partition_function(
            self._beta(T, k_B),
            arrs["dead_e"],
            arrs["live_e"],
            n_live=self.n_live,
            n_cull=self.n_cull,
        )[()]

    def log_Z(self, T: np.ndarray | float) -> np.ndarray:
        """``log Z(T)`` in reduced units (``k_B = 1``); see
        :meth:`partition_function`."""
        return self.partition_function(T, k_B=1.0)

    def free_energy(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Free energy ``F = -k_B T log Z`` (Gibbs ``G`` in NPT), up to the
        prior normalisation, a term proportional to ``T``."""
        beta = self._beta(T, k_B)
        return _free_energy(beta, self.partition_function(T, k_B))[()]

    def heat_capacity(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Heat capacity in units of ``k_B``, ``beta^2 Var(E)``; ``C_P`` in
        NPT."""
        arrs = self._arrays()
        return _heat_capacity(
            self._beta(T, k_B),
            arrs["dead_e"],
            arrs["live_e"],
            n_live=self.n_live,
            n_cull=self.n_cull,
        )[()]

    def mean_energy(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Mean energy ``<E>(T)``; the mean enthalpy ``<H>`` in NPT."""
        return weighted_mean(self.weights(T, k_B), self.point_energies)[()]

    def entropy(self, T: np.ndarray | float, k_B: float = 1.0) -> np.ndarray:
        """Entropy ``S = (<E> - F) / T = k_B (beta <E> + log Z)``.

        Known up to an additive constant: ``log Z`` is relative to the
        prior, whose normalisation is not included.  Differences in ``T``
        (and in ``P`` within one run) are exact; comparing systems needs
        the absolute normalisation.
        """
        beta = self._beta(T, k_B)
        w, log_Z = self._weights_and_log_Z(beta)
        return (k_B * (beta * (w @ self.point_energies) + log_Z))[()]

    def _weights_and_log_Z(self, beta: np.ndarray):
        arrs = self._arrays()
        return thermal_weights(
            beta,
            arrs["dead_e"],
            arrs["live_e"],
            n_live=self.n_live,
            n_cull=self.n_cull,
            return_log_Z=True,
        )

    def expectation(
        self,
        observable: np.ndarray,
        T: np.ndarray | float,
        k_B: float = 1.0,
    ) -> np.ndarray:
        """Thermal expectation ``<O>(T)``.

        Args:
            observable: Per-point values, ``(n_points,)`` or
                ``(n_points, k)``, dead points then live walkers.  Values
                on the dead points only, ``(n_dead,)`` or ``(n_dead, k)``,
                are averaged over the dead points alone (see
                :meth:`live_weight` for the weight left out).
            T: Temperature(s), scalar or 1-D.
            k_B: Boltzmann constant in the same energy units as the stored
                ``dead_energies``.  Defaults to ``1.0``; pass e.g.
                ``8.617e-5`` (eV/K) when ``T`` is in Kelvin and the energies
                are in eV.
        """
        w, (obs,) = self._weights_for(
            T, k_B, self._per_point(observable, "observable")
        )
        return weighted_mean(w, obs)[()]

    def variance(
        self,
        observable: np.ndarray,
        T: np.ndarray | float,
        k_B: float = 1.0,
    ) -> np.ndarray:
        """Thermal variance ``Var(O)(T)``; same arguments as
        :meth:`expectation`."""
        w, (obs,) = self._weights_for(
            T, k_B, self._per_point(observable, "observable")
        )
        return weighted_variance(w, obs)[()]

    def covariance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        T: np.ndarray | float,
        k_B: float = 1.0,
    ) -> np.ndarray:
        """Thermal covariance ``Cov(A, B)(T)``.

        ``a`` and ``b`` are per-point values as in :meth:`expectation`; a
        scalar against a vector observable gives one covariance per
        component.  If either is given on the dead points only, both are
        taken on the dead points.  In NPT, ``-d<A>/dP = beta Cov(A, V)`` (pass
        :attr:`point_volumes` as ``b``) and
        ``d<A>/dT = Cov(A, H) / (k_B T^2)`` (pass :attr:`point_energies`).
        """
        w, (a, b) = self._weights_for(
            T, k_B, self._per_point(a, "a"), self._per_point(b, "b")
        )
        return weighted_covariance(w, a, b)[()]

    def class_probability(
        self,
        mask: np.ndarray,
        T: np.ndarray | float,
        k_B: float = 1.0,
    ) -> np.ndarray:
        """Thermal probability of a class of points, ``<1_C>(T)``.

        Args:
            mask: Class membership, boolean ``(n_points,)`` or
                ``(n_points, k)`` for ``k`` classes, dead points then live
                walkers; or on the dead points only, as in
                :meth:`expectation`.
            T: Temperature(s), scalar or 1-D.
            k_B: Boltzmann constant.
        """
        w, (mask,) = self._weights_for(
            T, k_B, self._per_point(mask, "mask", dtype=bool)
        )
        return class_probability(w, mask)[()]

    def conditional_expectation(
        self,
        observable: np.ndarray,
        T: np.ndarray | float,
        k_B: float = 1.0,
        *,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Thermal average within a class, ``<O 1_C> / <1_C>``.

        Points where the observable is NaN are left out, so it also
        averages an observable that is undefined on some points.  NaN where
        the class has zero weight.

        Args:
            observable: Per-point values as in :meth:`expectation`, NaN for
                undefined.
            T: Temperature(s), scalar or 1-D.
            k_B: Boltzmann constant.
            mask: Class membership as in :meth:`class_probability`; None
                keeps every point.  If the observable or the mask is given
                on the dead points only, both are taken on the dead points.
        """
        items = [self._per_point(observable, "observable")]
        if mask is not None:
            items.append(self._per_point(mask, "mask", dtype=bool))
        w, values = self._weights_for(T, k_B, *items)
        mask = values[1] if mask is not None else None
        return conditional_mean(w, values[0], mask)[()]

    def __repr__(self) -> str:
        return (
            f"<Monitor label={self.label!r} n_dead={self.n_dead}"
            f" log_Z={self.log_evidence:.4f}>"
        )
