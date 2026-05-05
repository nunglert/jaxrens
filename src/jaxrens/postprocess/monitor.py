"""Single-run NS analysis: Monitor class.

Loads a jaxrens run from disk (checkpoint + energy log) and provides
thin wrappers over postprocess.thermodynamics for computing observables.
All scientific computation delegates to thermodynamics.py; this module
is pure data-loading and numpy-level orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    expectation as _expectation,
    free_energy as _free_energy,
    heat_capacity as _heat_capacity,
    log_evidence as _log_evidence,
    partition_function as _partition_function,
)


class Monitor:
    """Load and analyze a single jaxrens NS run.

    Reads the checkpoint HDF5 (preferred) plus the .energies file for the
    iteration trace.  All scientific computations delegate to
    postprocess.thermodynamics.

    Attributes:
        dead_energies: Dead-point energies, shape (n_dead,).
        dead_volumes: Dead-point volumes for NPT runs, shape (n_dead,), or None.
        live_energies: Live-walker energies at run end, shape (n_live_actual,).
        live_volumes: Live-walker volumes for NPT, shape (n_live_actual,), or None.
        log_evidence: Scalar log Z stored in the checkpoint.
        iteration: Iteration number at checkpoint.
        n_live: Number of live walkers used in the run.
        n_cull: Number of walkers culled per iteration.
        symbol_map: Mapping from integer type codes to element symbols, or None.
        energy_trace: Per-iteration culled energy from .energies file, or None.
        iteration_trace: Iteration indices matching energy_trace, or None.
        adaptation_trace: Per-move step sizes and acceptance rates, or None.
        label: Human-readable label for plots.
        path: Source directory, or None.
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

        # Source for dead arrays: prefer HDF5 (legacy / future write paths
        # that include them), fall back to the streamed ``.energies`` text
        # file (the canonical record under the current architecture, where
        # dead arrays no longer live in HDF5).
        de_h5 = state.get("dead_energies")
        dv_h5 = state.get("dead_volumes")
        h5_has_dead = (
            de_h5 is not None and np.asarray(de_h5).size > 0
        )
        energies_path = path / f"{prefix}.energies"
        # Pre-load the energies log when present — used for both the dead-
        # array fallback and the iteration trace below.
        energies_log = (
            EnergyLogger.read(energies_path) if energies_path.exists() else None
        )

        if h5_has_dead:
            # Prefer HDF5 dead arrays when present — preserves behaviour for
            # checkpoints written by older jaxrens versions.
            dead_energies = np.asarray(de_h5[:n_dead], dtype=np.float64)
            dead_volumes = (
                np.asarray(dv_h5[:n_dead], dtype=np.float64)
                if dv_h5 is not None else None
            )
        elif energies_log is not None:
            # New canonical path: read dead_energies (and dead_volumes when
            # NPT) from the per-iteration ``.energies`` log written by
            # ``EnergyLogger``.  The volume column is zero for NVT runs;
            # drop it when ``live_volumes`` is also absent (NVT signal).
            dead_energies = np.asarray(energies_log.energies, dtype=np.float64)
            if live_volumes is not None:
                dead_volumes = np.asarray(energies_log.volumes, dtype=np.float64)
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

        return cls(
            dead_energies=dead_energies,
            dead_volumes=dead_volumes,
            live_energies=live_energies,
            live_volumes=live_volumes,
            log_evidence=float(state["log_evidence"]),
            iteration=int(state["iteration"]),
            n_live=n_live,
            n_cull=1,  # jaxrens runs with n_cull=1 by default
            symbol_map=symbol_map,
            energy_trace=energy_trace,
            iteration_trace=iteration_trace,
            adaptation_trace=adaptation_trace,
            label=label,
            path=path,
        )

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

    # ------------------------------------------------------------------
    # Observable methods — thin wrappers over thermodynamics.py
    # ------------------------------------------------------------------

    def _beta(self, T: np.ndarray | float, k_B: float = 1.0) -> np.ndarray:
        """Convert temperature to inverse temperature array."""
        T = np.asarray(T, dtype=np.float64)
        return 1.0 / (k_B * T)

    def _jax_arrays(self):
        """Convert numpy arrays to jnp arrays once, cache for reuse."""
        import jax.numpy as jnp
        if not hasattr(self, "_cached_jax"):
            self._cached_jax = {
                "dead_e": jnp.asarray(self.dead_energies),
                "live_e": jnp.asarray(self.live_energies),
                "dead_v": (
                    jnp.asarray(self.dead_volumes)
                    if self.dead_volumes is not None else None
                ),
                "live_v": (
                    jnp.asarray(self.live_volumes)
                    if self.live_volumes is not None else None
                ),
            }
        return self._cached_jax

    def _vmap_over_beta(self, scalar_fn, betas, **fn_kwargs):
        """Apply scalar_fn(beta, ...) over an array of betas via vmap+jit.

        Returns a numpy array with one entry per beta.
        """
        import jax
        import jax.numpy as jnp

        jitted = jax.jit(scalar_fn)
        vmapped = jax.vmap(
            lambda b: jitted(b, **fn_kwargs),
            in_axes=0,
        )
        return np.asarray(vmapped(jnp.asarray(betas)))

    def log_Z(self, T: np.ndarray | float) -> np.ndarray:
        """Compute log partition function at each temperature."""
        T = np.asarray(T, dtype=np.float64)
        scalar = T.ndim == 0
        betas = 1.0 / np.atleast_1d(T)
        arrs = self._jax_arrays()

        def scalar_fn(beta):
            return _partition_function(
                beta, arrs["dead_e"], arrs["live_e"],
                n_live=self.n_live, n_cull=self.n_cull,
                dead_volumes=arrs["dead_v"], live_volumes=arrs["live_v"],
            )

        results = self._vmap_over_beta(scalar_fn, betas)
        return results[0] if scalar else results

    def heat_capacity(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Compute heat capacity C_v at each temperature."""
        T = np.asarray(T, dtype=np.float64)
        scalar = T.ndim == 0
        betas = self._beta(np.atleast_1d(T), k_B)
        arrs = self._jax_arrays()

        def scalar_fn(beta):
            return _heat_capacity(
                beta, arrs["dead_e"], arrs["live_e"],
                n_live=self.n_live, n_cull=self.n_cull,
            )

        results = self._vmap_over_beta(scalar_fn, betas)
        return results[0] if scalar else results

    def expectation(
        self, observable: np.ndarray, T: np.ndarray | float
    ) -> np.ndarray:
        """Compute thermal expectation <O>(T) for a per-dead-point observable."""
        import jax.numpy as jnp

        observable = np.asarray(observable, dtype=np.float64)
        if observable.shape != (self.n_dead,):
            raise ValueError(
                f"observable must have shape (n_dead,)=({self.n_dead},), "
                f"got {observable.shape}"
            )

        T = np.asarray(T, dtype=np.float64)
        scalar = T.ndim == 0
        betas = 1.0 / np.atleast_1d(T)
        arrs = self._jax_arrays()

        live_obs = jnp.full(self.live_energies.shape[0], float(np.mean(observable)))
        obs_full = jnp.concatenate([jnp.asarray(observable), live_obs])

        def scalar_fn(beta):
            return _expectation(
                obs_full, beta, arrs["dead_e"], arrs["live_e"],
                n_live=self.n_live, n_cull=self.n_cull,
            )

        results = self._vmap_over_beta(scalar_fn, betas)
        return results[0] if scalar else results

    def free_energy(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Compute Helmholtz free energy F = -log Z / beta at each temperature."""
        T = np.asarray(T, dtype=np.float64)
        scalar = T.ndim == 0
        betas = self._beta(np.atleast_1d(T), k_B)
        arrs = self._jax_arrays()

        def scalar_fn(beta):
            logZ = _partition_function(
                beta, arrs["dead_e"], arrs["live_e"],
                n_live=self.n_live, n_cull=self.n_cull,
                dead_volumes=arrs["dead_v"], live_volumes=arrs["live_v"],
            )
            return _free_energy(beta, logZ)

        results = self._vmap_over_beta(scalar_fn, betas)
        return results[0] if scalar else results

    def partition_function(
        self, T: np.ndarray | float, k_B: float = 1.0
    ) -> np.ndarray:
        """Compute log Z(T) at each temperature."""
        T = np.asarray(T, dtype=np.float64)
        scalar = T.ndim == 0
        betas = self._beta(np.atleast_1d(T), k_B)
        arrs = self._jax_arrays()

        def scalar_fn(beta):
            return _partition_function(
                beta, arrs["dead_e"], arrs["live_e"],
                n_live=self.n_live, n_cull=self.n_cull,
                dead_volumes=arrs["dead_v"], live_volumes=arrs["live_v"],
            )

        results = self._vmap_over_beta(scalar_fn, betas)
        return results[0] if scalar else results

    def __repr__(self) -> str:
        return (
            f"<Monitor label={self.label!r} n_dead={self.n_dead}"
            f" log_Z={self.log_evidence:.4f}>"
        )
