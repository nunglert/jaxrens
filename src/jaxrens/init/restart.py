"""Load a full NS checkpoint into (live walkers, NS-state bundle).

Used by Mode D: restart_file — crash-recovery restart of a single NS run.
Also provides shape-aware dispatch via ``load_restart`` for multi-run / multi-GPU
checkpoints produced by ``run_ns_parallel`` (ndim=1) and ``run_ns_multi_gpu``
(ndim=2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import jax
import jax.numpy as jnp
import numpy as np

from jaxrens.init.walker_set import WalkerSet
from jaxrens.sampling.batch_descriptor import (
    PmapVmapRuns,
    SingleRun,
    VmapRuns,
    from_shape_prefix,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestartBundle:
    """NS-state fields needed to resume a run after a crash.

    Live-walker state (positions, types, cells, energies, symbol_map) is
    carried separately in ResolvedInit; this bundle holds only the NS-state
    scalars and dead-point history that init_ns needs to seed NSState.

    Attributes:
        dead_energies: Dead-point potentials, shape (n_dead,).
        dead_positions: Dead-point positions, shape (n_dead, n_atoms, 3).
        dead_volumes: Dead-point volumes, shape (n_dead,), or None for NVT.
        log_evidence: Running log-evidence estimate at checkpoint time.
        iteration: NS iteration count at checkpoint time.
        emax: NS contour at checkpoint time (last enforced by ``ns_step``).
            Legacy checkpoints lacking the ``emax`` field surface here as
            ``+inf``; the first post-restart ``ns_step`` rewrites it from
            ``pop.energy``, and ``run_loop`` suppresses adapt at the
            restored iter via the ``i > 0`` gate so a stale +inf can't
            leak into trial-walker constraints.
        n_dead: Number of dead points stored.
        rng_key_data: Raw uint32 PRNG buffer (output of ``jax.random.key_data``)
            for this slot, or ``None`` for legacy checkpoints that pre-date
            persisting the PRNG state.  ``init_ns`` re-wraps with
            ``jax.random.wrap_key_data`` when present, so resumed runs
            continue the saved random stream instead of silently reseeding
            from the CLI seed.
    """

    dead_energies: jnp.ndarray
    dead_positions: jnp.ndarray
    dead_volumes: jnp.ndarray | None
    log_evidence: float
    iteration: int
    emax: float
    n_dead: int
    rng_key_data: np.ndarray | None = None


@dataclass(frozen=True)
class BatchedRestart:
    """Multi-replica restart payload, flat ``(n_total, K, ...)`` layout.

    Consumed by ``_resolve_multi_replica`` to seed both live-walker arrays
    and per-replica NS-state. ``bundles_2d`` is the ``(n_gpu, n_per_gpu)``
    nested list that ``run_ns_multi_gpu(restart_states=...)`` expects;
    ``bundles_flat`` is the same data ordered as a length-``n_total`` list
    for ``run_ns_parallel``.

    Attributes:
        positions: Live-walker positions, shape ``(n_total, K, n_atoms, 3)``.
        types: Species, shape ``(n_total, K, n_atoms)`` — sliced per replica
            from the stored array so it matches positions/cells in leading dim.
        cells: Periodic cells, shape ``(n_total, K, 3, 3)``.
        symbol_map: Atomic-number → element-symbol mapping.
        bundles_2d: Per-replica NS-state, nested list shape ``(n_gpu, n_per_gpu)``.
        n_gpu: GPU dimension recovered from the checkpoint shape.
        n_per_gpu: Per-GPU replica count recovered from the checkpoint shape.
    """

    positions: jnp.ndarray
    types: jnp.ndarray
    cells: jnp.ndarray
    symbol_map: dict[int, str]
    bundles_2d: list[list[RestartBundle]]
    n_gpu: int
    n_per_gpu: int

    @property
    def n_total(self) -> int:
        return self.n_gpu * self.n_per_gpu

    @property
    def bundles_flat(self) -> list[RestartBundle]:
        return [b for row in self.bundles_2d for b in row]


# ---------------------------------------------------------------------------
# Shape-aware checkpoint loader
# ---------------------------------------------------------------------------


_JNP_FIELDS = (
    "positions", "types", "energies", "cells",
    "dead_energies", "dead_positions", "dead_volumes", "live_volumes",
    "log_evidence", "emax",
)


def _coerce_ckpt_to_jnp(ckpt: dict) -> None:
    """In-place: convert checkpoint array fields from numpy to jax.Array.

    ``load_checkpoint`` is intentionally numpy-only (so headless postprocess
    paths stay jax-free).  Restart consumers run the loaded arrays through
    JIT'd NS code which jaxtyping-checks for ``jax.Array``; convert here
    once.  Scalars (n_dead / iteration / n_walkers / rng_key) are left
    as-is — they're consumed as Python ints / Optional[Key] downstream.
    """
    for name in _JNP_FIELDS:
        val = ckpt.get(name)
        if val is not None:
            ckpt[name] = jnp.asarray(val)


def _build_bundle_from_ckpt(
    ckpt: dict,
    idx: tuple[int, ...] | None = None,
) -> RestartBundle:
    """Construct a single ``RestartBundle`` from a checkpoint dict.

    For scalar (ndim=0) checkpoints, ``idx`` must be ``None`` and the full
    arrays/scalars are used directly.

    For batched checkpoints, ``idx`` is the integer index tuple (e.g.
    ``(r,)`` for ndim=1 or ``(g, p)`` for ndim=2) used to slice every field.
    Dead arrays are trimmed to ``n_dead[idx]`` entries so that the bundle
    carries only valid (non-padded) dead points — matching the convention
    used by the scalar path and expected by ``init_ns``.

    Args:
        ckpt: Dict returned by ``load_checkpoint``.
        idx: Index tuple for batched checkpoints; ``None`` for scalar.

    Returns:
        A ``RestartBundle`` for the given slot.
    """
    # Raw uint32 PRNG buffer carried verbatim — wrapping happens in init_ns.
    # ``rng_key_data`` has trailing dim 2 (the key payload) and a leading
    # shape matching the checkpoint topology (``()`` for scalar, ``(R,)`` for
    # VmapRuns, ``(G, P)`` for PmapVmapRuns).
    full_rng_data = ckpt.get("rng_key_data")

    if idx is None:
        # Scalar single-run checkpoint: load_checkpoint already trimmed dead arrays.
        n_dead = int(ckpt["n_dead"])
        dead_energies = ckpt["dead_energies"][:n_dead]
        dead_positions = ckpt["dead_positions"][:n_dead]
        dead_volumes: jnp.ndarray | None = None
        if ckpt.get("dead_volumes") is not None:
            dead_volumes = ckpt["dead_volumes"][:n_dead]
        # ``load_checkpoint`` substitutes ``+inf`` for legacy ckpts missing
        # the emax field, so this is always present.
        emax = float(np.asarray(ckpt["emax"]))
        rng_key_data = (
            np.asarray(full_rng_data) if full_rng_data is not None else None
        )
        return RestartBundle(
            dead_energies=dead_energies,
            dead_positions=dead_positions,
            dead_volumes=dead_volumes,
            log_evidence=float(ckpt["log_evidence"]),
            iteration=int(ckpt["iteration"]),
            emax=emax,
            n_dead=n_dead,
            rng_key_data=rng_key_data,
        )
    else:
        # Batched checkpoint: slice each field by idx.
        # n_dead is an array of shape (n_runs,) or (G, P).
        n_dead = int(np.asarray(ckpt["n_dead"])[idx])

        # dead_energies / dead_positions: shape (*batch, max_dead) or (*batch, max_dead, ...)
        # Slice the leading batch dims with idx, then trim to [:n_dead].
        dead_energies = jnp.asarray(ckpt["dead_energies"])[idx][:n_dead]
        dead_positions = jnp.asarray(ckpt["dead_positions"])[idx][:n_dead]
        dead_volumes = None
        if ckpt.get("dead_volumes") is not None:
            dead_volumes = jnp.asarray(ckpt["dead_volumes"])[idx][:n_dead]

        log_evidence = float(np.asarray(ckpt["log_evidence"])[idx])
        iteration = int(np.asarray(ckpt["iteration"])[idx])
        # emax may be batched (shape matches log_evidence) or scalar +inf
        # (legacy ckpts: load_checkpoint substituted a scalar).  Scalar
        # broadcasts to every replica via the float() cast below.
        emax_arr = np.asarray(ckpt["emax"])
        emax = float(emax_arr[idx]) if emax_arr.ndim > 0 else float(emax_arr)

        # rng_key_data: shape (*batch, 2).  Slice the leading batch dims when
        # the saved key matches the topology; otherwise fall back to None so
        # init_ns uses the caller's key.  Production batched runs always save
        # shape-prefix-shaped keys; the fallback covers test fixtures and
        # legacy mismatches where a scalar key was saved alongside batched
        # fields.
        if full_rng_data is not None and full_rng_data.ndim >= len(idx) + 1:
            rng_key_data = np.asarray(full_rng_data)[idx]
        else:
            rng_key_data = None

        return RestartBundle(
            dead_energies=dead_energies,
            dead_positions=dead_positions,
            dead_volumes=dead_volumes,
            log_evidence=log_evidence,
            iteration=iteration,
            emax=emax,
            n_dead=n_dead,
            rng_key_data=rng_key_data,
        )


RestartShape = Union[
    "tuple[WalkerSet, RestartBundle]",
    "list[RestartBundle]",
    "list[list[RestartBundle]]",
]


def load_restart(
    path: Path,
) -> RestartShape:
    """Load a checkpoint HDF5 file and dispatch based on batch shape.

    Reuses ``io/checkpoint.load_checkpoint`` as the low-level reader. The
    return type depends on the checkpoint's ``log_evidence`` shape:

    * ``ndim == 0`` (scalar) → ``(WalkerSet, RestartBundle)`` pair.
      Feed ``bundle`` into ``run_ns(restart_state=bundle)``.
      Backward-compatible with all existing callers that do
      ``ws, bundle = load_restart(path)``.

    * ``ndim == 1`` (1-D, shape ``(n_runs,)``) → ``list[RestartBundle]``
      of length ``n_runs``.
      Feed directly into
      ``run_ns_parallel(..., restart_states=load_restart(path))``.

    * ``ndim == 2`` (2-D, shape ``(G, P)``) → ``list[list[RestartBundle]]``
      of shape ``(G, P)``.
      Feed directly into
      ``run_ns_multi_gpu(..., restart_states=load_restart(path))``.

    * ``ndim >= 3`` → raises ``ValueError`` with a message mentioning ``ndim``.

    Dead arrays in each ``RestartBundle`` are the compact slices trimmed to
    ``n_dead`` entries (not padded). ``init_ns`` pads them into pre-allocated
    arrays of size ``max_dead``.

    The list of sliceable fields is derived from
    ``dataclasses.fields(RestartBundle)`` at call time, so adding a new field
    to ``RestartBundle`` is automatically reflected without modifying this
    function.

    Args:
        path: Path to the checkpoint HDF5 file.

    Returns:
        Shape-dependent: see above.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the file is missing required NS-state fields, or if
            ``log_evidence.ndim >= 3``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {path}")

    import json
    import h5py as _h5py

    with _h5py.File(path, "r") as _f:
        present_datasets = set(_f.keys())
        present_attrs = set(_f.attrs.keys())
        symbol_map: dict[int, str] | None = None
        if "symbol_map" in _f.attrs:
            raw = json.loads(_f.attrs["symbol_map"])
            symbol_map = {int(k): v for k, v in raw.items()}

    missing_datasets = {"energies", "dead_energies", "dead_positions"} - present_datasets
    # log_evidence, iteration, n_dead may be stored as either attr or dataset
    # (see save_checkpoint / _store_field); check for presence in either location.
    missing = set()
    with _h5py.File(path, "r") as _f:
        for field in ("log_evidence", "iteration", "n_dead"):
            if field not in _f and field not in _f.attrs:
                missing.add(field)
    missing |= missing_datasets
    if missing:
        raise ValueError(
            f"file at {path} is not a valid NS checkpoint; missing fields: "
            f"{sorted(missing)}. "
            f"If this is a bare walker-set file, use start_walker_set instead."
        )

    from jaxrens.io.checkpoint import load_checkpoint

    ckpt = load_checkpoint(path)
    # ``load_checkpoint`` returns numpy arrays (jax-free, for headless
    # postprocess).  Production restart consumers (init_ns, ns_step) check
    # types via jaxtyping → convert the live + dead arrays here.
    _coerce_ckpt_to_jnp(ckpt)

    # Recover the saved batcher from log_evidence.shape.  The shape prefix is
    # ``()`` for SingleRun checkpoints, ``(R,)`` for VmapRuns, ``(G, P)`` for
    # PmapVmapRuns; ``from_shape_prefix`` raises on higher ranks.
    log_ev_arr = np.asarray(ckpt["log_evidence"])
    try:
        saved = from_shape_prefix(log_ev_arr.shape)
    except ValueError as exc:
        raise ValueError(
            f"load_restart: checkpoint at {path!r} has unsupported log_evidence "
            f"shape {log_ev_arr.shape}. {exc}"
        ) from exc

    # Strict cross-topology check: a real multi-GPU checkpoint can only be
    # restarted on a host with the matching device count.  ``saved.n_gpu == 1``
    # is treated as topology-agnostic (single-shard pmap is equivalent to
    # single-device vmap), so it restarts fine on any host; SingleRun and
    # VmapRuns checkpoints carry no GPU dim either and never trigger the
    # check.
    if isinstance(saved, PmapVmapRuns) and saved.n_gpu > 1:
        n_local = len(jax.local_devices())
        if saved.n_gpu != n_local:
            raise ValueError(
                f"Checkpoint at {path!r} was saved on n_gpu={saved.n_gpu} "
                f"(log_evidence shape {log_ev_arr.shape}); current host has "
                f"{n_local} local device(s).  Cross-topology restart is not "
                f"supported.  Slice the checkpoint along the n_gpu axis to "
                f"match the current device count, or rerun on a "
                f"{saved.n_gpu}-GPU host."
            )

    if isinstance(saved, SingleRun):
        # -------------------------------------------------------------------
        # Scalar single-run checkpoint — backward-compatible path.
        # Returns (WalkerSet, RestartBundle).
        # -------------------------------------------------------------------
        bundle = _build_bundle_from_ckpt(ckpt, idx=None)

        positions: jnp.ndarray = ckpt["positions"]
        types: jnp.ndarray = ckpt["types"]
        cells: jnp.ndarray | None = ckpt["cells"]

        if cells is None:
            raise ValueError(
                f"checkpoint at {path} has no 'cells' dataset. "
                f"jaxrens restart requires a periodic-cell checkpoint."
            )

        walker_set = WalkerSet(
            positions=positions,
            types=types,
            cells=cells,
            symbol_map=symbol_map or {},
        )

        logger.info(
            "Restart loaded from %s: n_dead=%d, iteration=%d, log_Z=%.4f",
            path, bundle.n_dead, bundle.iteration, bundle.log_evidence,
        )
        return walker_set, bundle

    elif isinstance(saved, VmapRuns):
        # -------------------------------------------------------------------
        # 1-D checkpoint: shape (n_runs,).
        # Returns list[RestartBundle] of length n_runs.
        # -------------------------------------------------------------------
        bundles: list[RestartBundle] = [
            _build_bundle_from_ckpt(ckpt, idx=(r,))
            for r in range(saved.n_runs)
        ]
        logger.info(
            "Restart loaded from %s: %d runs, n_dead=%s",
            path, saved.n_runs, [b.n_dead for b in bundles],
        )
        return bundles

    else:
        # PmapVmapRuns: 2-D checkpoint with shape (G, P).
        # Returns list[list[RestartBundle]] of shape (G, P).
        G, P = saved.n_gpu, saved.n_per_gpu
        bundles_2d: list[list[RestartBundle]] = [
            [_build_bundle_from_ckpt(ckpt, idx=(g, p)) for p in range(P)]
            for g in range(G)
        ]
        logger.info(
            "Restart loaded from %s: G=%d, P=%d, n_dead (first run)=%d",
            path, G, P, bundles_2d[0][0].n_dead,
        )
        return bundles_2d


def load_restart_batched(path: Path | str) -> BatchedRestart:
    """Multi-replica restart loader: flat-layout walker arrays + nested bundles.

    Companion to :func:`load_restart`. Where ``load_restart`` returns one of
    three different shapes depending on the checkpoint, this loader is for
    callers that have already established they are in the multi-replica
    branch (``n_total > 1``) and want a uniform ``(n_total, K, ...)`` layout
    with the nested ``(n_gpu, n_per_gpu)`` bundle list that
    ``run_ns_multi_gpu`` consumes.

    Handles both ``VmapRuns`` (1-D) and ``PmapVmapRuns`` (2-D) checkpoints;
    the 1-D case is normalised to ``n_gpu=1, n_per_gpu=R``. Scalar
    ``SingleRun`` checkpoints raise ``ValueError`` — those go through
    ``load_restart`` + ``_resolve_init_restart``.

    Args:
        path: Path to the checkpoint HDF5 file.

    Returns:
        A :class:`BatchedRestart` carrying the live-walker arrays in flat
        ``(n_total, K, ...)`` layout plus the ``(n_gpu, n_per_gpu)`` nested
        bundle list.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is a scalar single-run checkpoint, lacks
            required fields, or has unsupported ``log_evidence`` rank.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {path}")

    import json
    import h5py as _h5py

    with _h5py.File(path, "r") as _f:
        present_datasets = set(_f.keys())
        symbol_map: dict[int, str] | None = None
        if "symbol_map" in _f.attrs:
            raw = json.loads(_f.attrs["symbol_map"])
            symbol_map = {int(k): v for k, v in raw.items()}

    missing_datasets = {"energies", "dead_energies", "dead_positions"} - present_datasets
    missing = set()
    with _h5py.File(path, "r") as _f:
        for field in ("log_evidence", "iteration", "n_dead"):
            if field not in _f and field not in _f.attrs:
                missing.add(field)
    missing |= missing_datasets
    if missing:
        raise ValueError(
            f"file at {path} is not a valid NS checkpoint; missing fields: "
            f"{sorted(missing)}."
        )

    from jaxrens.io.checkpoint import load_checkpoint

    ckpt = load_checkpoint(path)
    # ``load_checkpoint`` returns numpy arrays (jax-free, for headless
    # postprocess).  Production restart consumers (init_ns, ns_step) check
    # types via jaxtyping → convert the live + dead arrays here.
    _coerce_ckpt_to_jnp(ckpt)

    log_ev_arr = np.asarray(ckpt["log_evidence"])
    try:
        saved = from_shape_prefix(log_ev_arr.shape)
    except ValueError as exc:
        raise ValueError(
            f"load_restart_batched: checkpoint at {path!r} has unsupported "
            f"log_evidence shape {log_ev_arr.shape}. {exc}"
        ) from exc

    if isinstance(saved, SingleRun):
        raise ValueError(
            f"load_restart_batched: checkpoint at {path!r} is a scalar "
            f"single-run checkpoint (log_evidence.shape={log_ev_arr.shape}); "
            f"use load_restart() / _resolve_init_restart for single-run."
        )

    if isinstance(saved, PmapVmapRuns) and saved.n_gpu > 1:
        n_local = len(jax.local_devices())
        if saved.n_gpu != n_local:
            raise ValueError(
                f"Checkpoint at {path!r} was saved on n_gpu={saved.n_gpu} "
                f"(log_evidence shape {log_ev_arr.shape}); current host has "
                f"{n_local} local device(s). Cross-topology restart is not "
                f"supported."
            )

    if isinstance(saved, VmapRuns):
        n_gpu, n_per_gpu = 1, saved.n_runs
    else:
        n_gpu, n_per_gpu = saved.n_gpu, saved.n_per_gpu
    n_total = n_gpu * n_per_gpu

    cells = ckpt["cells"]
    if cells is None:
        raise ValueError(
            f"checkpoint at {path} has no 'cells' dataset. "
            f"jaxrens restart requires a periodic-cell checkpoint."
        )

    positions = ckpt["positions"]
    types = ckpt["types"]

    def _flatten(arr: jnp.ndarray, batch_ndim: int) -> jnp.ndarray:
        return arr.reshape((n_total,) + arr.shape[batch_ndim:])

    batch_ndim = log_ev_arr.ndim
    positions_flat = _flatten(positions, batch_ndim)
    cells_flat = _flatten(cells, batch_ndim)
    types_flat = _flatten(types, batch_ndim)

    if isinstance(saved, VmapRuns):
        bundles_2d: list[list[RestartBundle]] = [
            [_build_bundle_from_ckpt(ckpt, idx=(p,)) for p in range(n_per_gpu)]
        ]
    else:
        bundles_2d = [
            [_build_bundle_from_ckpt(ckpt, idx=(g, p)) for p in range(n_per_gpu)]
            for g in range(n_gpu)
        ]

    logger.info(
        "Multi-replica restart loaded from %s: n_gpu=%d, n_per_gpu=%d "
        "(n_total=%d), n_dead (first replica)=%d",
        path, n_gpu, n_per_gpu, n_total, bundles_2d[0][0].n_dead,
    )

    return BatchedRestart(
        positions=positions_flat,
        types=types_flat,
        cells=cells_flat,
        symbol_map=symbol_map or {},
        bundles_2d=bundles_2d,
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
    )


# ---------------------------------------------------------------------------
# Convenience: infer which entry-point to use from a loaded bundle
# ---------------------------------------------------------------------------


def infer_restart_shape(
    bundle: RestartShape,
) -> Literal["single", "parallel", "multi_gpu"]:
    """Infer which ``run_ns`` entry-point matches the loaded restart bundle.

    Inspects the Python type of the value returned by ``load_restart``:

    * ``(WalkerSet, RestartBundle)`` tuple → ``"single"`` — feed into
      ``run_ns(restart_state=bundle)``.
    * ``list[RestartBundle]`` → ``"parallel"`` — feed into
      ``run_ns_parallel(..., restart_states=bundle)``.
    * ``list[list[RestartBundle]]`` → ``"multi_gpu"`` — feed into
      ``run_ns_multi_gpu(..., restart_states=bundle)``.

    Args:
        bundle: Return value of ``load_restart(path)``.

    Returns:
        One of ``"single"``, ``"parallel"``, or ``"multi_gpu"``.

    Raises:
        TypeError: If ``bundle`` does not match any of the three expected
            shapes.

    Example::

        loaded = load_restart("run.checkpoint.h5")
        shape = infer_restart_shape(loaded)
        if shape == "single":
            ws, b = loaded
            run_ns(..., restart_state=b)
        elif shape == "parallel":
            run_ns_parallel(..., restart_states=loaded)
        else:  # "multi_gpu"
            run_ns_multi_gpu(..., restart_states=loaded)
    """
    # Tuple of (WalkerSet, RestartBundle) → single run.
    if isinstance(bundle, tuple) and len(bundle) == 2:
        ws, b = bundle
        if isinstance(ws, WalkerSet) and isinstance(b, RestartBundle):
            return "single"

    # Flat list → check element type.
    if isinstance(bundle, list) and len(bundle) > 0:
        first = bundle[0]
        if isinstance(first, RestartBundle):
            return "parallel"
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], RestartBundle):
            return "multi_gpu"

    # Empty list is ambiguous; default to parallel to avoid crashing.
    if isinstance(bundle, list) and len(bundle) == 0:
        return "parallel"

    raise TypeError(
        f"infer_restart_shape: unrecognised bundle type {type(bundle)!r}. "
        f"Expected a (WalkerSet, RestartBundle) tuple, list[RestartBundle], "
        f"or list[list[RestartBundle]] as returned by load_restart()."
    )
