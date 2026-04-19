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

import jax.numpy as jnp
import numpy as np

from jaxrens.init.walker_set import WalkerSet

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
        n_dead: Number of dead points stored.
    """

    dead_energies: jnp.ndarray
    dead_positions: jnp.ndarray
    dead_volumes: jnp.ndarray | None
    log_evidence: float
    iteration: int
    n_dead: int


# ---------------------------------------------------------------------------
# Shape-aware checkpoint loader
# ---------------------------------------------------------------------------

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
    if idx is None:
        # Scalar single-run checkpoint: load_checkpoint already trimmed dead arrays.
        n_dead = int(ckpt["n_dead"])
        dead_energies = ckpt["dead_energies"][:n_dead]
        dead_positions = ckpt["dead_positions"][:n_dead]
        dead_volumes: jnp.ndarray | None = None
        if ckpt.get("dead_volumes") is not None:
            dead_volumes = ckpt["dead_volumes"][:n_dead]
        return RestartBundle(
            dead_energies=dead_energies,
            dead_positions=dead_positions,
            dead_volumes=dead_volumes,
            log_evidence=float(ckpt["log_evidence"]),
            iteration=int(ckpt["iteration"]),
            n_dead=n_dead,
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

        return RestartBundle(
            dead_energies=dead_energies,
            dead_positions=dead_positions,
            dead_volumes=dead_volumes,
            log_evidence=log_evidence,
            iteration=iteration,
            n_dead=n_dead,
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

    # Determine batch shape from log_evidence.
    log_ev_ndim = int(np.asarray(ckpt["log_evidence"]).ndim)

    if log_ev_ndim >= 3:
        raise ValueError(
            f"load_restart: checkpoint at {path!r} has log_evidence.ndim="
            f"{log_ev_ndim} (shape "
            f"{np.asarray(ckpt['log_evidence']).shape}). "
            f"Only ndim 0 (single run), 1 (parallel), and 2 (multi-GPU) are "
            f"supported. Got ndim={log_ev_ndim}."
        )

    if log_ev_ndim == 0:
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

    elif log_ev_ndim == 1:
        # -------------------------------------------------------------------
        # 1-D checkpoint: shape (n_runs,).
        # Returns list[RestartBundle] of length n_runs.
        # -------------------------------------------------------------------
        n_runs = int(np.asarray(ckpt["log_evidence"]).shape[0])
        bundles: list[RestartBundle] = [
            _build_bundle_from_ckpt(ckpt, idx=(r,))
            for r in range(n_runs)
        ]
        logger.info(
            "Restart loaded from %s: %d runs, n_dead=%s",
            path, n_runs, [b.n_dead for b in bundles],
        )
        return bundles

    else:
        # log_ev_ndim == 2
        # -------------------------------------------------------------------
        # 2-D checkpoint: shape (G, P).
        # Returns list[list[RestartBundle]] of shape (G, P).
        # -------------------------------------------------------------------
        log_ev_arr = np.asarray(ckpt["log_evidence"])
        G, P = int(log_ev_arr.shape[0]), int(log_ev_arr.shape[1])
        bundles_2d: list[list[RestartBundle]] = [
            [_build_bundle_from_ckpt(ckpt, idx=(g, p)) for p in range(P)]
            for g in range(G)
        ]
        logger.info(
            "Restart loaded from %s: G=%d, P=%d, n_dead (first run)=%d",
            path, G, P, bundles_2d[0][0].n_dead,
        )
        return bundles_2d


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
