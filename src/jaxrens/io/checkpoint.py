"""Checkpoint save/load for NS run resumption.

HDF5-based checkpointing of full NS state.

Shape conventions
-----------------
``save_checkpoint`` and ``load_checkpoint`` are shape-agnostic.  The state
dict produced by ``run_ns`` (single run) has scalar ``log_evidence``,
``n_dead``, ``iteration``.  The dict produced by ``run_ns_parallel`` has
1-D ``(n_runs,)`` shapes for those fields.  The dict produced by
``run_ns_multi_gpu`` has 2-D ``(G, P)`` shapes.

All these shapes round-trip correctly:

* **Save**: arrays are stored as HDF5 datasets (any shape); scalars are stored
  as HDF5 attrs.  A field is treated as a scalar only when
  ``np.asarray(value).ndim == 0``.  ``float(...)`` / ``int(...)`` casts are
  avoided for fields that may be batched.
* **Load**: datasets are read back with their stored shape.  ``iteration``,
  ``n_dead``, and ``n_walkers`` are returned as plain Python ints (they are
  always stored as scalars — only ``log_evidence`` can be batched).
* **Dead-point padding** (``load_checkpoint``): works for 1-D dead arrays
  (``n_dead`` is a scalar).  For multi-run checkpoints, the caller is
  expected to save only the live-walker ``NSState``; use ``save_ns_state`` /
  ``load_ns_state`` (future) for batched restarts, or save per-run slices.

Batch-shape inference
---------------------
After loading, the caller can inspect ``state["log_evidence"].shape`` to
determine the batch shape:
* ``()`` → ``SingleRun``
* ``(n_runs,)`` → ``VmapRuns``
* ``(G, P)`` → ``PmapVmapRuns``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def _to_np(value: Any) -> np.ndarray:
    """Convert a JAX / numpy / Python numeric value to a numpy array."""
    return np.asarray(value)


def _store_field(f: h5py.File, key: str, value: Any) -> None:
    """Store *value* as an HDF5 dataset (any shape) or attr (scalar 0-D).

    Scalar 0-D numpy arrays are stored as attrs for backward compatibility
    with the ``f.attrs["iteration"]`` read pattern in old files.  Arrays with
    ``ndim >= 1`` are stored as datasets so their full shape is preserved.
    """
    arr = _to_np(value)
    if arr.ndim == 0:
        # Scalar — store as attr; use Python int/float for cleaner HDF5 type.
        py_val = arr.item()
        f.attrs[key] = py_val
    else:
        f.create_dataset(key, data=arr)


def save_checkpoint(
    path: Path | str,
    ns_state: dict,
    symbol_map: dict[int, str] | None = None,
) -> None:
    """Save NS state to HDF5 checkpoint.

    Handles any leading batch shape on ``log_evidence``, ``n_dead``,
    ``iteration``, ``dead_energies``, and ``dead_positions``:

    * ``SingleRun``: all scalar / 1-D — stored as attrs or 1-D datasets.
    * ``VmapRuns``: ``log_evidence`` / ``n_dead`` / ``iteration`` are 1-D
      ``(n_runs,)`` — stored as 1-D datasets.
    * ``PmapVmapRuns``: those fields are ``(G, P)`` — stored as 2-D datasets.

    The ``dead_energies`` slicing ``[:n_dead]`` is only applied when
    ``n_dead`` is a scalar (single run).  For batched runs the full padded
    array is saved (callers are expected to slice per-run before saving if
    space is a concern).

    Args:
        path: Path for the checkpoint file.
        ns_state: NS state dict from run_ns / run_ns_parallel / run_ns_multi_gpu.
        symbol_map: Optional mapping from type codes to element symbols.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_dead_arr = _to_np(ns_state["n_dead"])
    is_scalar_run = n_dead_arr.ndim == 0

    with h5py.File(path, "w") as f:
        f.create_dataset("positions", data=_to_np(ns_state["positions"]))
        f.create_dataset("types", data=_to_np(ns_state["types"]))
        f.create_dataset("energies", data=_to_np(ns_state["energies"]))
        if ns_state.get("cells") is not None:
            f.create_dataset("cells", data=_to_np(ns_state["cells"]))

        # Dead-point arrays are optional — current `_ns_state_to_checkpoint_dict`
        # omits them entirely (the canonical record lives in `.energies` /
        # `.traj`).  Older callers / tests that still populate them keep
        # working: write the array when present, skip otherwise.
        if is_scalar_run:
            n_dead_int = int(n_dead_arr)
            if ns_state.get("dead_energies") is not None:
                f.create_dataset(
                    "dead_energies",
                    data=_to_np(ns_state["dead_energies"][:n_dead_int]),
                )
            if ns_state.get("dead_positions") is not None:
                f.create_dataset(
                    "dead_positions",
                    data=_to_np(ns_state["dead_positions"][:n_dead_int]),
                )
            if ns_state.get("dead_volumes") is not None:
                f.create_dataset(
                    "dead_volumes",
                    data=_to_np(ns_state["dead_volumes"][:n_dead_int]),
                )
        else:
            # Batched run: save full padded arrays when present; batch dims
            # preserved.
            if ns_state.get("dead_energies") is not None:
                f.create_dataset(
                    "dead_energies", data=_to_np(ns_state["dead_energies"])
                )
            if ns_state.get("dead_positions") is not None:
                f.create_dataset(
                    "dead_positions", data=_to_np(ns_state["dead_positions"])
                )
            if ns_state.get("dead_volumes") is not None:
                f.create_dataset(
                    "dead_volumes", data=_to_np(ns_state["dead_volumes"])
                )

        if ns_state.get("live_volumes") is not None:
            f.create_dataset("live_volumes", data=_to_np(ns_state["live_volumes"]))

        # log_evidence may be batched — store as dataset when ndim >= 1.
        _store_field(f, "log_evidence", ns_state["log_evidence"])

        # iteration / n_dead / n_walkers are always scalar for single runs;
        # for batched runs they can be arrays.  Use _store_field to avoid
        # float()/int() casts on arrays.
        _store_field(f, "iteration", ns_state["iteration"])
        _store_field(f, "n_dead", ns_state["n_dead"])
        # n_walkers is always a plain Python int in all three result dicts.
        f.attrs["n_walkers"] = int(ns_state["n_walkers"])

        if symbol_map is not None:
            f.attrs["symbol_map"] = json.dumps(
                {str(k): v for k, v in symbol_map.items()}
            )

    # Log a safe iteration value regardless of batch shape.
    _iter_arr = _to_np(ns_state["iteration"])
    _iter_log = int(_iter_arr.flat[0]) if _iter_arr.size > 0 else -1
    logger.info("Checkpoint saved to %s (iteration %s)", path, _iter_log)


def _read_field(f: h5py.File, key: str) -> np.ndarray | int | float:
    """Read a field that may be stored either as an attr or a dataset.

    Returns a numpy array (dataset) or a Python scalar (attr).
    """
    if key in f:
        return f[key][()]   # numpy array, shape preserved
    elif key in f.attrs:
        return f.attrs[key]  # scalar attr
    else:
        raise KeyError(f"Field '{key}' not found in checkpoint (neither dataset nor attr).")


def load_checkpoint(
    path: Path | str,
    rng_key: jax.Array | None = None,
) -> dict:
    """Load NS state from HDF5 checkpoint.

    Handles checkpoints written by any of the three run variants:
    ``run_ns`` (scalar fields), ``run_ns_parallel`` (1-D), and
    ``run_ns_multi_gpu`` (2-D ``(G, P)``).

    The returned ``log_evidence`` has whatever shape was stored.  Call
    ``state["log_evidence"].shape`` to infer the batch configuration.

    Dead-array padding is only applied for **scalar** (single-run)
    checkpoints where ``n_dead`` is a plain integer.  For batched
    checkpoints the full stored arrays are returned as-is.

    Args:
        path: Path to checkpoint file.
        rng_key: New RNG key for resumed run. If None, uses key(0).

    Returns:
        NS state dict compatible with ns_step / run_ns and variants.
    """
    path = Path(path)
    if rng_key is None:
        rng_key = jax.random.key(0)

    with h5py.File(path, "r") as f:
        positions = jnp.array(f["positions"][()])
        types = jnp.array(f["types"][()])
        energies = jnp.array(f["energies"][()])
        cells = jnp.array(f["cells"][()]) if "cells" in f else None

        # dead_* are optional in the new HDF5 format (the canonical record
        # lives in `.energies` / `.traj`).  Treat missing fields as empty
        # so callers that only need live state work transparently; callers
        # that rely on dead-point data should source it from the streamed
        # files (see ``postprocess.Monitor.from_directory``).
        dead_energies_raw = (
            jnp.array(f["dead_energies"][()]) if "dead_energies" in f else None
        )
        dead_positions_raw = (
            jnp.array(f["dead_positions"][()]) if "dead_positions" in f else None
        )

        # log_evidence: stored as dataset (batched) or attr (scalar).
        log_evidence_raw = _read_field(f, "log_evidence")
        log_evidence = jnp.array(log_evidence_raw)

        # iteration / n_dead: may be dataset or attr.
        iteration_raw = _read_field(f, "iteration")
        n_dead_raw = _read_field(f, "n_dead")
        n_walkers = int(f.attrs["n_walkers"])

        dead_volumes_raw = jnp.array(f["dead_volumes"][()]) if "dead_volumes" in f else None
        live_volumes = jnp.array(f["live_volumes"][()]) if "live_volumes" in f else None

    # Determine whether this is a scalar (single-run) checkpoint.
    n_dead_np = np.asarray(n_dead_raw)
    is_scalar_run = n_dead_np.ndim == 0
    iteration_np = np.asarray(iteration_raw)

    if is_scalar_run:
        n_dead = int(n_dead_np)
        iteration = int(iteration_np)
        if dead_energies_raw is None:
            # New format: dead_* not in HDF5; consumer should read .energies.
            dead_energies = None
            dead_positions = None
            dead_volumes = None
        else:
            # Legacy format: pad dead arrays to a comfortable size for callers
            # that previously relied on padded shapes.
            max_dead = max(n_dead * 2, 50000)
            dead_energies = jnp.full(max_dead, jnp.inf)
            dead_energies = dead_energies.at[:n_dead].set(dead_energies_raw)
            dead_positions = jnp.zeros((max_dead, *positions.shape[1:]))
            dead_positions = dead_positions.at[:n_dead].set(dead_positions_raw)
            if dead_volumes_raw is not None:
                dead_volumes = jnp.zeros(max_dead)
                dead_volumes = dead_volumes.at[:n_dead].set(dead_volumes_raw)
            else:
                dead_volumes = None
    else:
        # Batched run: return stored arrays directly (no padding).
        n_dead = n_dead_np          # shape (n_runs,) or (G, P)
        iteration = iteration_np    # shape (n_runs,) or (G, P)
        dead_energies = dead_energies_raw
        dead_positions = dead_positions_raw
        dead_volumes = dead_volumes_raw

    _iter_log = int(np.asarray(iteration_raw).flat[0]) if np.asarray(iteration_raw).size > 0 else -1
    logger.info("Checkpoint loaded from %s (iteration %s)", path, _iter_log)

    return {
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": cells,
        "dead_energies": dead_energies,
        "dead_positions": dead_positions,
        "dead_volumes": dead_volumes,
        "live_volumes": live_volumes,
        "log_evidence": log_evidence,
        "iteration": iteration,
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": rng_key,
    }


def auto_detect_restart(working_dir: Path | str) -> Path | None:
    """Find the newest valid checkpoint in a directory.

    Args:
        working_dir: Directory to search.

    Returns:
        Path to newest checkpoint, or None if not found.
    """
    working_dir = Path(working_dir)
    checkpoints = sorted(
        working_dir.glob("*.checkpoint.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        return checkpoints[0]
    return None
