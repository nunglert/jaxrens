"""Checkpoint save/load for NS run resumption.

HDF5-based checkpointing of full NS state.
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


def save_checkpoint(
    path: Path | str,
    ns_state: dict,
    symbol_map: dict[int, str] | None = None,
) -> None:
    """Save NS state to HDF5 checkpoint.

    Args:
        path: Path for the checkpoint file.
        ns_state: NS state dict from run_ns / ns_step.
        symbol_map: Optional mapping from type codes to element symbols.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("positions", data=np.asarray(ns_state["positions"]))
        f.create_dataset("types", data=np.asarray(ns_state["types"]))
        f.create_dataset("energies", data=np.asarray(ns_state["energies"]))
        if ns_state.get("boxes") is not None:
            f.create_dataset("boxes", data=np.asarray(ns_state["boxes"]))
        f.create_dataset(
            "dead_energies",
            data=np.asarray(ns_state["dead_energies"][: ns_state["n_dead"]]),
        )
        f.create_dataset(
            "dead_positions",
            data=np.asarray(ns_state["dead_positions"][: ns_state["n_dead"]]),
        )
        if ns_state.get("dead_volumes") is not None:
            f.create_dataset(
                "dead_volumes",
                data=np.asarray(ns_state["dead_volumes"][: ns_state["n_dead"]]),
            )
        if ns_state.get("live_volumes") is not None:
            f.create_dataset(
                "live_volumes",
                data=np.asarray(ns_state["live_volumes"]),
            )
        f.attrs["log_evidence"] = float(ns_state["log_evidence"])
        f.attrs["iteration"] = int(ns_state["iteration"])
        f.attrs["n_dead"] = int(ns_state["n_dead"])
        f.attrs["n_walkers"] = int(ns_state["n_walkers"])
        if symbol_map is not None:
            f.attrs["symbol_map"] = json.dumps(
                {str(k): v for k, v in symbol_map.items()}
            )

    logger.info("Checkpoint saved to %s (iteration %d)", path, ns_state["iteration"])


def load_checkpoint(
    path: Path | str,
    rng_key: jax.Array | None = None,
) -> dict:
    """Load NS state from HDF5 checkpoint.

    Args:
        path: Path to checkpoint file.
        rng_key: New RNG key for resumed run. If None, uses key(0).

    Returns:
        NS state dict compatible with ns_step / run_ns.
    """
    path = Path(path)
    if rng_key is None:
        rng_key = jax.random.key(0)

    with h5py.File(path, "r") as f:
        positions = jnp.array(f["positions"][:])
        types = jnp.array(f["types"][:])
        energies = jnp.array(f["energies"][:])
        boxes = jnp.array(f["boxes"][:]) if "boxes" in f else None
        dead_energies_data = jnp.array(f["dead_energies"][:])
        dead_positions_data = jnp.array(f["dead_positions"][:])
        log_evidence = jnp.array(f.attrs["log_evidence"])
        iteration = int(f.attrs["iteration"])
        n_dead = int(f.attrs["n_dead"])
        n_walkers = int(f.attrs["n_walkers"])

    # Pad dead arrays to reasonable max_dead
    max_dead = max(n_dead * 2, 50000)
    dead_energies = jnp.full(max_dead, jnp.inf)
    dead_energies = dead_energies.at[:n_dead].set(dead_energies_data)
    dead_positions = jnp.zeros((max_dead, *positions.shape[1:]))
    dead_positions = dead_positions.at[:n_dead].set(dead_positions_data)

    with h5py.File(path, "r") as f:
        if "dead_volumes" in f:
            dead_volumes_data = jnp.array(f["dead_volumes"][:])
            dead_volumes = jnp.zeros(max_dead)
            dead_volumes = dead_volumes.at[:n_dead].set(dead_volumes_data)
        else:
            dead_volumes = None
        live_volumes = jnp.array(f["live_volumes"][:]) if "live_volumes" in f else None

    logger.info("Checkpoint loaded from %s (iteration %d)", path, iteration)

    return {
        "positions": positions,
        "types": types,
        "energies": energies,
        "boxes": boxes,
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
