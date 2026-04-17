"""Load a full NS checkpoint into (live walkers, NS-state bundle).

Used by Mode D: restart_file — crash-recovery restart of a single NS run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

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


def load_restart(path: Path) -> tuple[WalkerSet, RestartBundle]:
    """Load a checkpoint HDF5 file into (live walkers, NS-state bundle).

    Reuses io/checkpoint.load_checkpoint. Returns a WalkerSet for the live
    walker side and a RestartBundle for the dead-point history and NS-state
    scalars.

    The dead arrays in RestartBundle are the compact slices (length n_dead),
    not padded. init_ns pads them into pre-allocated arrays of size max_dead.

    Args:
        path: Path to the checkpoint HDF5 file.

    Returns:
        (WalkerSet, RestartBundle) pair.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the file is missing required NS-state fields (e.g. a
            bare walker-set file was passed instead of a checkpoint).
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
    missing_attrs = {"log_evidence", "iteration", "n_dead"} - present_attrs
    missing = missing_datasets | missing_attrs
    if missing:
        raise ValueError(
            f"file at {path} is not a valid NS checkpoint; missing fields: "
            f"{sorted(missing)}. "
            f"If this is a bare walker-set file, use start_walker_set instead."
        )

    from jaxrens.io.checkpoint import load_checkpoint

    ckpt = load_checkpoint(path)

    n_dead: int = int(ckpt["n_dead"])
    dead_energies: jnp.ndarray = ckpt["dead_energies"][:n_dead]
    dead_positions: jnp.ndarray = ckpt["dead_positions"][:n_dead]

    dead_volumes: jnp.ndarray | None = None
    if ckpt.get("dead_volumes") is not None:
        dead_volumes = ckpt["dead_volumes"][:n_dead]

    bundle = RestartBundle(
        dead_energies=dead_energies,
        dead_positions=dead_positions,
        dead_volumes=dead_volumes,
        log_evidence=float(ckpt["log_evidence"]),
        iteration=int(ckpt["iteration"]),
        n_dead=n_dead,
    )

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
        path, n_dead, bundle.iteration, bundle.log_evidence,
    )
    return walker_set, bundle
