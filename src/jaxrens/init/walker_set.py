"""Load a pre-computed set of N walker configurations from disk."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

logger = logging.getLogger(__name__)

_EXTXYZ_EXTENSIONS = {".extxyz", ".xyz"}
_HDF5_EXTENSIONS = {".h5", ".hdf5"}


@dataclass(frozen=True)
class WalkerSet:
    """A curated set of N walker configurations loaded from disk.

    Attributes:
        positions: (n_live, n_atoms, 3) float32 Cartesian coordinates (Å).
        types: (n_live, n_atoms) int32 contiguous 0-based type indices.
        cells: (n_live, 3, 3) float32 lattice vectors (Å).
        symbol_map: {type_index -> element_symbol} in first-appearance order.
    """

    positions: jnp.ndarray
    types: jnp.ndarray
    cells: jnp.ndarray
    symbol_map: dict[int, str]


def load_walker_set(
    path: Path,
    n_live_expected: int,
) -> WalkerSet:
    """Load a curated set of walker configurations from disk.

    Dispatches by file extension:
      - .extxyz, .xyz: ASE multi-frame read.
      - .h5, .hdf5: HDF5 read (matches io/checkpoint.py live-walker schema).

    Args:
        path: Path to the walker-set file.
        n_live_expected: Expected number of live walkers (frames/rows).

    Returns:
        WalkerSet with positions (n_live, n_atoms, 3), types (n_live, n_atoms),
        cells (n_live, 3, 3), symbol_map dict. Energies are NOT populated here —
        caller is expected to recompute with the current backend.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: On shape / count mismatches or unsupported extensions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"walker-set file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in _EXTXYZ_EXTENSIONS:
        return _load_walker_set_extxyz(path, n_live_expected)
    elif suffix in _HDF5_EXTENSIONS:
        return _load_walker_set_hdf5(path, n_live_expected)
    else:
        supported = sorted(_EXTXYZ_EXTENSIONS | _HDF5_EXTENSIONS)
        raise ValueError(
            f"Unsupported walker-set file extension {suffix!r} for {path}. "
            f"Supported extensions: {supported}"
        )


def _load_walker_set_extxyz(path: Path, n_live_expected: int) -> WalkerSet:
    """Load walker set from an ASE multi-frame extxyz/xyz file."""
    import ase.io
    import numpy as np

    frames = ase.io.read(str(path), index=":")

    if len(frames) != n_live_expected:
        raise ValueError(
            f"walker-set file {path} has {len(frames)} frames, "
            f"expected n_live={n_live_expected}"
        )

    n_atoms_0 = len(frames[0])
    symbols_0 = frames[0].get_chemical_symbols()

    for fi, atoms in enumerate(frames):
        if len(atoms) != n_atoms_0:
            raise ValueError(
                f"walker-set frames have inconsistent atom counts: "
                f"frame 0 has {n_atoms_0} atoms, frame {fi} has {len(atoms)}"
            )
        cell_np = np.array(atoms.get_cell()[:], dtype=np.float32)
        if not all(np.linalg.norm(cell_np[i]) > 0 for i in range(3)):
            raise ValueError(
                f"walker-set frame {fi} has a zero or missing simulation cell — "
                f"jaxrens requires a periodic cell"
            )
        if atoms.get_chemical_symbols() != symbols_0:
            raise ValueError(
                f"walker-set frames have divergent compositions: "
                f"frame 0 has {symbols_0}, frame {fi} has {atoms.get_chemical_symbols()}"
            )

    from jaxrens.init.structure import _build_symbol_map_from_symbols

    symbol_map, type_indices_0 = _build_symbol_map_from_symbols(symbols_0)

    positions_list = []
    types_list = []
    cells_list = []

    for atoms in frames:
        sym_to_idx = {sym: i for i, sym in symbol_map.items()}
        type_indices = [sym_to_idx[s] for s in atoms.get_chemical_symbols()]
        positions_list.append(np.array(atoms.get_positions(), dtype=np.float32))
        types_list.append(np.array(type_indices, dtype=np.int32))
        cells_list.append(np.array(atoms.get_cell()[:], dtype=np.float32))

    positions = jnp.array(np.stack(positions_list, axis=0), dtype=jnp.float32)
    types = jnp.array(np.stack(types_list, axis=0), dtype=jnp.int32)
    cells = jnp.array(np.stack(cells_list, axis=0), dtype=jnp.float32)

    return WalkerSet(
        positions=positions,
        types=types,
        cells=cells,
        symbol_map=symbol_map,
    )


def _load_walker_set_hdf5(path: Path, n_live_expected: int) -> WalkerSet:
    """Load walker set from an HDF5 file matching io/checkpoint.py live-walker schema."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        for key in ("positions", "types", "cells"):
            if key not in f:
                raise ValueError(
                    f"walker-set HDF5 file {path} is missing required dataset {key!r}"
                )

        positions_np = f["positions"][:]
        types_np = f["types"][:]
        cells_np = f["cells"][:]

        if positions_np.shape[0] != n_live_expected:
            raise ValueError(
                f"walker-set HDF5 file {path}: positions has shape {positions_np.shape}, "
                f"expected first dimension n_live={n_live_expected}"
            )
        if types_np.shape[0] != n_live_expected:
            raise ValueError(
                f"walker-set HDF5 file {path}: types has shape {types_np.shape}, "
                f"expected first dimension n_live={n_live_expected}"
            )
        if cells_np.shape[0] != n_live_expected:
            raise ValueError(
                f"walker-set HDF5 file {path}: cells has shape {cells_np.shape}, "
                f"expected first dimension n_live={n_live_expected}"
            )

        if "symbol_map" in f.attrs:
            raw = json.loads(f.attrs["symbol_map"])
            symbol_map: dict[int, str] = {int(k): v for k, v in raw.items()}
        else:
            logger.warning(
                "walker-set HDF5 file %s has no symbol_map attribute; "
                "element labels will be integer-coded from unique type indices.",
                path,
            )
            unique_types = sorted(set(types_np.flatten().tolist()))
            symbol_map = {i: str(t) for i, t in enumerate(unique_types)}

    positions = jnp.array(positions_np, dtype=jnp.float32)
    types = jnp.array(types_np, dtype=jnp.int32)
    cells = jnp.array(cells_np, dtype=jnp.float32)

    return WalkerSet(
        positions=positions,
        types=types,
        cells=cells,
        symbol_map=symbol_map,
    )
