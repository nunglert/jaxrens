"""Load a founder structure from disk via ASE for walker initialization."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp


def _build_symbol_map_from_symbols(
    symbols: list[str],
) -> tuple[dict[int, str], list[int]]:
    """Build a symbol_map and type-index list from a flat list of element symbols.

    Args:
        symbols: Ordered list of element symbols (one per atom), e.g.
            ["Si", "Si", "O", "O", "O"].

    Returns:
        (symbol_map, type_indices):
          - symbol_map: {type_index -> element_symbol} in first-appearance order.
          - type_indices: per-atom list of 0-based type indices.
    """
    seen: list[str] = []
    for sym in symbols:
        if sym not in seen:
            seen.append(sym)
    symbol_map: dict[int, str] = {i: sym for i, sym in enumerate(seen)}
    sym_to_idx = {sym: i for i, sym in symbol_map.items()}
    type_indices = [sym_to_idx[s] for s in symbols]
    return symbol_map, type_indices


def load_structure(
    path: Path | str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[int, str]]:
    """Read a single founder structure from disk via ASE.

    Args:
        path: Path to a structure file (any ASE-readable format: extxyz, cif,
            POSCAR, etc.).

    Returns:
        (positions, types, cell, symbol_map):
          - positions: (n_atoms, 3) float32 array in Cartesian coords (Å).
          - types: (n_atoms,) int32 array of contiguous 0-based type indices.
          - cell: (3, 3) float32 array (Å), rows are lattice vectors.
          - symbol_map: {type_index -> element_symbol}, e.g. {0: "Si", 1: "O"}.
            Ordering reflects first appearance in the atoms list.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file contains multiple frames, has no simulation
            cell, or contains no atoms.
    """
    import ase.io

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"structure file not found: {path}")

    all_frames = ase.io.read(str(path), index=":")

    if len(all_frames) != 1:
        raise ValueError(
            f"load_structure expects a single-frame file; "
            f"got {len(all_frames)} frames at {path}"
        )

    atoms = all_frames[0]

    if len(atoms) == 0:
        raise ValueError(f"structure at {path} contains no atoms")

    import numpy as np

    cell_np = np.array(atoms.get_cell()[:], dtype=np.float32)
    # Each lattice vector must be non-zero (all three norms > 0).
    if not all(np.linalg.norm(cell_np[i]) > 0 for i in range(3)):
        raise ValueError(
            f"structure at {path} has no simulation cell — "
            f"jaxrens requires a periodic cell"
        )

    symbols = atoms.get_chemical_symbols()
    symbol_map, type_indices = _build_symbol_map_from_symbols(symbols)

    positions = jnp.array(atoms.get_positions(), dtype=jnp.float32)
    types = jnp.array(type_indices, dtype=jnp.int32)
    cell = jnp.array(cell_np, dtype=jnp.float32)

    return positions, types, cell, symbol_map
