"""Format conversion utilities: WalkerState <-> ASE Atoms <-> HDF5.

Serialization contract
----------------------
The in-memory single-walker type is :class:`~jaxrens.state.walker.WalkerState`.
On disk / at the callback boundary a walker is a *loose dict* whose unit-cell
key is ``box`` (extxyz/HDF5 convention), and a *batched population* is a dict
with plural keys ``positions``/``types``/``energies``/``cells`` built by
:func:`population_record`.  This module is the only seam that converts between
the two: :func:`ensure_walker_state` / :func:`WalkerState.from_record` on the
way in, :func:`iter_walker_states` to expand a batched record, and the
``walker_to_*`` writers on the way out.  Everything else works with typed
``WalkerState`` objects and attribute access.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op
from jaxrens.state.walker import WalkerState


def ensure_walker_state(
    walker: WalkerState | Mapping[str, Any]
) -> WalkerState:
    """Normalize *walker* to a ``WalkerState``, converting a loose dict once.

    Idempotent for a ``WalkerState``; a dict is routed through
    :meth:`WalkerState.from_record` (which accepts ``cell`` or ``box``).  Lets
    every downstream consumer use attribute access on a single code path.
    """
    if isinstance(walker, WalkerState):
        return walker
    return WalkerState.from_record(walker)


def population_record(
    positions: Any,
    types: Any,
    energies: Any,
    cells: Any,
    **extra: Any,
) -> dict[str, Any]:
    """Build the canonical batched-population serialization dict.

    Single source of truth for the plural on-disk key names (``energies``,
    ``cells``) shared by the checkpoint writer and the snapshot writers.  Extra
    bookkeeping fields (``step_sizes``, ``log_evidence``, ``iteration`` …) are
    merged verbatim.  Consumed by :func:`iter_walker_states` and
    ``io.checkpoint.save_checkpoint``.
    """
    record = {
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": cells,
    }
    record.update(extra)
    return record


def iter_walker_states(record: Mapping[str, Any]) -> Iterator[WalkerState]:
    """Yield one ``WalkerState`` per walker from a batched population record.

    The batched record is the plural-keyed dict produced by
    :func:`population_record`.  This is the single reader that turns it back
    into the typed per-walker form the writers consume, so the writers no
    longer rebuild ad-hoc ``{"positions": …, "box": …}`` dicts per walker.
    Arrays are coerced to JAX arrays to satisfy ``WalkerState``'s field types.
    """
    positions = jnp.asarray(record["positions"])
    types = jnp.asarray(record["types"])
    energies = jnp.asarray(record["energies"])
    cells = record.get("cells")
    cells = jnp.asarray(cells) if cells is not None else None
    n_atoms = int(positions.shape[1])
    for i in range(positions.shape[0]):
        yield WalkerState(
            positions=positions[i],
            types=types[i] if types.ndim >= 2 else types,
            energy=energies[i],
            cell=None if cells is None else cells[i],
            n_atoms=n_atoms,
        )


def walker_to_ase_atoms(
    walker: WalkerState | Mapping[str, Any],
    symbol_map: dict[int, str],
) -> Any:
    """Convert a WalkerState (or loose walker dict) to ASE Atoms.

    Args:
        walker: WalkerState, or a dict with positions, types, energy, box keys
            (normalized via :func:`ensure_walker_state`).
        symbol_map: Mapping from integer type codes to element symbols.

    Returns:
        ase.Atoms object.
    """
    from ase import Atoms

    walker = ensure_walker_state(walker)
    positions = np.asarray(walker.positions)
    types = np.asarray(walker.types)
    energy = float(walker.energy)
    box = walker.cell

    symbols = [symbol_map[int(t)] for t in types]
    atoms = Atoms(symbols=symbols, positions=positions)

    if box is not None:
        atoms.set_cell(np.asarray(box))
        atoms.set_pbc(True)

    atoms.info["ns_energy"] = energy
    return atoms


def ase_atoms_to_walker(
    atoms: Any,
    symbol_map: dict[str, int],
) -> WalkerState:
    """Convert ASE Atoms to WalkerState.

    Args:
        atoms: ase.Atoms object.
        symbol_map: Mapping from element symbols to integer type codes.

    Returns:
        WalkerState.
    """
    positions = jnp.array(atoms.get_positions())
    types = jnp.array([symbol_map[s] for s in atoms.get_chemical_symbols()])
    energy = jnp.array(atoms.info.get("ns_energy", 0.0))
    cell = jnp.array(atoms.get_cell()[:]) if any(atoms.get_pbc()) else None

    return WalkerState(
        positions=positions,
        types=types,
        energy=energy,
        cell=cell,
        n_atoms=len(atoms),
    )


def walker_to_h5_group(
    group: Any, walker: WalkerState | Mapping[str, Any]
) -> None:
    """Write WalkerState arrays into an HDF5 group.

    Args:
        group: h5py.Group to write into.
        walker: WalkerState or loose walker dict (normalized via
            :func:`ensure_walker_state`).  The cell is stored under ``box``.
    """
    walker = ensure_walker_state(walker)
    group.create_dataset("positions", data=np.asarray(walker.positions))
    group.create_dataset("types", data=np.asarray(walker.types))
    group.create_dataset("energy", data=float(walker.energy))
    if walker.cell is not None:
        group.create_dataset("box", data=np.asarray(walker.cell))


def h5_group_to_walker(group: Any) -> WalkerState:
    """Read WalkerState from an HDF5 group.

    Args:
        group: h5py.Group to read from.

    Returns:
        WalkerState.
    """
    positions = jnp.array(group["positions"][:])
    types = jnp.array(group["types"][:])
    energy = jnp.array(float(group["energy"][()]))
    box = jnp.array(group["box"][:]) if "box" in group else None

    return WalkerState(
        positions=positions,
        types=types,
        energy=energy,
        cell=box,
        n_atoms=positions.shape[0],
    )
