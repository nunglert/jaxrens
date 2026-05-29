"""Format conversion utilities: WalkerState <-> ASE Atoms <-> HDF5."""

from __future__ import annotations

from typing import Any

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op
import jax.numpy as jnp
import numpy as np

from jaxrens.state.walker import WalkerState


def walker_to_ase_atoms(
    walker: WalkerState | dict,
    symbol_map: dict[int, str],
) -> Any:
    """Convert a WalkerState (or dict with walker fields) to ASE Atoms.

    Args:
        walker: WalkerState or dict with positions, types, energy, box keys.
        symbol_map: Mapping from integer type codes to element symbols.

    Returns:
        ase.Atoms object.
    """
    from ase import Atoms

    if isinstance(walker, dict):
        positions = np.asarray(walker["positions"])
        types = np.asarray(walker["types"])
        energy = float(walker["energy"])
        box = walker.get("box")
    else:
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


def walker_to_h5_group(group: Any, walker: WalkerState | dict) -> None:
    """Write WalkerState arrays into an HDF5 group.

    Args:
        group: h5py.Group to write into.
        walker: WalkerState or dict.
    """
    if isinstance(walker, dict):
        group.create_dataset("positions", data=np.asarray(walker["positions"]))
        group.create_dataset("types", data=np.asarray(walker["types"]))
        group.create_dataset("energy", data=float(walker["energy"]))
        if walker.get("box") is not None:
            group.create_dataset("box", data=np.asarray(walker["box"]))
    else:
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
