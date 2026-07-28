"""WalkerState: JAX-registered dataclass for a single walker in configuration space.

Follows JAX-MD's dataclass pattern: pytree-registered with .set() for functional
updates and static_field() for compile-time constants.

A single WalkerState represents one walker. In the NS loop, walkers are batched
as arrays with shape (G, P, K, ...) where G=n_gpu_parallel, P=n_runs_per_gpu,
K=n_walkers. The pytree registration ensures vmap/pmap work transparently.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def static_field(**kwargs: Any) -> Any:
    """Mark a dataclass field as static (compile-time constant).

    Static fields are not part of the pytree leaves -- they are stored in
    the pytree auxiliary data. Changing a static field triggers recompilation.
    """
    return dataclasses.field(metadata={"static": True}, **kwargs)


def flatten_dataclass(obj: Any) -> tuple[list, dict]:
    """Flatten any ``static_field``-aware dataclass into (leaves, aux).

    Dynamic fields become pytree leaves; ``static_field`` fields go into the
    auxiliary data. Shared by every JAX-registered state dataclass
    (``WalkerState``, ``NSState``, and the dynamic ``MCState``).
    """
    leaves = []
    aux = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if f.metadata.get("static", False):
            aux[f.name] = val
        else:
            leaves.append(val)
    return leaves, aux


def make_unflatten(cls: type) -> Any:
    """Build an unflatten function reconstructing ``cls`` from (aux, leaves)."""

    def unflatten(aux: dict, leaves: Sequence[Any]) -> Any:
        dynamic_fields = [
            f
            for f in dataclasses.fields(cls)
            if not f.metadata.get("static", False)
        ]
        kwargs = dict(aux)
        for f, val in zip(dynamic_fields, leaves):
            kwargs[f.name] = val
        return cls(**kwargs)

    return unflatten


def register_dataclass_pytree(cls: type) -> None:
    """Register ``cls`` as a JAX pytree via the generic flatten/unflatten."""
    jax.tree_util.register_pytree_node(
        cls,
        flatten_dataclass,
        make_unflatten(cls),
    )


@dataclasses.dataclass
class WalkerState:
    """State of a single walker in configuration space.

    JAX-registered pytree. Use .set() for functional updates.

    Shape annotations use a variadic batch prefix ``*B`` so the *same*
    annotation covers a single walker (``*B = ()``), the live population
    (``*B = (K,)``), vmapped runs (``*B = (P, K)``), and pmap+vmap
    (``*B = (G, P, K)``).

    Attributes:
        positions: Cartesian coordinates, shape ``(*B, n_atoms, 3)``.
        types: Integer atom type codes, shape ``(*B, n_atoms)``.
        energy: Current potential energy, shape ``(*B,)``.
        cell: Unit cell matrix, shape ``(*B, 3, 3)`` or ``None`` for
            non-periodic systems.
        n_atoms: Number of atoms (compile-time constant).
    """

    positions: Float[Array, "*B N 3"]
    types: Int[Array, "*B N"]
    energy: Float[Array, "*B"]
    cell: Float[Array, "*B 3 3"] | None = None
    n_atoms: int = static_field(default=0)

    def set(self, **kwargs: Any) -> WalkerState:
        """Functional update: return a new WalkerState with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        n_atoms: int | None = None,
    ) -> WalkerState:
        """Build a single-walker ``WalkerState`` from a loose serialization dict.

        The in-memory walker contract *is* this dataclass; on-disk artifacts and
        the callback boundary use plain dicts.  This classmethod is the single
        place that maps those loose keys back to the typed form, so callers stop
        hand-unpacking ``walker["positions"]`` alongside ``walker.positions``.

        The unit cell is read from ``cell`` or its on-disk alias ``box`` (the
        extxyz/HDF5 writers emit ``box``).  Arrays are coerced to JAX arrays so
        the result satisfies the dataclass's jaxtyped field contract regardless
        of whether the record carried numpy or jax arrays.

        Single walker only — for a batched population record use
        :func:`jaxrens.io.formats.iter_walker_states`.
        """
        positions = jnp.asarray(record["positions"])
        cell = record.get("cell")
        if cell is None:
            cell = record.get("box")
        return cls(
            positions=positions,
            types=jnp.asarray(record["types"]),
            energy=jnp.asarray(record["energy"]),
            cell=None if cell is None else jnp.asarray(cell),
            n_atoms=int(positions.shape[0]) if n_atoms is None else n_atoms,
        )


# Register as JAX pytree
register_dataclass_pytree(WalkerState)
