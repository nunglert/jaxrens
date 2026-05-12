"""WalkerState: JAX-registered dataclass for a single walker in configuration space.

Follows JAX-MD's dataclass pattern: pytree-registered with .set() for functional
updates and static_field() for compile-time constants.

A single WalkerState represents one walker. In the NS loop, walkers are batched
as arrays with shape (G, P, K, ...) where G=n_gpu_parallel, P=n_runs_per_gpu,
K=n_walkers. The pytree registration ensures vmap/pmap work transparently.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
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


def _walker_flatten(walker: WalkerState) -> tuple[list, dict]:
    """Flatten WalkerState into pytree leaves + aux data."""
    leaves = []
    aux = {}
    for f in dataclasses.fields(walker):
        val = getattr(walker, f.name)
        if f.metadata.get("static", False):
            aux[f.name] = val
        else:
            leaves.append(val)
    return leaves, aux


def _walker_unflatten(aux: dict, leaves: Sequence[Any]) -> WalkerState:
    """Reconstruct WalkerState from pytree leaves + aux data.

    JAX passes ``leaves`` as a tuple, but historical signature said ``list``;
    use ``Sequence`` so beartype's runtime check accepts both.
    """
    fields = dataclasses.fields(WalkerState)
    dynamic_fields = [f for f in fields if not f.metadata.get("static", False)]
    kwargs = dict(aux)
    for f, val in zip(dynamic_fields, leaves):
        kwargs[f.name] = val
    return WalkerState(**kwargs)


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


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    WalkerState,
    _walker_flatten,
    _walker_unflatten,
)
