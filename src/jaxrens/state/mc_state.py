"""Dynamic MCState factory for Monte Carlo moves.

MCState is built at runtime by the MWG factory based on the active move
set. Core fields (positions, types, energy, box, step_size, step_sizes,
n_accepted, n_proposed) are always present. Move-specific fields (e.g.
direction for Galilean) are added only when the corresponding move is
in the MWG descriptor list.

The factory uses dataclasses.make_dataclass() and registers each unique
class as a JAX pytree. Classes are cached by field names to avoid
duplicate registration.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.state.walker import static_field


# ---------------------------------------------------------------------------
# Generic pytree helpers (work on any dataclass with static_field metadata)
# ---------------------------------------------------------------------------


def _generic_flatten(obj):
    """Flatten a dataclass into pytree leaves + aux data."""
    leaves = []
    aux = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if f.metadata.get("static", False):
            aux[f.name] = val
        else:
            leaves.append(val)
    return leaves, aux


def _make_unflatten(cls):
    """Create an unflatten function for a specific dataclass type."""
    def unflatten(aux, leaves):
        fields = dataclasses.fields(cls)
        dynamic_fields = [f for f in fields if not f.metadata.get("static", False)]
        kwargs = dict(aux)
        for f, val in zip(dynamic_fields, leaves):
            kwargs[f.name] = val
        return cls(**kwargs)
    return unflatten


def _add_set_method(cls):
    """Add a .set(**kwargs) method for functional updates."""
    def set_method(self, **kwargs):
        return dataclasses.replace(self, **kwargs)
    cls.set = set_method


def _register_pytree(cls):
    """Register a dataclass as a JAX pytree."""
    jax.tree_util.register_pytree_node(
        cls,
        _generic_flatten,
        _make_unflatten(cls),
    )


# ---------------------------------------------------------------------------
# MCState factory
# ---------------------------------------------------------------------------

_MC_STATE_CACHE: dict[frozenset, type] = {}


def make_mc_state_class(extra_fields: dict[str, type] | None = None) -> type:
    """Create an MCState dataclass with core + move-specific fields.

    Args:
        extra_fields: Dict mapping field names to types for move-specific
            fields. E.g. {"direction": jnp.ndarray}. If None or empty,
            only core fields are included.

    Returns:
        A dataclass type registered as a JAX pytree with a .set() method.
        Cached by field names — same fields return the same class.
    """
    if extra_fields is None:
        extra_fields = {}

    cache_key = frozenset(extra_fields.keys())
    if cache_key in _MC_STATE_CACHE:
        return _MC_STATE_CACHE[cache_key]

    # Core fields — always present
    core = [
        ("positions", jnp.ndarray),
        ("types", jnp.ndarray),
        ("energy", jnp.ndarray),
        ("box", jnp.ndarray),
        ("step_size", jnp.ndarray),
        ("step_sizes", jnp.ndarray),
        ("n_accepted", jnp.ndarray),
        ("n_proposed", jnp.ndarray),
    ]

    # Move-specific fields
    extra = [(name, typ) for name, typ in sorted(extra_fields.items())]

    # Static fields (compile-time constants)
    static = [
        ("max_neighbors", int, static_field(default=0)),
        ("n_atoms", int, static_field(default=0)),
    ]

    cls = dataclasses.make_dataclass("MCState", core + extra + static)
    _add_set_method(cls)
    _register_pytree(cls)

    _MC_STATE_CACHE[cache_key] = cls
    return cls


# ---------------------------------------------------------------------------
# Convenience: default MCState with no extra fields
# ---------------------------------------------------------------------------

MCState = make_mc_state_class()
