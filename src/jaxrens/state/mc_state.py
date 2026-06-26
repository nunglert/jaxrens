"""Dynamic MCState factory for Monte Carlo moves.

MCState is built at runtime by the MWG factory based on the active move
set. Core fields (positions, types, energy, cell, step_size, step_sizes,
n_accepted, n_proposed) are always present. Move-specific fields (e.g.
direction for Galilean) are added only when the corresponding move is
in the MWG descriptor list.

The factory uses dataclasses.make_dataclass() and registers each unique
class as a JAX pytree. Classes are cached by field names to avoid
duplicate registration.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from jaxrens.state.walker import register_dataclass_pytree, static_field

# ---------------------------------------------------------------------------
# MCState-specific helper
# ---------------------------------------------------------------------------


def _add_set_method(cls):
    """Add a .set(**kwargs) method for functional updates.

    WalkerState/NSState define ``.set`` in their class body; the dynamically
    built MCState gets it injected here instead.
    """

    def set_method(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    cls.set = set_method


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
        ("cell", jnp.ndarray),
        ("step_size", jnp.ndarray),
        ("step_sizes", jnp.ndarray),
        ("n_accepted", jnp.ndarray),
        ("n_proposed", jnp.ndarray),
        ("max_neighbor_count", jnp.ndarray),  # actual max neighbors observed
        ("overflow", jnp.ndarray),  # bool — any overflow detected
        (
            "ensemble_params",
            dict,
        ),  # e.g. {"pressure": scalar, "chemical_potentials": (n_species,)}
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
    register_dataclass_pytree(cls)

    _MC_STATE_CACHE[cache_key] = cls
    return cls


# ---------------------------------------------------------------------------
# Convenience: default MCState with no extra fields
# ---------------------------------------------------------------------------

MCState = make_mc_state_class()
