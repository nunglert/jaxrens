"""NSState: full state of a nested sampling run.

Carries the live walker population (as a batched MCState), dead-point
accumulator, evidence estimate, and RNG state. Registered as a JAX pytree
for compatibility with JIT/vmap.

NSState is ensemble-agnostic — it doesn't know about pressure, chemical
potentials, or ensemble type. The full ensemble potential is stored in
MCState.energy (via EnsembleBackend), and ns_step reads it directly.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.state.walker import static_field


def _ns_flatten(state: NSState) -> tuple[list, dict]:
    leaves = []
    aux = {}
    for f in dataclasses.fields(state):
        val = getattr(state, f.name)
        if f.metadata.get("static", False):
            aux[f.name] = val
        else:
            leaves.append(val)
    return leaves, aux


def _ns_unflatten(aux: dict, leaves: list) -> NSState:
    fields = dataclasses.fields(NSState)
    dynamic_fields = [f for f in fields if not f.metadata.get("static", False)]
    kwargs = dict(aux)
    for f, val in zip(dynamic_fields, leaves):
        kwargs[f.name] = val
    return NSState(**kwargs)


@dataclasses.dataclass
class NSState:
    """Full state of a nested sampling run.

    Ensemble-agnostic: MCState.energy is the full potential (U, H, Ω, ...),
    computed by the backend (possibly wrapped in EnsembleBackend).

    Attributes:
        population: Batched MCState with shape (n_walkers, ...) on every field.
        dead_energies: Collected dead-point potentials, shape (max_dead,).
        dead_positions: Collected dead-point positions, shape (max_dead, n_atoms, 3).
        dead_volumes: Collected dead-point volumes, shape (max_dead,).
        log_evidence: Running log-evidence estimate (scalar).
        iteration: Current iteration count (scalar).
        n_dead: Number of dead points collected so far (scalar).
        rng_key: JAX PRNG key.
        n_walkers: Number of live walkers (compile-time constant).
        n_atoms: Number of atoms (compile-time constant).
        max_dead: Maximum dead points to store (compile-time constant).
    """

    # Walker population — batched MCState, (n_walkers, ...) on every field
    population: Any  # MCState instance (dynamic pytree)

    # Dead-point accumulator
    dead_energies: jnp.ndarray
    dead_positions: jnp.ndarray
    dead_volumes: jnp.ndarray  # zeros if no cell / NVT

    # NS bookkeeping
    log_evidence: jnp.ndarray
    iteration: jnp.ndarray
    n_dead: jnp.ndarray
    rng_key: jax.Array

    # Compile-time constants
    n_walkers: int = static_field(default=0)
    n_atoms: int = static_field(default=0)
    max_dead: int = static_field(default=0)

    def set(self, **kwargs: Any) -> NSState:
        return dataclasses.replace(self, **kwargs)


jax.tree_util.register_pytree_node(
    NSState,
    _ns_flatten,
    _ns_unflatten,
)
