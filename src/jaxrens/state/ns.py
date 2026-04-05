"""NSState: full state of a nested sampling run.

Carries the live walker population, dead-point accumulator, evidence estimate,
and RNG state. Registered as a JAX pytree for compatibility with JIT/vmap.
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

    The walker arrays have shape (G, P, K, ...) in the multi-GPU case,
    where G=n_gpu_parallel, P=n_runs_per_gpu, K=n_walkers.

    Attributes:
        positions: Walker positions, shape (..., n_atoms, 3).
        types: Walker atom types, shape (..., n_atoms).
        energies: Walker energies, shape (...,).
        boxes: Walker unit cells, shape (..., 3, 3) or None.
        dead_energies: Collected dead-point energies, shape (max_dead,).
        log_evidence: Running log-evidence estimate (scalar).
        iteration: Current iteration count (scalar).
        n_dead: Number of dead points collected so far (scalar).
        rng_key: JAX PRNG key.
        move_state: Opaque move-specific state (e.g., step sizes, velocities).
        n_live: Number of live walkers per run (compile-time constant).
        n_atoms: Number of atoms (compile-time constant).
    """

    # Walker population (dynamic, batched)
    positions: jnp.ndarray
    types: jnp.ndarray
    energies: jnp.ndarray
    boxes: jnp.ndarray | None = None

    # Dead-point accumulator
    dead_energies: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array([])
    )
    log_evidence: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array(-jnp.inf)
    )

    # Loop counters
    iteration: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32)
    )
    n_dead: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32)
    )

    # RNG
    rng_key: jax.Array = dataclasses.field(
        default_factory=lambda: jax.random.key(0)
    )

    # Move state (opaque pytree)
    move_state: Any = None

    # Compile-time constants
    n_live: int = static_field(default=0)
    n_atoms: int = static_field(default=0)

    def set(self, **kwargs: Any) -> NSState:
        return dataclasses.replace(self, **kwargs)


jax.tree_util.register_pytree_node(
    NSState,
    _ns_flatten,
    _ns_unflatten,
)
