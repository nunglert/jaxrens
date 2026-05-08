"""EnergyBackend protocol — unified interface for all energy backends.

Every backend satisfies this protocol:
    backend(positions, species, cell, max_neighbors) -> (energy, count, overflow)

max_neighbors is a static parameter passed per call (not stored on the
instance). JAX's compilation cache handles retrace when it changes.

Design doc: experiments/jaxrens_design/energy_backend_design.md
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class EnergyBackend(Protocol):
    """Protocol that all energy backends must satisfy.

    Backends are callable objects that compute potential energy from
    atomic configuration. Model weights live on the instance — no
    ``params`` argument in the call.

    ``max_neighbors`` controls compiled array shapes:
    - NeuralIL: lexsort truncation in descriptor computation
    - GNN backends: edge buffer size ``max_edges = N * max_neighbors``
    - Toy/LJ: ignored (no neighbors)

    Implementations must be compatible with jax.jit and jax.vmap.
    """

    r_cutoff: float

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        """Compute total potential energy.

        Args:
            positions: Atomic positions, shape (n_atoms, 3).
            species: Integer atom type codes, shape (n_atoms,).
            cell: Unit cell matrix (3, 3). Zeros for non-periodic.
            max_neighbors: Static parameter controlling compiled shapes.
            ensemble_params: Optional per-run ensemble parameters
                (used by EnsembleBackend for per-run vmap).

        Returns:
            (energy, actual_max_neighbor_count, overflow) where:
            - energy: scalar potential energy
            - actual_max_neighbor_count: max neighbors any atom has
            - overflow: True if actual > max_neighbors
        """
        ...
