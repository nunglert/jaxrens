"""EnergyBackend protocol — unified interface for all energy backends.

Every backend is a callable that computes potential energy from an atomic
configuration and returns a :class:`BackendResult` pytree::

    backend(positions, species, cell, max_neighbors) -> BackendResult

``max_neighbors`` is a static parameter passed per call (not stored on the
instance). JAX's compilation cache handles retrace when it changes.

Backends may *optionally* implement ``energy_and_forces`` (same signature) to
return forces natively; callers should go through :func:`eval_energy_and_forces`,
which dispatches to the native method when present and otherwise falls back to
reverse-mode autodiff of ``__call__``.

Design doc: experiments/jaxrens_design/energy_backend_design.md
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class BackendResult(NamedTuple):
    """Structured return value of every :class:`EnergyBackend` call.

    Registered automatically as a JAX pytree (``NamedTuple``), so it threads
    through ``jax.jit`` / ``vmap`` / ``scan`` unchanged. ``energy`` is the only
    universally-meaningful field; the rest are backend-specific.

    A given backend must return the **same field set on every call** (None-vs-
    array is part of the pytree treedef and a changing structure would break
    ``lax.scan`` / ``lax.cond``). Backends that don't produce a control field
    leave it at its sentinel default (``0`` / ``False``); fields they never
    produce stay ``None``.

    Fields:
        energy: Total potential energy. Shape ``*B`` (scalar per walker).
        forces: ``-dE/dx`` atomic forces, shape ``*B N 3``. ``None`` on the
            energy-only ``__call__`` path; populated by ``energy_and_forces``.
            NOT required to be the exact gradient of ``energy`` — backends may
            return independent / non-conservative forces.
        max_neighbor_count: Largest neighbor count any atom saw this call
            (control signal for the bucket manager). Sentinel ``0`` for
            backends without a neighbor list (LJ, toy, all-pairs jax-md).
        overflow: True if ``max_neighbor_count`` exceeded the requested
            ``max_neighbors`` buffer. Sentinel ``False`` for neighbor-free
            backends.
        energy_members: Reserved — per-ensemble-member energies, shape
            ``*B n_ens``, for active-learning uncertainty. Unpopulated for now.
        forces_members: Reserved — per-ensemble-member forces, shape
            ``*B n_ens N 3``. Unpopulated for now.
    """

    energy: Float[Array, "*B"]
    forces: Float[Array, "*B N 3"] | None = None
    # --- control plane (consumed by the neighbor-bucket manager) ---
    max_neighbor_count: Int[Array, "*B"] = 0
    overflow: Bool[Array, "*B"] = False
    # --- diagnostics plane (reserved for AL uncertainty; unpopulated) ---
    energy_members: Float[Array, "*B n_ens"] | None = None
    forces_members: Float[Array, "*B n_ens N 3"] | None = None

    def legacy(self) -> tuple[Float[Array, "*B"], Int[Array, "*B"], Bool[Array, "*B"]]:
        """Return the historical ``(energy, max_neighbor_count, overflow)`` tuple.

        Transitional shim so consumers can be migrated one at a time while
        producers already return :class:`BackendResult`. Removed once every
        call site reads the named fields directly.
        """
        return self.energy, self.max_neighbor_count, self.overflow


def eval_energy_and_forces(
    backend: "EnergyBackend",
    positions: jnp.ndarray,
    species: jnp.ndarray,
    cell: jnp.ndarray,
    max_neighbors: int,
    ensemble_params: dict[str, Any] | None = None,
) -> BackendResult:
    """Evaluate energy and forces, using a native force path when available.

    If ``backend`` defines ``energy_and_forces``, it is called directly.
    Otherwise forces are obtained by reverse-mode autodiff of ``backend``'s
    energy (``forces = -dE/dx``), preserving the control fields via ``has_aux``.

    Returns a :class:`BackendResult` with ``forces`` populated.
    """
    native = getattr(backend, "energy_and_forces", None)
    if native is not None:
        return native(
            positions, species, cell, max_neighbors,
            ensemble_params=ensemble_params,
        )

    def energy_of(pos: jnp.ndarray) -> tuple[jnp.ndarray, BackendResult]:
        res = backend(
            pos, species, cell, max_neighbors,
            ensemble_params=ensemble_params,
        )
        return res.energy, res

    (energy, res), grad = jax.value_and_grad(energy_of, has_aux=True)(positions)
    return res._replace(energy=energy, forces=-grad)


@runtime_checkable
class EnergyBackend(Protocol):
    """Protocol that all energy backends must satisfy.

    Backends are callable objects that compute potential energy from an
    atomic configuration. Model weights live on the instance — no ``params``
    argument in the call.

    ``max_neighbors`` controls compiled array shapes:
    - NeuralIL: lexsort truncation in descriptor computation
    - GNN backends: edge buffer size ``max_edges = N * max_neighbors``
    - Toy/LJ: ignored (no neighbors)

    Implementations must be compatible with jax.jit and jax.vmap.

    Optional capability (not part of the required protocol): a backend may
    define ``energy_and_forces(positions, species, cell, max_neighbors,
    ensemble_params=None) -> BackendResult`` with ``forces`` populated. Callers
    should use :func:`eval_energy_and_forces` rather than probing for it.
    """

    r_cutoff: float

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Compute total potential energy.

        Args:
            positions: Atomic positions, shape (n_atoms, 3).
            species: Integer atom type codes, shape (n_atoms,).
            cell: Unit cell matrix (3, 3). Zeros for non-periodic.
            max_neighbors: Static parameter controlling compiled shapes.
            ensemble_params: Optional per-run ensemble parameters
                (used by EnsembleBackend for per-run vmap).

        Returns:
            A :class:`BackendResult` with ``energy`` (and the control fields
            ``max_neighbor_count`` / ``overflow``) populated. ``forces`` is
            ``None`` on this path — use :func:`eval_energy_and_forces` for forces.
        """
        ...
