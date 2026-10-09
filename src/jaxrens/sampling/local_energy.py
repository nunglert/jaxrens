"""Shared patch-invariant helper for local ("few atoms changed") moves.

Every ``affects="local"`` move kernel (see :mod:`jaxrens.sampling.move_kernel`)
calls :func:`local_energy_update` instead of a bare ``backend(...)`` call —
one implementation of the delta-patch identity, not one per move.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.backends.hamiltonian import unwrap_model


def local_energy_update(
    backend: Any,
    atomic_energies: jnp.ndarray,
    affected: jnp.ndarray,
    new_positions: jnp.ndarray,
    new_types: jnp.ndarray,
    cell: jnp.ndarray,
    max_neighbors: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the raw-energy delta from patching a per-atom energy cache.

    ``new_U = old_U + (new_local_sum - old_local_sum)``, where both sums run
    over exactly the (deduplicated) ``affected`` index set — the atoms whose
    per-atom energy contribution could have changed as a result of the move.

    Args:
        backend: Must implement ``atomic_energies_for`` (see
            ``backends/neuralil.py``); callers are expected to have already
            checked this via ``jaxrens.backends.locality.probe_local_capability``.
        atomic_energies: ``(n_atoms,)`` cached per-atom energies, from
            *before* the move.
        affected: ``(K,)`` int32 index array (sentinel ``n_atoms`` for
            unused slots), e.g. from
            ``neighbor_list.affected_indices_from_mask``.
        new_positions, new_types, cell: The system's arrays *after* the
            proposed move.
        max_neighbors: Buffer-shape control, forwarded to the backend.

    Returns:
        (delta_U, new_local_vec):
            - ``delta_U``: scalar, ``new_local_sum - old_local_sum``. Add
              this to the cached raw total energy to get the proposed raw
              total energy.
            - ``new_local_vec``: ``(K,)`` per-atom energies for `affected`
              under the new configuration (0.0 at sentinel slots) — the
              caller scatters this into the cache at `affected` on accept:
              ``atomic_energies.at[affected].set(new_local_vec, mode="drop")``
              (``mode="drop"`` makes the sentinel-index writes safe no-ops
              instead of relying on JAX's default out-of-bounds behaviour).
    """
    n_atoms = atomic_energies.shape[0]
    valid = affected < n_atoms
    safe_idx = jnp.where(valid, affected, 0)

    old_local_sum = jnp.sum(jnp.where(valid, atomic_energies[safe_idx], 0.0))

    new_local_vec = backend.atomic_energies_for(
        new_positions, new_types, cell, affected, max_neighbors
    )
    new_local_vec = jnp.where(valid, new_local_vec, 0.0)
    new_local_sum = jnp.sum(new_local_vec)

    delta_U = new_local_sum - old_local_sum
    return delta_U, new_local_vec


def patch_term_energies(
    backend: Any,
    term_energies: jnp.ndarray,
    touched_idx: jnp.ndarray,
    old_positions: jnp.ndarray,
    old_types: jnp.ndarray,
    new_positions: jnp.ndarray,
    new_types: jnp.ndarray,
    cell: jnp.ndarray,
    ensemble_params: dict[str, Any] | None,
) -> jnp.ndarray:
    """Update the cached per-``EnergyTerm`` energies for a local proposal.

    For each term in ``backend.terms`` (empty for a bare, term-less
    backend — returns ``term_energies`` unchanged), uses
    ``term.local_energy_delta(...)`` when the term supports it (e.g.
    ``SoftCoreTerm``, exact in O(K) instead of O(N^2)); otherwise falls back
    to a full ``term.energy(new) - term.energy(old)`` for that term alone
    (cheap for an O(1) term like ``EnsembleTerm`` — never re-evaluates the
    whole Hamiltonian, just the terms that don't have a local path).

    Which branch each term takes is a static property of its class (``None``
    vs. an array is never data-dependent), so this is a plain Python loop
    unrolled once per term at trace time, not a runtime branch.

    Args:
        backend: The (possibly term-less) energy backend; only
            ``getattr(backend, "terms", ())`` is read.
        term_energies: ``(n_terms,)`` cached per-term totals from *before*
            the move, in the same order as ``backend.terms``.
        touched_idx: Index array of atoms the move touches (see
            ``neighbor_list.local_affected_mask`` / the move's own touched
            set — this is the SAME small set, not the dilated "affected"
            neighborhood used for the model's own cache).
        old_positions, old_types: Pre-move configuration.
        new_positions, new_types, cell: Post-move configuration.
        ensemble_params: Per-run ensemble parameters (``state.ensemble_params``).

    Returns:
        ``(n_terms,)`` updated cache. Add ``jnp.sum(...)`` of this to the
        model's own (separately patched) raw energy to get the move's full
        proposed energy.
    """
    terms = getattr(backend, "terms", ())
    if not terms:
        return term_energies

    deltas = []
    for term in terms:
        delta = term.local_energy_delta(
            touched_idx,
            old_positions,
            old_types,
            new_positions,
            new_types,
            cell,
            ensemble_params,
        )
        if delta is None:
            delta = term.energy(
                new_positions, new_types, cell, ensemble_params
            ) - term.energy(old_positions, old_types, cell, ensemble_params)
        deltas.append(delta)

    return term_energies + jnp.stack(deltas)


def seed_local_energy_cache(population: Any, base_backend: Any) -> Any:
    """Seed ``atomic_energies``/``raw_energy`` on a freshly-constructed
    population with real values, via one full backend evaluation per walker.

    ``mwg.build_mwg``'s ``init_fn`` creates these two extra fields (when an
    ``affects="local"`` move is active) at a zero-valued placeholder,
    because the generic ``extra_state_fields`` initializer signature
    ``(positions, types) -> value`` has no backend access. Every caller of
    ``init_fn`` (``init_ns``, ``init_ns_parallel``, ``init_ns_multi_gpu``,
    ``init_ns_sharded``) MUST call this immediately on the returned
    population before any local-update move can run — otherwise the first
    local move patches a delta onto a bogus zero baseline, silently
    corrupting every subsequent energy.

    A no-op (returns ``population`` unchanged) when no local-update move is
    active (no ``atomic_energies`` field on ``population``), so callers can
    apply this unconditionally without checking first.

    Args:
        population: A (possibly batched, any ``*B`` prefix) ``MCState``
            instance with ``positions``/``types``/``cell`` shaped
            ``(*B, n_atoms, 3)``/``(*B, n_atoms)``/``(*B, 3, 3)`` and a
            static ``max_neighbors`` field. Works for any batch topology
            (SingleRun, VmapRuns, PmapVmapRuns) by flattening every leading
            batch dimension into one before a single ``jax.vmap`` call.
        base_backend: The energy backend used by the run (model alone, or a
            ``Hamiltonian`` wrapping it with ``EnergyTerm``s). The model's
            own ``atomic_energies`` is reached via ``unwrap_model`` — the
            per-atom cache always holds the bare model's energies, never the
            terms' contribution (see ``local_energy_update``). Each term in
            ``backend.terms`` (if any) gets its own cached total in
            ``term_energies`` (see ``patch_term_energies``), seeded here by
            one full ``term.energy(...)`` evaluation per term, per walker.

    Returns:
        ``population`` with ``atomic_energies``/``raw_energy`` (and
        ``term_energies``, when present) set to real, backend-consistent
        values (unchanged otherwise).
    """
    if not hasattr(population, "atomic_energies"):
        return population

    positions = population.positions
    types = population.types
    cell = population.cell
    max_neighbors = population.max_neighbors
    ensemble_params = population.ensemble_params

    n_atoms = positions.shape[-2]
    batch_shape = positions.shape[:-2]

    flat_positions = positions.reshape((-1, n_atoms, 3))
    flat_types = types.reshape((-1, n_atoms))
    flat_cell = cell.reshape((-1, 3, 3))

    model = unwrap_model(base_backend)
    atomic_flat = jax.vmap(
        lambda p, t, c: model.atomic_energies(p, t, c, max_neighbors)
    )(flat_positions, flat_types, flat_cell)

    shift = getattr(model, "energy_shift_per_atom", 0.0)
    n_real_flat = jnp.sum(flat_types >= 0, axis=-1).astype(jnp.float32)
    raw_flat = atomic_flat.sum(axis=-1) + shift * n_real_flat

    atomic = atomic_flat.reshape(batch_shape + (n_atoms,))
    raw = raw_flat.reshape(batch_shape)

    updates: dict[str, jnp.ndarray] = dict(atomic_energies=atomic, raw_energy=raw)

    terms = getattr(base_backend, "terms", ())
    if terms and hasattr(population, "term_energies"):
        # ``ensemble_params`` is one shared dict for the whole population in
        # every currently-supported local_update path (single replica) —
        # closed over rather than vmapped, matching how move kernels already
        # read ``state.ensemble_params`` without a per-walker leading axis.
        def _term_energies(p, t, c):
            return jnp.stack(
                [term.energy(p, t, c, ensemble_params) for term in terms]
            )

        term_energies_flat = jax.vmap(_term_energies)(
            flat_positions, flat_types, flat_cell
        )
        updates["term_energies"] = term_energies_flat.reshape(
            batch_shape + (len(terms),)
        )

    return population.set(**updates)
