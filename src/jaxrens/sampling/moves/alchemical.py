"""Alchemical move kernels for nested sampling.

- build_morph_kernel: change one atom's species (semi-grand-canonical)

Single-walker function, designed for shard_map(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.sampling.base import MoveInfo
from jaxrens.sampling.local_energy import local_energy_update, patch_term_energies
from jaxrens.sampling.neighbor_list import (
    affected_indices_from_mask,
    build_symmetric_image_offsets,
    image_count_needed_for,
    local_affected_mask,
)
from jaxrens.unvalidated import unvalidated


@unvalidated(
    concern=("no production NS run has used this move."),
    since="0.2.2",
    clears_when=(
        "Production sGC runs delivering correct physics for a binary system."
    ),
)
def build_morph_kernel(
    backend: Any,
    n_species: int,
    max_affected: int | None = None,
):
    """Build an atom morph kernel.

    Selects a random atom, changes its type to a random different species,
    and accepts if the new energy is below the NS constraint.

    Args:
        backend: EnergyBackend instance.
        n_species: Total number of distinct species.
        max_affected: Static upper bound on "the touched atom plus its
            neighbors". When given, patches a per-atom energy cache
            incrementally instead of a full-system recompute — the
            affected set is recomputed fresh every proposal from CURRENT
            ``state.positions``/``state.cell`` (see
            ``sampling/neighbor_list.py::local_affected_mask``), and the
            periodic-image geometry itself is rebuilt every proposal from
            the CURRENT ``state.image_bucket`` (see
            ``sampling/bucket_manager.py``), so this stays correct even if
            another move in the same set changes positions or cell.
            Requires a backend that declares ``kind="single_cutoff"``
            local energy capability (checked by ``build_mwg``, not here).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    local_update = max_affected is not None
    if local_update:
        r_cutoff = backend.r_cutoff

    def step(rng_key, state, likelihood_constraint):
        key_atom, key_species = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        atom_idx = jax.random.randint(key_atom, (), 0, n_atoms)

        current_type = state.types[atom_idx]
        candidate = jax.random.randint(key_species, (), 0, n_species - 1)
        new_type = candidate + (candidate >= current_type).astype(jnp.int32)

        new_types = state.types.at[atom_idx].set(new_type)

        if not local_update:
            result = backend(
                state.positions,
                new_types,
                state.cell,
                state.max_neighbors,
                ensemble_params=state.ensemble_params,
            )
            new_energy, count, overflow = (
                result.energy,
                result.max_neighbor_count,
                result.overflow,
            )

            accepted = new_energy < likelihood_constraint

            new_state = state.set(
                types=jnp.where(accepted, new_types, state.types),
                energy=jnp.where(accepted, new_energy, state.energy),
                max_neighbor_count=jnp.maximum(
                    state.max_neighbor_count, count
                ),
                overflow=state.overflow | overflow,
            )

            info = MoveInfo(
                accepted=accepted,
                log_likelihood=-new_state.energy,
                n_evaluations=1,
            )

            return new_state, info

        # --- Local/incremental path: no full-system backend call ---
        image_offsets = build_symmetric_image_offsets(state.image_bucket)
        image_count_needed = image_count_needed_for(state.cell, r_cutoff)
        image_overflow = image_count_needed > state.image_bucket

        # A single touched atom, duplicated: local_affected_mask's mask
        # semantics make a duplicate harmless (see neighbor_list.py
        # docstring).
        touched = jnp.array([atom_idx, atom_idx])
        mask = local_affected_mask(
            touched, state.positions, state.cell, r_cutoff, image_offsets
        )
        affected, overflow = affected_indices_from_mask(mask, max_affected)

        delta_U, new_local_vec = local_energy_update(
            backend,
            state.atomic_energies,
            affected,
            state.positions,
            new_types,
            state.cell,
            state.max_neighbors,
        )
        new_raw_energy = state.raw_energy + delta_U

        new_term_energies = patch_term_energies(
            backend,
            state.term_energies,
            touched,
            state.positions,
            state.types,
            state.positions,
            new_types,
            state.cell,
            state.ensemble_params,
        )
        new_energy = new_raw_energy + jnp.sum(new_term_energies)

        accepted = (
            (new_energy < likelihood_constraint) & ~overflow & ~image_overflow
        )

        new_atomic_energies = state.atomic_energies.at[affected].set(
            new_local_vec, mode="drop"
        )

        new_state = state.set(
            types=jnp.where(accepted, new_types, state.types),
            energy=jnp.where(accepted, new_energy, state.energy),
            atomic_energies=jnp.where(
                accepted, new_atomic_energies, state.atomic_energies
            ),
            raw_energy=jnp.where(accepted, new_raw_energy, state.raw_energy),
            term_energies=jnp.where(
                accepted, new_term_energies, state.term_energies
            ),
            image_count_needed=jnp.maximum(
                state.image_count_needed, image_count_needed
            ),
            image_overflow=state.image_overflow | image_overflow,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=0,
        )

        return new_state, info

    return step
