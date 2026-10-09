"""Alchemical move kernels for nested sampling.

- build_morph_kernel: change one atom's species (semi-grand-canonical)
- build_shift_kernel: random translation of all atoms (rigid shift)

Single-walker functions, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.backends.ensemble import ensemble_correction
from jaxrens.base import MoveInfo
from jaxrens.sampling.local_energy import local_energy_update
from jaxrens.sampling.neighbor_list import (
    affected_indices_from_mask,
    build_symmetric_image_offsets,
    image_count_needed_for,
    local_affected_mask,
)
from jaxrens.utils.cell import wrap_positions


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
        default_pressure = getattr(backend, "pressure", 0.0)
        default_mu = getattr(backend, "chemical_potentials", None)
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

        pressure = default_pressure
        mu = default_mu
        ep = state.ensemble_params
        if isinstance(ep, dict):
            pressure = ep.get("pressure", pressure)
            mu = ep.get("chemical_potentials", mu)
        new_energy = new_raw_energy + ensemble_correction(
            state.cell, new_types, pressure, mu
        )

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


def build_shift_kernel(
    backend: Any, assume_translation_invariant: bool = False
):
    """Build a random shift kernel.

    Applies a uniform random translation to all atoms simultaneously.

    Args:
        backend: EnergyBackend instance.
        assume_translation_invariant: If True, skip the energy evaluation
            entirely and always accept. Sound only for backends whose energy
            depends solely on *relative* interatomic distances (e.g.
            NeuralIL, LJ, MACE, nequix under periodic boundary conditions) —
            a uniform rigid shift then leaves every pairwise difference, and
            hence the total energy, exactly unchanged (verified against
            NeuralIL's descriptor math: every neighbor delta is a difference
            of two identically-shifted coordinates, so it is bitwise
            invariant under the shift). Do NOT enable this for backends
            whose energy depends on absolute position (e.g. a fixed
            external field/trap, or ``gaussian_mixture``'s center-of-mass
            term), where a shift genuinely changes the energy.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """

    def step(rng_key, state, likelihood_constraint):
        shift = state.step_size * jax.random.normal(rng_key, (3,))
        new_positions = state.positions + shift[None, :]

        if assume_translation_invariant:
            # Energy is provably unchanged (see docstring) — skip the
            # backend call entirely and always accept. Wrap into the home
            # cell purely for numerical hygiene (keeps positions from
            # drifting unboundedly over a long run); this does not affect
            # the energy, since wrap_positions is itself a lattice
            # translation per atom-group and the energy is invariant under
            # those too.
            wrapped_positions = wrap_positions(new_positions, state.cell)
            new_state = state.set(positions=wrapped_positions)
            info = MoveInfo(
                accepted=jnp.asarray(True),
                log_likelihood=-state.energy,
                n_evaluations=0,
            )
            return new_state, info

        result = backend(
            new_positions,
            state.types,
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
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            max_neighbor_count=jnp.maximum(state.max_neighbor_count, count),
            overflow=state.overflow | overflow,
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
        )

        return new_state, info

    return step
