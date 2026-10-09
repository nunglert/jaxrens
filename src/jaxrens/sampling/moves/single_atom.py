"""Single-atom move kernels for nested sampling.

Moves that perturb one atom at a time, efficient for large systems where
displacing all atoms simultaneously has low acceptance.

- build_kernel: move one random atom per step
- build_sweep_kernel: sweep through all atoms sequentially via lax.scan
- build_swap_kernel: swap species of two random atoms (multi-component)

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
    dynamic_affected_mask,
    image_count_needed_for,
    local_affected_mask,
)


def build_kernel(backend: Any, max_affected: int | None = None):
    """Build a single-atom random displacement kernel.

    Args:
        backend: EnergyBackend instance.
        max_affected: Static upper bound on "the moved atom plus its
            neighbors under its old and new position". When given, the
            kernel patches a per-atom energy cache incrementally instead
            of recomputing the whole system's energy — the affected set
            is recomputed fresh every proposal (see
            ``sampling/neighbor_list.py`` module docstring for why not a
            Verlet-style skin list), and the periodic-image geometry
            itself is rebuilt every proposal from the CURRENT
            ``state.image_bucket`` (a static field managed by the
            image-count bucket ladder — see
            ``sampling/bucket_manager.py``), so this stays correct even
            if the cell changes (a volume move, or per-walker cell
            diversity). Requires a backend that declares
            ``kind="single_cutoff"`` local energy capability (checked by
            ``build_mwg``, not here).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    local_update = max_affected is not None
    if local_update:
        default_pressure = getattr(backend, "pressure", 0.0)
        default_mu = getattr(backend, "chemical_potentials", None)
        r_cutoff = backend.r_cutoff

    def step(rng_key, state, likelihood_constraint):
        key_atom, key_disp = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        atom_idx = jax.random.randint(key_atom, (), 0, n_atoms)

        displacement = state.step_size * jax.random.normal(key_disp, (3,))
        new_positions = state.positions.at[atom_idx].add(displacement)

        if not local_update:
            result = backend(
                new_positions,
                state.types,
                state.cell,
                state.max_neighbors,
                ensemble_params=state.ensemble_params,
            )
            accepted = result.energy < likelihood_constraint

            new_state = state.set(
                positions=jnp.where(accepted, new_positions, state.positions),
                energy=jnp.where(accepted, result.energy, state.energy),
                max_neighbor_count=jnp.maximum(
                    state.max_neighbor_count, result.max_neighbor_count
                ),
                overflow=state.overflow | result.overflow,
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

        mask = dynamic_affected_mask(
            atom_idx,
            state.positions,
            new_positions,
            state.cell,
            r_cutoff,
            image_offsets,
        )
        affected, overflow = affected_indices_from_mask(mask, max_affected)

        delta_U, new_local_vec = local_energy_update(
            backend,
            state.atomic_energies,
            affected,
            new_positions,
            state.types,
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
            state.cell, state.types, pressure, mu
        )

        accepted = (
            (new_energy < likelihood_constraint) & ~overflow & ~image_overflow
        )

        new_atomic_energies = state.atomic_energies.at[affected].set(
            new_local_vec, mode="drop"
        )

        new_state = state.set(
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            atomic_energies=jnp.where(
                accepted, new_atomic_energies, state.atomic_energies
            ),
            raw_energy=jnp.where(accepted, new_raw_energy, state.raw_energy),
            # Unconditional (not gated by accepted) — mirrors how
            # max_neighbor_count/overflow are updated above: the outer
            # loop's overflow-retry must see the true geometric need even
            # for a step that would otherwise have been rejected, since
            # that rejection itself could be spurious with too small a
            # bucket.
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


def build_sweep_kernel(
    backend: Any,
    n_atoms: int,
    max_affected: int | None = None,
):
    """Build a single-atom sweep kernel.

    Sweeps through all atoms sequentially (via lax.scan), displacing each.

    Args:
        backend: EnergyBackend instance.
        n_atoms: Number of atoms (needed for scan length).
        max_affected: Same meaning as in :func:`build_kernel` — when
            given, every iteration of the sweep uses the local/incremental
            path instead of a full-system recompute.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    local_update = max_affected is not None
    if local_update:
        default_pressure = getattr(backend, "pressure", 0.0)
        default_mu = getattr(backend, "chemical_potentials", None)
        r_cutoff = backend.r_cutoff

    def step(rng_key, state, likelihood_constraint):
        keys = jax.random.split(rng_key, n_atoms)
        max_neighbors = state.max_neighbors
        ensemble_params = state.ensemble_params

        if not local_update:

            def sweep_one(carry, key_and_idx):
                positions, energy, n_accepted, acc_count, acc_overflow = carry
                key, idx = key_and_idx

                displacement = state.step_size * jax.random.normal(key, (3,))
                new_positions = positions.at[idx].add(displacement)

                result = backend(
                    new_positions,
                    state.types,
                    state.cell,
                    max_neighbors,
                    ensemble_params=ensemble_params,
                )
                accepted = result.energy < likelihood_constraint

                out_positions = jnp.where(accepted, new_positions, positions)
                out_energy = jnp.where(accepted, result.energy, energy)
                acc_count = jnp.maximum(acc_count, result.max_neighbor_count)
                acc_overflow = acc_overflow | result.overflow

                return (
                    out_positions,
                    out_energy,
                    n_accepted + accepted.astype(jnp.int32),
                    acc_count,
                    acc_overflow,
                ), None

            atom_indices = jnp.arange(n_atoms)
            init_carry = (
                state.positions,
                state.energy,
                jnp.int32(0),
                state.max_neighbor_count,
                state.overflow,
            )
            (
                final_positions,
                final_energy,
                n_accepted,
                acc_count,
                acc_overflow,
            ), _ = jax.lax.scan(sweep_one, init_carry, (keys, atom_indices))

            new_state = state.set(
                positions=final_positions,
                energy=final_energy,
                max_neighbor_count=acc_count,
                overflow=acc_overflow,
            )

            info = MoveInfo(
                accepted=n_accepted > 0,
                log_likelihood=-new_state.energy,
                n_evaluations=n_atoms,
            )

            return new_state, info

        # --- Local/incremental path: no full-system backend call ---
        # image_bucket never changes mid-sweep (this move never mutates
        # "cell"), so the offsets array is built once, outside the scan.
        image_offsets = build_symmetric_image_offsets(state.image_bucket)
        image_count_needed = image_count_needed_for(state.cell, r_cutoff)
        image_overflow = image_count_needed > state.image_bucket

        def sweep_one_local(carry, key_and_idx):
            positions, energy, atomic_energies, raw_energy, n_accepted = carry
            key, idx = key_and_idx

            displacement = state.step_size * jax.random.normal(key, (3,))
            new_positions = positions.at[idx].add(displacement)

            mask = dynamic_affected_mask(
                idx,
                positions,
                new_positions,
                state.cell,
                r_cutoff,
                image_offsets,
            )
            affected, overflow = affected_indices_from_mask(mask, max_affected)

            delta_U, new_local_vec = local_energy_update(
                backend,
                atomic_energies,
                affected,
                new_positions,
                state.types,
                state.cell,
                max_neighbors,
            )
            new_raw_energy = raw_energy + delta_U

            pressure = default_pressure
            mu = default_mu
            if isinstance(ensemble_params, dict):
                pressure = ensemble_params.get("pressure", pressure)
                mu = ensemble_params.get("chemical_potentials", mu)
            new_energy = new_raw_energy + ensemble_correction(
                state.cell, state.types, pressure, mu
            )

            accepted = (
                (new_energy < likelihood_constraint)
                & ~overflow
                & ~image_overflow
            )

            new_atomic_energies = atomic_energies.at[affected].set(
                new_local_vec, mode="drop"
            )

            out_positions = jnp.where(accepted, new_positions, positions)
            out_energy = jnp.where(accepted, new_energy, energy)
            out_atomic_energies = jnp.where(
                accepted, new_atomic_energies, atomic_energies
            )
            out_raw_energy = jnp.where(accepted, new_raw_energy, raw_energy)

            return (
                out_positions,
                out_energy,
                out_atomic_energies,
                out_raw_energy,
                n_accepted + accepted.astype(jnp.int32),
            ), None

        atom_indices = jnp.arange(n_atoms)
        init_carry = (
            state.positions,
            state.energy,
            state.atomic_energies,
            state.raw_energy,
            jnp.int32(0),
        )
        (
            final_positions,
            final_energy,
            final_atomic_energies,
            final_raw_energy,
            n_accepted,
        ), _ = jax.lax.scan(sweep_one_local, init_carry, (keys, atom_indices))

        new_state = state.set(
            positions=final_positions,
            energy=final_energy,
            atomic_energies=final_atomic_energies,
            raw_energy=final_raw_energy,
            image_count_needed=jnp.maximum(
                state.image_count_needed, image_count_needed
            ),
            image_overflow=state.image_overflow | image_overflow,
        )

        info = MoveInfo(
            accepted=n_accepted > 0,
            log_likelihood=-new_state.energy,
            n_evaluations=0,
        )

        return new_state, info

    return step


def build_swap_kernel(
    backend: Any,
    max_affected: int | None = None,
):
    """Build a single-atom species swap kernel.

    Selects two atoms of different species and swaps their types.

    Args:
        backend: EnergyBackend instance.
        max_affected: Static upper bound on the size of "the two touched
            atoms plus their neighbors". When given, the kernel patches a
            per-atom energy cache (``state.atomic_energies``/
            ``state.raw_energy``) incrementally instead of recomputing the
            whole system's energy — the affected set is recomputed fresh
            every proposal from CURRENT ``state.positions``/``state.cell``
            (see ``sampling/neighbor_list.py::local_affected_mask``), and
            the periodic-image geometry itself is rebuilt every proposal
            from the CURRENT ``state.image_bucket`` (see
            ``sampling/bucket_manager.py``), so this stays correct even if
            another move in the same set changes positions or cell.
            Requires a backend that declares ``kind="single_cutoff"``
            local energy capability (checked by ``build_mwg``, not here).

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)
    """
    local_update = max_affected is not None
    if local_update:
        # Mirrors EnsembleBackend.__call__'s own default-then-override
        # logic: read the closured pressure/mu off the (possibly
        # EnsembleBackend-wrapped) backend once, then let per-run
        # state.ensemble_params override at call time.
        default_pressure = getattr(backend, "pressure", 0.0)
        default_mu = getattr(backend, "chemical_potentials", None)
        r_cutoff = backend.r_cutoff

    def step(rng_key, state, likelihood_constraint):
        key_a, key_b = jax.random.split(rng_key)

        n_atoms = state.positions.shape[0]
        idx_a = jax.random.randint(key_a, (), 0, n_atoms)
        idx_b = jax.random.randint(key_b, (), 0, n_atoms)

        type_a = state.types[idx_a]
        type_b = state.types[idx_b]
        new_types = state.types.at[idx_a].set(type_b).at[idx_b].set(type_a)
        different_species = type_a != type_b

        if not local_update:
            result = backend(
                state.positions,
                new_types,
                state.cell,
                state.max_neighbors,
                ensemble_params=state.ensemble_params,
            )

            accepted = (
                result.energy < likelihood_constraint
            ) & different_species

            new_state = state.set(
                types=jnp.where(accepted, new_types, state.types),
                energy=jnp.where(accepted, result.energy, state.energy),
                max_neighbor_count=jnp.maximum(
                    state.max_neighbor_count, result.max_neighbor_count
                ),
                overflow=state.overflow | result.overflow,
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

        touched = jnp.array([idx_a, idx_b])
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

        # Overflow (either the affected set didn't fit max_affected, or
        # the image bucket was too small for the current cell) means the
        # computed delta can't be trusted — reject rather than risk a
        # silently-wrong energy (mirrors how other moves reject on cell/
        # prior violations rather than accepting an unverified proposal).
        accepted = (
            (new_energy < likelihood_constraint)
            & different_species
            & ~overflow
            & ~image_overflow
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
