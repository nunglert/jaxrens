"""Single-atom move kernels for nested sampling.

Moves that perturb one atom at a time, efficient for large systems where
displacing all atoms simultaneously has low acceptance.

- build_kernel: move one random atom per step
- build_sweep_kernel: sweep through all atoms sequentially via lax.scan

Single-walker functions, designed for shard_map(vmap(vmap(...))) wrapping.
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
    dynamic_affected_mask,
    image_count_needed_for,
)
from jaxrens.unvalidated import unvalidated


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

        touched = jnp.array([atom_idx])
        new_term_energies = patch_term_energies(
            backend,
            state.term_energies,
            touched,
            state.positions,
            state.types,
            new_positions,
            state.types,
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
            positions=jnp.where(accepted, new_positions, state.positions),
            energy=jnp.where(accepted, new_energy, state.energy),
            atomic_energies=jnp.where(
                accepted, new_atomic_energies, state.atomic_energies
            ),
            raw_energy=jnp.where(accepted, new_raw_energy, state.raw_energy),
            term_energies=jnp.where(
                accepted, new_term_energies, state.term_energies
            ),
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


@unvalidated(
    concern=("no production NS run has used this move."),
    since="0.2.2",
    clears_when=("Benchmark e.g. on LJ comparing with GMC/random walk"),
)
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
            (
                positions,
                energy,
                atomic_energies,
                raw_energy,
                term_energies,
                n_accepted,
            ) = carry
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

            touched = jnp.array([idx])
            new_term_energies = patch_term_energies(
                backend,
                term_energies,
                touched,
                positions,
                state.types,
                new_positions,
                state.types,
                state.cell,
                ensemble_params,
            )
            new_energy = new_raw_energy + jnp.sum(new_term_energies)

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
            out_term_energies = jnp.where(
                accepted, new_term_energies, term_energies
            )

            return (
                out_positions,
                out_energy,
                out_atomic_energies,
                out_raw_energy,
                out_term_energies,
                n_accepted + accepted.astype(jnp.int32),
            ), None

        atom_indices = jnp.arange(n_atoms)
        init_carry = (
            state.positions,
            state.energy,
            state.atomic_energies,
            state.raw_energy,
            state.term_energies,
            jnp.int32(0),
        )
        (
            final_positions,
            final_energy,
            final_atomic_energies,
            final_raw_energy,
            final_term_energies,
            n_accepted,
        ), _ = jax.lax.scan(sweep_one_local, init_carry, (keys, atom_indices))

        new_state = state.set(
            positions=final_positions,
            energy=final_energy,
            atomic_energies=final_atomic_energies,
            raw_energy=final_raw_energy,
            term_energies=final_term_energies,
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
