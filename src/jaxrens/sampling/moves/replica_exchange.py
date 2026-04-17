"""Replica exchange moves across parallel NS runs.

Unlike standard moves that operate on a single walker, replica exchange
swaps walkers between different parallel runs (across the P dimension).
This improves mixing when runs have different energy constraints.

Supports both simple energy-based RE and pressure RE with enthalpies.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def get_swap_pairs(n_runs: int, phase: int) -> jnp.ndarray:
    """Get non-overlapping swap pairs for replica exchange.

    Phase 0 (even): [(0,1), (2,3), (4,5), ...]
    Phase 1 (odd):  [(1,2), (3,4), (5,6), ...]

    Args:
        n_runs: Number of parallel runs.
        phase: 0 for even pairs, 1 for odd pairs.

    Returns:
        Array of shape (n_pairs, 2) with run indices.
    """
    # Maximum possible pairs
    starts = jnp.arange(phase, n_runs - 1, 2)
    pairs = jnp.stack([starts, starts + 1], axis=-1)
    return pairs


def perform_swap(
    energies_pair: jnp.ndarray,
    emax_pair: jnp.ndarray,
    volumes_pair: jnp.ndarray | None = None,
    pressures_pair: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Decide whether to accept a replica exchange swap.

    For pressure RE, the acceptance checks enthalpies:
      H_A_in_j = E_A + P_j * V_A < Emax_j
      H_B_in_i = E_B + P_i * V_B < Emax_i

    For simple RE (no pressure):
      E_A < Emax_j AND E_B < Emax_i

    Args:
        energies_pair: (2,) energies of the two walkers.
        emax_pair: (2,) Emax constraints of the two runs.
        volumes_pair: (2,) or None, volumes for pressure RE.
        pressures_pair: (2,) or None, pressures for pressure RE.

    Returns:
        accepted: bool scalar.
    """
    e_a, e_b = energies_pair[0], energies_pair[1]
    emax_i, emax_j = emax_pair[0], emax_pair[1]

    use_pressure = (pressures_pair is not None) & (volumes_pair is not None)

    if pressures_pair is not None and volumes_pair is not None:
        p_i, p_j = pressures_pair[0], pressures_pair[1]
        v_a, v_b = volumes_pair[0], volumes_pair[1]
        # Enthalpy of A evaluated at j's pressure
        h_a_in_j = e_a + p_j * v_a
        # Enthalpy of B evaluated at i's pressure
        h_b_in_i = e_b + p_i * v_b
        accepted = (h_a_in_j < emax_j) & (h_b_in_i < emax_i)
    else:
        # Simple energy-based RE
        accepted = (e_a < emax_j) & (e_b < emax_i)

    return accepted


def _get_volume(cell: jnp.ndarray) -> jnp.ndarray:
    """Compute volume from cell matrix. cell shape: (3, 3)."""
    return jnp.abs(jnp.linalg.det(cell))


def replica_exchange_step(
    rng_key: jax.Array,
    all_positions: jnp.ndarray,
    all_types: jnp.ndarray,
    all_energies: jnp.ndarray,
    all_cells: jnp.ndarray | None,
    all_emax: jnp.ndarray,
    pressures: jnp.ndarray | None = None,
    n_swap_cycles: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, dict]:
    """Perform replica exchange across P parallel runs.

    For each swap cycle:
      1. Even phase: attempt swaps (0,1), (2,3), ...
      2. Odd phase: attempt swaps (1,2), (3,4), ...

    In each swap, randomly pick one walker from each run in the pair,
    attempt the swap, and update if accepted.

    Args:
        rng_key: JAX PRNG key.
        all_positions: (P, K, n_atoms, 3) positions for all runs.
        all_energies: (P, K) energies for all runs.
        all_types: (P, K, n_atoms) or (P, n_atoms) types.
        all_cells: (P, K, 3, 3) or None.
        all_emax: (P,) current Emax for each run.
        pressures: (P,) or None.
        n_swap_cycles: number of even+odd phase cycles.

    Returns:
        (new_positions, new_types, new_energies, new_cells, swap_info)
        where swap_info is a dict with 'n_accepted', 'n_attempted'.
    """
    n_runs = all_positions.shape[0]
    n_walkers = all_positions.shape[1]

    # Handle n_runs=1: no swaps possible
    # We still go through the logic but pairs will be empty

    # Pre-compute swap pairs for both phases
    even_pairs = get_swap_pairs(n_runs, 0)  # (n_even, 2)
    odd_pairs = get_swap_pairs(n_runs, 1)  # (n_odd, 2)

    # Pad to same length so we can use a uniform loop body
    n_even = even_pairs.shape[0]
    n_odd = odd_pairs.shape[0]
    max_pairs = max(n_even, n_odd) if (n_even > 0 or n_odd > 0) else 0

    if max_pairs == 0:
        # No swaps possible (n_runs < 2)
        return (
            all_positions,
            all_types,
            all_energies,
            all_cells,
            {"n_accepted": jnp.array(0), "n_attempted": jnp.array(0)},
        )

    # Pad pairs arrays to max_pairs with dummy pair (0, 0) and mask
    def _pad_pairs(pairs, n_valid, max_len):
        if pairs.shape[0] < max_len:
            padding = jnp.zeros((max_len - pairs.shape[0], 2), dtype=pairs.dtype)
            pairs = jnp.concatenate([pairs, padding], axis=0)
        mask = jnp.arange(max_len) < n_valid
        return pairs, mask

    even_pairs_padded, even_mask = _pad_pairs(even_pairs, n_even, max_pairs)
    odd_pairs_padded, odd_mask = _pad_pairs(odd_pairs, n_odd, max_pairs)

    # Stack phases: (2, max_pairs, 2) for pairs, (2, max_pairs) for masks
    all_pairs = jnp.stack([even_pairs_padded, odd_pairs_padded], axis=0)
    all_masks = jnp.stack([even_mask, odd_mask], axis=0)

    # Types might be (P, n_atoms) — broadcast to (P, K, n_atoms) if needed
    types_broadcastable = all_types
    types_are_per_walker = all_types.ndim >= 3

    def _do_one_phase(carry, phase_input):
        """Process one phase (even or odd) of swap attempts."""
        positions, types, energies, cells, n_acc, n_att, key = carry
        pairs, mask, phase_key = phase_input

        # Generate random walker indices for all pairs
        pair_keys = jax.random.split(phase_key, max_pairs)

        def _attempt_one_swap(carry_inner, swap_input):
            """Attempt a single swap between a pair of runs."""
            pos, typ, ene, bxs, acc_count = carry_inner
            pair, valid, swap_key = swap_input

            k1, k2 = jax.random.split(swap_key)
            run_i, run_j = pair[0], pair[1]

            # Random walker from each run
            wi = jax.random.randint(k1, (), 0, n_walkers)
            wj = jax.random.randint(k2, (), 0, n_walkers)

            # Extract energies
            e_i = ene[run_i, wi]
            e_j = ene[run_j, wj]
            energies_pair = jnp.array([e_i, e_j])
            emax_pair = jnp.array([all_emax[run_i], all_emax[run_j]])

            # Volumes and pressures for pressure RE
            if cells is not None and pressures is not None:
                v_i = _get_volume(bxs[run_i, wi])
                v_j = _get_volume(bxs[run_j, wj])
                volumes_pair = jnp.array([v_i, v_j])
                pressures_pair = jnp.array([pressures[run_i], pressures[run_j]])
                accepted = perform_swap(energies_pair, emax_pair, volumes_pair, pressures_pair)
            else:
                accepted = perform_swap(energies_pair, emax_pair)

            # Only actually swap if valid (not padding) and accepted
            do_swap = accepted & valid

            # Swap positions
            pos_i = pos[run_i, wi]
            pos_j = pos[run_j, wj]
            new_pos = pos.at[run_i, wi].set(jnp.where(do_swap, pos_j, pos_i))
            new_pos = new_pos.at[run_j, wj].set(jnp.where(do_swap, pos_i, pos_j))

            # Swap energies
            new_ene = ene.at[run_i, wi].set(jnp.where(do_swap, e_j, e_i))
            new_ene = new_ene.at[run_j, wj].set(jnp.where(do_swap, e_i, e_j))

            # Swap types if per-walker
            if types_are_per_walker:
                t_i = typ[run_i, wi]
                t_j = typ[run_j, wj]
                new_typ = typ.at[run_i, wi].set(
                    jnp.where(do_swap, t_j, t_i)
                )
                new_typ = new_typ.at[run_j, wj].set(
                    jnp.where(do_swap, t_i, t_j)
                )
            else:
                new_typ = typ

            # Swap cells if present
            if cells is not None:
                b_i = bxs[run_i, wi]
                b_j = bxs[run_j, wj]
                new_bxs = bxs.at[run_i, wi].set(jnp.where(do_swap, b_j, b_i))
                new_bxs = new_bxs.at[run_j, wj].set(jnp.where(do_swap, b_i, b_j))
            else:
                new_bxs = bxs

            new_acc = acc_count + valid.astype(jnp.int32) * do_swap.astype(jnp.int32)

            return (new_pos, new_typ, new_ene, new_bxs, new_acc), None

        scan_inputs = (pairs, mask, pair_keys)
        (positions, types, energies, cells, n_acc), _ = jax.lax.scan(
            _attempt_one_swap,
            (positions, types, energies, cells, n_acc),
            scan_inputs,
        )
        n_att = n_att + jnp.sum(mask.astype(jnp.int32))

        return (positions, types, energies, cells, n_acc, n_att, key), None

    # Run n_swap_cycles, each with even + odd phase
    def _do_one_cycle(carry, cycle_key):
        positions, types, energies, cells, n_acc, n_att = carry
        even_key, odd_key = jax.random.split(cycle_key)

        # Even phase
        (positions, types, energies, cells, n_acc, n_att, _), _ = _do_one_phase(
            (positions, types, energies, cells, n_acc, n_att, even_key),
            (all_pairs[0], all_masks[0], even_key),
        )
        # Odd phase
        (positions, types, energies, cells, n_acc, n_att, _), _ = _do_one_phase(
            (positions, types, energies, cells, n_acc, n_att, odd_key),
            (all_pairs[1], all_masks[1], odd_key),
        )

        return (positions, types, energies, cells, n_acc, n_att), None

    cycle_keys = jax.random.split(rng_key, n_swap_cycles)
    init_carry = (
        all_positions,
        types_broadcastable,
        all_energies,
        all_cells,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )
    (new_pos, new_types, new_ene, new_cells, total_acc, total_att), _ = jax.lax.scan(
        _do_one_cycle, init_carry, cycle_keys
    )

    swap_info = {
        "n_accepted": total_acc,
        "n_attempted": total_att,
    }

    return new_pos, new_types, new_ene, new_cells, swap_info
