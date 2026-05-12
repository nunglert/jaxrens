"""Replica exchange moves across parallel NS runs.

Unlike standard moves that operate on a single walker, replica exchange
swaps walkers between different parallel runs (across the P dimension).
This improves mixing when runs have different energy constraints.

Supports both simple energy-based RE and pressure RE with enthalpies.

SwapKernel abstraction:
    New code should use ``PressureRENSSwap`` (or any ``SwapKernel`` subclass)
    directly and pass it to ``replica_exchange_step`` via the ``swap_kernel``
    parameter.  The standalone ``perform_swap`` function is retained for
    back-compat only; see its docstring for details.

XRENS:
    ``XRENSSwap`` performs composition-morphing RE.  Because ``propose``
    requires a backend call (energy re-evaluation after morphing), XRENS
    uses a separate ``xrens_replica_exchange_step`` that properly invokes
    both ``propose`` and ``accept`` for every pair.  Do not pass an
    ``XRENSSwap`` instance to the generic ``replica_exchange_step``.

SemiGrand:
    ``SemiGrandSwap`` performs chemical-potential RE.  Walkers' positions,
    cells, and types stay unchanged; only the μ assignment is swapped.  The
    grand-canonical energy ``Ω = U - μ·N`` is recomputed with the receiving
    run's μ and compared to Emax.  Zero backend calls are required.  Use
    ``semi_grand_replica_exchange_step`` (not the generic
    ``replica_exchange_step``) for this kernel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.sampling.morph import morph_types_to_composition


# ---------------------------------------------------------------------------
# SwapKernel abstraction
# ---------------------------------------------------------------------------


class SwapKernel(ABC):
    """One flavor of inter-RE swap logic.

    Subclasses implement :meth:`propose` (produce swap candidate state) and
    :meth:`accept` (decide whether to accept it).  The split keeps the
    proposal logic (which may involve backend energy evaluations for XRENS)
    separate from the deterministic acceptance check.
    """

    @abstractmethod
    def propose(
        self,
        state_a: Any,
        state_b: Any,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
        rng_key: jax.Array,
        backend: Any,
    ) -> tuple[Any, int, int]:
        """Produce swap candidate states plus evaluation counts.

        Args:
            state_a: State dict / pytree for replica A.
            state_b: State dict / pytree for replica B.
            ensemble_params_a: Ensemble parameters (pressure, μ, …) for run A.
            ensemble_params_b: Ensemble parameters for run B.
            rng_key: JAX PRNG key.
            backend: Energy/force backend (may be ``None`` for zero-eval kernels).

        Returns:
            Tuple ``(proposed, n_energy_evals, n_grad_evals)`` where
            ``proposed`` is a dict with keys
            ``{'positions_a', 'positions_b', 'cell_a', 'cell_b',
               'energy_a', 'energy_b', 'types_a', 'types_b'}``,
            and the two integers count backend calls made.
        """

    @abstractmethod
    def accept(
        self,
        proposed: Any,
        emax_a: jnp.ndarray,
        emax_b: jnp.ndarray,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
    ) -> jnp.ndarray:
        """Decide whether to accept the proposed swap.

        Args:
            proposed: Output of :meth:`propose`.
            emax_a: Emax scalar for run A.
            emax_b: Emax scalar for run B.
            ensemble_params_a: Ensemble parameters for run A.
            ensemble_params_b: Ensemble parameters for run B.

        Returns:
            Boolean scalar (JIT-compatible).
        """


# ---------------------------------------------------------------------------
# PressureRENSSwap — concrete kernel (commit 1)
# ---------------------------------------------------------------------------


class PressureRENSSwap(SwapKernel):
    """Pressure-RENS swap: identity proposal with enthalpy acceptance check.

    ``propose`` is identity — the positions/cell/energy pair is passed
    through unchanged as the swap candidate (no morphing, no backend call).

    ``accept`` checks enthalpies:
        H_A_in_j = E_A + P_j * V_A < Emax_j
        H_B_in_i = E_B + P_i * V_B < Emax_i

    For simple energy-based RE (``volumes``/``pressures`` not provided in
    the proposed dict or set to ``None``), falls back to direct energy
    comparison:
        E_A < Emax_j AND E_B < Emax_i

    Equivalent to the legacy :func:`perform_swap` function.
    """

    def propose(
        self,
        state_a: Any,
        state_b: Any,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
        rng_key: jax.Array,
        backend: Any,
    ) -> tuple[Any, int, int]:
        """Identity proposal: return the walker pair as-is with zero eval counts.

        Args:
            state_a: Dict with at minimum keys ``'energy'``, and optionally
                ``'positions'``, ``'cell'``, ``'types'``.
            state_b: Same structure as ``state_a`` for the other replica.
            ensemble_params_a: Must contain ``'pressure'`` (scalar float) if
                pressure RE is desired; otherwise ``None``.
            ensemble_params_b: Same structure as ``ensemble_params_a``.
            rng_key: Unused for identity proposal; accepted for API uniformity.
            backend: Unused; accepted for API uniformity.

        Returns:
            ``(proposed, 0, 0)`` where ``proposed`` is a dict mirroring the
            input states (positions, cell, energy, types for each replica).
        """
        proposed = {
            "positions_a": state_a.get("positions"),
            "positions_b": state_b.get("positions"),
            "cell_a": state_a.get("cell"),
            "cell_b": state_b.get("cell"),
            "energy_a": state_a.get("energy"),
            "energy_b": state_b.get("energy"),
            "types_a": state_a.get("types"),
            "types_b": state_b.get("types"),
        }
        return proposed, 0, 0

    def accept(
        self,
        proposed: Any,
        emax_a: jnp.ndarray,
        emax_b: jnp.ndarray,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
    ) -> jnp.ndarray:
        """Enthalpy-based acceptance check (or simple energy check without pressure).

        Replicates the logic of the legacy :func:`perform_swap` function.

        Args:
            proposed: Dict with keys ``'energy_a'``, ``'energy_b'``, and
                optionally ``'cell_a'``, ``'cell_b'``.
            emax_a: Emax scalar for run A (the Emax constraint walker A lives in).
            emax_b: Emax scalar for run B.
            ensemble_params_a: Dict with optional key ``'pressure'`` (scalar).
            ensemble_params_b: Dict with optional key ``'pressure'`` (scalar).

        Returns:
            Boolean scalar: ``True`` iff the swap is accepted.
        """
        e_a = proposed["energy_a"]
        e_b = proposed["energy_b"]

        p_a = ensemble_params_a.get("pressure") if isinstance(ensemble_params_a, dict) else None
        p_b = ensemble_params_b.get("pressure") if isinstance(ensemble_params_b, dict) else None

        use_pressure = (
            p_a is not None
            and p_b is not None
            and proposed.get("cell_a") is not None
            and proposed.get("cell_b") is not None
        )

        if use_pressure:
            v_a = _get_volume(proposed["cell_a"])
            v_b = _get_volume(proposed["cell_b"])
            # Enthalpy of A evaluated at B's pressure (run_b's constraint)
            h_a_in_b = e_a + p_b * v_a
            # Enthalpy of B evaluated at A's pressure (run_a's constraint)
            h_b_in_a = e_b + p_a * v_b
            accepted = (h_a_in_b < emax_b) & (h_b_in_a < emax_a)
        else:
            accepted = (e_a < emax_b) & (e_b < emax_a)

        return accepted


# ---------------------------------------------------------------------------
# Pair-generation helper (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Back-compat shim: perform_swap
#
# RETAINED FOR BACK-COMPAT.  New code should use PressureRENSSwap directly.
# This shim delegates to PressureRENSSwap().accept() after constructing the
# minimal proposed/ensemble_params dicts that the kernel expects.
# ---------------------------------------------------------------------------

_PRESSURE_RENS_SWAP_INSTANCE = PressureRENSSwap()


def perform_swap(
    energies_pair: jnp.ndarray,
    emax_pair: jnp.ndarray,
    volumes_pair: jnp.ndarray | None = None,
    pressures_pair: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Decide whether to accept a replica exchange swap.

    .. deprecated::
        This is a back-compat shim retained so existing call sites and tests
        continue to work unchanged.  New code should instantiate
        :class:`PressureRENSSwap` and call its :meth:`~PressureRENSSwap.accept`
        method directly.

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
    # Build the proposed / ensemble_params dicts that PressureRENSSwap.accept expects.
    proposed: dict[str, Any] = {
        "energy_a": energies_pair[0],
        "energy_b": energies_pair[1],
    }

    if volumes_pair is not None and pressures_pair is not None:
        # Fake cell matrices whose determinant equals the supplied volumes.
        # _get_volume calls jnp.linalg.det, so we need actual (3,3) matrices.
        # Use a diagonal matrix: det(diag(cbrt(V), cbrt(V), cbrt(V))) = V.
        def _volume_to_cell(v: jnp.ndarray) -> jnp.ndarray:
            side = jnp.cbrt(v)
            return jnp.diag(jnp.array([side, side, side]))

        proposed["cell_a"] = _volume_to_cell(volumes_pair[0])
        proposed["cell_b"] = _volume_to_cell(volumes_pair[1])
        ensemble_params_a: dict[str, Any] = {"pressure": pressures_pair[0]}
        ensemble_params_b: dict[str, Any] = {"pressure": pressures_pair[1]}
    else:
        ensemble_params_a = {}
        ensemble_params_b = {}

    return _PRESSURE_RENS_SWAP_INSTANCE.accept(
        proposed,
        emax_pair[0],
        emax_pair[1],
        ensemble_params_a,
        ensemble_params_b,
    )


# ---------------------------------------------------------------------------
# Volume helper
# ---------------------------------------------------------------------------


def _get_volume(cell: jnp.ndarray) -> jnp.ndarray:
    """Compute volume from cell matrix. cell shape: (3, 3)."""
    return jnp.abs(jnp.linalg.det(cell))


# ---------------------------------------------------------------------------
# replica_exchange_step
# ---------------------------------------------------------------------------


def replica_exchange_step(
    rng_key: jax.Array,
    all_positions: jnp.ndarray,
    all_types: jnp.ndarray,
    all_energies: jnp.ndarray,
    all_cells: jnp.ndarray | None,
    all_emax: jnp.ndarray,
    pressures: jnp.ndarray | None = None,
    n_swap_cycles: int = 1,
    swap_kernel: SwapKernel | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, dict]:
    """Perform replica exchange across P parallel runs.

    .. note::
        Passing ``swap_kernel`` explicitly is preferred.  When omitted,
        the function defaults to :class:`PressureRENSSwap` (back-compat).

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
        swap_kernel: :class:`SwapKernel` instance to use for acceptance logic.
            Defaults to ``PressureRENSSwap()`` for back-compat.

    Returns:
        (new_positions, new_types, new_energies, new_cells, swap_info)
        where swap_info is a dict with 'n_accepted', 'n_attempted'.
    """
    if swap_kernel is None:
        swap_kernel = PressureRENSSwap()

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

    n_pairs = max(n_runs - 1, 0)

    if max_pairs == 0:
        # No swaps possible (n_runs < 2)
        return (
            all_positions,
            all_types,
            all_energies,
            all_cells,
            {
                "n_accepted": jnp.array(0),
                "n_attempted": jnp.array(0),
                "n_accepted_per_pair": jnp.zeros(n_pairs, dtype=jnp.int32),
                "n_attempted_per_pair": jnp.zeros(n_pairs, dtype=jnp.int32),
            },
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
        positions, types, energies, cells, acc_per_pair, att_per_pair, key = carry
        pairs, mask, phase_key = phase_input

        # Generate random walker indices for all pairs
        pair_keys = jax.random.split(phase_key, max_pairs)

        def _attempt_one_swap(carry_inner, swap_input):
            """Attempt a single swap between a pair of runs."""
            pos, typ, ene, bxs, acc_per_pair_in = carry_inner
            pair, valid, swap_key = swap_input

            k1, k2 = jax.random.split(swap_key)
            run_i, run_j = pair[0], pair[1]

            # Random walker from each run
            wi = jax.random.randint(k1, (), 0, n_walkers)
            wj = jax.random.randint(k2, (), 0, n_walkers)

            # Extract energies
            e_i = ene[run_i, wi]
            e_j = ene[run_j, wj]

            # Build proposed dict for the kernel
            proposed: dict[str, Any] = {
                "energy_a": e_i,
                "energy_b": e_j,
            }
            if bxs is not None:
                proposed["cell_a"] = bxs[run_i, wi]
                proposed["cell_b"] = bxs[run_j, wj]
            else:
                proposed["cell_a"] = None
                proposed["cell_b"] = None

            emax_a = all_emax[run_i]
            emax_b = all_emax[run_j]

            # Build per-pair ensemble params
            if pressures is not None:
                ens_a = {"pressure": pressures[run_i]}
                ens_b = {"pressure": pressures[run_j]}
            else:
                ens_a = {}
                ens_b = {}

            accepted = swap_kernel.accept(proposed, emax_a, emax_b, ens_a, ens_b)

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
            if bxs is not None:
                b_i = bxs[run_i, wi]
                b_j = bxs[run_j, wj]
                new_bxs = bxs.at[run_i, wi].set(jnp.where(do_swap, b_j, b_i))
                new_bxs = new_bxs.at[run_j, wj].set(jnp.where(do_swap, b_i, b_j))
            else:
                new_bxs = bxs

            # Per-pair scatter-add: pair_id = min(run_i, run_j) since
            # pairs are always (k, k+1).  Even/odd phases write to
            # disjoint pair_ids; padding entries have valid=False so
            # contribute zero.
            pair_id = jnp.minimum(run_i, run_j)
            delta = valid.astype(jnp.int32) * do_swap.astype(jnp.int32)
            new_acc_per_pair = acc_per_pair_in.at[pair_id].add(delta)

            return (new_pos, new_typ, new_ene, new_bxs, new_acc_per_pair), None

        scan_inputs = (pairs, mask, pair_keys)
        (positions, types, energies, cells, acc_per_pair), _ = jax.lax.scan(
            _attempt_one_swap,
            (positions, types, energies, cells, acc_per_pair),
            scan_inputs,
        )
        # Per-pair attempted: scatter the phase mask by pair_id.  No
        # collisions within a phase (each pair_id appears at most once
        # in the unpadded slice; padded entries have mask=0).
        pair_ids_phase = jnp.minimum(pairs[:, 0], pairs[:, 1])
        att_per_pair = att_per_pair.at[pair_ids_phase].add(mask.astype(jnp.int32))

        return (positions, types, energies, cells, acc_per_pair, att_per_pair, key), None

    # Run n_swap_cycles, each with even + odd phase
    def _do_one_cycle(carry, cycle_key):
        positions, types, energies, cells, acc_per_pair, att_per_pair = carry
        even_key, odd_key = jax.random.split(cycle_key)

        # Even phase
        (positions, types, energies, cells, acc_per_pair, att_per_pair, _), _ = _do_one_phase(
            (positions, types, energies, cells, acc_per_pair, att_per_pair, even_key),
            (all_pairs[0], all_masks[0], even_key),
        )
        # Odd phase
        (positions, types, energies, cells, acc_per_pair, att_per_pair, _), _ = _do_one_phase(
            (positions, types, energies, cells, acc_per_pair, att_per_pair, odd_key),
            (all_pairs[1], all_masks[1], odd_key),
        )

        return (positions, types, energies, cells, acc_per_pair, att_per_pair), None

    cycle_keys = jax.random.split(rng_key, n_swap_cycles)
    init_carry = (
        all_positions,
        types_broadcastable,
        all_energies,
        all_cells,
        jnp.zeros(n_pairs, dtype=jnp.int32),
        jnp.zeros(n_pairs, dtype=jnp.int32),
    )
    (new_pos, new_types, new_ene, new_cells, acc_per_pair, att_per_pair), _ = jax.lax.scan(
        _do_one_cycle, init_carry, cycle_keys
    )

    swap_info = {
        "n_accepted": jnp.sum(acc_per_pair),
        "n_attempted": jnp.sum(att_per_pair),
        "n_accepted_per_pair": acc_per_pair,
        "n_attempted_per_pair": att_per_pair,
    }

    return new_pos, new_types, new_ene, new_cells, swap_info


# ---------------------------------------------------------------------------
# XRENSSwap — composition-morphing swap kernel
# ---------------------------------------------------------------------------


class XRENSSwap(SwapKernel):
    """Composition-swap RE: morph types + re-evaluate energy + enthalpy check.

    When swapping replica A (at composition ``target_a``) with replica B
    (at ``target_b``):

    - Replica A receives walker from B; its types are morphed to ``target_a``.
    - Replica B receives walker from A; its types are morphed to ``target_b``.
    - Each morphed config's energy is re-evaluated under the new types via
      the backend.
    - Enthalpy check (E + P·V, or pure energy if no pressure) against each
      replica's emax decides acceptance.

    Each swap makes exactly 2 energy evaluations, 0 gradient evaluations.

    Args:
        n_species: Static int; number of distinct species labels.  Must match
            the length of each ``target_composition`` array passed at call time.
    """

    def __init__(self, n_species: int) -> None:
        if not isinstance(n_species, int) or n_species < 1:
            raise ValueError(
                f"n_species must be a positive Python int, got {n_species!r}"
            )
        self.n_species = n_species

    def propose(
        self,
        state_a: Any,
        state_b: Any,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
        rng_key: jax.Array,
        backend: Any,
    ) -> tuple[Any, int, int]:
        """Morph types and re-evaluate energies for both replicas.

        Args:
            state_a: Dict with keys ``'positions'``, ``'types'``, ``'energy'``,
                ``'cell'``.
            state_b: Same structure as ``state_a``.
            ensemble_params_a: Dict with key ``'target_composition'`` — int
                array of shape ``(n_species,)`` summing to ``n_atoms``.
            ensemble_params_b: Same structure as ``ensemble_params_a``.
            rng_key: JAX PRNG key; split internally for the two morphs.
            backend: Energy/force callable with signature
                ``backend(positions, types, cell, max_neighbors, ensemble_params)
                -> (energy, n_evals, overflow)``.

        Returns:
            ``(proposed, 2, 0)`` where ``proposed`` is a dict with keys
            ``{'positions_a', 'cell_a', 'types_a', 'energy_a',
               'positions_b', 'cell_b', 'types_b', 'energy_b'}``.

        Raises:
            ValueError: If ``target_composition`` is missing from either
                ensemble_params dict.
        """
        if not isinstance(ensemble_params_a, dict) or "target_composition" not in ensemble_params_a:
            raise ValueError(
                "XRENSSwap.propose: ensemble_params_a must contain "
                "'target_composition' (int array of shape (n_species,)). "
                f"Got keys: {list(ensemble_params_a.keys()) if isinstance(ensemble_params_a, dict) else type(ensemble_params_a)}"
            )
        if not isinstance(ensemble_params_b, dict) or "target_composition" not in ensemble_params_b:
            raise ValueError(
                "XRENSSwap.propose: ensemble_params_b must contain "
                "'target_composition' (int array of shape (n_species,)). "
                f"Got keys: {list(ensemble_params_b.keys()) if isinstance(ensemble_params_b, dict) else type(ensemble_params_b)}"
            )

        target_a = jnp.asarray(ensemble_params_a["target_composition"], dtype=jnp.int32)
        target_b = jnp.asarray(ensemble_params_b["target_composition"], dtype=jnp.int32)

        key_a, key_b = jax.random.split(rng_key)

        # A receives B's walker: morph B's types to A's target composition.
        morphed_types_for_a = morph_types_to_composition(
            key_a, state_b["types"], target_a, self.n_species
        )
        # B receives A's walker: morph A's types to B's target composition.
        morphed_types_for_b = morph_types_to_composition(
            key_b, state_a["types"], target_b, self.n_species
        )

        # Re-evaluate energies under morphed types.
        e_a_new, _, _ = backend(
            state_b["positions"],
            morphed_types_for_a,
            state_b["cell"],
            0,  # max_neighbors
        )
        e_b_new, _, _ = backend(
            state_a["positions"],
            morphed_types_for_b,
            state_a["cell"],
            0,
        )

        proposed = {
            "positions_a": state_b["positions"],
            "cell_a": state_b["cell"],
            "types_a": morphed_types_for_a,
            "energy_a": e_a_new,
            "positions_b": state_a["positions"],
            "cell_b": state_a["cell"],
            "types_b": morphed_types_for_b,
            "energy_b": e_b_new,
        }
        return proposed, 2, 0

    def accept(
        self,
        proposed: Any,
        emax_a: jnp.ndarray,
        emax_b: jnp.ndarray,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
    ) -> jnp.ndarray:
        """Enthalpy-based acceptance on the morphed + re-evaluated energies.

        Delegates to :class:`PressureRENSSwap` acceptance logic: the proposed
        dict already contains the re-evaluated post-morph energies so the
        math is identical to pressure-RENS once energies are set.

        Args:
            proposed: Dict from :meth:`propose` with keys ``'energy_a'``,
                ``'energy_b'``, ``'cell_a'``, ``'cell_b'``.
            emax_a: Emax scalar for run A.
            emax_b: Emax scalar for run B.
            ensemble_params_a: Dict with optional key ``'pressure'``.
            ensemble_params_b: Dict with optional key ``'pressure'``.

        Returns:
            Boolean scalar: ``True`` iff the swap is accepted.
        """
        return _PRESSURE_RENS_SWAP_INSTANCE.accept(
            proposed, emax_a, emax_b, ensemble_params_a, ensemble_params_b
        )


# ---------------------------------------------------------------------------
# xrens_replica_exchange_step
# ---------------------------------------------------------------------------


def xrens_replica_exchange_step(
    rng_key: jax.Array,
    all_positions: jnp.ndarray,
    all_types: jnp.ndarray,
    all_energies: jnp.ndarray,
    all_cells: jnp.ndarray | None,
    all_emax: jnp.ndarray,
    composition_targets: jnp.ndarray,
    backend: Any,
    xrens_kernel: XRENSSwap,
    pressures: jnp.ndarray | None = None,
    n_swap_cycles: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, dict]:
    """Perform XRENS (composition-morphing) replica exchange across P parallel runs.

    Unlike :func:`replica_exchange_step` (identity proposal), each swap here
    morphs atom types and re-evaluates the energy via the backend.

    For each swap cycle:
      1. Even phase: attempt swaps (0,1), (2,3), ...
      2. Odd phase: attempt swaps (1,2), (3,4), ...

    In each swap, randomly select one walker from each run in the pair,
    morph types to the target composition of the receiving run, re-evaluate
    energy, and accept/reject via enthalpy check.

    Args:
        rng_key: JAX PRNG key.
        all_positions: ``(P, K, n_atoms, 3)`` positions for all runs.
        all_types: ``(P, K, n_atoms)`` per-walker types (XRENS requires
            per-walker types since each run has a different composition).
        all_energies: ``(P, K)`` energies for all runs.
        all_cells: ``(P, K, 3, 3)`` or None.
        all_emax: ``(P,)`` current Emax for each run.
        composition_targets: ``(P, n_species)`` int array of target
            compositions; row *p* is the target for run *p*.
        backend: Energy backend callable.
        xrens_kernel: ``XRENSSwap`` instance.
        pressures: ``(P,)`` or None for enthalpy check.
        n_swap_cycles: Number of even+odd swap phases per call.

    Returns:
        ``(new_positions, new_types, new_energies, new_cells, swap_info)``
        where ``swap_info`` has keys ``'n_accepted'``, ``'n_attempted'``,
        ``'n_energy_evals'``.
    """
    n_runs = all_positions.shape[0]
    n_walkers = all_positions.shape[1]

    # Pre-compute swap pairs for both phases.
    even_pairs = get_swap_pairs(n_runs, 0)  # (n_even, 2)
    odd_pairs = get_swap_pairs(n_runs, 1)   # (n_odd, 2)

    n_even = even_pairs.shape[0]
    n_odd = odd_pairs.shape[0]
    max_pairs = max(n_even, n_odd) if (n_even > 0 or n_odd > 0) else 0
    n_pairs = max(n_runs - 1, 0)

    if max_pairs == 0:
        return (
            all_positions,
            all_types,
            all_energies,
            all_cells,
            {
                "n_accepted": jnp.array(0),
                "n_attempted": jnp.array(0),
                "n_energy_evals": jnp.array(0),
                "n_accepted_per_pair": jnp.zeros(n_pairs, dtype=jnp.int32),
                "n_attempted_per_pair": jnp.zeros(n_pairs, dtype=jnp.int32),
            },
        )

    def _pad_pairs(pairs, n_valid, max_len):
        if pairs.shape[0] < max_len:
            padding = jnp.zeros((max_len - pairs.shape[0], 2), dtype=pairs.dtype)
            pairs = jnp.concatenate([pairs, padding], axis=0)
        mask = jnp.arange(max_len) < n_valid
        return pairs, mask

    even_pairs_padded, even_mask = _pad_pairs(even_pairs, n_even, max_pairs)
    odd_pairs_padded, odd_mask = _pad_pairs(odd_pairs, n_odd, max_pairs)

    all_pairs = jnp.stack([even_pairs_padded, odd_pairs_padded], axis=0)
    all_masks = jnp.stack([even_mask, odd_mask], axis=0)

    def _do_one_phase(carry, phase_input):
        """Process one phase of XRENS swap attempts."""
        positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total, key = carry
        pairs, mask, phase_key = phase_input

        pair_keys = jax.random.split(phase_key, max_pairs)

        def _attempt_one_xrens_swap(carry_inner, swap_input):
            pos, typ, ene, bxs, acc_per_pair_in, eval_count = carry_inner
            pair, valid, swap_key = swap_input

            k1, k2, morph_key = jax.random.split(swap_key, 3)
            run_i, run_j = pair[0], pair[1]

            # Random walker from each run.
            wi = jax.random.randint(k1, (), 0, n_walkers)
            wj = jax.random.randint(k2, (), 0, n_walkers)

            # Build state dicts for propose().
            cell_i = bxs[run_i, wi] if bxs is not None else jnp.zeros((3, 3))
            cell_j = bxs[run_j, wj] if bxs is not None else jnp.zeros((3, 3))
            state_i = {
                "positions": pos[run_i, wi],
                "types": typ[run_i, wi],
                "energy": ene[run_i, wi],
                "cell": cell_i,
            }
            state_j = {
                "positions": pos[run_j, wj],
                "types": typ[run_j, wj],
                "energy": ene[run_j, wj],
                "cell": cell_j,
            }

            # Ensemble params with target_composition for each run.
            ens_i: dict[str, Any] = {
                "target_composition": composition_targets[run_i]
            }
            ens_j: dict[str, Any] = {
                "target_composition": composition_targets[run_j]
            }
            if pressures is not None:
                ens_i["pressure"] = pressures[run_i]
                ens_j["pressure"] = pressures[run_j]

            # propose() morphs types + re-evaluates energies.
            proposed, n_e, _ = xrens_kernel.propose(
                state_i, state_j, ens_i, ens_j, morph_key, backend
            )

            emax_i = all_emax[run_i]
            emax_j = all_emax[run_j]

            accepted = xrens_kernel.accept(proposed, emax_i, emax_j, ens_i, ens_j)

            # Only apply if valid (not padding) and accepted.
            do_swap = accepted & valid

            # Update positions.
            new_pos = pos.at[run_i, wi].set(
                jnp.where(do_swap, proposed["positions_a"], pos[run_i, wi])
            )
            new_pos = new_pos.at[run_j, wj].set(
                jnp.where(do_swap, proposed["positions_b"], pos[run_j, wj])
            )

            # Update types (always per-walker for XRENS).
            new_typ = typ.at[run_i, wi].set(
                jnp.where(do_swap, proposed["types_a"], typ[run_i, wi])
            )
            new_typ = new_typ.at[run_j, wj].set(
                jnp.where(do_swap, proposed["types_b"], typ[run_j, wj])
            )

            # Update energies.
            new_ene = ene.at[run_i, wi].set(
                jnp.where(do_swap, proposed["energy_a"], ene[run_i, wi])
            )
            new_ene = new_ene.at[run_j, wj].set(
                jnp.where(do_swap, proposed["energy_b"], ene[run_j, wj])
            )

            # Update cells if present.
            if bxs is not None:
                new_bxs = bxs.at[run_i, wi].set(
                    jnp.where(do_swap, proposed["cell_a"], bxs[run_i, wi])
                )
                new_bxs = new_bxs.at[run_j, wj].set(
                    jnp.where(do_swap, proposed["cell_b"], bxs[run_j, wj])
                )
            else:
                new_bxs = bxs

            # Per-pair scatter-add (see replica_exchange_step for the
            # convention rationale).
            pair_id = jnp.minimum(run_i, run_j)
            delta = valid.astype(jnp.int32) * do_swap.astype(jnp.int32)
            new_acc_per_pair = acc_per_pair_in.at[pair_id].add(delta)
            # We always call propose (2 evals) when valid, regardless of acceptance.
            new_eval_count = eval_count + valid.astype(jnp.int32) * n_e

            return (new_pos, new_typ, new_ene, new_bxs, new_acc_per_pair, new_eval_count), None

        scan_inputs = (pairs, mask, pair_keys)
        (positions, types, energies, cells, acc_per_pair, n_evals_total), _ = jax.lax.scan(
            _attempt_one_xrens_swap,
            (positions, types, energies, cells, acc_per_pair, n_evals_total),
            scan_inputs,
        )
        pair_ids_phase = jnp.minimum(pairs[:, 0], pairs[:, 1])
        att_per_pair = att_per_pair.at[pair_ids_phase].add(mask.astype(jnp.int32))
        return (positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total, key), None

    def _do_one_cycle(carry, cycle_key):
        positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total = carry
        even_key, odd_key = jax.random.split(cycle_key)

        (positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total, _), _ = _do_one_phase(
            (positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total, even_key),
            (all_pairs[0], all_masks[0], even_key),
        )
        (positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total, _), _ = _do_one_phase(
            (positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total, odd_key),
            (all_pairs[1], all_masks[1], odd_key),
        )
        return (positions, types, energies, cells, acc_per_pair, att_per_pair, n_evals_total), None

    cycle_keys = jax.random.split(rng_key, n_swap_cycles)
    init_carry = (
        all_positions,
        all_types,
        all_energies,
        all_cells,
        jnp.zeros(n_pairs, dtype=jnp.int32),
        jnp.zeros(n_pairs, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )
    (new_pos, new_types, new_ene, new_cells, acc_per_pair, att_per_pair, total_evals), _ = (
        jax.lax.scan(_do_one_cycle, init_carry, cycle_keys)
    )

    swap_info = {
        "n_accepted": jnp.sum(acc_per_pair),
        "n_attempted": jnp.sum(att_per_pair),
        "n_energy_evals": total_evals,
        "n_accepted_per_pair": acc_per_pair,
        "n_attempted_per_pair": att_per_pair,
    }
    return new_pos, new_types, new_ene, new_cells, swap_info


# ---------------------------------------------------------------------------
# SemiGrandSwap — chemical-potential swap kernel
# ---------------------------------------------------------------------------


class SemiGrandSwap(SwapKernel):
    """Semi-grand (μVT / μPT) RE: swap chemical-potential assignments.

    Each replica i holds a per-species chemical potential vector μ_i of
    shape ``(n_species,)``.  On a swap between replicas A and B, replica A
    adopts μ_B and replica B adopts μ_A.  Positions, cells, and types are
    **never** changed — only the μ assignment moves.

    **Sign convention (matching legacy jaxnest
    ``create_perform_semi_grand_swap`` at
    ``jaxns-devAS/src/jaxnest/replica_exchange.py:195-249``):**

    The stored ``state.energy`` is the raw potential energy *U* (not the
    grand-canonical energy Ω).  The grand-canonical energy under chemical
    potential μ and species count array N is::

        Ω = U - μ · N

    where ``N[s] = number of atoms of species s`` (i.e.
    ``jnp.bincount(types, length=n_species)``).

    Under the swap, replica A (with Emax_A) receives μ_B::

        Ω_A_new = U_A - μ_B · N_A

    and replica B (with Emax_B) receives μ_A::

        Ω_B_new = U_B - μ_A · N_B

    The swap is accepted iff both ``Ω_A_new < Emax_A`` and
    ``Ω_B_new < Emax_B``.

    Zero backend calls are made.  No morphing, no position change.

    Args:
        n_species: Static Python int; number of distinct species labels.
            Must match the length of each ``chemical_potentials`` array
            passed at call time.
    """

    def __init__(self, n_species: int) -> None:
        if not isinstance(n_species, int) or n_species < 1:
            raise ValueError(
                f"n_species must be a positive Python int, got {n_species!r}"
            )
        self.n_species = n_species

    def propose(
        self,
        state_a: Any,
        state_b: Any,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
        rng_key: jax.Array,
        backend: Any,
    ) -> tuple[Any, int, int]:
        """Compute grand-canonical energies under swapped μ assignments.

        Positions, cells, and types are copied through unchanged.  The
        ``energy`` fields in the returned ``proposed`` dict are the
        grand-canonical energies ``Ω = U - μ_new · N`` with the *swapped*
        chemical potentials.

        Args:
            state_a: Dict with keys ``'positions'``, ``'cell'``, ``'types'``,
                ``'energy'`` (raw potential energy U_A).
            state_b: Same structure for replica B.
            ensemble_params_a: Dict with key ``'chemical_potentials'`` — float
                array of shape ``(n_species,)``.
            ensemble_params_b: Same structure for replica B.
            rng_key: Unused; accepted for API uniformity.
            backend: Unused; accepted for API uniformity (zero backend calls).

        Returns:
            ``(proposed, 0, 0)`` where ``proposed`` is a dict with keys
            ``{'positions_a', 'cell_a', 'types_a', 'energy_a',
               'positions_b', 'cell_b', 'types_b', 'energy_b'}``.
            ``energy_a`` = ``U_A - μ_B · N_A``.
            ``energy_b`` = ``U_B - μ_A · N_B``.

        Raises:
            ValueError: If ``'chemical_potentials'`` is absent from either
                ensemble_params dict.
        """
        if not isinstance(ensemble_params_a, dict) or "chemical_potentials" not in ensemble_params_a:
            raise ValueError(
                "SemiGrandSwap.propose: ensemble_params_a must contain "
                "'chemical_potentials' (float array of shape (n_species,)). "
                f"Got keys: {list(ensemble_params_a.keys()) if isinstance(ensemble_params_a, dict) else type(ensemble_params_a)}"
            )
        if not isinstance(ensemble_params_b, dict) or "chemical_potentials" not in ensemble_params_b:
            raise ValueError(
                "SemiGrandSwap.propose: ensemble_params_b must contain "
                "'chemical_potentials' (float array of shape (n_species,)). "
                f"Got keys: {list(ensemble_params_b.keys()) if isinstance(ensemble_params_b, dict) else type(ensemble_params_b)}"
            )

        mu_a = jnp.asarray(ensemble_params_a["chemical_potentials"], dtype=jnp.float32)
        mu_b = jnp.asarray(ensemble_params_b["chemical_potentials"], dtype=jnp.float32)

        n_species = self.n_species
        if mu_a.shape != (n_species,) or mu_b.shape != (n_species,):
            raise ValueError(
                f"SemiGrandSwap(n_species={n_species}): chemical_potentials must "
                f"have shape ({n_species},). Got mu_a.shape={mu_a.shape}, "
                f"mu_b.shape={mu_b.shape}."
            )

        types_a = jnp.asarray(state_a["types"], dtype=jnp.int32)
        types_b = jnp.asarray(state_b["types"], dtype=jnp.int32)

        # Species counts: N_A[s] = number of atoms of species s in walker A.
        N_A = jnp.bincount(types_a, length=n_species)  # (n_species,)
        N_B = jnp.bincount(types_b, length=n_species)  # (n_species,)

        # Grand-canonical energy under the *swapped* chemical potential.
        # Replica A adopts μ_B: Ω_A_new = U_A - μ_B · N_A
        # Replica B adopts μ_A: Ω_B_new = U_B - μ_A · N_B
        omega_a = state_a["energy"] - jnp.dot(mu_b, N_A)
        omega_b = state_b["energy"] - jnp.dot(mu_a, N_B)

        proposed = {
            "positions_a": state_a.get("positions"),
            "cell_a": state_a.get("cell"),
            "types_a": types_a,
            "energy_a": omega_a,   # grand-canonical energy under new μ
            "positions_b": state_b.get("positions"),
            "cell_b": state_b.get("cell"),
            "types_b": types_b,
            "energy_b": omega_b,   # grand-canonical energy under new μ
        }
        return proposed, 0, 0

    def accept(
        self,
        proposed: Any,
        emax_a: jnp.ndarray,
        emax_b: jnp.ndarray,
        ensemble_params_a: Any,
        ensemble_params_b: Any,
    ) -> jnp.ndarray:
        """Accept iff grand-canonical energies are below Emax on both sides.

        ``proposed['energy_a']`` and ``proposed['energy_b']`` are already the
        grand-canonical energies Ω = U - μ_new · N computed by :meth:`propose`,
        so the acceptance criterion reduces to a simple threshold check::

            Ω_A_new < Emax_A  AND  Ω_B_new < Emax_B

        Args:
            proposed: Dict from :meth:`propose` with keys ``'energy_a'``,
                ``'energy_b'``.
            emax_a: Emax scalar for run A.
            emax_b: Emax scalar for run B.
            ensemble_params_a: Unused; accepted for API uniformity.
            ensemble_params_b: Unused; accepted for API uniformity.

        Returns:
            Boolean scalar: ``True`` iff the swap is accepted.
        """
        return (proposed["energy_a"] < emax_a) & (proposed["energy_b"] < emax_b)


# ---------------------------------------------------------------------------
# semi_grand_replica_exchange_step
# ---------------------------------------------------------------------------


def semi_grand_replica_exchange_step(
    rng_key: jax.Array,
    all_positions: jnp.ndarray,
    all_types: jnp.ndarray,
    all_energies: jnp.ndarray,
    all_cells: jnp.ndarray | None,
    all_emax: jnp.ndarray,
    chemical_potentials: jnp.ndarray,
    semi_grand_kernel: "SemiGrandSwap",
    pressures: jnp.ndarray | None = None,
    n_swap_cycles: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, dict]:
    """Perform semi-grand (μVT/μPT) replica exchange across P parallel runs.

    Swaps chemical-potential *assignments* between pairs of runs.  Positions,
    cells, and types are never changed.  Each run's walkers are updated only
    when the grand-canonical energy under the new μ falls below Emax.

    Zero backend calls per swap cycle.

    For each swap cycle:
      1. Even phase: attempt swaps (0,1), (2,3), ...
      2. Odd phase: attempt swaps (1,2), (3,4), ...

    In each swap, randomly select one walker from each run in the pair.
    The acceptance condition is::

        (U_i - μ_j · N_i) < Emax_i  AND  (U_j - μ_i · N_j) < Emax_j

    When a swap is accepted, positions and types stay put; only the walkers'
    energy fields are updated to their new grand-canonical values.

    Args:
        rng_key: JAX PRNG key.
        all_positions: ``(P, K, n_atoms, 3)`` positions for all runs.
        all_types: ``(P, K, n_atoms)`` per-walker types.
        all_energies: ``(P, K)`` raw potential energies for all runs.
        all_cells: ``(P, K, 3, 3)`` or None.
        all_emax: ``(P,)`` current Emax for each run.
        chemical_potentials: ``(P, n_species)`` float array; row p is μ for run p.
        semi_grand_kernel: ``SemiGrandSwap`` instance.
        pressures: ``(P,)`` or None (unused by semi-grand; accepted for
            interface parity).
        n_swap_cycles: Number of even+odd swap phases per call.

    Returns:
        ``(new_positions, new_types, new_energies, new_cells, swap_info)``
        where ``swap_info`` has keys ``'n_accepted'``, ``'n_attempted'``,
        ``'n_energy_evals'`` (always 0 for semi-grand).
    """
    n_runs = all_positions.shape[0]
    n_walkers = all_positions.shape[1]
    n_species = semi_grand_kernel.n_species

    even_pairs = get_swap_pairs(n_runs, 0)
    odd_pairs = get_swap_pairs(n_runs, 1)

    n_even = even_pairs.shape[0]
    n_odd = odd_pairs.shape[0]
    max_pairs = max(n_even, n_odd) if (n_even > 0 or n_odd > 0) else 0
    n_pairs = max(n_runs - 1, 0)

    if max_pairs == 0:
        return (
            all_positions,
            all_types,
            all_energies,
            all_cells,
            {
                "n_accepted": jnp.array(0),
                "n_attempted": jnp.array(0),
                "n_energy_evals": jnp.array(0),
                "n_accepted_per_pair": jnp.zeros(n_pairs, dtype=jnp.int32),
                "n_attempted_per_pair": jnp.zeros(n_pairs, dtype=jnp.int32),
            },
        )

    def _pad_pairs(pairs, n_valid, max_len):
        if pairs.shape[0] < max_len:
            padding = jnp.zeros((max_len - pairs.shape[0], 2), dtype=pairs.dtype)
            pairs = jnp.concatenate([pairs, padding], axis=0)
        mask = jnp.arange(max_len) < n_valid
        return pairs, mask

    even_pairs_padded, even_mask = _pad_pairs(even_pairs, n_even, max_pairs)
    odd_pairs_padded, odd_mask = _pad_pairs(odd_pairs, n_odd, max_pairs)

    all_pairs = jnp.stack([even_pairs_padded, odd_pairs_padded], axis=0)
    all_masks = jnp.stack([even_mask, odd_mask], axis=0)

    def _do_one_phase(carry, phase_input):
        """Process one phase of semi-grand swap attempts."""
        positions, types, energies, cells, acc_per_pair, att_per_pair = carry
        pairs, mask, phase_key = phase_input

        pair_keys = jax.random.split(phase_key, max_pairs)

        def _attempt_one_sg_swap(carry_inner, swap_input):
            pos, typ, ene, bxs, acc_per_pair_in = carry_inner
            pair, valid, swap_key = swap_input

            k1, k2 = jax.random.split(swap_key)
            run_i, run_j = pair[0], pair[1]

            wi = jax.random.randint(k1, (), 0, n_walkers)
            wj = jax.random.randint(k2, (), 0, n_walkers)

            # Build per-walker state dicts.
            cell_i = bxs[run_i, wi] if bxs is not None else jnp.zeros((3, 3))
            cell_j = bxs[run_j, wj] if bxs is not None else jnp.zeros((3, 3))
            state_i = {
                "positions": pos[run_i, wi],
                "types": typ[run_i, wi],
                "energy": ene[run_i, wi],
                "cell": cell_i,
            }
            state_j = {
                "positions": pos[run_j, wj],
                "types": typ[run_j, wj],
                "energy": ene[run_j, wj],
                "cell": cell_j,
            }

            ens_i: dict[str, Any] = {"chemical_potentials": chemical_potentials[run_i]}
            ens_j: dict[str, Any] = {"chemical_potentials": chemical_potentials[run_j]}

            proposed, _, _ = semi_grand_kernel.propose(
                state_i, state_j, ens_i, ens_j, swap_key, None
            )

            emax_i = all_emax[run_i]
            emax_j = all_emax[run_j]

            accepted = semi_grand_kernel.accept(proposed, emax_i, emax_j, ens_i, ens_j)
            do_swap = accepted & valid

            # Positions, cells, types are unchanged; only energies update
            # (to the grand-canonical value under the new μ).
            # Walkers do NOT physically move to the other run; instead the
            # energy field is updated to reflect the new μ assignment.
            # The swap is symmetric: run i's selected walker now "lives" under
            # μ_j, and run j's selected walker now "lives" under μ_i.
            new_ene = ene.at[run_i, wi].set(
                jnp.where(do_swap, proposed["energy_a"], ene[run_i, wi])
            )
            new_ene = new_ene.at[run_j, wj].set(
                jnp.where(do_swap, proposed["energy_b"], ene[run_j, wj])
            )

            # Per-pair scatter-add (see replica_exchange_step for the
            # convention rationale).
            pair_id = jnp.minimum(run_i, run_j)
            delta = valid.astype(jnp.int32) * do_swap.astype(jnp.int32)
            new_acc_per_pair = acc_per_pair_in.at[pair_id].add(delta)

            return (pos, typ, new_ene, bxs, new_acc_per_pair), None

        scan_inputs = (pairs, mask, pair_keys)
        (positions, types, energies, cells, acc_per_pair), _ = jax.lax.scan(
            _attempt_one_sg_swap,
            (positions, types, energies, cells, acc_per_pair),
            scan_inputs,
        )
        pair_ids_phase = jnp.minimum(pairs[:, 0], pairs[:, 1])
        att_per_pair = att_per_pair.at[pair_ids_phase].add(mask.astype(jnp.int32))
        return (positions, types, energies, cells, acc_per_pair, att_per_pair), None

    def _do_one_cycle(carry, cycle_key):
        positions, types, energies, cells, acc_per_pair, att_per_pair = carry
        even_key, odd_key = jax.random.split(cycle_key)

        (positions, types, energies, cells, acc_per_pair, att_per_pair), _ = _do_one_phase(
            (positions, types, energies, cells, acc_per_pair, att_per_pair),
            (all_pairs[0], all_masks[0], even_key),
        )
        (positions, types, energies, cells, acc_per_pair, att_per_pair), _ = _do_one_phase(
            (positions, types, energies, cells, acc_per_pair, att_per_pair),
            (all_pairs[1], all_masks[1], odd_key),
        )
        return (positions, types, energies, cells, acc_per_pair, att_per_pair), None

    cycle_keys = jax.random.split(rng_key, n_swap_cycles)
    init_carry = (
        all_positions,
        all_types,
        all_energies,
        all_cells,
        jnp.zeros(n_pairs, dtype=jnp.int32),
        jnp.zeros(n_pairs, dtype=jnp.int32),
    )
    (new_pos, new_types, new_ene, new_cells, acc_per_pair, att_per_pair), _ = jax.lax.scan(
        _do_one_cycle, init_carry, cycle_keys
    )

    swap_info = {
        "n_accepted": jnp.sum(acc_per_pair),
        "n_attempted": jnp.sum(att_per_pair),
        "n_energy_evals": jnp.array(0, dtype=jnp.int32),
        "n_accepted_per_pair": acc_per_pair,
        "n_attempted_per_pair": att_per_pair,
    }
    return new_pos, new_types, new_ene, new_cells, swap_info
