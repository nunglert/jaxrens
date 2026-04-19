"""Core nested sampling loop.

Two-loop architecture:
- Outer loop (Python): adaptation, overflow retry, termination, callbacks, logging
- Inner step (JIT): ns_step — one dead-point replacement via lax.scan over MCMC

Functions:
- init_ns(): Initialize NSState from walker data + MWG init_fn
- ns_step(): One dead-point replacement (fully JIT-compatible)
- run_ns(): Full NS run (outer Python loop calling ns_step)
- init_ns_parallel() / run_ns_parallel(): Multi-run vmap support
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from jaxrens.base import NSCallback
from jaxrens.sampling.termination import (
    IterationTermination,
    PriorMassTermination,
    TerminationCriterion,
    check_any,
)
from jaxrens.state.ns import NSState
from jaxrens.utils.cell import get_volume

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_worst_walker(
    potentials: jnp.ndarray,
    rng_key: jax.Array,
    n_atoms: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find the walker with the highest potential (worst likelihood).

    Fully JIT-compatible. Uses per-atom comparison for float precision
    and random tie-breaking among walkers sharing the maximum.

    Args:
        potentials: Per-walker potentials, shape (n_walkers,).
        rng_key: PRNG key for random tie-breaking.
        n_atoms: Number of atoms per walker (for per-atom precision).

    Returns:
        (index, value) of the worst walker.
    """
    comparison = potentials / n_atoms
    max_val = jnp.max(comparison)
    is_max = jnp.isclose(comparison, max_val, rtol=1e-7, atol=0.0)
    noise = jax.random.uniform(rng_key, shape=comparison.shape)
    tie_scores = jnp.where(is_max, noise, -jnp.inf)
    idx = jnp.argmax(tie_scores)
    return idx, potentials[idx]


def _find_worst_walkers(
    energies: jnp.ndarray,
    n_cull: int,
    rng_key: jax.Array | None = None,
    n_atoms: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find the n_cull walkers with the highest energies.

    Returns:
        (indices, values) — both shape (n_cull,), sorted by descending energy.
    """
    comparison = energies / n_atoms if n_atoms is not None else energies

    if rng_key is not None:
        noise = 1e-10 * jax.random.uniform(
            rng_key, shape=comparison.shape, minval=-1.0, maxval=1.0
        )
        comparison = comparison + noise

    sorted_indices = jnp.argsort(-comparison)
    worst_indices = sorted_indices[:n_cull]
    worst_values = energies[worst_indices]
    return worst_indices, worst_values


def _get_extra_indices(
    worst_idx: jnp.ndarray,
    n_walkers: int,
    n_extra: int,
    rng_key: jax.Array,
) -> jnp.ndarray:
    """Pick n_extra random walker indices, excluding the worst walker."""
    noise = jax.random.uniform(rng_key, shape=(n_walkers,))
    noise = noise.at[worst_idx].set(-jnp.inf)
    return jnp.argsort(-noise)[:n_extra]


# ---------------------------------------------------------------------------
# init_ns
# ---------------------------------------------------------------------------


def init_ns(
    init_fn: Callable,
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    cells: jnp.ndarray | None,
    rng_key: jax.Array,
    max_dead: int = 50000,
    step_sizes: jnp.ndarray | None = None,
    ensemble_params: dict | None = None,
    restart_state=None,
) -> NSState:
    """Initialize NSState from walker data.

    Builds a batched MCState population by calling init_fn per walker
    and stacking into a single pytree with (n_walkers, ...) batch dims.

    The energies passed in are raw backend energies. init_fn applies
    any ensemble correction (if the backend is an EnsembleBackend).

    Args:
        init_fn: MCState constructor from build_mwg().
        positions: Initial walker positions, shape (n_walkers, n_atoms, 3).
        types: Atom types, shape (n_walkers, n_atoms) or (n_atoms,).
        energies: Initial walker energies, shape (n_walkers,).
        cells: Unit cells, shape (n_walkers, 3, 3) or None.
        rng_key: JAX PRNG key.
        max_dead: Maximum number of dead points to collect.
        step_sizes: Per-move step size array. None = use descriptor defaults.
        ensemble_params: Ensemble parameters dict for MCState.
        restart_state: Optional RestartBundle from load_restart(). When
            provided, seeds NSState dead-point history and bookkeeping
            scalars from the checkpoint instead of zero-initializing.
            The live-walker side is always taken from positions/energies.

    Returns:
        Initialized NSState with batched MCState population.
    """
    n_walkers = positions.shape[0]
    n_atoms = positions.shape[1] if positions.ndim >= 2 else 1

    # Broadcast types if not batched
    if types.ndim == 1:
        types_batched = jnp.broadcast_to(types[None, :], (n_walkers, types.shape[0]))
    else:
        types_batched = types

    # Build batched population by stacking per-walker MCStates
    walkers = []
    for i in range(n_walkers):
        w = init_fn(
            positions=positions[i],
            types=types_batched[i],
            energy=energies[i],
            cell=cells[i] if cells is not None else None,
            step_sizes=step_sizes,
            ensemble_params=ensemble_params,
        )
        walkers.append(w)
    population = jax.tree.map(lambda *xs: jnp.stack(xs), *walkers)

    if restart_state is None:
        dead_energies = jnp.full(max_dead, jnp.inf)
        dead_positions = jnp.zeros((max_dead, n_atoms, 3))
        dead_volumes = jnp.zeros(max_dead)
        log_evidence = jnp.array(-jnp.inf)
        iteration = jnp.array(0, dtype=jnp.int32)
        n_dead = jnp.array(0, dtype=jnp.int32)
    else:
        rs = restart_state
        nd = rs.n_dead

        dead_energies = jnp.full(max_dead, jnp.inf)
        dead_energies = dead_energies.at[:nd].set(jnp.asarray(rs.dead_energies[:nd]))

        dead_positions = jnp.zeros((max_dead, n_atoms, 3))
        dead_positions = dead_positions.at[:nd].set(
            jnp.asarray(rs.dead_positions[:nd])
        )

        if rs.dead_volumes is not None:
            dead_volumes = jnp.zeros(max_dead)
            dead_volumes = dead_volumes.at[:nd].set(
                jnp.asarray(rs.dead_volumes[:nd])
            )
        else:
            dead_volumes = jnp.zeros(max_dead)

        log_evidence = jnp.array(rs.log_evidence)
        iteration = jnp.array(rs.iteration, dtype=jnp.int32)
        n_dead = jnp.array(nd, dtype=jnp.int32)

    state = NSState(
        population=population,
        dead_energies=dead_energies,
        dead_positions=dead_positions,
        dead_volumes=dead_volumes,
        log_evidence=log_evidence,
        iteration=iteration,
        n_dead=n_dead,
        rng_key=rng_key,
        n_walkers=n_walkers,
        n_atoms=n_atoms,
        max_dead=max_dead,
    )

    logger.debug(
        "NS state initialized: %d walkers, energy range [%.4g, %.4g], max_dead=%d%s",
        n_walkers, float(jnp.min(energies)), float(jnp.max(energies)), max_dead,
        f" (restart from iteration {restart_state.iteration})" if restart_state is not None else "",
    )
    return state


# ---------------------------------------------------------------------------
# ns_step — fully JIT-compatible
# ---------------------------------------------------------------------------


def ns_step(
    ns_state: NSState,
    step_fn: Callable,
    n_mcmc_steps: int = 20,
    n_extra: int = 0,
) -> tuple[NSState, dict]:
    """One NS iteration: replace the worst walker, with optional extra walks.

    Fully JIT-compatible. MCState.energy is the full ensemble potential
    (computed by the backend, possibly wrapped in EnsembleBackend).
    No enthalpy computation needed here — ns_step is ensemble-agnostic.

    When n_extra > 0, additional randomly chosen walkers from the
    population are MCMC-walked alongside the clone via vmap.

    Args:
        ns_state: Current NSState with batched MCState population.
        step_fn: MCMC step function from build_mwg().
        n_mcmc_steps: Number of MCMC steps per walker (static).
        n_extra: Number of additional walkers to walk (static).

    Returns:
        (updated_ns_state, info_dict).
    """
    pop = ns_state.population
    potentials = pop.energy  # (n_walkers,) — full ensemble potential
    key = ns_state.rng_key

    # 1. Find worst walker (highest potential)
    key, key_worst = jax.random.split(key)
    worst_idx, potential_max = _find_worst_walker(
        potentials, rng_key=key_worst, n_atoms=ns_state.n_atoms,
    )

    # 2. Record dead point
    n_dead = ns_state.n_dead
    dead_energies = ns_state.dead_energies.at[n_dead].set(potential_max)
    dead_positions = ns_state.dead_positions.at[n_dead].set(
        pop.positions[worst_idx]
    )
    volumes = jax.vmap(get_volume)(pop.cell)  # (n_walkers,)
    dead_volumes = ns_state.dead_volumes.at[n_dead].set(volumes[worst_idx])

    # 3. Update evidence estimate
    n_walkers = ns_state.n_walkers
    log_weight = -ns_state.iteration / n_walkers + jnp.log(1.0 / n_walkers)
    log_evidence = jnp.logaddexp(
        ns_state.log_evidence, log_weight + (-potential_max)
    )

    # 4. Clone random survivor to replace worst
    key, key_clone, key_extra, key_mcmc = jax.random.split(key, 4)
    clone_idx = jax.random.randint(key_clone, (), 0, n_walkers - 1)
    clone_idx = jnp.where(clone_idx >= worst_idx, clone_idx + 1, clone_idx)

    clone = jax.tree.map(lambda x: x[clone_idx], pop)
    n_moves = clone.n_accepted.shape[-1]
    clone = clone.set(
        n_accepted=jnp.zeros(n_moves, dtype=jnp.int32),
        n_proposed=jnp.zeros(n_moves, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
    )
    # Write clone into population at worst_idx (before walking)
    pop = jax.tree.map(
        lambda pop_field, clone_field: pop_field.at[worst_idx].set(clone_field),
        pop, clone,
    )

    # 5. Gather walkers to walk: worst_idx (the clone) + n_extra survivors
    if n_extra > 0:
        extra_indices = _get_extra_indices(worst_idx, n_walkers, n_extra, key_extra)
        walk_indices = jnp.concatenate([worst_idx[None], extra_indices])
    else:
        walk_indices = worst_idx[None]

    n_walk = 1 + n_extra
    walk_batch = jax.tree.map(lambda x: x[walk_indices], pop)

    # 6. Run MCMC chains in parallel via vmap(lax.scan)
    chain_keys = jax.random.split(key_mcmc, n_walk)
    all_mcmc_keys = jax.vmap(lambda k: jax.random.split(k, n_mcmc_steps))(chain_keys)

    def run_one_chain(walker, chain_keys):
        # Bucket convention for reject_reason_counts_per_move:
        #   axis 1 index 0 = accepted count
        #   axis 1 index 1 = energy reject count
        #   axis 1 index 2 = cell reject count
        #   axis 1 index 3 = prior reject count
        # Each row sums to n_proposed_per_move for that move.
        # This makes the 4 buckets exclusive and exhaustive.
        def scan_body(state, step_key):
            new_state, info = step_fn(step_key, state, potential_max)
            # Per-move accepted count: scatter accepted flag onto move_idx
            n_acc = jnp.zeros(n_moves, dtype=jnp.int32).at[info.move_idx].add(
                info.accepted.astype(jnp.int32)
            )
            # Per-move proposed count: always +1 for this move
            n_prop = jnp.zeros(n_moves, dtype=jnp.int32).at[info.move_idx].add(1)
            # Per-move reject-reason bucket: index into (move_idx, reason)
            # reason=0 means accepted, so accepted moves go into bucket 0.
            # Gate rejected moves: non-cell moves leave reject_reason at default
            # 0 (same as accepted), so force any rejected move with reason=0
            # into bucket 1 (energy). This preserves the invariant
            # rr[:, 0] == n_accepted_per_move without touching move kernels.
            scatter_col = jnp.where(
                info.accepted, jnp.int32(0), jnp.maximum(info.reject_reason, jnp.int32(1))
            )
            rr_counts = jnp.zeros((n_moves, 4), dtype=jnp.int32).at[
                info.move_idx, scatter_col
            ].add(1)
            return new_state, (info.accepted, n_acc, n_prop, rr_counts)

        final, (accepted_arr, n_acc_arr, n_prop_arr, rr_arr) = jax.lax.scan(
            scan_body, walker, chain_keys
        )
        # Sum over the scan axis (n_mcmc_steps)
        return (
            final,
            jnp.sum(accepted_arr),
            jnp.sum(n_acc_arr, axis=0),   # (n_moves,)
            jnp.sum(n_prop_arr, axis=0),  # (n_moves,)
            jnp.sum(rr_arr, axis=0),      # (n_moves, 4)
        )

    finals, acc_counts, chain_n_acc, chain_n_prop, chain_rr = jax.vmap(
        run_one_chain
    )(walk_batch, all_mcmc_keys)

    # 7. Scatter walked walkers back into population
    new_pop = jax.tree.map(
        lambda pop_field, walked_field: pop_field.at[walk_indices].set(walked_field),
        pop, finals,
    )

    # 8. Build new state
    new_ns_state = ns_state.set(
        population=new_pop,
        dead_energies=dead_energies,
        dead_positions=dead_positions,
        dead_volumes=dead_volumes,
        log_evidence=log_evidence,
        iteration=ns_state.iteration + 1,
        n_dead=n_dead + 1,
        rng_key=key,
    )

    total_accepted = jnp.sum(acc_counts)
    total_steps = n_walk * n_mcmc_steps
    # Aggregate per-move counters across all walked walkers (vmap axis 0).
    # chain_n_acc / chain_n_prop / chain_rr have shape (n_walk, n_moves[, 4]).
    agg_n_accepted = jnp.sum(chain_n_acc, axis=0)    # (n_moves,)
    agg_n_proposed = jnp.sum(chain_n_prop, axis=0)   # (n_moves,)
    agg_rr_counts = jnp.sum(chain_rr, axis=0)        # (n_moves, 4)

    info = {
        "emax": potential_max,
        "hmax": potential_max,  # backward compat — same as emax now
        "worst_idx": worst_idx,
        "clone_idx": clone_idx,
        "acceptance_rate": total_accepted / total_steps,
        # Chain-level per-move statistics (raw counts; let consumers compute rates).
        # Bucket convention: reject_reason_counts_per_move[:, 0] = accepted,
        #   [:, 1] = energy reject, [:, 2] = cell reject, [:, 3] = prior reject.
        # Each row sums to n_proposed_per_move for that move.
        "n_accepted_per_move": agg_n_accepted,          # (n_moves,) int32
        "n_proposed_per_move": agg_n_proposed,          # (n_moves,) int32
        "reject_reason_counts_per_move": agg_rr_counts,  # (n_moves, 4) int32
    }
    return new_ns_state, info


# ---------------------------------------------------------------------------
# run_ns — outer Python loop (NOT JIT'd)
# ---------------------------------------------------------------------------


def _ns_state_to_result_dict(ns_state: NSState) -> dict:
    """Convert NSState to backward-compatible result dict."""
    pop = ns_state.population
    ep = pop.ensemble_params if hasattr(pop, "ensemble_params") else {}
    is_npt = isinstance(ep, dict) and "pressure" in ep
    result = {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "dead_energies": ns_state.dead_energies,
        "dead_positions": ns_state.dead_positions,
        "dead_volumes": ns_state.dead_volumes if is_npt else None,
        "log_evidence": ns_state.log_evidence,
        "iteration": int(ns_state.iteration),
        "n_dead": int(ns_state.n_dead),
        "n_walkers": ns_state.n_walkers,
        "rng_key": ns_state.rng_key,
    }
    if is_npt:
        result["live_volumes"] = jax.vmap(get_volume)(pop.cell)
    else:
        result["live_volumes"] = None
    return result


def run_ns(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    cells: jnp.ndarray | None,
    init_fn: Callable,
    step_fn: Callable,
    rng_key: jax.Array,
    n_walkers: int | None = None,
    max_iterations: int = 10000,
    n_mcmc_steps: int = 20,
    n_extra: int = 0,
    convergence_threshold: float = 0.1,
    initial_step_size: float = 0.1,
    target_acceptance: float = 0.5,
    callbacks: list[Any] | None = None,
    termination_criteria: list[TerminationCriterion] | None = None,
    ensemble_params: dict | None = None,
    per_move_fns: list[Callable] | None = None,
    move_descriptors: list | None = None,
    adjust_interval: int = 0,
    adjust_n_samples: int = 50,
    adjust_max_rounds: int = 15,
    adjust_factor: float = 1.5,
    restart_state=None,
) -> dict:
    """Run a full nested sampling calculation.

    Outer Python loop calling ns_step(). Handles adaptation, overflow
    retry, termination, callbacks, logging.

    Returns a backward-compatible result dict.
    """
    if callbacks is None:
        callbacks = []

    if n_walkers is None:
        n_walkers = positions.shape[0]

    if termination_criteria is None:
        termination_criteria = [
            IterationTermination(max_iterations),
            PriorMassTermination(n_walkers, convergence_threshold),
        ]

    # Initialize NSState with batched MCState population.
    # Size step_sizes per number of moves when known; falls back to length-1
    # broadcast for legacy callers that don't pass per_move_fns / descriptors.
    if per_move_fns is not None:
        n_moves = len(per_move_fns)
    elif move_descriptors is not None:
        n_moves = len(move_descriptors)
    else:
        n_moves = 1
    ns_state = init_ns(
        init_fn, positions, types, energies, cells, rng_key,
        max_dead=max_iterations,
        step_sizes=jnp.full(n_moves, initial_step_size),
        ensemble_params=ensemble_params,
        restart_state=restart_state,
    )

    n_atoms = positions.shape[1] if positions.ndim >= 2 else None
    logger.info(
        "Starting NS run: %d walkers, %s atoms, max_iter=%d, n_mcmc=%d",
        n_walkers, n_atoms, max_iterations, n_mcmc_steps,
    )

    info_interval = max(1, max_iterations // 20)

    # JIT-compile ns_step with step_fn, n_mcmc_steps, n_extra as static args
    jit_ns_step = jax.jit(ns_step, static_argnums=(1, 2, 3))

    # Determine adaptation mode
    use_full_auto = (
        per_move_fns is not None
        and move_descriptors is not None
        and adjust_interval > 0
    )

    # Pre-compile per-move adjustment functions (once, before the loop)
    if use_full_auto:
        from jaxrens.sampling.adaptation.stepsize_handler import adjust_step_size

        jit_adjust_fns = []
        for move_idx, desc in enumerate(move_descriptors):
            # Each move gets its own JIT'd function with its static config baked in
            jit_adjust_fns.append(
                jax.jit(adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10))
            )

    # Track per-move step sizes
    pop = ns_state.population
    current_step_sizes = pop.step_sizes[0]  # (n_move_types,) — same for all walkers

    for i in range(max_iterations):
        # Step size adaptation via bisection (full_auto only; no fallback path)
        _adjust_fired = False
        per_move_rates: list = []
        per_move_counts: list = []
        per_move_n_rounds: list = []
        per_move_converged: list = []
        per_move_cap_hits: list = []
        per_move_floor_hits: list = []
        per_move_bracket_detected: list = []
        if use_full_auto and i > 0 and i % adjust_interval == 0:
            pop = ns_state.population
            emax = jnp.max(pop.energy)
            for move_idx, desc in enumerate(move_descriptors):
                rng_key, key_adjust = jax.random.split(rng_key)
                (new_ss, rate, counts,
                 n_rounds, converged, cap_hits, floor_hits, bracket_detected
                 ) = jit_adjust_fns[move_idx](
                    pop, per_move_fns[move_idx],
                    current_step_sizes[move_idx], emax, key_adjust,
                    adjust_n_samples, desc.min_rate, desc.max_rate,
                    adjust_factor, desc.step_size_max, adjust_max_rounds,
                )
                current_step_sizes = current_step_sizes.at[move_idx].set(new_ss)
                per_move_rates.append(float(rate))
                per_move_counts.append(counts)
                per_move_n_rounds.append(n_rounds)
                per_move_converged.append(converged)
                per_move_cap_hits.append(cap_hits)
                per_move_floor_hits.append(floor_hits)
                per_move_bracket_detected.append(bracket_detected)
                logger.debug(
                    "Adjusted %s: ss=%.4g rate=%.3f rounds=%d converged=%s",
                    desc.name, float(new_ss), float(rate),
                    int(n_rounds), bool(converged),
                )
            # Broadcast to all walkers
            new_ss_pop = jnp.broadcast_to(
                current_step_sizes, (n_walkers, current_step_sizes.shape[0]),
            )
            ns_state = ns_state.set(population=pop.set(step_sizes=new_ss_pop))
            _adjust_fired = True

        # Run one NS iteration (JIT'd)
        new_ns_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps, n_extra)

        # Overflow check — retry with larger neighbor list
        pop = new_ns_state.population
        if jnp.any(pop.overflow):
            new_max = int(pop.max_neighbor_count.max() * 1.25) + 1
            logger.warning(
                "Overflow at iter %d: resizing max_neighbors %d -> %d",
                i, pop.max_neighbors, new_max,
            )
            ns_state = ns_state.set(
                population=ns_state.population.set(max_neighbors=new_max),
            )
            continue

        ns_state = new_ns_state

        # Populate per-move adaptation data into info dict for callbacks.
        # Only present on iterations where full_auto just fired (adjust_interval).
        if _adjust_fired:
            info["step_sizes_per_move"] = current_step_sizes        # (n_moves,)
            info["acceptance_rates_per_move"] = jnp.array(per_move_rates)  # (n_moves,)
            info["reject_counts_per_move"] = jnp.stack(per_move_counts, axis=0)  # (n_moves, 4)
            info["adjustment_n_rounds"] = jnp.array(per_move_n_rounds, dtype=jnp.int32)  # (n_moves,)
            info["adjustment_converged"] = jnp.array(per_move_converged)        # (n_moves,)
            info["adjustment_cap_hits"] = jnp.array(per_move_cap_hits, dtype=jnp.int32)  # (n_moves,)
            info["adjustment_floor_hits"] = jnp.array(per_move_floor_hits, dtype=jnp.int32)  # (n_moves,)
            info["adjustment_bracket_detected"] = jnp.array(per_move_bracket_detected)  # (n_moves,)
            move_name_list = [d.name for d in move_descriptors]
            info["move_names"] = move_name_list
            info["move_reject_reasons"] = tuple(
                frozenset(d.reject_reasons) for d in move_descriptors
            )

        # The canonical periodic summary is printed by ProgressCallback;
        # do NOT add a duplicate logger.info call here.

        # Callbacks
        for cb in callbacks:
            if hasattr(cb, "on_iteration"):
                cb.on_iteration(i, ns_state, info)

        # Update evidence in PriorMassTermination if present
        for criterion in termination_criteria:
            if isinstance(criterion, PriorMassTermination):
                criterion.update_evidence(float(ns_state.log_evidence))

        # Termination check
        emax = float(info["hmax"])
        should_stop, stop_msg = check_any(termination_criteria, i, emax)
        if should_stop:
            logger.info(
                "Terminated at iteration %d: %s (log_evidence=%.4f)",
                i, stop_msg, float(ns_state.log_evidence),
            )
            break

    # Final evidence: add contribution from remaining live walkers
    remaining_potentials = ns_state.population.energy
    log_remaining_mass = -ns_state.iteration / n_walkers
    log_avg_likelihood = -jnp.mean(remaining_potentials)
    log_final_contribution = log_remaining_mass + log_avg_likelihood
    ns_state = ns_state.set(
        log_evidence=jnp.logaddexp(ns_state.log_evidence, log_final_contribution),
    )

    logger.info(
        "NS complete: %d dead points, log_Z=%.4f",
        int(ns_state.n_dead), float(ns_state.log_evidence),
    )

    for cb in callbacks:
        if hasattr(cb, "on_finish"):
            cb.on_finish(ns_state)

    return _ns_state_to_result_dict(ns_state)


# ---------------------------------------------------------------------------
# Multi-run parallel NS (vmap)
# ---------------------------------------------------------------------------


def init_ns_parallel(
    init_fn: Callable,
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    cells: jnp.ndarray | None,
    rng_keys: jax.Array,
    max_dead: int = 50000,
    step_sizes: jnp.ndarray | None = None,
    ensemble_params_per_run: list[dict] | None = None,
) -> NSState:
    """Create batched NSState for n_runs parallel NS runs.

    Args:
        positions: (n_runs, n_walkers, n_atoms, 3)
        types: (n_runs, n_walkers, n_atoms) or (n_atoms,)
        energies: (n_runs, n_walkers)
        cells: (n_runs, n_walkers, 3, 3) or None
        rng_keys: (n_runs,) per-run PRNG keys
        max_dead: Max dead points per run
        step_sizes: Per-move step sizes (shared across runs)
        ensemble_params_per_run: List of per-run ensemble param dicts

    Returns:
        NSState with (n_runs, ...) on all dynamic fields.
        Static fields shared: n_walkers, n_atoms, max_dead.
    """
    n_runs = positions.shape[0]
    runs = []
    for i in range(n_runs):
        ep = ensemble_params_per_run[i] if ensemble_params_per_run else None
        run_types = types if types.ndim == 1 else types[i]
        run_state = init_ns(
            init_fn,
            positions[i], run_types, energies[i],
            cells[i] if cells is not None else None,
            rng_keys[i], max_dead, step_sizes, ep,
        )
        runs.append(run_state)
    return jax.tree.map(lambda *xs: jnp.stack(xs), *runs)


def run_ns_parallel(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    cells: jnp.ndarray | None,
    init_fn: Callable,
    step_fn: Callable,
    rng_keys: jax.Array,
    n_walkers: int | None = None,
    max_iterations: int = 10000,
    n_mcmc_steps: int = 20,
    n_extra: int = 0,
    convergence_threshold: float = 0.1,
    initial_step_size: float = 0.1,
    target_acceptance: float = 0.5,
    termination_criteria: list[TerminationCriterion] | None = None,
    ensemble_params_per_run: list[dict] | None = None,
    per_move_fns: list[Callable] | None = None,
    move_descriptors: list | None = None,
    adjust_interval: int = 0,
    adjust_n_samples: int = 50,
    adjust_max_rounds: int = 15,
    adjust_factor: float = 1.5,
) -> dict:
    """Run multiple NS calculations in parallel via vmap(ns_step).

    All runs share the same step_fn, n_mcmc_steps, and n_extra.
    Each run has independent RNG and (optionally) ensemble parameters.
    When per_move_fns / move_descriptors / adjust_interval are provided,
    step sizes are adapted via vmapped adjust_step_size bisection.
    Without them, step sizes remain at initial_step_size throughout.

    Args:
        positions: (n_runs, n_walkers, n_atoms, 3)
        types: (n_atoms,) shared, or (n_runs, n_walkers, n_atoms)
        energies: (n_runs, n_walkers)
        cells: (n_runs, n_walkers, 3, 3) or None
        init_fn: MCState constructor from build_mwg().
        step_fn: MCMC step function from build_mwg().
        rng_keys: (n_runs,) per-run PRNG keys.
        n_walkers: Inferred from positions if None.
        max_iterations: Max NS iterations per run.
        n_mcmc_steps: MCMC steps per walker (static).
        n_extra: Extra walkers to walk per iteration (static).
        convergence_threshold: For PriorMassTermination.
        initial_step_size: Starting step size.
        target_acceptance: Target acceptance rate (unused directly; kept for
            interface parity with run_ns).
        termination_criteria: Optional. If None, uses defaults.
        ensemble_params_per_run: Per-run ensemble params (e.g. different pressures).
        per_move_fns: Per-move step functions for bisection adaptation.
        move_descriptors: MoveKernel descriptors carrying rate bounds + max ss.
        adjust_interval: Adapt every N iterations. 0 = no adaptation.
        adjust_n_samples: Walkers to sample per bisection trial round (static).
        adjust_max_rounds: Max bisection rounds per adjust call (static).
        adjust_factor: Multiplicative bisection factor (static).

    Returns:
        Dict with (n_runs, ...) shaped arrays on all fields.
    """
    n_runs = positions.shape[0]
    if n_walkers is None:
        n_walkers = positions.shape[1]

    if termination_criteria is None:
        termination_criteria = [
            IterationTermination(max_iterations),
            PriorMassTermination(n_walkers, convergence_threshold),
        ]

    # Determine adaptation mode
    use_full_auto = (
        per_move_fns is not None
        and move_descriptors is not None
        and adjust_interval > 0
    )

    if use_full_auto:
        n_moves = len(move_descriptors)
    elif move_descriptors is not None:
        n_moves = len(move_descriptors)
    else:
        n_moves = 1

    # Initialize batched NSState: (n_runs, ...) on all dynamic fields
    ns_states = init_ns_parallel(
        init_fn, positions, types, energies, cells, rng_keys,
        max_dead=max_iterations,
        step_sizes=jnp.full(n_moves, initial_step_size),
        ensemble_params_per_run=ensemble_params_per_run,
    )

    # Pre-compile vmapped adjust_step_size (once per move, before the loop)
    if use_full_auto:
        from jaxrens.sampling.adaptation.stepsize_handler import adjust_step_size

        # Build per-move vmapped+JIT'd functions.
        # vmap is over the n_runs dimension of (population, step_size, emax, key).
        # Static args (move_fn, n_samples, bounds, factor, max_ss, max_rounds)
        # remain outside vmap.
        jit_vmap_adjust_fns = []
        for midx, desc in enumerate(move_descriptors):
            def _make_vmap_fn(move_fn, n_samp, min_r, max_r, afac, max_ss, max_r2):
                def _per_run(pop, ss, emax, key):
                    return adjust_step_size(
                        pop, move_fn, ss, emax, key,
                        n_samp, min_r, max_r, afac, max_ss, max_r2,
                    )
                return jax.jit(jax.vmap(_per_run))
            jit_vmap_adjust_fns.append(
                _make_vmap_fn(
                    per_move_fns[midx],
                    adjust_n_samples,
                    desc.min_rate,
                    desc.max_rate,
                    adjust_factor,
                    desc.step_size_max,
                    adjust_max_rounds,
                )
            )

        # (n_runs, n_moves) — current step sizes, one per run per move
        current_step_sizes = ns_states.population.step_sizes[:, 0, :]  # (n_runs, n_moves)

    # Per-run PRNG keys used during adaptation (independent of ns_states.rng_key)
    adapt_keys = jax.vmap(lambda k: jax.random.split(k)[0])(rng_keys)  # (n_runs,)

    logger.info(
        "Starting parallel NS: %d runs, %d walkers, max_iter=%d, n_mcmc=%d, n_extra=%d",
        n_runs, n_walkers, max_iterations, n_mcmc_steps, n_extra,
    )

    info_interval = max(1, max_iterations // 20)

    # JIT-compile vmapped ns_step
    def step_all_runs(ns_states):
        return jax.vmap(
            lambda s: ns_step(s, step_fn, n_mcmc_steps, n_extra)
        )(ns_states)

    jit_step = jax.jit(step_all_runs)

    for i in range(max_iterations):
        # Step size adaptation via vmapped bisection (full_auto only)
        if use_full_auto and i > 0 and i % adjust_interval == 0:
            pop = ns_states.population
            # (n_runs,) per-run emax
            emax_per_run = jnp.max(pop.energy, axis=1)  # (n_runs,)
            # current_step_sizes: (n_runs, n_moves)
            for move_idx, desc in enumerate(move_descriptors):
                # Advance adapt_keys and derive per-run trial keys for this move.
                # split_pairs: (n_runs, 2) — first column = new carry, second = trial key
                split_pairs = jax.vmap(jax.random.split)(adapt_keys)  # (n_runs, 2)
                adapt_keys = split_pairs[:, 0]   # carry forward
                new_keys = split_pairs[:, 1]     # per-run trial keys

                ss_move = current_step_sizes[:, move_idx]  # (n_runs,)
                (new_ss_runs, rates, counts, n_rounds,
                 converged, cap_hits, floor_hits, bracket_detected
                 ) = jit_vmap_adjust_fns[move_idx](
                    pop, ss_move, emax_per_run, new_keys,
                )
                current_step_sizes = current_step_sizes.at[:, move_idx].set(new_ss_runs)
                logger.debug(
                    "Adjusted %s (all runs): ss_mean=%.4g rate_mean=%.3f",
                    desc.name, float(jnp.mean(new_ss_runs)), float(jnp.mean(rates)),
                )
            # Broadcast (n_runs, n_moves) -> (n_runs, n_walkers, n_moves)
            new_ss_pop = jnp.broadcast_to(
                current_step_sizes[:, None, :],
                (n_runs, n_walkers, n_moves),
            )
            ns_states = ns_states.set(population=pop.set(step_sizes=new_ss_pop))

        # Batched NS step (all runs in parallel)
        new_ns_states, infos = jit_step(ns_states)

        # Overflow: check ANY run. Conservative — retries all runs.
        if jnp.any(new_ns_states.population.overflow):
            new_max = int(new_ns_states.population.max_neighbor_count.max() * 1.25) + 1
            logger.warning(
                "Overflow at iter %d: resizing max_neighbors -> %d", i, new_max,
            )
            ns_states = ns_states.set(
                population=ns_states.population.set(max_neighbors=new_max),
            )
            continue

        ns_states = new_ns_states

        # Periodic log
        if i % info_interval == 0 or i == max_iterations - 1:
            for r in range(n_runs):
                logger.info(
                    "run=%d iter=%d  Emax=%.6g  acc=%.2f  log_Z=%.4f",
                    r, int(ns_states.iteration[r]),
                    float(infos["emax"][r]),
                    float(infos["acceptance_rate"][r]),
                    float(ns_states.log_evidence[r]),
                )

        # Update evidence in PriorMassTermination (use worst-case across runs)
        for criterion in termination_criteria:
            if isinstance(criterion, PriorMassTermination):
                worst_evidence = float(jnp.min(ns_states.log_evidence))
                criterion.update_evidence(worst_evidence)

        # Termination: use worst hmax across runs
        worst_hmax = float(jnp.max(infos["hmax"]))
        should_stop, stop_msg = check_any(termination_criteria, i, worst_hmax)
        if should_stop:
            logger.info("Terminated at iteration %d: %s", i, stop_msg)
            break

    # Final evidence: per-run contribution from remaining live walkers
    remaining_potentials = ns_states.population.energy  # (n_runs, n_walkers)
    log_remaining_mass = -ns_states.iteration / n_walkers  # (n_runs,)
    log_avg_likelihood = -jnp.mean(remaining_potentials, axis=-1)  # (n_runs,)
    log_final = log_remaining_mass + log_avg_likelihood
    ns_states = ns_states.set(
        log_evidence=jnp.logaddexp(ns_states.log_evidence, log_final),
    )

    for r in range(n_runs):
        logger.info(
            "Run %d complete: %d dead points, log_Z=%.4f",
            r, int(ns_states.n_dead[r]), float(ns_states.log_evidence[r]),
        )

    # Return dict with (n_runs, ...) shaped arrays
    pop = ns_states.population
    return {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "dead_energies": ns_states.dead_energies,
        "dead_positions": ns_states.dead_positions,
        "dead_volumes": ns_states.dead_volumes,
        "log_evidence": ns_states.log_evidence,
        "iteration": ns_states.iteration,
        "n_dead": ns_states.n_dead,
        "n_walkers": n_walkers,
        "n_runs": n_runs,
    }
