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
from jaxtyping import Array, Float, Int, Key

from jaxrens.base import NSCallback
from jaxrens.sampling.adaptation.manager import AdaptationManager
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun, VmapRuns
from jaxrens.sampling.inter_re_manager import InterREManager
from jaxrens.sampling.moves.replica_exchange import PressureRENSSwap, SemiGrandSwap, XRENSSwap
from jaxrens.sampling.run_loop import (
    _bump_cumulative_counters,
    _dispatch_callbacks,
    _inject_cumulative_into_info,
    _pack_adjustment_info,
    _pick_next_bucket,
    _run_loop,
)
from jaxrens.sampling.termination import (
    IterationTermination,
    PriorMassTermination,
    TerminationCriterion,
    check_any,
)
from jaxrens.state.ns import NSState
from jaxrens.utils.cell import get_volume

logger = logging.getLogger(__name__)


def _choose_starting_bucket(
    initial_max_neighbor_counts: jnp.ndarray | None,
    ladder: tuple[int, ...],
    offset: int,
) -> int:
    """Pick the smallest ladder entry that covers ``max(counts) + offset``.

    When ``initial_max_neighbor_counts`` is None (no backend, or backend
    that doesn't expose ``max_neighbors_for``), fall back to ``ladder[0]``
    — the legacy "start small, grow on overflow" path.

    When counts are available the starting bucket is chosen to accommodate
    every walker's observed count plus ``offset``, so iter 0 compiles
    against the right bucket and the overflow-retry loop only fires when
    chain dynamics push counts past the current ladder entry.
    """
    if initial_max_neighbor_counts is None:
        return ladder[0]
    peak = int(jnp.max(initial_max_neighbor_counts))
    return _pick_next_bucket(peak, 0, ladder, offset)


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
    positions: Float[Array, "K N 3"],
    types: Int[Array, "K N"] | Int[Array, "N"],
    energies: Float[Array, "K"],
    cells: Float[Array, "K 3 3"] | None,
    rng_key: Key[Array, ""],
    step_sizes: Float[Array, "n_moves"] | None = None,
    ensemble_params: dict | None = None,
    restart_state=None,
    max_neighbors: int = 0,
    max_neighbor_counts: Int[Array, "K"] | None = None,
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
        step_sizes: Per-move step size array. None = use descriptor defaults.
        ensemble_params: Ensemble parameters dict for MCState.
        restart_state: Optional RestartBundle from load_restart(). When
            provided, seeds NSState bookkeeping scalars (iteration,
            log_evidence) from the checkpoint.  Dead-point history is not
            re-seeded — the canonical record on disk (``.energies`` /
            ``.traj``) is append-only and survives across restart.

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
            max_neighbors=max_neighbors,
            max_neighbor_count_init=(
                int(max_neighbor_counts[i]) if max_neighbor_counts is not None else 0
            ),
        )
        walkers.append(w)
    population = jax.tree.map(lambda *xs: jnp.stack(xs), *walkers)

    if restart_state is None:
        log_evidence = jnp.array(-jnp.inf)
        iteration = jnp.array(0, dtype=jnp.int32)
    else:
        rs = restart_state
        log_evidence = jnp.array(rs.log_evidence)
        iteration = jnp.array(rs.iteration, dtype=jnp.int32)

    state = NSState(
        population=population,
        log_evidence=log_evidence,
        iteration=iteration,
        rng_key=rng_key,
        n_walkers=n_walkers,
        n_atoms=n_atoms,
    )

    logger.debug(
        "NS state initialized: %d walkers, energy range [%.4g, %.4g]%s",
        n_walkers, float(jnp.min(energies)), float(jnp.max(energies)),
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

    # 2. Identify dead point.  We no longer write it onto NSState — the
    # values are surfaced via the ``info`` dict instead, and the
    # per-iteration callbacks (``EnergyLogger``, ``TrajectoryCallback``)
    # persist them straight to disk.  Inside JIT, the dead point only
    # matters for the log-evidence accumulator below; the rest is pure
    # history / output.
    volumes = jax.vmap(get_volume)(pop.cell)  # (n_walkers,)
    dead_position = pop.positions[worst_idx]
    dead_volume = volumes[worst_idx]

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
            # Per-move energy evaluation counters
            n_evals = jnp.zeros(n_moves, dtype=jnp.int32).at[info.move_idx].add(
                jnp.int32(info.n_evaluations)
            )
            n_grad_evals = jnp.zeros(n_moves, dtype=jnp.int32).at[info.move_idx].add(
                jnp.int32(info.n_grad_evaluations)
            )
            return new_state, (info.accepted, n_acc, n_prop, rr_counts, n_evals, n_grad_evals)

        final, (accepted_arr, n_acc_arr, n_prop_arr, rr_arr, n_evals_arr, n_grad_evals_arr) = jax.lax.scan(
            scan_body, walker, chain_keys
        )
        # Sum over the scan axis (n_mcmc_steps)
        return (
            final,
            jnp.sum(accepted_arr),
            jnp.sum(n_acc_arr, axis=0),        # (n_moves,)
            jnp.sum(n_prop_arr, axis=0),        # (n_moves,)
            jnp.sum(rr_arr, axis=0),            # (n_moves, 4)
            jnp.sum(n_evals_arr, axis=0),       # (n_moves,)
            jnp.sum(n_grad_evals_arr, axis=0),  # (n_moves,)
        )

    finals, acc_counts, chain_n_acc, chain_n_prop, chain_rr, chain_n_evals, chain_n_grad_evals = jax.vmap(
        run_one_chain
    )(walk_batch, all_mcmc_keys)

    # 7. Scatter walked walkers back into population
    new_pop = jax.tree.map(
        lambda pop_field, walked_field: pop_field.at[walk_indices].set(walked_field),
        pop, finals,
    )

    # 8. Build new state — algorithm fields only; dead-point history is
    # handled host-side by ``_run_loop`` via the info dict below.
    new_ns_state = ns_state.set(
        population=new_pop,
        log_evidence=log_evidence,
        iteration=ns_state.iteration + 1,
        rng_key=key,
    )

    total_accepted = jnp.sum(acc_counts)
    total_steps = n_walk * n_mcmc_steps
    # Aggregate per-move counters across all walked walkers (vmap axis 0).
    # chain_n_acc / chain_n_prop / chain_rr have shape (n_walk, n_moves[, 4]).
    agg_n_accepted = jnp.sum(chain_n_acc, axis=0)          # (n_moves,)
    agg_n_proposed = jnp.sum(chain_n_prop, axis=0)         # (n_moves,)
    agg_rr_counts = jnp.sum(chain_rr, axis=0)              # (n_moves, 4)
    agg_n_evals = jnp.sum(chain_n_evals, axis=0)           # (n_moves,)
    agg_n_grad_evals = jnp.sum(chain_n_grad_evals, axis=0) # (n_moves,)

    info = {
        "emax": potential_max,
        "hmax": potential_max,  # backward compat — same as emax now
        "worst_idx": worst_idx,
        "clone_idx": clone_idx,
        "acceptance_rate": total_accepted / total_steps,
        # Per-iteration culled-walker (the "dead point").  Read by
        # ``EnergyLogger`` / ``TrajectoryCallback`` for streaming-to-disk;
        # not stored anywhere on NSState.
        "dead_energy": potential_max,
        "dead_position": dead_position,
        "dead_volume": dead_volume,
        # Chain-level per-move statistics (raw counts; let consumers compute rates).
        # Bucket convention: reject_reason_counts_per_move[:, 0] = accepted,
        #   [:, 1] = energy reject, [:, 2] = cell reject, [:, 3] = prior reject.
        # Each row sums to n_proposed_per_move for that move.
        "n_accepted_per_move": agg_n_accepted,               # (n_moves,) int32
        "n_proposed_per_move": agg_n_proposed,               # (n_moves,) int32
        "reject_reason_counts_per_move": agg_rr_counts,       # (n_moves, 4) int32
        # Energy evaluation counters (summed over walkers and chain steps).
        "n_evaluations_per_move": agg_n_evals,               # (n_moves,) int32
        "n_grad_evaluations_per_move": agg_n_grad_evals,     # (n_moves,) int32
    }
    return new_ns_state, info


# ---------------------------------------------------------------------------
# run_ns — thin wrapper around _run_loop
# ---------------------------------------------------------------------------


def _ns_state_to_result_dict(ns_state: NSState, n_dead: int | jnp.ndarray = 0) -> dict:
    """Convert NSState to a result dict (live-walker state + scalars only).

    Dead-point history is *not* in the return value — it's persisted to disk
    by the callbacks (``EnergyLogger``, ``TrajectoryCallback``,
    ``CheckpointCallback``) during the run.  Postprocessing reads it from
    disk via ``postprocess.Monitor.from_directory(...)`` rather than from
    the result dict, so carrying the full arrays in the return would
    duplicate persisted data and waste memory.

    ``n_dead`` is taken from ``len(history)`` at call time and forwarded
    here as a small scalar / batched scalar for log lines and for
    interactive callers that just want the count.
    """
    pop = ns_state.population
    ep = pop.ensemble_params if hasattr(pop, "ensemble_params") else {}
    is_npt = isinstance(ep, dict) and "pressure" in ep
    result = {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "log_evidence": ns_state.log_evidence,
        "iteration": int(ns_state.iteration),
        "n_dead": n_dead,
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
    max_iterations: int | None = None,
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
    max_neighbors_list: tuple[int, ...] | list[int] = (30, 35, 40, 45, 50),
    max_neighbors_offset: int = 5,
    initial_max_neighbor_counts: jnp.ndarray | None = None,
) -> dict:
    """Run a full nested sampling calculation.

    Thin wrapper: validates args, initialises NSState, constructs
    descriptor + AdaptationManager, delegates to ``_run_loop``, and
    packages the result dict (live state + scalars only — dead-point
    history is persisted to disk by the per-iteration callbacks, not
    returned).

    ``max_iterations`` is optional; when ``None`` the loop runs until
    another termination criterion fires (typically ``PriorMassTermination``).
    Pass an explicit integer to add an ``IterationTermination`` to the
    default criteria.

    Returns a backward-compatible result dict.
    """
    if callbacks is None:
        callbacks = []
    if n_walkers is None:
        n_walkers = positions.shape[0]
    if termination_criteria is None:
        termination_criteria = [
            PriorMassTermination(n_walkers, convergence_threshold),
        ]
        if max_iterations is not None:
            termination_criteria.append(IterationTermination(max_iterations))
    ladder = tuple(int(x) for x in max_neighbors_list)
    if not ladder:
        raise ValueError("max_neighbors_list must be non-empty.")
    starting_bucket = _choose_starting_bucket(
        initial_max_neighbor_counts, ladder, max_neighbors_offset,
    )

    if per_move_fns is not None:
        n_moves = len(per_move_fns)
    elif move_descriptors is not None:
        n_moves = len(move_descriptors)
    else:
        n_moves = 1

    ns_state = init_ns(
        init_fn, positions, types, energies, cells, rng_key,
        step_sizes=jnp.full(n_moves, initial_step_size),
        ensemble_params=ensemble_params,
        restart_state=restart_state,
        max_neighbors=starting_bucket,
        max_neighbor_counts=initial_max_neighbor_counts,
    )

    n_atoms = positions.shape[1] if positions.ndim >= 2 else None

    logger.info(
        "Starting NS run: %d walkers, %s atoms, max_iter=%s, n_mcmc=%d",
        n_walkers, n_atoms, max_iterations, n_mcmc_steps,
    )

    batcher = SingleRun()
    adapt_mgr = AdaptationManager(
        move_descriptors=move_descriptors or [],
        per_move_fns=per_move_fns,
        batcher=batcher,
        adjust_n_samples=adjust_n_samples,
        adjust_factor=adjust_factor,
        adjust_max_rounds=adjust_max_rounds,
        adjust_interval=adjust_interval,
    )

    ns_state, rng_key, _cumulative = _run_loop(
        batcher=batcher,
        adapt_mgr=adapt_mgr,
        ns_state=ns_state,
        step_fn=step_fn,
        n_mcmc_steps=n_mcmc_steps,
        n_extra=n_extra,
        termination_criteria=termination_criteria,
        callbacks=callbacks,
        n_moves=n_moves,
        move_descriptors=move_descriptors,
        rng_key=rng_key,
        info_interval=max(1, (max_iterations or 1000) // 20),
        max_neighbors_list=ladder,
        max_neighbors_offset=int(max_neighbors_offset),
    )

    # Final evidence: add contribution from remaining live walkers
    remaining_potentials = ns_state.population.energy
    log_remaining_mass = -ns_state.iteration / n_walkers
    log_avg_likelihood = -jnp.mean(remaining_potentials)
    ns_state = ns_state.set(
        log_evidence=jnp.logaddexp(
            ns_state.log_evidence,
            log_remaining_mass + log_avg_likelihood,
        ),
    )

    logger.info(
        "NS complete: %d dead points, log_Z=%.4f",
        int(ns_state.iteration), float(ns_state.log_evidence),
    )

    for cb in callbacks:
        if hasattr(cb, "on_finish"):
            cb.on_finish(ns_state)

    return _ns_state_to_result_dict(ns_state, n_dead=int(ns_state.iteration))


# ---------------------------------------------------------------------------
# Multi-run parallel NS (vmap)
# ---------------------------------------------------------------------------


def init_ns_parallel(
    init_fn: Callable,
    positions: Float[Array, "R K N 3"],
    types: Int[Array, "R K N"] | Int[Array, "N"],
    energies: Float[Array, "R K"],
    cells: Float[Array, "R K 3 3"] | None,
    rng_keys: Key[Array, "R"],
    step_sizes: Float[Array, "n_moves"] | None = None,
    ensemble_params_per_run: list[dict] | None = None,
    restart_states: list | None = None,
    max_neighbors: int = 0,
    max_neighbor_counts: Int[Array, "R K"] | None = None,
) -> NSState:
    """Create batched NSState for n_runs parallel NS runs.

    Args:
        positions: (n_runs, n_walkers, n_atoms, 3)
        types: (n_runs, n_walkers, n_atoms) or (n_atoms,)
        energies: (n_runs, n_walkers)
        cells: (n_runs, n_walkers, 3, 3) or None
        rng_keys: (n_runs,) per-run PRNG keys
        step_sizes: Per-move step sizes (shared across runs)
        ensemble_params_per_run: List of per-run ensemble param dicts
        restart_states: Optional list of ``RestartBundle`` objects, one per run.
            When provided, ``init_ns`` for run *i* is seeded from
            ``restart_states[i]`` (iteration counter and log-evidence from
            the checkpoint).  Dead-point history is not re-seeded — the
            canonical disk record (``.energies`` / ``.traj``) is append-only
            and survives across restart.  Pass ``None`` for any run that
            should start fresh.  ``len(restart_states)`` must equal
            ``n_runs`` when provided.

    Returns:
        NSState with (n_runs, ...) on all dynamic fields.
        Static fields shared: n_walkers, n_atoms.
    """
    n_runs = positions.shape[0]
    if restart_states is not None and len(restart_states) != n_runs:
        raise ValueError(
            f"restart_states length ({len(restart_states)}) must equal "
            f"n_runs ({n_runs})"
        )
    runs = []
    for i in range(n_runs):
        ep = ensemble_params_per_run[i] if ensemble_params_per_run else None
        run_types = types if types.ndim == 1 else types[i]
        rs = restart_states[i] if restart_states is not None else None
        run_state = init_ns(
            init_fn,
            positions[i], run_types, energies[i],
            cells[i] if cells is not None else None,
            rng_keys[i], step_sizes, ep,
            restart_state=rs,
            max_neighbors=max_neighbors,
            max_neighbor_counts=(
                max_neighbor_counts[i] if max_neighbor_counts is not None else None
            ),
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
    restart_states: list | None = None,
    inter_re_config=None,
    backend=None,
    callbacks: list | None = None,
    max_neighbors_list: tuple[int, ...] | list[int] = (30, 35, 40, 45, 50),
    max_neighbors_offset: int = 5,
    initial_max_neighbor_counts: jnp.ndarray | None = None,
) -> dict:
    """Run multiple NS calculations in parallel via vmap(ns_step).

    Thin wrapper: validates args, initialises batched NSState, constructs
    VmapRuns descriptor + AdaptationManager, delegates to ``_run_loop``,
    and packages the result dict.

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
        restart_states: Optional list of ``RestartBundle`` objects, one per run.
            When provided, run *i* resumes from the checkpoint at
            ``restart_states[i]`` — dead-point history, iteration counter, and
            log-evidence are seeded from the bundle.  Pass ``None`` entries for
            any run that should start fresh.  ``len(restart_states)`` must equal
            ``n_runs``.

    Returns:
        Dict with (n_runs, ...) shaped arrays on all fields.
    """
    n_runs = positions.shape[0]
    if n_walkers is None:
        n_walkers = positions.shape[1]
    if termination_criteria is None:
        termination_criteria = [
            PriorMassTermination(n_walkers, convergence_threshold),
        ]
        if max_iterations is not None:
            termination_criteria.append(IterationTermination(max_iterations))
    if move_descriptors is not None:
        n_moves = len(move_descriptors)
    else:
        n_moves = 1

    ladder = tuple(int(x) for x in max_neighbors_list)
    if not ladder:
        raise ValueError("max_neighbors_list must be non-empty.")
    starting_bucket = _choose_starting_bucket(
        initial_max_neighbor_counts, ladder, max_neighbors_offset,
    )

    ns_states = init_ns_parallel(
        init_fn, positions, types, energies, cells, rng_keys,
        step_sizes=jnp.full(n_moves, initial_step_size),
        ensemble_params_per_run=ensemble_params_per_run,
        restart_states=restart_states,
        max_neighbors=starting_bucket,
        max_neighbor_counts=initial_max_neighbor_counts,
    )

    logger.info(
        "Starting parallel NS: %d runs, %d walkers, max_iter=%s, n_mcmc=%d, n_extra=%d",
        n_runs, n_walkers, max_iterations, n_mcmc_steps, n_extra,
    )

    batcher = VmapRuns(n_runs=n_runs)
    adapt_mgr = AdaptationManager(
        move_descriptors=move_descriptors or [],
        per_move_fns=per_move_fns,
        batcher=batcher,
        adjust_n_samples=adjust_n_samples,
        adjust_factor=adjust_factor,
        adjust_max_rounds=adjust_max_rounds,
        adjust_interval=adjust_interval,
    )

    # Construct InterREManager when inter_re_config is provided.
    inter_re_mgr = None
    if inter_re_config is not None:
        from jaxrens.state.config import InterREConfig
        cfg: InterREConfig = inter_re_config
        if cfg.flavor == "pressure":
            swap_kernel = PressureRENSSwap()
        elif cfg.flavor == "xrens":
            if cfg.composition_targets is None:
                raise ValueError(
                    "inter_re flavor 'xrens' requires composition_targets in "
                    "InterREConfig. Each run must have a target composition."
                )
            n_species = len(cfg.composition_targets[0])
            swap_kernel = XRENSSwap(n_species=n_species)
            # Inject target_composition into ensemble_params for each run so
            # InterREManager can extract them from ns_state.population.ensemble_params.
            # Store as a JAX array (not a plain list) so that pytree stacking across
            # walkers produces shape (n_walkers, n_species) rather than (n_species, n_walkers).
            if ensemble_params_per_run is None:
                ensemble_params_per_run = [
                    {"target_composition": jnp.array(cfg.composition_targets[i], dtype=jnp.int32)}
                    for i in range(n_runs)
                ]
            else:
                ensemble_params_per_run = [
                    dict(ensemble_params_per_run[i],
                         target_composition=jnp.array(cfg.composition_targets[i], dtype=jnp.int32))
                    for i in range(n_runs)
                ]
            # Re-initialize with updated ensemble_params_per_run now that we've
            # injected target_composition. Re-run init_ns_parallel.
            ns_states = init_ns_parallel(
                init_fn, positions, types, energies, cells, rng_keys,
                step_sizes=jnp.full(n_moves, initial_step_size),
                ensemble_params_per_run=ensemble_params_per_run,
                restart_states=restart_states,
                max_neighbors=starting_bucket,
                max_neighbor_counts=initial_max_neighbor_counts,
            )
        elif cfg.flavor == "semi_grand":
            if cfg.chemical_potentials is None:
                raise ValueError(
                    "inter_re flavor 'semi_grand' requires chemical_potentials in "
                    "InterREConfig. Each run must have a per-species μ vector."
                )
            n_species = len(cfg.chemical_potentials[0])
            swap_kernel = SemiGrandSwap(n_species=n_species)
            # Inject chemical_potentials into ensemble_params for each run so
            # InterREManager can extract them from ns_state.population.ensemble_params.
            if ensemble_params_per_run is None:
                ensemble_params_per_run = [
                    {"chemical_potentials": jnp.array(cfg.chemical_potentials[i], dtype=jnp.float32)}
                    for i in range(n_runs)
                ]
            else:
                ensemble_params_per_run = [
                    dict(ensemble_params_per_run[i],
                         chemical_potentials=jnp.array(cfg.chemical_potentials[i], dtype=jnp.float32))
                    for i in range(n_runs)
                ]
            # Re-initialize with updated ensemble_params_per_run.
            ns_states = init_ns_parallel(
                init_fn, positions, types, energies, cells, rng_keys,
                step_sizes=jnp.full(n_moves, initial_step_size),
                ensemble_params_per_run=ensemble_params_per_run,
                restart_states=restart_states,
                max_neighbors=starting_bucket,
                max_neighbor_counts=initial_max_neighbor_counts,
            )
        else:
            raise NotImplementedError(
                f"inter_re flavor {cfg.flavor!r} is not yet implemented. "
                f"Supported flavors: 'pressure', 'xrens', 'semi_grand'."
            )
        inter_re_mgr = InterREManager(
            swap_kernel=swap_kernel,
            batcher=batcher,
            backend=backend,
            every=cfg.every,
            n_swap_cycles=cfg.n_swap_cycles,
        )

    # Per-run PRNG keys used during adaptation (independent of ns_states.rng_key).
    # _run_loop will consume and advance these via adapt_mgr.
    adapt_keys = jax.vmap(lambda k: jax.random.split(k)[0])(rng_keys)  # (n_runs,)

    ns_states, adapt_keys, _cumulative = _run_loop(
        batcher=batcher,
        adapt_mgr=adapt_mgr,
        ns_state=ns_states,
        step_fn=step_fn,
        n_mcmc_steps=n_mcmc_steps,
        n_extra=n_extra,
        termination_criteria=termination_criteria,
        callbacks=callbacks or [],
        n_moves=n_moves,
        move_descriptors=move_descriptors,
        rng_key=adapt_keys,
        info_interval=max(1, (max_iterations or 1000) // 20),
        inter_re_mgr=inter_re_mgr,
        max_neighbors_list=ladder,
        max_neighbors_offset=int(max_neighbors_offset),
    )

    # Final evidence: per-run contribution from remaining live walkers
    remaining_potentials = ns_states.population.energy  # (n_runs, n_walkers)
    log_remaining_mass = -ns_states.iteration / n_walkers  # (n_runs,)
    log_avg_likelihood = -jnp.mean(remaining_potentials, axis=-1)  # (n_runs,)
    ns_states = ns_states.set(
        log_evidence=jnp.logaddexp(
            ns_states.log_evidence,
            log_remaining_mass + log_avg_likelihood,
        ),
    )

    for r in range(n_runs):
        logger.info(
            "Run %d complete: %d dead points, log_Z=%.4f",
            r, int(ns_states.iteration[r]), float(ns_states.log_evidence[r]),
        )

    # Dead-point history is persisted to disk by the callbacks; not in
    # the return.  ``n_dead`` ≡ ``iteration`` (both increment per ns_step).
    pop = ns_states.population
    return {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "log_evidence": ns_states.log_evidence,
        "iteration": ns_states.iteration,
        "n_dead": ns_states.iteration,
        "n_walkers": n_walkers,
        "n_runs": n_runs,
    }


# ---------------------------------------------------------------------------
# Multi-GPU NS: pmap(vmap(ns_step))
# ---------------------------------------------------------------------------


def init_ns_multi_gpu(
    init_fn: Callable,
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    cells: jnp.ndarray | None,
    rng_keys: jax.Array,
    n_gpu: int,
    n_per_gpu: int,
    step_sizes: jnp.ndarray | None = None,
    ensemble_params_per_run: list[dict] | None = None,
    restart_states: list[list] | None = None,
    max_neighbors: int = 0,
    max_neighbor_counts: jnp.ndarray | None = None,
) -> NSState:
    """Initialize a ``(G, P, ...)``-shaped NSState for pmap(vmap) execution.

    Delegates to ``init_ns_parallel`` with ``n_runs = G*P``, then reshapes
    all dynamic fields from ``(G*P, ...)`` to ``(G, P, ...)``.

    Args:
        init_fn: MCState constructor from ``build_mwg()``.
        positions: Shape ``(G*P, n_walkers, n_atoms, 3)`` or ``(G, P, n_walkers, n_atoms, 3)``.
            If 5-D, interpreted as already having the ``(G, P)`` prefix.
        types: Shape ``(n_atoms,)`` shared, or ``(G*P, n_walkers, n_atoms)``.
        energies: Shape ``(G*P, n_walkers)`` or ``(G, P, n_walkers)``.
        cells: Shape ``(G*P, n_walkers, 3, 3)`` or ``(G, P, n_walkers, 3, 3)`` or None.
        rng_keys: Shape ``(G*P,)`` or ``(G, P)`` — one key per run.
        n_gpu: Number of GPU devices (G).
        n_per_gpu: Number of NS runs per GPU (P).
        step_sizes: Per-move step sizes (shared across all runs).
        ensemble_params_per_run: Flat list of ``G*P`` dicts, or ``None``.
        restart_states: ``(G, P)`` nested list of ``RestartBundle | None``, or
            ``None`` for a fresh start on all runs.  Mixed restart (some runs
            fresh, some from checkpoint) is supported.  Dead-point history
            is not re-seeded — the disk record (``.energies`` / ``.traj``)
            is append-only and survives across restart.  Only the
            bookkeeping scalars (iteration, log_evidence) are seeded into
            NSState here.

    Returns:
        NSState with shape ``(G, P, ...)`` on all dynamic fields.
    """
    n_total = n_gpu * n_per_gpu

    # Flatten (G, P, ...) -> (G*P, ...) if inputs arrive in 2D-prefix form.
    # We detect 2D-prefix by checking if arr.shape[:2] == (n_gpu, n_per_gpu).
    def _flatten_leading(arr):
        """If arr has a (G, P) leading prefix, merge to (G*P)."""
        if arr is None:
            return arr
        if arr.ndim >= 2 and arr.shape[0] == n_gpu and arr.shape[1] == n_per_gpu:
            return arr.reshape((n_total,) + arr.shape[2:])
        return arr

    positions_flat = _flatten_leading(positions)   # (G*P, K, A, 3)
    energies_flat = _flatten_leading(energies)      # (G*P, K)
    cells_flat = _flatten_leading(cells) if cells is not None else None
    mnc_flat = (
        _flatten_leading(max_neighbor_counts)
        if max_neighbor_counts is not None else None
    )
    rng_keys_flat = rng_keys.reshape(n_total) if rng_keys.ndim == 2 else rng_keys

    # Flatten restart_states from (G, P) nested list to flat list of G*P.
    rs_flat: list | None = None
    if restart_states is not None:
        rs_flat = []
        for gpu_list in restart_states:
            rs_flat.extend(gpu_list)

    # Build flat (G*P, ...) NSState via init_ns_parallel.
    flat_states = init_ns_parallel(
        init_fn, positions_flat, types, energies_flat, cells_flat,
        rng_keys_flat, step_sizes,
        ensemble_params_per_run=ensemble_params_per_run,
        restart_states=rs_flat,
        max_neighbors=max_neighbors,
        max_neighbor_counts=mnc_flat,
    )

    # Reshape all dynamic fields from (G*P, ...) to (G, P, ...) and explicitly
    # shard along the GPU axis. The explicit shard is load-bearing for the
    # post-burn-in path: when ``positions`` / ``energies`` / ``cells`` arrive
    # already pmap-sharded (``NamedSharding(spec=P('gpu',))``),
    # ``init_ns_parallel``'s per-run ``positions[i]`` + ``jnp.stack`` produces
    # replicated (``spec=P()``) leaves, which the downstream ``jit_ns_step``
    # pmap rejects. ``jax.device_put`` with the gpu-axis sharding repairs the
    # replicated leaves and is a no-op when the data is already correctly
    # sharded or uncommitted.
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    gpu_sharding = NamedSharding(
        Mesh(jax.local_devices()[:n_gpu], ("gpu",)),
        PartitionSpec("gpu"),
    )

    def _reshape_and_shard(x):
        if x is None:
            return x
        arr = jnp.asarray(x)
        arr = arr.reshape((n_gpu, n_per_gpu) + arr.shape[1:])
        return jax.device_put(arr, gpu_sharding)

    return jax.tree.map(_reshape_and_shard, flat_states)


def run_ns_multi_gpu(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    cells: jnp.ndarray | None,
    init_fn: Callable,
    step_fn: Callable,
    rng_keys: jax.Array,
    n_gpu: int,
    n_per_gpu: int,
    n_walkers: int | None = None,
    max_iterations: int = 10000,
    n_mcmc_steps: int = 20,
    n_extra: int = 0,
    convergence_threshold: float = 0.1,
    initial_step_size: float = 0.1,
    target_acceptance: float = 0.5,
    callbacks: list[Any] | None = None,
    termination_criteria: list[TerminationCriterion] | None = None,
    ensemble_params_per_run: list[dict] | None = None,
    per_move_fns: list[Callable] | None = None,
    move_descriptors: list | None = None,
    adjust_interval: int = 0,
    adjust_n_samples: int = 50,
    adjust_max_rounds: int = 15,
    adjust_factor: float = 1.5,
    restart_states: list[list] | None = None,
    inter_re_config=None,
    backend=None,
    max_neighbors_list: tuple[int, ...] | list[int] = (30, 35, 40, 45, 50),
    max_neighbors_offset: int = 5,
    initial_max_neighbor_counts: jnp.ndarray | None = None,
    batcher: PmapVmapRuns | None = None,
) -> dict:
    """Run NS with ``pmap(vmap(ns_step))`` dispatch across G GPUs × P runs each.

    Shape convention: ``(G, P, ...)`` where G=n_gpu, P=n_per_gpu.

    Args:
        positions: ``(G*P, n_walkers, n_atoms, 3)`` or ``(G, P, n_walkers, n_atoms, 3)``.
        types: ``(n_atoms,)`` shared, or ``(G*P, n_walkers, n_atoms)``.
        energies: ``(G*P, n_walkers)`` or ``(G, P, n_walkers)``.
        cells: ``(G*P, n_walkers, 3, 3)`` / ``(G, P, n_walkers, 3, 3)`` or None.
        init_fn: MCState constructor from ``build_mwg()``.
        step_fn: MCMC step function from ``build_mwg()``.
        rng_keys: ``(G*P,)`` or ``(G, P)`` per-run PRNG keys.
        n_gpu: Number of GPU devices to use (G).  Must satisfy
            ``n_gpu >= 1`` and ``n_gpu <= len(jax.devices())``.
        n_per_gpu: Number of independent NS runs per GPU (P).
            Must be ``>= 1``.
        n_walkers: Inferred from positions if None.
        max_iterations: Max NS iterations per run.
        n_mcmc_steps: MCMC steps per walker (static).
        n_extra: Extra walkers to walk per iteration (static).
        convergence_threshold: For ``PriorMassTermination``.
        initial_step_size: Starting step size.
        target_acceptance: Kept for interface parity; unused directly.
        callbacks: List of callback objects with optional ``on_iteration`` /
            ``on_finish`` methods.  Callbacks receive the ``(G, P, ...)``-shaped
            ``NSState`` directly.  Each ``info`` dict contains
            ``info["_batcher"] = PmapVmapRuns(n_gpu, n_per_gpu)`` so callbacks
            can identify the batch shape via ``info["_batcher"].is_batched``.
            ``log_evidence`` in the associated ``NSState`` has shape ``(G, P)``.
        termination_criteria: Optional.  Defaults to
            ``[IterationTermination(max_iterations),
               PriorMassTermination(n_walkers, convergence_threshold)]``.
        ensemble_params_per_run: Flat list of ``G*P`` dicts, or ``None``.
        per_move_fns: Per-move step functions for bisection adaptation.
        move_descriptors: ``MoveKernel`` descriptors carrying rate bounds + max ss.
        adjust_interval: Adapt every N iters.  0 = no adaptation.
        adjust_n_samples: Walkers sampled per bisection trial round (static).
        adjust_max_rounds: Max bisection rounds per adjust call (static).
        adjust_factor: Multiplicative bisection factor (static).
        restart_states: ``(G, P)`` nested list of ``RestartBundle | None``.
            Pass ``None`` (default) for a fresh start on all runs.
            Mixed restart (some checkpointed, some fresh) is supported
            within a single call.

    Returns:
        Dict with shapes ``(G, P, ...)`` on all array fields:
        ``log_evidence``, ``n_dead``, ``iteration`` → ``(G, P)``;
        ``dead_energies`` → ``(G, P, max_dead)``;
        ``positions`` → ``(G, P, n_walkers, n_atoms, 3)``; etc.

    Raises:
        ValueError: If ``n_gpu < 1``, ``n_per_gpu < 1``, or
            ``n_gpu > len(jax.devices())``.
    """
    if callbacks is None:
        callbacks = []

    if n_gpu < 1:
        raise ValueError(f"n_gpu must be >= 1, got {n_gpu}")
    if n_per_gpu < 1:
        raise ValueError(f"n_per_gpu must be >= 1, got {n_per_gpu}")
    n_available = len(jax.devices())
    if n_gpu > n_available:
        raise ValueError(
            f"n_gpu={n_gpu} exceeds available devices={n_available}. "
            f"Available: {jax.devices()}"
        )

    n_total = n_gpu * n_per_gpu

    # Infer n_walkers from the position array.
    # Positions may be (G*P, K, A, 3) or (G, P, K, A, 3).
    if n_walkers is None:
        if positions.ndim == 5:
            # (G, P, K, A, 3) form
            n_walkers = positions.shape[2]
        else:
            # (G*P, K, A, 3) form
            n_walkers = positions.shape[1]

    if termination_criteria is None:
        termination_criteria = [
            PriorMassTermination(n_walkers, convergence_threshold),
        ]
        if max_iterations is not None:
            termination_criteria.append(IterationTermination(max_iterations))
    if move_descriptors is not None:
        n_moves = len(move_descriptors)
    elif per_move_fns is not None:
        n_moves = len(per_move_fns)
    else:
        n_moves = 1

    # Flatten rng_keys to (G*P,) for init.
    rng_keys_flat = rng_keys.reshape(n_total) if rng_keys.ndim == 2 else rng_keys

    ladder = tuple(int(x) for x in max_neighbors_list)
    if not ladder:
        raise ValueError("max_neighbors_list must be non-empty.")
    logger.debug(
        "[stage] run_ns_multi_gpu: choose_starting_bucket "
        "(initial_max_neighbor_counts shape=%s, ladder=%s, offset=%d)",
        None if initial_max_neighbor_counts is None
        else initial_max_neighbor_counts.shape,
        ladder, max_neighbors_offset,
    )
    starting_bucket = _choose_starting_bucket(
        initial_max_neighbor_counts, ladder, max_neighbors_offset,
    )
    logger.debug(
        "[stage] run_ns_multi_gpu: starting_bucket=%d", starting_bucket,
    )

    # Initialize (G, P, ...) state via init_ns_multi_gpu.
    logger.debug("[stage] run_ns_multi_gpu: init_ns_multi_gpu — starting")
    ns_states = init_ns_multi_gpu(
        init_fn, positions, types, energies, cells,
        rng_keys_flat, n_gpu, n_per_gpu,
        step_sizes=jnp.full(n_moves, initial_step_size),
        ensemble_params_per_run=ensemble_params_per_run,
        restart_states=restart_states,
        max_neighbors=starting_bucket,
        max_neighbor_counts=initial_max_neighbor_counts,
    )
    if logger.isEnabledFor(logging.DEBUG):
        try:
            jax.block_until_ready(ns_states.population.positions)
        except Exception:
            pass
        logger.debug("[stage] run_ns_multi_gpu: init_ns_multi_gpu — done")

    logger.info(
        "Starting multi-GPU NS: n_gpu=%d, n_per_gpu=%d (%d total runs), "
        "n_walkers=%d, max_iter=%s, n_mcmc=%d, n_extra=%d",
        n_gpu, n_per_gpu, n_total, n_walkers, max_iterations, n_mcmc_steps, n_extra,
    )

    if batcher is None:
        batcher = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)
    elif batcher.n_gpu != n_gpu or batcher.n_per_gpu != n_per_gpu:
        raise ValueError(
            f"run_ns_multi_gpu: batcher topology ({batcher.n_gpu}, "
            f"{batcher.n_per_gpu}) disagrees with explicit n_gpu={n_gpu}, "
            f"n_per_gpu={n_per_gpu}. Pass either form, not both with "
            f"conflicting values."
        )
    adapt_mgr = AdaptationManager(
        move_descriptors=move_descriptors or [],
        per_move_fns=per_move_fns,
        batcher=batcher,
        adjust_n_samples=adjust_n_samples,
        adjust_factor=adjust_factor,
        adjust_max_rounds=adjust_max_rounds,
        adjust_interval=adjust_interval,
    )

    # Construct InterREManager when inter_re_config is provided.
    inter_re_mgr = None
    if inter_re_config is not None:
        from jaxrens.state.config import InterREConfig
        cfg: InterREConfig = inter_re_config
        if cfg.flavor == "pressure":
            swap_kernel_mg = PressureRENSSwap()
        elif cfg.flavor == "xrens":
            if cfg.composition_targets is None:
                raise ValueError(
                    "inter_re flavor 'xrens' requires composition_targets in "
                    "InterREConfig."
                )
            n_species_mg = len(cfg.composition_targets[0])
            swap_kernel_mg = XRENSSwap(n_species=n_species_mg)
            # Inject target_composition into ensemble_params for each run.
            # Store as a JAX array to ensure correct pytree stacking across walkers.
            n_total_runs = n_gpu * n_per_gpu
            if ensemble_params_per_run is None:
                ensemble_params_per_run = [
                    {"target_composition": jnp.array(cfg.composition_targets[i], dtype=jnp.int32)}
                    for i in range(n_total_runs)
                ]
            else:
                ensemble_params_per_run = [
                    dict(ensemble_params_per_run[i],
                         target_composition=jnp.array(cfg.composition_targets[i], dtype=jnp.int32))
                    for i in range(n_total_runs)
                ]
            # Re-initialize with updated ensemble_params_per_run.
            ns_states = init_ns_multi_gpu(
                init_fn, positions, types, energies, cells,
                rng_keys_flat, n_gpu, n_per_gpu,
                step_sizes=jnp.full(n_moves, initial_step_size),
                ensemble_params_per_run=ensemble_params_per_run,
                restart_states=restart_states,
                max_neighbors=starting_bucket,
                max_neighbor_counts=initial_max_neighbor_counts,
            )
        elif cfg.flavor == "semi_grand":
            if cfg.chemical_potentials is None:
                raise ValueError(
                    "inter_re flavor 'semi_grand' requires chemical_potentials in "
                    "InterREConfig."
                )
            n_species_mg = len(cfg.chemical_potentials[0])
            swap_kernel_mg = SemiGrandSwap(n_species=n_species_mg)
            n_total_runs = n_gpu * n_per_gpu
            if ensemble_params_per_run is None:
                ensemble_params_per_run = [
                    {"chemical_potentials": jnp.array(cfg.chemical_potentials[i], dtype=jnp.float32)}
                    for i in range(n_total_runs)
                ]
            else:
                ensemble_params_per_run = [
                    dict(ensemble_params_per_run[i],
                         chemical_potentials=jnp.array(cfg.chemical_potentials[i], dtype=jnp.float32))
                    for i in range(n_total_runs)
                ]
            ns_states = init_ns_multi_gpu(
                init_fn, positions, types, energies, cells,
                rng_keys_flat, n_gpu, n_per_gpu,
                step_sizes=jnp.full(n_moves, initial_step_size),
                ensemble_params_per_run=ensemble_params_per_run,
                restart_states=restart_states,
                max_neighbors=starting_bucket,
                max_neighbor_counts=initial_max_neighbor_counts,
            )
        else:
            raise NotImplementedError(
                f"inter_re flavor {cfg.flavor!r} is not yet implemented. "
                f"Supported flavors: 'pressure', 'xrens', 'semi_grand'."
            )
        inter_re_mgr = InterREManager(
            swap_kernel=swap_kernel_mg,
            batcher=batcher,
            backend=backend,
            every=cfg.every,
            n_swap_cycles=cfg.n_swap_cycles,
        )

    # Per-run PRNG keys for adaptation (shape (G, P)).
    # Derive from rng_keys_flat by splitting, then reshape to (G, P).
    adapt_keys_flat = jax.vmap(lambda k: jax.random.split(k)[0])(rng_keys_flat)
    adapt_keys = adapt_keys_flat.reshape(n_gpu, n_per_gpu)

    ns_states, adapt_keys, _cumulative = _run_loop(
        batcher=batcher,
        adapt_mgr=adapt_mgr,
        ns_state=ns_states,
        step_fn=step_fn,
        n_mcmc_steps=n_mcmc_steps,
        n_extra=n_extra,
        termination_criteria=termination_criteria,
        callbacks=callbacks,
        n_moves=n_moves,
        move_descriptors=move_descriptors,
        rng_key=adapt_keys,
        info_interval=max(1, (max_iterations or 1000) // 20),
        inter_re_mgr=inter_re_mgr,
        max_neighbors_list=ladder,
        max_neighbors_offset=int(max_neighbors_offset),
    )

    # Final evidence: per-run contribution from remaining live walkers.
    # ns_states.population.energy shape: (G, P, K)
    remaining_potentials = ns_states.population.energy  # (G, P, K)
    log_remaining_mass = -ns_states.iteration / n_walkers  # (G, P)
    log_avg_likelihood = -jnp.mean(remaining_potentials, axis=-1)  # (G, P)
    ns_states = ns_states.set(
        log_evidence=jnp.logaddexp(
            ns_states.log_evidence,
            log_remaining_mass + log_avg_likelihood,
        ),
    )

    for g in range(n_gpu):
        for p in range(n_per_gpu):
            logger.info(
                "GPU %d run %d complete: %d dead points, log_Z=%.4f",
                g, p, int(ns_states.iteration[g, p]),
                float(ns_states.log_evidence[g, p]),
            )

    for cb in callbacks:
        if hasattr(cb, "on_finish"):
            cb.on_finish(ns_states)

    # Dead-point history is persisted to disk by the callbacks; not in the
    # return.  ``n_dead`` ≡ ``iteration`` (both increment per ns_step).
    pop = ns_states.population
    return {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "log_evidence": ns_states.log_evidence,    # (G, P)
        "iteration": ns_states.iteration,           # (G, P)
        "n_dead": ns_states.iteration,              # (G, P) — same as iteration
        "n_walkers": n_walkers,
        "n_gpu": n_gpu,
        "n_per_gpu": n_per_gpu,
        "n_runs": n_total,
    }
