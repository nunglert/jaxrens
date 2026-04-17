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

from jaxrens.base import NSCallback
from jaxrens.sampling.adaptation.step_size import (
    AdaptationState,
    dual_averaging_update,
    get_step_size,
    init_adaptation,
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

    state = NSState(
        population=population,
        dead_energies=jnp.full(max_dead, jnp.inf),
        dead_positions=jnp.zeros((max_dead, n_atoms, 3)),
        dead_volumes=jnp.zeros(max_dead),
        log_evidence=jnp.array(-jnp.inf),
        iteration=jnp.array(0, dtype=jnp.int32),
        n_dead=jnp.array(0, dtype=jnp.int32),
        rng_key=rng_key,
        n_walkers=n_walkers,
        n_atoms=n_atoms,
        max_dead=max_dead,
    )

    logger.debug(
        "NS state initialized: %d walkers, energy range [%.4g, %.4g], max_dead=%d",
        n_walkers, float(jnp.min(energies)), float(jnp.max(energies)), max_dead,
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
        def scan_body(state, step_key):
            new_state, info = step_fn(step_key, state, potential_max)
            return new_state, info.accepted
        final, accepted_arr = jax.lax.scan(scan_body, walker, chain_keys)
        return final, jnp.sum(accepted_arr)

    finals, acc_counts = jax.vmap(run_one_chain)(walk_batch, all_mcmc_keys)

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
    info = {
        "emax": potential_max,
        "hmax": potential_max,  # backward compat — same as emax now
        "worst_idx": worst_idx,
        "clone_idx": clone_idx,
        "acceptance_rate": total_accepted / total_steps,
    }
    return new_ns_state, info


# ---------------------------------------------------------------------------
# run_ns — outer Python loop (NOT JIT'd)
# ---------------------------------------------------------------------------


def _ns_state_to_result_dict(ns_state: NSState) -> dict:
    """Convert NSState to backward-compatible result dict."""
    pop = ns_state.population
    ep = pop.ensemble_params if hasattr(pop, "ensemble_params") else {}
    is_npt = isinstance(ep, dict) and float(ep.get("pressure", 0.0)) != 0.0
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
    adapt_warmup: int = 100,
    callbacks: list[Any] | None = None,
    termination_criteria: list[TerminationCriterion] | None = None,
    ensemble_params: dict | None = None,
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

    # Initialize NSState with batched MCState population
    ns_state = init_ns(
        init_fn, positions, types, energies, cells, rng_key,
        max_dead=max_iterations,
        step_sizes=jnp.full(1, initial_step_size),
        ensemble_params=ensemble_params,
    )

    adapt_state = init_adaptation(
        initial_step_size=initial_step_size,
        target_acceptance=target_acceptance,
    )

    n_atoms = positions.shape[1] if positions.ndim >= 2 else None
    logger.info(
        "Starting NS run: %d walkers, %s atoms, max_iter=%d, n_mcmc=%d",
        n_walkers, n_atoms, max_iterations, n_mcmc_steps,
    )

    info_interval = max(1, max_iterations // 20)

    # JIT-compile ns_step with step_fn, n_mcmc_steps, n_extra as static args
    jit_ns_step = jax.jit(ns_step, static_argnums=(1, 2, 3))

    for i in range(max_iterations):
        # Adaptation: update step_sizes on population before step
        if i < adapt_warmup:
            step_size = get_step_size(adapt_state)
            pop = ns_state.population
            n_move_types = pop.step_sizes.shape[-1]
            new_step_sizes = jnp.broadcast_to(
                step_size, (n_walkers, n_move_types)
            )
            ns_state = ns_state.set(
                population=pop.set(step_sizes=new_step_sizes),
            )

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

        # Adaptation update (outside JIT)
        if i < adapt_warmup:
            adapt_state = dual_averaging_update(
                adapt_state,
                accepted=jnp.array(info["acceptance_rate"] > target_acceptance),
                target_acceptance=target_acceptance,
            )

        # Periodic INFO log
        if i % info_interval == 0 or i == max_iterations - 1:
            logger.info(
                "iter=%d  Emax=%.6g  log_Z=%.4f  acc=%.2f  ss=%.4g",
                int(ns_state.iteration),
                float(info["emax"]),
                float(info["acceptance_rate"]),
                float(get_step_size(adapt_state)),
                float(ns_state.log_evidence),
            )

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
    adapt_warmup: int = 100,
    termination_criteria: list[TerminationCriterion] | None = None,
    ensemble_params_per_run: list[dict] | None = None,
) -> dict:
    """Run multiple NS calculations in parallel via vmap(ns_step).

    All runs share the same step_fn, n_mcmc_steps, and n_extra.
    Each run has independent RNG, adaptation, and (optionally) ensemble
    parameters. The outer Python loop handles adaptation, overflow
    detection, termination, and logging.

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
        target_acceptance: Target acceptance rate.
        adapt_warmup: Iterations for step size warmup.
        termination_criteria: Optional. If None, uses defaults.
        ensemble_params_per_run: Per-run ensemble params (e.g. different pressures).

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

    # Initialize batched NSState: (n_runs, ...) on all dynamic fields
    ns_states = init_ns_parallel(
        init_fn, positions, types, energies, cells, rng_keys,
        max_dead=max_iterations,
        step_sizes=jnp.full(1, initial_step_size),
        ensemble_params_per_run=ensemble_params_per_run,
    )

    # Per-run adaptation: stack n_runs copies of AdaptationState
    adapt_states_list = [
        init_adaptation(initial_step_size=initial_step_size,
                        target_acceptance=target_acceptance)
        for _ in range(n_runs)
    ]
    adapt_states = jax.tree.map(lambda *xs: jnp.stack(xs), *adapt_states_list)

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
        # Adaptation: per-run step_size broadcast
        if i < adapt_warmup:
            step_sizes_per_run = jax.vmap(get_step_size)(adapt_states)  # (n_runs,)
            pop = ns_states.population
            n_move_types = pop.step_sizes.shape[-1]
            # Broadcast: (n_runs,) -> (n_runs, n_walkers, n_move_types)
            new_step_sizes = jnp.broadcast_to(
                step_sizes_per_run[:, None, None],
                (n_runs, n_walkers, n_move_types),
            )
            ns_states = ns_states.set(
                population=pop.set(step_sizes=new_step_sizes),
            )

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

        # Per-run adaptation (vmapped)
        if i < adapt_warmup:
            adapt_states = jax.vmap(
                lambda s, acc: dual_averaging_update(
                    s, accepted=acc > target_acceptance,
                    target_acceptance=target_acceptance,
                )
            )(adapt_states, infos["acceptance_rate"])

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
