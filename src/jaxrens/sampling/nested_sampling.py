"""Core nested sampling loop.

Two-loop architecture:
- Outer loop (Python): kernel dispatch, dead-point collection,
  termination check, callbacks, checkpointing
- Inner loop (lax.scan): n_mcmc_steps within the selected compiled kernel

Functions:
- init_ns(): Initialize NS state from configuration
- ns_step(): One dead-point replacement (inner lax.scan over MCMC steps)
- run_ns(): Full NS run (outer Python loop calling ns_step)
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

logger = logging.getLogger(__name__)


def init_ns(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    boxes: jnp.ndarray | None,
    rng_key: jax.Array,
    max_dead: int = 50000,
) -> dict:
    """Initialize NS state.

    Args:
        positions: Initial walker positions, shape (n_walkers, n_atoms, 3).
        types: Atom types, shape (n_walkers, n_atoms) or (n_atoms,).
        energies: Initial energies, shape (n_walkers,).
        boxes: Unit cells, shape (n_walkers, 3, 3) or None.
        rng_key: JAX PRNG key.
        max_dead: Maximum number of dead points to collect.

    Returns:
        NS state dict with all fields needed for the loop.
    """
    n_walkers = positions.shape[0]

    # Broadcast types if not batched
    if types.ndim == 1:
        types = jnp.broadcast_to(types[None, :], (n_walkers, types.shape[0]))

    return {
        "positions": positions,
        "types": types,
        "energies": energies,
        "boxes": boxes,
        "dead_energies": jnp.full(max_dead, jnp.inf),
        "dead_positions": jnp.zeros((max_dead, *positions.shape[1:])),
        "log_evidence": jnp.array(-jnp.inf),
        "iteration": 0,
        "n_dead": 0,
        "n_walkers": n_walkers,
        "rng_key": rng_key,
    }


def _find_worst_walker(energies: jnp.ndarray) -> tuple[int, jnp.ndarray]:
    """Find the walker with the highest energy (worst likelihood).

    Returns:
        (index, energy) of the worst walker.
    """
    idx = jnp.argmax(energies)
    return idx, energies[idx]


def ns_step(
    ns_state: dict,
    step_fn: Callable,
    n_mcmc_steps: int = 20,
    adapt_state: AdaptationState | None = None,
    target_acceptance: float = 0.5,
) -> tuple[dict, dict, AdaptationState | None]:
    """One NS iteration: replace the worst walker.

    1. Find worst walker (highest energy)
    2. Record it as a dead point
    3. Clone a random survivor
    4. Run n_mcmc_steps MCMC steps on the clone (via lax.scan)
    5. Update evidence estimate

    Args:
        ns_state: Current NS state dict.
        step_fn: MCMC step function (key, state, Emax) -> (state, info).
        n_mcmc_steps: Number of MCMC steps per dead-point replacement.
        adapt_state: Optional adaptation state for step size tuning.
        target_acceptance: Target acceptance rate for adaptation.

    Returns:
        (updated_ns_state, iteration_info, updated_adapt_state)
    """
    positions = ns_state["positions"]
    types = ns_state["types"]
    energies = ns_state["energies"]
    boxes = ns_state["boxes"]
    n_walkers = ns_state["n_walkers"]
    iteration = ns_state["iteration"]
    key = ns_state["rng_key"]

    # 1. Find worst walker
    worst_idx, emax = _find_worst_walker(energies)

    # 2. Record dead point
    n_dead = ns_state["n_dead"]
    dead_energies = ns_state["dead_energies"].at[n_dead].set(emax)
    dead_positions = ns_state["dead_positions"].at[n_dead].set(
        positions[worst_idx]
    )

    # 3. Update evidence estimate
    # log(weight) = log(1 - exp(-1/n_walkers)) + log(prior_mass_at_step)
    # Simplified: log_weight ~ -iteration/n_walkers + log(1/n_walkers)
    log_weight = -iteration / n_walkers + jnp.log(1.0 / n_walkers)
    log_likelihood = -emax  # E = -log(L)
    log_evidence_contribution = log_weight + log_likelihood
    log_evidence = jnp.logaddexp(
        ns_state["log_evidence"], log_evidence_contribution
    )

    # 4. Clone a random survivor (not the worst)
    key, key_clone, key_mcmc = jax.random.split(key, 3)
    # Pick random index excluding worst
    clone_idx = jax.random.randint(key_clone, (), 0, n_walkers - 1)
    clone_idx = jnp.where(clone_idx >= worst_idx, clone_idx + 1, clone_idx)

    # Build initial state for MCMC from cloned walker
    from jaxrens.sampling.moves.random_walk import RandomWalkState

    step_size = (
        get_step_size(adapt_state) if adapt_state is not None else jnp.array(0.1)
    )

    clone_state = RandomWalkState(
        positions=positions[clone_idx],
        types=types[clone_idx],
        energy=energies[clone_idx],
        box=boxes[clone_idx] if boxes is not None else None,
        step_size=step_size,
    )

    # 5. Run MCMC chain via lax.scan
    mcmc_keys = jax.random.split(key_mcmc, n_mcmc_steps)

    def scan_step(state, step_key):
        new_state, info = step_fn(step_key, state, emax)
        return new_state, info.accepted

    final_state, accepted_arr = jax.lax.scan(scan_step, clone_state, mcmc_keys)
    n_accepted = jnp.sum(accepted_arr)

    # 6. Replace worst walker with MCMC result
    new_positions = positions.at[worst_idx].set(final_state.positions)
    new_energies = energies.at[worst_idx].set(final_state.energy)
    new_types = types  # types don't change
    new_boxes = boxes
    if boxes is not None:
        new_boxes = boxes.at[worst_idx].set(
            final_state.box if final_state.box is not None else boxes[worst_idx]
        )

    # 7. Update adaptation
    acc_rate = n_accepted / n_mcmc_steps
    if adapt_state is not None:
        adapt_state = dual_averaging_update(
            adapt_state,
            accepted=jnp.array(acc_rate > target_acceptance),
            target_acceptance=target_acceptance,
        )

    # Build updated state
    new_ns_state = {
        **ns_state,
        "positions": new_positions,
        "types": new_types,
        "energies": new_energies,
        "boxes": new_boxes,
        "dead_energies": dead_energies,
        "dead_positions": dead_positions,
        "log_evidence": log_evidence,
        "iteration": iteration + 1,
        "n_dead": n_dead + 1,
        "rng_key": key,
    }

    info = {
        "emax": emax,
        "worst_idx": worst_idx,
        "clone_idx": clone_idx,
        "n_accepted": n_accepted,
        "acceptance_rate": acc_rate,
        "log_evidence": log_evidence,
        "step_size": step_size,
    }

    return new_ns_state, info, adapt_state


def run_ns(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    boxes: jnp.ndarray | None,
    step_fn: Callable,
    rng_key: jax.Array,
    n_walkers: int | None = None,
    max_iterations: int = 10000,
    n_mcmc_steps: int = 20,
    convergence_threshold: float = 0.1,
    initial_step_size: float = 0.1,
    target_acceptance: float = 0.5,
    adapt_warmup: int = 100,
    callbacks: list[Any] | None = None,
) -> dict:
    """Run a full nested sampling calculation.

    This is the outer Python loop that calls ns_step() repeatedly.

    Args:
        positions: Initial walker positions, shape (n_walkers, n_atoms, 3).
        types: Atom types, shape (n_atoms,).
        energies: Initial walker energies, shape (n_walkers,).
        boxes: Unit cells, shape (n_walkers, 3, 3) or None.
        step_fn: MCMC step function from build_kernel().
        rng_key: JAX PRNG key.
        n_walkers: Number of live walkers (inferred from positions if None).
        max_iterations: Maximum number of NS iterations.
        n_mcmc_steps: MCMC steps per dead-point replacement.
        convergence_threshold: Log-evidence convergence threshold.
        initial_step_size: Starting step size for MCMC.
        target_acceptance: Target acceptance rate.
        adapt_warmup: Number of iterations for step size warmup.
        callbacks: Optional list of NSCallback objects.

    Returns:
        Final NS state dict with dead points, evidence, etc.
    """
    if callbacks is None:
        callbacks = []

    if n_walkers is None:
        n_walkers = positions.shape[0]

    # Initialize
    ns_state = init_ns(
        positions, types, energies, boxes, rng_key, max_dead=max_iterations
    )
    adapt_state = init_adaptation(
        initial_step_size=initial_step_size,
        target_acceptance=target_acceptance,
    )

    # Outer loop
    for i in range(max_iterations):
        # Use adaptation during warmup, fixed after
        current_adapt = adapt_state if i < adapt_warmup else None

        ns_state, info, adapt_state_new = ns_step(
            ns_state,
            step_fn,
            n_mcmc_steps=n_mcmc_steps,
            adapt_state=adapt_state,
            target_acceptance=target_acceptance,
        )
        adapt_state = adapt_state_new

        # Callbacks
        for cb in callbacks:
            if hasattr(cb, "on_iteration"):
                cb.on_iteration(i, ns_state, info)

        # Termination check
        if i > n_walkers:
            log_remaining = -i / n_walkers
            if log_remaining < float(ns_state["log_evidence"]) - convergence_threshold:
                logger.info(
                    "Converged at iteration %d, log_evidence=%.4f",
                    i,
                    float(ns_state["log_evidence"]),
                )
                break

    # Final evidence: add contribution from remaining live walkers
    remaining_energies = ns_state["energies"]
    log_remaining_mass = -ns_state["iteration"] / n_walkers
    log_avg_likelihood = -jnp.mean(remaining_energies)
    log_final_contribution = log_remaining_mass + log_avg_likelihood
    ns_state["log_evidence"] = jnp.logaddexp(
        ns_state["log_evidence"], log_final_contribution
    )

    # Callbacks: on_finish
    for cb in callbacks:
        if hasattr(cb, "on_finish"):
            cb.on_finish(ns_state)

    return ns_state
