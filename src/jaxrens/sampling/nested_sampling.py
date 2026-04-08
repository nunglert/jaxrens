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
from jaxrens.sampling.termination import (
    IterationTermination,
    PriorMassTermination,
    TerminationCriterion,
    check_any,
)
from jaxrens.utils.cell import get_volume

logger = logging.getLogger(__name__)


def init_ns(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    energies: jnp.ndarray,
    boxes: jnp.ndarray | None,
    rng_key: jax.Array,
    max_dead: int = 50000,
    pressure: float | None = None,
) -> dict:
    """Initialize NS state.

    Args:
        positions: Initial walker positions, shape (n_walkers, n_atoms, 3).
        types: Atom types, shape (n_walkers, n_atoms) or (n_atoms,).
        energies: Initial energies, shape (n_walkers,).
        boxes: Unit cells, shape (n_walkers, 3, 3) or None.
        rng_key: JAX PRNG key.
        max_dead: Maximum number of dead points to collect.
        pressure: Pressure for NPT ensemble. When set, dead_volumes are tracked.

    Returns:
        NS state dict with all fields needed for the loop.
    """
    n_walkers = positions.shape[0]

    # Broadcast types if not batched
    if types.ndim == 1:
        types = jnp.broadcast_to(types[None, :], (n_walkers, types.shape[0]))

    state = {
        "positions": positions,
        "types": types,
        "energies": energies,
        "boxes": boxes,
        "dead_energies": jnp.full(max_dead, jnp.inf),
        "dead_positions": jnp.zeros((max_dead, *positions.shape[1:])),
        "dead_volumes": jnp.zeros(max_dead) if pressure else None,
        "log_evidence": jnp.array(-jnp.inf),
        "iteration": 0,
        "n_dead": 0,
        "n_walkers": n_walkers,
        "rng_key": rng_key,
    }
    return state


def _compute_enthalpies(
    energies: jnp.ndarray,
    boxes: jnp.ndarray | None,
    pressure: float | None,
) -> jnp.ndarray:
    """Compute enthalpies H = E + P*V for each walker.

    When pressure is None or 0, returns energies unchanged.
    """
    if pressure is None or pressure == 0.0:
        return energies
    if boxes is None:
        raise ValueError("pressure is set but boxes are None; NPT requires periodic cells")
    volumes = jax.vmap(get_volume)(boxes)  # (n_walkers,)
    return energies + pressure * volumes


def _find_worst_walker(
    energies: jnp.ndarray,
    rng_key: jax.Array | None = None,
    n_atoms: int | None = None,
) -> tuple[int, jnp.ndarray]:
    """Find the walker with the highest energy/enthalpy (worst likelihood).

    Uses energy per atom for comparison when n_atoms is provided, which
    improves numerical precision for large systems (avoids float32 range
    issues). When multiple walkers share the same
    maximum energy, one is selected uniformly at random (tie-breaking).

    Args:
        energies: Per-walker energies/enthalpies, shape (n_walkers,).
        rng_key: PRNG key for random tie-breaking. If None, uses argmax
            (deterministic, picks first max).
        n_atoms: Number of atoms per walker. When provided, comparison
            uses energy/n_atoms for better precision.

    Returns:
        (index, value) of the worst walker. Value is the original energy
        (not per-atom), so the NS evidence calculation is unaffected.
    """
    # Compare using per-atom energy for better float precision
    comparison = energies / n_atoms if n_atoms is not None else energies

    if rng_key is not None:
        # Random tie-breaking: among all walkers sharing the max value,
        # select one uniformly at random
        max_val = jnp.max(comparison)
        is_max = jnp.isclose(comparison, max_val, rtol=1e-7, atol=0.0)
        # Use Gumbel-max trick for differentiable random argmax among ties
        noise = jax.random.uniform(rng_key, shape=comparison.shape, minval=0.0, maxval=1.0)
        # Set non-max entries to -inf so they're never selected
        tie_scores = jnp.where(is_max, noise, -jnp.inf)
        idx = jnp.argmax(tie_scores)
    else:
        idx = jnp.argmax(comparison)

    return idx, energies[idx]


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
        # Add small noise for tie-breaking
        noise = 1e-10 * jax.random.uniform(
            rng_key, shape=comparison.shape, minval=-1.0, maxval=1.0
        )
        comparison = comparison + noise

    # top-k via negative argsort
    sorted_indices = jnp.argsort(-comparison)
    worst_indices = sorted_indices[:n_cull]
    worst_values = energies[worst_indices]
    return worst_indices, worst_values


def ns_step(
    ns_state: dict,
    step_fn: Callable,
    n_mcmc_steps: int = 20,
    adapt_state: AdaptationState | None = None,
    target_acceptance: float = 0.5,
    pressure: float | None = None,
    n_cull: int = 1,
) -> tuple[dict, dict, AdaptationState | None]:
    """One NS iteration: replace the worst walker(s).

    When n_cull > 1, removes the n_cull worst walkers simultaneously,
    records all as dead points, clones survivors, and runs MCMC on each.

    Args:
        ns_state: Current NS state dict.
        step_fn: MCMC step function (key, state, Emax) -> (state, info).
        n_mcmc_steps: Number of MCMC steps per dead-point replacement.
        adapt_state: Optional adaptation state for step size tuning.
        target_acceptance: Target acceptance rate for adaptation.
        pressure: Pressure for NPT ensemble. None or 0 = NVT (energy only).
        n_cull: Number of walkers to replace per iteration (default 1).

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

    # Use enthalpy if pressure is set
    comparison_values = _compute_enthalpies(energies, boxes, pressure)
    n_atoms = positions.shape[1] if positions.ndim >= 2 else None

    from jaxrens.sampling.moves.random_walk import RandomWalkState

    step_size = (
        get_step_size(adapt_state) if adapt_state is not None else jnp.array(0.1)
    )

    if n_cull == 1:
        # --- Fast path: single walker culling (original logic) ---

        # 1. Find worst walker
        key, key_worst = jax.random.split(key)
        worst_idx, hmax = _find_worst_walker(
            comparison_values, rng_key=key_worst, n_atoms=n_atoms
        )

        # 2. Record dead point
        emax = energies[worst_idx]
        n_dead = ns_state["n_dead"]
        dead_energies = ns_state["dead_energies"].at[n_dead].set(hmax)
        dead_positions = ns_state["dead_positions"].at[n_dead].set(
            positions[worst_idx]
        )

        dead_volumes = ns_state["dead_volumes"]
        if dead_volumes is not None and boxes is not None:
            dead_volumes = dead_volumes.at[n_dead].set(get_volume(boxes[worst_idx]))

        # 3. Update evidence estimate
        log_weight = -iteration / n_walkers + jnp.log(1.0 / n_walkers)
        log_likelihood = -hmax
        log_evidence_contribution = log_weight + log_likelihood
        log_evidence = jnp.logaddexp(
            ns_state["log_evidence"], log_evidence_contribution
        )

        # 4. Clone a random survivor
        key, key_clone, key_mcmc = jax.random.split(key, 3)
        clone_idx = jax.random.randint(key_clone, (), 0, n_walkers - 1)
        clone_idx = jnp.where(clone_idx >= worst_idx, clone_idx + 1, clone_idx)

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
            new_state, info = step_fn(step_key, state, hmax)
            return new_state, info.accepted

        final_state, accepted_arr = jax.lax.scan(scan_step, clone_state, mcmc_keys)
        n_accepted = jnp.sum(accepted_arr)

        # 6. Replace worst walker with MCMC result
        new_positions = positions.at[worst_idx].set(final_state.positions)
        new_energies = energies.at[worst_idx].set(final_state.energy)
        new_types = types
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

        new_ns_state = {
            **ns_state,
            "positions": new_positions,
            "types": new_types,
            "energies": new_energies,
            "boxes": new_boxes,
            "dead_energies": dead_energies,
            "dead_positions": dead_positions,
            "dead_volumes": dead_volumes,
            "log_evidence": log_evidence,
            "iteration": iteration + 1,
            "n_dead": n_dead + 1,
            "rng_key": key,
        }

        info = {
            "emax": emax,
            "hmax": hmax,
            "worst_idx": worst_idx,
            "clone_idx": clone_idx,
            "n_accepted": n_accepted,
            "acceptance_rate": acc_rate,
            "log_evidence": log_evidence,
            "step_size": step_size,
        }

    else:
        # --- Multi-cull path: remove n_cull worst walkers simultaneously ---

        # 1. Find n_cull worst walkers
        key, key_worst = jax.random.split(key)
        worst_indices, worst_hvals = _find_worst_walkers(
            comparison_values, n_cull, rng_key=key_worst, n_atoms=n_atoms
        )

        # hmax = the highest enthalpy among the culled walkers (the NS constraint)
        hmax = worst_hvals[0]  # sorted descending

        # 2. Record all dead points
        n_dead = ns_state["n_dead"]
        dead_energies = ns_state["dead_energies"]
        dead_positions = ns_state["dead_positions"]
        dead_volumes = ns_state["dead_volumes"]

        for j in range(n_cull):
            w_idx = worst_indices[j]
            dead_energies = dead_energies.at[n_dead + j].set(worst_hvals[j])
            dead_positions = dead_positions.at[n_dead + j].set(positions[w_idx])
            if dead_volumes is not None and boxes is not None:
                dead_volumes = dead_volumes.at[n_dead + j].set(
                    get_volume(boxes[w_idx])
                )

        # 3. Update evidence: one contribution per dead point
        log_evidence = ns_state["log_evidence"]
        for j in range(n_cull):
            iter_j = iteration + j
            log_weight = -iter_j / n_walkers + jnp.log(1.0 / n_walkers)
            log_likelihood = -worst_hvals[j]
            log_evidence = jnp.logaddexp(
                log_evidence, log_weight + log_likelihood
            )

        # 4 & 5. For each culled walker: clone a survivor, run MCMC
        # Build mask of survivor indices
        worst_set = set()
        for j in range(n_cull):
            worst_set.add(int(worst_indices[j]))

        survivor_mask = jnp.array(
            [i not in worst_set for i in range(n_walkers)], dtype=jnp.bool_
        )
        survivor_indices = jnp.where(survivor_mask, size=n_walkers - n_cull)[0]

        new_positions = positions
        new_energies = energies
        new_types = types
        new_boxes = boxes
        total_accepted = 0
        total_mcmc = 0

        key, *cull_keys = jax.random.split(key, 1 + 2 * n_cull)
        clone_indices = []

        for j in range(n_cull):
            w_idx = worst_indices[j]
            key_clone = cull_keys[2 * j]
            key_mcmc = cull_keys[2 * j + 1]

            # Clone random survivor
            n_survivors = n_walkers - n_cull
            clone_pick = jax.random.randint(key_clone, (), 0, n_survivors)
            clone_idx = survivor_indices[clone_pick]
            clone_indices.append(clone_idx)

            clone_state = RandomWalkState(
                positions=new_positions[clone_idx],
                types=new_types[clone_idx],
                energy=new_energies[clone_idx],
                box=new_boxes[clone_idx] if new_boxes is not None else None,
                step_size=step_size,
            )

            # Run MCMC with the overall worst enthalpy as constraint
            mcmc_keys = jax.random.split(key_mcmc, n_mcmc_steps)

            def scan_step(state, step_key, _hmax=hmax):
                new_state, move_info = step_fn(step_key, state, _hmax)
                return new_state, move_info.accepted

            final_state, accepted_arr = jax.lax.scan(
                scan_step, clone_state, mcmc_keys
            )
            total_accepted += int(jnp.sum(accepted_arr))
            total_mcmc += n_mcmc_steps

            # Replace culled walker
            new_positions = new_positions.at[w_idx].set(final_state.positions)
            new_energies = new_energies.at[w_idx].set(final_state.energy)
            if new_boxes is not None:
                new_boxes = new_boxes.at[w_idx].set(
                    final_state.box if final_state.box is not None else new_boxes[w_idx]
                )

        # 7. Update adaptation (average acceptance over all culled walkers)
        acc_rate = total_accepted / max(total_mcmc, 1)
        if adapt_state is not None:
            adapt_state = dual_averaging_update(
                adapt_state,
                accepted=jnp.array(acc_rate > target_acceptance),
                target_acceptance=target_acceptance,
            )

        new_ns_state = {
            **ns_state,
            "positions": new_positions,
            "types": new_types,
            "energies": new_energies,
            "boxes": new_boxes,
            "dead_energies": dead_energies,
            "dead_positions": dead_positions,
            "dead_volumes": dead_volumes,
            "log_evidence": log_evidence,
            "iteration": iteration + n_cull,
            "n_dead": n_dead + n_cull,
            "rng_key": key,
        }

        info = {
            "emax": energies[worst_indices[0]],
            "hmax": hmax,
            "worst_idx": worst_indices,
            "clone_idx": jnp.array(clone_indices),
            "n_accepted": total_accepted,
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
    pressure: float | None = None,
    termination_criteria: list[TerminationCriterion] | None = None,
    n_cull: int = 1,
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
        pressure: Pressure for NPT ensemble. None or 0 = NVT (energy only).
        termination_criteria: Optional list of TerminationCriterion objects.
            If None, defaults to IterationTermination + PriorMassTermination.
        n_cull: Number of walkers to replace per iteration (default 1).

    Returns:
        Final NS state dict with dead points, evidence, etc.
    """
    if callbacks is None:
        callbacks = []

    if n_walkers is None:
        n_walkers = positions.shape[0]

    # Build default termination criteria if none provided
    if termination_criteria is None:
        termination_criteria = [
            IterationTermination(max_iterations),
            PriorMassTermination(n_walkers, convergence_threshold),
        ]

    # Initialize
    ns_state = init_ns(
        positions, types, energies, boxes, rng_key,
        max_dead=max_iterations, pressure=pressure,
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
            pressure=pressure,
            n_cull=n_cull,
        )
        adapt_state = adapt_state_new

        # Callbacks
        for cb in callbacks:
            if hasattr(cb, "on_iteration"):
                cb.on_iteration(i, ns_state, info)

        # Update evidence in PriorMassTermination if present
        for criterion in termination_criteria:
            if isinstance(criterion, PriorMassTermination):
                criterion.update_evidence(float(ns_state["log_evidence"]))

        # Termination check
        emax = float(info["hmax"])
        should_stop, stop_msg = check_any(termination_criteria, i, emax)
        if should_stop:
            logger.info(
                "Terminated at iteration %d: %s (log_evidence=%.4f)",
                i, stop_msg, float(ns_state["log_evidence"]),
            )
            break

    # Final evidence: add contribution from remaining live walkers
    remaining_energies = ns_state["energies"]
    remaining_values = _compute_enthalpies(
        remaining_energies, ns_state["boxes"], pressure
    )
    log_remaining_mass = -ns_state["iteration"] / n_walkers
    log_avg_likelihood = -jnp.mean(remaining_values)
    log_final_contribution = log_remaining_mass + log_avg_likelihood
    ns_state["log_evidence"] = jnp.logaddexp(
        ns_state["log_evidence"], log_final_contribution
    )

    # Store live volumes for post-processing (NPT only)
    if pressure and ns_state["boxes"] is not None:
        ns_state["live_volumes"] = jax.vmap(get_volume)(ns_state["boxes"])
    else:
        ns_state["live_volumes"] = None

    # Callbacks: on_finish
    for cb in callbacks:
        if hasattr(cb, "on_finish"):
            cb.on_finish(ns_state)

    return ns_state
