"""Step size adjustment via trial moves — pure JAX function.

Core function: ``adjust_step_size`` adjusts the step size for ONE move
type on ONE run by running trial moves on sampled walkers and iterating
until the acceptance rate falls within a target window.

Properties:
- Pure function on arrays — no class, no mutable state
- JIT-compilable (``lax.while_loop`` for the bisection)
- Vmappable over runs for multi-run parallelism
- Layout-agnostic: same function under vmap/pmap

The outer loop (which moves to adjust, when to trigger, how to write
back) is handled by ``run_ns`` — this module only does the core
adjustment for a single (move, run) pair.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def _process_rate_jax(
    rate: jnp.ndarray,
    step_size: jnp.ndarray,
    step_size_prev: jnp.ndarray,
    rate_prev: jnp.ndarray,
    min_rate: float,
    max_rate: float,
    adjust_factor: float,
    max_step_size: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Branchless rate processing for step size adjustment.

    Given an acceptance rate and the target window [min_rate, max_rate],
    decide whether the step size has converged and compute the next value.

    Uses rate_prev < 0 as sentinel for "no previous rate".

    Returns:
        (new_step_size, converged, cap_hit, floor_hit, too_high, too_low)
        where:
          - converged: bool scalar, True iff within window or bracketed
          - cap_hit: bool scalar, True iff the proposed adjusted ss hit max_step_size
          - floor_hit: bool scalar, True iff the proposed adjusted ss hit the 1e-20 floor
          - too_high: bool scalar, True iff rate >= max_rate (used to track bracket formation)
          - too_low: bool scalar, True iff rate < min_rate (used to track bracket formation)
    """
    target = 0.5 * (min_rate + max_rate)
    in_window = (rate > min_rate) & (rate < max_rate)

    # Bracket detection: previous and current straddle the target window
    has_prev = rate_prev >= 0.0
    brackets = (
        has_prev
        & (jnp.minimum(rate, rate_prev) < min_rate)
        & (jnp.maximum(rate, rate_prev) > max_rate)
    )
    prev_closer = jnp.abs(rate_prev - target) < jnp.abs(rate - target)
    bracket_ss = jnp.where(prev_closer, step_size_prev, step_size)

    # Direction: scale up if rate too high, down if too low
    too_high = rate >= max_rate
    too_low = rate < min_rate
    scale = jnp.where(
        too_low,
        1.0 / adjust_factor,
        jnp.where(too_high, adjust_factor, 1.0),
    )

    # Check for cap/floor hits before clamp
    proposed_ss = step_size * scale
    cap_hit = proposed_ss >= max_step_size
    floor_hit = proposed_ss <= 1e-20
    adjusted_ss = jnp.clip(proposed_ss, 1e-20, max_step_size)

    # Pick result: in_window > brackets > adjusted
    new_ss = jnp.where(
        in_window, step_size, jnp.where(brackets, bracket_ss, adjusted_ss)
    )
    converged = in_window | brackets

    return new_ss, converged, cap_hit, floor_hit, too_high, too_low


def adjust_step_size(
    population: Any,
    move_fn: Callable,
    step_size: jnp.ndarray,
    emax: jnp.ndarray,
    rng_key: jax.Array,
    n_samples: int,
    min_rate: float,
    max_rate: float,
    adjust_factor: float,
    max_step_size: float,
    max_rounds: int,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray,
]:
    """Adjust step size for one move type until acceptance rate is in window.

    Pure JAX function. JIT-compilable. Vmappable over runs.

    Iterates a bisection-like loop: sample walkers → run trial moves →
    measure acceptance rate → scale step size up/down. Stops when rate
    falls within [min_rate, max_rate] or brackets are detected.

    Args:
        population: Batched MCState, shape (n_walkers, ...) on every field.
        move_fn: Single-move step function (static under JIT).
            Signature: move_fn(state, key, constraint) -> (state, MoveInfo)
        step_size: Current step size (scalar).
        emax: Likelihood constraint — max energy in population (scalar).
        rng_key: PRNG key.
        n_samples: Number of walkers to sample per trial round (static).
        min_rate: Lower bound of target acceptance window (static).
        max_rate: Upper bound of target acceptance window (static).
        adjust_factor: Multiplicative factor for step size scaling (static).
        max_step_size: Upper bound on step size (static).
        max_rounds: Maximum number of adjustment rounds (static).

    Returns:
        (new_step_size, final_rate, final_counts, n_rounds, converged,
         cap_hits, floor_hits, bracket_detected,
         trial_n_evaluations, trial_n_grad_evaluations)
        - new_step_size: scalar float — adjusted step size
        - final_rate: scalar float — acceptance rate at final round
        - final_counts: (4,) int32 — rejection reason counts from final round
        - n_rounds: scalar int32 — number of bisection rounds executed
        - converged: scalar bool — True iff loop converged within max_rounds
        - cap_hits: scalar int32 — rounds where proposed ss hit max_step_size
        - floor_hits: scalar int32 — rounds where proposed ss hit 1e-20 floor
        - bracket_detected: scalar bool — True iff both too-high and too-low
          rates were observed during bisection (proper bracket formed)
        - trial_n_evaluations: scalar int32 — total backend calls made during
          all trial rounds (summed across walkers and rounds)
        - trial_n_grad_evaluations: scalar int32 — value_and_grad subset of
          trial_n_evaluations
    """
    n_walkers = population.energy.shape[0]

    # Carry: (step_size, step_size_prev, rate_prev, rng_key, round_idx,
    #         converged, reject_counts, cap_hits, floor_hits,
    #         saw_too_high, saw_too_low,
    #         cumulative_n_evals, cumulative_n_grad_evals)
    def cond_fn(carry):
        _, _, _, _, round_idx, converged, _, _, _, _, _, _, _ = carry
        return ~converged & (round_idx < max_rounds)

    def body_fn(carry):
        (ss, ss_prev, rate_prev, key, round_idx, converged, _,
         cap_hits, floor_hits, saw_too_high, saw_too_low,
         cum_evals, cum_grad_evals) = carry

        # 1. Sample walkers
        key, key_sample, key_trials = jax.random.split(key, 3)
        indices = jax.random.choice(
            key_sample, n_walkers, shape=(n_samples,), replace=True
        )
        sample = jax.tree.map(lambda x: x[indices], population)

        # 2. Inject test step size into both step_size (scalar) and step_sizes
        # (per-move array).  The MWG wrapper reads state.step_sizes[move_idx]
        # rather than state.step_size, so we must update the array as well.
        # Broadcasting ss into all positions is safe because the trial
        # function only calls one specific move kernel per adjust_step_size
        # call, so only the target move_idx entry is read.
        sample = sample.set(
            step_size=jnp.full(n_samples, ss),
            step_sizes=jnp.broadcast_to(
                ss[None, None], (n_samples, sample.step_sizes.shape[-1])
            ),
        )

        # 3. Run trial moves (vmapped over sampled walkers)
        trial_keys = jax.random.split(key_trials, n_samples)

        def trial_one(state, trial_key):
            _, info = move_fn(state, trial_key, emax)
            return info.accepted, info.reject_reason, info.n_evaluations, info.n_grad_evaluations

        accepted, reasons, n_evals_per_sample, n_grad_evals_per_sample = jax.vmap(trial_one)(sample, trial_keys)
        rate = jnp.mean(accepted.astype(jnp.float32))

        # Per-reason counts (code 0=accepted, 1=energy, 2=cell, 3=prior)
        counts = jnp.array([
            jnp.sum(reasons == 0),
            jnp.sum(reasons == 1),
            jnp.sum(reasons == 2),
            jnp.sum(reasons == 3),
        ], dtype=jnp.int32)

        # Accumulate evaluation counts over all trial rounds
        round_evals = jnp.sum(n_evals_per_sample.astype(jnp.int32))
        round_grad_evals = jnp.sum(n_grad_evals_per_sample.astype(jnp.int32))
        new_cum_evals = cum_evals + round_evals
        new_cum_grad_evals = cum_grad_evals + round_grad_evals

        # 4. Process rate → new step size + convergence flag + diagnostics
        new_ss, new_converged, cap_hit, floor_hit, too_high, too_low = _process_rate_jax(
            rate, ss, ss_prev, rate_prev,
            min_rate, max_rate, adjust_factor, max_step_size,
        )

        new_cap_hits = cap_hits + cap_hit.astype(jnp.int32)
        new_floor_hits = floor_hits + floor_hit.astype(jnp.int32)
        new_saw_too_high = saw_too_high | too_high
        new_saw_too_low = saw_too_low | too_low

        return (new_ss, ss, rate, key, round_idx + 1,
                converged | new_converged, counts,
                new_cap_hits, new_floor_hits,
                new_saw_too_high, new_saw_too_low,
                new_cum_evals, new_cum_grad_evals)

    init_carry = (
        step_size,
        step_size,
        jnp.array(-1.0),  # sentinel: no previous rate
        rng_key,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),   # cap_hits
        jnp.array(0, dtype=jnp.int32),   # floor_hits
        jnp.array(False),                 # saw_too_high
        jnp.array(False),                 # saw_too_low
        jnp.array(0, dtype=jnp.int32),   # cumulative_n_evals
        jnp.array(0, dtype=jnp.int32),   # cumulative_n_grad_evals
    )

    final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    (final_ss, _, final_rate, _, n_rounds, converged, final_counts,
     cap_hits, floor_hits, saw_too_high, saw_too_low,
     trial_n_evals, trial_n_grad_evals) = final

    bracket_detected = saw_too_high & saw_too_low

    return (final_ss, final_rate, final_counts,
            n_rounds, converged, cap_hits, floor_hits, bracket_detected,
            trial_n_evals, trial_n_grad_evals)
