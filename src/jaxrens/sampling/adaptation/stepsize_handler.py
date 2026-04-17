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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Branchless rate processing for step size adjustment.

    Given an acceptance rate and the target window [min_rate, max_rate],
    decide whether the step size has converged and compute the next value.

    Uses rate_prev < 0 as sentinel for "no previous rate".

    Returns:
        (new_step_size, converged) where converged is a bool scalar.
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
    scale = jnp.where(
        rate < min_rate,
        1.0 / adjust_factor,
        jnp.where(rate >= max_rate, adjust_factor, 1.0),
    )
    adjusted_ss = jnp.clip(step_size * scale, 1e-20, max_step_size)

    # Pick result: in_window > brackets > adjusted
    new_ss = jnp.where(
        in_window, step_size, jnp.where(brackets, bracket_ss, adjusted_ss)
    )
    converged = in_window | brackets

    return new_ss, converged


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        (new_step_size, final_acceptance_rate) — both scalars.
    """
    n_walkers = population.energy.shape[0]

    # Carry: (step_size, step_size_prev, rate_prev, rng_key, round_idx, converged)
    def cond_fn(carry):
        _, _, _, _, round_idx, converged = carry
        return ~converged & (round_idx < max_rounds)

    def body_fn(carry):
        ss, ss_prev, rate_prev, key, round_idx, converged = carry

        # 1. Sample walkers
        key, key_sample, key_trials = jax.random.split(key, 3)
        indices = jax.random.choice(
            key_sample, n_walkers, shape=(n_samples,), replace=True
        )
        sample = jax.tree.map(lambda x: x[indices], population)

        # 2. Inject test step size
        sample = sample.set(step_size=jnp.full(n_samples, ss))

        # 3. Run trial moves (vmapped over sampled walkers)
        trial_keys = jax.random.split(key_trials, n_samples)

        def trial_one(state, trial_key):
            _, info = move_fn(state, trial_key, emax)
            return info.accepted

        accepted = jax.vmap(trial_one)(sample, trial_keys)
        rate = jnp.mean(accepted.astype(jnp.float32))

        # 4. Process rate → new step size + convergence flag
        new_ss, new_converged = _process_rate_jax(
            rate, ss, ss_prev, rate_prev,
            min_rate, max_rate, adjust_factor, max_step_size,
        )

        return (new_ss, ss, rate, key, round_idx + 1, converged | new_converged)

    init_carry = (
        step_size,
        step_size,
        jnp.array(-1.0),  # sentinel: no previous rate
        rng_key,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
    )

    final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    final_ss, _, final_rate, _, _, _ = final

    return final_ss, final_rate
