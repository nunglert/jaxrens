"""Step size adjustment via trial moves — pure JAX functions.

Two production entry points, one per batcher regime:

* :func:`adjust_step_size` — full in-XLA bisection (``lax.while_loop``) for
  SingleRun / VmapRuns / PmapVmapRuns.  Vmaps cleanly over runs;
  per-replica convergence handled by the vmap-of-while_loop "any-active"
  semantics.

* :func:`adjust_step_size_sharded` — sibling for ShardedSingleRun.  Same
  in-XLA ``lax.while_loop`` structure, with cross-shard ``lax.psum``
  reductions inside the body so each shard sees the global trial rate.
  Must be called inside ``jax.pmap(axis_name="shard")``.

Both share :func:`_process_rate_jax` (branchless rate-window decision).
The outer loop (when to trigger, how to write back into the population)
lives in :func:`jaxrens.sampling.adaptation.manager.build_adapt_step` and
its caller ``_run_loop``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from jaxrens.utils.padding import pad_to_multiple


def _process_rate_jax(
    rate: Float[Array, ""],
    step_size: Float[Array, ""],
    step_size_prev: Float[Array, ""],
    rate_prev: Float[Array, ""],
    min_rate: float,
    max_rate: float,
    adjust_factor: float,
    max_step_size: float,
) -> tuple[
    Float[Array, ""], Bool[Array, ""], Bool[Array, ""],
    Bool[Array, ""], Bool[Array, ""], Bool[Array, ""],
]:
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


# ---------------------------------------------------------------------------
# adjust_step_size — in-XLA bisection for SingleRun / VmapRuns / PmapVmapRuns
# ---------------------------------------------------------------------------


def adjust_step_size(
    population: Any,
    move_fn: Callable,
    step_size: Float[Array, ""],
    emax: Float[Array, ""],
    rng_key: Key[Array, ""],
    n_samples: int,
    min_rate: float,
    max_rate: float,
    adjust_factor: float,
    max_step_size: float,
    max_rounds: int,
    *,
    trial_batch_size: int | None = None,
) -> tuple[
    Float[Array, ""], Float[Array, ""], Int[Array, "4"],
    Int[Array, ""], Bool[Array, ""], Int[Array, ""], Int[Array, ""], Bool[Array, ""],
    Int[Array, ""], Int[Array, ""],
]:
    """Adjust step size for one move type until acceptance rate is in window.

    Pure JAX function.  JIT-compilable.  Vmappable over runs.

    Iterates a bisection-like loop: sample walkers → run trial moves →
    measure acceptance rate → scale step size up/down.  Stops when the
    rate falls within ``[min_rate, max_rate]`` or brackets are detected.
    Under vmap-over-runs, ``lax.while_loop`` runs until **all** replicas
    converge or ``max_rounds`` is hit; per-replica convergence is masked
    by the loop's any-active semantics combined with the per-replica
    ``converged`` carry.

    Args:
        population: Batched MCState, shape (n_walkers, ...) on every field.
        move_fn: Single-move step function (static under JIT).
            Signature: move_fn(state, key, constraint) -> (state, MoveInfo).
        step_size: Current step size (scalar under each replica).
        emax: Likelihood constraint — max energy in population (scalar).
        rng_key: PRNG key.
        n_samples: Walkers to sample per trial round (static).
        min_rate, max_rate: Target acceptance window (static).
        adjust_factor: Multiplicative step-size factor (static).
        max_step_size: Upper bound on step size (static).
        max_rounds: Maximum bisection rounds (static).
        trial_batch_size: Optional chunk size for the trial vmap (kwarg-only,
            static).  ``None`` (default) → ``jax.vmap(trial_one)`` over all
            ``n_samples`` walkers.  When set to an int, the trial vmap is
            replaced by ``jax.lax.map(..., batch_size=trial_batch_size)`` so
            peak memory scales with the chunk rather than ``n_samples`` —
            needed when the move's backward tape exceeds device memory.
            If ``n_samples`` is not a multiple of ``trial_batch_size`` the
            trial population is padded with copies of the last sample and
            the pad is sliced off before rate/count aggregation.

    Returns:
        ``(new_step_size, final_rate, final_counts, n_rounds, converged,
        cap_hits, floor_hits, bracket_detected, trial_n_evaluations,
        trial_n_grad_evaluations)``.
    """
    n_walkers = population.energy.shape[0]

    def cond_fn(carry):
        _, _, _, _, round_idx, converged, _, _, _, _, _, _, _ = carry
        return ~converged & (round_idx < max_rounds)

    def body_fn(carry):
        (ss, ss_prev, rate_prev, key, round_idx, converged, _,
         cap_hits, floor_hits, saw_too_high, saw_too_low,
         cum_evals, cum_grad_evals) = carry

        # 1. Sample walkers from the K-population.
        key, key_sample, key_trials = jax.random.split(key, 3)
        indices = jax.random.choice(
            key_sample, n_walkers, shape=(n_samples,), replace=True,
        )
        sample = jax.tree.map(lambda x: x[indices], population)

        # 2. Inject test step size into both ``step_size`` (scalar) and
        # ``step_sizes`` (per-move array).  The MWG wrapper reads
        # ``state.step_sizes[move_idx]`` rather than ``state.step_size``,
        # so we must update the array as well.  Broadcasting ``ss`` into
        # all positions is safe because the trial function only calls one
        # specific move kernel per ``adjust_step_size`` call.
        sample = sample.set(
            step_size=jnp.full(n_samples, ss),
            step_sizes=jnp.broadcast_to(
                ss[None, None], (n_samples, sample.step_sizes.shape[-1]),
            ),
        )

        # 3. Run trial moves (vmapped over sampled walkers; optionally chunked).
        trial_keys = jax.random.split(key_trials, n_samples)

        def trial_one(state, trial_key):
            _, info = move_fn(state, trial_key, emax)
            return (
                info.accepted, info.reject_reason,
                info.n_evaluations, info.n_grad_evaluations,
            )

        if trial_batch_size is None:
            accepted, reasons, n_evals_per_sample, n_grad_evals_per_sample = jax.vmap(
                trial_one
            )(sample, trial_keys)
        else:
            padded_sample, n_pad = pad_to_multiple(
                sample, n_samples, trial_batch_size,
            )
            padded_trial_keys, _ = pad_to_multiple(
                trial_keys, n_samples, trial_batch_size,
            )
            accepted, reasons, n_evals_per_sample, n_grad_evals_per_sample = jax.lax.map(
                lambda x: trial_one(x[0], x[1]),
                (padded_sample, padded_trial_keys),
                batch_size=trial_batch_size,
            )
            if n_pad > 0:
                accepted = accepted[:n_samples]
                reasons = reasons[:n_samples]
                n_evals_per_sample = n_evals_per_sample[:n_samples]
                n_grad_evals_per_sample = n_grad_evals_per_sample[:n_samples]
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


# ---------------------------------------------------------------------------
# adjust_step_size_sharded — sibling for ShardedSingleRun population layout
# ---------------------------------------------------------------------------


def adjust_step_size_sharded(
    population: Any,
    move_fn: Callable,
    step_size: Float[Array, ""],
    emax: Float[Array, ""],
    rng_key: Key[Array, ""],
    n_samples: int,
    min_rate: float,
    max_rate: float,
    adjust_factor: float,
    max_step_size: float,
    max_rounds: int,
    *,
    trial_batch_size: int | None = None,
) -> tuple[
    Float[Array, ""], Float[Array, ""], Int[Array, "4"],
    Int[Array, ""], Bool[Array, ""], Int[Array, ""], Int[Array, ""], Bool[Array, ""],
    Int[Array, ""], Int[Array, ""],
]:
    """Adjust step size with the population sharded across ``n_gpu`` GPUs.

    Sibling of :func:`adjust_step_size`.  Must be called inside a
    ``jax.pmap`` with ``axis_name="shard"`` (see
    :meth:`ShardedSingleRun.wrap_for_batch`).

    Sampling strategy: each shard independently samples ``n_samples``
    walkers from its *local* ``(K_per_gpu, ...)`` slice.  Effective
    trial-pool size is therefore ``G * n_samples`` per round — larger
    than ``SingleRun(K=G*K_per_gpu)`` would use at the same
    ``adjust_n_samples`` config.  Statistically valid (samples are
    independent across shards).

    Aggregation: per-shard accepted / reasons / eval counters are
    ``lax.psum``'d across the shard axis inside the bisection
    ``lax.while_loop`` body, before the rate-vs-window comparison.
    All shards see identical ``rate`` → identical
    ``_process_rate_jax`` decisions → identical updated ``ss`` →
    bisection paths stay coherent across shards.

    Returns ``(final_ss, final_rate, final_counts, n_rounds, converged,
    cap_hits, floor_hits, bracket_detected, trial_n_evals,
    trial_n_grad_evals)`` — values are identical on every shard.
    """
    n_walkers_local = population.energy.shape[0]

    def cond_fn(carry):
        _, _, _, _, round_idx, converged, _, _, _, _, _, _, _ = carry
        return ~converged & (round_idx < max_rounds)

    def body_fn(carry):
        (ss, ss_prev, rate_prev, key, round_idx, converged, _,
         cap_hits, floor_hits, saw_too_high, saw_too_low,
         cum_evals, cum_grad_evals) = carry

        # 1. Sample walkers from LOCAL population.  RNG is independent
        # across shards (split_keys broadcast a single key to all shards
        # at apply-time, but the per-round key advances identically; the
        # `key_sample` derived here is the same on every shard, so every
        # shard picks the SAME local indices — but indexes into
        # DIFFERENT local populations, so the effective sample is
        # already drawn from disjoint walker subsets).
        key, key_sample, key_trials = jax.random.split(key, 3)
        indices = jax.random.choice(
            key_sample, n_walkers_local, shape=(n_samples,), replace=True,
        )
        sample = jax.tree.map(lambda x: x[indices], population)

        # 2. Inject test step size (same as adjust_step_size).
        sample = sample.set(
            step_size=jnp.full(n_samples, ss),
            step_sizes=jnp.broadcast_to(
                ss[None, None], (n_samples, sample.step_sizes.shape[-1]),
            ),
        )

        # 3. Run trial moves on the local sample.
        trial_keys = jax.random.split(key_trials, n_samples)

        def trial_one(state, trial_key):
            _, info = move_fn(state, trial_key, emax)
            return info.accepted, info.reject_reason, info.n_evaluations, info.n_grad_evaluations

        if trial_batch_size is None:
            accepted, reasons, n_evals_per_sample, n_grad_evals_per_sample = jax.vmap(
                trial_one
            )(sample, trial_keys)
        else:
            padded_sample, n_pad = pad_to_multiple(
                sample, n_samples, trial_batch_size,
            )
            padded_trial_keys, _ = pad_to_multiple(
                trial_keys, n_samples, trial_batch_size,
            )
            accepted, reasons, n_evals_per_sample, n_grad_evals_per_sample = jax.lax.map(
                lambda x: trial_one(x[0], x[1]),
                (padded_sample, padded_trial_keys),
                batch_size=trial_batch_size,
            )
            if n_pad > 0:
                accepted = accepted[:n_samples]
                reasons = reasons[:n_samples]
                n_evals_per_sample = n_evals_per_sample[:n_samples]
                n_grad_evals_per_sample = n_grad_evals_per_sample[:n_samples]

        # 4. Cross-shard reduction: accepted-count, per-reason counts,
        # and eval counters are summed across shards so the rate
        # reflects the GLOBAL trial pool (G * n_samples walkers).
        local_accepted = jnp.sum(accepted.astype(jnp.int32))
        global_accepted = jax.lax.psum(local_accepted, axis_name="shard")
        global_n = jax.lax.psum(
            jnp.array(n_samples, dtype=jnp.int32), axis_name="shard",
        )
        rate = global_accepted.astype(jnp.float32) / global_n.astype(jnp.float32)

        local_counts = jnp.array([
            jnp.sum(reasons == 0),
            jnp.sum(reasons == 1),
            jnp.sum(reasons == 2),
            jnp.sum(reasons == 3),
        ], dtype=jnp.int32)
        counts = jax.lax.psum(local_counts, axis_name="shard")

        local_round_evals = jnp.sum(n_evals_per_sample.astype(jnp.int32))
        local_round_grad_evals = jnp.sum(n_grad_evals_per_sample.astype(jnp.int32))
        round_evals = jax.lax.psum(local_round_evals, axis_name="shard")
        round_grad_evals = jax.lax.psum(local_round_grad_evals, axis_name="shard")
        new_cum_evals = cum_evals + round_evals
        new_cum_grad_evals = cum_grad_evals + round_grad_evals

        # 5. Process rate → new step size + convergence flag (identical
        # decision on every shard because `rate` is post-psum).
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
        jnp.array(-1.0),
        rng_key,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

    final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    (final_ss, _, final_rate, _, n_rounds, converged, final_counts,
     cap_hits, floor_hits, saw_too_high, saw_too_low,
     trial_n_evals, trial_n_grad_evals) = final

    bracket_detected = saw_too_high & saw_too_low

    return (final_ss, final_rate, final_counts,
            n_rounds, converged, cap_hits, floor_hits, bracket_detected,
            trial_n_evals, trial_n_grad_evals)
