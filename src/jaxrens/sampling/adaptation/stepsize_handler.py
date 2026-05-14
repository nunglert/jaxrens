"""Step size adjustment via trial moves — pure JAX functions.

Two production entry points, one per batcher regime:

* :func:`_one_bisection_round` — one bisection round on a pre-gathered
  trial sample.  The Python loop in
  :func:`jaxrens.sampling.adaptation.manager.build_adapt_step` drives it
  for SingleRun / VmapRuns / PmapVmapRuns.  Removes one outer ``while``
  from the gradient body's XLA nesting (see WORKLOG 2026-05-14).

* :func:`adjust_step_size_sharded` — full in-XLA bisection
  (``lax.while_loop``) with cross-shard ``lax.psum`` inside the body.
  Used for ShardedSingleRun, where the psum has to live inside the
  pmap'd region.

Both share :func:`_process_rate_jax` (branchless rate-window decision)
and :func:`_gather_trial_walkers` (single random.choice + indexed
gather over the K-walker population).  The outer loop (when to trigger,
how to write back) lives in ``build_adapt_step`` / ``_run_loop``.
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


def _gather_trial_walkers(
    population: Any,
    rng_key: Key[Array, ""],
    n_samples: int,
) -> Any:
    """Sample ``n_samples`` walker indices from the K-population and gather.

    Hoisted out of :func:`_one_bisection_round` so the JIT body sees a
    ``(n_samples, ...)`` input shape instead of doing a 500→50 indexing
    pattern inside the gradient body.  Cheap to JIT-wrap (one
    ``random.choice`` + one indexed gather per leaf); the compile is
    shared across all bisection rounds via JAX's per-input-shape cache.
    """
    n_walkers = population.energy.shape[0]
    indices = jax.random.choice(
        rng_key, n_walkers, shape=(n_samples,), replace=True,
    )
    return jax.tree.map(lambda x: x[indices], population)


def _one_bisection_round(
    sample: Any,
    move_fn: Callable,
    step_size: Float[Array, ""],
    step_size_prev: Float[Array, ""],
    rate_prev: Float[Array, ""],
    rng_key: Key[Array, ""],
    emax: Float[Array, ""],
    n_samples: int,
    min_rate: float,
    max_rate: float,
    adjust_factor: float,
    max_step_size: float,
    *,
    trial_batch_size: int | None = None,
):
    """One bisection round — JIT-friendly body driven by the Python loop in
    :func:`jaxrens.sampling.adaptation.manager.build_adapt_step`.

    Trial-vmap, rate computation, and :func:`_process_rate_jax` only.  The
    ``sample`` argument is the **pre-gathered** ``n_samples``-walker
    MCState: walker selection from the K-population happens at Python
    level above the JIT boundary so the body sees an input shape matching
    ``ns_step`` (only the walkers actually being walked enter the JIT
    body, not the full K-population that gets indexed down inside).

    Used by :func:`jaxrens.sampling.adaptation.manager.build_adapt_step` —
    that builder wraps this body in ``batcher.wrap_for_batch`` (JIT'd) and
    drives the bisection at Python level above the JIT boundary.  This
    structure removes one outer ``while`` from the gradient body's XLA
    nesting and the in-body gather pattern — the differences that
    explained why ``ns_step`` compiles fast at bucket=25 while the
    previous in-JIT bisection hung (see WORKLOG 2026-05-14).

    Returns:
        ``(new_step_size, new_step_size_prev, new_rate_prev, new_key,
        new_converged_round, rate, counts, cap_hit, floor_hit, too_high,
        too_low, round_evals, round_grad_evals)``.  ``new_*_prev`` and
        ``new_rate_prev`` are the unupdated previous-round values that the
        driver should adopt iff convergence has not yet occurred — masking
        is the driver's responsibility.
    """
    # 1. (Gather moved to Python driver above the JIT boundary.)

    # 2. Inject test step size into both ``step_size`` (scalar) and
    # ``step_sizes`` (per-move array).  The MWG wrapper reads
    # ``state.step_sizes[move_idx]`` rather than ``state.step_size``,
    # so we must update the array as well.  Broadcasting ``ss`` into
    # all positions is safe because the trial function only calls one
    # specific move kernel per ``_one_bisection_round`` call.
    sample = sample.set(
        step_size=jnp.full(n_samples, step_size),
        step_sizes=jnp.broadcast_to(
            step_size[None, None], (n_samples, sample.step_sizes.shape[-1]),
        ),
    )

    # 3. Run trial moves
    key, key_trials = jax.random.split(rng_key)
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
        padded_sample, n_pad = pad_to_multiple(sample, n_samples, trial_batch_size)
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

    counts = jnp.array([
        jnp.sum(reasons == 0),
        jnp.sum(reasons == 1),
        jnp.sum(reasons == 2),
        jnp.sum(reasons == 3),
    ], dtype=jnp.int32)
    round_evals = jnp.sum(n_evals_per_sample.astype(jnp.int32))
    round_grad_evals = jnp.sum(n_grad_evals_per_sample.astype(jnp.int32))

    # 4. Process rate → new step size + convergence flag
    new_ss, new_converged_round, cap_hit, floor_hit, too_high, too_low = _process_rate_jax(
        rate, step_size, step_size_prev, rate_prev,
        min_rate, max_rate, adjust_factor, max_step_size,
    )

    return (
        new_ss,                  # 0: proposed new step size (driver: next ss carry)
        step_size,               # 1: input ss → next ss_prev carry
        rate,                    # 2: this round's rate → next rate_prev carry
        key,                     # 3: advanced rng_key
        new_converged_round,     # 4: did this round trip convergence?
        counts,                  # 5: per-reason counts (4,)
        cap_hit,                 # 6: bool
        floor_hit,               # 7: bool
        too_high,                # 8: bool
        too_low,                 # 9: bool
        round_evals,             # 10: int32 — backend calls this round
        round_grad_evals,        # 11: int32 — grad calls this round
    )


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

    The cross-shard counterpart to :func:`_one_bisection_round`'s
    Python-driven bisection: this one keeps the bisection inside an
    in-XLA ``lax.while_loop`` so the per-shard ``lax.psum`` calls live
    inside the pmap'd region.  Must be called inside a ``jax.pmap`` with
    ``axis_name="shard"`` (see :meth:`ShardedSingleRun.wrap_for_batch`).

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

        # 2. Inject test step size (same as _one_bisection_round).
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
