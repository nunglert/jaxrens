"""Build the per-iteration step-size adapt step.

Public surface: :func:`build_adapt_step`.  Returns either:

  * ``None`` — adaptation is inactive (no moves, no per-move kernels, or
    ``adjust_interval <= 0``).  Caller should skip adaptation entirely.
  * A callable ``adapt_step(ns_state, emax, rng_key) -> (new_ns_state,
    diag, new_rng_key)`` that runs one full per-move bisection pass and
    writes the updated step sizes back into ``ns_state.population``.

The closure carries no run state.  It is constructed once per NS run /
burn-in before the outer loop; the caller owns the firing schedule
(``i > 0 and i % adjust_interval == 0``) and the loop-level RNG carry.

All batcher-specific dispatch is concentrated in this builder.  Each
per-move callable is a single JIT'd in-XLA bisection (``lax.while_loop``)
— same structure for SingleRun / VmapRuns / PmapVmapRuns, with the
sharded variant adding ``lax.psum`` reductions for cross-shard rate
aggregation.

Historical note: this module is named ``manager.py`` for legacy import
reasons.  The previous ``AdaptationManager`` class held no state and has
been collapsed to this builder; rename of the file is a trivial
follow-up.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from jaxrens.sampling.adaptation.stepsize_handler import (
    adjust_step_size,
    adjust_step_size_sharded,
)
from jaxrens.sampling.batch_descriptor import (
    BatchDescriptor,
    ShardedSingleRun,
)
from jaxrens.sampling.move_kernel import MoveKernel

logger = logging.getLogger(__name__)

# Diagnostic keys returned by the per-move callable (indices 1..9 of its
# 10-tuple).  Keep order in sync with ``adjust_step_size`` /
# ``adjust_step_size_sharded``.
_DIAG_KEYS = (
    "rate",
    "counts",
    "n_rounds",
    "converged",
    "cap_hits",
    "floor_hits",
    "bracket_detected",
    "trial_n_evaluations",
    "trial_n_grad_evaluations",
)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_adapt_step(
    move_descriptors: Sequence[MoveKernel],
    per_move_fns: Sequence[Callable] | None,
    batcher: BatchDescriptor,
    adjust_n_samples: int,
    adjust_factor: float,
    adjust_max_rounds: int,
    adjust_interval: int,
    trial_batch_size: int | None = None,
) -> Callable[
    [Any, Float[Array, "*B"], Key[Array, "*B"]],
    tuple[Any, dict, Key[Array, "*B"]],
] | None:
    """Build the per-iteration adapt step.

    Returns ``None`` when adaptation is inactive — when ``per_move_fns``
    or ``move_descriptors`` is empty, or ``adjust_interval <= 0``.
    Callers detect inactivity via ``if adapt_step is None``.

    Otherwise returns ``adapt_step(ns_state, emax, rng_key)`` which:

    1. Extracts current step sizes from ``ns_state.population``.
    2. Iterates over moves: per-move bisection on a fresh trial sample, with
       early exit per-replica once every replica has converged.
    3. Writes the new step sizes back into the population and returns the
       full ``(new_ns_state, diag, new_rng_key)`` tuple.

    ``rng_key`` MUST already match ``batcher.shape_prefix``: scalar for
    ``SingleRun``, ``(R,)`` for ``VmapRuns(R)``, ``(G, P)`` for
    ``PmapVmapRuns(G, P)``, and ``(n_gpu,)`` (broadcast — same key on
    every shard, required for ``lax.psum`` bisection coherence) for
    ``ShardedSingleRun``.  Callers that carry a scalar key (e.g. burn-in)
    must promote it themselves before calling.

    Args:
        move_descriptors: One ``MoveKernel`` per move type.
        per_move_fns: One step function per move type.  ``None`` / empty →
            inactive.
        batcher: ``BatchDescriptor`` (``SingleRun`` / ``VmapRuns`` /
            ``PmapVmapRuns`` / ``ShardedSingleRun``).
        adjust_n_samples: Walkers per bisection trial round (static).
        adjust_factor: Multiplicative step-size factor (static).
        adjust_max_rounds: Max bisection rounds (static).
        adjust_interval: Iterations between fires.  ``<= 0`` → inactive.
        trial_batch_size: Optional chunk for the per-round trial vmap
            (``jax.lax.map(batch_size=...)``).  Bounds peak memory at
            ``trial_batch_size * per-walker tape``.  ``None`` → full vmap.

    Returns:
        ``adapt_step`` closure (signature above), or ``None`` when inactive.

    The ``diag`` dict it returns has keys
    ``rate``/``counts``/``n_rounds``/``converged``/``cap_hits``/
    ``floor_hits``/``bracket_detected``/``trial_n_evaluations``/
    ``trial_n_grad_evaluations``.  Stacked over moves along the axis right
    after the batch prefix: shape ``(n_moves, ...)`` for SingleRun,
    ``(n_runs, n_moves, ...)`` for VmapRuns, ``(G, P, n_moves, ...)`` for
    PmapVmapRuns.
    """
    if not per_move_fns or not move_descriptors or adjust_interval <= 0:
        return None

    descriptors = list(move_descriptors)
    move_fns = list(per_move_fns)

    # === Batcher-dispatch (concentrated here) =============================
    # Pull out the small handful of batcher operations the adapt step needs.
    # The rest of the closure uses these names; no further isinstance checks
    # or polymorphic dispatch live inside the returned callable.
    is_sharded = isinstance(batcher, ShardedSingleRun)
    shape_prefix = batcher.shape_prefix
    stack_axis = len(shape_prefix)
    walker_axis = batcher.walker_axis
    extract_ss = batcher.extract_step_sizes
    broadcast_ss = batcher.broadcast_step_sizes
    split_keys = batcher.split_keys

    # === Per-move JIT callables (one per move type) =======================
    builder = _build_sharded_per_move if is_sharded else _build_per_move
    per_move_jit = [
        builder(
            desc, fn, batcher,
            adjust_n_samples, adjust_factor, adjust_max_rounds,
            trial_batch_size,
        )
        for desc, fn in zip(descriptors, move_fns)
    ]

    # === The actual closure ===============================================
    def adapt_step(ns_state, emax, rng_key):
        pop = ns_state.population
        ss = extract_ss(pop)
        adapt_key = rng_key

        per_move_results: list[tuple] = []

        for move_idx, desc in enumerate(descriptors):
            pairs = split_keys(adapt_key, 2)
            adapt_key = pairs[..., 0]
            key_adjust = pairs[..., 1]
            ss_move = ss[..., move_idx]
            result = per_move_jit[move_idx](pop, ss_move, emax, key_adjust)
            new_ss = result[0]
            ss = ss.at[..., move_idx].set(new_ss)

            if logger.isEnabledFor(logging.DEBUG):
                # result indexing follows _DIAG_KEYS (offset by +1 for new_ss
                # at result[0]): rate=1, n_rounds=3, converged=4, cap_hits=5,
                # floor_hits=6, bracket_detected=7.  Flatten over the batcher
                # prefix so SingleRun/VmapRuns/PmapVmapRuns/ShardedSingleRun
                # all yield 1-D arrays printable as scalars.
                new_ss_flat = jnp.asarray(new_ss).reshape(-1)
                rate_flat = jnp.asarray(result[1]).reshape(-1)
                n_rounds_flat = jnp.asarray(result[3]).reshape(-1)
                converged_flat = jnp.asarray(result[4]).reshape(-1)
                cap_hits_flat = jnp.asarray(result[5]).reshape(-1)
                floor_hits_flat = jnp.asarray(result[6]).reshape(-1)
                bracket_flat = jnp.asarray(result[7]).reshape(-1)
                n_total = int(new_ss_flat.shape[0])
                logger.debug(
                    "adapted %s: ss=%.3e±%.1e  rate=%.3f±%.3f  "
                    "rounds=%d (max)  conv=%d/%d  bracket=%d  floor=%d  cap=%d",
                    desc.name,
                    float(new_ss_flat.mean()), float(new_ss_flat.std()),
                    float(rate_flat.mean()), float(rate_flat.std()),
                    int(n_rounds_flat.max()),
                    int(converged_flat.sum()), n_total,
                    int(bracket_flat.sum()),
                    int(floor_hits_flat.sum()),
                    int(cap_hits_flat.sum()),
                )

            per_move_results.append(result[1:])  # skip new_ss

        diag: dict = {}
        for k, name in enumerate(_DIAG_KEYS):
            diag[name] = jnp.stack(
                [r[k] for r in per_move_results], axis=stack_axis,
            )

        n_walkers = pop.step_sizes.shape[walker_axis]
        new_pop = pop.set(step_sizes=broadcast_ss(ss, n_walkers))
        new_ns_state = ns_state.set(population=new_pop)
        return new_ns_state, diag, adapt_key

    return adapt_step


# ---------------------------------------------------------------------------
# Internal: per-move callable factories
# ---------------------------------------------------------------------------


def _build_per_move(
    desc: MoveKernel,
    move_fn: Callable,
    batcher: BatchDescriptor,
    n_samp: int,
    afac: float,
    max_rounds: int,
    trial_chunk: int | None,
) -> Callable:
    """Per-move closure for SingleRun / VmapRuns / PmapVmapRuns.

    Wraps :func:`adjust_step_size` (single in-XLA ``lax.while_loop``
    bisection) in the batcher's vmap/pmap. The trace-time ``logger.info``
    fires once per JIT cache miss for this move type — the gap between
    that line and the next iteration log is the compile + first-execution
    duration for this move's adapt kernel.
    """
    min_r = desc.min_rate
    max_r = desc.max_rate
    max_ss = desc.step_size_max
    name = desc.name

    def _per_replica(
        pop, ss, emax, key,
        _move_fn=move_fn,
        _n_samp=n_samp,
        _min_r=min_r,
        _max_r=max_r,
        _afac=afac,
        _max_ss=max_ss,
        _max_rounds=max_rounds,
        _trial_chunk=trial_chunk,
        _desc_name=name,
    ):
        logger.info(
            "adapt tracing: move=%s  pop_shape=%s  max_neighbors=%d  "
            "n_samp=%d  max_rounds=%d",
            _desc_name,
            pop.positions.shape,
            int(pop.max_neighbors),
            int(_n_samp),
            int(_max_rounds),
        )
        if _trial_chunk is None:
            return adjust_step_size(
                pop, _move_fn, ss, emax, key,
                _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
            )
        return adjust_step_size(
            pop, _move_fn, ss, emax, key,
            _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
            trial_batch_size=_trial_chunk,
        )

    return batcher.wrap_for_batch(_per_replica)


def _build_sharded_per_move(
    desc: MoveKernel,
    move_fn: Callable,
    batcher: BatchDescriptor,
    n_samp: int,
    afac: float,
    max_rounds: int,
    trial_chunk: int | None,
) -> Callable:
    """Per-move closure for ShardedSingleRun.

    Wraps :func:`adjust_step_size_sharded` (in-XLA ``lax.while_loop`` with
    cross-shard ``lax.psum`` reductions inside the body) in the
    sharded-pmap batcher.
    """
    min_r = desc.min_rate
    max_r = desc.max_rate
    max_ss = desc.step_size_max
    name = desc.name

    def _per_replica(
        pop, ss, emax, key,
        _move_fn=move_fn,
        _n_samp=n_samp,
        _min_r=min_r,
        _max_r=max_r,
        _afac=afac,
        _max_ss=max_ss,
        _max_rounds=max_rounds,
        _trial_chunk=trial_chunk,
        _desc_name=name,
    ):
        logger.info(
            "adapt tracing (sharded): move=%s  pop_shape=%s  "
            "max_neighbors=%d  n_samp=%d  max_rounds=%d",
            _desc_name,
            pop.positions.shape,
            int(pop.max_neighbors),
            int(_n_samp),
            int(_max_rounds),
        )
        if _trial_chunk is None:
            return adjust_step_size_sharded(
                pop, _move_fn, ss, emax, key,
                _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
            )
        return adjust_step_size_sharded(
            pop, _move_fn, ss, emax, key,
            _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
            trial_batch_size=_trial_chunk,
        )

    return batcher.wrap_for_batch(_per_replica)
