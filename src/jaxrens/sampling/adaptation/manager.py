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

All batcher-specific dispatch is concentrated in this builder — the
returned closure runs the same Python-driver bisection for SingleRun /
VmapRuns / PmapVmapRuns, and a separate legacy in-XLA bisection for
ShardedSingleRun (whose ``lax.psum`` reduction doesn't compose with a
Python-level driver).

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
    _gather_trial_walkers,
    _one_bisection_round,
    adjust_step_size_sharded,
)
from jaxrens.sampling.batch_descriptor import (
    BatchDescriptor,
    ShardedSingleRun,
)
from jaxrens.sampling.move_kernel import MoveKernel

logger = logging.getLogger(__name__)

# Diagnostic keys returned by the per-move callable (indices 1..9 of its
# 10-tuple).  Keep order in sync with ``_one_bisection_round`` /
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
    2. Promotes ``rng_key`` from scalar to ``shape_prefix``-shaped if needed
       (broadcast for ShardedSingleRun, split for VmapRuns / PmapVmapRuns).
    3. Iterates over moves: per-move bisection on a fresh trial sample, with
       early exit per-replica once every replica has converged.
    4. Writes the new step sizes back into the population and returns the
       full ``(new_ns_state, diag, new_rng_key)`` tuple.

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
    promote_key = _make_key_promoter(batcher)

    # === Per-move JIT callables (one per move type) =======================
    if is_sharded:
        per_move_jit = [
            _build_sharded_per_move(
                desc, fn, batcher,
                adjust_n_samples, adjust_factor, adjust_max_rounds,
                trial_batch_size,
            )
            for desc, fn in zip(descriptors, move_fns)
        ]
    else:
        per_move_jit = [
            _build_python_driver_per_move(
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
        adapt_key = promote_key(rng_key)

        per_move_results: list[tuple] = []
        log_per_move = not batcher.is_batched

        for move_idx, desc in enumerate(descriptors):
            pairs = split_keys(adapt_key, 2)
            adapt_key = pairs[..., 0]
            key_adjust = pairs[..., 1]
            ss_move = ss[..., move_idx]
            result = per_move_jit[move_idx](pop, ss_move, emax, key_adjust)
            new_ss = result[0]
            ss = ss.at[..., move_idx].set(new_ss)

            if log_per_move:
                logger.debug(
                    "Adjusted %s: ss=%.4g rate=%.3f rounds=%d converged=%s",
                    desc.name, float(new_ss), float(result[1]),
                    int(result[3]), bool(result[4]),
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
# Internal: key promotion (scalar → shape_prefix-shaped, batcher-specific)
# ---------------------------------------------------------------------------


def _make_key_promoter(batcher: BatchDescriptor) -> Callable[[Any], Any]:
    """Return a function that promotes a scalar key to ``shape_prefix`` shape.

    * SingleRun:        identity.
    * VmapRuns(R):      ``split(key, R) → (R,)``.
    * PmapVmapRuns(G,P):``split(key, G*P).reshape(G, P)``.
    * ShardedSingleRun: BROADCAST (not split) onto the ``'shard'`` mesh —
      coherent bisection requires identical RNG on every shard so each
      shard's ``lax.psum``'d rate is over a shared trial decision.

    Shape-correct inputs (already shape_prefix-shaped) flow through
    unchanged so the NS loop's pre-shaped key works alongside burn-in's
    scalar-key chain.
    """
    prefix = batcher.shape_prefix
    if not prefix:
        return lambda k: k

    if isinstance(batcher, ShardedSingleRun):
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        n_gpu = batcher.n_gpu
        shard_mesh = NamedSharding(
            Mesh(jax.local_devices()[:n_gpu], ("shard",)),
            PartitionSpec("shard"),
        )

        def promote_sharded(rng_key):
            key_ndim = jnp.asarray(rng_key).ndim
            if key_ndim != 0:
                return rng_key
            return jax.device_put(
                jnp.broadcast_to(rng_key[None], (n_gpu,)), shard_mesh,
            )

        return promote_sharded

    def promote_split(rng_key):
        key_ndim = jnp.asarray(rng_key).ndim
        if key_ndim != 0:
            return rng_key
        return jax.random.split(rng_key, batcher.n_runs).reshape(prefix)

    return promote_split


# ---------------------------------------------------------------------------
# Internal: per-move callable factories
# ---------------------------------------------------------------------------


def _build_python_driver_per_move(
    desc: MoveKernel,
    move_fn: Callable,
    batcher: BatchDescriptor,
    n_samp: int,
    afac: float,
    max_rounds: int,
    trial_chunk: int | None,
) -> Callable:
    """Per-move closure for SingleRun / VmapRuns / PmapVmapRuns.

    Two small JIT'd kernels:

    * ``_per_replica_gather`` — splits the key, draws ``n_samp`` walkers,
      returns ``(sample, advanced_key)``.
    * ``_per_replica_one_round`` — runs ONE bisection round on the
      pre-gathered sample.

    A Python loop above the JIT boundary drives up to ``max_rounds`` rounds
    with early exit on full convergence.  This structure removes one outer
    ``while`` from the gradient body's XLA nesting (matches ``ns_step``'s
    pattern; see WORKLOG 2026-05-14).
    """
    min_r = desc.min_rate
    max_r = desc.max_rate
    max_ss = desc.step_size_max
    name = desc.name

    def _per_replica_gather(pop, key, _n_samp=n_samp):
        # Split key inside the JIT so the Python driver only sees per-
        # replica-shaped keys (splitting along a batch axis above the JIT
        # boundary is awkward under pmap+vmap).  Returns advanced key.
        key, key_sample = jax.random.split(key)
        sample = _gather_trial_walkers(pop, key_sample, _n_samp)
        return sample, key

    def _per_replica_one_round(
        sample, ss, ss_prev, rate_prev, key, emax,
        _move_fn=move_fn,
        _n_samp=n_samp,
        _min_r=min_r,
        _max_r=max_r,
        _afac=afac,
        _max_ss=max_ss,
        _trial_chunk=trial_chunk,
        _desc_name=name,
    ):
        # Trace-time logger fires once per JIT cache miss of the one-round
        # body (i.e. once per distinct signature per move type).
        #
        # When two trace lines fire back-to-back for the same move, diff
        # the ``leaves=`` summary below — the differing entry is the cache
        # key that flipped (shape, dtype, or weak_type).  If the leaves
        # look identical, compare ``treedef_hash`` (pytree structure) and
        # ``fn_id`` (the wrapped function's Python id — if it differs the
        # JIT wrapper itself was rebuilt, so cache was never reusable).
        _args_pytree = (sample, ss, ss_prev, rate_prev, key, emax)
        _leaves_flat, _treedef = jax.tree_util.tree_flatten(_args_pytree)
        def _fmt(v):
            return (
                f"{getattr(v, 'shape', '?')}:{getattr(v, 'dtype', type(v).__name__)}"
                f"{'(weak)' if getattr(v, 'weak_type', False) else ''}"
            )
        _leaf_str = "  ".join(_fmt(v) for v in _leaves_flat)
        logger.info(
            "adapt _per_replica_one_round tracing: move=%s  "
            "sample_shape=%s  max_neighbors=%d  n_samp=%d  "
            "n_leaves=%d  treedef_hash=%x  fn_id=%x  move_fn_id=%x  "
            "leaves=[%s]",
            _desc_name,
            sample.positions.shape,
            int(sample.max_neighbors),
            int(_n_samp),
            len(_leaves_flat),
            hash(_treedef) & 0xFFFFFFFF,
            id(_per_replica_one_round) & 0xFFFFFFFF,
            id(_move_fn) & 0xFFFFFFFF,
            _leaf_str,
        )
        return _one_bisection_round(
            sample, _move_fn, ss, ss_prev, rate_prev, key, emax,
            _n_samp, _min_r, _max_r, _afac, _max_ss,
            trial_batch_size=_trial_chunk,
        )

    jit_gather = batcher.wrap_for_batch(_per_replica_gather)
    jit_one_round = batcher.wrap_for_batch(_per_replica_one_round)

    def _driver(
        pop, ss, emax, key,
        _jit_gather=jit_gather,
        _jit_one_round=jit_one_round,
        _max_rounds=max_rounds,
    ):
        # Python-level bisection above the JIT boundary.  Batch-axis
        # shapes (scalar / (R,) / (G, P)) flow through unchanged: every
        # operand to ``_jit_one_round`` carries the batcher's shape
        # prefix and the JIT body strips it via vmap/pmap.
        ss_carry = ss
        ss_prev = ss
        rate_prev = jnp.broadcast_to(
            jnp.array(-1.0, dtype=jnp.float32), ss.shape,
        )
        converged = jnp.zeros(ss.shape, dtype=jnp.bool_)

        final_rate = jnp.zeros(ss.shape, dtype=jnp.float32)
        final_counts = jnp.zeros(ss.shape + (4,), dtype=jnp.int32)
        cap_hits = jnp.zeros(ss.shape, dtype=jnp.int32)
        floor_hits = jnp.zeros(ss.shape, dtype=jnp.int32)
        saw_too_high = jnp.zeros(ss.shape, dtype=jnp.bool_)
        saw_too_low = jnp.zeros(ss.shape, dtype=jnp.bool_)
        trial_n_evals = jnp.zeros(ss.shape, dtype=jnp.int32)
        trial_n_grad_evals = jnp.zeros(ss.shape, dtype=jnp.int32)
        n_rounds = jnp.zeros(ss.shape, dtype=jnp.int32)

        for _ in range(_max_rounds):
            # One device-to-host scalar sync per round — microseconds.
            if bool(jnp.all(converged)):
                break

            sample, key = _jit_gather(pop, key)
            (new_ss_prop, new_ss_prev_prop, new_rate_prev_prop,
             key, new_conv_round, counts, cap_hit, floor_hit,
             too_high, too_low, round_evals, round_grad_evals) = \
                _jit_one_round(
                    sample, ss_carry, ss_prev, rate_prev, key, emax,
                )

            active = ~converged

            ss_carry = jnp.where(active, new_ss_prop, ss_carry)
            ss_prev = jnp.where(active, new_ss_prev_prop, ss_prev)
            rate_prev = jnp.where(active, new_rate_prev_prop, rate_prev)

            cap_hits = cap_hits + jnp.where(
                active, cap_hit.astype(jnp.int32), 0,
            )
            floor_hits = floor_hits + jnp.where(
                active, floor_hit.astype(jnp.int32), 0,
            )
            saw_too_high = saw_too_high | (too_high & active)
            saw_too_low = saw_too_low | (too_low & active)
            trial_n_evals = trial_n_evals + jnp.where(
                active, round_evals, 0,
            )
            trial_n_grad_evals = trial_n_grad_evals + jnp.where(
                active, round_grad_evals, 0,
            )
            n_rounds = n_rounds + jnp.where(active, 1, 0)

            final_rate = jnp.where(active, rate_prev, final_rate)
            final_counts = jnp.where(
                active[..., None], counts, final_counts,
            )

            converged = converged | new_conv_round

        bracket_detected = saw_too_high & saw_too_low

        return (
            ss_carry, final_rate, final_counts,
            n_rounds, converged, cap_hits, floor_hits, bracket_detected,
            trial_n_evals, trial_n_grad_evals,
        )

    return _driver


def _build_sharded_per_move(
    desc: MoveKernel,
    move_fn: Callable,
    batcher: BatchDescriptor,
    n_samp: int,
    afac: float,
    max_rounds: int,
    trial_chunk: int | None,
) -> Callable:
    """Per-move closure for ShardedSingleRun (legacy in-XLA bisection).

    Cross-shard ``lax.psum`` inside the bisection body doesn't compose
    with the Python-driver pattern (the psum has to live inside the pmap'd
    body), so we keep the legacy in-XLA ``lax.while_loop`` path here.
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
        _args_pytree = (pop, ss, emax, key)
        _leaves_flat, _ = jax.tree_util.tree_flatten(_args_pytree)
        def _fmt(v):
            return (
                f"{getattr(v, 'shape', '?')}:{getattr(v, 'dtype', type(v).__name__)}"
                f"{'(weak)' if getattr(v, 'weak_type', False) else ''}"
            )
        _leaf_str = "  ".join(_fmt(v) for v in _leaves_flat)
        logger.info(
            "adapt _per_replica (sharded) tracing: move=%s  "
            "pop_shape=%s  max_neighbors=%d  n_samp=%d  max_rounds=%d  "
            "n_leaves=%d  leaves=[%s]",
            _desc_name,
            pop.positions.shape,
            int(pop.max_neighbors),
            int(_n_samp),
            int(_max_rounds),
            len(_leaves_flat),
            _leaf_str,
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
