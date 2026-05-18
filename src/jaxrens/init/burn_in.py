"""Fixed-Emax MCMC burn-in to decorrelate live walkers before NS proper.

The burn-in runs the existing MWG step_fn at a constant Emax computed once
from the live-walker population. No new MCMC kernel; no modification to ns_step.

Public API:
    initial_walk(key, ns_state, step_fn, ..., batcher=None)  ->  NSState

The ``batcher`` argument routes the run-axis dispatch:

* ``SingleRun`` (default) — single-run burn-in.
* ``VmapRuns(R)`` — ``jax.vmap`` over R runs.
* ``PmapVmapRuns(G, P)`` — ``jax.pmap(jax.vmap(...))`` over G GPUs × P runs.

Parallelism axes:
    walker_batch_size: chunk the walker vmap via lax.map(batch_size=N).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import (
    BatchDescriptor,
    ShardedSingleRun,
    SingleRun,
    VmapRuns,
)
from jaxrens.sampling.bucket_manager import BucketManager
from jaxrens.state.ns import NSState
from jaxrens.utils.padding import pad_to_multiple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _one_walk(
    key: jax.Array,
    ns_state: NSState,
    step_fn: Callable,
    walklength: int,
    emax: jnp.ndarray,
    walker_batch_size: int | None,
) -> tuple[NSState, Any]:
    """Advance all walkers in a single-run NSState by walklength steps.

    Args:
        key: PRNG key. Consumed to produce (n_walkers, walklength) sub-keys.
        ns_state: Single-run NSState. Population shape (n_walkers, ...).
        step_fn: MWG step fn. Signature: (rng_key, mcstate, emax) -> (mcstate, info).
        walklength: Number of MCMC steps per walker.
        emax: Scalar energy ceiling.
        walker_batch_size: If None, vmap over all walkers. If int, chunk via
            lax.map(batch_size=walker_batch_size). Any positive int is
            accepted; non-divisors are handled by padding the population with
            copies of the last walker and slicing the padding off the output.

    Returns:
        (new_ns_state, last_accepted) where last_accepted is (n_walkers, walklength).
    """
    # Fires once per cache miss; subsequent walks with identical shapes /
    # static fields reuse the compiled binary and stay silent.  A second
    # entry after iter-N indicates the burn-in bucket changed (overflow
    # retry).
    logger.info(
        "burnin _one_walk tracing: pop_shape=%s  max_neighbors=%d  "
        "walklength=%d  walker_batch_size=%s",
        ns_state.population.positions.shape,
        int(ns_state.population.max_neighbors),
        int(walklength),
        walker_batch_size,
    )
    population = ns_state.population
    # Use the leading dim of the local population, not ``ns_state.n_walkers``.
    # ``n_walkers`` is a static field that holds the GLOBAL count (matters for
    # ShardedSingleRun: per-shard local count is ``K // G`` while
    # ``ns_state.n_walkers == K``).  All other batchers run with local ==
    # global so the two agree.
    n_walkers = population.energy.shape[0]

    walker_keys = jax.random.split(key, n_walkers)
    # chain_keys: (n_walkers, walklength)
    chain_keys = jax.vmap(lambda k: jax.random.split(k, walklength))(walker_keys)

    def walker_fn(walker_state: Any, wkeys: jax.Array) -> tuple[Any, Any]:
        """Scan step_fn over walklength keys for one walker."""
        def scan_body(state, k):
            new_state, info = step_fn(k, state, emax)
            return new_state, info.accepted
        final, accepted_arr = jax.lax.scan(scan_body, walker_state, wkeys)
        return final, accepted_arr

    if walker_batch_size is None:
        new_population, accepted = jax.vmap(walker_fn)(population, chain_keys)
    else:
        # lax.map processes (population[i], chain_keys[i]) pairs in chunks.
        # Pad to a multiple of walker_batch_size when n_walkers isn't a
        # divisor; padded slots are stepped through and discarded.
        padded_pop, n_pad = pad_to_multiple(
            population, n_walkers, walker_batch_size
        )
        padded_keys, _ = pad_to_multiple(
            chain_keys, n_walkers, walker_batch_size
        )
        new_pop_padded, accepted_padded = jax.lax.map(
            lambda x: walker_fn(x[0], x[1]),
            (padded_pop, padded_keys),
            batch_size=walker_batch_size,
        )
        if n_pad == 0:
            new_population, accepted = new_pop_padded, accepted_padded
        else:
            new_population = jax.tree.map(
                lambda x: x[:n_walkers], new_pop_padded
            )
            accepted = accepted_padded[:n_walkers]

    return ns_state.set(population=new_population), accepted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initial_walk(
    key: jax.Array,
    ns_state: NSState,
    step_fn: Callable,
    *,
    n_walks: int,
    walklength: int,
    adjust_interval: int,
    emax_offset_per_atom: float,
    n_atoms: int,
    batcher: BatchDescriptor | None = None,
    batched: bool | None = None,  # deprecated; pass ``batcher`` instead
    walker_batch_size: int | None = None,
    per_move_fns: list[Callable] | None = None,
    adaptation_policies: tuple | None = None,
    adjust_n_samples: int = 50,
    adjust_max_rounds: int = 15,
    max_neighbors_list: tuple[int, ...] = (30, 35, 40, 45, 50),
    max_neighbors_offset: int = 5,
    max_neighbors_shrink_dwell: int = 0,
    max_neighbors_shrink_margin: int | None = None,
) -> NSState:
    """Run fixed-Emax MCMC to decorrelate walkers from their initialization.

    Emax = max(live_energies) + emax_offset_per_atom * n_atoms, held constant
    across all n_walks * walklength steps.

    Parallelism:
        walker_batch_size=None: vmap over all walkers simultaneously (fastest).
        walker_batch_size=N: chunk via lax.map(batch_size=N). Any positive
            int; non-divisors are handled internally by padding.

    Args:
        key: JAX PRNG key.
        ns_state: NSState after init_ns / init_ns_parallel / init_ns_multi_gpu.
            Population shape ``(*shape_prefix, n_walkers, ...)`` per the
            descriptor (``()`` for SingleRun, ``(R,)`` for VmapRuns,
            ``(G, P)`` for PmapVmapRuns).
        step_fn: MWG step function from build_mwg().
            Signature: (rng_key, mcstate, emax) -> (mcstate, MoveInfo).
        n_walks: Number of outer walk iterations. n_walks=0 is a no-op.
        walklength: MCMC steps per walk per walker.
        adjust_interval: Run step-size adaptation every this many walks (>0).
        emax_offset_per_atom: Added to max(live_energies)/max across runs
            to set Emax. Units: energy per atom.
        n_atoms: Number of atoms per walker (Python int).
        batcher: ``BatchDescriptor`` selecting the run-axis dispatch.
            Defaults to ``SingleRun()`` for back-compat.  Mutually exclusive
            with the deprecated ``batched`` boolean.
        batched: **Deprecated.** When set, synthesises a ``VmapRuns`` from
            the input shape and emits a ``DeprecationWarning``.  Pass
            ``batcher`` instead.
        walker_batch_size: Chunk size for walker vmap. None = full vmap.
            Any positive int; non-divisors are handled internally by padding.
        per_move_fns: Per-move step functions (third return of build_mwg).
            Required for step-size adaptation; None disables it.
        adaptation_policies: Tuple of ResolvedAdaptationPolicy, one per move.
            Required when per_move_fns is not None.
        adjust_n_samples: Number of trial walkers per adaptation round.
        adjust_max_rounds: Max bisection rounds per adaptation call.
        max_neighbors_list: Bucket ladder for neighbor-overflow retries.
            On overflow after a walk the next entry in this ladder is
            selected via ``_pick_next_bucket`` and the walk is retried.
            Mirrors the NS-loop parameter of the same name.
        max_neighbors_offset: Headroom added to the observed peak when
            picking the next bucket.  Mirrors the NS-loop parameter.
        max_neighbors_shrink_dwell: Hysteresis dwell window for downsizing
            the bucket — see :class:`~jaxrens.sampling.bucket_manager.BucketManager`.
            ``0`` (default) disables shrinking, preserving the existing
            growth-only behaviour.  Mirrors the NS-loop parameter.
        max_neighbors_shrink_margin: Hysteresis gap below the next-smaller
            ladder entry that must hold for ``shrink_dwell`` iterations
            before the bucket is shrunk.  ``None`` reuses
            ``max_neighbors_offset``.  Mirrors the NS-loop parameter.

    Returns:
        New NSState with live walkers advanced. Same pytree shape as input.
        Dead-point arrays, log_evidence, iteration, n_dead are unchanged.
    """
    if batched is not None:
        warnings.warn(
            "initial_walk(batched=...) is deprecated; pass `batcher` instead "
            "(SingleRun / VmapRuns / PmapVmapRuns).",
            DeprecationWarning,
            stacklevel=2,
        )
        if batcher is not None:
            raise ValueError(
                "initial_walk: pass either `batcher` or the deprecated "
                "`batched` flag, not both."
            )
        if batched:
            batcher = VmapRuns(n_runs=int(ns_state.population.energy.shape[0]))
        else:
            batcher = SingleRun()
    if batcher is None:
        batcher = SingleRun()

    if n_walks == 0:
        return ns_state

    n_walkers = int(ns_state.population.energy.shape[batcher.walker_axis])

    # --- Compute fixed Emax (per replica) ---
    population = ns_state.population
    emax_offset = float(emax_offset_per_atom) * n_atoms
    emax = batcher.reduce_emax(population.energy) + emax_offset

    # --- Setup adaptation ---
    use_adaptation = (
        per_move_fns is not None
        and adaptation_policies is not None
        and adjust_interval > 0
    )

    if use_adaptation:
        # ``build_adapt_step`` only reads ``name``/``min_rate``/
        # ``max_rate``/``step_size_max`` from each move descriptor.
        # Burn-in receives ``ResolvedAdaptationPolicy`` (no ``name``);
        # construct a real :class:`MoveKernel` with a no-op
        # ``build_kernel`` (the builder never calls it — only reads the
        # four rate/cap fields).  ``MoveKernel`` is a frozen dataclass
        # with strict beartype-checked typing on its sequence
        # consumers, so a real instance is required.  Per-move
        # ``adjust_factor`` collapses to a single value (the builder's
        # contract) — burn-in adopts the first policy's factor,
        # mirroring how the NS-loop adapt step is constructed in
        # ``cli/run.py``.
        from jaxrens.sampling.move_kernel import MoveKernel

        def _noop_build_kernel(*_args, **_kwargs):  # pragma: no cover
            raise RuntimeError(
                "burn-in's MoveKernel shim build_kernel was invoked — "
                "build_adapt_step should only read static fields."
            )

        descs = [
            MoveKernel(
                name=f"move_{i}",
                build_kernel=_noop_build_kernel,
                min_rate=p.min_rate,
                max_rate=p.max_rate,
                step_size_max=p.step_size_max,
            )
            for i, p in enumerate(adaptation_policies)
        ]
        adjust_factor = float(adaptation_policies[0].adjust_factor)
        # Propagate burn-in's ``walker_batch_size`` to the adapt step as
        # ``trial_batch_size`` — same memory motivation (chunk the per-
        # round trial vmap).  Zero-overhead when ``None``.
        adapt_step = build_adapt_step(
            move_descriptors=descs,
            per_move_fns=per_move_fns,
            batcher=batcher,
            adjust_n_samples=adjust_n_samples,
            adjust_factor=adjust_factor,
            adjust_max_rounds=adjust_max_rounds,
            adjust_interval=adjust_interval,
            trial_batch_size=walker_batch_size,
        )
    else:
        adapt_step = None

    # --- Build one-walk function via the descriptor ---
    # ``wrap_for_batch`` adds the right jit / vmap / pmap composition for
    # the chosen descriptor (identity for SingleRun, vmap for VmapRuns,
    # pmap-of-vmap for PmapVmapRuns).
    def _per_replica(k, run_state, run_emax):
        return _one_walk(
            k, run_state, step_fn, walklength, run_emax, walker_batch_size,
        )

    jit_one_walk = batcher.wrap_for_batch(_per_replica)

    # Bucket-ladder manager shared with ``_run_loop``.  Growth always
    # active; shrink path opt-in via ``shrink_dwell > 0``.
    bucket_mgr = BucketManager(
        ladder=max_neighbors_list,
        offset=max_neighbors_offset,
        shrink_dwell=max_neighbors_shrink_dwell,
        shrink_margin=max_neighbors_shrink_margin,
    )

    # --- Outer walk loop (while-loop so overflow retries don't advance walk_i) ---
    walk_i = 0
    while walk_i < n_walks:
        if adapt_step is not None and walk_i > 0 and walk_i % adjust_interval == 0:
            key, key_adapt = jax.random.split(key)
            # adapt_step requires a shape-prefix-shaped key; burn-in
            # carries a scalar so promote here.  Sharded: BROADCAST
            # (identical per shard — required for lax.psum coherence).
            # Other batched: SPLIT (independent per replica).
            if isinstance(batcher, ShardedSingleRun):
                key_adapt = jnp.broadcast_to(
                    key_adapt, (batcher.n_gpu,) + key_adapt.shape,
                )
            elif batcher.is_batched:
                key_adapt = jax.random.split(
                    key_adapt, batcher.n_runs,
                ).reshape(batcher.shape_prefix)
            ns_state, _diag, _new_key = adapt_step(
                ns_state, emax, key_adapt,
            )

        key, sub = jax.random.split(key)

        # ``distinct_keys`` returns shape_prefix-shaped INDEPENDENT keys
        # (scalar for SingleRun) on the right mesh per batcher.  Each
        # replica/shard walks its own walkers.
        walk_keys = batcher.distinct_keys(sub)
        new_ns_state, _ = jit_one_walk(walk_keys, ns_state, emax)

        # Bucket overflow → grow and retry the same walk.  JAX re-traces
        # ``jit_one_walk`` automatically because ``max_neighbors`` is a
        # static field on the MCState pytree.  Counter ``walk_i`` is not
        # advanced.
        ns_state, retry = bucket_mgr.grow_if_overflow(
            ns_state, new_ns_state,
            label="burn-in walk", iteration=walk_i,
        )
        if retry:
            continue

        # Optional shrink path — no-op when ``shrink_dwell == 0``.
        ns_state = bucket_mgr.maybe_shrink(ns_state, iteration=walk_i)
        walk_i += 1

    return ns_state
