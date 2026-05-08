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
    run_batch_size: chunk the run vmap via lax.map(batch_size=M).  Only
        applies to ``VmapRuns``; ignored for SingleRun and PmapVmapRuns.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jaxrens.sampling.adaptation.stepsize_handler import adjust_step_size
from jaxrens.sampling.batch_descriptor import (
    BatchDescriptor,
    SingleRun,
    VmapRuns,
)
from jaxrens.state.ns import NSState


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
            lax.map(batch_size=walker_batch_size). Must divide n_walkers.

    Returns:
        (new_ns_state, last_accepted) where last_accepted is (n_walkers, walklength).
    """
    population = ns_state.population
    n_walkers = ns_state.n_walkers

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
        new_population, accepted = jax.lax.map(
            lambda x: walker_fn(x[0], x[1]),
            (population, chain_keys),
            batch_size=walker_batch_size,
        )

    return ns_state.set(population=new_population), accepted


def _apply_adaptation(
    ns_state: NSState,
    emax: jnp.ndarray,
    key: jax.Array,
    per_move_fns: list[Callable],
    adaptation_policies: tuple,
    current_step_sizes: jnp.ndarray,
    adjust_n_samples: int,
    adjust_max_rounds: int,
    jit_adjust_fns: list[Callable],
    batcher: BatchDescriptor,
    run_batch_size: int | None = None,
) -> tuple[NSState, jnp.ndarray]:
    """Update step sizes for all move types.

    Routing is descriptor-driven via ``batcher``:

    * ``SingleRun`` — direct call (no vmap).
    * ``VmapRuns`` — ``jax.vmap`` over the run axis (or
      ``jax.lax.map(batch_size=run_batch_size)`` when set).
    * ``PmapVmapRuns`` — ``jax.pmap(jax.vmap(...))`` over G × P.

    Args:
        ns_state: Current NSState whose population has shape
            ``(*shape_prefix, n_walkers, ...)``.
        emax: ``(*shape_prefix,)``-shaped energy ceiling (scalar for SingleRun).
        key: PRNG key.
        per_move_fns: Per-move step functions.
        adaptation_policies: Tuple of ResolvedAdaptationPolicy.
        current_step_sizes: ``(*shape_prefix, n_move_types)``.
        adjust_n_samples: Trial walker count per adaptation round.
        adjust_max_rounds: Max bisection rounds.
        jit_adjust_fns: Pre-JIT'd adjust functions, one per move.  These should
            already close over ``trial_batch_size`` (set in :func:`initial_walk`)
            so each per-run call internally chunks its trial vmap.
        batcher: ``BatchDescriptor`` selecting the run-axis dispatch.
        run_batch_size: Optional chunk size for the run-axis vmap.  Only
            consumed when ``batcher`` is ``VmapRuns``; ignored otherwise.
            ``jax.lax.map(..., batch_size=run_batch_size)`` replaces the full
            vmap, bounding peak memory.

    Returns:
        (new_ns_state, new_step_sizes)
    """
    population = ns_state.population
    n_walkers = population.step_sizes.shape[batcher.walker_axis]

    use_chunked_vmap = (
        run_batch_size is not None and isinstance(batcher, VmapRuns)
    )

    for move_idx, policy in enumerate(adaptation_policies):
        key, key_adj = jax.random.split(key)

        def adjust_one_replica(
            pop, ss, e, k,
            _move_idx=move_idx,
            _policy=policy,
        ):
            new_ss, *_ = jit_adjust_fns[_move_idx](
                pop,
                per_move_fns[_move_idx],
                ss,
                e,
                k,
                adjust_n_samples,
                _policy.min_rate,
                _policy.max_rate,
                _policy.adjust_factor,
                _policy.step_size_max,
                adjust_max_rounds,
            )
            return new_ss

        ss_move = current_step_sizes[..., move_idx]

        if not batcher.is_batched:
            new_ss = adjust_one_replica(population, ss_move, emax, key_adj)
        else:
            run_keys = jax.random.split(key_adj, batcher.n_runs).reshape(
                batcher.shape_prefix
            )
            if use_chunked_vmap:
                new_ss = jax.lax.map(
                    lambda x: adjust_one_replica(x[0], x[1], x[2], x[3]),
                    (population, ss_move, emax, run_keys),
                    batch_size=run_batch_size,
                )
            else:
                wrapped = batcher.wrap_for_batch(adjust_one_replica)
                new_ss = wrapped(population, ss_move, emax, run_keys)

        current_step_sizes = current_step_sizes.at[..., move_idx].set(new_ss)

    new_ss_pop = batcher.broadcast_step_sizes(current_step_sizes, n_walkers)
    new_population = population.set(step_sizes=new_ss_pop)
    return ns_state.set(population=new_population), current_step_sizes


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
    run_batch_size: int | None = None,
    per_move_fns: list[Callable] | None = None,
    adaptation_policies: tuple | None = None,
    adjust_n_samples: int = 50,
    adjust_max_rounds: int = 15,
) -> NSState:
    """Run fixed-Emax MCMC to decorrelate walkers from their initialization.

    Emax = max(live_energies) + emax_offset_per_atom * n_atoms, held constant
    across all n_walks * walklength steps.

    Parallelism:
        walker_batch_size=None: vmap over all walkers simultaneously (fastest).
        walker_batch_size=N: chunk via lax.map(batch_size=N). Must divide n_walkers.
        run_batch_size=None: vmap over all runs simultaneously (VmapRuns only).
        run_batch_size=M: chunk via lax.map(batch_size=M). Must divide n_runs.
            Ignored for SingleRun and PmapVmapRuns.

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
            Must divide n_walkers evenly.
        run_batch_size: Chunk size for run vmap (VmapRuns only). None =
            full vmap over runs. Must divide n_runs evenly.
        per_move_fns: Per-move step functions (third return of build_mwg).
            Required for step-size adaptation; None disables it.
        adaptation_policies: Tuple of ResolvedAdaptationPolicy, one per move.
            Required when per_move_fns is not None.
        adjust_n_samples: Number of trial walkers per adaptation round.
        adjust_max_rounds: Max bisection rounds per adaptation call.

    Returns:
        New NSState with live walkers advanced. Same pytree shape as input.
        Dead-point arrays, log_evidence, iteration, n_dead are unchanged.

    Raises:
        ValueError: If walker_batch_size does not evenly divide n_walkers, or
            if run_batch_size does not evenly divide n_runs.
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

    # --- Validate chunking parameters ---
    n_walkers = int(ns_state.population.energy.shape[batcher.walker_axis])
    if walker_batch_size is not None and n_walkers % walker_batch_size != 0:
        raise ValueError(
            f"walker_batch_size={walker_batch_size} does not evenly divide "
            f"n_walkers={n_walkers}. "
            f"Choose a divisor of {n_walkers}."
        )
    if (
        isinstance(batcher, VmapRuns)
        and run_batch_size is not None
        and batcher.n_runs % run_batch_size != 0
    ):
        raise ValueError(
            f"run_batch_size={run_batch_size} does not evenly divide "
            f"n_runs={batcher.n_runs}. "
            f"Choose a divisor of {batcher.n_runs}."
        )

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
        n_moves = len(per_move_fns)
        # Close over walker_batch_size as trial_batch_size: when set, the inner
        # trial vmap inside adjust_step_size becomes a chunked-vmap so peak
        # memory tracks the chunk rather than adjust_n_samples.  The closed-over
        # value is captured at JIT-trace time, so different walker_batch_size
        # values produce distinct compilation traces (correct cache behaviour).
        if walker_batch_size is None:
            jit_adjust_fns = [
                jax.jit(adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10))
                for _ in range(n_moves)
            ]
        else:
            def _make_adjust_fn(_trial_chunk=walker_batch_size):
                def _wrapped(*args):
                    return adjust_step_size(*args, trial_batch_size=_trial_chunk)
                return jax.jit(_wrapped, static_argnums=(1, 5, 6, 7, 8, 9, 10))
            jit_adjust_fns = [_make_adjust_fn() for _ in range(n_moves)]
        current_step_sizes = batcher.extract_step_sizes(population)
    else:
        jit_adjust_fns = None
        current_step_sizes = None

    # --- Build one-walk function via the descriptor ---
    # Per-replica callable; ``wrap_for_batch`` adds the right jit/vmap/pmap
    # composition for the chosen descriptor.  ``run_batch_size`` (chunked
    # vmap along the run axis) is only meaningful for ``VmapRuns`` — for
    # SingleRun there is no run axis, and for PmapVmapRuns the replicas are
    # already split by GPU.  The chunked path is left explicit below.
    def _per_replica(k, run_state, run_emax):
        return _one_walk(
            k, run_state, step_fn, walklength, run_emax, walker_batch_size,
        )

    use_chunked_run_vmap = (
        run_batch_size is not None and isinstance(batcher, VmapRuns)
    )
    if use_chunked_run_vmap:
        def _chunked_walk(key_in, batched_state, batched_emax):
            run_keys = jax.random.split(key_in, batcher.n_runs)
            return jax.lax.map(
                lambda x: _per_replica(x[0], x[1], x[2]),
                (run_keys, batched_state, batched_emax),
                batch_size=run_batch_size,
            )
        jit_one_walk = jax.jit(_chunked_walk)
    else:
        jit_one_walk = batcher.wrap_for_batch(_per_replica)

    # --- Outer walk loop ---
    for walk_i in range(n_walks):
        if use_adaptation and walk_i > 0 and walk_i % adjust_interval == 0:
            key, key_adapt = jax.random.split(key)
            ns_state, current_step_sizes = _apply_adaptation(
                ns_state=ns_state,
                emax=emax,
                key=key_adapt,
                per_move_fns=per_move_fns,
                adaptation_policies=adaptation_policies,
                current_step_sizes=current_step_sizes,
                adjust_n_samples=adjust_n_samples,
                adjust_max_rounds=adjust_max_rounds,
                jit_adjust_fns=jit_adjust_fns,
                batcher=batcher,
                run_batch_size=run_batch_size,
            )

        key, sub = jax.random.split(key)

        if not batcher.is_batched:
            # SingleRun: scalar key flows straight through to ``_per_replica``.
            ns_state, _ = jit_one_walk(sub, ns_state, emax)
        elif use_chunked_run_vmap:
            # Chunked path consumes the scalar key and splits internally.
            ns_state, _ = jit_one_walk(sub, ns_state, emax)
        else:
            # Batched path (vmap or pmap-vmap): split into per-replica keys
            # before invoking the wrapped callable.
            run_keys = jax.random.split(sub, batcher.n_runs).reshape(
                batcher.shape_prefix
            )
            ns_state, _ = jit_one_walk(run_keys, ns_state, emax)

    return ns_state
