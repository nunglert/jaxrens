"""Unified outer NS loop: _run_loop and its private helpers.

This module is commit 4 of the nested_sampling.py modularisation plan.
Both ``run_ns`` and ``run_ns_parallel`` are thin wrappers that call
``_run_loop`` after initialising the descriptor- and manager-level objects.

The helpers ``_bump_cumulative_counters``, ``_inject_cumulative_into_info``,
``_pack_adjustment_info``, and ``_dispatch_callbacks`` were previously private
to ``nested_sampling.py``; they now live here.  ``nested_sampling.py`` imports
them from here.

Design invariants
-----------------
- ``_run_loop`` is pure-Python control flow; every JAX call inside it is
  either already JIT-compiled (``jit_ns_step`` from ``batcher.wrap_step``)
  or is a Python-level numpy / host operation.
- The descriptor carries all shape differences (single vs. vmap); there is NO
  ``isinstance`` sniffing inside ``_run_loop``.
- Overflow retry preserves the invariant that the iteration counter only
  advances after a successful step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from jaxrens.sampling.adaptation.manager import build_adapt_step  # noqa: F401
from jaxrens.sampling.batch_descriptor import BatchDescriptor
from jaxrens.sampling.inter_re_manager import InterREManager
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.termination import PriorMassTermination, check_any
from jaxrens.state.ns import NSState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers (previously in nested_sampling.py)
# ---------------------------------------------------------------------------


def _bump_cumulative_counters(
    cumulative: dict[str, np.ndarray],
    info: dict[str, Any],
    *,
    include_trial: bool = False,
) -> None:
    """Accumulate per-move evaluation counters from *info* into *cumulative*.

    Mutates *cumulative* in place.  Works for both the single-run case
    (shapes ``(n_moves,)``) and the parallel case (shapes
    ``(n_runs, n_moves)``) because numpy broadcasting handles both.

    Args:
        cumulative: Mutable dict with keys
            ``"n_evaluations"`` and ``"n_grad_evaluations"`` as
            numpy int64 arrays.
        info: Info dict emitted by ``ns_step`` (or the batched variant).
            Must contain ``"n_evaluations_per_move"`` and
            ``"n_grad_evaluations_per_move"``.
        include_trial: When *True*, also accumulate
            ``"trial_n_evaluations_per_move"`` and
            ``"trial_n_grad_evaluations_per_move"`` from *info* (only
            present on adjust-fired iterations).
    """
    cumulative["n_evaluations"] += np.asarray(
        info["n_evaluations_per_move"], dtype=np.int64
    )
    cumulative["n_grad_evaluations"] += np.asarray(
        info["n_grad_evaluations_per_move"], dtype=np.int64
    )
    if include_trial:
        cumulative["n_evaluations"] += np.asarray(
            info["trial_n_evaluations_per_move"], dtype=np.int64
        )
        cumulative["n_grad_evaluations"] += np.asarray(
            info["trial_n_grad_evaluations_per_move"], dtype=np.int64
        )


def _inject_cumulative_into_info(
    info: dict[str, Any], cumulative: dict[str, np.ndarray],
) -> None:
    """Write cumulative counter snapshots into *info* for downstream callbacks.

    Sets ``info["cumulative_n_evaluations_per_move"]`` and
    ``info["cumulative_n_grad_evaluations_per_move"]`` to copies of the
    current cumulative arrays.  Mutates *info* in place.

    Args:
        info: Info dict to update.
        cumulative: Dict with keys ``"n_evaluations"`` and
            ``"n_grad_evaluations"`` (numpy int64 arrays).
    """
    info["cumulative_n_evaluations_per_move"] = cumulative["n_evaluations"].copy()
    info["cumulative_n_grad_evaluations_per_move"] = (
        cumulative["n_grad_evaluations"].copy()
    )


def _pack_adjustment_info(
    info: dict[str, Any],
    *,
    current_step_sizes: Float[Array, "*B n_moves"],
    adjust_info: dict[str, Any],
    move_descriptors: list[MoveKernel],
) -> None:
    """Pack per-move adjustment diagnostics into *info*.

    Mutates *info* in place.  Also sets the trial-phase evaluation
    counter keys (``"trial_n_evaluations_per_move"`` /
    ``"trial_n_grad_evaluations_per_move"``) that
    ``_bump_cumulative_counters`` consumes when ``include_trial=True``.

    All array values are passed through with shape ``(*shape_prefix,
    n_moves, ...)`` exactly as ``adapt_step`` returned
    them — no list-of-scalars round-trip — so this works uniformly for
    SingleRun (``(n_moves,)`` / ``(n_moves, 4)``), VmapRuns
    (``(n_runs, n_moves, ...)``), and PmapVmapRuns (``(G, P,
    n_moves, ...)``).  Downstream ``AdaptationCallback`` calls
    ``batcher.flatten`` to coerce these to ``(n_runs_total, n_moves,
    ...)`` for the HDF5 writer.

    Args:
        info: Info dict to update.
        current_step_sizes: Per-move step sizes after the adjust call,
            shape ``(*shape_prefix, n_moves)``.
        adjust_info: Dict returned by ``adapt_step`` —
            keys ``rate``, ``counts``, ``n_rounds``, ``converged``,
            ``cap_hits``, ``floor_hits``, ``bracket_detected``,
            ``trial_n_evaluations``, ``trial_n_grad_evaluations``.
        move_descriptors: MoveKernel descriptors (provide ``.name`` and
            ``.reject_reasons``).
    """
    info["step_sizes_per_move"] = current_step_sizes
    info["acceptance_rates_per_move"] = adjust_info["rate"]
    info["reject_counts_per_move"] = adjust_info["counts"]
    info["adjustment_n_rounds"] = adjust_info["n_rounds"]
    info["adjustment_converged"] = adjust_info["converged"]
    info["adjustment_cap_hits"] = adjust_info["cap_hits"]
    info["adjustment_floor_hits"] = adjust_info["floor_hits"]
    info["adjustment_bracket_detected"] = adjust_info["bracket_detected"]
    info["trial_n_evaluations_per_move"] = adjust_info["trial_n_evaluations"]
    info["trial_n_grad_evaluations_per_move"] = adjust_info[
        "trial_n_grad_evaluations"
    ]
    info["move_names"] = [d.name for d in move_descriptors]
    info["move_reject_reasons"] = tuple(
        frozenset(d.reject_reasons) for d in move_descriptors
    )


_PRESERVE_INFO_KEYS = frozenset({"move_names"})
_DROP_INFO_KEYS_AFTER_SCALARIZE = frozenset({"_batcher"})


def _gather_sharded_ns_state(ns_state: NSState) -> NSState:
    """Return an NSState with population leaves gathered to ``(K, ...)``.

    For ShardedSingleRun, ``ns_state.population`` carries leaves shaped
    ``(G, K // G, ...)`` distributed across G devices.  Callbacks that
    were written for SingleRun expect ``(K, ...)``-shaped leaves.  This
    helper gathers + reshapes; non-population scalar leaves carry a
    redundant (G,) axis (post-`lax.psum`) — collapse via ``[0]``.
    """
    pop = ns_state.population

    def _gather_leaf(leaf):
        arr = jnp.asarray(leaf)
        if arr.ndim == 0:
            return arr
        # Population leaves: (G, K_per_gpu, ...) → (K, ...).
        # Per-shard scalars stored on pop (e.g. step_size, n_atoms): (G,) →
        # take [0] (every shard identical).
        if arr.ndim >= 2:
            return arr.reshape((arr.shape[0] * arr.shape[1],) + arr.shape[2:])
        return arr[0]

    new_pop = jax.tree.map(_gather_leaf, pop)

    # Top-level NSState scalar leaves (log_evidence, iteration, rng_key)
    # also carry a (G,) redundant axis — collapse via [0].
    def _collapse(leaf):
        arr = jnp.asarray(leaf)
        return arr[0] if arr.ndim > 0 else arr

    new_ns_state = ns_state.set(
        population=new_pop,
        log_evidence=_collapse(ns_state.log_evidence),
        iteration=_collapse(ns_state.iteration),
        emax=_collapse(ns_state.emax),
        rng_key=_collapse(ns_state.rng_key),
    )
    return new_ns_state


def _scalarize_sharded_info(info: dict[str, Any]) -> dict[str, Any]:
    """Collapse the redundant leading-G axis from every leaf in ``info``.

    ShardedSingleRun's ``ns_step_sharded`` and ``adjust_step_size_sharded``
    write per-shard scalars / per-shard arrays where every shard's entry
    is identical (post-`lax.psum` invariant).  Take ``[0]`` along that
    axis so SingleRun-shaped callbacks (ProgressCallback,
    TrajectoryCallback, etc.) see scalar / unsharded shapes without
    needing to know about the underlying batcher.

    Preserves a small allowlist of keys that are intentionally
    already-collapsed (``move_names`` is a Python list,
    ``_batcher`` is the descriptor itself).
    """
    out: dict[str, Any] = {}
    for k, v in info.items():
        if k in _DROP_INFO_KEYS_AFTER_SCALARIZE:
            # Hide the sharded batcher reference: the scalarized dict
            # now looks like SingleRun, so callbacks that branch on
            # ``info["_batcher"]`` (e.g. AdaptationCallback's
            # ``batcher.flatten``) should fall back to their
            # SingleRun-shaped ndim heuristics.
            continue
        if k in _PRESERVE_INFO_KEYS or v is None:
            out[k] = v
            continue
        try:
            arr = jnp.asarray(v)
        except (TypeError, ValueError):
            out[k] = v
            continue
        out[k] = arr[0] if arr.ndim > 0 else arr
    return out


def _dispatch_callbacks(
    callbacks: list,
    iteration: int,
    ns_state: NSState,
    info: dict[str, Any],
) -> None:
    """Call ``on_iteration`` on every callback that exposes it.

    Args:
        callbacks: List of callback objects.
        iteration: Current iteration index (Python int).
        ns_state: Current ``NSState``.
        info: Info dict for this iteration.
    """
    for cb in callbacks:
        if hasattr(cb, "on_iteration"):
            cb.on_iteration(iteration, ns_state, info)


# ---------------------------------------------------------------------------
# Unified outer loop
# ---------------------------------------------------------------------------


# Backward-compatible re-exports.  The pickers (and the BucketManager) now
# live in ``sampling.bucket_manager``; importing them from this module is
# preserved so older tests and downstream code keep working.
from jaxrens.sampling.bucket_manager import (  # noqa: E402
    BucketManager,
    _pick_next_bucket,
    _pick_prev_bucket,
)


def _run_loop(
    *,
    batcher: BatchDescriptor,
    adapt_step: Callable | None,
    adjust_interval: int,
    ns_state: NSState,
    step_fn: Callable,
    n_mcmc_steps: int,
    n_extra: int,
    termination_criteria: list,
    callbacks: list,
    n_moves: int,
    move_descriptors: list[MoveKernel] | None,
    rng_key: Key[Array, "*B"],
    info_interval: int,
    inter_re_mgr: InterREManager | None = None,
    max_neighbors_list: tuple[int, ...] = (30, 35, 40, 45, 50),
    max_neighbors_offset: int = 5,
    max_neighbors_shrink_dwell: int = 0,
    ns_step_fn: Callable | None = None,
) -> tuple[NSState, Key[Array, "*B"], dict[str, np.ndarray]]:
    """Unified NS outer loop shared by ``run_ns`` and ``run_ns_parallel``.

    All shape differences between single-run and multi-run execution are
    encapsulated by *batcher*; ``_run_loop`` contains no ``isinstance``
    checks on *batcher*.

    Per-iteration culled walker data lives only in the ``info`` dict —
    callbacks that need it (``EnergyLogger``, ``TrajectoryCallback``) read
    ``info["dead_walker"]`` (a ``WalkerState`` pytree with positions /
    types / energy / cell) and persist to disk on the spot.  No host-side
    history buffer is accumulated; the canonical record of dead points
    lives on disk via those callbacks.

    Args:
        batcher: ``BatchDescriptor`` instance controlling JIT-compilation,
            key splitting, and termination reduction.
        adapt_step: Closure built by :func:`build_adapt_step`, or ``None``
            when no step-size adaptation is configured.  Called as
            ``(ns_state, emax, key) -> (new_ns_state, diag, new_key)``.
        adjust_interval: Iterations between adapt fires.  ``<= 0`` (or
            ``adapt_step is None``) disables adaptation.
        ns_state: Initial ``NSState`` (single or batched).
        step_fn: MCMC step function from ``build_mwg()``.
        n_mcmc_steps: Number of MCMC steps per walker (static under JIT).
        n_extra: Number of extra walkers to walk per iteration (static).
        termination_criteria: List of ``TerminationCriterion`` objects.  The
            loop runs until any one of these fires; there is no static
            iteration cap.  Pass an ``IterationTermination(N)`` to bound the
            run by iteration count.
        callbacks: List of callback objects with optional ``on_iteration``
            methods.
        n_moves: Number of move types (determines counter array shapes).
        move_descriptors: List of ``MoveKernel`` objects; may be ``None`` or
            empty when no adaptation is configured.
        rng_key: PRNG key for adaptation key splitting.
            * ``SingleRun``: scalar key.
            * ``VmapRuns``: shape ``(n_runs,)`` per-run keys.
        info_interval: Unused by this function (kept for forward-compat).
            Logging is handled entirely by callbacks (e.g. ProgressCallback).
        inter_re_mgr: Optional :class:`InterREManager` instance.  When
            provided and ``inter_re_mgr.is_active`` is True, a swap pass
            fires after each ``ns_step`` call on iterations where
            ``inter_re_mgr.fires(i)`` is True.  ``None`` → zero overhead.
        max_neighbors_shrink_dwell: Number of consecutive iterations the
            observed peak must sit at or below ``next_smaller_entry -
            offset`` before the bucket is downsized one step.  ``0``
            (default) disables shrinking entirely so existing runs are
            byte-identical.  JAX's compilation cache reuses earlier compiles
            when the bucket revisits a known size, so the only cost paid by
            a wrong downsize is the next overflow re-grow.

    Returns:
        ``(ns_state, rng_key, cumulative)`` where:

        * ``ns_state``: Updated ``NSState`` after the loop completes.
        * ``rng_key``: Advanced PRNG key (carry for callers that restart).
        * ``cumulative``: Dict with keys ``"n_evaluations"`` and
          ``"n_grad_evaluations"`` as numpy int64 arrays.  Shape
          ``(n_moves,)`` for ``SingleRun``, ``(n_runs, n_moves)`` for
          ``VmapRuns``.
    """
    from jaxrens.sampling.nested_sampling import ns_step  # avoid circular at module level

    # JIT-compile ns_step once before the loop.  Callers may override
    # the underlying step function (e.g. ``ns_step_sharded`` for the
    # ShardedSingleRun mode) via ``ns_step_fn``; default preserves the
    # SingleRun / VmapRuns / PmapVmapRuns behaviour.
    step_to_wrap = ns_step_fn if ns_step_fn is not None else ns_step
    jit_ns_step = batcher.wrap_step(step_to_wrap, step_fn, n_mcmc_steps, n_extra)

    # Per-replica step sizes: pop.step_sizes is (*shape_prefix, K, n_moves).
    # The descriptor drops the walker axis to give (*shape_prefix, n_moves).
    pop = ns_state.population
    current_step_sizes = batcher.extract_step_sizes(pop)

    # Cumulative evaluation counters: shape (*shape_prefix, n_moves).
    # Empty prefix collapses to (n_moves,) for SingleRun.
    cumulative: dict = {
        "n_evaluations": np.zeros(
            batcher.shape_prefix + (n_moves,), dtype=np.int64,
        ),
        "n_grad_evaluations": np.zeros(
            batcher.shape_prefix + (n_moves,), dtype=np.int64,
        ),
    }

    # Dedicated scalar PRNG key for inter-RE swaps.
    # The adaptation ``rng_key`` may be (n_runs,) shaped for VmapRuns; we need
    # a single scalar key for replica_exchange_step.  Derive from rng_key once
    # before the loop.  For SingleRun rng_key is already scalar; for batched
    # descriptors we take the first per-run key.
    inter_re_key = None
    if inter_re_mgr is not None and inter_re_mgr.is_active:
        inter_re_key = jax.random.split(batcher.scalar_key(rng_key), 1)[0]

    # ---- on_start callbacks (fired once, post-init, pre-loop) ----
    # CheckpointCallback uses this hook to snapshot the fully-initialised
    # NSState before iteration 0, giving a reproducible entry point for
    # post-hoc debugging of init/overflow issues.  AdaptationCallback uses
    # it to write the mandatory iter-0 step-size baseline row.
    #
    # ``start_info`` carries ``_batcher`` and the initial
    # ``step_sizes_per_move`` so callbacks can reuse the same shape
    # coercion they apply on ``on_iteration`` info dicts.
    start_info = {
        "_batcher": batcher,
        "step_sizes_per_move": current_step_sizes,
    }
    for cb in callbacks:
        if hasattr(cb, "on_start"):
            cb.on_start(ns_state, start_info)

    # Bucket-ladder hysteresis manager — owns _pick_next/_prev_bucket calls
    # and the ``low_count`` shrink counter.  Shared with burn-in's
    # ``initial_walk`` so both sites use the same growth + shrink logic.
    bucket_mgr = BucketManager(
        ladder=max_neighbors_list,
        offset=max_neighbors_offset,
        shrink_dwell=max_neighbors_shrink_dwell,
    )

    # ``i`` is the segment-local loop counter (0-based each invocation) and
    # drives termination + adapt phase — keeping it segment-local preserves
    # "run N more iterations per restart" semantics.  ``record_iteration``
    # is the *global* iteration used for every on-disk artifact (dead-point
    # labels, output cadence): on a fresh run ``restart_iteration == 0`` so
    # it equals ``i``; on restart it continues monotonically from the
    # checkpoint's iteration, so the ``.energies`` / ``.traj`` / HDF5 streams
    # stay continuous and truncate-on-restart has a well-defined cut point.
    restart_iteration = int(jnp.asarray(ns_state.iteration).reshape(-1)[0])

    i = 0
    while True:
        record_iteration = i + restart_iteration
        # ---- Adaptation ----
        adjust_info = None
        if adapt_step is not None and i % adjust_interval == 0:
            # ``ns_state.emax`` holds the contour the previous ``ns_step``
            # just culled at (algorithm state, not a re-derived
            # ``max(pop.energy)``).  Adapt trial walkers experience the
            # same constraint the population was sampled under — minor
            # off-by-one vs. the upcoming step's L_{i+1}, but principled.
            ns_state, per_move_outputs, rng_key = adapt_step(
                ns_state, ns_state.emax, rng_key,
            )
            # Re-extract for the downstream callbacks / info dict.
            current_step_sizes = batcher.extract_step_sizes(
                ns_state.population,
            )
            adjust_info = per_move_outputs

        # ---- NS step ----
        # Batched callables (vmap / pmap-vmap) close over step_fn etc.;
        # the SingleRun callable is a plain jit'd fn expecting explicit args.
        if batcher.is_batched:
            new_ns_state, info = jit_ns_step(ns_state)
        else:
            new_ns_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps, n_extra)

        # ---- Overflow retry ----
        # For PmapVmapRuns, any overflow across any (G, P) shard triggers a retry.
        ns_state, retry = bucket_mgr.grow_if_overflow(
            ns_state, new_ns_state, label="iter", iteration=record_iteration,
        )
        if retry:
            continue

        # ---- Bucket shrink (hysteresis-gated) ----
        # No-op when shrink_dwell == 0; otherwise steps the bucket down one
        # ladder entry once the observed peak has stayed below the next-
        # smaller bucket (with margin) for ``shrink_dwell`` iterations.
        ns_state = bucket_mgr.maybe_shrink(ns_state, iteration=record_iteration)

        # ---- Inter-RE phase ----
        # Fires after ns_step, before cumulative counter bump and callbacks.
        # Zero overhead when inter_re_mgr is None or is_active=False.
        if inter_re_mgr is not None and inter_re_mgr.is_active and inter_re_mgr.fires(i):
            inter_re_key, key_re = jax.random.split(inter_re_key)
            # RE reads the contour off ``ns_state.emax`` (set by the
            # ns_step that just ran).  No emax recomputation here, no
            # plumbing — the contour lives on state precisely so every
            # consumer reads the same algorithm-defined value.
            ns_state, re_stats, _ = inter_re_mgr.apply(ns_state, key_re)
            info["inter_re_stats"] = re_stats
            # Roll RE energy evals into the cumulative counters so that the
            # monitor's nE= / nG= tally stays accurate.
            # PressureRENSSwap has zero eval counts; XRENSSwap will have non-zero.
            # We add a scalar to all per-move counters (uniform distribution
            # assumption — simplest approach that keeps the tally consistent).
            re_n_evals = re_stats.get("n_energy_evals", 0)
            re_n_grad = re_stats.get("n_grad_evals", 0)
            if re_n_evals > 0:
                cumulative["n_evaluations"] += np.int64(re_n_evals)
            if re_n_grad > 0:
                cumulative["n_grad_evaluations"] += np.int64(re_n_grad)

        # ---- Cumulative counters (chain phase) ----
        _bump_cumulative_counters(cumulative, info)

        # ---- Pack adjustment info (only when fires this iter) ----
        # Works uniformly for SingleRun, VmapRuns, and PmapVmapRuns:
        # adjust_info values are already shaped ``(*shape_prefix, n_moves, ...)``
        # straight from ``adapt_step``; the packer is shape-agnostic
        # and ``AdaptationCallback.on_iteration`` will call ``batcher.flatten``
        # to land them in ``(n_runs_total, n_moves, ...)`` before HDF5.
        if adjust_info is not None:
            _pack_adjustment_info(
                info,
                current_step_sizes=current_step_sizes,
                adjust_info=adjust_info,
                move_descriptors=move_descriptors or [],
            )
            # Also accumulate trial-phase evals into cumulative counters.
            # cumulative has shape (*shape_prefix, n_moves); the trial arrays
            # already match, so the in-place add is shape-correct for all
            # batchers (numpy broadcasting handles the SingleRun case where
            # shape_prefix is empty).
            _bump_cumulative_counters(cumulative, info, include_trial=True)

        # ---- Inject cumulative snapshot ----
        _inject_cumulative_into_info(info, cumulative)

        # ---- Attach the batcher so callbacks can introspect ----
        info["_batcher"] = batcher

        # ---- Callback dispatch ----
        # ShardedSingleRun: ``info`` carries a redundant leading (G,)
        # axis on every leaf (each shard reports the same global value
        # post-`lax.psum`); the population leaves carry a non-redundant
        # leading G axis (each shard owns ``K // G`` walkers).
        # Collapse info via ``[0]`` and gather+reshape population leaves
        # to ``(K, ...)`` so SingleRun-shaped callbacks see uniform
        # shapes.
        from jaxrens.sampling.batch_descriptor import ShardedSingleRun
        if isinstance(batcher, ShardedSingleRun):
            info = _scalarize_sharded_info(info)
            ns_state_for_cb = _gather_sharded_ns_state(ns_state)
        else:
            ns_state_for_cb = ns_state

        _dispatch_callbacks(callbacks, record_iteration, ns_state_for_cb, info)

        # ---- Termination ----
        log_z_scalar, hmax_scalar = batcher.reduce_for_termination(
            ns_state.log_evidence, info.get("hmax", jnp.inf),
        )
        for criterion in termination_criteria:
            if isinstance(criterion, PriorMassTermination):
                criterion.update_evidence(log_z_scalar)

        should_stop, reason = check_any(termination_criteria, i, hmax_scalar)
        if should_stop:
            logger.info("NS terminated at iter %d: %s", i, reason)
            break

        i += 1

    return ns_state, rng_key, cumulative


