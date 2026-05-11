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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxrens.sampling.adaptation.manager import AdaptationManager
from jaxrens.sampling.batch_descriptor import BatchDescriptor
from jaxrens.sampling.termination import PriorMassTermination, check_any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers (previously in nested_sampling.py)
# ---------------------------------------------------------------------------


def _bump_cumulative_counters(
    cumulative: dict,
    info: dict,
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


def _inject_cumulative_into_info(info: dict, cumulative: dict) -> None:
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
    info: dict,
    *,
    current_step_sizes,
    per_move_rates: list,
    per_move_counts: list,
    per_move_n_rounds: list,
    per_move_converged: list,
    per_move_cap_hits: list,
    per_move_floor_hits: list,
    per_move_bracket_detected: list,
    per_move_trial_n_evals: list,
    per_move_trial_n_grad_evals: list,
    move_descriptors: list,
) -> None:
    """Pack per-move adjustment diagnostics into *info*.

    Mutates *info* in place.  Also sets the trial-phase evaluation
    counter keys (``"trial_n_evaluations_per_move"`` /
    ``"trial_n_grad_evaluations_per_move"``) that
    ``_bump_cumulative_counters`` consumes when ``include_trial=True``.

    Args:
        info: Info dict to update.
        current_step_sizes: Current per-move step sizes array ``(n_moves,)``.
        per_move_rates: List of float acceptance rates, one per move.
        per_move_counts: List of reject-count arrays, one per move.
        per_move_n_rounds: List of bisection round counts, one per move.
        per_move_converged: List of bool convergence flags, one per move.
        per_move_cap_hits: List of int cap-hit counts, one per move.
        per_move_floor_hits: List of int floor-hit counts, one per move.
        per_move_bracket_detected: List of bool flags, one per move.
        per_move_trial_n_evals: List of int trial eval counts, one per move.
        per_move_trial_n_grad_evals: List of int trial grad-eval counts, one per move.
        move_descriptors: MoveKernel descriptors (provide ``.name`` and
            ``.reject_reasons``).
    """
    trial_n_evals_arr = np.array(
        [int(v) for v in per_move_trial_n_evals], dtype=np.int64
    )
    trial_n_grad_evals_arr = np.array(
        [int(v) for v in per_move_trial_n_grad_evals], dtype=np.int64
    )
    info["step_sizes_per_move"] = current_step_sizes
    info["acceptance_rates_per_move"] = jnp.array(per_move_rates)
    info["reject_counts_per_move"] = jnp.stack(per_move_counts, axis=0)
    info["adjustment_n_rounds"] = jnp.array(per_move_n_rounds, dtype=jnp.int32)
    info["adjustment_converged"] = jnp.array(per_move_converged)
    info["adjustment_cap_hits"] = jnp.array(per_move_cap_hits, dtype=jnp.int32)
    info["adjustment_floor_hits"] = jnp.array(per_move_floor_hits, dtype=jnp.int32)
    info["adjustment_bracket_detected"] = jnp.array(per_move_bracket_detected)
    info["trial_n_evaluations_per_move"] = trial_n_evals_arr
    info["trial_n_grad_evaluations_per_move"] = trial_n_grad_evals_arr
    info["move_names"] = [d.name for d in move_descriptors]
    info["move_reject_reasons"] = tuple(
        frozenset(d.reject_reasons) for d in move_descriptors
    )


def _dispatch_callbacks(
    callbacks: list,
    iteration: int,
    ns_state: Any,
    info: dict,
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


def _pick_next_bucket(
    true_max: int,
    current: int,
    ladder: tuple[int, ...],
    offset: int,
) -> int:
    """Return the smallest ladder entry that accommodates the observed need.

    The target size is ``true_max + offset`` — adding headroom prevents the
    very next MCMC step from tripping the same overflow after a trivial cell
    fluctuation.  Restricting the choice to ``ladder`` bounds the number of
    distinct JIT recompilations to ``len(ladder)`` over a whole run.

    Raises ``RuntimeError`` when the ladder is exhausted or cannot make
    progress; both conditions are user-actionable (extend the list).
    """
    target = int(true_max) + int(offset)
    for b in ladder:
        if b >= target and b > current:
            return b
    raise RuntimeError(
        f"Overflow retry cannot make progress: observed max neighbor count "
        f"{int(true_max)} (+ offset {offset}) requires bucket >= {target}, "
        f"but no entry in max_neighbors_list={list(ladder)} satisfies both "
        f"> current bucket {int(current)} and >= {target}. "
        f"Extend backend.max_neighbors_list to cover this regime."
    )


def _run_loop(
    *,
    batcher: BatchDescriptor,
    adapt_mgr: AdaptationManager,
    ns_state: Any,
    step_fn,
    n_mcmc_steps: int,
    n_extra: int,
    termination_criteria: list,
    callbacks: list,
    n_moves: int,
    move_descriptors: Any,
    rng_key,
    info_interval: int,
    inter_re_mgr: Any = None,
    max_neighbors_list: tuple[int, ...] = (30, 35, 40, 45, 50),
    max_neighbors_offset: int = 5,
) -> tuple[Any, Any, dict]:
    """Unified NS outer loop shared by ``run_ns`` and ``run_ns_parallel``.

    All shape differences between single-run and multi-run execution are
    encapsulated by *batcher*; ``_run_loop`` contains no ``isinstance``
    checks on *batcher*.

    Per-iteration culled walker data lives only in the ``info`` dict —
    callbacks that need it (``EnergyLogger``, ``TrajectoryCallback``) read
    ``info["dead_energy"]`` / ``info["dead_position"]`` / ``info["dead_volume"]``
    directly and persist to disk on the spot.  No host-side history buffer
    is accumulated; the canonical record of dead points lives on disk via
    those callbacks.

    Args:
        batcher: ``BatchDescriptor`` instance controlling JIT-compilation,
            key splitting, and termination reduction.
        adapt_mgr: ``AdaptationManager`` instance. May be inactive
            (``is_active=False``) when no step-size adaptation is configured.
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

    # JIT-compile ns_step once before the loop.
    jit_ns_step = batcher.wrap_step(ns_step, step_fn, n_mcmc_steps, n_extra)

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
    # post-hoc debugging of init/overflow issues.
    for cb in callbacks:
        if hasattr(cb, "on_start"):
            cb.on_start(ns_state)

    i = 0
    while True:
        # ---- Adaptation ----
        adjust_info = None
        if adapt_mgr.fires(i):
            pop = ns_state.population
            emax = batcher.reduce_emax(pop.energy)
            # adapt_mgr.apply does its own key splitting internally;
            # pass rng_key directly and get back the advanced carry.
            current_step_sizes, per_move_outputs, rng_key = adapt_mgr.apply(
                pop, emax, rng_key, current_step_sizes,
            )
            # Re-broadcast updated step sizes across the walker axis.
            n_walkers = pop.step_sizes.shape[batcher.walker_axis]
            new_ss_pop = batcher.broadcast_step_sizes(
                current_step_sizes, n_walkers,
            )
            ns_state = ns_state.set(population=pop.set(step_sizes=new_ss_pop))
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
        if jnp.any(new_ns_state.population.overflow):
            true_max = int(new_ns_state.population.max_neighbor_count.max())
            current = int(ns_state.population.max_neighbors)
            new_max = _pick_next_bucket(
                true_max, current, max_neighbors_list, max_neighbors_offset,
            )
            logger.warning(
                "Overflow at iter %d: observed max_neighbors=%d, "
                "resizing bucket %d -> %d (ladder=%s, offset=%d)",
                i, true_max, current, new_max,
                list(max_neighbors_list), max_neighbors_offset,
            )
            ns_state = ns_state.set(
                population=ns_state.population.set(max_neighbors=new_max),
            )
            continue

        ns_state = new_ns_state

        # ---- Inter-RE phase ----
        # Fires after ns_step, before cumulative counter bump and callbacks.
        # Zero overhead when inter_re_mgr is None or is_active=False.
        if inter_re_mgr is not None and inter_re_mgr.is_active and inter_re_mgr.fires(i):
            inter_re_key, key_re = jax.random.split(inter_re_key)
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

        # ---- Pack adjustment info (only when fires this iter, single-run only) ----
        # Batched per_move_outputs have shape (*shape_prefix, n_moves, ...) which
        # is incompatible with the scalar-per-move pattern of _pack_adjustment_info.
        if adjust_info is not None and not batcher.is_batched:
            _pack_adjustment_info(
                info,
                current_step_sizes=current_step_sizes,
                per_move_rates=list(adjust_info["rate"]),
                per_move_counts=list(adjust_info["counts"]),
                per_move_n_rounds=list(adjust_info["n_rounds"]),
                per_move_converged=list(adjust_info["converged"]),
                per_move_cap_hits=list(adjust_info["cap_hits"]),
                per_move_floor_hits=list(adjust_info["floor_hits"]),
                per_move_bracket_detected=list(adjust_info["bracket_detected"]),
                per_move_trial_n_evals=list(adjust_info["trial_n_evaluations"]),
                per_move_trial_n_grad_evals=list(adjust_info["trial_n_grad_evaluations"]),
                move_descriptors=move_descriptors or [],
            )
            # Also accumulate trial-phase evals into cumulative counters
            _bump_cumulative_counters(cumulative, info, include_trial=True)

        # ---- Inject cumulative snapshot ----
        _inject_cumulative_into_info(info, cumulative)

        # ---- Attach the batcher so callbacks can introspect ----
        info["_batcher"] = batcher

        # ---- Callback dispatch ----
        _dispatch_callbacks(callbacks, i, ns_state, info)

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


