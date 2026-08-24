"""Neighbor-bucket ladder management shared by the NS loop and burn-in.

Both ``_run_loop`` (NS proper) and ``initial_walk`` (burn-in) advance MCMC
state in steps that can overflow the per-walker neighbor budget configured
via ``max_neighbors`` on the population.  When that happens the outer
Python loop has to:

1. Roll back to the pre-step ``ns_state``,
2. Grow the bucket to the next ladder entry,
3. Retry the same step (the JAX cache hands us the kernel compiled for the
   new bucket, or compiles it once on first hit).

Optionally — when ``shrink_dwell > 0`` — the loop also tracks how long the
observed neighbor peak has stayed below the next-smaller ladder entry
(modulo the offset headroom) and steps the bucket back down once the dwell
is satisfied.  Temporal hysteresis via ``shrink_dwell`` and the ``offset``
slack already keep the bucket from thrashing at a boundary; no extra
shrink-side margin is needed on top.

Both pickers (``_pick_next_bucket`` / ``_pick_prev_bucket``) and the
``BucketManager`` orchestrator live here; the two outer loops own only the
Python control flow that decides *when* to call them.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

logger = logging.getLogger(__name__)


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


def _pick_prev_bucket(
    true_max: int,
    current: int,
    ladder: tuple[int, ...],
    offset: int,
) -> int | None:
    """Return the largest ladder entry strictly less than ``current`` that
    still safely accommodates ``true_max + offset``, or ``None`` when no
    smaller entry qualifies.

    ``offset`` plays the dual role of post-grow headroom and post-shrink
    slack: a shrink only commits when the next-smaller bucket still leaves
    ``offset`` slots above the observed peak.  Temporal hysteresis comes
    from ``shrink_dwell`` in the orchestrator.

    Going back to a previously-visited bucket reuses the JAX compilation
    cache, so the only cost of a wrong decision is the next overflow re-grow
    — not a fresh compile.
    """
    target = int(true_max) + int(offset)
    best: int | None = None
    for b in ladder:
        if b >= current:
            break  # ladder is strictly ascending; nothing smaller follows
        if b >= target:
            best = b  # keep scanning for a closer fit just below current
    return best


class BucketManager:
    """Outer-loop helper that owns ``low_count`` and the picker calls.

    A single instance is used by ``_run_loop`` (one per NS run) and by
    ``initial_walk`` (one per burn-in invocation).  The class is *not*
    JIT-traced — it is pure-Python control-flow glue that decides whether
    to grow / shrink / keep the ``max_neighbors`` static field on the
    population pytree.

    ``low_count`` is the only mutable state: the count of consecutive
    successful iterations satisfying the hysteresis gap condition.  It is
    reset to zero on (a) a gap violation, (b) any successful shrink, and
    (c) any overflow growth.
    """

    def __init__(
        self,
        ladder: tuple[int, ...] | list[int],
        offset: int,
        shrink_dwell: int = 0,
    ) -> None:
        self.ladder = tuple(int(x) for x in ladder)
        if not self.ladder:
            raise ValueError(
                "BucketManager ladder is empty. It holds the max_neighbors "
                "capacities the run escalates through on overflow, so it "
                "needs at least one entry (e.g. (64, 96, 128))."
            )
        self.offset = int(offset)
        self.shrink_dwell = int(shrink_dwell)
        if self.shrink_dwell < 0:
            raise ValueError(
                f"shrink_dwell must be >= 0, got {self.shrink_dwell}."
            )
        self.low_count = 0

    def grow_if_overflow(
        self,
        ns_state: Any,
        new_ns_state: Any,
        *,
        label: str,
        iteration: int,
    ) -> tuple[Any, bool]:
        """Detect and handle a neighbor-bucket overflow.

        Args:
            ns_state: Pre-step state (the rollback target if overflow fired).
            new_ns_state: Post-step state, which may carry
                ``population.overflow=True`` from the JIT kernel.
            label: Short tag used in the warning log, e.g. ``"iter"`` or
                ``"burn-in walk"``.  Helps disambiguate the two call sites.
            iteration: Counter value (NS iteration or burn-in walk index).

        Returns:
            ``(state_to_use_next, retry_needed)``.  When ``retry_needed`` is
            True, the caller must NOT advance its outer counter and should
            re-run the JIT kernel against ``state_to_use_next`` (which has a
            larger bucket).  When False, ``state_to_use_next`` is just
            ``new_ns_state`` unchanged.
        """
        if not bool(jnp.any(new_ns_state.population.overflow)):
            return new_ns_state, False

        true_max = int(new_ns_state.population.max_neighbor_count.max())
        current = int(ns_state.population.max_neighbors)
        new_max = _pick_next_bucket(
            true_max, current, self.ladder, self.offset
        )
        logger.warning(
            "Overflow at %s %d: observed max_neighbors=%d, "
            "resizing bucket %d -> %d (ladder=%s, offset=%d)",
            label,
            iteration,
            true_max,
            current,
            new_max,
            list(self.ladder),
            self.offset,
        )
        # Growing invalidates any pending shrink streak.
        self.low_count = 0
        retried_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=new_max),
        )
        return retried_state, True

    def maybe_shrink(
        self,
        ns_state: Any,
        *,
        iteration: int,
    ) -> Any:
        """Step the bucket down one entry when the dwell window is satisfied.

        Reads ``ns_state.population.max_neighbor_count`` (the per-walker peak
        observed during the last successful step) and compares the next-
        smaller ladder entry against ``observed + offset``.  Increments
        ``low_count`` on a qualifying iteration, resets to zero on a
        violation, and shrinks when the counter reaches ``shrink_dwell``.

        ``shrink_dwell == 0`` (default) bypasses the entire check so the
        method becomes a cheap no-op for users who never opted in.

        Returns the (possibly updated) state.  The caller assigns the result
        back to its working ``ns_state``.
        """
        if self.shrink_dwell <= 0:
            return ns_state

        obs_max = int(ns_state.population.max_neighbor_count.max())
        current = int(ns_state.population.max_neighbors)
        smaller = _pick_prev_bucket(
            obs_max,
            current,
            self.ladder,
            self.offset,
        )
        if smaller is None:
            self.low_count = 0
            return ns_state

        self.low_count += 1
        if self.low_count < self.shrink_dwell:
            return ns_state

        logger.info(
            "Shrinking bucket at iter %d: observed max_neighbors=%d, "
            "resizing bucket %d -> %d (ladder=%s, offset=%d, dwell=%d)",
            iteration,
            obs_max,
            current,
            smaller,
            list(self.ladder),
            self.offset,
            self.shrink_dwell,
        )
        self.low_count = 0
        return ns_state.set(
            population=ns_state.population.set(max_neighbors=smaller),
        )
