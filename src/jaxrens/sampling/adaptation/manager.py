"""AdaptationManager: owns per-move JIT'd adjust_step_size callables.

Constructed once per NS run before the outer loop.  Carries no run state —
just callables and static move descriptors.  Dynamic state (pop, rng_key,
step sizes) flows through ``.apply(...)``.

Supports both single-run (``SingleRun`` batch descriptor) and multi-run
vmap (``VmapRuns``) dispatch via the batch descriptor passed at construction.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence, TypedDict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from jaxrens.sampling.adaptation.stepsize_handler import (
    adjust_step_size,
    adjust_step_size_sharded,
)
from jaxrens.sampling.batch_descriptor import BatchDescriptor, ShardedSingleRun
from jaxrens.sampling.move_kernel import MoveKernel

logger = logging.getLogger(__name__)

# Diagnostic keys returned by adjust_step_size (indices 1..9 of the 10-tuple).
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


class PerMoveDiag(TypedDict):
    """Stacked-over-moves return type of ``AdaptationManager.apply``.

    Shape prefix ``*B`` is empty for ``SingleRun``, ``(n_runs,)`` for
    ``VmapRuns``, and ``(G, P)`` for ``PmapVmapRuns``.
    """

    rate: Float[Array, "*B n_moves"]
    counts: Int[Array, "*B n_moves 4"]
    n_rounds: Int[Array, "*B n_moves"]
    converged: Bool[Array, "*B n_moves"]
    cap_hits: Int[Array, "*B n_moves"]
    floor_hits: Int[Array, "*B n_moves"]
    bracket_detected: Bool[Array, "*B n_moves"]
    trial_n_evaluations: Int[Array, "*B n_moves"]
    trial_n_grad_evaluations: Int[Array, "*B n_moves"]


class AdaptationManager:
    """Owns per-move JIT'd adjust_step_size functions + dispatches their application.

    Constructed once per NS run, before the outer loop.  Does not carry run state —
    just callables and static move descriptors.  Thread-through of dynamic state
    (pop, rng_key, step sizes) goes through ``.apply(...)``.

    Args:
        move_descriptors: Sequence of ``MoveKernel`` objects, one per move type.
        per_move_fns: Sequence of move step functions, one per move.
            When ``None`` (or empty), adaptation is inactive — ``is_active``
            returns ``False`` and ``apply`` raises ``RuntimeError``.
        batcher: ``BatchDescriptor`` instance (``SingleRun`` / ``VmapRuns`` /
            ``PmapVmapRuns``) that determines whether to use plain calls,
            ``jax.vmap``, or ``jax.pmap(jax.vmap(...))`` for the per-move
            adaptation kernels.
        adjust_n_samples: Number of walkers to sample per bisection trial round.
            Static under JIT.
        adjust_factor: Multiplicative scaling factor for step size adjustment.
            Static under JIT.
        adjust_max_rounds: Maximum bisection rounds per ``apply`` call.
            Static under JIT.
        adjust_interval: Trigger ``apply`` every this many iterations.
            ``0`` means adaptation is disabled.
    """

    def __init__(
        self,
        move_descriptors: Sequence[MoveKernel],
        per_move_fns: Sequence[Callable] | None,
        batcher: BatchDescriptor,
        adjust_n_samples: int,
        adjust_factor: float,
        adjust_max_rounds: int,
        adjust_interval: int,
    ) -> None:
        self._move_descriptors = list(move_descriptors) if move_descriptors else []
        self._per_move_fns = list(per_move_fns) if per_move_fns else []
        self._batcher = batcher
        self._adjust_n_samples = adjust_n_samples
        self._adjust_factor = adjust_factor
        self._adjust_max_rounds = adjust_max_rounds
        self._adjust_interval = adjust_interval

        # Build per-move JIT'd callables (once, at construction time).
        # The callable signature depends on the batch descriptor type:
        #   SingleRun: fn(pop, ss_scalar, emax_scalar, key) -> 10-tuple
        #   VmapRuns:  fn(pop_batch, ss_vec, emax_vec, keys_vec) -> 10-tuple of batched arrays
        self._jit_fns: list[Callable] = []
        if self.is_active:
            self._jit_fns = self._build_jit_fns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True iff adaptation is wired (per_move_fns + descriptors + adjust_interval>0)."""
        return (
            bool(self._per_move_fns)
            and bool(self._move_descriptors)
            and self._adjust_interval > 0
        )

    def fires(self, iteration: int) -> bool:
        """True iff *iteration* should trigger an adaptation step.

        Args:
            iteration: Current outer-loop iteration index (Python int).

        Returns:
            ``True`` when ``adjust_interval > 0``, ``iteration > 0``, and
            ``iteration % adjust_interval == 0``.
        """
        return (
            self._adjust_interval > 0
            and iteration > 0
            and iteration % self._adjust_interval == 0
        )

    def apply(
        self,
        pop: Any,
        emax: Float[Array, "*B"],
        rng_key: Key[Array, "*B"],
        current_step_sizes: Float[Array, "*B n_moves"],
    ) -> tuple[Float[Array, "*B n_moves"], PerMoveDiag, Key[Array, "*B"]]:
        """Run one adjust pass across all moves.

        Args:
            pop: Batched MCState population.
                * ``SingleRun``: shape ``(n_walkers, ...)``.
                * ``VmapRuns``:  shape ``(n_runs, n_walkers, ...)``.
            emax: Likelihood constraint.
                * ``SingleRun``: scalar.
                * ``VmapRuns``:  shape ``(n_runs,)``.
            rng_key: PRNG key.
                * ``SingleRun``: scalar key.
                * ``VmapRuns``:  shape ``(n_runs,)`` per-run keys.
            current_step_sizes: Per-move step sizes.
                * ``SingleRun``: shape ``(n_moves,)``.
                * ``VmapRuns``:  shape ``(n_runs, n_moves)``.

        Returns:
            ``(new_step_sizes, per_move_outputs, new_rng_key)`` where:

            * ``new_step_sizes`` has the same shape as ``current_step_sizes``.
            * ``per_move_outputs`` is a dict keyed by diagnostic name
              (``"rate"``, ``"counts"``, ``"n_rounds"``, ``"converged"``,
              ``"cap_hits"``, ``"floor_hits"``, ``"bracket_detected"``,
              ``"trial_n_evaluations"``, ``"trial_n_grad_evaluations"``).
              Each value is:
              - ``(n_moves, ...)`` array for ``SingleRun``
                (``rate``/``converged``/etc. are scalar per move, ``counts``
                is ``(4,)`` per move → stacked to ``(n_moves, 4)``).
              - ``(n_runs, n_moves, ...)`` array for ``VmapRuns``.
            * ``new_rng_key``: Updated PRNG key (carry-forward for caller).
              Same shape as input ``rng_key``.

        Raises:
            RuntimeError: If called when ``is_active`` is ``False``.
        """
        if not self.is_active:
            raise RuntimeError(
                "AdaptationManager.apply() called when is_active=False. "
                "Check that per_move_fns, move_descriptors, and adjust_interval>0 are set."
            )

        # Collect per-move outputs as lists (one item per move).  Shape per
        # entry: ``(*shape_prefix, ...)`` — scalar leaves collapse to ``()``
        # for SingleRun, ``(n_runs,)`` for VmapRuns, ``(G, P)`` for
        # PmapVmapRuns.  No descriptor branching needed: indexing + key
        # splitting use ``...`` and ``descriptor.split_keys`` uniformly.
        per_move_results: list[tuple] = []
        log_per_move = not self._batcher.is_batched  # only meaningful for SingleRun

        for move_idx, desc in enumerate(self._move_descriptors):
            pairs = self._batcher.split_keys(rng_key, 2)
            rng_key = pairs[..., 0]
            key_adjust = pairs[..., 1]
            ss_move = current_step_sizes[..., move_idx]
            result = self._jit_fns[move_idx](pop, ss_move, emax, key_adjust)
            new_ss = result[0]
            current_step_sizes = current_step_sizes.at[..., move_idx].set(new_ss)

            if log_per_move:
                logger.debug(
                    "Adjusted %s: ss=%.4g rate=%.3f rounds=%d converged=%s",
                    desc.name, float(new_ss), float(result[1]),
                    int(result[3]), bool(result[4]),
                )

            per_move_results.append(result[1:])  # skip new_ss (index 0)

        # Stack per-move results.  Each value has shape ``(*shape_prefix, ...)``;
        # stacking ``n_moves`` of them along axis ``len(shape_prefix)`` produces
        # ``(*shape_prefix, n_moves, ...)`` uniformly across descriptor types.
        stack_axis = len(self._batcher.shape_prefix)
        per_move_outputs: dict = {}
        for k, key_name in enumerate(_DIAG_KEYS):
            values = [r[k] for r in per_move_results]
            per_move_outputs[key_name] = jnp.stack(values, axis=stack_axis)

        # Return the advanced rng_key so the caller can carry it forward.
        return current_step_sizes, per_move_outputs, rng_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_jit_fns(self) -> list[Callable]:
        """Build one JIT'd callable per move type.

        Each callable closes over the move's static config (rate bounds,
        ``step_size_max``, etc.) and is wrapped via
        :meth:`BatchDescriptor.wrap_for_batch` so the same factory works for
        SingleRun (``jit``), VmapRuns (``jit(vmap(...))``), and PmapVmapRuns
        (``pmap(vmap(...))``).
        """
        # Select the underlying step-size adjustment function once.
        # For ShardedSingleRun the bisection's accept/eval counters need
        # ``lax.psum`` across the shard axis (since each shard samples
        # locally) — ``adjust_step_size_sharded`` does that internally.
        # All other batchers (SingleRun / VmapRuns / PmapVmapRuns) use
        # the plain function; cross-replica aggregation is not desired
        # because those modes run *independent* replicas.
        if isinstance(self._batcher, ShardedSingleRun):
            adjust_fn = adjust_step_size_sharded
        else:
            adjust_fn = adjust_step_size

        fns: list[Callable] = []
        for move_idx, desc in enumerate(self._move_descriptors):
            move_fn = self._per_move_fns[move_idx]
            n_samp = self._adjust_n_samples
            min_r = desc.min_rate
            max_r = desc.max_rate
            afac = self._adjust_factor
            max_ss = desc.step_size_max
            max_rounds = self._adjust_max_rounds

            def _per_replica(
                pop, ss, emax, key,
                _move_fn=move_fn,
                _n_samp=n_samp,
                _min_r=min_r,
                _max_r=max_r,
                _afac=afac,
                _max_ss=max_ss,
                _max_rounds=max_rounds,
                _adjust_fn=adjust_fn,
            ):
                return _adjust_fn(
                    pop, _move_fn, ss, emax, key,
                    _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
                )

            fns.append(self._batcher.wrap_for_batch(_per_replica))

        return fns
