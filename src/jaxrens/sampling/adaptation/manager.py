"""AdaptationManager: owns per-move JIT'd adjust_step_size callables.

Constructed once per NS run before the outer loop.  Carries no run state —
just callables and static move descriptors.  Dynamic state (pop, rng_key,
step sizes) flows through ``.apply(...)``.

Supports both single-run (``SingleRun`` batch descriptor) and multi-run
vmap (``VmapRuns``) dispatch via the batch descriptor passed at construction.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from jaxrens.sampling.adaptation.stepsize_handler import adjust_step_size
from jaxrens.sampling.batch_descriptor import BatchDescriptor, SingleRun, VmapRuns

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
        batch_descriptor: ``BatchDescriptor`` instance (``SingleRun`` or
            ``VmapRuns``) that determines whether to use plain calls or
            ``jax.vmap`` over the run dimension.
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
        move_descriptors: Sequence,
        per_move_fns: Sequence[Callable] | None,
        batch_descriptor: BatchDescriptor,
        adjust_n_samples: int,
        adjust_factor: float,
        adjust_max_rounds: int,
        adjust_interval: int,
    ) -> None:
        self._move_descriptors = list(move_descriptors) if move_descriptors else []
        self._per_move_fns = list(per_move_fns) if per_move_fns else []
        self._batch_descriptor = batch_descriptor
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
        emax: jax.Array,
        rng_key: jax.Array,
        current_step_sizes: jax.Array,
    ) -> tuple[jax.Array, dict, jax.Array]:
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

        is_vmap = isinstance(self._batch_descriptor, VmapRuns)

        # Collect per-move outputs as lists (one item per move).
        # For SingleRun: each item is a scalar/array.
        # For VmapRuns:  each item is a (n_runs, ...) array.
        per_move_results: list[tuple] = []

        for move_idx, desc in enumerate(self._move_descriptors):
            if is_vmap:
                # current_step_sizes: (n_runs, n_moves); rng_key: (n_runs,)
                # emax: (n_runs,)
                rng_key, key_adjust = self._split_vmap_keys(rng_key)
                ss_move = current_step_sizes[:, move_idx]  # (n_runs,)
                result = self._jit_fns[move_idx](pop, ss_move, emax, key_adjust)
                new_ss = result[0]  # (n_runs,)
                current_step_sizes = current_step_sizes.at[:, move_idx].set(new_ss)
            else:
                # current_step_sizes: (n_moves,); rng_key: scalar key
                rng_key, key_adjust = jax.random.split(rng_key)
                ss_move = current_step_sizes[move_idx]  # scalar
                result = self._jit_fns[move_idx](pop, ss_move, emax, key_adjust)
                new_ss = result[0]  # scalar
                current_step_sizes = current_step_sizes.at[move_idx].set(new_ss)
                logger.debug(
                    "Adjusted %s: ss=%.4g rate=%.3f rounds=%d converged=%s",
                    desc.name, float(new_ss), float(result[1]),
                    int(result[3]), bool(result[4]),
                )

            per_move_results.append(result[1:])  # skip new_ss (index 0)

        # Stack per-move results into arrays keyed by diagnostic name.
        # For SingleRun, each result[k] is a scalar/array; stack along axis 0 → (n_moves, ...).
        # For VmapRuns, each result[k] is (n_runs, ...) → stack along axis 1 (new axis after n_runs).
        # Convenient: jnp.stack(..., axis=0) then transpose to (n_runs, n_moves, ...) if needed.
        per_move_outputs: dict = {}
        for k, key_name in enumerate(_DIAG_KEYS):
            values = [r[k] for r in per_move_results]
            if is_vmap:
                # Each value is (n_runs,) or (n_runs, 4) etc.
                # Stack along a new axis 1 to get (n_runs, n_moves) or (n_runs, n_moves, 4).
                stacked = jnp.stack(values, axis=1)  # (n_runs, n_moves, ...)
            else:
                # Each value is scalar or (4,) etc.
                # Stack along axis 0 to get (n_moves,) or (n_moves, 4).
                stacked = jnp.stack(values, axis=0)  # (n_moves, ...)
            per_move_outputs[key_name] = stacked

        # Return the advanced rng_key so the caller can carry it forward.
        # For SingleRun: rng_key is the residual after n_moves splits.
        # For VmapRuns: rng_key is the (n_runs,) carry after n_moves per-run splits.
        return current_step_sizes, per_move_outputs, rng_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_jit_fns(self) -> list[Callable]:
        """Build one JIT'd callable per move type.

        For ``SingleRun``:
            ``jax.jit(adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10))``
            with the static config baked in via a closure.

        For ``VmapRuns``:
            ``jax.jit(jax.vmap(lambda pop_r, ss_r, emax_r, key_r: adjust_step_size(...)))``
            where the static config is captured in the closure.
        """
        is_vmap = isinstance(self._batch_descriptor, VmapRuns)
        fns = []
        for move_idx, desc in enumerate(self._move_descriptors):
            move_fn = self._per_move_fns[move_idx]
            n_samp = self._adjust_n_samples
            min_r = desc.min_rate
            max_r = desc.max_rate
            afac = self._adjust_factor
            max_ss = desc.step_size_max
            max_rounds = self._adjust_max_rounds

            if is_vmap:
                # Close over all static args so vmap sees no static arg boundary issues.
                def _make_vmap_fn(
                    _move_fn=move_fn,
                    _n_samp=n_samp,
                    _min_r=min_r,
                    _max_r=max_r,
                    _afac=afac,
                    _max_ss=max_ss,
                    _max_rounds=max_rounds,
                ) -> Callable:
                    def _per_run(pop_r, ss_r, emax_r, key_r):
                        return adjust_step_size(
                            pop_r, _move_fn, ss_r, emax_r, key_r,
                            _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
                        )
                    return jax.jit(jax.vmap(_per_run))

                fns.append(_make_vmap_fn())
            else:
                # SingleRun: each call is a scalar pop/ss/emax/key.
                # Close over static args so the same jit object is reused
                # across different step_size values (only dynamic: pop, ss, emax, key).
                def _make_single_fn(
                    _move_fn=move_fn,
                    _n_samp=n_samp,
                    _min_r=min_r,
                    _max_r=max_r,
                    _afac=afac,
                    _max_ss=max_ss,
                    _max_rounds=max_rounds,
                ) -> Callable:
                    def _call(pop, ss, emax, key):
                        return adjust_step_size(
                            pop, _move_fn, ss, emax, key,
                            _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
                        )
                    return jax.jit(_call)

                fns.append(_make_single_fn())

        return fns

    def _split_vmap_keys(
        self, rng_key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Split a (n_runs,) key array into two (n_runs,) arrays.

        Returns ``(carry_keys, trial_keys)`` each of shape ``(n_runs,)``.
        """
        # jax.vmap(jax.random.split)(rng_key) → (n_runs, 2) typed-key array
        pairs = jax.vmap(jax.random.split)(rng_key)  # (n_runs, 2)
        carry = pairs[:, 0]
        trial = pairs[:, 1]
        return carry, trial
