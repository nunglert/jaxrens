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
        trial_batch_size: Optional chunk size for the bisection's inner
            trial-walker vmap.  When set, the per-move closure passes
            ``trial_batch_size`` through to ``adjust_step_size`` /
            ``adjust_step_size_sharded``, converting the inner
            ``vmap(move_fn)`` over ``adjust_n_samples`` trial walkers
            into ``jax.lax.map(batch_size=trial_batch_size)``.  Bounds
            peak memory at ``trial_batch_size * per-walker-state``.
            Useful when the move kernel is expensive (e.g. MACE,
            NeuralIL) and ``adjust_n_samples`` is large.  ``None``
            (default) keeps the full trial vmap — zero overhead.
        run_batch_size: Optional chunk size for the run-axis vmap.
            Only consumed when ``batcher`` is :class:`VmapRuns`;
            ignored for the other batchers (``SingleRun`` has no run
            axis, ``PmapVmapRuns`` is already split by GPU,
            ``ShardedSingleRun`` is one logical run).  When set,
            ``_build_jit_fns`` builds a chunked wrapper that calls
            ``jax.lax.map(per_replica_fn, ..., batch_size=run_batch_size)``
            instead of ``jax.vmap``, bounding peak memory at the
            chunk-replica granularity.  ``None`` (default) keeps the
            full run-axis vmap.
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
        trial_batch_size: int | None = None,
        run_batch_size: int | None = None,
    ) -> None:
        self._move_descriptors = list(move_descriptors) if move_descriptors else []
        self._per_move_fns = list(per_move_fns) if per_move_fns else []
        self._batcher = batcher
        self._adjust_n_samples = adjust_n_samples
        self._adjust_factor = adjust_factor
        self._adjust_max_rounds = adjust_max_rounds
        self._adjust_interval = adjust_interval
        self._trial_batch_size = trial_batch_size
        self._run_batch_size = run_batch_size

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

    def apply_to_state(
        self,
        ns_state,
        emax,
        rng_key,
    ):
        """One-shot adapt + write-back for callers holding an NSState.

        Convenience over :meth:`apply` that does what every caller does
        post-adapt:

        1. Extracts ``current_step_sizes`` from ``ns_state.population``
           via the batcher's ``extract_step_sizes``.
        2. Promotes ``rng_key`` from scalar to ``shape_prefix``-shaped
           per batcher (broadcast onto the ``'shard'`` mesh for
           ShardedSingleRun, ``jax.random.split`` for VmapRuns /
           PmapVmapRuns, identity for SingleRun).  Burn-in's outer
           loop hands us a scalar; the NS loop's pre-shaped key can
           also flow through (the broadcast / split is shape-aware).
        3. Calls :meth:`apply`.
        4. Broadcasts the new per-move step sizes back across the
           walker axis via ``batcher.broadcast_step_sizes`` and writes
           the result into ``ns_state.population.step_sizes``.

        Returns:
            ``(new_ns_state, per_move_outputs, new_rng_key)``.  The
            advanced key matches the input ``rng_key`` shape; callers
            that gave us a scalar get a scalar back, callers that gave
            us a shape_prefix-shaped key get the same shape.
        """
        pop = ns_state.population
        current_step_sizes = self._batcher.extract_step_sizes(pop)

        # Promote scalar rng_key to shape_prefix-shaped on the right mesh.
        # Broadcast (not split) for ShardedSingleRun so every shard sees
        # the same key — load-bearing for coherent-bisection invariants
        # inside ``adjust_step_size_sharded`` (`lax.psum` aggregates
        # per-shard accept counts; identical RNG ⇒ same trial picks ⇒
        # identical bisection path on every shard).
        adapt_key = self._promote_adapt_key(rng_key)

        new_step_sizes, diag, new_key = self.apply(
            pop=pop,
            emax=emax,
            rng_key=adapt_key,
            current_step_sizes=current_step_sizes,
        )

        n_walkers = pop.step_sizes.shape[self._batcher.walker_axis]
        new_ss_pop = self._batcher.broadcast_step_sizes(
            new_step_sizes, n_walkers,
        )
        new_pop = pop.set(step_sizes=new_ss_pop)
        new_ns_state = ns_state.set(population=new_pop)
        return new_ns_state, diag, new_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _promote_adapt_key(self, rng_key: jax.Array) -> jax.Array:
        """Promote a scalar key to ``shape_prefix``-shaped for ``apply``.

        For batched batchers, the per-replica adapt path expects
        ``shape_prefix``-shaped keys.  Burn-in hands in a scalar from
        its outer-loop split chain; the NS loop already has a properly
        shaped key from ``init_ns_*``.  This helper is no-op for
        shape-correct inputs and promotes scalars per batcher:

        * **SingleRun**: identity.
        * **VmapRuns(R)**: ``jax.random.split(key, R)`` → ``(R,)``.
        * **PmapVmapRuns(G, P)**: ``split + reshape`` → ``(G, P)``.
        * **ShardedSingleRun(G)**: BROADCAST (not split) onto the
          ``'shard'``-named mesh → identical key on every shard, on
          the right mesh for the pmap'd per-move adjuster.
        """
        key_ndim = jnp.asarray(rng_key).ndim
        prefix_ndim = len(self._batcher.shape_prefix)
        if key_ndim != 0 or prefix_ndim == 0:
            return rng_key
        if isinstance(self._batcher, ShardedSingleRun):
            from jax.sharding import Mesh, NamedSharding, PartitionSpec
            shard_mesh = NamedSharding(
                Mesh(jax.local_devices()[:self._batcher.n_gpu], ("shard",)),
                PartitionSpec("shard"),
            )
            return jax.device_put(
                jnp.broadcast_to(rng_key[None], (self._batcher.n_gpu,)),
                shard_mesh,
            )
        return jax.random.split(rng_key, self._batcher.n_runs).reshape(
            self._batcher.shape_prefix,
        )

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
            trial_chunk = self._trial_batch_size

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
                _trial_chunk=trial_chunk,
                _desc_name=desc.name,
            ):
                # Fires once per cache miss — i.e. once per distinct
                # signature *per move type*.  After a bucket bump this fires
                # again for every move whose adapt kernel hasn't yet been
                # compiled against the new ``max_neighbors``.
                logger.info(
                    "adapt _per_replica tracing: move=%s  pop_shape=%s  "
                    "max_neighbors=%d  n_samp=%d  max_rounds=%d",
                    _desc_name,
                    pop.positions.shape,
                    int(pop.max_neighbors),
                    int(_n_samp),
                    int(_max_rounds),
                )
                if _trial_chunk is None:
                    return _adjust_fn(
                        pop, _move_fn, ss, emax, key,
                        _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
                    )
                return _adjust_fn(
                    pop, _move_fn, ss, emax, key,
                    _n_samp, _min_r, _max_r, _afac, _max_ss, _max_rounds,
                    trial_batch_size=_trial_chunk,
                )

            fns.append(self._wrap_per_replica(_per_replica))

        return fns

    def _wrap_per_replica(self, per_replica_fn: Callable) -> Callable:
        """Wrap ``per_replica_fn`` for the active batcher + chunking config.

        Equivalent to ``self._batcher.wrap_for_batch(per_replica_fn)`` for
        the default un-chunked path.  When ``self._run_batch_size`` is set
        AND the batcher is :class:`VmapRuns`, replaces the run-axis
        ``jax.vmap`` with ``jax.lax.map(batch_size=...)`` so peak memory
        tracks the chunk-replica granularity instead of the full vmap.
        """
        from jaxrens.sampling.batch_descriptor import VmapRuns

        chunked = (
            self._run_batch_size is not None
            and isinstance(self._batcher, VmapRuns)
        )
        if not chunked:
            return self._batcher.wrap_for_batch(per_replica_fn)

        run_chunk = self._run_batch_size

        def _chunked(pop, ss, emax, run_keys):
            return jax.lax.map(
                lambda x: per_replica_fn(x[0], x[1], x[2], x[3]),
                (pop, ss, emax, run_keys),
                batch_size=run_chunk,
            )

        return jax.jit(_chunked)
