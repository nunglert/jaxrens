"""BatchDescriptor abstraction for the NS outer loop.

Encapsulates the single-run vs. multi-run (vmap) vs. pmap dispatch so that
``run_ns`` and ``run_ns_parallel`` can share the same three concerns without
duplicating them:

1. **wrap_step** — JIT-compiles (and optionally vmaps) ``ns_step``.
2. **split_keys** — Splits a PRNG key appropriately for the batch shape.
3. **reduce_for_termination** — Reduces batched scalars to a single scalar
   for ``PriorMassTermination`` / ``IterationTermination``.

Design intent
-------------
This module is commit 2 of the nested_sampling.py modularization plan.  The
three concrete subclasses mirror the three execution modes:

* ``SingleRun``      — ``run_ns``          uses this.
* ``VmapRuns``       — ``run_ns_parallel`` uses this.
* ``PmapVmapRuns``   — stub only; commit 4 (``run_loop.py``) will flesh it out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Shaped


class BatchDescriptor(ABC):
    """Encapsulates the single/vmap/pmap differences for the NS outer loop.

    Single: identity wrap, scalar termination reductions.
    Vmap:   jax.vmap wrap over n_runs, worst-of reductions.
    Pmap:   pmap(vmap(...)) wrap, not implemented yet.

    Attributes
    ----------
    n_runs : int
        Total number of independent NS runs (1 for SingleRun).
    shape_prefix : tuple[int, ...]
        Leading shape of batched arrays:
        ``()`` for SingleRun, ``(n_runs,)`` for VmapRuns,
        ``(n_gpu, n_per_gpu)`` for PmapVmapRuns.
    """

    n_runs: int
    shape_prefix: tuple[int, ...]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def is_batched(self) -> bool:
        """True when this descriptor represents multiple parallel NS runs.

        Used by ``_run_loop`` to attach ``info["_batcher"]`` and by
        ``_is_batched`` in ``cli/monitor.py`` to prefer a descriptor-based
        check over the ndim-sniff fallback.

        * ``SingleRun.is_batched``    → ``False``
        * ``VmapRuns.is_batched``     → ``True``
        * ``PmapVmapRuns.is_batched`` → ``True`` (stub, raises on use)
        """

    @abstractmethod
    def wrap_step(
        self,
        ns_step_fn,
        step_fn,
        n_mcmc_steps: int,
        n_extra: int,
    ):
        """Return a JIT-compiled callable for the NS step.

        Called *once* before the outer loop to produce ``jit_ns_step``.

        Parameters
        ----------
        ns_step_fn : callable
            The raw ``ns_step`` function (or any compatible replacement).
        step_fn : callable
            MCMC step function from ``build_mwg()``; passed as a static arg
            to ``ns_step_fn``.
        n_mcmc_steps : int
            Number of MCMC steps per walker (static).
        n_extra : int
            Number of additional walkers to walk per iteration (static).

        Returns
        -------
        callable
            * **SingleRun** — ``jax.jit(ns_step_fn, static_argnums=(1, 2, 3))``
              so the returned callable has signature
              ``jit_step(ns_state, step_fn, n_mcmc_steps, n_extra)``.
            * **VmapRuns** — ``jax.jit(lambda ns_states: jax.vmap(lambda s:
              ns_step_fn(s, step_fn, n_mcmc_steps, n_extra))(ns_states))``
              so the returned callable has signature
              ``jit_step(ns_states)``.
            * **PmapVmapRuns** — raises ``NotImplementedError``.
        """

    @abstractmethod
    def split_keys(self, rng_key: jax.Array, n_sub_keys: int) -> jax.Array:
        """Split a PRNG key appropriate for this batch shape.

        Called inside the outer loop whenever fresh sub-keys are needed
        (e.g. for per-move adaptation key splitting).

        Parameters
        ----------
        rng_key : jax.Array
            * **SingleRun** — a scalar JAX key (shape ``(2,)`` for typed-key
              arrays or ``()`` for new-style keys).
            * **VmapRuns** — an ``(n_runs,)`` array of per-run keys.
        n_sub_keys : int
            Number of sub-keys to generate per run.

        Returns
        -------
        jax.Array
            * **SingleRun** — shape ``(n_sub_keys,)`` with typed-key dtype
              (matches ``jax.random.split(rng_key, n_sub_keys)``).
            * **VmapRuns** — shape ``(n_runs, n_sub_keys)`` with typed-key dtype
              (``vmap(jax.random.split)`` applied to each run key).
            * **PmapVmapRuns** — raises ``NotImplementedError``.
        """

    @abstractmethod
    def reduce_for_termination(
        self,
        log_evidence: jax.typing.ArrayLike,
        hmax: jax.typing.ArrayLike,
    ) -> tuple[float, float]:
        """Reduce batched scalars for the termination-check interface.

        ``PriorMassTermination.update_evidence`` and
        ``check_any`` / ``IterationTermination.check`` both require plain
        Python floats.  This method performs any worst-case aggregation
        and returns Python floats ready to pass to those interfaces.

        Parameters
        ----------
        log_evidence : jax.Array
            * **SingleRun** — scalar.
            * **VmapRuns** — shape ``(n_runs,)``.
        hmax : jax.Array
            Same shape as ``log_evidence``.

        Returns
        -------
        (log_evidence_scalar, hmax_scalar) : tuple[float, float]
            * **SingleRun** — identity: ``(float(log_evidence), float(hmax))``.
            * **VmapRuns** — worst-of:
              ``(float(jnp.min(log_evidence)), float(jnp.max(hmax)))``.
            * **PmapVmapRuns** — raises ``NotImplementedError``.
        """

    # ------------------------------------------------------------------
    # Derived helpers (shape-prefix-driven; concrete classes inherit)
    # ------------------------------------------------------------------

    @property
    def walker_axis(self) -> int:
        """Axis index of the per-walker (K) dimension in population arrays.

        ``0`` for SingleRun, ``1`` for VmapRuns, ``2`` for PmapVmapRuns.
        """
        return len(self.shape_prefix)

    def flatten(
        self, arr: Shaped[np.ndarray | Array, "*P ..."]
    ) -> Shaped[np.ndarray | Array, "n_runs ..."]:
        """Collapse the leading shape-prefix dims into a single ``(n_runs,)`` axis.

        For SingleRun (no prefix) prepends a length-1 axis so the output is
        always ``(n_runs, *trailing)``.
        """
        n_prefix = len(self.shape_prefix)
        if n_prefix == 0:
            return arr[None, ...]
        if n_prefix == 1:
            return arr
        return arr.reshape((self.n_runs,) + arr.shape[n_prefix:])

    def unflatten(
        self, arr_flat: Shaped[np.ndarray | Array, "n_runs ..."]
    ) -> Shaped[np.ndarray | Array, "*P ..."]:
        """Inverse of :meth:`flatten`."""
        n_prefix = len(self.shape_prefix)
        if n_prefix == 0:
            return arr_flat[0]
        if n_prefix == 1:
            return arr_flat
        return arr_flat.reshape(self.shape_prefix + arr_flat.shape[1:])

    def extract_step_sizes(self, pop) -> jnp.ndarray:
        """Per-replica step sizes from ``pop.step_sizes (*prefix, K, n_moves)``.

        Returns shape ``(*shape_prefix, n_moves)`` — walker axis dropped.
        """
        return jnp.take(pop.step_sizes, 0, axis=self.walker_axis)

    def broadcast_step_sizes(
        self, per_move_ss: jnp.ndarray, n_walkers: int,
    ) -> jnp.ndarray:
        """Inverse of :meth:`extract_step_sizes` — re-insert the walker axis.

        Given ``per_move_ss`` of shape ``(*shape_prefix, n_moves)``, returns
        shape ``(*shape_prefix, K, n_moves)`` broadcast across the walker axis.
        """
        n_moves = per_move_ss.shape[-1]
        target_shape = self.shape_prefix + (n_walkers, n_moves)
        return jnp.broadcast_to(
            jnp.expand_dims(per_move_ss, axis=self.walker_axis),
            target_shape,
        )

    def reduce_emax(
        self, energy: Float[Array, "*P K"]
    ) -> Float[Array, "*P"]:
        """Per-replica Emax: ``max`` along the walker axis.

        Returns scalar for SingleRun, ``(*shape_prefix,)`` otherwise.
        """
        return jnp.max(energy, axis=self.walker_axis)

    def scalar_key(self, rng_key: jax.Array) -> jax.Array:
        """Reduce a per-replica key array to a single scalar key.

        Identity for SingleRun (already scalar); takes replica ``(0,)``
        for VmapRuns / PmapVmapRuns.  Used by ``_run_loop`` to derive the
        single key the inter-RE swap path needs from the per-replica
        adaptation key carry.
        """
        arr = jnp.asarray(rng_key)
        return arr if arr.ndim == 0 else arr.reshape(-1)[0]

    def distinct_keys(self, rng_key: jax.Array) -> jax.Array:
        """Split a scalar key into ``shape_prefix``-shaped INDEPENDENT keys.

        Distinct from :meth:`split_keys`: that one returns COHERENT
        (same-key-broadcast for ShardedSingleRun) sub-keys for adapt
        bisection.  This one returns INDEPENDENT keys — each replica /
        shard gets a different RNG stream.  Used by burn-in's walking
        step where each replica's walkers evolve independently.

        Default: ``jax.random.split(rng_key, n_runs).reshape(shape_prefix)``
        for batched batchers; identity for SingleRun.  ShardedSingleRun
        overrides to place the result on the ``'shard'``-named mesh.
        """
        if not self.is_batched:
            return rng_key
        return jax.random.split(rng_key, self.n_runs).reshape(
            self.shape_prefix,
        )

    def wrap_for_batch(self, per_element_fn):
        """Wrap a per-replica callable with jit/vmap/pmap as appropriate.

        Generic version of :meth:`wrap_step` for callables that don't take
        ``static_argnums``-style sentinels.  *per_element_fn* receives its
        arguments at single-replica shape; the returned callable accepts
        them at ``(*shape_prefix, ...)`` shape:

        * **SingleRun** — ``jax.jit(per_element_fn)``.
        * **VmapRuns** — ``jax.jit(jax.vmap(per_element_fn))``.
        * **PmapVmapRuns** — ``jax.pmap(jax.vmap(per_element_fn), axis_name="gpu")``.

        Default implementation routes by ``shape_prefix`` length so concrete
        classes inherit unchanged.
        """
        n_prefix = len(self.shape_prefix)
        if n_prefix == 0:
            return jax.jit(per_element_fn)
        if n_prefix == 1:
            return jax.jit(jax.vmap(per_element_fn))
        # n_prefix == 2 (PmapVmapRuns): outer pmap over G, inner vmap over P.
        # pmap is self-JIT-compiling — do NOT wrap in jax.jit.
        return jax.pmap(jax.vmap(per_element_fn), axis_name="gpu")


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleRun(BatchDescriptor):
    """Descriptor for a single NS run (no batching).

    Used by ``run_ns``.  All three methods are identity / thin wrappers
    around plain JAX primitives.

    Attributes
    ----------
    n_runs : int
        Always 1.
    shape_prefix : tuple[int, ...]
        Always ``()``.
    """

    n_runs: int = 1
    shape_prefix: tuple[int, ...] = ()

    @property
    def is_batched(self) -> bool:
        """Always ``False`` — single NS run, no batch dimension."""
        return False

    def wrap_step(
        self,
        ns_step_fn,
        step_fn,
        n_mcmc_steps: int,
        n_extra: int,
    ):
        """Return ``jax.jit(ns_step_fn, static_argnums=(1, 2, 3))``.

        The caller uses the returned function as::

            jit_step(ns_state, step_fn, n_mcmc_steps, n_extra)

        which is exactly the current ``run_ns`` pattern.
        """
        return jax.jit(ns_step_fn, static_argnums=(1, 2, 3))

    def split_keys(self, rng_key: jax.Array, n_sub_keys: int) -> jax.Array:
        """Delegate directly to ``jax.random.split``.

        Returns shape ``(n_sub_keys,)`` with typed-key dtype.
        """
        return jax.random.split(rng_key, n_sub_keys)

    def reduce_for_termination(
        self,
        log_evidence: jax.typing.ArrayLike,
        hmax: jax.typing.ArrayLike,
    ) -> tuple[float, float]:
        """Identity reduction — single run has no worst-case to aggregate.

        Returns
        -------
        (float(log_evidence), float(hmax))
        """
        return float(log_evidence), float(hmax)


@dataclass(frozen=True)
class VmapRuns(BatchDescriptor):
    """Descriptor for n_runs independent NS runs batched via ``jax.vmap``.

    Used by ``run_ns_parallel``.

    Attributes
    ----------
    n_runs : int
        Number of independent NS runs.
    shape_prefix : tuple[int, ...]
        ``(n_runs,)`` — leading dimension of all batched arrays.
    """

    n_runs: int
    shape_prefix: tuple[int, ...] = ()  # overridden by __post_init__

    def __post_init__(self) -> None:
        # dataclass(frozen=True) requires object.__setattr__ for mutation
        object.__setattr__(self, "shape_prefix", (self.n_runs,))

    @property
    def is_batched(self) -> bool:
        """Always ``True`` — multiple NS runs are stacked along axis 0."""
        return True

    def wrap_step(
        self,
        ns_step_fn,
        step_fn,
        n_mcmc_steps: int,
        n_extra: int,
    ):
        """Return a JIT-compiled vmapped NS step.

        The closure captures ``step_fn``, ``n_mcmc_steps``, and ``n_extra``
        as Python-level static values so vmap sees no dynamic static args.

        The returned function has signature::

            jit_step(ns_states)  ->  (ns_states, infos)

        which matches the current ``run_ns_parallel`` inner step pattern.
        """
        def step_all_runs(ns_states):
            return jax.vmap(
                lambda s: ns_step_fn(s, step_fn, n_mcmc_steps, n_extra)
            )(ns_states)

        return jax.jit(step_all_runs)

    def split_keys(self, rng_key: jax.Array, n_sub_keys: int) -> jax.Array:
        """Vmap ``jax.random.split`` over the ``(n_runs,)`` key array.

        Parameters
        ----------
        rng_key : jax.Array
            Shape ``(n_runs,)`` — one key per run.
        n_sub_keys : int
            Number of sub-keys to produce per run.

        Returns
        -------
        jax.Array
            Shape ``(n_runs, n_sub_keys)`` with typed-key dtype.
        """
        return jax.vmap(
            lambda k: jax.random.split(k, n_sub_keys)
        )(rng_key)

    def reduce_for_termination(
        self,
        log_evidence: jax.typing.ArrayLike,
        hmax: jax.typing.ArrayLike,
    ) -> tuple[float, float]:
        """Worst-of reduction across all runs.

        Matches the current ``run_ns_parallel`` termination logic::

            worst_evidence = float(jnp.min(ns_states.log_evidence))
            worst_hmax     = float(jnp.max(infos["hmax"]))

        Returns
        -------
        (float(jnp.min(log_evidence)), float(jnp.max(hmax)))
        """
        return float(jnp.min(log_evidence)), float(jnp.max(hmax))


@dataclass(frozen=True)
class PmapVmapRuns(BatchDescriptor):
    """Descriptor for ``pmap(vmap(...))`` multi-GPU multi-run NS.

    Shape convention: ``(G, P, K, ...)`` where

    * ``G = n_gpu`` — pmap axis (one shard per GPU device).
    * ``P = n_per_gpu`` — vmap axis (independent NS runs per GPU).
    * ``K`` — walker axis (already handled inside ``ns_step`` via vmap).

    For ``n_gpu=1`` this degenerates to ``(1, P, K, ...)`` and pmap runs on
    the single available device, which is useful for testing.

    **wrap_step** composes ``jax.pmap`` (G axis) over ``jax.vmap`` (P axis):

    .. code-block:: python

        per_run   = lambda s: ns_step_fn(s, step_fn, n_mcmc_steps, n_extra)
        per_device = jax.vmap(per_run)
        jit_step   = jax.pmap(per_device, axis_name="gpu")

    ``jax.pmap`` is self-JIT-compiling; do **not** wrap in an additional
    ``jax.jit`` — that causes XLA sharding conflicts.

    **split_keys** produces ``(G, P, n_sub_keys)``-shaped key arrays via two
    nested splits: first ``(G,)`` per-GPU keys, then ``(P, n_sub_keys)`` per
    GPU via ``jax.vmap``.

    **reduce_for_termination** takes the worst-of across both G and P axes
    (same as VmapRuns but over a 2-D input rather than 1-D).

    Attributes
    ----------
    n_gpu : int
        Number of GPU devices (G).
    n_per_gpu : int
        Number of NS runs per GPU (P).
    n_runs : int
        Total runs: ``n_gpu * n_per_gpu``.
    shape_prefix : tuple[int, ...]
        ``(n_gpu, n_per_gpu)`` — leading shape of all batched arrays.
    """

    n_gpu: int
    n_per_gpu: int
    n_runs: int = 0           # set by __post_init__
    shape_prefix: tuple[int, ...] = ()  # set by __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_runs", self.n_gpu * self.n_per_gpu)
        object.__setattr__(self, "shape_prefix", (self.n_gpu, self.n_per_gpu))

    @property
    def is_batched(self) -> bool:
        """Always ``True`` — multiple NS runs are distributed across G×P."""
        return True

    def wrap_step(
        self,
        ns_step_fn,
        step_fn,
        n_mcmc_steps: int,
        n_extra: int,
    ):
        """Return a pmap(vmap(...)) NS step callable.

        The returned callable has signature::

            pmap_step(ns_states)  ->  (ns_states, infos)

        where ``ns_states`` has leading shape ``(G, P, ...)``.

        ``jax.pmap`` is already JIT-compiled internally; the returned
        function is **not** additionally wrapped in ``jax.jit``.

        Parameters
        ----------
        ns_step_fn : callable
            Raw ``ns_step`` (or compatible replacement).
        step_fn : callable
            MCMC step function captured in the closure (static for vmap).
        n_mcmc_steps : int
            Number of MCMC steps per walker (static).
        n_extra : int
            Number of additional walkers per iteration (static).

        Returns
        -------
        callable
            ``jax.pmap(jax.vmap(per_run), axis_name="gpu")`` where
            ``per_run(s) = ns_step_fn(s, step_fn, n_mcmc_steps, n_extra)``.
        """
        # Close over Python-level statics to avoid static_argnums under pmap.
        def per_run(s):
            return ns_step_fn(s, step_fn, n_mcmc_steps, n_extra)

        per_device = jax.vmap(per_run)
        # pmap handles JIT internally — do NOT add jax.jit here.
        return jax.pmap(per_device, axis_name="gpu")

    def split_keys(self, rng_key: jax.Array, n_sub_keys: int) -> jax.Array:
        """Split a per-replica PRNG key array into ``(G, P, n_sub_keys)``.

        Matches the ABC contract: input shape equals ``shape_prefix``
        (``(G, P)`` here), output prepends an ``n_sub_keys`` axis at the end.
        Implemented as a 2-D vmap (one for each prefix axis) of
        ``jax.random.split``.

        Parameters
        ----------
        rng_key : jax.Array
            Shape ``(G, P)`` — one key per replica.  Same convention as
            ``NSState.rng_key`` for PmapVmapRuns.
        n_sub_keys : int
            Number of sub-keys per replica.

        Returns
        -------
        jax.Array
            Shape ``(G, P, n_sub_keys)`` with typed-key dtype.
        """
        return jax.vmap(jax.vmap(
            lambda k: jax.random.split(k, n_sub_keys)
        ))(rng_key)

    def reduce_for_termination(
        self,
        log_evidence: jax.typing.ArrayLike,
        hmax: jax.typing.ArrayLike,
    ) -> tuple[float, float]:
        """Worst-of reduction across both G and P axes.

        Parameters
        ----------
        log_evidence : jax.Array
            Shape ``(G, P)``.
        hmax : jax.Array
            Shape ``(G, P)``.

        Returns
        -------
        (float(jnp.min(log_evidence)), float(jnp.max(hmax)))
            Worst-case scalars across all runs on all GPUs.
        """
        return float(jnp.min(log_evidence)), float(jnp.max(hmax))


@dataclass(frozen=True)
class ShardedSingleRun(BatchDescriptor):
    """Descriptor for a single NS run sharded across ``n_gpu`` GPUs.

    Logically one NS run; physically the ``K``-walker population is
    split into ``n_gpu`` chunks of ``K // n_gpu`` walkers, one per
    device.  ``ns_step_sharded`` uses ``lax.all_gather`` /
    ``lax.argmax`` / ``lax.psum`` collectives across the
    ``"shard"`` axis to act on the global population coherently
    while letting each device hold only its share.  Use case:
    memory scaling for heavy backends (MACE, NeuralIL, Nequix) on
    large populations that overflow a single GPU.

    Distinct from :class:`PmapVmapRuns` — that class runs ``G * P``
    *independent* NS replicas.  ``ShardedSingleRun`` runs one
    population spread across G devices.

    Shape conventions:

    * Population leaves: ``(G, K // G, ...)``.
    * ``shape_prefix = (n_gpu,)`` — matches the physical layout.
      The ``n_runs == prod(shape_prefix)`` invariant that the other
      batchers happen to satisfy does NOT hold here: ``n_runs = 1``
      logically, but the leading ``G`` axis is the sharding axis,
      not a replica axis.  Consumers iterating ``shape_prefix`` (the
      cumulative counters in ``_run_loop``, the per-move
      ``stack_axis`` in ``build_adapt_step``) end up with a length-G
      axis where each row is identical post-``lax.psum``.  Correct;
      G× redundant memory on small counters.  Acceptable cost.

    Reductions (``reduce_emax``, ``reduce_for_termination``) collapse
    the (G,) axis to a scalar — every shard sees the same global
    value after the in-step collectives.

    Inputs to :meth:`wrap_for_batch`-wrapped callables are scalar
    (ss / emax / key); the wrapper broadcasts to (G,) before
    ``pmap`` and takes ``[0]`` of the result on the way out.  This
    keeps the per-replica callable signature identical to
    SingleRun so :func:`build_adapt_step` doesn't need a different
    closure shape — only the underlying ``adjust_step_size_sharded``
    call needs to know it's running under pmap.

    Attributes
    ----------
    n_gpu : int
        Number of GPU devices (G) the population is sharded across.
    n_runs : int
        Always 1 — one logical NS run.
    shape_prefix : tuple[int, ...]
        ``(n_gpu,)`` — physical sharding axis.
    """

    n_gpu: int
    n_runs: int = 1
    shape_prefix: tuple[int, ...] = ()  # set by __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_prefix", (self.n_gpu,))

    @property
    def is_batched(self) -> bool:
        """Always ``True`` — the population is distributed across G devices."""
        return True

    def wrap_step(
        self,
        ns_step_fn,
        step_fn,
        n_mcmc_steps: int,
        n_extra: int,
    ):
        """Return a ``pmap``-compiled NS step callable.

        ``ns_step_fn`` here should be ``ns_step_sharded`` (not the
        plain ``ns_step``) — it uses ``lax.all_gather`` / ``lax.psum``
        collectives that only work inside a ``pmap`` context with
        matching ``axis_name="shard"``.

        The returned callable has signature::

            pmap_step(ns_state)  ->  (ns_state, info)

        with leading shape ``(G, ...)`` on every leaf.
        """
        def per_shard(ns_state):
            return ns_step_fn(ns_state, step_fn, n_mcmc_steps, n_extra)

        return jax.pmap(per_shard, axis_name="shard")

    def split_keys(self, rng_key: jax.Array, n_sub_keys: int) -> jax.Array:
        """Split a (G,)-broadcast key into ``(G, n_sub_keys)`` sub-keys.

        ``rng_key`` may be either:

        * A scalar key — auto-broadcast to ``(G,)`` (every shard then
          gets the same sub-keys post-split).
        * A ``(G,)`` array of identical broadcast keys — same outcome,
          just skips the broadcast step.

        The load-bearing invariant for ``ns_step_sharded`` and
        ``adjust_step_size_sharded`` is that every shard makes the
        *same* RNG decisions, so the broadcast-then-split pattern is
        equivalent to "every shard splits the same key" — used by
        the ``adapt_step`` closure (which sees a ``(G,)``-shaped
        key from its ``rng_key`` parameter).

        Returns shape ``(G, n_sub_keys)`` with typed-key dtype.
        """
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        # Collapse (G,) → scalar so jax.random.split (which only accepts
        # a single key) works.  All G entries are identical by
        # construction (broadcast upstream); take [0] is exact.
        scalar_key = rng_key[0] if rng_key.ndim > 0 else rng_key
        sub = jax.random.split(scalar_key, n_sub_keys)  # (n_sub_keys,)
        broadcast = jnp.broadcast_to(sub[None, ...], (self.n_gpu,) + sub.shape)
        # Place on the same shard mesh as ``reduce_emax`` /
        # ``extract_step_sizes`` so the per-move pmap accepts every
        # input uniformly without per-call ``jax.device_put``.
        shard_mesh = NamedSharding(
            Mesh(jax.local_devices()[:self.n_gpu], ("shard",)),
            PartitionSpec("shard"),
        )
        return jax.device_put(broadcast, shard_mesh)

    def reduce_for_termination(
        self,
        log_evidence: jax.typing.ArrayLike,
        hmax: jax.typing.ArrayLike,
    ) -> tuple[float, float]:
        """Take ``[0]`` from the (G,) axis — every shard has the same value.

        ``ns_step_sharded`` writes identical ``log_evidence`` /
        ``hmax`` on every shard (post-psum), so ``[0]`` is exact.
        """
        log_z_arr = jnp.asarray(log_evidence)
        hmax_arr = jnp.asarray(hmax)
        log_z = log_z_arr[0] if log_z_arr.ndim > 0 else log_z_arr
        h = hmax_arr[0] if hmax_arr.ndim > 0 else hmax_arr
        return float(log_z), float(h)

    # ------------------------------------------------------------------
    # Override base helpers where the (G,) prefix needs special treatment
    # ------------------------------------------------------------------

    def reduce_emax(
        self, energy: Float[Array, "G K"]
    ) -> Float[Array, "G"]:
        """Global ``max`` over the entire (G, K_per_gpu) population, broadcast to (G,).

        Returns shape ``(G,)`` (every entry identical) on the
        ``'shard'``-named mesh — same convention as
        :meth:`VmapRuns.reduce_emax` returning ``(R,)`` and
        :meth:`PmapVmapRuns.reduce_emax` returning ``(G, P)``.
        Following the shape_prefix convention everywhere lets
        consumers (the ``adapt_step`` closure, `_run_loop`'s
        adaptation block) handle every batcher uniformly without
        per-batcher emax broadcast/reshape.

        The G entries are physically identical (one logical Emax for
        one logical population); the redundancy is G floats per
        adapt call — trivial vs. the per-shard isinstance branches it
        eliminates.
        """
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        scalar = jnp.max(energy)
        broadcast = jnp.broadcast_to(scalar[None], (self.n_gpu,))
        shard_mesh = NamedSharding(
            Mesh(jax.local_devices()[:self.n_gpu], ("shard",)),
            PartitionSpec("shard"),
        )
        return jax.device_put(broadcast, shard_mesh)

    def flatten(
        self, arr: Shaped[np.ndarray | Array, "G ..."]
    ) -> Shaped[np.ndarray | Array, "1 ..."]:
        """Take ``[0:1]`` from the G axis — one logical run.

        All shards hold identical per-run data after ``lax.psum`` in
        ``ns_step_sharded`` / ``adjust_step_size_sharded``, so
        slicing one shard is exact.  Returns a length-1 leading
        axis to match the ``flatten`` contract (``(n_runs, ...)``).
        """
        arr_np = np.asarray(arr) if isinstance(arr, np.ndarray) else arr
        return arr_np[:1]

    def unflatten(
        self, arr_flat: Shaped[np.ndarray | Array, "1 ..."]
    ) -> Shaped[np.ndarray | Array, "G ..."]:
        """Re-broadcast a length-1 leading axis to (G, ...)."""
        return jnp.broadcast_to(arr_flat[0:1], (self.n_gpu,) + arr_flat.shape[1:])

    def wrap_for_batch(self, per_element_fn):
        """Wrap a per-replica callable in ``pmap`` over the shard axis.

        Signature contract: every input has a leading ``(G,)`` axis on
        the ``'shard'``-named mesh; outputs preserve that axis.  This
        matches :meth:`VmapRuns.wrap_for_batch` and
        :meth:`PmapVmapRuns.wrap_for_batch`'s contracts (each "row"
        of the leading axis is fed to ``per_element_fn``).

        For ShardedSingleRun the per-shard "row" of every coherent
        input (ss / emax / key) is identical by construction — that's
        the load-bearing invariant for ``adjust_step_size_sharded``'s
        coherent bisection.  Producers (``reduce_emax``,
        ``split_keys``, ``extract_step_sizes``) all return shape
        ``(G, ...)`` on the shard mesh, so callers don't need to
        massage shapes themselves.
        """
        return jax.pmap(per_element_fn, axis_name="shard")

    def distinct_keys(self, rng_key: jax.Array) -> jax.Array:
        """Split a scalar key into G INDEPENDENT keys on the shard mesh.

        Overrides the base default (``split + reshape``) to also
        ``device_put`` the result onto the ``'shard'``-named mesh so
        the per-shard pmap accepts it alongside the sharded NSState.

        Returns shape ``(G,)`` typed-key array sharded along axis 0.
        """
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
        keys = jax.random.split(rng_key, self.n_gpu)
        shard_mesh = NamedSharding(
            Mesh(jax.local_devices()[:self.n_gpu], ("shard",)),
            PartitionSpec("shard"),
        )
        return jax.device_put(keys, shard_mesh)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def from_shape_prefix(shape_prefix: tuple[int, ...]) -> BatchDescriptor:
    """Return the batcher whose ``shape_prefix`` matches *shape_prefix*.

    * ``()``      → :class:`SingleRun`.
    * ``(R,)``    → :class:`VmapRuns(n_runs=R)`.
    * ``(G, P)``  → :class:`PmapVmapRuns(n_gpu=G, n_per_gpu=P)`.

    Used by ``init/restart.py``, ``io/checkpoint.py``, and ``cli/monitor.py``
    to recover the batcher from a stored array's leading shape (``log_evidence``
    is the canonical witness — its shape is always exactly the prefix).

    Raises
    ------
    ValueError
        If *shape_prefix* has more than two leading dims.
    """
    prefix = tuple(int(d) for d in shape_prefix)
    if len(prefix) == 0:
        return SingleRun()
    if len(prefix) == 1:
        return VmapRuns(n_runs=prefix[0])
    if len(prefix) == 2:
        return PmapVmapRuns(n_gpu=prefix[0], n_per_gpu=prefix[1])
    raise ValueError(
        f"from_shape_prefix: only rank-0/1/2 prefixes are supported, got "
        f"shape_prefix={shape_prefix} (rank {len(prefix)})."
    )
