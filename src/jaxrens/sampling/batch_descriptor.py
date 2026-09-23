"""Execution-topology descriptors for the NS outer loop.

Four execution topologies exist — ``SingleRun``, ``VmapRuns``,
``PmapVmapRuns``, ``ShardedSingleRun`` — encapsulating the shape/RNG/
reduction bookkeeping so ``_run_loop`` (``run_loop.py``) and its callbacks
can share one orchestration body without per-topology ``isinstance``
branching for most concerns:

1. **wrap_step** — JIT-compiles (and vmaps/shard_maps as appropriate) the NS
   step.
2. **split_keys** — Splits a PRNG key appropriately for the batch shape.
3. **reduce_for_termination** — Reduces batched scalars to a single scalar
   for ``PriorMassTermination`` / ``IterationTermination``.

``BatchDescriptor`` (bottom of this module) is the ``SingleRun | VmapRuns |
PmapVmapRuns | ShardedSingleRun`` type used to annotate "any of the four" —
not a base class. ``SingleRun``, ``VmapRuns``, and ``PmapVmapRuns`` share a
concrete (non-abstract) base, ``_UniformBatcher``, because they genuinely
share behavior: each satisfies ``n_runs == prod(shape_prefix)`` and treats
its leading shape axes as independent-replica axes. ``ShardedSingleRun``
does **not** inherit from it — its ``shape_prefix = (n_gpu,)`` is a
*sharding* axis, not a replica axis (``n_runs`` is always 1), so several of
``_UniformBatcher``'s methods don't apply to it as written (see its own
docstring). Rather than force it to override most of a shared contract to
fit a hierarchy whose invariants it breaks, it stands alone with its own
implementations — some duplicated from ``_UniformBatcher`` where the
generic, shape_prefix-driven logic happens to still be correct for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Shaped

from jaxrens.sampling.mesh import build_mesh, pmap_like


class _UniformBatcher:
    """Shared behavior for the three "uniform shape-prefix" topologies.

    ``SingleRun``, ``VmapRuns``, and ``PmapVmapRuns`` all treat their
    leading ``shape_prefix`` axes as independent-replica axes, with
    ``n_runs == prod(shape_prefix)`` holding exactly. That shared
    invariant is what the "derived helpers" below rely on. ``ShardedSingleRun``
    intentionally does not inherit from this class — see the module
    docstring.

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
        self,
        per_move_ss: jnp.ndarray,
        n_walkers: int,
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

    def reduce_emax(self, energy: Float[Array, "*P K"]) -> Float[Array, "*P"]:
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
        bisection.  This one returns INDEPENDENT keys — each replica
        gets a different RNG stream.  Used by burn-in's walking step
        where each replica's walkers evolve independently.

        ``jax.random.split(rng_key, n_runs).reshape(shape_prefix)`` for
        batched batchers; identity for SingleRun.
        """
        if not self.is_batched:  # type: ignore[attr-defined]
            return rng_key
        return jax.random.split(rng_key, self.n_runs).reshape(
            self.shape_prefix,
        )

    def wrap_for_batch(self, per_element_fn, *, check_vma: bool = False):
        """Wrap a per-replica callable with jit/vmap/shard_map as appropriate.

        Generic version of ``wrap_step`` for callables that don't take
        ``static_argnums``-style sentinels.  *per_element_fn* receives its
        arguments at single-replica shape; the returned callable accepts
        them at ``(*shape_prefix, ...)`` shape:

        * **SingleRun** — ``jax.jit(per_element_fn)``.
        * **VmapRuns** — ``jax.jit(jax.vmap(per_element_fn))``.
        * **PmapVmapRuns** — ``jax.jit(shard_map(jax.vmap(per_element_fn)))``,
          outer ``"gpu"``-axis shard_map (pmap-equivalent, see
          :mod:`jaxrens.sampling.mesh`) over G, inner vmap over P.

        ``check_vma`` is forwarded to :func:`jaxrens.sampling.mesh.pmap_like`
        for the PmapVmapRuns case (ignored otherwise, since SingleRun/
        VmapRuns never enter a mapped context) — see that function's
        docstring.

        Default implementation routes by ``shape_prefix`` length so concrete
        classes inherit unchanged.
        """
        n_prefix = len(self.shape_prefix)
        if n_prefix == 0:
            return jax.jit(per_element_fn)
        if n_prefix == 1:
            return jax.jit(jax.vmap(per_element_fn))
        # n_prefix == 2 (PmapVmapRuns): outer shard_map over G, inner vmap
        # over P. Unlike jax.pmap, shard_map composes with jax.jit.
        # self.mesh: only PmapVmapRuns reaches this branch, and it defines
        # the cached `mesh` property this base class doesn't declare.
        mesh = self.mesh  # type: ignore[attr-defined]
        return jax.jit(
            pmap_like(
                jax.vmap(per_element_fn), "gpu", mesh, check_vma=check_vma
            )
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleRun(_UniformBatcher):
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
class VmapRuns(_UniformBatcher):
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
        return jax.vmap(lambda k: jax.random.split(k, n_sub_keys))(rng_key)

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
class PmapVmapRuns(_UniformBatcher):
    """Descriptor for ``shard_map(vmap(...))`` multi-GPU multi-run NS.

    Shape convention: ``(G, P, K, ...)`` where

    * ``G = n_gpu`` — sharded axis (one shard per GPU device).
    * ``P = n_per_gpu`` — vmap axis (independent NS runs per GPU).
    * ``K`` — walker axis (already handled inside ``ns_step`` via vmap).

    For ``n_gpu=1`` this degenerates to ``(1, P, K, ...)`` and runs on the
    single available device, which is useful for testing.

    **wrap_step** composes a ``"gpu"``-axis ``jax.shard_map`` (G axis, via
    :func:`jaxrens.sampling.mesh.pmap_like`, pmap-equivalent) over
    ``jax.vmap`` (P axis):

    .. code-block:: python

        per_run    = lambda s: ns_step_fn(s, step_fn, n_mcmc_steps, n_extra)
        per_device = jax.vmap(per_run)
        jit_step   = jax.jit(pmap_like(per_device, "gpu", mesh))

    Unlike ``jax.pmap``, ``shard_map`` composes cleanly with ``jax.jit``, so
    the result is explicitly jit-wrapped.

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
    n_runs: int = 0  # set by __post_init__
    shape_prefix: tuple[int, ...] = ()  # set by __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_runs", self.n_gpu * self.n_per_gpu)
        object.__setattr__(self, "shape_prefix", (self.n_gpu, self.n_per_gpu))

    @property
    def is_batched(self) -> bool:
        """Always ``True`` — multiple NS runs are distributed across G×P."""
        return True

    @cached_property
    def mesh(self):
        """The ``"gpu"``-axis :class:`jax.sharding.Mesh` for this topology.

        Built once per instance (not per ``wrap_step``/``wrap_for_batch``
        call) and shared by every consumer — ``InterREManager`` pulls this
        same object rather than building its own equivalent-but-distinct
        mesh.
        """
        return build_mesh("gpu", self.n_gpu)

    def wrap_step(
        self,
        ns_step_fn,
        step_fn,
        n_mcmc_steps: int,
        n_extra: int,
    ):
        """Return a shard_map(vmap(...)) NS step callable (pmap-equivalent).

        The returned callable has signature::

            pmap_step(ns_states)  ->  (ns_states, infos)

        where ``ns_states`` has leading shape ``(G, P, ...)``.

        Built on ``jax.shard_map`` via :func:`jaxrens.sampling.mesh.pmap_like`
        rather than ``jax.pmap`` — this is the ``"gpu"``-axis leg of the
        pmap -> shard_map migration. Unlike ``jax.pmap``, ``shard_map``
        composes with ``jax.jit``, so the returned callable is explicitly
        jit-wrapped.

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
            ``jax.jit(pmap_like(jax.vmap(per_run), "gpu", mesh))`` where
            ``per_run(s) = ns_step_fn(s, step_fn, n_mcmc_steps, n_extra)``.
        """

        # Close over Python-level statics, same as under the old pmap path.
        def per_run(s):
            return ns_step_fn(s, step_fn, n_mcmc_steps, n_extra)

        per_device = jax.vmap(per_run)
        return jax.jit(pmap_like(per_device, "gpu", self.mesh))

    def split_keys(self, rng_key: jax.Array, n_sub_keys: int) -> jax.Array:
        """Split a per-replica PRNG key array into ``(G, P, n_sub_keys)``.

        Input shape equals ``shape_prefix`` (``(G, P)`` here), output
        prepends an ``n_sub_keys`` axis at the end.  Implemented as a 2-D
        vmap (one for each prefix axis) of ``jax.random.split``.

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
        return jax.vmap(jax.vmap(lambda k: jax.random.split(k, n_sub_keys)))(
            rng_key
        )

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
class ShardedSingleRun:
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

    Deliberately does **not** inherit from ``_UniformBatcher`` (see the
    module docstring): ``n_runs`` is always 1 here, so the
    ``n_runs == prod(shape_prefix)`` invariant the other three batchers
    share does not hold — the leading ``G`` axis is a *sharding* axis, not
    a replica axis. Four small, purely shape_prefix-driven methods
    (``walker_axis``, ``extract_step_sizes``, ``broadcast_step_sizes``,
    ``scalar_key``) are still correct as written for this shape and are
    duplicated below rather than shared through inheritance, to avoid
    pretending this class belongs to that family.

    Shape conventions:

    * Population leaves: ``(G, K // G, ...)``.
    * ``shape_prefix = (n_gpu,)`` — matches the physical layout.
      Consumers iterating ``shape_prefix`` (the cumulative counters in
      ``_run_loop``, the per-move ``stack_axis`` in ``build_adapt_step``)
      end up with a length-G axis where each row is identical post-
      ``lax.psum``.  Correct; G× redundant memory on small counters.
      Acceptable cost.

    ``reduce_for_termination`` collapses the (G,) axis to a scalar —
    every shard sees the same global value after the in-step
    collectives, so taking ``[0]`` is exact.

    ``split_keys`` returns a plain, unbroadcast ``(n_sub_keys,)`` array
    rather than a redundant ``(G,)`` copy — :meth:`wrap_for_batch`'s
    ``replicated=`` argument tells ``shard_map`` to hand it to every
    device as-is (``in_specs=P()``), so no pre-broadcast/``device_put``
    is needed. ``reduce_emax`` keeps its ``(G,)`` broadcast (see its own
    docstring for why — its output sometimes substitutes for a value
    that's genuinely ``(G,)``-shaped elsewhere, unlike ``split_keys``'s
    output). ``distinct_keys`` is different again — it returns
    *genuinely independent* per-shard keys, a real ``(G,)``-sharded
    array, not a coherent broadcast, so it keeps its ``device_put``-onto-
    the-mesh placement. See ``experiments/shard_map_rewrite.md`` (item 4)
    for the full account, including what was tried and reverted.

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

    @cached_property
    def mesh(self):
        """The ``"shard"``-axis :class:`jax.sharding.Mesh` for this topology.

        Built once per instance and shared by every consumer, same as
        :attr:`PmapVmapRuns.mesh`.
        """
        return build_mesh("shard", self.n_gpu)

    # ------------------------------------------------------------------
    # Small shape_prefix-driven helpers — same logic as _UniformBatcher's,
    # duplicated (not inherited) since this class isn't part of that family.
    # ------------------------------------------------------------------

    @property
    def walker_axis(self) -> int:
        """Axis index of the per-walker (K) dimension — always ``1`` here."""
        return len(self.shape_prefix)

    def extract_step_sizes(self, pop) -> jnp.ndarray:
        """Per-replica step sizes from ``pop.step_sizes (G, K, n_moves)``.

        Returns shape ``(G, n_moves)`` — walker axis dropped.
        """
        return jnp.take(pop.step_sizes, 0, axis=self.walker_axis)

    def broadcast_step_sizes(
        self,
        per_move_ss: jnp.ndarray,
        n_walkers: int,
    ) -> jnp.ndarray:
        """Inverse of :meth:`extract_step_sizes` — re-insert the walker axis."""
        n_moves = per_move_ss.shape[-1]
        target_shape = self.shape_prefix + (n_walkers, n_moves)
        return jnp.broadcast_to(
            jnp.expand_dims(per_move_ss, axis=self.walker_axis),
            target_shape,
        )

    def scalar_key(self, rng_key: jax.Array) -> jax.Array:
        """Reduce a per-shard key array to a single scalar key.

        Every shard carries an identical broadcast key by construction, so
        taking entry 0 is exact — same as ``_UniformBatcher.scalar_key``.
        """
        arr = jnp.asarray(rng_key)
        return arr if arr.ndim == 0 else arr.reshape(-1)[0]

    # ------------------------------------------------------------------
    # Regime-specific behavior
    # ------------------------------------------------------------------

    def wrap_step(
        self,
        ns_step_fn,
        step_fn,
        n_mcmc_steps: int,
        n_extra: int,
    ):
        """Return a shard_map-compiled NS step callable (pmap-equivalent).

        ``ns_step_fn`` here should be ``ns_step_sharded`` (not the
        plain ``ns_step``) — it uses ``lax.all_gather`` / ``lax.psum``
        collectives that only work inside a mapped context with
        matching ``axis_name="shard"`` (``jax.shard_map`` via
        :func:`jaxrens.sampling.mesh.pmap_like`, same as ``jax.pmap``
        provided).

        The returned callable has signature::

            step(ns_state)  ->  (ns_state, info)

        with leading shape ``(G, ...)`` on every leaf. Unlike the
        ``jax.pmap`` this replaces, the result composes with ``jax.jit``.
        """

        def per_shard(ns_state):
            return ns_step_fn(ns_state, step_fn, n_mcmc_steps, n_extra)

        return jax.jit(pmap_like(per_shard, "shard", self.mesh))

    def split_keys(self, rng_key: jax.Array, n_sub_keys: int) -> jax.Array:
        """Split a key into ``n_sub_keys`` coherent (shard-independent) sub-keys.

        ``rng_key`` may be either:

        * A scalar key.
        * A ``(G,)`` array of identical broadcast keys (legacy callers) —
          collapsed via ``[0]`` before splitting.

        The load-bearing invariant for ``ns_step_sharded`` and
        ``adjust_step_size_sharded`` is that every shard makes the
        *same* RNG decisions — this returns one plain, unbroadcast
        ``(n_sub_keys,)`` array rather than a redundant ``(G,)``-broadcast
        copy of it, since ``wrap_for_batch``'s ``replicated=`` argument
        marks it as a replicated (``in_specs=P()``) input instead: no
        pre-broadcast/``device_put`` needed, ``shard_map`` hands the same
        value to every device directly. See ``experiments/shard_map_rewrite.md``
        ("Per-argument in_specs instead of force-broadcasting scalars").

        Returns shape ``(n_sub_keys,)`` with typed-key dtype.
        """
        scalar_key = rng_key[0] if rng_key.ndim > 0 else rng_key
        return jax.random.split(scalar_key, n_sub_keys)

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

    def reduce_emax(self, energy: Float[Array, "G K"]) -> Float[Array, "G"]:
        """Global ``max`` over the entire (G, K_per_gpu) population, broadcast to (G,).

        Returns shape ``(G,)`` (every entry identical) on the
        ``'shard'``-named mesh — same convention as
        :meth:`VmapRuns.reduce_emax` returning ``(R,)`` and
        :meth:`PmapVmapRuns.reduce_emax` returning ``(G, P)``.

        Unlike :meth:`split_keys`, this one keeps its ``(G,)`` broadcast
        rather than returning a plain scalar for ``wrap_for_batch``'s
        ``replicated=`` — burn-in's ``initial_walk`` feeds this value into
        the *same* ``adapt_step`` machinery that the regular NS loop feeds
        with ``ns_state.emax`` (a real, structurally ``(G,)``-shaped
        field). Simplifying this to a scalar was tried and reverted: it
        broke burn-in's adaptation call, since the shared
        ``_build_sharded_per_move`` closure can't use a different
        ``replicated`` mask depending on which caller supplied ``emax``.
        See ``experiments/shard_map_rewrite.md`` (item 4) for the finding.
        """
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        scalar = jnp.max(energy)
        broadcast = jnp.broadcast_to(scalar[None], (self.n_gpu,))
        shard_mesh = NamedSharding(
            Mesh(jax.local_devices()[: self.n_gpu], ("shard",)),
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
        return jnp.broadcast_to(
            arr_flat[0:1], (self.n_gpu,) + arr_flat.shape[1:]
        )

    def wrap_for_batch(
        self,
        per_element_fn,
        *,
        check_vma: bool = False,
        replicated: tuple[bool, ...] = (),
    ):
        """Wrap a per-replica callable in shard_map over the shard axis.

        Default signature contract (``replicated=()``): every input has a
        leading ``(G,)`` axis on the ``'shard'``-named mesh; outputs
        preserve that axis. This matches :meth:`VmapRuns.wrap_for_batch`
        and :meth:`PmapVmapRuns.wrap_for_batch`'s contracts (each "row" of
        the leading axis is fed to ``per_element_fn``).

        ``replicated``: one bool per positional argument of
        *per_element_fn*, in order; ``True`` marks that argument as an
        already-replicated plain value (no ``(G,)`` axis at all — e.g.
        :meth:`reduce_emax` / :meth:`split_keys`'s outputs, post-
        simplification) rather than a genuinely ``(G, ...)``-sharded one.
        Those arguments get ``in_specs=P()`` (handed to every device
        as-is, no squeeze/broadcast) instead of ``P("shard")``. Leaving
        it empty preserves the old uniform-``P("shard")``-for-everything
        contract exactly. Outputs are always ``P("shard")`` — this only
        changes how *inputs* are described, not outputs (see
        ``experiments/shard_map_rewrite.md``, item 4, for why the
        output-replication case needs a different, costlier approach).

        Not built on :func:`jaxrens.sampling.mesh.pmap_like` when
        ``replicated`` is non-empty — that helper's contract is one
        uniform spec for every argument, which can't express this.
        """
        if not replicated:
            return jax.jit(
                pmap_like(
                    per_element_fn, "shard", self.mesh, check_vma=check_vma
                )
            )

        def wrapped(*args):
            squeezed = tuple(
                a
                if is_rep
                else jax.tree.map(lambda x: jnp.squeeze(x, axis=0), a)
                for a, is_rep in zip(args, replicated)
            )
            out = per_element_fn(*squeezed)
            return jax.tree.map(lambda x: x[None, ...], out)

        in_specs = tuple(
            PartitionSpec() if is_rep else PartitionSpec("shard")
            for is_rep in replicated
        )
        return jax.jit(
            jax.shard_map(
                wrapped,
                mesh=self.mesh,
                in_specs=in_specs,
                out_specs=PartitionSpec("shard"),
                check_vma=check_vma,
            )
        )

    def distinct_keys(self, rng_key: jax.Array) -> jax.Array:
        """Split a scalar key into G INDEPENDENT keys on the shard mesh.

        Overrides the ``_UniformBatcher`` default (``split + reshape``,
        duplicated here since this class doesn't inherit from it) to also
        ``device_put`` the result onto the ``'shard'``-named mesh so
        the per-shard pmap accepts it alongside the sharded NSState.

        Returns shape ``(G,)`` typed-key array sharded along axis 0.
        """
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        keys = jax.random.split(rng_key, self.n_gpu)
        shard_mesh = NamedSharding(
            Mesh(jax.local_devices()[: self.n_gpu], ("shard",)),
            PartitionSpec("shard"),
        )
        return jax.device_put(keys, shard_mesh)


# ---------------------------------------------------------------------------
# "Any of the four" type, and the module-level factory
# ---------------------------------------------------------------------------

# Not a base class — SingleRun / VmapRuns / PmapVmapRuns / ShardedSingleRun
# are four independent concrete types (three sharing _UniformBatcher because
# their behavior genuinely overlaps; ShardedSingleRun standalone because it
# doesn't). This alias is what code elsewhere means by "any batcher" — it
# supports isinstance() checks against it directly (`isinstance(x, A | B)`
# is valid since Python 3.10) exactly like the old ABC did.
BatchDescriptor = SingleRun | VmapRuns | PmapVmapRuns | ShardedSingleRun


def from_shape_prefix(shape_prefix: tuple[int, ...]) -> BatchDescriptor:
    """Return the batcher whose ``shape_prefix`` matches *shape_prefix*.

    * ``()``      → :class:`SingleRun`.
    * ``(R,)``    → :class:`VmapRuns(n_runs=R)`.
    * ``(G, P)``  → :class:`PmapVmapRuns(n_gpu=G, n_per_gpu=P)`.

    Used by ``init/restart.py``, ``io/checkpoint.py``, and ``cli/monitor.py``
    to recover the batcher from a stored array's leading shape (``log_evidence``
    is the canonical witness — its shape is always exactly the prefix).

    Note: ``ShardedSingleRun`` checkpoints are stored in the SingleRun
    ``(1, ...)``/scalar on-disk convention (via its own ``flatten``), so
    they never round-trip through the rank-1 branch here — restoring them
    is handled separately by ``run.shard_n_gpu``, not by shape-sniffing.

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
