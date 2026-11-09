"""InterREManager: orchestrates inter-replica-exchange swap passes.

One swap pass fires after each ``ns_step`` call in ``_run_loop`` when configured.
The manager is descriptor-aware:

- ``SingleRun``: ``apply`` is a no-op (returns state unchanged with zero stats).
- ``VmapRuns``: operates on ``(n_runs, K, ...)`` directly via
  ``replica_exchange_step``.
- ``PmapVmapRuns``: uses ``lax.all_gather(axis_name="gpu")`` to replicate the
  population across devices, swaps on each device's identical view (same RNG =
  same swap decisions), then re-shards by slicing the device's own offset.
  For ``n_gpu=1`` the all_gather is a no-op (zero extra cost).

Design: built once at construction time, JIT'd swap step cached; ``_run_loop``
calls ``fires(i)`` / ``apply(state, key)`` at each iteration.

The manager is flavor-agnostic.  It hands the population to
:func:`~jaxrens.sampling.moves.replica_exchange.replica_exchange_step` as a
:class:`~jaxrens.sampling.moves.replica_exchange.SwapWalker` pytree, plus the
per-run ensemble params the kernel declares in ``param_keys``.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jaxtyping import Array, Key

logger = logging.getLogger(__name__)

from jaxrens.sampling.batch_descriptor import BatchDescriptor, PmapVmapRuns
from jaxrens.sampling.moves.replica_exchange import (
    SwapKernel,
    SwapWalker,
    replica_exchange_step,
)
from jaxrens.state.ns import NSState


class SwapStats(TypedDict):
    """Public return type of ``InterREManager.apply``'s stats dict.

    The aggregate fields (``n_swap_pairs_attempted``,
    ``n_swap_pairs_accepted``, ``acceptance_rate``) are sums over all
    pairs and all swap cycles in the fire.  The per-pair arrays carry
    the same totals broken down by pair_id (``min(left, right)`` of
    the (k, k+1) pair).  Always shape ``(n_runs - 1,)`` int32; when
    ``n_runs < 2`` the arrays are length-zero.
    """

    n_swap_pairs_attempted: int
    n_swap_pairs_accepted: int
    acceptance_rate: float
    n_energy_evals: int
    n_grad_evals: int
    n_accepted_per_pair: np.ndarray
    n_attempted_per_pair: np.ndarray


_EMPTY_STATS: SwapStats = {
    "n_swap_pairs_attempted": 0,
    "n_swap_pairs_accepted": 0,
    "acceptance_rate": 0.0,
    "n_energy_evals": 0,
    "n_grad_evals": 0,
    "n_accepted_per_pair": np.zeros(0, dtype=np.int32),
    "n_attempted_per_pair": np.zeros(0, dtype=np.int32),
}

# Per-run ensemble params a swap kernel may ask for: trailing rank of one
# run's value and the dtype it is normalised to (None keeps the stored one).
_RUN_PARAM_SPECS: dict[str, tuple[int, Any]] = {
    "pressure": (0, None),
    "chemical_potentials": (1, jnp.float32),
    "target_composition": (1, jnp.int32),
}


def _certify_replicated(tree, axis_name: str, n_devices: int):
    """psum-then-divide every leaf of *tree* across *axis_name*.

    All devices already hold numerically identical values here (same RNG +
    same all-gathered data => same swap decision on every device), but
    ``shard_map``'s VMA type checker doesn't take that on faith — it only
    trusts a handful of collectives (``psum`` among them) as *proof* of
    replication, which ``out_specs=PartitionSpec()`` requires. Summing G
    identical integer copies and dividing by G recovers the exact original
    value while producing a provably-reduced type. See
    ``experiments/shard_map_rewrite.md`` ("out_specs=P() ... with a
    caveat") for the discovery that motivated this.
    """
    return jax.tree.map(
        lambda x: jax.lax.psum(x, axis_name=axis_name) // n_devices, tree
    )


def _wrap_swap_body(fn, axis_name: str, mesh):
    """``pmap_like``-equivalent for the ``"gpu"``-axis swap body, except the
    swap-stats dict (2nd return value) is replicated (``out_specs=P()``)
    rather than concatenated across devices.

    *fn* takes ``(rng_key, walkers, emax, run_params)`` with the mapped axis
    stripped and returns ``(walkers, swap_info)``; it must have run
    ``swap_info`` through :func:`_certify_replicated` before returning it —
    this wrapper passes it straight through shard_map's replicated path.

    ``check_vma=False``: the swap driver this wraps has its own internal
    ``lax.scan``/``lax.cond`` control flow (including ``morph.py`` under
    XRENS) that has not been audited for VMA-correctness — turning this on
    surfaced a real, unrelated ``lax.cond`` branch-type mismatch inside
    ``morph.py:pick_from_donor_species`` on real 2-GPU hardware. Auditing
    and fixing that is a separate, larger undertaking than this wrapper's
    swap-info replication; ``_certify_replicated`` still makes the
    ``out_specs=P()`` promise numerically correct (every device provably
    computed the identical swap), it just isn't statically checked here.
    """

    def wrapped(*args):
        args = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), args)
        shard_walkers, swap_info = fn(*args)
        return jax.tree.map(lambda x: x[None, ...], shard_walkers), swap_info

    spec = PartitionSpec(axis_name)
    return jax.shard_map(
        wrapped,
        mesh=mesh,
        in_specs=spec,
        out_specs=(spec, PartitionSpec()),
        check_vma=False,
    )


def _flatten_gp(x):
    """``(G, P, *trailing) -> (G*P, *trailing)``."""
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


class InterREManager:
    """Manages inter-replica-exchange swap passes in the NS outer loop.

    Constructed once before the loop starts.  ``_run_loop`` calls
    ``fires(i)`` to decide whether to fire on iteration ``i``, then calls
    ``apply(ns_state, rng_key)`` to run the swap pass.

    Args:
        swap_kernel: :class:`SwapKernel` instance, typically from
            :func:`~jaxrens.sampling.moves.replica_exchange.make_swap_kernel`.
        batcher: ``BatchDescriptor`` controlling the execution mode.
        backend: Energy backend, passed to kernels that re-evaluate
            energies (``XRENSSwap``).
        re_interval: Fire a swap pass every this many NS iterations.  0 → never fire.
        n_swap_cycles: Number of even+odd swap phases per fire.
    """

    def __init__(
        self,
        swap_kernel: SwapKernel,
        batcher: BatchDescriptor,
        backend: Any,
        re_interval: int = 1,
        n_swap_cycles: int = 1,
    ) -> None:
        unknown = set(swap_kernel.param_keys) - set(_RUN_PARAM_SPECS)
        if unknown:
            raise ValueError(
                f"{type(swap_kernel).__name__} asks for per-run params "
                f"{sorted(unknown)} that InterREManager does not know how to "
                f"extract; known: {sorted(_RUN_PARAM_SPECS)}."
            )
        self._swap_kernel = swap_kernel
        self._batcher = batcher
        self._backend = backend
        self._re_interval = re_interval
        self._n_swap_cycles = n_swap_cycles

        # Build and cache the JIT-compiled swap step.
        self._jit_swap = None
        if batcher.is_batched:
            self._jit_swap = self._build_jit_fn()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fires(self, iteration: int) -> bool:
        """Return True iff a swap pass should fire on this iteration.

        Rules: re_interval > 0 AND iteration > 0 AND iteration % re_interval == 0.
        Iteration 0 is skipped to match the adapt-step firing convention.

        Args:
            iteration: Current NS iteration index (Python int).

        Returns:
            True when a swap should be attempted.
        """
        return (
            self._re_interval > 0
            and iteration > 0
            and iteration % self._re_interval == 0
        )

    @property
    def is_active(self) -> bool:
        """True iff this manager will actually do work.

        ``SingleRun`` descriptors return False (no batched population to swap).
        ``VmapRuns`` / ``PmapVmapRuns`` return True when ``re_interval > 0``.
        """
        return self._batcher.is_batched and self._re_interval > 0

    def apply(
        self, ns_state: NSState, rng_key: Key[Array, ""]
    ) -> tuple[NSState, SwapStats, Key[Array, ""]]:
        """Run one inter-RE swap pass.

        For ``SingleRun``: returns state unchanged with zero stats.
        For ``VmapRuns``: operates on ``(n_runs, K, ...)`` population directly.
        For ``PmapVmapRuns``: all_gathers across the pmap axis, swaps, re-shards.

        The swap-acceptance constraint is read off ``ns_state.emax`` — the
        per-replica NS contour the most recent ``ns_step`` culled at.
        This is algorithm state set by ``ns_step``, not a value re-derived
        from ``pop.energy``: recomputing here would be strictly tighter
        post-MCMC and would reject otherwise-legal swaps.

        Args:
            ns_state: Current ``NSState`` (single, vmapped, or pmap-vmapped).
            rng_key: Scalar PRNG key for swap randomness.

        Returns:
            Tuple ``(new_ns_state, swap_stats, new_rng_key)`` where:

            * ``new_ns_state``: Updated ``NSState`` after swaps.
            * ``swap_stats``: Dict with keys ``{"n_swap_pairs_attempted": int, "n_swap_pairs_accepted": int, "acceptance_rate": float, "n_energy_evals": int, "n_grad_evals": int}``.
              All zeros for ``SingleRun`` (no-op).
            * ``new_rng_key``: Advanced PRNG key carry (scalar).
        """
        if not self._batcher.is_batched:
            # SingleRun: no-op
            new_key = jax.random.split(rng_key)[0]
            return ns_state, dict(_EMPTY_STATS), new_key

        rng_key, swap_key = jax.random.split(rng_key)
        walkers, emax, run_params = self._extract_swap_inputs(ns_state)

        if isinstance(self._batcher, PmapVmapRuns):
            # Same key on every device => identical swap decisions everywhere.
            swap_key = jnp.broadcast_to(
                swap_key[None], (self._batcher.n_gpu,) + swap_key.shape
            )
        new_walkers, swap_info = self._jit_swap(
            swap_key, walkers, emax, run_params
        )

        new_pop = ns_state.population.set(
            positions=new_walkers.positions,
            types=new_walkers.types,
            energy=new_walkers.energy,
            cell=new_walkers.cell,
        )
        new_ns_state = ns_state.set(population=new_pop)
        return new_ns_state, self._build_stats(swap_info), rng_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_jit_fn(self):
        """Build the JIT-compiled swap function for this batcher.

        Signature ``fn(rng_key, walkers, emax, run_params) -> (walkers,
        swap_info)``.  For ``VmapRuns`` the inputs are ``(n_runs, K, ...)``;
        for ``PmapVmapRuns`` they are ``(G, P, K, ...)`` (and ``rng_key`` is
        ``(G,)``), handled by a ``"gpu"``-axis ``shard_map`` (pmap-equivalent,
        see :mod:`jaxrens.sampling.mesh`) around an all_gather swap body.
        """
        kernel = self._swap_kernel
        backend = self._backend
        n_swap_cycles = self._n_swap_cycles

        def _swap(rng_key, walkers, emax, run_params):
            logger.info(
                "inter_re tracing: kernel=%s  pop_shape=%s  n_swap_cycles=%d",
                type(kernel).__name__,
                walkers.positions.shape,
                int(n_swap_cycles),
            )
            return replica_exchange_step(
                rng_key,
                walkers,
                emax,
                run_params,
                kernel,
                backend=backend,
                n_swap_cycles=n_swap_cycles,
            )

        if not isinstance(self._batcher, PmapVmapRuns):
            return jax.jit(_swap)

        def _gpu_body(rng_key, walkers, emax, run_params):
            """all_gather everything across ``"gpu"`` so each device sees the
            full ``(G, P, ...)`` population, swap on the flattened
            ``(G*P, ...)`` view (same RNG on every device => same decisions),
            then slice this device's own ``(P, ...)`` shard back out."""
            full = jax.tree.map(
                lambda x: jax.lax.all_gather(x, axis_name="gpu", axis=0),
                (walkers, emax, run_params),
            )
            full_walkers = full[0]
            flat_walkers, flat_emax, flat_params = jax.tree.map(
                _flatten_gp, full
            )
            new_flat, swap_info = _swap(
                rng_key, flat_walkers, flat_emax, flat_params
            )
            dev_idx = jax.lax.axis_index("gpu")
            shard = jax.tree.map(
                lambda new, ref: new.reshape(ref.shape)[dev_idx],
                new_flat,
                full_walkers,
            )
            n_gpu = jax.tree.leaves(full_walkers)[0].shape[0]
            return shard, _certify_replicated(swap_info, "gpu", n_gpu)

        # self._batcher.mesh: the same cached Mesh object wrap_step /
        # wrap_for_batch use for this batcher, not a fresh equivalent one.
        return jax.jit(_wrap_swap_body(_gpu_body, "gpu", self._batcher.mesh))

    def _extract_swap_inputs(
        self, ns_state: NSState
    ) -> tuple[SwapWalker, Array, dict[str, Array]]:
        """Pull the swap population, Emax and per-run params off the state.

        For VmapRuns the population has shape ``(n_runs, K, ...)``.
        For PmapVmapRuns the population has shape ``(G, P, K, ...)``.

        ``emax`` is read off ``ns_state.emax`` — the per-replica NS
        contour set by the most recent ``ns_step``.  Not re-derived from
        ``pop.energy`` (which would be strictly tighter post-MCMC).

        Per-run params are the kernel's ``param_keys`` found in
        ``population.ensemble_params``.  After vmapping init_ns those carry
        a walker axis (all walkers of a run hold the same value), which is
        dropped here; a scalar value is broadcast to every run.

        Raises:
            ValueError: If a key in the kernel's ``required_params`` is
                missing.
        """
        pop = ns_state.population
        walkers = SwapWalker(
            positions=pop.positions,
            types=pop.types,
            energy=pop.energy,
            cell=pop.cell,
        )

        shape_prefix = tuple(self._batcher.shape_prefix)
        walker_axis = self._batcher.walker_axis
        ep = getattr(pop, "ensemble_params", None)
        ep = ep if isinstance(ep, dict) else {}

        run_params: dict[str, Array] = {}
        for key in self._swap_kernel.param_keys:
            if key not in ep:
                continue
            rank, dtype = _RUN_PARAM_SPECS[key]
            arr = jnp.asarray(ep[key], dtype=dtype)
            if arr.ndim == rank:
                arr = jnp.broadcast_to(arr, shape_prefix + arr.shape)
            elif arr.ndim == len(shape_prefix) + rank + 1:
                arr = jnp.take(arr, 0, axis=walker_axis)
            run_params[key] = arr

        missing = [
            k for k in self._swap_kernel.required_params if k not in run_params
        ]
        if missing:
            raise ValueError(
                f"{type(self._swap_kernel).__name__} requires {missing} in "
                f"ensemble_params for every run. Ensure they were injected at "
                f"init time via ensemble_params_per_run."
            )
        return walkers, ns_state.emax, run_params

    @staticmethod
    def _build_stats(swap_info: dict[str, Any]) -> SwapStats:
        """Convert ``replica_exchange_step`` swap_info to the public stats dict.

        Args:
            swap_info: Dict with keys ``"n_accepted"``, ``"n_attempted"``,
                ``"n_energy_evals"`` and the per-pair arrays.

        Returns:
            Dict with keys matching the public API:
            ``n_swap_pairs_attempted``, ``n_swap_pairs_accepted``,
            ``acceptance_rate``, ``n_energy_evals``, ``n_grad_evals``.
        """
        n_att = int(jnp.asarray(swap_info["n_attempted"]))
        n_acc = int(jnp.asarray(swap_info["n_accepted"]))
        return {
            "n_swap_pairs_attempted": n_att,
            "n_swap_pairs_accepted": n_acc,
            "acceptance_rate": n_acc / max(n_att, 1),
            "n_energy_evals": int(jnp.asarray(swap_info["n_energy_evals"])),
            "n_grad_evals": 0,
            "n_accepted_per_pair": np.asarray(
                swap_info["n_accepted_per_pair"], dtype=np.int32
            ),
            "n_attempted_per_pair": np.asarray(
                swap_info["n_attempted_per_pair"], dtype=np.int32
            ),
        }
