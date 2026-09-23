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

Three swap-kernel flavors (pressure / XRENS / semi-grand) share the exact
same shape/collective plumbing in both the vmap and shard_map builders below
— they differ only in which kernel function gets called and whether there's
an extra per-replica ensemble argument (``composition_targets`` /
``chemical_potentials`` / none). That's factored into a small ``kernel_call``
closure per flavor (``_pressure_kernel_call`` / ``_xrens_kernel_call`` /
``_semi_grand_kernel_call``); ``_build_vmap_swap_fn`` and ``_build_pmap_body``
each take one such closure and stay flavor-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jaxtyping import Array, Key

logger = logging.getLogger(__name__)

from jaxrens.sampling.batch_descriptor import (
    BatchDescriptor,
    PmapVmapRuns,
    SingleRun,
    VmapRuns,
)
from jaxrens.sampling.moves.replica_exchange import (
    PressureRENSSwap,
    SemiGrandSwap,
    SwapKernel,
    XRENSSwap,
    replica_exchange_step,
    semi_grand_replica_exchange_step,
    xrens_replica_exchange_step,
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
    5th return value (the swap-stats dict) is replicated (``out_specs=P()``)
    rather than concatenated across devices.

    *fn* must be written like :func:`jaxrens.sampling.mesh.pmap_like`'s
    contract for its first 4 (sharded) return values, and must have already
    run its swap-stats dict through :func:`_certify_replicated` before
    returning it — this wrapper does not unsqueeze/re-shape that 5th value
    at all, it passes it straight through shard_map's replicated path.

    ``check_vma=False``: the swap kernels this wraps (``replica_exchange_step``
    / ``xrens_replica_exchange_step`` / ``semi_grand_replica_exchange_step``,
    and ``morph.py`` underneath the XRENS/semi-grand composition swap) have
    their own internal ``lax.scan``/``lax.cond`` control flow that has not
    been audited for VMA-correctness — turning this on surfaced a real,
    unrelated ``lax.cond`` branch-type mismatch inside
    ``morph.py:pick_from_donor_species`` on real 2-GPU hardware. Auditing
    and fixing that is a separate, larger undertaking than this wrapper's
    swap-info replication; ``_certify_replicated`` still makes the
    ``out_specs=P()`` promise numerically correct (every device provably
    computed the identical swap), it just isn't statically checked here.
    """

    def wrapped(*args):
        args = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), args)
        shard_pos, shard_typ, shard_ene, shard_bxs, swap_info = fn(*args)
        sharded_outs = jax.tree.map(
            lambda x: x[None, ...],
            (shard_pos, shard_typ, shard_ene, shard_bxs),
        )
        return (*sharded_outs, swap_info)

    spec = PartitionSpec(axis_name)
    out_specs = (spec, spec, spec, spec, PartitionSpec())
    return jax.shard_map(
        wrapped, mesh=mesh, in_specs=spec, out_specs=out_specs, check_vma=False
    )


# ---------------------------------------------------------------------------
# Per-flavor kernel_call closures: map a uniform positional calling
# convention (rng_key, positions, types, energies, cells, emax, pressures[,
# extra]) onto each swap kernel's actual (differently-named) kwargs. This is
# the *only* flavor-specific piece; the shape/collective plumbing around it
# (_build_vmap_swap_fn, _build_pmap_body below) is shared.
# ---------------------------------------------------------------------------


def _pressure_kernel_call(kernel: SwapKernel, n_swap_cycles: int) -> Callable:
    def call(rng_key, positions, types, energies, cells, emax, pressures):
        return replica_exchange_step(
            rng_key=rng_key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=cells,
            all_emax=emax,
            pressures=pressures,
            n_swap_cycles=n_swap_cycles,
            swap_kernel=kernel,
        )

    return call


def _xrens_kernel_call(
    kernel: SwapKernel, backend: Any, n_swap_cycles: int
) -> Callable:
    def call(
        rng_key,
        positions,
        types,
        energies,
        cells,
        emax,
        pressures,
        composition_targets,
    ):
        return xrens_replica_exchange_step(
            rng_key=rng_key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=cells,
            all_emax=emax,
            composition_targets=composition_targets,
            backend=backend,
            xrens_kernel=kernel,
            pressures=pressures,
            n_swap_cycles=n_swap_cycles,
        )

    return call


def _semi_grand_kernel_call(
    kernel: SwapKernel, n_swap_cycles: int
) -> Callable:
    def call(
        rng_key,
        positions,
        types,
        energies,
        cells,
        emax,
        pressures,
        chemical_potentials,
    ):
        return semi_grand_replica_exchange_step(
            rng_key=rng_key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=cells,
            all_emax=emax,
            chemical_potentials=chemical_potentials,
            semi_grand_kernel=kernel,
            pressures=pressures,
            n_swap_cycles=n_swap_cycles,
        )

    return call


def _build_vmap_swap_fn(
    kernel_call: Callable,
    flavor_name: str,
    has_extra: bool,
    n_swap_cycles: int,
) -> Callable:
    """JIT-compile *kernel_call* with the ``(n_runs, K, ...)`` calling
    convention, logging the trace-time flavor/shape once per JIT cache miss
    (the gap to the next iteration log is the compile + first-execution
    duration).
    """

    if has_extra:

        def _swap_fn(
            rng_key, positions, types, energies, cells, emax, pressures, extra
        ):
            logger.info(
                "inter_re tracing: flavor=%s  pop_shape=%s  n_swap_cycles=%d",
                flavor_name,
                positions.shape,
                int(n_swap_cycles),
            )
            return kernel_call(
                rng_key,
                positions,
                types,
                energies,
                cells,
                emax,
                pressures,
                extra,
            )

    else:

        def _swap_fn(
            rng_key, positions, types, energies, cells, emax, pressures
        ):
            logger.info(
                "inter_re tracing: flavor=%s  pop_shape=%s  n_swap_cycles=%d",
                flavor_name,
                positions.shape,
                int(n_swap_cycles),
            )
            return kernel_call(
                rng_key, positions, types, energies, cells, emax, pressures
            )

    return jax.jit(_swap_fn)


def _all_gather_maybe(x, axis_name: str):
    """``lax.all_gather`` (default ``tiled=False``: prepends a new size-G
    axis) unless *x* is ``None``."""
    if x is None:
        return None
    return jax.lax.all_gather(x, axis_name=axis_name, axis=0)


def _flatten_gp(x):
    """``(G, P, *trailing) -> (G*P, *trailing)``; passes ``None`` through.

    Works uniformly for both walker-indexed arrays (positions/types/
    energies/cells, real trailing dims) and pure per-replica scalars
    (emax/pressures, empty trailing dims) since ``x.shape[2:]`` is ``()``
    for the latter.
    """
    if x is None:
        return None
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def _build_pmap_body(kernel_call: Callable, has_extra: bool) -> Callable:
    """Build the ``"gpu"``-axis swap body shared by all three flavors.

    ``all_gather`` each input across the ``"gpu"`` axis (each device then
    sees the full ``(G, P, ...)`` population), flatten to ``(G*P, ...)``,
    run *kernel_call* on the full population (same RNG on every device =>
    same swap decisions everywhere), reshape back and slice out this
    device's own ``(P, ...)`` shard, and certify the swap-stats dict as
    replicated (see :func:`_certify_replicated`) so the caller's
    ``out_specs=P()`` is valid.

    Written for :func:`jaxrens.sampling.mesh.pmap_like`'s calling
    convention (mapped axis already stripped from every input) — actually
    wrapped via :func:`_wrap_swap_body`, not ``pmap_like`` itself, since
    the swap-stats output needs the replicated (not concatenated) out_spec
    that ``pmap_like``'s uniform-spec contract can't express.
    """

    def _run(rng_key_per_device, pos, typ, ene, bxs, em, pres, extra):
        full_pos = _all_gather_maybe(pos, "gpu")
        full_typ = _all_gather_maybe(typ, "gpu")
        full_ene = _all_gather_maybe(ene, "gpu")
        full_bxs = _all_gather_maybe(bxs, "gpu")
        full_em = _all_gather_maybe(em, "gpu")
        full_pres = _all_gather_maybe(pres, "gpu")
        full_extra = _all_gather_maybe(extra, "gpu")
        G = full_pos.shape[0]

        call_args = (
            rng_key_per_device,
            _flatten_gp(full_pos),
            _flatten_gp(full_typ),
            _flatten_gp(full_ene),
            _flatten_gp(full_bxs),
            _flatten_gp(full_em),
            _flatten_gp(full_pres),
        )
        if has_extra:
            call_args = call_args + (_flatten_gp(full_extra),)

        (
            new_pos_flat,
            new_typ_flat,
            new_ene_flat,
            new_bxs_flat,
            swap_info,
        ) = kernel_call(*call_args)

        new_pos_full = new_pos_flat.reshape(full_pos.shape)
        new_typ_full = new_typ_flat.reshape(full_typ.shape)
        new_ene_full = new_ene_flat.reshape(full_ene.shape)
        new_bxs_full = (
            new_bxs_flat.reshape(full_bxs.shape)
            if new_bxs_flat is not None
            else None
        )

        dev_idx = jax.lax.axis_index("gpu")
        shard_pos = new_pos_full[dev_idx]
        shard_typ = new_typ_full[dev_idx]
        shard_ene = new_ene_full[dev_idx]
        shard_bxs = new_bxs_full[dev_idx] if new_bxs_full is not None else None
        # Certify swap_info as provably-replicated so out_specs=P() (in
        # _wrap_swap_body) is valid.
        swap_info = _certify_replicated(swap_info, "gpu", G)
        return shard_pos, shard_typ, shard_ene, shard_bxs, swap_info

    if has_extra:

        def _pmap_body(
            rng_key_per_device, pos, typ, ene, bxs, em, pres, extra
        ):
            return _run(
                rng_key_per_device, pos, typ, ene, bxs, em, pres, extra
            )

    else:

        def _pmap_body(rng_key_per_device, pos, typ, ene, bxs, em, pres):
            return _run(rng_key_per_device, pos, typ, ene, bxs, em, pres, None)

    return _pmap_body


class InterREManager:
    """Manages inter-replica-exchange swap passes in the NS outer loop.

    Constructed once before the loop starts.  ``_run_loop`` calls
    ``fires(i)`` to decide whether to fire on iteration ``i``, then calls
    ``apply(ns_state, rng_key)`` to run the swap pass.

    Args:
        swap_kernel: :class:`SwapKernel` instance (e.g. ``PressureRENSSwap()``).
        batcher: ``BatchDescriptor`` controlling the execution mode.
        backend: Energy backend (unused for ``PressureRENSSwap`` but part of
            the general API for future kernels such as ``XRENSSwap``).
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
        self._swap_kernel = swap_kernel
        self._batcher = batcher
        self._backend = backend
        self._re_interval = re_interval
        self._n_swap_cycles = n_swap_cycles

        # Kernel flavor flags (mutually exclusive).
        self._is_xrens = isinstance(swap_kernel, XRENSSwap)
        self._is_semi_grand = isinstance(swap_kernel, SemiGrandSwap)

        # Build and cache the JIT-compiled swap step.
        self._jit_vmap_swap = None
        self._jit_pmap_swap = None
        if batcher.is_batched:
            self._jit_vmap_swap, self._jit_pmap_swap = self._build_jit_fns()

        # Resolved once here (fixed for this instance's lifetime) rather
        # than re-checked via isinstance on every apply() call.
        self._apply_impl = (
            self._apply_pmap_vmap
            if isinstance(batcher, PmapVmapRuns)
            else self._apply_vmap
        )

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
        new_ns_state, swap_stats = self._apply_impl(ns_state, swap_key)
        return new_ns_state, swap_stats, rng_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_jit_fns(self):
        """Build JIT-compiled swap functions for VmapRuns and PmapVmapRuns.

        Returns:
            ``(jit_vmap_swap, jit_pmap_swap)`` where each is a compiled
            callable with signature::

                fn(rng_key, positions, types, energies, cells, emax, pressures)
                    -> (new_pos, new_types, new_ene, new_cells, swap_info)

            ``jit_pmap_swap`` operates on ``(G, P, K, ...)`` shaped inputs via
            a ``"gpu"``-axis ``shard_map`` (pmap-equivalent, see
            :mod:`jaxrens.sampling.mesh`) wrapping an all_gather swap body.
            ``jit_vmap_swap`` operates on ``(n_runs, K, ...)`` shaped inputs.

            For XRENS, the signature is extended with ``composition_targets``::

                fn(rng_key, positions, types, energies, cells, emax,
                   pressures, composition_targets)
                    -> (new_pos, new_types, new_ene, new_cells, swap_info)

            For SemiGrand, the signature is extended with ``chemical_potentials``::

                fn(rng_key, positions, types, energies, cells, emax,
                   pressures, chemical_potentials)
                    -> (new_pos, new_types, new_ene, new_cells, swap_info)
        """
        n_swap_cycles = self._n_swap_cycles
        kernel = self._swap_kernel
        backend = self._backend

        if self._is_xrens:
            kernel_call = _xrens_kernel_call(kernel, backend, n_swap_cycles)
            flavor_name = "xrens"
            has_extra = True
        elif self._is_semi_grand:
            kernel_call = _semi_grand_kernel_call(kernel, n_swap_cycles)
            flavor_name = "semi_grand"
            has_extra = True
        else:
            kernel_call = _pressure_kernel_call(kernel, n_swap_cycles)
            flavor_name = "pressure"
            has_extra = False

        jit_vmap = _build_vmap_swap_fn(
            kernel_call, flavor_name, has_extra, n_swap_cycles
        )

        # jit_pmap is only ever invoked for PmapVmapRuns (see `apply`); other
        # batchers don't carry a mesh to build from and don't need this
        # callable, so leave it unbuilt.
        jit_pmap = None
        if isinstance(self._batcher, PmapVmapRuns):
            pmap_body = _build_pmap_body(kernel_call, has_extra)
            # self._batcher.mesh: the same cached Mesh object wrap_step /
            # wrap_for_batch use for this batcher, not a fresh equivalent one.
            jit_pmap = jax.jit(
                _wrap_swap_body(pmap_body, "gpu", self._batcher.mesh)
            )

        return jit_vmap, jit_pmap

    def _extract_swap_inputs(self, ns_state: NSState):
        """Extract swap inputs from state.

        For VmapRuns the population has shape ``(n_runs, K, ...)``.
        For PmapVmapRuns the population has shape ``(G, P, K, ...)``.

        ``emax`` is read off ``ns_state.emax`` — the per-replica NS
        contour set by the most recent ``ns_step``.  Not re-derived from
        ``pop.energy`` (which would be strictly tighter post-MCMC).

        Returns:
            Tuple of JAX arrays:
            ``(positions, types, energies, cells, emax, pressures,
               composition_targets, chemical_potentials)``
            where ``composition_targets`` is ``None`` unless ``XRENSSwap``
            mode is active, and ``chemical_potentials`` is ``None`` unless
            ``SemiGrandSwap`` mode is active.
        """
        pop = ns_state.population
        emax = ns_state.emax  # (*shape_prefix,)

        positions = pop.positions  # (*shape_prefix, K, n_atoms, 3)
        types = pop.types  # varies
        energies = pop.energy  # (*shape_prefix, K)
        cells = pop.cell  # (*shape_prefix, K, 3, 3)

        # Pressures, composition_targets, chemical_potentials: extract from
        # ensemble_params.  After vmapping init_ns, per-replica scalar values
        # carry an extra walker axis (shape ``(*shape_prefix, K)``); per-replica
        # vectors carry it before the trailing per-replica vector axis.  Drop
        # the walker axis when present.
        n_prefix = len(self._batcher.shape_prefix)
        walker_axis = self._batcher.walker_axis

        def _drop_walker_axis_if_present(
            arr: jnp.ndarray, has_vector: bool
        ) -> jnp.ndarray:
            target_ndim = n_prefix + (1 if has_vector else 0)
            if arr.ndim == target_ndim + 1:
                return jnp.take(arr, 0, axis=walker_axis)
            return arr

        pressures = None
        composition_targets = None
        chemical_potentials = None
        ep = getattr(pop, "ensemble_params", None)
        if ep is not None and isinstance(ep, dict):
            if "pressure" in ep:
                arr = jnp.asarray(ep["pressure"])
                if arr.ndim == 0:
                    # Scalar pressure — replicate across all replicas.
                    pressures = jnp.broadcast_to(
                        arr,
                        self._batcher.shape_prefix or (1,),
                    )
                else:
                    pressures = _drop_walker_axis_if_present(
                        arr, has_vector=False
                    )

            if "target_composition" in ep and self._is_xrens:
                tc_arr = jnp.asarray(ep["target_composition"], dtype=jnp.int32)
                composition_targets = _drop_walker_axis_if_present(
                    tc_arr,
                    has_vector=True,
                )

            if "chemical_potentials" in ep and self._is_semi_grand:
                cp_arr = jnp.asarray(
                    ep["chemical_potentials"], dtype=jnp.float32
                )
                chemical_potentials = _drop_walker_axis_if_present(
                    cp_arr,
                    has_vector=True,
                )

        return (
            positions,
            types,
            energies,
            cells,
            emax,
            pressures,
            composition_targets,
            chemical_potentials,
        )

    def _apply_vmap(
        self, ns_state: NSState, swap_key: Key[Array, ""]
    ) -> tuple[NSState, SwapStats]:
        """Apply swap pass for VmapRuns descriptor.

        State population has shape ``(n_runs, K, ...)``.
        """
        (
            positions,
            types,
            energies,
            cells,
            emax,
            pressures,
            composition_targets,
            chemical_potentials,
        ) = self._extract_swap_inputs(ns_state)

        if self._is_xrens:
            if composition_targets is None:
                raise ValueError(
                    "XRENSSwap requires 'target_composition' in ensemble_params "
                    "for every run. Ensure composition_targets were injected at "
                    "init time via ensemble_params_per_run."
                )
            (
                new_pos,
                new_types,
                new_ene,
                new_cells,
                swap_info,
            ) = self._jit_vmap_swap(
                swap_key,
                positions,
                types,
                energies,
                cells,
                emax,
                pressures,
                composition_targets,
            )
        elif self._is_semi_grand:
            if chemical_potentials is None:
                raise ValueError(
                    "SemiGrandSwap requires 'chemical_potentials' in ensemble_params "
                    "for every run. Ensure chemical_potentials were injected at "
                    "init time via ensemble_params_per_run."
                )
            (
                new_pos,
                new_types,
                new_ene,
                new_cells,
                swap_info,
            ) = self._jit_vmap_swap(
                swap_key,
                positions,
                types,
                energies,
                cells,
                emax,
                pressures,
                chemical_potentials,
            )
        else:
            (
                new_pos,
                new_types,
                new_ene,
                new_cells,
                swap_info,
            ) = self._jit_vmap_swap(
                swap_key, positions, types, energies, cells, emax, pressures
            )

        new_pop = ns_state.population.set(
            positions=new_pos,
            types=new_types,
            energy=new_ene,
            cell=new_cells,
        )
        new_ns_state = ns_state.set(population=new_pop)
        stats = self._build_stats(swap_info)
        return new_ns_state, stats

    def _apply_pmap_vmap(
        self, ns_state: NSState, swap_key: Key[Array, ""]
    ) -> tuple[NSState, SwapStats]:
        """Apply swap pass for PmapVmapRuns descriptor.

        Population has shape ``(G, P, K, ...)``.  Uses ``lax.all_gather``
        so every device sees the full ``(G*P, K, ...)`` population.

        For ``n_gpu=1`` the all_gather is a no-op (cost zero).
        The same code runs unconditionally for all n_gpu values so that
        multi-GPU correctness can be tested without a fork.
        """
        (
            positions,
            types,
            energies,
            cells,
            emax,
            pressures,
            composition_targets,
            chemical_potentials,
        ) = self._extract_swap_inputs(ns_state)
        G = self._batcher.n_gpu

        # Broadcast the same rng_key to all devices so every device makes
        # identical swap decisions (deterministic = same output on all devices).
        per_device_key = jnp.broadcast_to(
            swap_key[None], (G,) + swap_key.shape
        )

        if self._is_xrens:
            if composition_targets is None:
                raise ValueError(
                    "XRENSSwap requires 'target_composition' in ensemble_params."
                )
            (
                new_pos,
                new_types,
                new_ene,
                new_cells_out,
                swap_info_sharded,
            ) = self._jit_pmap_swap(
                per_device_key,
                positions,
                types,
                energies,
                cells,
                emax,
                pressures,
                composition_targets,
            )
        elif self._is_semi_grand:
            if chemical_potentials is None:
                raise ValueError(
                    "SemiGrandSwap requires 'chemical_potentials' in ensemble_params."
                )
            (
                new_pos,
                new_types,
                new_ene,
                new_cells_out,
                swap_info_sharded,
            ) = self._jit_pmap_swap(
                per_device_key,
                positions,
                types,
                energies,
                cells,
                emax,
                pressures,
                chemical_potentials,
            )
        else:
            (
                new_pos,
                new_types,
                new_ene,
                new_cells_out,
                swap_info_sharded,
            ) = self._jit_pmap_swap(
                per_device_key,
                positions,
                types,
                energies,
                cells,
                emax,
                pressures,
            )

        new_pop = ns_state.population.set(
            positions=new_pos,
            types=new_types,
            energy=new_ene,
            cell=new_cells_out,
        )
        new_ns_state = ns_state.set(population=new_pop)

        # swap_info_sharded arrives already replicated (out_specs=P() in
        # _wrap_swap_body, certified via _certify_replicated) -- no more
        # per-device axis to index away, unlike the old pmap path.
        stats = self._build_stats(swap_info_sharded)
        return new_ns_state, stats

    @staticmethod
    def _build_stats(swap_info: dict[str, Any]) -> SwapStats:
        """Convert ``replica_exchange_step`` swap_info to the public stats dict.

        Args:
            swap_info: Dict with keys ``"n_accepted"``, ``"n_attempted"``,
                and optionally ``"n_energy_evals"`` (int32 scalars).

        Returns:
            Dict with keys matching the public API:
            ``n_swap_pairs_attempted``, ``n_swap_pairs_accepted``,
            ``acceptance_rate``, ``n_energy_evals``, ``n_grad_evals``.
        """
        n_att = int(jnp.asarray(swap_info["n_attempted"]))
        n_acc = int(jnp.asarray(swap_info["n_accepted"]))
        rate = n_acc / max(n_att, 1)
        n_evals = (
            int(jnp.asarray(swap_info["n_energy_evals"]))
            if "n_energy_evals" in swap_info
            else 0
        )
        # Per-pair arrays — host-side numpy copies for downstream
        # logging.  Always present in swap_info post-2026-05 kernel
        # extension; defensively zero-filled for legacy callers.
        n_acc_pp = swap_info.get("n_accepted_per_pair")
        n_att_pp = swap_info.get("n_attempted_per_pair")
        n_acc_pp = (
            np.asarray(n_acc_pp, dtype=np.int32)
            if n_acc_pp is not None
            else np.zeros(0, dtype=np.int32)
        )
        n_att_pp = (
            np.asarray(n_att_pp, dtype=np.int32)
            if n_att_pp is not None
            else np.zeros(0, dtype=np.int32)
        )
        return {
            "n_swap_pairs_attempted": n_att,
            "n_swap_pairs_accepted": n_acc,
            "acceptance_rate": rate,
            "n_energy_evals": n_evals,
            "n_grad_evals": 0,
            "n_accepted_per_pair": n_acc_pp,
            "n_attempted_per_pair": n_att_pp,
        }
