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

Design mirrors ``AdaptationManager``: built once at construction time, JIT'd
swap step cached; ``_run_loop`` calls ``fires(i)`` / ``apply(state, key)`` at
each iteration.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

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


_EMPTY_STATS: dict = {
    "n_swap_pairs_attempted": 0,
    "n_swap_pairs_accepted": 0,
    "acceptance_rate": 0.0,
    "n_energy_evals": 0,
    "n_grad_evals": 0,
}


class InterREManager:
    """Manages inter-replica-exchange swap passes in the NS outer loop.

    Constructed once before the loop starts.  ``_run_loop`` calls
    ``fires(i)`` to decide whether to fire on iteration ``i``, then calls
    ``apply(ns_state, rng_key)`` to run the swap pass.

    Args:
        swap_kernel: :class:`SwapKernel` instance (e.g. ``PressureRENSSwap()``).
        batch_descriptor: ``BatchDescriptor`` controlling the execution mode.
        backend: Energy backend (unused for ``PressureRENSSwap`` but part of
            the general API for future kernels such as ``XRENSSwap``).
        every: Fire a swap pass every this many NS iterations.  0 → never fire.
        n_swap_cycles: Number of even+odd swap phases per fire.
    """

    def __init__(
        self,
        swap_kernel: SwapKernel,
        batch_descriptor: BatchDescriptor,
        backend: Any,
        every: int = 1,
        n_swap_cycles: int = 1,
    ) -> None:
        self._swap_kernel = swap_kernel
        self._descriptor = batch_descriptor
        self._backend = backend
        self._every = every
        self._n_swap_cycles = n_swap_cycles

        # Kernel flavor flags (mutually exclusive).
        self._is_xrens = isinstance(swap_kernel, XRENSSwap)
        self._is_semi_grand = isinstance(swap_kernel, SemiGrandSwap)

        # Build and cache the JIT-compiled swap step.
        self._jit_vmap_swap = None
        self._jit_pmap_swap = None
        self._jit_xrens_vmap_swap = None
        self._jit_xrens_pmap_swap = None
        if batch_descriptor.is_batched:
            self._jit_vmap_swap, self._jit_pmap_swap = self._build_jit_fns()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fires(self, iteration: int) -> bool:
        """Return True iff a swap pass should fire on this iteration.

        Rules: every > 0 AND iteration > 0 AND iteration % every == 0.
        Iteration 0 is skipped to match ``AdaptationManager.fires`` convention.

        Args:
            iteration: Current NS iteration index (Python int).

        Returns:
            True when a swap should be attempted.
        """
        return self._every > 0 and iteration > 0 and iteration % self._every == 0

    @property
    def is_active(self) -> bool:
        """True iff this manager will actually do work.

        ``SingleRun`` descriptors return False (no batched population to swap).
        ``VmapRuns`` / ``PmapVmapRuns`` return True when ``every > 0``.
        """
        return self._descriptor.is_batched and self._every > 0

    def apply(
        self, ns_state: Any, rng_key: jax.Array
    ) -> tuple[Any, dict, jax.Array]:
        """Run one inter-RE swap pass.

        For ``SingleRun``: returns state unchanged with zero stats.
        For ``VmapRuns``: operates on ``(n_runs, K, ...)`` population directly.
        For ``PmapVmapRuns``: all_gathers across the pmap axis, swaps, re-shards.

        Args:
            ns_state: Current ``NSState`` (single, vmapped, or pmap-vmapped).
            rng_key: Scalar PRNG key for swap randomness.

        Returns:
            Tuple ``(new_ns_state, swap_stats, new_rng_key)`` where:

            * ``new_ns_state``: Updated ``NSState`` after swaps.
            * ``swap_stats``: Dict with keys
              ``{"n_swap_pairs_attempted": int,
                 "n_swap_pairs_accepted": int,
                 "acceptance_rate": float,
                 "n_energy_evals": int,
                 "n_grad_evals": int}``.
              All zeros for ``SingleRun`` (no-op).
            * ``new_rng_key``: Advanced PRNG key carry (scalar).
        """
        if not self._descriptor.is_batched:
            # SingleRun: no-op
            new_key = jax.random.split(rng_key)[0]
            return ns_state, dict(_EMPTY_STATS), new_key

        rng_key, swap_key = jax.random.split(rng_key)

        if isinstance(self._descriptor, PmapVmapRuns):
            new_ns_state, swap_stats = self._apply_pmap_vmap(ns_state, swap_key)
        else:
            # VmapRuns
            new_ns_state, swap_stats = self._apply_vmap(ns_state, swap_key)

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
            ``pmap(vmap_all_gather_swap, axis_name="gpu")``.
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
            # XRENS path: requires composition_targets and backend.
            def _xrens_swap_fn(rng_key, positions, types, energies, cells,
                               emax, pressures, composition_targets):
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
            jit_vmap = jax.jit(_xrens_swap_fn)
        elif self._is_semi_grand:
            # Semi-grand path: requires chemical_potentials; zero backend calls.
            def _sg_swap_fn(rng_key, positions, types, energies, cells,
                            emax, pressures, chemical_potentials):
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
            jit_vmap = jax.jit(_sg_swap_fn)
        else:
            def _swap_fn(rng_key, positions, types, energies, cells, emax, pressures):
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
            jit_vmap = jax.jit(_swap_fn)

        # Build pmap version: each device receives its own shard (P, K, ...).
        # We use lax.all_gather to replicate across devices, swap on the full
        # population, then slice back the device's shard.
        if self._is_xrens:
            def _pmap_body(rng_key_per_device, pos, typ, ene, bxs, em, pres, comp_targets):
                full_pos = jax.lax.all_gather(pos, axis_name="gpu", axis=0)
                full_typ = jax.lax.all_gather(typ, axis_name="gpu", axis=0)
                full_ene = jax.lax.all_gather(ene, axis_name="gpu", axis=0)
                full_em  = jax.lax.all_gather(em,  axis_name="gpu", axis=0)
                full_bxs = (
                    jax.lax.all_gather(bxs, axis_name="gpu", axis=0)
                    if bxs is not None else None
                )
                full_pres = (
                    jax.lax.all_gather(pres, axis_name="gpu", axis=0)
                    if pres is not None else None
                )
                full_comp = jax.lax.all_gather(comp_targets, axis_name="gpu", axis=0)

                G, P = full_pos.shape[0], full_pos.shape[1]
                gp = G * P
                flat_pos  = full_pos.reshape((gp,) + full_pos.shape[2:])
                flat_typ  = full_typ.reshape((gp,) + full_typ.shape[2:])
                flat_ene  = full_ene.reshape((gp,) + full_ene.shape[2:])
                flat_em   = full_em.reshape((gp,))
                flat_bxs  = full_bxs.reshape((gp,) + full_bxs.shape[2:]) if full_bxs is not None else None
                flat_pres = full_pres.reshape((gp,)) if full_pres is not None else None
                flat_comp = full_comp.reshape((gp,) + full_comp.shape[2:])

                new_pos_flat, new_typ_flat, new_ene_flat, new_bxs_flat, swap_info = (
                    xrens_replica_exchange_step(
                        rng_key=rng_key_per_device,
                        all_positions=flat_pos,
                        all_types=flat_typ,
                        all_energies=flat_ene,
                        all_cells=flat_bxs,
                        all_emax=flat_em,
                        composition_targets=flat_comp,
                        backend=backend,
                        xrens_kernel=kernel,
                        pressures=flat_pres,
                        n_swap_cycles=n_swap_cycles,
                    )
                )

                new_pos_full = new_pos_flat.reshape(full_pos.shape)
                new_typ_full = new_typ_flat.reshape(full_typ.shape)
                new_ene_full = new_ene_flat.reshape(full_ene.shape)
                new_bxs_full = (
                    new_bxs_flat.reshape(full_bxs.shape) if new_bxs_flat is not None else None
                )

                dev_idx = jax.lax.axis_index("gpu")
                shard_pos = new_pos_full[dev_idx]
                shard_typ = new_typ_full[dev_idx]
                shard_ene = new_ene_full[dev_idx]
                shard_bxs = new_bxs_full[dev_idx] if new_bxs_full is not None else None
                return shard_pos, shard_typ, shard_ene, shard_bxs, swap_info

        elif self._is_semi_grand:
            def _pmap_body(rng_key_per_device, pos, typ, ene, bxs, em, pres, chem_pots):
                full_pos = jax.lax.all_gather(pos, axis_name="gpu", axis=0)
                full_typ = jax.lax.all_gather(typ, axis_name="gpu", axis=0)
                full_ene = jax.lax.all_gather(ene, axis_name="gpu", axis=0)
                full_em  = jax.lax.all_gather(em,  axis_name="gpu", axis=0)
                full_bxs = (
                    jax.lax.all_gather(bxs, axis_name="gpu", axis=0)
                    if bxs is not None else None
                )
                full_pres = (
                    jax.lax.all_gather(pres, axis_name="gpu", axis=0)
                    if pres is not None else None
                )
                full_chem = jax.lax.all_gather(chem_pots, axis_name="gpu", axis=0)

                G, P = full_pos.shape[0], full_pos.shape[1]
                gp = G * P
                flat_pos  = full_pos.reshape((gp,) + full_pos.shape[2:])
                flat_typ  = full_typ.reshape((gp,) + full_typ.shape[2:])
                flat_ene  = full_ene.reshape((gp,) + full_ene.shape[2:])
                flat_em   = full_em.reshape((gp,))
                flat_bxs  = full_bxs.reshape((gp,) + full_bxs.shape[2:]) if full_bxs is not None else None
                flat_pres = full_pres.reshape((gp,)) if full_pres is not None else None
                flat_chem = full_chem.reshape((gp,) + full_chem.shape[2:])

                new_pos_flat, new_typ_flat, new_ene_flat, new_bxs_flat, swap_info = (
                    semi_grand_replica_exchange_step(
                        rng_key=rng_key_per_device,
                        all_positions=flat_pos,
                        all_types=flat_typ,
                        all_energies=flat_ene,
                        all_cells=flat_bxs,
                        all_emax=flat_em,
                        chemical_potentials=flat_chem,
                        semi_grand_kernel=kernel,
                        pressures=flat_pres,
                        n_swap_cycles=n_swap_cycles,
                    )
                )

                new_pos_full = new_pos_flat.reshape(full_pos.shape)
                new_typ_full = new_typ_flat.reshape(full_typ.shape)
                new_ene_full = new_ene_flat.reshape(full_ene.shape)
                new_bxs_full = (
                    new_bxs_flat.reshape(full_bxs.shape) if new_bxs_flat is not None else None
                )

                dev_idx = jax.lax.axis_index("gpu")
                shard_pos = new_pos_full[dev_idx]
                shard_typ = new_typ_full[dev_idx]
                shard_ene = new_ene_full[dev_idx]
                shard_bxs = new_bxs_full[dev_idx] if new_bxs_full is not None else None
                return shard_pos, shard_typ, shard_ene, shard_bxs, swap_info

        else:
            def _pmap_body(rng_key_per_device, pos, typ, ene, bxs, em, pres):
                # Inside pmap each shard has shape (P, K, ...).
                # all_gather collects all G shards → (G, P, K, ...) on each device.
                full_pos = jax.lax.all_gather(pos, axis_name="gpu", axis=0)
                full_typ = jax.lax.all_gather(typ, axis_name="gpu", axis=0)
                full_ene = jax.lax.all_gather(ene, axis_name="gpu", axis=0)
                full_em  = jax.lax.all_gather(em,  axis_name="gpu", axis=0)
                full_bxs = (
                    jax.lax.all_gather(bxs, axis_name="gpu", axis=0)
                    if bxs is not None else None
                )
                full_pres = (
                    jax.lax.all_gather(pres, axis_name="gpu", axis=0)
                    if pres is not None else None
                )

                # Flatten (G, P, K, ...) → (G*P, K, ...) for the swap function.
                G, P = full_pos.shape[0], full_pos.shape[1]
                gp = G * P
                flat_pos  = full_pos.reshape((gp,) + full_pos.shape[2:])
                flat_typ  = full_typ.reshape((gp,) + full_typ.shape[2:])
                flat_ene  = full_ene.reshape((gp,) + full_ene.shape[2:])
                flat_em   = full_em.reshape((gp,))
                flat_bxs  = full_bxs.reshape((gp,) + full_bxs.shape[2:]) if full_bxs is not None else None
                flat_pres = full_pres.reshape((gp,)) if full_pres is not None else None

                new_pos_flat, new_typ_flat, new_ene_flat, new_bxs_flat, swap_info = (
                    replica_exchange_step(
                        rng_key=rng_key_per_device,
                        all_positions=flat_pos,
                        all_types=flat_typ,
                        all_energies=flat_ene,
                        all_cells=flat_bxs,
                        all_emax=flat_em,
                        pressures=flat_pres,
                        n_swap_cycles=n_swap_cycles,
                        swap_kernel=kernel,
                    )
                )

                # Reshape back and take this device's shard.
                new_pos_full = new_pos_flat.reshape(full_pos.shape)
                new_typ_full = new_typ_flat.reshape(full_typ.shape)
                new_ene_full = new_ene_flat.reshape(full_ene.shape)
                new_bxs_full = (
                    new_bxs_flat.reshape(full_bxs.shape) if new_bxs_flat is not None else None
                )

                dev_idx = jax.lax.axis_index("gpu")
                shard_pos  = new_pos_full[dev_idx]
                shard_typ  = new_typ_full[dev_idx]
                shard_ene  = new_ene_full[dev_idx]
                shard_bxs  = new_bxs_full[dev_idx] if new_bxs_full is not None else None

                return shard_pos, shard_typ, shard_ene, shard_bxs, swap_info

        jit_pmap = jax.pmap(_pmap_body, axis_name="gpu")

        return jit_vmap, jit_pmap

    def _extract_swap_inputs(self, ns_state: Any):
        """Extract swap inputs from state.

        For VmapRuns the population has shape ``(n_runs, K, ...)``.
        For PmapVmapRuns the population has shape ``(G, P, K, ...)``.

        Returns:
            Tuple of JAX arrays:
            ``(positions, types, energies, cells, emax, pressures,
               composition_targets, chemical_potentials)``
            where ``composition_targets`` is ``None`` unless ``XRENSSwap``
            mode is active, and ``chemical_potentials`` is ``None`` unless
            ``SemiGrandSwap`` mode is active.
        """
        pop = ns_state.population

        positions = pop.positions   # (R, K, n_atoms, 3) or (G, P, K, n_atoms, 3)
        types = pop.types           # varies
        energies = pop.energy       # (R, K) or (G, P, K)
        cells = pop.cell            # (R, K, 3, 3) or (G, P, K, 3, 3)

        # emax per run: max over the walker axis.
        # VmapRuns: walker axis = 1 → shape (R,)
        # PmapVmapRuns: walker axis = 2 → shape (G, P)
        if isinstance(self._descriptor, PmapVmapRuns):
            emax = jnp.max(energies, axis=2)  # (G, P)
        else:
            emax = jnp.max(energies, axis=1)  # (R,)

        # Pressures, composition_targets, chemical_potentials: extract from
        # ensemble_params.  After vmapping init_ns, scalar per-run values have
        # shape (n_runs, n_walkers) or (G, P, n_walkers).  We need per-run
        # vectors: take first-walker slice.
        pressures = None
        composition_targets = None
        chemical_potentials = None
        ep = getattr(pop, "ensemble_params", None)
        if ep is not None and isinstance(ep, dict):
            if "pressure" in ep:
                p_val = ep["pressure"]
                arr = jnp.asarray(p_val)
                if arr.ndim == 0:
                    # Scalar pressure — replicate across all runs.
                    if isinstance(self._descriptor, PmapVmapRuns):
                        G, P = self._descriptor.n_gpu, self._descriptor.n_per_gpu
                        pressures = jnp.broadcast_to(arr, (G, P))
                    else:
                        pressures = jnp.broadcast_to(arr, (self._descriptor.n_runs,))
                elif arr.ndim == 1:
                    # Already (n_runs,) — use directly.
                    pressures = arr
                elif arr.ndim == 2:
                    # (n_runs, n_walkers) or (G, P) — take first walker per run.
                    if isinstance(self._descriptor, PmapVmapRuns):
                        pressures = arr
                    else:
                        pressures = arr[:, 0]
                elif arr.ndim == 3:
                    # (G, P, n_walkers) for PmapVmapRuns — take first walker.
                    pressures = arr[:, :, 0]

            if "target_composition" in ep and self._is_xrens:
                # target_composition per run: shape (n_runs, n_walkers, n_species)
                # after vmapping — take first walker's composition per run.
                tc_val = ep["target_composition"]
                tc_arr = jnp.asarray(tc_val, dtype=jnp.int32)
                # VmapRuns: (n_runs, n_walkers, n_species) → (n_runs, n_species)
                # PmapVmapRuns: (G, P, n_walkers, n_species) → (G, P, n_species)
                if isinstance(self._descriptor, PmapVmapRuns):
                    if tc_arr.ndim == 4:
                        # (G, P, n_walkers, n_species)
                        composition_targets = tc_arr[:, :, 0, :]
                    elif tc_arr.ndim == 3:
                        # (G, P, n_species) — already per-run
                        composition_targets = tc_arr
                    else:
                        composition_targets = tc_arr
                else:
                    if tc_arr.ndim == 3:
                        # (n_runs, n_walkers, n_species) → (n_runs, n_species)
                        composition_targets = tc_arr[:, 0, :]
                    elif tc_arr.ndim == 2:
                        # (n_runs, n_species) — already per-run
                        composition_targets = tc_arr
                    else:
                        composition_targets = tc_arr

            if "chemical_potentials" in ep and self._is_semi_grand:
                # chemical_potentials per run: shape (n_runs, n_walkers, n_species)
                # after vmapping — take first walker's vector per run.
                cp_val = ep["chemical_potentials"]
                cp_arr = jnp.asarray(cp_val, dtype=jnp.float32)
                # VmapRuns: (n_runs, n_walkers, n_species) → (n_runs, n_species)
                # PmapVmapRuns: (G, P, n_walkers, n_species) → (G, P, n_species)
                if isinstance(self._descriptor, PmapVmapRuns):
                    if cp_arr.ndim == 4:
                        chemical_potentials = cp_arr[:, :, 0, :]
                    elif cp_arr.ndim == 3:
                        chemical_potentials = cp_arr
                    else:
                        chemical_potentials = cp_arr
                else:
                    if cp_arr.ndim == 3:
                        chemical_potentials = cp_arr[:, 0, :]
                    elif cp_arr.ndim == 2:
                        chemical_potentials = cp_arr
                    else:
                        chemical_potentials = cp_arr

        return (
            positions, types, energies, cells, emax,
            pressures, composition_targets, chemical_potentials,
        )

    def _apply_vmap(self, ns_state: Any, swap_key: jax.Array):
        """Apply swap pass for VmapRuns descriptor.

        State population has shape ``(n_runs, K, ...)``.
        """
        (positions, types, energies, cells, emax,
         pressures, composition_targets, chemical_potentials) = (
            self._extract_swap_inputs(ns_state)
        )

        if self._is_xrens:
            if composition_targets is None:
                raise ValueError(
                    "XRENSSwap requires 'target_composition' in ensemble_params "
                    "for every run. Ensure composition_targets were injected at "
                    "init time via ensemble_params_per_run."
                )
            new_pos, new_types, new_ene, new_cells, swap_info = self._jit_vmap_swap(
                swap_key, positions, types, energies, cells, emax,
                pressures, composition_targets,
            )
        elif self._is_semi_grand:
            if chemical_potentials is None:
                raise ValueError(
                    "SemiGrandSwap requires 'chemical_potentials' in ensemble_params "
                    "for every run. Ensure chemical_potentials were injected at "
                    "init time via ensemble_params_per_run."
                )
            new_pos, new_types, new_ene, new_cells, swap_info = self._jit_vmap_swap(
                swap_key, positions, types, energies, cells, emax,
                pressures, chemical_potentials,
            )
        else:
            new_pos, new_types, new_ene, new_cells, swap_info = self._jit_vmap_swap(
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

    def _apply_pmap_vmap(self, ns_state: Any, swap_key: jax.Array):
        """Apply swap pass for PmapVmapRuns descriptor.

        Population has shape ``(G, P, K, ...)``.  Uses ``lax.all_gather``
        so every device sees the full ``(G*P, K, ...)`` population.

        For ``n_gpu=1`` the all_gather is a no-op (cost zero).
        The same code runs unconditionally for all n_gpu values so that
        multi-GPU correctness can be tested without a fork.
        """
        (positions, types, energies, cells, emax,
         pressures, composition_targets, chemical_potentials) = (
            self._extract_swap_inputs(ns_state)
        )
        G = self._descriptor.n_gpu

        # Broadcast the same rng_key to all devices so every device makes
        # identical swap decisions (deterministic = same output on all devices).
        per_device_key = jnp.broadcast_to(swap_key[None], (G,) + swap_key.shape)

        if self._is_xrens:
            if composition_targets is None:
                raise ValueError(
                    "XRENSSwap requires 'target_composition' in ensemble_params."
                )
            new_pos, new_types, new_ene, new_cells_out, swap_info_sharded = (
                self._jit_pmap_swap(
                    per_device_key, positions, types, energies, cells, emax,
                    pressures, composition_targets,
                )
            )
        elif self._is_semi_grand:
            if chemical_potentials is None:
                raise ValueError(
                    "SemiGrandSwap requires 'chemical_potentials' in ensemble_params."
                )
            new_pos, new_types, new_ene, new_cells_out, swap_info_sharded = (
                self._jit_pmap_swap(
                    per_device_key, positions, types, energies, cells, emax,
                    pressures, chemical_potentials,
                )
            )
        else:
            new_pos, new_types, new_ene, new_cells_out, swap_info_sharded = (
                self._jit_pmap_swap(
                    per_device_key, positions, types, energies, cells, emax, pressures
                )
            )

        new_pop = ns_state.population.set(
            positions=new_pos,
            types=new_types,
            energy=new_ene,
            cell=new_cells_out,
        )
        new_ns_state = ns_state.set(population=new_pop)

        # All devices ran the same swap (same RNG + same data after all_gather),
        # so device 0's swap_info is representative.
        stats = self._build_stats({
            "n_accepted": swap_info_sharded["n_accepted"][0],
            "n_attempted": swap_info_sharded["n_attempted"][0],
        })
        return new_ns_state, stats

    @staticmethod
    def _build_stats(swap_info: dict) -> dict:
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
        n_evals = int(jnp.asarray(swap_info["n_energy_evals"])) if "n_energy_evals" in swap_info else 0
        return {
            "n_swap_pairs_attempted": n_att,
            "n_swap_pairs_accepted": n_acc,
            "acceptance_rate": rate,
            "n_energy_evals": n_evals,
            "n_grad_evals": 0,
        }
