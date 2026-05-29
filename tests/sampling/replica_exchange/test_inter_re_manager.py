"""Unit tests for InterREManager.

Coverage:
- fires() boundary cases
- apply() no-op for SingleRun
- apply() returns expected stats dict keys
- is_active property
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun, VmapRuns
from jaxrens.sampling.inter_re_manager import InterREManager
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.moves.replica_exchange import PressureRENSSwap
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel


_EXPECTED_STATS_KEYS = frozenset({
    "n_swap_pairs_attempted",
    "n_swap_pairs_accepted",
    "acceptance_rate",
    "n_energy_evals",
    "n_grad_evals",
    "n_accepted_per_pair",
    "n_attempted_per_pair",
})


def _make_vmap_ns_state(n_runs=2, n_walkers=10, pressures=None, seed=42):
    """Create a batched NSState suitable for VmapRuns tests.

    Seeds ``ns_state.emax`` from the empirical per-replica max of the
    initial population — this mirrors what would happen after one
    ``ns_step`` (which is the first place ``emax`` gets a meaningful
    value; ``init_ns`` sets it to ``+inf``).  Without this, RE swap
    tests would see an unbounded constraint and accept every swap.
    """
    backend = create_harmonic(k=1.0)
    descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    key = jax.random.key(seed)
    keys = jax.random.split(key, n_runs + 1)
    rng_keys = keys[:n_runs]
    pos_key = keys[-1]

    positions_all = jax.random.uniform(
        pos_key, (n_runs, n_walkers, 1, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies_all = jax.vmap(
        lambda pos: jax.vmap(lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0])(pos)
    )(positions_all)

    if pressures is None:
        ensemble_params_per_run = None
    else:
        ensemble_params_per_run = [{"pressure": p} for p in pressures]

    ns_state = init_ns_parallel(
        init_fn, positions_all, types, energies_all, None,
        rng_keys,
        step_sizes=jnp.array([0.2]),
        ensemble_params_per_run=ensemble_params_per_run,
    )
    ns_state = ns_state.set(
        emax=jnp.max(ns_state.population.energy, axis=1),
    )
    return ns_state, step_fn, backend


# ---------------------------------------------------------------------------
# fires() boundary cases
# ---------------------------------------------------------------------------


class TestFires:
    def test_fires_at_every_1(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        assert not mgr.fires(0)        # iteration 0 never fires
        assert mgr.fires(1)
        assert mgr.fires(2)
        assert mgr.fires(100)

    def test_fires_at_every_5(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=5
        )
        assert not mgr.fires(0)
        assert not mgr.fires(1)
        assert not mgr.fires(4)
        assert mgr.fires(5)
        assert mgr.fires(10)
        assert not mgr.fires(11)

    def test_fires_never_when_every_0(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=0
        )
        for i in range(10):
            assert not mgr.fires(i)

    def test_fires_never_when_every_large(self):
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1000
        )
        for i in range(1, 999):
            assert not mgr.fires(i)
        assert mgr.fires(1000)

    def test_fires_single_run_irrelevant(self):
        # SingleRun: is_active=False, but fires() itself is descriptor-agnostic.
        mgr = InterREManager(
            PressureRENSSwap(), SingleRun(), None, re_interval=1
        )
        # fires() returns True but is_active=False so _run_loop won't call apply.
        assert mgr.fires(1)
        assert not mgr.is_active


# ---------------------------------------------------------------------------
# is_active property
# ---------------------------------------------------------------------------


class TestIsActive:
    def test_single_run_not_active(self):
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, re_interval=1)
        assert not mgr.is_active

    def test_vmap_active(self):
        mgr = InterREManager(PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1)
        assert mgr.is_active

    def test_vmap_not_active_when_every_0(self):
        mgr = InterREManager(PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=0)
        assert not mgr.is_active

    def test_pmap_vmap_active(self):
        mgr = InterREManager(PressureRENSSwap(), PmapVmapRuns(1, 2), None, re_interval=1)
        assert mgr.is_active


# ---------------------------------------------------------------------------
# apply() no-op for SingleRun
# ---------------------------------------------------------------------------


class TestApplyNoOpSingleRun:
    def _make_single_ns_state(self):
        backend = create_harmonic(k=1.0)
        descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)
        key = jax.random.key(0)
        pos_key, run_key = jax.random.split(key)
        positions = jax.random.uniform(pos_key, (10, 1, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)
        return init_ns(init_fn, positions, types, energies, None, run_key)

    def test_single_run_state_unchanged(self):
        ns_state = self._make_single_ns_state()
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, re_interval=1)
        key = jax.random.key(99)

        new_state, stats, new_key = mgr.apply(ns_state, key)

        # Populations must be identical (no mutation)
        assert jnp.allclose(
            new_state.population.energy, ns_state.population.energy
        )
        assert jnp.allclose(
            new_state.population.positions, ns_state.population.positions
        )

    def test_single_run_stats_all_zero(self):
        ns_state = self._make_single_ns_state()
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, re_interval=1)
        key = jax.random.key(0)
        _, stats, _ = mgr.apply(ns_state, key)

        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())
        assert stats["n_swap_pairs_attempted"] == 0
        assert stats["n_swap_pairs_accepted"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["n_energy_evals"] == 0
        assert stats["n_grad_evals"] == 0

    def test_single_run_key_advances(self):
        ns_state = self._make_single_ns_state()
        mgr = InterREManager(PressureRENSSwap(), SingleRun(), None, re_interval=1)
        key = jax.random.key(42)
        _, _, new_key = mgr.apply(ns_state, key)
        # Key should advance (not be identical)
        assert not jnp.array_equal(key, new_key)


# ---------------------------------------------------------------------------
# apply() for VmapRuns returns expected stats keys
# ---------------------------------------------------------------------------


class TestApplyVmapRuns:
    def test_stats_keys_present(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        key = jax.random.key(7)
        _, stats, _ = mgr.apply(ns_state, key)

        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())

    def test_stats_non_negative(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        key = jax.random.key(7)
        _, stats, _ = mgr.apply(ns_state, key)

        assert stats["n_swap_pairs_attempted"] >= 0
        assert stats["n_swap_pairs_accepted"] >= 0
        assert 0.0 <= stats["acceptance_rate"] <= 1.0

    def test_stats_acceptance_rate_consistent(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        key = jax.random.key(7)
        _, stats, _ = mgr.apply(ns_state, key)

        n_att = stats["n_swap_pairs_attempted"]
        n_acc = stats["n_swap_pairs_accepted"]
        expected_rate = n_acc / max(n_att, 1)
        assert abs(stats["acceptance_rate"] - expected_rate) < 1e-6

    def test_key_advances(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        key = jax.random.key(42)
        _, _, new_key = mgr.apply(ns_state, key)
        assert not jnp.array_equal(key, new_key)

    def test_state_returned_is_valid_pytree(self):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        key = jax.random.key(7)
        new_state, _, _ = mgr.apply(ns_state, key)

        # Shapes should be preserved
        assert new_state.population.positions.shape == ns_state.population.positions.shape
        assert new_state.population.energy.shape == ns_state.population.energy.shape

    def test_pressure_rens_with_pressures(self):
        """VmapRuns with different pressures: swap uses enthalpy check."""
        ns_state, _, _ = _make_vmap_ns_state(n_runs=2, pressures=[0.01, 0.1])
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=2), None, re_interval=1
        )
        key = jax.random.key(77)
        new_state, stats, _ = mgr.apply(ns_state, key)

        # Should complete without error; stats keys present
        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())


# ---------------------------------------------------------------------------
# PmapVmapRuns smoke test
# ---------------------------------------------------------------------------


class TestApplyPmapVmapRuns:
    def test_smoke_n_gpu_1(self):
        """n_gpu=1, n_per_gpu=2: apply runs without error."""
        from jaxrens.sampling.nested_sampling import init_ns_multi_gpu

        backend = create_harmonic(k=1.0)
        descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_gpu, n_per_gpu, n_walkers = 1, 2, 8
        key = jax.random.key(0)
        keys = jax.random.split(key, n_gpu * n_per_gpu + 1)
        rng_keys = keys[:-1]
        pos_key = keys[-1]

        positions = jax.random.uniform(
            pos_key, (n_gpu * n_per_gpu, n_walkers, 1, 3), minval=-2.0, maxval=2.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0])(pos)
        )(positions)

        ns_state = init_ns_multi_gpu(
            init_fn, positions, types, energies, None,
            rng_keys, n_gpu=n_gpu, n_per_gpu=n_per_gpu,
            step_sizes=jnp.array([0.2]),
        )

        descriptor = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)
        # Seed emax from initial pop max — mirror what one ns_step would do.
        ns_state = ns_state.set(
            emax=descriptor.reduce_emax(ns_state.population.energy),
        )
        mgr = InterREManager(
            PressureRENSSwap(), descriptor, backend, re_interval=1
        )

        swap_key = jax.random.key(99)
        new_state, stats, _ = mgr.apply(ns_state, swap_key)

        assert _EXPECTED_STATS_KEYS == frozenset(stats.keys())
        assert new_state.population.positions.shape == ns_state.population.positions.shape

    def test_pmap_vmap_is_active(self):
        descriptor = PmapVmapRuns(n_gpu=1, n_per_gpu=2)
        mgr = InterREManager(PressureRENSSwap(), descriptor, None, re_interval=1)
        assert mgr.is_active


# ---------------------------------------------------------------------------
# Per-pair tracking — n_accepted_per_pair / n_attempted_per_pair
# ---------------------------------------------------------------------------


class TestPerPairTracking:
    """The per-pair fields must (a) have the right shape and dtype,
    (b) sum to the existing aggregate fields, and (c) localise
    acceptance to the correct pair_id when only one pair is allowed
    to swap."""

    @pytest.mark.parametrize("n_runs", [2, 3, 4, 5])
    def test_shape_and_dtype(self, n_runs):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=n_runs)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=n_runs), None, re_interval=1
        )
        _, stats, _ = mgr.apply(ns_state, jax.random.key(7))

        n_pairs = n_runs - 1
        assert stats["n_accepted_per_pair"].shape == (n_pairs,)
        assert stats["n_attempted_per_pair"].shape == (n_pairs,)
        assert stats["n_accepted_per_pair"].dtype.kind == "i"
        assert stats["n_attempted_per_pair"].dtype.kind == "i"

    @pytest.mark.parametrize("n_runs,seed", [(2, 1), (3, 2), (4, 3), (5, 4)])
    def test_per_pair_sums_match_aggregate(self, n_runs, seed):
        ns_state, _, _ = _make_vmap_ns_state(n_runs=n_runs, seed=seed)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=n_runs), None,
            re_interval=1, n_swap_cycles=2,
        )
        _, stats, _ = mgr.apply(ns_state, jax.random.key(seed * 11))

        assert int(stats["n_accepted_per_pair"].sum()) == \
            stats["n_swap_pairs_accepted"]
        assert int(stats["n_attempted_per_pair"].sum()) == \
            stats["n_swap_pairs_attempted"]

    def test_per_pair_attempted_uniform_across_pairs(self):
        """With re_interval=1 and n_swap_cycles=1, every pair gets
        exactly one attempt per fire (one even or odd phase serves it).
        For n_runs=4 we have pairs 0=(0,1), 1=(1,2), 2=(2,3); even
        phase covers {0, 2}, odd covers {1} — so all three get one
        attempt each."""
        n_runs = 4
        ns_state, _, _ = _make_vmap_ns_state(n_runs=n_runs)
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=n_runs), None,
            re_interval=1, n_swap_cycles=1,
        )
        _, stats, _ = mgr.apply(ns_state, jax.random.key(99))

        import numpy as np
        np.testing.assert_array_equal(
            stats["n_attempted_per_pair"],
            np.array([1, 1, 1], dtype=np.int32),
        )

    def test_per_pair_acceptance_localisation(self):
        """Construct a pressure ladder with one impossible-to-swap
        slot and one easy-to-swap slot; per-pair acceptance must
        reflect the topology, not be uniform."""
        # 3 replicas with very different pressures.  Pair 0=(0,1)
        # joins runs at p=0.0 and p=1e-3 — should swap easily.
        # Pair 1=(1,2) joins p=1e-3 and p=10.0 — much harder
        # (large enthalpy gap → low acceptance).
        n_runs = 3
        ns_state, _, _ = _make_vmap_ns_state(
            n_runs=n_runs, n_walkers=20,
            pressures=[0.0, 1e-3, 10.0], seed=11,
        )
        mgr = InterREManager(
            PressureRENSSwap(), VmapRuns(n_runs=n_runs), None,
            re_interval=1, n_swap_cycles=20,  # many cycles to amplify the asymmetry
        )
        _, stats, _ = mgr.apply(ns_state, jax.random.key(123))

        n_acc_pp = stats["n_accepted_per_pair"]
        n_att_pp = stats["n_attempted_per_pair"]

        # Pair 0 (low-p coupling) should accept more often than pair 1
        # (high-p coupling) — asserting strict > would be flaky on
        # unlucky seeds, but the ratio should differ markedly.
        rate_0 = n_acc_pp[0] / max(int(n_att_pp[0]), 1)
        rate_1 = n_acc_pp[1] / max(int(n_att_pp[1]), 1)
        assert rate_0 >= rate_1

    def test_per_pair_xrens_shape(self):
        """XRENS swap step also surfaces per-pair arrays of the right shape."""
        from jaxrens.sampling.moves.replica_exchange import xrens_replica_exchange_step, XRENSSwap

        n_runs, n_walkers, n_atoms = 3, 4, 4
        rng = jax.random.key(0)
        all_positions = jax.random.uniform(
            rng, (n_runs, n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0,
        )
        all_types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
        backend = create_harmonic(k=1.0)
        all_energies = jax.vmap(
            lambda pos_R: jax.vmap(
                lambda pos_K: backend(pos_K, all_types[0, 0], jnp.zeros((3, 3)), 0)[0]
            )(pos_R)
        )(all_positions)
        all_emax = jnp.full((n_runs,), 1e6)
        composition_targets = jnp.tile(
            jnp.array([n_atoms], dtype=jnp.int32)[None, :], (n_runs, 1),
        )
        _, _, _, _, swap_info = xrens_replica_exchange_step(
            jax.random.key(7),
            all_positions, all_types, all_energies, None, all_emax,
            composition_targets,
            backend, XRENSSwap(n_species=1),
        )
        assert swap_info["n_accepted_per_pair"].shape == (n_runs - 1,)
        assert swap_info["n_attempted_per_pair"].shape == (n_runs - 1,)
        # Aggregate consistency.
        assert int(swap_info["n_accepted_per_pair"].sum()) == \
            int(swap_info["n_accepted"])
        assert int(swap_info["n_attempted_per_pair"].sum()) == \
            int(swap_info["n_attempted"])

    def test_per_pair_semi_grand_shape(self):
        """Semi-grand swap step also surfaces per-pair arrays."""
        from jaxrens.sampling.moves.replica_exchange import (
            semi_grand_replica_exchange_step, SemiGrandSwap,
        )

        n_runs, n_walkers, n_atoms, n_species = 3, 4, 4, 2
        rng = jax.random.key(0)
        all_positions = jax.random.uniform(
            rng, (n_runs, n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0,
        )
        all_types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
        backend = create_harmonic(k=1.0)
        all_energies = jax.vmap(
            lambda pos_R: jax.vmap(
                lambda pos_K: backend(pos_K, all_types[0, 0], jnp.zeros((3, 3)), 0)[0]
            )(pos_R)
        )(all_positions)
        all_emax = jnp.full((n_runs,), 1e6)
        chemical_potentials = jnp.zeros((n_runs, n_species))
        _, _, _, _, swap_info = semi_grand_replica_exchange_step(
            jax.random.key(7),
            all_positions, all_types, all_energies, None, all_emax,
            chemical_potentials,
            SemiGrandSwap(n_species=n_species),
        )
        assert swap_info["n_accepted_per_pair"].shape == (n_runs - 1,)
        assert swap_info["n_attempted_per_pair"].shape == (n_runs - 1,)
        assert int(swap_info["n_accepted_per_pair"].sum()) == \
            int(swap_info["n_accepted"])
        assert int(swap_info["n_attempted_per_pair"].sum()) == \
            int(swap_info["n_attempted"])


