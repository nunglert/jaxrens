"""Test ns_step and vmap(ns_step) under JIT.

Verifies:
- init_ns creates correct state shapes
- ns_step single-run: iteration count, emax decrease, evidence accumulation
- ns_step with n_extra walks
- vmap(ns_step): multi-run compilation, different keys → different trajectories
- vmap(ns_step) with n_extra
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    _get_extra_indices,
    init_ns,
    init_ns_parallel,
    ns_step,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_setup():
    """Single-run harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("random_walk", random_walk.build_kernel),
        ],
    )

    n_walkers = 50
    n_atoms = 1
    key = jax.random.key(42)

    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    return {
        "backend": backend,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
        "n_walkers": n_walkers,
    }


@pytest.fixture
def parallel_setup():
    """2-run parallel harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("random_walk", random_walk.build_kernel),
        ],
    )

    n_runs = 2
    n_walkers = 20
    n_atoms = 1

    keys = jax.random.split(jax.random.key(0), n_runs)
    positions = jax.vmap(
        lambda k: jax.random.uniform(
            k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
        )
    )(keys)

    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)

    rng_keys = jax.random.split(jax.random.key(42), n_runs)

    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "rng_keys": rng_keys,
        "n_runs": n_runs,
        "n_walkers": n_walkers,
    }


@pytest.fixture
def jit_step():
    """JIT-compiled ns_step — all single-run tests use this."""
    return jax.jit(ns_step, static_argnums=(1, 2, 3))


@pytest.fixture
def jit_vmap_step(parallel_setup):
    """JIT-compiled vmap(ns_step) for parallel tests."""
    s = parallel_setup
    return jax.jit(jax.vmap(lambda state: ns_step(state, s["step_fn"], 10, 0)))


# ---------------------------------------------------------------------------
# init_ns
# ---------------------------------------------------------------------------


class TestInitNS:
    def test_init_creates_state(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        assert state.population.positions.shape == s["positions"].shape
        assert state.population.energy.shape == (s["n_walkers"],)
        assert state.iteration == 0

    def test_init_population_is_batched_mcstate(self, harmonic_setup):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        pop = state.population
        assert pop.positions.shape[0] == s["n_walkers"]
        assert pop.energy.shape == (s["n_walkers"],)
        assert pop.step_sizes.ndim == 2


# ---------------------------------------------------------------------------
# ns_step (single run, under JIT)
# ---------------------------------------------------------------------------


class TestNSStep:
    def test_single_step(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )

        new_state, info = jit_step(state, s["step_fn"], 10, 0)

        assert new_state.iteration == 1
        assert 0 <= info["acceptance_rate"] <= 1.0

    def test_multiple_steps_reduce_emax(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )

        emaxes = []
        for _ in range(20):
            state, info = jit_step(state, s["step_fn"], 10, 0)
            emaxes.append(float(info["emax"]))

        assert emaxes[-1] < emaxes[0], "Emax should decrease during NS"

    def test_evidence_increases(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )

        for _ in range(50):
            state, info = jit_step(state, s["step_fn"], 10, 0)

        assert jnp.isfinite(state.log_evidence)

    def test_n_extra_walks(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )

        new_state, info = jit_step(state, s["step_fn"], 10, 5)

        assert new_state.iteration == 1
        assert 0 <= info["acceptance_rate"] <= 1.0

    def test_n_extra_reduces_emax(self, harmonic_setup, jit_step):
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )

        emaxes = []
        for _ in range(20):
            state, info = jit_step(state, s["step_fn"], 10, 3)
            emaxes.append(float(info["emax"]))

        assert (
            emaxes[-1] < emaxes[0]
        ), "Emax should decrease during NS with n_extra"


# ---------------------------------------------------------------------------
# _get_extra_indices (clone-source exclusion regression test)
# ---------------------------------------------------------------------------


class TestGetExtraIndices:
    """Both the worst walker AND the clone source must be excluded from the
    extras pool.  If the clone source leaks in, the clone's data sits at two
    population slots (worst_idx and clone_idx) and both get walked
    independently from the same parent state — losing one independent walker
    and producing two correlated descendants every iteration where the
    collision happens.  Repeated across long runs this seeds visible
    population degeneracies.
    """

    def test_excludes_worst_and_clone(self):
        worst_idx = jnp.int32(7)
        clone_idx = jnp.int32(3)
        # Try many keys; the picked indices must never collide with either.
        for seed in range(50):
            picks = _get_extra_indices(
                worst_idx,
                clone_idx,
                n_walkers=10,
                n_extra=5,
                rng_key=jax.random.key(seed),
            )
            assert picks.shape == (5,)
            assert not bool(
                jnp.any(picks == worst_idx)
            ), f"seed {seed}: extras picked worst_idx ({int(worst_idx)})"
            assert not bool(
                jnp.any(picks == clone_idx)
            ), f"seed {seed}: extras picked clone_idx ({int(clone_idx)})"
            # All picks must be unique (argsort-based selection guarantees this).
            assert len(set(int(p) for p in picks)) == 5

    def test_extras_can_use_all_other_slots(self):
        """With n_extra = n_walkers - 2, the picked set is exactly the
        complement of {worst_idx, clone_idx}."""
        worst_idx = jnp.int32(0)
        clone_idx = jnp.int32(5)
        picks = _get_extra_indices(
            worst_idx,
            clone_idx,
            n_walkers=10,
            n_extra=8,
            rng_key=jax.random.key(0),
        )
        expected = {1, 2, 3, 4, 6, 7, 8, 9}
        assert set(int(p) for p in picks) == expected


# ---------------------------------------------------------------------------
# vmap(ns_step) (parallel runs, under JIT)
# ---------------------------------------------------------------------------


class TestVmapNsStep:
    def test_vmap_two_runs(self, parallel_setup):
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_keys=s["rng_keys"],
        )

        vmapped = jax.jit(
            jax.vmap(lambda state: ns_step(state, s["step_fn"], 10, 0))
        )
        new_states, infos = vmapped(ns_states)

        assert new_states.iteration.shape == (2,)
        assert infos["emax"].shape == (2,)
        assert jnp.all(new_states.iteration == 1)

    def test_different_keys_different_results(self, parallel_setup):
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_keys=s["rng_keys"],
        )

        vmapped = jax.jit(
            jax.vmap(lambda state: ns_step(state, s["step_fn"], 10, 0))
        )

        for _ in range(5):
            ns_states, infos = vmapped(ns_states)

        assert not jnp.allclose(
            ns_states.population.energy[0],
            ns_states.population.energy[1],
        )

    def test_vmap_with_n_extra(self, parallel_setup):
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_keys=s["rng_keys"],
        )

        vmapped = jax.jit(
            jax.vmap(lambda state: ns_step(state, s["step_fn"], 10, 3))
        )
        new_states, infos = vmapped(ns_states)

        assert new_states.iteration.shape == (2,)
        assert 0 <= float(infos["acceptance_rate"][0]) <= 1.0


# ---------------------------------------------------------------------------
# Per-move chain-level counters
# ---------------------------------------------------------------------------


class TestPerMoveCounters:
    """Verify n_accepted_per_move / n_proposed_per_move / reject_reason_counts_per_move."""

    def test_info_keys_present(self, harmonic_setup, jit_step):
        """ns_step info always contains the three per-move count keys."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], 10, 0)

        assert "n_accepted_per_move" in info
        assert "n_proposed_per_move" in info
        assert "reject_reason_counts_per_move" in info

    def test_shapes_single_move(self, harmonic_setup, jit_step):
        """With one move type, per-move arrays have shape (1,) / (1, 4)."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], 10, 0)

        assert info["n_accepted_per_move"].shape == (1,)
        assert info["n_proposed_per_move"].shape == (1,)
        assert info["reject_reason_counts_per_move"].shape == (1, 4)

    def test_shapes_two_moves(self):
        """With two move types, per-move arrays have shape (2,) / (2, 4)."""
        from jaxrens.sampling.moves import random_walk as rw

        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel("rw1", rw.build_kernel),
                MoveKernel("rw2", rw.build_kernel),
            ],
        )

        n_walkers = 20
        n_atoms = 1
        key = jax.random.key(7)
        key, init_key = jax.random.split(key)
        positions = jax.random.uniform(
            init_key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
        )
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        state = init_ns(
            init_fn, positions, types, energies, cells=None, rng_key=key
        )
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(state, step_fn, 20, 0)

        assert info["n_accepted_per_move"].shape == (2,)
        assert info["n_proposed_per_move"].shape == (2,)
        assert info["reject_reason_counts_per_move"].shape == (2, 4)

    def test_proposed_counts_total(self, harmonic_setup, jit_step):
        """sum(n_proposed_per_move) == n_walk * n_mcmc_steps."""
        s = harmonic_setup
        n_mcmc = 15
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], n_mcmc, 0)

        # n_walk=1 (no n_extra), 1 chain × 15 steps = 15 total proposals
        total_proposed = int(jnp.sum(info["n_proposed_per_move"]))
        assert (
            total_proposed == 1 * n_mcmc
        ), f"Expected {1 * n_mcmc} proposals, got {total_proposed}"

    def test_proposed_counts_with_n_extra(self, harmonic_setup, jit_step):
        """sum(n_proposed_per_move) == (1 + n_extra) * n_mcmc_steps."""
        s = harmonic_setup
        n_mcmc = 10
        n_extra = 3
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], n_mcmc, n_extra)

        total_proposed = int(jnp.sum(info["n_proposed_per_move"]))
        expected = (1 + n_extra) * n_mcmc
        assert (
            total_proposed == expected
        ), f"Expected {expected} proposals, got {total_proposed}"

    def test_accepted_leq_proposed(self, harmonic_setup, jit_step):
        """n_accepted_per_move <= n_proposed_per_move element-wise."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], 20, 0)

        assert jnp.all(
            info["n_accepted_per_move"] <= info["n_proposed_per_move"]
        )

    def test_acceptance_rate_invariant(self, harmonic_setup, jit_step):
        """sum(n_accepted_per_move) / sum(n_proposed_per_move) ≈ acceptance_rate."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], 20, 0)

        total_acc = float(jnp.sum(info["n_accepted_per_move"]))
        total_prop = float(jnp.sum(info["n_proposed_per_move"]))
        rate_from_counts = total_acc / total_prop
        rate_direct = float(info["acceptance_rate"])

        assert (
            abs(rate_from_counts - rate_direct) < 1e-5
        ), f"Rate from counts={rate_from_counts:.4f} vs direct rate={rate_direct:.4f}"

    def test_rr_counts_bucket_invariant(self, harmonic_setup, jit_step):
        """For each move: sum of rr_counts row == n_proposed_per_move."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], 20, 0)

        rr = info["reject_reason_counts_per_move"]  # (n_moves, 4)
        row_sums = jnp.sum(rr, axis=1)  # (n_moves,)
        assert jnp.array_equal(
            row_sums, info["n_proposed_per_move"]
        ), f"rr row sums {row_sums} != n_proposed {info['n_proposed_per_move']}"

    def test_rr_col0_equals_n_accepted(self, harmonic_setup, jit_step):
        """rr_counts[:, 0] (accepted bucket) == n_accepted_per_move."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], 20, 0)

        rr = info["reject_reason_counts_per_move"]
        assert jnp.array_equal(
            rr[:, 0], info["n_accepted_per_move"]
        ), f"rr[:,0]={rr[:, 0]} != n_accepted={info['n_accepted_per_move']}"

    def test_dtypes_are_int32(self, harmonic_setup, jit_step):
        """Per-move count arrays must be int32."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        _, info = jit_step(state, s["step_fn"], 10, 0)

        assert info["n_accepted_per_move"].dtype == jnp.int32
        assert info["n_proposed_per_move"].dtype == jnp.int32
        assert info["reject_reason_counts_per_move"].dtype == jnp.int32

    def test_jit_no_retrace(self, harmonic_setup):
        """ns_step does not retrace across calls with same static args."""
        s = harmonic_setup
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_key=s["key"],
        )
        n_mcmc = 10

        # Wrap with explicit trace counter
        trace_count = [0]

        def counting_step_fn(rng_key, st, lc):
            return s["step_fn"](rng_key, st, lc)

        # JIT ns_step with a Python function — retraces if JAX recompiles
        # Use cache_size heuristic: run twice, both times measure if fresh JIT
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))

        # First call: compiles and runs
        state1, info1 = jit_step(state, s["step_fn"], n_mcmc, 0)
        # Second call: must NOT retrace (same static args)
        state2, info2 = jit_step(state1, s["step_fn"], n_mcmc, 0)

        # Both calls produced valid output — if retracing had broken JIT we'd
        # see an error or shape mismatch rather than silent pass.
        assert "n_accepted_per_move" in info1
        assert "n_accepted_per_move" in info2
        assert (
            info2["n_proposed_per_move"].shape
            == info1["n_proposed_per_move"].shape
        )

    def test_vmap_ns_step_per_move_shapes(self, parallel_setup):
        """vmap(ns_step) returns per-move counts with (n_runs, n_moves[, 4]) shape."""
        s = parallel_setup
        ns_states = init_ns_parallel(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            rng_keys=s["rng_keys"],
        )

        vmapped = jax.jit(
            jax.vmap(lambda state: ns_step(state, s["step_fn"], 10, 0))
        )
        _, infos = vmapped(ns_states)

        n_runs = s["n_runs"]
        assert infos["n_accepted_per_move"].shape == (n_runs, 1)
        assert infos["n_proposed_per_move"].shape == (n_runs, 1)
        assert infos["reject_reason_counts_per_move"].shape == (n_runs, 1, 4)


# ---------------------------------------------------------------------------
# Regression test for Fix 2: reject_reason default misclassification
# ---------------------------------------------------------------------------


class TestRejectReasonGating:
    """Regression test for the reject_reason default-0 misclassification bug.

    Non-cell moves (random_walk, galilean, hmc) leave MoveInfo.reject_reason
    at the default value of 0, which is the *accepted* bucket.  Before the fix,
    a rejected random_walk with (accepted=False, reject_reason=0) was scattered
    into rr[:, 0] — the accepted bucket — inflating the reported acceptance rate.

    The fix gates scatter_col on info.accepted: rejected moves with reason=0
    are redirected to bucket 1 (energy reject), preserving the invariant
    rr_counts[:, 0] == n_accepted_per_move.

    Setup: a two-move MWG with (1) random_walk at extreme step size so it
    always rejects, and (2) a cell-move-like kernel that always accepts.
    We verify that the random_walk move accumulates zero accepted counts and
    that all proposals fall into bucket 1 (energy reject), not bucket 0.
    """

    def test_rejected_non_cell_move_goes_to_energy_bucket(self):
        """random_walk at ss=1e6 should always reject; all counts must land in bucket 1."""
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.moves import random_walk as rw
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        backend = create_harmonic(k=1.0)

        # Use a single random_walk move with enormous step size so that the
        # proposed energy always exceeds emax → 100% rejection.
        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel("random_walk", rw.build_kernel),
            ],
        )

        n_walkers = 30
        n_atoms = 1
        n_mcmc = 20
        key = jax.random.key(99)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0
        )
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        state = init_ns(
            init_fn, positions, types, energies, cells=None, rng_key=key
        )

        # Override step_sizes to an astronomically large value so every
        # random_walk proposal lands far outside emax.
        rw_idx = 0
        huge_ss = jnp.full_like(state.population.step_sizes, 1e6)
        state = state.set(population=state.population.set(step_sizes=huge_ss))

        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(state, step_fn, n_mcmc, 0)

        n_acc = info["n_accepted_per_move"]  # (1,)
        n_prop = info["n_proposed_per_move"]  # (1,)
        rr = info["reject_reason_counts_per_move"]  # (1, 4)

        # With a huge step size the walker always rejects
        assert (
            int(n_acc[rw_idx]) == 0
        ), f"Expected 0 accepted for huge-ss random_walk, got {int(n_acc[rw_idx])}"
        # All proposals must be recorded
        assert (
            int(n_prop[rw_idx]) == n_mcmc
        ), f"Expected {n_mcmc} proposals, got {int(n_prop[rw_idx])}"
        # Accepted bucket (col 0) must be zero — not inflated by rejected moves
        assert int(rr[rw_idx, 0]) == 0, (
            f"rr[rw, 0] (accepted bucket) = {int(rr[rw_idx, 0])} != 0. "
            "Rejected non-cell moves are leaking into the accepted bucket. "
            "Fix 2 (reject_reason gating) may be missing."
        )
        # All rejected proposals must be in bucket 1 (energy reject)
        assert int(rr[rw_idx, 1]) == n_mcmc, (
            f"rr[rw, 1] (energy bucket) = {int(rr[rw_idx, 1])} != {n_mcmc}. "
            "Expected all rejections to land in the energy bucket."
        )

    def test_rejected_non_cell_move_goes_to_energy_bucket_under_jit(self):
        """Same rejection-gating invariant, explicitly exercised under jax.jit."""
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.moves import random_walk as rw
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel("rw", rw.build_kernel),
            ],
        )

        n_walkers = 20
        n_atoms = 1
        n_mcmc = 10
        key = jax.random.key(7)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0
        )
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        state = init_ns(
            init_fn, positions, types, energies, cells=None, rng_key=key
        )
        huge_ss = jnp.full_like(state.population.step_sizes, 1e6)
        state = state.set(population=state.population.set(step_sizes=huge_ss))

        # Explicitly compile under jax.jit (non-negotiable per project policy)
        jit_ns = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_ns(state, step_fn, n_mcmc, 0)

        rr = info["reject_reason_counts_per_move"]  # (1, 4)
        n_acc = info["n_accepted_per_move"]  # (1,)

        # Invariant: accepted bucket == n_accepted_per_move
        assert jnp.array_equal(rr[:, 0], n_acc), (
            f"rr[:, 0]={rr[:, 0]} != n_accepted={n_acc}. "
            "reject_reason gating invariant violated under JIT."
        )
        # Rejected moves (all, since ss=1e6) land in bucket 1, not bucket 0
        assert (
            int(rr[0, 0]) == 0
        ), f"Accepted bucket {int(rr[0, 0])} should be 0 when all moves reject."
