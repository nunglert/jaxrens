"""Tests for the split energy / gradient evaluation counters (Task 2).

Verifies:
- MoveInfo.n_grad_evaluations field exists with correct default (0)
- Per-move n_evaluations and n_grad_evaluations are correctly populated in ns_step info
- Galilean: n_evaluations == n_grad_evaluations (all reflect calls use value_and_grad)
- Random-walk: n_grad_evaluations == 0
- Cumulative counters in run_ns grow monotonically
- trial_n_evaluations_per_move / trial_n_grad_evaluations_per_move emitted on adjust iters

The AdaptationLog v3 schema round-trip (which folds in n_evaluations /
n_grad_evaluations) lives in tests/log_io/test_adaptation_log.py alongside
the v1/v2 schema tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.base import MoveInfo
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import galilean, random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_rw_setup():
    """Harmonic oscillator with random_walk only."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [
        MoveKernel("random_walk", random_walk.build_kernel),
    ])
    n_walkers = 20
    key = jax.random.key(77)
    key, init_key = jax.random.split(key)
    positions = 0.5 * jax.random.normal(init_key, (n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)
    return {
        "init_fn": init_fn, "step_fn": step_fn,
        "positions": positions, "types": types,
        "energies": energies, "ns_state": ns_state, "key": key,
    }


@pytest.fixture
def galilean_setup():
    """Harmonic oscillator with galilean move (n_reflect=5).

    Galilean requires a 'direction' extra state field — pass it via MoveKernel.
    """
    n_reflect = 5
    backend = create_harmonic(k=1.0)

    init_fn, step_fn, _ = build_mwg(backend, [
        MoveKernel(
            "galilean",
            galilean.build_kernel,
            kernel_kwargs={"n_reflect": n_reflect, "use_forces": True},
            extra_state_fields={
                "direction": (jnp.ndarray, lambda pos, types: jnp.zeros_like(pos))
            },
        ),
    ])
    n_walkers = 20
    key = jax.random.key(88)
    key, init_key = jax.random.split(key)
    positions = 0.1 * jax.random.normal(init_key, (n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)
    return {
        "init_fn": init_fn, "step_fn": step_fn,
        "positions": positions, "types": types,
        "energies": energies, "ns_state": ns_state, "key": key,
        "n_reflect": n_reflect,
    }


# ---------------------------------------------------------------------------
# 1. MoveInfo default field
# ---------------------------------------------------------------------------


class TestMoveInfoDefault:
    def test_n_grad_evaluations_defaults_to_zero(self):
        """MoveInfo without n_grad_evaluations kwarg should default to 0."""
        info = MoveInfo(
            accepted=jnp.array(True),
            log_likelihood=jnp.array(-1.0),
            n_evaluations=jnp.int32(1),
        )
        assert int(info.n_grad_evaluations) == 0
        assert info.n_grad_evaluations.dtype == jnp.int32

    def test_n_grad_evaluations_explicit(self):
        """MoveInfo with explicit n_grad_evaluations stores it correctly."""
        info = MoveInfo(
            accepted=jnp.array(True),
            log_likelihood=jnp.array(-2.0),
            n_evaluations=jnp.int32(5),
            n_grad_evaluations=jnp.int32(5),
        )
        assert int(info.n_grad_evaluations) == 5

    def test_n_grad_evaluations_dtype(self):
        """n_grad_evaluations should be int32."""
        info = MoveInfo(
            accepted=jnp.array(False),
            log_likelihood=jnp.array(-3.0),
            n_evaluations=jnp.int32(1),
            n_grad_evaluations=jnp.int32(0),
        )
        assert info.n_grad_evaluations.dtype == jnp.int32


# ---------------------------------------------------------------------------
# 2. Random-walk: n_evaluations=1, n_grad_evaluations=0
# ---------------------------------------------------------------------------


class TestRandomWalkCounters:
    def test_rw_eval_counts_in_info(self, harmonic_rw_setup):
        """Random-walk ns_step info: n_evaluations_per_move=1 per chain step per walker,
        n_grad_evaluations_per_move=0."""
        s = harmonic_rw_setup
        n_mcmc_steps = 10
        n_walk = 1  # n_extra=0

        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], n_mcmc_steps, 0)

        assert "n_evaluations_per_move" in info
        assert "n_grad_evaluations_per_move" in info
        n_e = np.asarray(info["n_evaluations_per_move"])
        n_g = np.asarray(info["n_grad_evaluations_per_move"])
        assert n_e.shape == (1,)  # 1 move type
        assert n_g.shape == (1,)
        # random_walk: 1 eval per step * n_walk * n_mcmc_steps
        assert int(n_e[0]) == n_walk * n_mcmc_steps
        assert int(n_g[0]) == 0

    def test_rw_grad_evals_always_zero_under_jit(self, harmonic_rw_setup):
        """Under JIT: n_grad_evaluations_per_move[0] == 0 (no grad calls in RW)."""
        s = harmonic_rw_setup
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], 5, 0)
        assert int(info["n_grad_evaluations_per_move"][0]) == 0


# ---------------------------------------------------------------------------
# 3. Galilean: n_evaluations == n_grad_evaluations == n_reflect per chain step
# ---------------------------------------------------------------------------


class TestGalileanCounters:
    def test_galilean_eval_counts(self, galilean_setup):
        """Galilean ns_step info: n_evaluations_per_move == n_grad_evaluations_per_move
        == n_reflect * n_walk * n_mcmc_steps (all reflect calls use value_and_grad)."""
        s = galilean_setup
        n_mcmc_steps = 4
        n_walk = 1

        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], n_mcmc_steps, 0)

        n_e = int(info["n_evaluations_per_move"][0])
        n_g = int(info["n_grad_evaluations_per_move"][0])

        expected = s["n_reflect"] * n_walk * n_mcmc_steps
        assert n_e == expected, f"n_evaluations={n_e}, expected={expected}"
        assert n_g == expected, f"n_grad_evaluations={n_g}, expected={expected}"
        # galilean: every eval is a value_and_grad call
        assert n_e == n_g

    def test_galilean_all_evals_are_grad_evals_under_jit(self, galilean_setup):
        """Under JIT: galilean n_evaluations == n_grad_evaluations."""
        s = galilean_setup
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], 3, 0)
        assert int(info["n_evaluations_per_move"][0]) == int(
            info["n_grad_evaluations_per_move"][0]
        )


# ---------------------------------------------------------------------------
# 4. Mixed MWG (galilean + random_walk)
# ---------------------------------------------------------------------------


class TestMixedMWGCounters:
    @pytest.fixture
    def mixed_setup(self):
        n_reflect = 5
        backend = create_harmonic(k=1.0)

        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel(
                "galilean",
                galilean.build_kernel,
                kernel_kwargs={"n_reflect": n_reflect, "use_forces": True},
                extra_state_fields={
                    "direction": (jnp.ndarray, lambda pos, types: jnp.zeros_like(pos))
                },
            ),  # idx 0
            MoveKernel("random_walk", random_walk.build_kernel),  # idx 1
        ])
        n_walkers = 20
        key = jax.random.key(99)
        key, init_key = jax.random.split(key)
        positions = 0.1 * jax.random.normal(init_key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)
        ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)
        return {
            "init_fn": init_fn, "step_fn": step_fn,
            "positions": positions, "types": types, "energies": energies,
            "ns_state": ns_state, "key": key, "n_reflect": n_reflect,
        }

    def test_mixed_eval_shapes(self, mixed_setup):
        """With 2 moves, eval-count arrays have shape (2,)."""
        s = mixed_setup
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], 10, 0)
        assert info["n_evaluations_per_move"].shape == (2,)
        assert info["n_grad_evaluations_per_move"].shape == (2,)

    def test_random_walk_grad_evals_zero_in_mixed(self, mixed_setup):
        """In a mixed MWG, random_walk's n_grad_evaluations_per_move == 0."""
        s = mixed_setup
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], 20, 0)
        # Index 1 is random_walk
        assert int(info["n_grad_evaluations_per_move"][1]) == 0

    def test_galilean_grad_evals_positive_in_mixed(self, mixed_setup):
        """In a mixed MWG, galilean's n_grad_evaluations_per_move > 0 when it runs."""
        s = mixed_setup
        # Run enough steps so galilean is likely chosen at least once
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        # Run several iterations to accumulate some galilean draws
        ns_state = s["ns_state"]
        total_g_evals = 0
        for _ in range(10):
            ns_state, info = jit_step(ns_state, s["step_fn"], 10, 0)
            total_g_evals += int(info["n_grad_evaluations_per_move"][0])
        # After 100 MCMC steps total, galilean (50% weight) should have fired
        assert total_g_evals > 0


# ---------------------------------------------------------------------------
# 5. Cumulative counters in run_ns
# ---------------------------------------------------------------------------


class TestCumulativeCounters:
    def test_cumulative_grows_monotonically(self, harmonic_rw_setup):
        """cumulative_n_evaluations_per_move should strictly increase each iter."""
        s = harmonic_rw_setup
        seen_values = []

        def capture_cb(iteration, ns_state, info):
            cum = info.get("cumulative_n_evaluations_per_move")
            if cum is not None:
                seen_values.append(int(np.asarray(cum).sum()))

        class _CB:
            def on_iteration(self, iteration, ns_state, info):
                capture_cb(iteration, ns_state, info)
            def on_dead_point(self, *a): pass
            def on_finish(self, *a): pass

        run_ns(
            s["positions"], s["types"], s["energies"], cells=None,
            init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=10,
            n_mcmc_steps=5,
            callbacks=[_CB()],
        )
        assert len(seen_values) > 0
        # Strictly monotonically non-decreasing
        for i in range(1, len(seen_values)):
            assert seen_values[i] >= seen_values[i - 1], \
                f"cumulative decreased at step {i}: {seen_values[i-1]} -> {seen_values[i]}"

    def test_cumulative_grad_always_leq_evals(self, harmonic_rw_setup):
        """cumulative_n_grad_evaluations_per_move <= cumulative_n_evaluations_per_move."""
        s = harmonic_rw_setup
        final_e = [None]
        final_g = [None]

        class _CB:
            def on_iteration(self, iteration, ns_state, info):
                e = info.get("cumulative_n_evaluations_per_move")
                g = info.get("cumulative_n_grad_evaluations_per_move")
                if e is not None:
                    final_e[0] = np.asarray(e)
                    final_g[0] = np.asarray(g)
            def on_dead_point(self, *a): pass
            def on_finish(self, *a): pass

        run_ns(
            s["positions"], s["types"], s["energies"], cells=None,
            init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=5,
            n_mcmc_steps=3,
            callbacks=[_CB()],
        )
        assert final_e[0] is not None
        # For random_walk only: grad evals are always 0
        assert int(final_g[0].sum()) == 0
        assert int(final_e[0].sum()) > 0


# ---------------------------------------------------------------------------
# 6. Trial eval counts on adjust iterations
# ---------------------------------------------------------------------------


class TestTrialEvalCounters:
    def test_trial_n_evaluations_on_adjust_iter(self, harmonic_rw_setup):
        """On adjustment iterations, trial_n_evaluations_per_move is in info and > 0."""
        s = harmonic_rw_setup
        found_trial = [False]

        class _CB:
            def on_iteration(self, iteration, ns_state, info):
                trial_e = info.get("trial_n_evaluations_per_move")
                if trial_e is not None:
                    assert int(np.asarray(trial_e).sum()) > 0
                    found_trial[0] = True
            def on_dead_point(self, *a): pass
            def on_finish(self, *a): pass

        desc = [MoveKernel(
            "random_walk", random_walk.build_kernel,
            step_size=0.1, step_size_max=5.0, min_rate=0.2, max_rate=0.7,
        )]
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, per_move_fns = build_mwg(backend, desc)
        run_ns(
            s["positions"], s["types"], s["energies"], cells=None,
            init_fn=init_fn, step_fn=step_fn,
            rng_key=s["key"],
            max_iterations=30,
            n_mcmc_steps=3,
            per_move_fns=per_move_fns,
            move_descriptors=desc,
            adjust_interval=10,
            adjust_n_samples=5,
            callbacks=[_CB()],
        )
        assert found_trial[0], "trial_n_evaluations_per_move was never set"


# ---------------------------------------------------------------------------
# HMC: 2*n_leapfrog+1 evals, 2*n_leapfrog grad evals
# ---------------------------------------------------------------------------


class TestHMCCounters:
    def test_hmc_eval_counts(self):
        """HMC kernel: n_evaluations = 2*n_leapfrog+1, n_grad_evaluations = 2*n_leapfrog."""
        from jaxrens.sampling.moves import hmc

        backend = create_harmonic(k=1.0)
        n_leapfrog = 4

        def build_hmc_kernel(be):
            return hmc.build_kernel(be, n_leapfrog=n_leapfrog)

        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("hmc", build_hmc_kernel),
        ])
        n_walkers = 10
        key = jax.random.key(55)
        key, init_key = jax.random.split(key)
        positions = 0.1 * jax.random.normal(init_key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)
        ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)

        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        n_mcmc_steps = 3
        _, info = jit_step(ns_state, step_fn, n_mcmc_steps, 0)

        n_e = int(info["n_evaluations_per_move"][0])
        n_g = int(info["n_grad_evaluations_per_move"][0])
        n_walk = 1

        expected_evals = (2 * n_leapfrog + 1) * n_walk * n_mcmc_steps
        expected_grad_evals = (2 * n_leapfrog) * n_walk * n_mcmc_steps
        assert n_e == expected_evals, f"n_evals={n_e}, expected={expected_evals}"
        assert n_g == expected_grad_evals, f"n_grad_evals={n_g}, expected={expected_grad_evals}"
