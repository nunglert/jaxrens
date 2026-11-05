"""Tests for _process_rate_jax and adjust_step_size."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.stepsize_handler import (
    _process_rate_jax,
    adjust_step_size,
)
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns


@pytest.fixture(scope="module")
def harmonic_mwg():
    """Harmonic MWG after 50 NS steps, so that Emax constrains proposals."""
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "random_walk",
            random_walk.build_kernel,
            step_size=0.1,
            step_size_max=5.0,
            min_rate=0.2,
            max_rate=0.7,
        ),
    ]
    init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

    n_walkers = 30
    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    positions = 0.5 * jax.random.normal(init_key, (n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    ns_state = init_ns(
        init_fn,
        positions,
        types,
        energies,
        cells=None,
        rng_key=key,
    )

    for _ in range(50):
        ns_state, _ = ns_step(ns_state, step_fn, n_mcmc_steps=5)

    return {
        "backend": backend,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "per_move_fns": per_move_fns,
        "descriptors": descriptors,
        "ns_state": ns_state,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
    }


class TestProcessRateJax:
    """Check the convergence decision for one measured acceptance rate."""

    def test_within_window_converges(self):
        (
            ss,
            converged,
            cap_hit,
            floor_hit,
            too_high,
            too_low,
        ) = _process_rate_jax(
            jnp.array(0.4),
            jnp.array(0.1),
            jnp.array(0.1),
            jnp.array(-1.0),
            0.2,
            0.7,
            1.5,
            10.0,
        )
        assert converged
        assert float(ss) == pytest.approx(0.1)
        assert not cap_hit
        assert not floor_hit
        assert not too_high
        assert not too_low

    def test_below_min_decreases(self):
        (
            ss,
            converged,
            cap_hit,
            floor_hit,
            too_high,
            too_low,
        ) = _process_rate_jax(
            jnp.array(0.1),
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(-1.0),
            0.2,
            0.7,
            1.5,
            10.0,
        )
        assert not converged
        assert float(ss) < 1.0
        assert not too_high
        assert too_low

    def test_above_max_increases(self):
        (
            ss,
            converged,
            cap_hit,
            floor_hit,
            too_high,
            too_low,
        ) = _process_rate_jax(
            jnp.array(0.9),
            jnp.array(0.1),
            jnp.array(0.1),
            jnp.array(-1.0),
            0.2,
            0.7,
            1.5,
            10.0,
        )
        assert not converged
        assert float(ss) > 0.1
        assert too_high
        assert not too_low

    def test_max_enforced_and_cap_hit(self):
        # rate 0.95 > max 0.7: proposed = 9.0 * 1.5 = 13.5 > max_step_size 10
        ss, _, cap_hit, floor_hit, too_high, _ = _process_rate_jax(
            jnp.array(0.95),
            jnp.array(9.0),
            jnp.array(9.0),
            jnp.array(-1.0),
            0.2,
            0.7,
            1.5,
            10.0,
        )
        assert float(ss) <= 10.0
        assert cap_hit
        assert not floor_hit

    def test_floor_hit(self):
        # rate 0.01 < min 0.2: proposed = 1e-20 / 1.5 falls below the floor
        ss, _, cap_hit, floor_hit, _, too_low = _process_rate_jax(
            jnp.array(0.01),
            jnp.array(1e-20),
            jnp.array(1e-20),
            jnp.array(-1.0),
            0.2,
            0.7,
            1.5,
            10.0,
        )
        assert float(ss) > 0.0
        assert floor_hit
        assert not cap_hit
        assert too_low

    @pytest.mark.parametrize(
        "rate,rate_prev,expected_ss",
        [
            # The previous rate (0.15) is closer to the target 0.45.
            (0.8, 0.15, 0.3),
            # The current rate (0.75) is closer.
            (0.75, 0.05, 0.5),
        ],
    )
    def test_bracket_keeps_the_closer_step_size(
        self, rate, rate_prev, expected_ss
    ):
        ss, converged, *_ = _process_rate_jax(
            jnp.array(rate),
            jnp.array(0.5),
            jnp.array(0.3),
            jnp.array(rate_prev),
            0.2,
            0.7,
            1.5,
            10.0,
        )
        assert converged
        assert float(ss) == pytest.approx(expected_ss)

    def test_jit_compatible(self):
        jit_fn = jax.jit(_process_rate_jax, static_argnums=(4, 5, 6, 7))
        ss, converged, cap_hit, floor_hit, too_high, too_low = jit_fn(
            jnp.array(0.4),
            jnp.array(0.1),
            jnp.array(0.1),
            jnp.array(-1.0),
            0.2,
            0.7,
            1.5,
            10.0,
        )
        assert converged
        assert not cap_hit
        assert not floor_hit


class TestAdjustStepSize:
    """Check the bisection that adjusts one move's step size."""

    def test_converges_from_small(self, harmonic_mwg):
        """A tiny step size (high acceptance) grows."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        jit_adjust = jax.jit(
            adjust_step_size,
            static_argnums=(1, 5, 6, 7, 8, 9, 10),
        )
        (
            new_ss,
            rate,
            _,
            n_rounds,
            converged,
            cap_hits,
            floor_hits,
            bracket_detected,
            trial_n_evals,
            trial_n_grad_evals,
        ) = jit_adjust(
            pop,
            s["per_move_fns"][0],
            jnp.array(0.001),
            emax,
            jax.random.key(123),
            30,
            0.2,
            0.7,
            1.5,
            5.0,
            15,
        )
        assert float(new_ss) > 0.001
        assert int(n_rounds) > 0

    def test_converges_to_window(self, harmonic_mwg):
        """With enough samples and rounds the bisection converges."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        jit_adjust = jax.jit(
            adjust_step_size,
            static_argnums=(1, 5, 6, 7, 8, 9, 10),
        )
        _, rate, _, _, converged, *_ = jit_adjust(
            pop,
            s["per_move_fns"][0],
            jnp.array(0.5),
            emax,
            jax.random.key(456),
            100,
            0.2,
            0.7,
            1.5,
            5.0,
            30,
        )
        assert converged
        assert 0.2 <= float(rate) <= 0.7

    def test_n_rounds_bounded_by_max_rounds(self, harmonic_mwg):
        """n_rounds never exceeds max_rounds."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)
        max_rounds = 5

        jit_adjust = jax.jit(
            adjust_step_size,
            static_argnums=(1, 5, 6, 7, 8, 9, 10),
        )
        _, _, _, n_rounds, *_ = jit_adjust(
            pop,
            s["per_move_fns"][0],
            jnp.array(0.5),
            emax,
            jax.random.key(7),
            20,
            0.2,
            0.7,
            1.5,
            5.0,
            max_rounds,
        )
        assert int(n_rounds) <= max_rounds

    def test_cap_hits_when_everything_is_accepted(self, harmonic_mwg):
        """A huge emax accepts every move, so the step size hits the cap."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.asarray(1e6, dtype=pop.energy.dtype)

        jit_adjust = jax.jit(
            adjust_step_size,
            static_argnums=(1, 5, 6, 7, 8, 9, 10),
        )
        new_ss, _, _, _, _, cap_hits, *_ = jit_adjust(
            pop,
            s["per_move_fns"][0],
            jnp.array(4.5),
            emax,
            jax.random.key(99),
            30,
            0.2,
            0.7,
            1.5,
            5.0,
            10,
        )
        assert int(cap_hits) > 0
        assert float(new_ss) <= 5.0

    def test_jit_compiles_and_returns_correct_dtypes(self, harmonic_mwg):
        """Under JIT every output has the expected shape and dtype."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        jit_adjust = jax.jit(
            adjust_step_size,
            static_argnums=(1, 5, 6, 7, 8, 9, 10),
        )
        (
            new_ss,
            final_rate,
            final_counts,
            n_rounds,
            converged,
            cap_hits,
            floor_hits,
            bracket_detected,
            trial_n_evals,
            trial_n_grad_evals,
        ) = jit_adjust(
            pop,
            s["per_move_fns"][0],
            jnp.array(0.3),
            emax,
            jax.random.key(1),
            20,
            0.2,
            0.7,
            1.5,
            5.0,
            10,
        )
        assert new_ss.shape == ()
        assert final_rate.shape == ()
        assert final_counts.shape == (4,)
        assert final_counts.dtype == jnp.int32
        assert n_rounds.shape == ()
        assert n_rounds.dtype == jnp.int32
        assert converged.shape == ()
        assert converged.dtype == jnp.bool_
        assert cap_hits.shape == ()
        assert cap_hits.dtype == jnp.int32
        assert floor_hits.shape == ()
        assert floor_hits.dtype == jnp.int32
        assert bracket_detected.shape == ()
        assert bracket_detected.dtype == jnp.bool_
        assert trial_n_evals.shape == ()
        assert trial_n_evals.dtype == jnp.int32
        assert trial_n_grad_evals.shape == ()
        assert trial_n_grad_evals.dtype == jnp.int32

    def test_vmap_compatible(self, harmonic_mwg):
        """adjust_step_size can be vmapped over rng_key."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        keys = jax.random.split(jax.random.key(42), 3)

        def single(key):
            return adjust_step_size(
                pop,
                s["per_move_fns"][0],
                jnp.array(0.3),
                emax,
                key,
                20,
                0.2,
                0.7,
                1.5,
                5.0,
                8,
            )

        results = jax.vmap(single)(keys)
        assert results[0].shape == (3,)
        assert results[3].shape == (3,)
        assert results[3].dtype == jnp.int32

    def test_trial_batch_size_matches_full_vmap(self, harmonic_mwg):
        """A trial_batch_size of 5 gives the same outputs as a full vmap.

        ``lax.map(batch_size=N)`` is a scan of vmaps, so with the same key,
        population and n_samples the step size and counts are identical.
        """
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)
        common_kwargs = dict(
            population=pop,
            move_fn=s["per_move_fns"][0],
            step_size=jnp.array(0.3),
            emax=emax,
            rng_key=jax.random.key(2026),
            n_samples=20,
            min_rate=0.2,
            max_rate=0.7,
            adjust_factor=1.5,
            max_step_size=5.0,
            max_rounds=10,
        )
        full = adjust_step_size(**common_kwargs)
        chunked = adjust_step_size(**common_kwargs, trial_batch_size=5)
        assert jnp.allclose(full[0], chunked[0])
        assert jnp.allclose(full[1], chunked[1])
        assert jnp.array_equal(full[2], chunked[2])

    def test_trial_batch_size_jit(self, harmonic_mwg):
        """With trial_batch_size, jit and eager calls give the same result."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)
        args = (
            pop,
            s["per_move_fns"][0],
            jnp.array(0.3),
            emax,
            jax.random.key(7),
            20,
            0.2,
            0.7,
            1.5,
            5.0,
            10,
        )

        def _wrapped(*a):
            return adjust_step_size(*a, trial_batch_size=4)

        jit_fn = jax.jit(_wrapped, static_argnums=(1, 5, 6, 7, 8, 9, 10))
        eager = _wrapped(*args)
        jitted = jit_fn(*args)
        assert jnp.allclose(jitted[0], eager[0])
        assert jnp.allclose(jitted[1], eager[1])
        assert jnp.array_equal(jitted[2], eager[2])


class TestAdaptIntegration:
    """Check run_ns with per-move step-size adaptation switched on."""

    def test_run_ns_with_full_auto(self, harmonic_mwg):
        """run_ns with per_move_fns and adjust_interval completes."""
        s = harmonic_mwg
        result = run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=50,
            n_mcmc_steps=5,
            per_move_fns=s["per_move_fns"],
            move_descriptors=s["descriptors"],
            adjust_interval=20,
            adjust_n_samples=15,
        )
        assert result["iteration"] > 0
        assert result["n_dead"] > 0
        assert jnp.isfinite(result["log_evidence"])
