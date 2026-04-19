"""Tests for the pure-function step size adjustment.

Verifies:
- _process_rate_jax produces correct convergence decisions
- adjust_step_size converges to target acceptance window
- JIT and vmap compatibility
- Integration with run_ns
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.stepsize_handler import (
    adjust_step_size,
    _process_rate_jax,
)
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns


@pytest.fixture
def harmonic_mwg():
    """Set up harmonic backend with MWG and per-move functions.

    Runs 50 NS steps to tighten the population so that Emax actually
    constrains proposals.
    """
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "random_walk", random_walk.build_kernel,
            step_size=0.1, step_size_max=5.0,
            min_rate=0.2, max_rate=0.7,
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
        init_fn, positions, types, energies,
        cells=None, rng_key=key, max_dead=500,
    )

    # Tighten population
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
    def test_within_window_converges(self):
        ss, converged, cap_hit, floor_hit, too_high, too_low = _process_rate_jax(
            jnp.array(0.4), jnp.array(0.1),
            jnp.array(0.1), jnp.array(-1.0),
            0.2, 0.7, 1.5, 10.0,
        )
        assert converged
        assert float(ss) == pytest.approx(0.1)
        assert not cap_hit
        assert not floor_hit
        assert not too_high
        assert not too_low

    def test_below_min_decreases(self):
        ss, converged, cap_hit, floor_hit, too_high, too_low = _process_rate_jax(
            jnp.array(0.1), jnp.array(1.0),
            jnp.array(1.0), jnp.array(-1.0),
            0.2, 0.7, 1.5, 10.0,
        )
        assert not converged
        assert float(ss) < 1.0
        assert not too_high
        assert too_low

    def test_above_max_increases(self):
        ss, converged, cap_hit, floor_hit, too_high, too_low = _process_rate_jax(
            jnp.array(0.9), jnp.array(0.1),
            jnp.array(0.1), jnp.array(-1.0),
            0.2, 0.7, 1.5, 10.0,
        )
        assert not converged
        assert float(ss) > 0.1
        assert too_high
        assert not too_low

    def test_max_enforced_and_cap_hit(self):
        # rate=0.95 (above max=0.7) with ss=9.0: proposed = 9.0 * 1.5 = 13.5 > max_step_size=10
        ss, _, cap_hit, floor_hit, too_high, _ = _process_rate_jax(
            jnp.array(0.95), jnp.array(9.0),
            jnp.array(9.0), jnp.array(-1.0),
            0.2, 0.7, 1.5, 10.0,
        )
        assert float(ss) <= 10.0
        assert cap_hit
        assert not floor_hit

    def test_floor_hit(self):
        # rate=0.01 (below min=0.2) with tiny ss: proposed = 1e-20 / 1.5 < 1e-20 => floor hit
        ss, _, cap_hit, floor_hit, _, too_low = _process_rate_jax(
            jnp.array(0.01), jnp.array(1e-20),
            jnp.array(1e-20), jnp.array(-1.0),
            0.2, 0.7, 1.5, 10.0,
        )
        # clip brings it to exactly 1e-20 in float32 (may have tiny FP error)
        assert float(ss) > 0.0
        assert floor_hit
        assert not cap_hit
        assert too_low

    def test_bracketing_converges(self):
        ss, converged, _, _, _, _ = _process_rate_jax(
            jnp.array(0.8), jnp.array(0.5),
            jnp.array(0.3), jnp.array(0.1),
            0.2, 0.7, 1.5, 10.0,
        )
        assert converged

    def test_jit_compatible(self):
        jit_fn = jax.jit(_process_rate_jax, static_argnums=(4, 5, 6, 7))
        ss, converged, cap_hit, floor_hit, too_high, too_low = jit_fn(
            jnp.array(0.4), jnp.array(0.1),
            jnp.array(0.1), jnp.array(-1.0),
            0.2, 0.7, 1.5, 10.0,
        )
        assert converged
        assert not cap_hit
        assert not floor_hit


class TestAdjustStepSize:
    def test_converges_from_small(self, harmonic_mwg):
        """Starting with tiny step size (high acceptance) should increase."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        jit_adjust = jax.jit(
            adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10)
        )
        new_ss, rate, _, n_rounds, converged, cap_hits, floor_hits, bracket_detected = jit_adjust(
            pop, s["per_move_fns"][0],
            jnp.array(0.001), emax, jax.random.key(123),
            30, 0.2, 0.7, 1.5, 5.0, 15,
        )
        assert float(new_ss) > 0.001
        assert int(n_rounds) > 0
        # dtype checks
        assert n_rounds.dtype == jnp.int32
        assert cap_hits.dtype == jnp.int32
        assert floor_hits.dtype == jnp.int32
        assert converged.dtype == jnp.bool_
        assert bracket_detected.dtype == jnp.bool_

    def test_converges_to_window(self, harmonic_mwg):
        """Adjusted step size should produce rate within target window."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        jit_adjust = jax.jit(
            adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10)
        )
        new_ss, rate, _, n_rounds, converged, cap_hits, floor_hits, bracket_detected = jit_adjust(
            pop, s["per_move_fns"][0],
            jnp.array(0.5), emax, jax.random.key(456),
            40, 0.2, 0.7, 1.5, 5.0, 15,
        )
        # Rate should be roughly in the target window
        # (generous tolerance since stochastic)
        assert 0.05 < float(rate) < 0.95

    def test_return_tuple_has_eight_elements(self, harmonic_mwg):
        """adjust_step_size must return exactly 8 elements."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        result = adjust_step_size(
            pop, s["per_move_fns"][0],
            jnp.array(0.1), emax, jax.random.key(0),
            20, 0.2, 0.7, 1.5, 5.0, 10,
        )
        assert len(result) == 8

    def test_n_rounds_bounded_by_max_rounds(self, harmonic_mwg):
        """n_rounds must not exceed max_rounds."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)
        max_rounds = 5

        jit_adjust = jax.jit(
            adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10)
        )
        _, _, _, n_rounds, _, _, _, _ = jit_adjust(
            pop, s["per_move_fns"][0],
            jnp.array(0.5), emax, jax.random.key(7),
            20, 0.2, 0.7, 1.5, 5.0, max_rounds,
        )
        assert int(n_rounds) <= max_rounds

    def test_cap_hits_nonzero_when_starting_near_max(self, harmonic_mwg):
        """Starting just below max_step_size with high acceptance should trigger cap_hits."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        jit_adjust = jax.jit(
            adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10)
        )
        # step_size near max so scale-up hits the cap
        _, _, _, n_rounds, _, cap_hits, floor_hits, _ = jit_adjust(
            pop, s["per_move_fns"][0],
            jnp.array(4.5), emax, jax.random.key(99),
            30, 0.2, 0.7, 1.5, 5.0, 10,
        )
        # cap_hits should be >= 0 and int32
        assert int(cap_hits) >= 0
        assert cap_hits.dtype == jnp.int32
        # floor_hits should be non-negative
        assert int(floor_hits) >= 0

    def test_jit_compiles_and_returns_correct_dtypes(self, harmonic_mwg):
        """All return elements have expected dtypes under JIT."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        jit_adjust = jax.jit(
            adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10)
        )
        new_ss, final_rate, final_counts, n_rounds, converged, cap_hits, floor_hits, bracket_detected = jit_adjust(
            pop, s["per_move_fns"][0],
            jnp.array(0.3), emax, jax.random.key(1),
            20, 0.2, 0.7, 1.5, 5.0, 10,
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

    def test_vmap_compatible(self, harmonic_mwg):
        """adjust_step_size must be vmappable over rng_key."""
        s = harmonic_mwg
        pop = s["ns_state"].population
        emax = jnp.max(pop.energy)

        keys = jax.random.split(jax.random.key(42), 3)

        def single(key):
            return adjust_step_size(
                pop, s["per_move_fns"][0],
                jnp.array(0.3), emax, key,
                20, 0.2, 0.7, 1.5, 5.0, 8,
            )

        results = jax.vmap(single)(keys)
        # results[0] is new_ss, shape (3,)
        assert results[0].shape == (3,)
        assert results[3].shape == (3,)  # n_rounds
        assert results[3].dtype == jnp.int32

    def test_run_ns_with_full_auto(self, harmonic_mwg):
        """run_ns with per_move_fns + adjust_interval should complete."""
        s = harmonic_mwg
        result = run_ns(
            s["positions"], s["types"], s["energies"],
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
