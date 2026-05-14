"""Tests for the pure-function step size adjustment helpers.

Verifies:
- _process_rate_jax produces correct convergence decisions
- run_ns with full-auto adaptation completes end-to-end
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.stepsize_handler import _process_rate_jax
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
        cells=None, rng_key=key,
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


class TestAdaptIntegration:
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
