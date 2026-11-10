"""Tests for the energy and gradient evaluation counters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.base import MoveInfo
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import galilean, hmc, random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns


@pytest.fixture
def harmonic_rw_setup():
    """Harmonic oscillator with random_walk only."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("random_walk", random_walk.build_kernel),
        ],
    )
    n_walkers = 20
    key = jax.random.key(77)
    key, init_key = jax.random.split(key)
    positions = 0.5 * jax.random.normal(init_key, (n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    ns_state = init_ns(
        init_fn, positions, types, energies, cells=None, rng_key=key
    )
    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "ns_state": ns_state,
        "key": key,
    }


@pytest.fixture
def galilean_setup():
    """Harmonic oscillator with a galilean move (n_reflect=5)."""
    n_reflect = 5
    backend = create_harmonic(k=1.0)

    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel(
                "galilean",
                galilean.build_kernel,
                kernel_kwargs={"n_reflect": n_reflect, "use_forces": True},
                extra_state_fields={
                    "direction": (
                        jnp.ndarray,
                        lambda pos, types, cell: jnp.zeros_like(pos),
                    )
                },
            ),
        ],
    )
    n_walkers = 20
    key = jax.random.key(88)
    key, init_key = jax.random.split(key)
    positions = 0.1 * jax.random.normal(init_key, (n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    ns_state = init_ns(
        init_fn, positions, types, energies, cells=None, rng_key=key
    )
    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "ns_state": ns_state,
        "key": key,
        "n_reflect": n_reflect,
    }


class TestMoveInfoDefault:
    """Check the n_grad_evaluations field of MoveInfo."""

    def test_n_grad_evaluations_defaults_to_zero(self):
        """MoveInfo without n_grad_evaluations kwarg should default to 0."""
        info = MoveInfo(
            accepted=jnp.array(True),
            log_likelihood=jnp.array(-1.0),
            n_evaluations=jnp.int32(1),
        )
        assert int(info.n_grad_evaluations) == 0
        assert info.n_grad_evaluations.dtype == jnp.int32


class TestRandomWalkCounters:
    """Check that random_walk counts one energy and no gradient call."""

    @pytest.mark.parametrize("n_cull,n_extra", [(1, 0), (1, 3), (4, 3)])
    def test_rw_eval_counts_in_info(self, harmonic_rw_setup, n_cull, n_extra):
        """One evaluation per walk step for each of n_cull + n_extra walkers.

        No gradient evaluations are counted.
        """
        s = harmonic_rw_setup
        n_mcmc_steps = 10

        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3, 4))
        _, info = jit_step(
            s["ns_state"], s["step_fn"], n_mcmc_steps, n_extra, n_cull
        )

        n_e = np.asarray(info["n_evaluations_per_move"])
        n_g = np.asarray(info["n_grad_evaluations_per_move"])
        assert n_e.shape == (1,)
        assert n_g.shape == (1,)
        assert int(n_e[0]) == (n_cull + n_extra) * n_mcmc_steps
        assert int(n_g[0]) == 0


class TestGalileanCounters:
    """Check that galilean counts n_reflect value_and_grad calls per step."""

    def test_galilean_eval_counts(self, galilean_setup):
        """Every reflection is a value_and_grad call: n_reflect per step."""
        s = galilean_setup
        n_mcmc_steps = 4
        n_walk = 1

        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], n_mcmc_steps, 0)

        n_e = int(info["n_evaluations_per_move"][0])
        n_g = int(info["n_grad_evaluations_per_move"][0])

        expected = s["n_reflect"] * n_walk * n_mcmc_steps
        assert n_e == expected
        assert n_g == expected


class TestMixedMWGCounters:
    """Check the per-move counters of a galilean + random_walk MWG."""

    @pytest.fixture
    def mixed_setup(self):
        n_reflect = 5
        backend = create_harmonic(k=1.0)

        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel(
                    "galilean",
                    galilean.build_kernel,
                    kernel_kwargs={"n_reflect": n_reflect, "use_forces": True},
                    extra_state_fields={
                        "direction": (
                            jnp.ndarray,
                            lambda pos, types, cell: jnp.zeros_like(pos),
                        )
                    },
                ),  # idx 0
                MoveKernel("random_walk", random_walk.build_kernel),  # idx 1
            ],
        )
        n_walkers = 20
        key = jax.random.key(99)
        key, init_key = jax.random.split(key)
        positions = 0.1 * jax.random.normal(init_key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)
        ns_state = init_ns(
            init_fn, positions, types, energies, cells=None, rng_key=key
        )
        return {
            "init_fn": init_fn,
            "step_fn": step_fn,
            "positions": positions,
            "types": types,
            "energies": energies,
            "ns_state": ns_state,
            "key": key,
            "n_reflect": n_reflect,
        }

    def test_mixed_eval_shapes(self, mixed_setup):
        """With 2 moves, eval-count arrays have shape (2,)."""
        s = mixed_setup
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], 10, 0)
        assert info["n_evaluations_per_move"].shape == (2,)
        assert info["n_grad_evaluations_per_move"].shape == (2,)

    def test_random_walk_grad_evals_zero_in_mixed(self, mixed_setup):
        """In a mixed MWG, random_walk counts evaluations but no gradients."""
        s = mixed_setup
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(s["ns_state"], s["step_fn"], 20, 0)
        assert int(info["n_evaluations_per_move"][1]) > 0
        assert int(info["n_grad_evaluations_per_move"][1]) == 0

    def test_galilean_grad_evals_positive_in_mixed(self, mixed_setup):
        """Galilean accumulates gradient evaluations over 100 MCMC steps."""
        s = mixed_setup
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        ns_state = s["ns_state"]
        total_g_evals = 0
        for _ in range(10):
            ns_state, info = jit_step(ns_state, s["step_fn"], 10, 0)
            total_g_evals += int(info["n_grad_evaluations_per_move"][0])
        assert total_g_evals > 0


class TestCumulativeCounters:
    """Check the cumulative counters that run_ns adds to info."""

    def test_cumulative_grows_monotonically(self, harmonic_rw_setup):
        """The summed cumulative evaluation count never decreases."""
        s = harmonic_rw_setup
        seen_values = []

        def capture_cb(iteration, ns_state, info):
            cum = info.get("cumulative_n_evaluations_per_move")
            if cum is not None:
                seen_values.append(int(np.asarray(cum).sum()))

        class _CB:
            def on_iteration(self, iteration, ns_state, info):
                capture_cb(iteration, ns_state, info)

            def on_dead_point(self, *a):
                pass

            def on_finish(self, *a):
                pass

        run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=10,
            n_mcmc_steps=5,
            callbacks=[_CB()],
        )
        assert len(seen_values) > 0
        assert np.all(np.diff(seen_values) >= 0)

    def test_cumulative_random_walk_has_no_grad_evals(self, harmonic_rw_setup):
        """Random walk ends with evaluations > 0 and zero gradient ones."""
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

            def on_dead_point(self, *a):
                pass

            def on_finish(self, *a):
                pass

        run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=5,
            n_mcmc_steps=3,
            callbacks=[_CB()],
        )
        assert final_e[0] is not None
        assert int(final_g[0].sum()) == 0
        assert int(final_e[0].sum()) > 0


class TestTrialEvalCounters:
    """Check the trial counters emitted on step-size adjustment iterations."""

    def test_trial_n_evaluations_on_adjust_iter(self, harmonic_rw_setup):
        """Adjustment iterations report positive trial evaluation counts."""
        s = harmonic_rw_setup
        found_trial = [False]

        class _CB:
            def on_iteration(self, iteration, ns_state, info):
                trial_e = info.get("trial_n_evaluations_per_move")
                if trial_e is not None:
                    assert int(np.asarray(trial_e).sum()) > 0
                    found_trial[0] = True

            def on_dead_point(self, *a):
                pass

            def on_finish(self, *a):
                pass

        desc = [
            MoveKernel(
                "random_walk",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            )
        ]
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, per_move_fns = build_mwg(backend, desc)
        run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=s["key"],
            max_iterations=30,
            n_mcmc_steps=3,
            per_move_fns=per_move_fns,
            move_descriptors=desc,
            adjust_interval=10,
            adjust_n_samples=5,
            callbacks=[_CB()],
        )
        assert found_trial[0]


def test_hmc_eval_counts():
    """HMC counts 2*n_leapfrog+1 evaluations, 2*n_leapfrog of them grads."""
    backend = create_harmonic(k=1.0)
    n_leapfrog = 4

    def build_hmc_kernel(be):
        return hmc.build_kernel(be, n_leapfrog=n_leapfrog)

    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("hmc", build_hmc_kernel),
        ],
    )
    n_walkers = 10
    key = jax.random.key(55)
    key, init_key = jax.random.split(key)
    positions = 0.1 * jax.random.normal(init_key, (n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    ns_state = init_ns(
        init_fn, positions, types, energies, cells=None, rng_key=key
    )

    jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
    n_mcmc_steps = 3
    _, info = jit_step(ns_state, step_fn, n_mcmc_steps, 0)

    n_e = int(info["n_evaluations_per_move"][0])
    n_g = int(info["n_grad_evaluations_per_move"][0])
    n_walk = 1

    expected_evals = (2 * n_leapfrog + 1) * n_walk * n_mcmc_steps
    expected_grad_evals = (2 * n_leapfrog) * n_walk * n_mcmc_steps
    assert n_e == expected_evals, f"n_evals={n_e}, expected={expected_evals}"
    assert (
        n_g == expected_grad_evals
    ), f"n_grad_evals={n_g}, expected={expected_grad_evals}"
