"""Tests for the step-size adaptation closure from build_adapt_step."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import SingleRun, VmapRuns
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel, ns_step


@pytest.fixture(scope="module")
def harmonic_setup():
    """Harmonic backend with two random-walk moves and 20 walkers.

    Returns a dict with a single-run ``ns_state``, a two-run
    ``ns_states_batched``, ``per_move_fns``, ``descriptors``, ``n_runs``,
    ``n_walkers`` and ``step_fn``.
    """
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw_1",
            random_walk.build_kernel,
            step_size=0.2,
            step_size_max=5.0,
            min_rate=0.2,
            max_rate=0.7,
        ),
        MoveKernel(
            "rw_2",
            random_walk.build_kernel,
            step_size=0.3,
            step_size_max=5.0,
            min_rate=0.2,
            max_rate=0.7,
        ),
    ]
    init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

    n_walkers = 20
    n_runs = 2
    key = jax.random.key(0)
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
    for _ in range(20):
        ns_state, _ = ns_step(ns_state, step_fn, n_mcmc_steps=5)

    key2 = jax.random.key(1)
    run_keys = jax.random.split(key2, n_runs)
    pos_batch = jnp.broadcast_to(
        positions[None, :, :, :],
        (n_runs, n_walkers, 1, 3),
    )
    energies_batch = jnp.broadcast_to(energies[None, :], (n_runs, n_walkers))
    ns_states_batched = init_ns_parallel(
        init_fn,
        pos_batch,
        types,
        energies_batch,
        cells=None,
        rng_keys=run_keys,
    )

    return {
        "ns_state": ns_state,
        "ns_states_batched": ns_states_batched,
        "per_move_fns": per_move_fns,
        "descriptors": descriptors,
        "n_runs": n_runs,
        "n_walkers": n_walkers,
        "step_fn": step_fn,
    }


class TestInactiveReturnsNone:
    """Check that build_adapt_step returns None when adaptation is off.

    That happens with no per_move_fns, no descriptors or adjust_interval <= 0.
    """

    @pytest.mark.parametrize(
        "use_descriptors,per_move_fns,interval",
        [
            pytest.param(True, None, 50, id="none_fns"),
            pytest.param(True, [], 50, id="empty_fns"),
            pytest.param(False, "setup", 50, id="no_descriptors"),
            pytest.param(True, "setup", 0, id="zero_interval"),
        ],
    )
    def test_inactive(
        self, harmonic_setup, use_descriptors, per_move_fns, interval
    ):
        if per_move_fns == "setup":
            per_move_fns = harmonic_setup["per_move_fns"]
        adapt = build_adapt_step(
            move_descriptors=(
                harmonic_setup["descriptors"] if use_descriptors else []
            ),
            per_move_fns=per_move_fns,
            batcher=SingleRun(),
            adjust_n_samples=10,
            adjust_factor=1.5,
            adjust_max_rounds=5,
            adjust_interval=interval,
        )
        assert adapt is None


class TestSingleRun:
    """Check the diagnostics, step sizes and rng key of a single run."""

    def _build(self, harmonic_setup):
        return build_adapt_step(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )

    def test_diag_keys_present(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(10)
        _, diag, _ = adapt(ns_state, emax, key)
        assert set(diag.keys()) == {
            "rate",
            "counts",
            "n_rounds",
            "converged",
            "cap_hits",
            "floor_hits",
            "bracket_detected",
            "trial_n_evaluations",
            "trial_n_grad_evaluations",
        }

    def test_diag_shapes(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(11)
        _, diag, _ = adapt(ns_state, emax, key)
        n_moves = 2
        assert diag["rate"].shape == (n_moves,)
        assert diag["counts"].shape == (n_moves, 4)
        for k in (
            "n_rounds",
            "converged",
            "cap_hits",
            "floor_hits",
            "bracket_detected",
            "trial_n_evaluations",
            "trial_n_grad_evaluations",
        ):
            assert diag[k].shape == (n_moves,), k

    def test_step_sizes_written_back(self, harmonic_setup):
        """The new step sizes are written into ``population.step_sizes``."""
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(12)
        new_ns_state, _, _ = adapt(ns_state, emax, key)
        ss_pop = new_ns_state.population.step_sizes
        assert ss_pop.shape == ns_state.population.step_sizes.shape
        assert jnp.all(ss_pop > 0.0)
        assert jnp.all(jnp.isfinite(ss_pop))
        first_row = ss_pop[0]
        assert jnp.all(ss_pop == first_row[None, :])

    def test_rates_in_unit_interval(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(13)
        _, diag, _ = adapt(ns_state, emax, key)
        assert jnp.all(diag["rate"] >= 0.0)
        assert jnp.all(diag["rate"] <= 1.0)

    def test_rng_key_advances(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key_in = jax.random.key(14)
        _, _, key_out = adapt(ns_state, emax, key_in)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(key_in)),
            np.asarray(jax.random.key_data(key_out)),
        )

    @pytest.mark.parametrize(
        "start,grows",
        [(50.0, False), (1e-4, True)],
        ids=["too_large_shrinks", "too_small_grows"],
    )
    def test_step_size_moves_towards_target_rate(
        self, harmonic_setup, start, grows
    ):
        adapt = self._build(harmonic_setup)
        ns_state = harmonic_setup["ns_state"]
        pop = ns_state.population
        ns_state = ns_state.set(
            population=pop.set(step_sizes=jnp.full_like(pop.step_sizes, start))
        )
        emax = jnp.max(ns_state.population.energy)
        new_state, _, _ = adapt(ns_state, emax, jax.random.key(15))
        new_ss = new_state.population.step_sizes[0]
        if grows:
            assert jnp.all(new_ss > start)
        else:
            assert jnp.all(new_ss < start)

    def test_step_size_max_caps_growth(self, harmonic_setup):
        # 8 rounds of x1.5 from 1e-4 would reach 2.6e-3, above the cap.
        cap = 1e-3
        capped = [
            dataclasses.replace(d, step_size_max=cap)
            for d in harmonic_setup["descriptors"]
        ]
        adapt = build_adapt_step(
            move_descriptors=capped,
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )
        ns_state = harmonic_setup["ns_state"]
        pop = ns_state.population
        ns_state = ns_state.set(
            population=pop.set(step_sizes=jnp.full_like(pop.step_sizes, 1e-4))
        )
        emax = jnp.max(ns_state.population.energy)
        new_state, diag, _ = adapt(ns_state, emax, jax.random.key(16))
        assert jnp.all(diag["cap_hits"] > 0)
        assert jnp.all(new_state.population.step_sizes <= cap)

    def test_trial_batch_size_matches_full_vmap(self, harmonic_setup):
        """trial_batch_size=4 gives the same step sizes as a full vmap."""
        common = dict(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=8,
            adjust_factor=1.5,
            adjust_max_rounds=3,
            adjust_interval=50,
        )
        adapt_full = build_adapt_step(**common)
        adapt_chunked = build_adapt_step(trial_batch_size=4, **common)

        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        key = jax.random.key(17)
        state_full, _, _ = adapt_full(ns_state, emax, key)
        state_chunked, _, _ = adapt_chunked(ns_state, emax, key)

        np.testing.assert_allclose(
            state_full.population.step_sizes[0],
            state_chunked.population.step_sizes[0],
            rtol=1e-5,
        )


class TestVmapRuns:
    """Check the same outputs with a leading run axis (n_runs=2)."""

    def _build(self, harmonic_setup):
        return build_adapt_step(
            move_descriptors=harmonic_setup["descriptors"],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=VmapRuns(n_runs=harmonic_setup["n_runs"]),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )

    def test_diag_shapes(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_states = harmonic_setup["ns_states_batched"]
        n_runs = harmonic_setup["n_runs"]
        n_moves = 2
        emax_per_run = jnp.max(ns_states.population.energy, axis=1)
        run_keys = jax.random.split(jax.random.key(20), n_runs)
        _, diag, _ = adapt(ns_states, emax_per_run, run_keys)
        assert diag["rate"].shape == (n_runs, n_moves)
        assert diag["counts"].shape == (n_runs, n_moves, 4)
        for k in (
            "n_rounds",
            "converged",
            "cap_hits",
            "floor_hits",
            "bracket_detected",
            "trial_n_evaluations",
            "trial_n_grad_evaluations",
        ):
            assert diag[k].shape == (n_runs, n_moves), k

    def test_step_sizes_positive_finite(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_states = harmonic_setup["ns_states_batched"]
        n_runs = harmonic_setup["n_runs"]
        emax_per_run = jnp.max(ns_states.population.energy, axis=1)
        run_keys = jax.random.split(jax.random.key(22), n_runs)
        new_ns_states, _, _ = adapt(ns_states, emax_per_run, run_keys)
        ss_pop = new_ns_states.population.step_sizes
        assert jnp.all(ss_pop > 0.0)
        assert jnp.all(jnp.isfinite(ss_pop))

    def test_rng_keys_advance(self, harmonic_setup):
        adapt = self._build(harmonic_setup)
        ns_states = harmonic_setup["ns_states_batched"]
        n_runs = harmonic_setup["n_runs"]
        emax_per_run = jnp.max(ns_states.population.energy, axis=1)
        run_keys_in = jax.random.split(jax.random.key(21), n_runs)
        _, _, run_keys_out = adapt(ns_states, emax_per_run, run_keys_in)
        assert run_keys_out.shape == run_keys_in.shape
        assert not np.array_equal(
            np.asarray(jax.random.key_data(run_keys_in)),
            np.asarray(jax.random.key_data(run_keys_out)),
        )


# ---------------------------------------------------------------------------
# Moves with adaptive=False are left alone
# ---------------------------------------------------------------------------


class TestNotAdaptedMoves:
    """``rw_2`` is marked non-adaptive; ``rw_1`` is adapted as usual."""

    def _build(self, harmonic_setup, batcher, adaptive=(True, False)):
        descriptors = [
            dataclasses.replace(d, adaptive=a)
            for d, a in zip(harmonic_setup["descriptors"], adaptive)
        ]
        return build_adapt_step(
            move_descriptors=descriptors,
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=batcher,
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )

    def test_all_moves_not_adaptive_is_inactive(self, harmonic_setup):
        adapt = self._build(harmonic_setup, SingleRun(), (False, False))
        assert adapt is None

    def test_single_run(self, harmonic_setup):
        adapt = self._build(harmonic_setup, SingleRun())
        ns_state = harmonic_setup["ns_state"]
        emax = jnp.max(ns_state.population.energy)
        new_state, diag, _ = adapt(ns_state, emax, jax.random.key(30))
        np.testing.assert_array_equal(
            new_state.population.step_sizes[:, 1],
            ns_state.population.step_sizes[:, 1],
        )
        assert np.isnan(float(diag["rate"][1]))
        assert bool(diag["converged"][1])
        assert int(diag["n_rounds"][1]) == 0
        assert int(diag["trial_n_evaluations"][1]) == 0
        assert np.all(np.asarray(diag["counts"][1]) == 0)
        # the adaptive move still runs its trials
        assert np.isfinite(float(diag["rate"][0]))
        assert int(diag["n_rounds"][0]) >= 1

    def test_vmap_runs(self, harmonic_setup):
        n_runs = harmonic_setup["n_runs"]
        adapt = self._build(harmonic_setup, VmapRuns(n_runs=n_runs))
        ns_states = harmonic_setup["ns_states_batched"]
        emax = jnp.max(ns_states.population.energy, axis=1)
        keys = jax.random.split(jax.random.key(31), n_runs)
        new_states, diag, _ = adapt(ns_states, emax, keys)
        np.testing.assert_array_equal(
            new_states.population.step_sizes[..., 1],
            ns_states.population.step_sizes[..., 1],
        )
        assert diag["rate"].shape == (n_runs, 2)
        assert np.all(np.isnan(np.asarray(diag["rate"][:, 1])))
        assert np.all(np.isfinite(np.asarray(diag["rate"][:, 0])))

    def test_per_move_adjust_factor(self, harmonic_setup, monkeypatch):
        """Each move's builder gets its own factor, else the run-wide one."""
        import jaxrens.sampling.adaptation.manager as manager

        seen = {}
        real = manager._build_per_move

        def spy(desc, fn, batcher, n_samp, afac, *rest):
            seen[desc.name] = afac
            return real(desc, fn, batcher, n_samp, afac, *rest)

        monkeypatch.setattr(manager, "_build_per_move", spy)
        d1, d2 = harmonic_setup["descriptors"]
        build_adapt_step(
            move_descriptors=[d1, dataclasses.replace(d2, adjust_factor=2.0)],
            per_move_fns=harmonic_setup["per_move_fns"],
            batcher=SingleRun(),
            adjust_n_samples=20,
            adjust_factor=1.5,
            adjust_max_rounds=8,
            adjust_interval=50,
        )
        assert seen == {"rw_1": 1.5, "rw_2": 2.0}
