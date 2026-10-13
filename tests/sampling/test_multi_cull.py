"""Multi-walker culling (``n_cull > 1``) in the NS step.

Verifies:
- ``_select_culls``: top-``n_cull`` selection (descending, random ties),
  contour = lowest culled potential, distinct clone sources disjoint from the
  culls, extras disjoint from both.
- ``ns_step(n_cull=c)``: info shapes, untouched survivors, contour invariant,
  first-step evidence equals the pymatnest-weighted dead points.
- ``n_cull = 1`` keeps the historical single-cull path bit-for-bit.
- The dead-point ladder of a full run tracks the analytic prior mass of a
  3-D harmonic well for both ``n_cull = 1`` and ``n_cull > 1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.prior_mass import log_dead_weights, log_prior_mass
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    _find_worst_walker,
    _get_extra_indices,
    _select_culls,
    init_ns,
    ns_step,
    run_ns,
)


def _keys(seed=0):
    return jax.random.split(jax.random.key(seed), 3)


# ---------------------------------------------------------------------------
# _select_culls
# ---------------------------------------------------------------------------


class TestSelectCulls:
    @pytest.mark.parametrize("n_cull,n_extra", [(2, 0), (4, 3), (5, 10)])
    def test_selection_invariants(self, n_cull, n_extra):
        n = 30
        # Coarse grid → plenty of exact ties, including at the contour.
        pot = jnp.round(jax.random.normal(jax.random.key(7), (n,)) * 2.0)
        worst, clone, walk, contour = _select_culls(
            pot, n_cull, n_extra, *_keys(1)
        )
        worst, clone, walk = map(np.asarray, (worst, clone, walk))
        p = np.asarray(pot)

        assert worst.shape == clone.shape == (n_cull,)
        assert walk.shape == (n_cull + n_extra,)
        assert len(set(worst)) == n_cull
        # Descending potential; the culls are a true top-n_cull set.
        assert np.all(np.diff(p[worst]) <= 0)
        assert np.sort(p)[::-1][:n_cull].tolist() == p[worst].tolist()
        assert float(contour) == p[worst[-1]]
        survivors = np.setdiff1d(np.arange(n), worst)
        assert np.all(p[survivors] <= float(contour))
        # Clone sources: distinct survivors.
        assert len(set(clone)) == n_cull
        assert not set(clone) & set(worst)
        # Walk set: the culled slots first, then extras disjoint from both.
        assert walk[:n_cull].tolist() == worst.tolist()
        extras = walk[n_cull:]
        assert len(set(extras)) == n_extra
        assert not set(extras) & (set(worst) | set(clone))

    def test_all_tied_still_distinct(self):
        worst, clone, _, contour = _select_culls(
            jnp.zeros(12), 4, 2, *_keys(2)
        )
        assert len(set(np.asarray(worst))) == 4
        assert len(set(np.asarray(clone))) == 4
        assert float(contour) == 0.0

    def test_ties_broken_randomly(self):
        pot = jnp.zeros(20)
        picks = {
            tuple(np.asarray(_select_culls(pot, 3, 0, *_keys(s))[0]))
            for s in range(8)
        }
        assert len(picks) > 1

    def test_too_wide_raises(self):
        with pytest.raises(ValueError, match="2\\*n_cull"):
            _select_culls(jnp.arange(10.0), 5, 1, *_keys())

    @pytest.mark.parametrize("n_extra", [0, 3])
    def test_n_cull_1_is_historical_path(self, n_extra):
        pot = jax.random.normal(jax.random.key(3), (25,))
        kw, kc, ke = _keys(4)
        worst, clone, walk, contour = _select_culls(
            pot, 1, n_extra, kw, kc, ke
        )

        w_ref, c_ref = _find_worst_walker(pot, kw)
        clone_ref = jax.random.randint(kc, (), 0, 24)
        clone_ref = jnp.where(clone_ref >= w_ref, clone_ref + 1, clone_ref)
        assert int(worst[0]) == int(w_ref)
        assert float(contour) == float(c_ref)
        assert int(clone[0]) == int(clone_ref)
        if n_extra:
            extras = _get_extra_indices(w_ref, clone_ref, 25, n_extra, ke)
            np.testing.assert_array_equal(walk[1:], extras)


# ---------------------------------------------------------------------------
# ns_step with n_cull > 1
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_state():
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend, [MoveKernel("random_walk", random_walk.build_kernel)]
    )
    n_walkers = 40
    key, init_key = jax.random.split(jax.random.key(11))
    positions = jax.random.uniform(
        init_key, (n_walkers, 1, 3), minval=-3.0, maxval=3.0
    )
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    state = init_ns(init_fn, positions, types, energies, None, key)
    return state, step_fn


class TestNsStepMultiCull:
    N_CULL = 4
    N_EXTRA = 3

    @pytest.fixture
    def stepped(self, harmonic_state):
        state, step_fn = harmonic_state
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3, 4))
        new_state, info = jit_step(
            state, step_fn, 10, self.N_EXTRA, self.N_CULL
        )
        return state, new_state, info

    def test_info_shapes(self, stepped):
        state, _, info = stepped
        c = self.N_CULL
        dw = info["dead_walkers"]
        assert dw.positions.shape == (c, state.n_atoms, 3)
        assert dw.energy.shape == (c,)
        assert info["worst_indices"].shape == (c,)
        assert info["clone_indices"].shape == (c,)
        # Single-walker views are the contour cull.
        assert info["dead_walker"].energy.shape == ()
        assert float(info["dead_walker"].energy) == float(dw.energy[-1])
        assert float(info["emax"]) == float(dw.energy[-1])
        assert int(info["worst_idx"]) == int(info["worst_indices"][-1])

    def test_dead_points_are_top_culls(self, stepped):
        state, _, info = stepped
        e0 = np.asarray(state.population.energy)
        dead_e = np.asarray(info["dead_walkers"].energy)
        np.testing.assert_array_equal(dead_e, np.sort(e0)[::-1][: self.N_CULL])
        np.testing.assert_array_equal(
            dead_e, e0[np.asarray(info["worst_indices"])]
        )

    def test_population_respects_contour(self, stepped):
        _, new_state, info = stepped
        assert np.all(
            np.asarray(new_state.population.energy) <= float(info["emax"])
        )
        assert int(new_state.iteration) == 1
        assert float(new_state.emax) == float(info["emax"])

    def test_unwalked_survivors_untouched(self, stepped, harmonic_state):
        state, new_state, info = stepped
        walked = set(np.asarray(info["worst_indices"]).tolist())
        # Extras are the other walked slots; recover them as the walkers
        # whose positions changed outside the culled set.
        old = np.asarray(state.population.positions)
        new = np.asarray(new_state.population.positions)
        changed = {
            i
            for i in range(old.shape[0])
            if not np.array_equal(old[i], new[i])
        }
        assert len(changed - walked) <= self.N_EXTRA

    def test_first_step_evidence(self, stepped):
        state, new_state, info = stepped
        n = state.n_walkers
        w = log_dead_weights(self.N_CULL, n, self.N_CULL)
        expected = np.logaddexp.reduce(
            w - np.asarray(info["dead_walkers"].energy, dtype=np.float64)
        )
        np.testing.assert_allclose(
            float(new_state.log_evidence), expected, rtol=1e-5
        )


class TestNCull1Unchanged:
    def test_single_cull_info_has_unit_cull_axis(self, harmonic_state):
        state, step_fn = harmonic_state
        jit_step = jax.jit(ns_step, static_argnums=(1, 2, 3))
        _, info = jit_step(state, step_fn, 5, 2)
        assert info["dead_walkers"].energy.shape == (1,)
        assert info["worst_idx"].shape == ()
        assert float(info["emax"]) == float(jnp.max(state.population.energy))


# ---------------------------------------------------------------------------
# Full run: dead-point ladder vs the analytic prior mass
# ---------------------------------------------------------------------------


class _DeadEnergies:
    def __init__(self):
        self.energies = []

    def on_iteration(self, iteration, ns_state, info):
        self.energies.extend(
            np.asarray(info["dead_walkers"].energy, dtype=np.float64).tolist()
        )


@pytest.mark.parametrize("n_cull", [1, 5])
def test_ladder_tracks_analytic_prior_mass(n_cull):
    """3-D harmonic well ``E = r^2 / 2`` with a uniform prior on
    ``[-L, L]^3``: inside the box, ``X(E) = (4/3) pi (2E)^{3/2} / (2L)^3``.

    The estimated ``log X_n`` of each dead point (pymatnest ladder with this
    ``n_cull``) must track ``log X(E_n)`` to within NS noise
    (``~ sqrt(n) / N``).  Treating a 5-wide cull as a single cull would put
    the ladder off by a factor ~5 in ``log X`` and fail by a wide margin.
    """
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend, [MoveKernel("random_walk", random_walk.build_kernel)]
    )
    n_live, L = 100, 3.0
    key, init_key = jax.random.split(jax.random.key(5))
    positions = jax.random.uniform(
        init_key, (n_live, 1, 3), minval=-L, maxval=L
    )
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    n_dead_target = 600  # log X ≈ -6
    rec = _DeadEnergies()
    run_ns(
        positions,
        types,
        energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_key=key,
        n_cull=n_cull,
        max_iterations=n_dead_target // n_cull,
        n_mcmc_steps=30,
        initial_step_size=0.3,
        convergence_threshold=1e-12,
        callbacks=[rec],
    )

    e = np.asarray(rec.energies)
    assert e.shape == (n_dead_target,)
    log_x_est = log_prior_mass(e.size, n_live, n_cull)
    inside = e < 0.5 * L**2  # contour sphere fits in the prior box
    log_x_true = np.log(4.0 / 3.0 * np.pi * (2.0 * e[inside]) ** 1.5) - np.log(
        (2 * L) ** 3
    )
    resid = log_x_est[inside] - log_x_true
    assert np.abs(resid).max() < 1.0, resid
    assert abs(resid[-100:].mean()) < 0.5, resid[-100:].mean()
