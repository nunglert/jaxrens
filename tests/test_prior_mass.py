"""Tests for jaxrens.prior_mass — the shared n_cull-aware prior-mass ladder.

The reference is pymatnest's ``ns_analyse.calc_log_a`` (SENS paper, PRX 4,
031034 (2014), Eq. 3), transcribed verbatim below so the comparison doesn't
depend on a pymatnest checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxrens.prior_mass import (
    iteration_offsets,
    log_alpha,
    log_dead_weights,
    log_live_weight,
    log_prior_mass,
)


def _pymatnest_calc_log_a(n_Es, n_walkers, n_cull):
    """Verbatim pymatnest ``calc_log_a`` (interval=1)."""
    i_range = np.array(range(n_Es))
    i_range_mod_n_cull = np.mod(i_range, n_cull)
    i_range_plus_1_mod_n_cull = np.mod(i_range + 1, n_cull)
    log_X_n_term = np.log(n_walkers - i_range_mod_n_cull) - np.log(
        n_walkers + 1 - i_range_mod_n_cull
    )
    log_X_n = np.cumsum(log_X_n_term)
    return log_X_n - np.log(n_walkers + 1 - i_range_plus_1_mod_n_cull)


CASES = [(10, 1), (600, 1), (50, 3), (600, 10), (7, 3)]


@pytest.mark.parametrize("n_live,n_cull", CASES)
def test_dead_weights_match_pymatnest(n_live, n_cull):
    n_dead = 41 * n_cull
    np.testing.assert_allclose(
        log_dead_weights(n_dead, n_live, n_cull),
        _pymatnest_calc_log_a(n_dead, n_live, n_cull),
        rtol=0,
        atol=1e-12,
    )


def test_n_cull_1_is_N_over_N_plus_1():
    """Regression: the old formula used ``(N-1)/N`` per dead point."""
    n = 600
    log_x = log_prior_mass(5, n, 1)
    np.testing.assert_allclose(np.diff(log_x), np.log(n / (n + 1.0)))


@pytest.mark.parametrize("n_live,n_cull", CASES)
def test_full_iteration_contracts_by_alpha(n_live, n_cull):
    log_x = log_prior_mass(4 * n_cull, n_live, n_cull)
    per_iter = log_x[n_cull - 1 :: n_cull]
    np.testing.assert_allclose(
        np.diff(per_iter), log_alpha(n_live, n_cull), atol=1e-12
    )
    np.testing.assert_allclose(
        log_alpha(n_live, n_cull),
        np.log((n_live + 1.0 - n_cull) / (n_live + 1.0)),
    )


@pytest.mark.parametrize("n_live,n_cull", CASES)
def test_iteration_offsets_closed_form(n_live, n_cull):
    n_iter = 25
    x_off, w_off = iteration_offsets(n_live, n_cull)
    la = log_alpha(n_live, n_cull)
    closed_x = np.concatenate([i * la + x_off for i in range(n_iter)])
    closed_w = np.concatenate([i * la + w_off for i in range(n_iter)])
    np.testing.assert_allclose(
        closed_x, log_prior_mass(n_iter * n_cull, n_live, n_cull), atol=1e-10
    )
    np.testing.assert_allclose(
        closed_w, log_dead_weights(n_iter * n_cull, n_live, n_cull), atol=1e-10
    )


@pytest.mark.parametrize("n_live,n_cull", CASES)
@pytest.mark.parametrize("n_iter", [0, 1, 13])
def test_dead_plus_live_weights_conserve_mass(n_live, n_cull, n_iter):
    """Dead + live weights sum to the constant ``X_0 = N / (N + 1)`` at
    every iteration boundary, so no prior mass is lost or double-counted."""
    n_dead = n_iter * n_cull
    total = np.logaddexp(
        np.logaddexp.reduce(log_dead_weights(n_dead, n_live, n_cull))
        if n_dead
        else -np.inf,
        float(log_live_weight(n_dead, n_live, n_cull)) + np.log(n_live),
    )
    np.testing.assert_allclose(
        total, np.log(n_live / (n_live + 1.0)), atol=1e-10
    )


@pytest.mark.parametrize("n_live,n_cull", [(10, 0), (10, 10), (10, 11)])
def test_invalid_n_cull_rejected(n_live, n_cull):
    with pytest.raises(ValueError):
        log_alpha(n_live, n_cull)
