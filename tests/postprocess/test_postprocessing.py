"""Tests for postprocess.thermodynamics.

Checks calc_log_weights, log_point_weights, thermal_weights, log_evidence,
partition_function, heat_capacity, expectation and free_energy against the
analytic results for a 3D harmonic potential, that each function maps a
(G, P, n) batch slice by slice, and that a 1-D beta gives the same values
as one call per temperature.  Also checks the estimators on a weight
matrix: weighted_mean / variance / covariance, class_probability and
conditional_mean.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    calc_log_weights_live,
    class_probability,
    conditional_mean,
    expectation,
    free_energy,
    heat_capacity,
    log_evidence,
    log_point_weights,
    partition_function,
    thermal_weights,
    weighted_covariance,
    weighted_mean,
    weighted_variance,
)

N_DIM = 3
N_LIVE = 200


def _harmonic_ns_data(
    n_dead: int = 3000,
    n_live: int = N_LIVE,
    n_dim: int = N_DIM,
    e_max: float = 50.0,
):
    """Nested-sampling energies for E = |x|^2 / 2 with a uniform prior in
    a ball, where the enclosed prior mass is X(E) = (E / e_max)^(n_dim/2).

    Dead point k sits at the expected NS prior mass
    X_k = (n_live / (n_live + 1))^(k + 1), so the energies decrease along
    the dead sequence, as NS produces them, and match the weights of
    calc_log_weights. The live points spread evenly in X below the last
    dead point. With e_max large enough for the Boltzmann weight to fit
    inside the ball, the analytic results hold to about 1e-7:
    log Z(beta) = -n_dim/2 log(beta) + const, C_v = n_dim/2 and
    <E> = n_dim / (2 beta).
    """
    x_dead = (n_live / (n_live + 1.0)) ** np.arange(1, n_dead + 1)
    x_live = x_dead[-1] * (np.arange(n_live) + 0.5) / n_live
    dead = e_max * x_dead ** (2.0 / n_dim)
    live = e_max * x_live ** (2.0 / n_dim)
    return dead, live


@pytest.fixture
def harmonic():
    return _harmonic_ns_data()


class TestCalcLogWeights:
    """Check the dead-point log prior-mass weights. The formula itself is
    tested in tests/test_prior_mass.py."""

    def test_shape(self):
        log_w = calc_log_weights(n_dead=50, n_live=20, n_cull=1)
        assert log_w.shape == (50,)

    def test_known_values_n_cull_1(self):
        """pymatnest convention for n_live=10, n_cull=1: per-dead-point
        shrinkage ``N/(N+1)`` and ``log a_n = (n+1) log(N/(N+1)) - log(N+1)``
        (``ns_analyse.calc_log_a``)."""
        n_live = 10
        n_dead = 5

        log_t = np.log(n_live / (n_live + 1.0))
        expected = (np.arange(n_dead) + 1.0) * log_t - np.log(n_live + 1.0)

        log_w = calc_log_weights(n_dead=n_dead, n_live=n_live, n_cull=1)
        np.testing.assert_allclose(log_w, expected, atol=1e-12)


class TestLogPointWeights:
    """Check the dead-then-live layout of log_point_weights."""

    def test_layout(self):
        log_w = log_point_weights(n_dead=30, n_live=10, n_cull=2)
        np.testing.assert_array_equal(log_w[:30], calc_log_weights(30, 10, 2))
        np.testing.assert_array_equal(
            log_w[30:], np.full(10, float(calc_log_weights_live(30, 10, 2)))
        )

    def test_n_live_points(self):
        log_w = log_point_weights(n_dead=30, n_live=10, n_live_points=3)
        assert log_w.shape == (33,)

    @pytest.mark.parametrize("n_cull", [1, 3])
    def test_total_mass_independent_of_n_dead(self, n_cull):
        """Dead plus live weights add up to the same X_0 after any number of
        dead points."""
        totals = [
            logsumexp(log_point_weights(n_dead, 10, n_cull))
            for n_dead in (0, 7, 300)
        ]
        np.testing.assert_allclose(totals, totals[0], rtol=1e-12)


class TestThermalWeights:
    """Check the normalised weight matrix the estimators are built on."""

    def test_normalised(self, harmonic):
        dead_E, live_E = harmonic
        w = thermal_weights(1.0, dead_E, live_E, n_live=N_LIVE)
        assert w.shape == (dead_E.shape[0] + live_E.shape[0],)
        assert np.all(w >= 0.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)

    def test_weighted_sum_is_expectation(self, harmonic):
        dead_E, live_E = harmonic
        all_E = np.concatenate([dead_E, live_E])
        w = thermal_weights(2.0, dead_E, live_E, n_live=N_LIVE)
        ref = expectation(all_E, 2.0, dead_E, live_E, n_live=N_LIVE)
        assert w @ all_E == pytest.approx(float(ref), rel=1e-12)

    def test_return_log_z_is_partition_function(self, harmonic):
        dead_E, live_E = harmonic
        w, log_Z = thermal_weights(
            _BETAS, dead_E, live_E, N_LIVE, return_log_Z=True
        )
        assert log_Z.shape == w.shape[:-1] == (len(_BETAS),)
        np.testing.assert_allclose(
            log_Z,
            partition_function(_BETAS, dead_E, live_E, n_live=N_LIVE),
            rtol=1e-12,
        )

    def test_2d_beta_raises(self, harmonic):
        dead_E, live_E = harmonic
        with pytest.raises(ValueError, match="scalar or 1-D"):
            thermal_weights(np.ones((2, 2)), dead_E, live_E, N_LIVE)


class TestLogEvidence:
    """Check log_evidence."""

    def test_scalar_output(self, harmonic):
        dead_E, live_E = harmonic
        log_Z = log_evidence(dead_E, live_E, n_live=N_LIVE, n_cull=1)
        assert log_Z.ndim == 0

    def test_equals_partition_function_at_beta_1(self, harmonic):
        """With E = -log L the evidence is Z(beta) at beta = 1."""
        dead_E, live_E = harmonic
        log_Z = log_evidence(dead_E, live_E, n_live=N_LIVE)
        log_Z_1 = partition_function(1.0, dead_E, live_E, n_live=N_LIVE)
        np.testing.assert_allclose(log_Z, log_Z_1, rtol=1e-12)


class TestPartitionFunction:
    """Check log Z(beta) for the harmonic potential, where
    d(log Z)/d(log beta) = -n_dim/2. The prior volume only adds a
    constant."""

    def test_scalar_output(self, harmonic):
        dead_E, live_E = harmonic
        log_Z = partition_function(1.0, dead_E, live_E, n_live=N_LIVE)
        assert log_Z.ndim == 0
        assert np.isfinite(log_Z)

    def test_log_z_slope_vs_beta(self, harmonic):
        """A linear fit of log Z against log beta has slope -n_dim/2."""
        dead_E, live_E = harmonic
        betas = np.array([0.5, 1.0, 2.0, 4.0])
        log_Zs = [
            partition_function(b, dead_E, live_E, n_live=N_LIVE) for b in betas
        ]
        slope = np.polyfit(np.log(betas), log_Zs, 1)[0]
        assert slope == pytest.approx(-N_DIM / 2.0, abs=1e-3)

    def test_z_increases_with_temperature(self, harmonic):
        """Z grows as beta decreases."""
        dead_E, live_E = harmonic
        log_Z_cold = partition_function(5.0, dead_E, live_E, n_live=N_LIVE)
        log_Z_hot = partition_function(0.5, dead_E, live_E, n_live=N_LIVE)
        assert float(log_Z_hot) > float(log_Z_cold)


class TestHeatCapacity:
    """Check C_v for the harmonic potential; equipartition gives
    n_dim/2 in units of k_B."""

    def test_scalar_output(self, harmonic):
        dead_E, live_E = harmonic
        cv = heat_capacity(1.0, dead_E, live_E, n_live=N_LIVE)
        assert cv.ndim == 0
        assert np.isfinite(cv)

    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 5.0])
    def test_equipartition(self, harmonic, beta):
        dead_E, live_E = harmonic
        cv = heat_capacity(beta, dead_E, live_E, n_live=N_LIVE)
        assert float(cv) == pytest.approx(N_DIM / 2.0, abs=1e-3)


class TestExpectation:
    """Check thermal expectations; for the harmonic potential
    <E> = n_dim / (2 beta)."""

    def test_scalar_output(self, harmonic):
        dead_E, live_E = harmonic
        all_E = np.concatenate([dead_E, live_E])
        mean_E = expectation(all_E, 1.0, dead_E, live_E, n_live=N_LIVE)
        assert mean_E.ndim == 0

    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 5.0])
    def test_mean_energy(self, harmonic, beta):
        dead_E, live_E = harmonic
        all_E = np.concatenate([dead_E, live_E])
        mean_E = expectation(all_E, beta, dead_E, live_E, n_live=N_LIVE)
        assert float(beta * mean_E) == pytest.approx(N_DIM / 2.0, abs=1e-3)

    def test_mean_energy_increases_with_T(self, harmonic):
        """<E> grows as beta decreases."""
        dead_E, live_E = harmonic
        all_E = np.concatenate([dead_E, live_E])
        mean_cold = expectation(all_E, 5.0, dead_E, live_E, n_live=N_LIVE)
        mean_hot = expectation(all_E, 0.5, dead_E, live_E, n_live=N_LIVE)
        assert float(mean_hot) > float(mean_cold)

    def test_constant_observable(self, harmonic):
        """<c> = c for a constant observable."""
        dead_E, live_E = harmonic
        c = 42.0
        obs = np.full(dead_E.shape[0] + live_E.shape[0], c)
        result = expectation(obs, 1.0, dead_E, live_E, n_live=N_LIVE)
        assert result == pytest.approx(c, abs=1e-4)


class TestFreeEnergy:
    """Check that F = -log Z / beta and the Gibbs-Helmholtz relation."""

    def test_multiple_temperatures(self, harmonic):
        dead_E, live_E = harmonic
        for beta in [0.5, 1.0, 2.0, 5.0]:
            log_Z = partition_function(beta, dead_E, live_E, n_live=N_LIVE)
            F = free_energy(beta, log_Z)
            assert np.isfinite(F), f"beta={beta}"
            assert F == pytest.approx(-log_Z / beta, abs=1e-6)

    def test_gibbs_helmholtz(self, harmonic):
        """d(beta F)/d(beta) = <E>, by central finite difference."""
        dead_E, live_E = harmonic
        all_E = np.concatenate([dead_E, live_E])
        beta, dbeta = 1.0, 0.01

        def beta_f(b):
            log_Z = partition_function(b, dead_E, live_E, n_live=N_LIVE)
            return b * free_energy(b, log_Z)

        d_beta_f = (beta_f(beta + dbeta) - beta_f(beta - dbeta)) / (2 * dbeta)
        mean_E = expectation(all_E, beta, dead_E, live_E, n_live=N_LIVE)
        assert float(d_beta_f) == pytest.approx(float(mean_E), abs=1e-3)


def _make_gp_data(G: int = 2, P: int = 2, n_dead: int = 500, n_live: int = 50):
    """Stack 1-D runs with different e_max into JAX arrays dead
    (G, P, n_dead) and live (G, P, n_live), so every slice differs and the
    JAX-input path is exercised."""
    dead = np.empty((G, P, n_dead))
    live = np.empty((G, P, n_live))
    for g in range(G):
        for p in range(P):
            e_max = 50.0 * (g * P + p + 1)
            dead[g, p], live[g, p] = _harmonic_ns_data(
                n_dead=n_dead, n_live=n_live, e_max=e_max
            )
    return jnp.asarray(dead), jnp.asarray(live)


_BATCH_N_LIVE = 50
_BATCH_FNS = {
    "log_evidence": lambda d, l: log_evidence(d, l, n_live=_BATCH_N_LIVE),
    "partition_function": lambda d, l: partition_function(
        1.0, d, l, n_live=_BATCH_N_LIVE
    ),
    "heat_capacity": lambda d, l: heat_capacity(
        1.0, d, l, n_live=_BATCH_N_LIVE
    ),
    "expectation": lambda d, l: expectation(
        jnp.concatenate([d, l], axis=-1), 1.0, d, l, n_live=_BATCH_N_LIVE
    ),
    "free_energy": lambda d, l: free_energy(
        1.0, partition_function(1.0, d, l, n_live=_BATCH_N_LIVE)
    ),
}


@pytest.mark.parametrize("name", list(_BATCH_FNS))
def test_batch_matches_per_slice(name):
    """A (G, P, n) input gives a (G, P) output equal to the 1-D result of
    each slice. G = P = 2 catches swapped batch axes."""
    fn = _BATCH_FNS[name]
    dead, live = _make_gp_data()
    batched = fn(dead, live)
    assert batched.shape == (2, 2)
    for g in range(2):
        for p in range(2):
            ref = fn(dead[g, p], live[g, p])
            np.testing.assert_allclose(
                batched[g, p], ref, rtol=1e-12, err_msg=f"({g}, {p})"
            )


_BETAS = np.array([0.5, 1.0, 3.0])
_TEMPERATURE_FNS = {
    "thermal_weights": lambda b, d, l: thermal_weights(
        b, d, l, n_live=_BATCH_N_LIVE
    ),
    "partition_function": lambda b, d, l: partition_function(
        b, d, l, n_live=_BATCH_N_LIVE
    ),
    "heat_capacity": lambda b, d, l: heat_capacity(
        b, d, l, n_live=_BATCH_N_LIVE
    ),
    "expectation": lambda b, d, l: expectation(
        jnp.concatenate([d, l], axis=-1), b, d, l, n_live=_BATCH_N_LIVE
    ),
}


@pytest.mark.parametrize("name", list(_TEMPERATURE_FNS))
def test_beta_grid_matches_per_beta(name):
    """A 1-D beta on a (G, P, n) batch adds an n_T axis after the batch
    axes, equal to one scalar-beta call per temperature."""
    fn = _TEMPERATURE_FNS[name]
    dead, live = _make_gp_data()
    grid = fn(_BETAS, dead, live)
    per_beta = [fn(b, dead, live) for b in _BETAS]
    if name == "thermal_weights":
        assert grid.shape == (2, 2, len(_BETAS), 550)
        ref = np.stack(per_beta, axis=-2)
    else:
        assert grid.shape == (2, 2, len(_BETAS))
        ref = np.stack(per_beta, axis=-1)
    np.testing.assert_allclose(grid, ref, rtol=1e-12)


@pytest.fixture
def grid(harmonic):
    """Weights on a beta grid, all energies, and a second observable."""
    dead_E, live_E = harmonic
    all_E = np.concatenate([dead_E, live_E])
    w = thermal_weights(_BETAS, dead_E, live_E, n_live=N_LIVE)
    other = np.sin(all_E) + 0.1 * all_E
    return w, all_E, other, dead_E, live_E


class TestWeightedMoments:
    """Check weighted_mean, weighted_variance and weighted_covariance."""

    def test_mean_is_expectation(self, grid):
        w, all_E, _, dead_E, live_E = grid
        ref = expectation(all_E, _BETAS, dead_E, live_E, n_live=N_LIVE)
        np.testing.assert_allclose(weighted_mean(w, all_E), ref, rtol=1e-12)

    def test_variance_of_energy_is_heat_capacity(self, grid):
        w, all_E, _, dead_E, live_E = grid
        cv = heat_capacity(_BETAS, dead_E, live_E, n_live=N_LIVE)
        np.testing.assert_allclose(
            _BETAS**2 * weighted_variance(w, all_E), cv, rtol=1e-12
        )

    def test_covariance_properties(self, grid):
        """Symmetric, Cov(A, A) = Var(A), zero against a constant, linear
        in each argument."""
        w, a, b, _, _ = grid
        c = np.cos(a)
        np.testing.assert_allclose(
            weighted_covariance(w, a, b), weighted_covariance(w, b, a)
        )
        np.testing.assert_allclose(
            weighted_covariance(w, a, a), weighted_variance(w, a)
        )
        np.testing.assert_allclose(
            weighted_covariance(w, a, np.full_like(a, 3.0)), 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            weighted_covariance(w, 2.0 * a + b, c),
            2.0 * weighted_covariance(w, a, c) + weighted_covariance(w, b, c),
            rtol=1e-10,
        )

    def test_variance_survives_a_large_offset(self, grid):
        """The deviation form keeps the variance of O + 1e8 equal to that
        of O, where <O^2> - <O>^2 would lose it to rounding."""
        w, a, _, _, _ = grid
        np.testing.assert_allclose(
            weighted_variance(w, a + 1e8), weighted_variance(w, a), rtol=1e-6
        )

    def test_vector_observable_matches_columns(self, grid):
        w, a, b, _, _ = grid
        vec = np.stack([a, b, a * b], axis=1)
        mean = weighted_mean(w, vec)
        var = weighted_variance(w, vec)
        cov = weighted_covariance(w, a, vec)
        assert mean.shape == var.shape == cov.shape == (len(_BETAS), 3)
        for j in range(3):
            np.testing.assert_allclose(mean[:, j], weighted_mean(w, vec[:, j]))
            np.testing.assert_allclose(
                var[:, j], weighted_variance(w, vec[:, j]), rtol=1e-12
            )
            np.testing.assert_allclose(
                cov[:, j], weighted_covariance(w, a, vec[:, j]), rtol=1e-12
            )

    def test_batch_leading_axes(self):
        """Weights with (G, P, n_T) leading axes give a (G, P, n_T) result,
        slice by slice equal to the 1-D weights."""
        dead, live = _make_gp_data()
        w = thermal_weights(_BETAS, dead, live, n_live=_BATCH_N_LIVE)
        obs = np.linspace(0.0, 1.0, w.shape[-1]) ** 2
        vec = np.stack([obs, 1.0 - obs], axis=1)
        var = weighted_variance(w, obs)
        var_vec = weighted_variance(w, vec)
        assert var.shape == (2, 2, len(_BETAS))
        assert var_vec.shape == (2, 2, len(_BETAS), 2)
        np.testing.assert_allclose(
            var[1, 0], weighted_variance(w[1, 0], obs), rtol=1e-12
        )
        np.testing.assert_allclose(var_vec[..., 0], var, rtol=1e-12)

    def test_wrong_point_count_raises(self, grid):
        w, a, _, _, _ = grid
        with pytest.raises(ValueError, match="n_pts="):
            weighted_mean(w, a[:-1])


class TestClassAverages:
    """Check class_probability and conditional_mean."""

    def test_complementary_classes_sum_to_one(self, grid):
        w, a, _, _, _ = grid
        mask = a < np.median(a)
        np.testing.assert_allclose(
            class_probability(w, mask) + class_probability(w, ~mask), 1.0
        )
        np.testing.assert_allclose(
            class_probability(w, np.ones_like(mask)), 1.0
        )

    def test_conditional_mean_definition(self, grid):
        """<O | C> = <O 1_C> / <1_C>."""
        w, a, b, _, _ = grid
        mask = a < np.median(a)
        ref = weighted_mean(w, b * mask) / class_probability(w, mask)
        np.testing.assert_allclose(
            conditional_mean(w, b, mask), ref, rtol=1e-12
        )

    def test_conditional_mean_without_mask_is_mean(self, grid):
        w, a, _, _, _ = grid
        np.testing.assert_allclose(
            conditional_mean(w, a), weighted_mean(w, a), rtol=1e-12
        )

    def test_nan_points_are_left_out(self, grid):
        w, a, b, _, _ = grid
        undefined = a > np.median(a)
        with_nan = np.where(undefined, np.nan, b)
        np.testing.assert_allclose(
            conditional_mean(w, with_nan),
            conditional_mean(w, b, ~undefined),
            rtol=1e-12,
        )

    def test_empty_class_is_nan(self, grid):
        w, a, _, _, _ = grid
        out = conditional_mean(w, a, np.zeros(a.shape, dtype=bool))
        assert out.shape == (len(_BETAS),)
        assert np.all(np.isnan(out))

    def test_vector_masks(self, grid):
        """A (n_pts, k) mask gives k classes at once, column by column."""
        w, a, b, _, _ = grid
        masks = np.stack([a < np.median(a), a >= np.median(a)], axis=1)
        prob = class_probability(w, masks)
        cond = conditional_mean(w, b, masks)
        assert prob.shape == cond.shape == (len(_BETAS), 2)
        for j in range(2):
            np.testing.assert_allclose(
                cond[:, j], conditional_mean(w, b, masks[:, j]), rtol=1e-12
            )
