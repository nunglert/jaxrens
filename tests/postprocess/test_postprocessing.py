"""Tests for postprocess.thermodynamics.

Checks calc_log_weights, log_evidence, partition_function, heat_capacity,
expectation and free_energy against the analytic results for a 3D
harmonic potential, and that each function maps a (G, P, n) batch slice
by slice.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    expectation,
    free_energy,
    heat_capacity,
    log_evidence,
    partition_function,
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
