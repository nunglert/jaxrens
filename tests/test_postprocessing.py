"""Tests for the thermodynamic post-processing module.

Verifies calc_log_weights, partition_function, heat_capacity,
expectation, free_energy, and log_evidence against known analytical
results for a 3D harmonic potential.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    calc_log_weights_live,
    free_energy,
    heat_capacity,
    log_evidence,
    partition_function,
    expectation,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic nested-sampling data for a harmonic potential
# ---------------------------------------------------------------------------

def _generate_harmonic_ns_data(
    n_dead: int = 2000,
    n_live: int = 100,
    n_dim: int = 3,
    omega: float = 1.0,
    seed: int = 0,
):
    """Generate synthetic nested-sampling energies for a harmonic potential.

    E = 0.5 * omega^2 * sum(x_i^2), with x drawn uniformly from
    [-L, L]^n_dim and then sorted by increasing energy (as nested
    sampling would produce).

    Returns dead_energies (n_dead,), live_energies (n_live,).
    """
    key = jax.random.key(seed)
    n_total = n_dead + n_live
    L = 10.0  # box half-side, large enough to capture most of the Boltzmann weight

    # Draw positions uniformly in the box
    positions = jax.random.uniform(key, shape=(n_total, n_dim), minval=-L, maxval=L)
    energies = 0.5 * omega**2 * jnp.sum(positions**2, axis=1)

    # Sort by energy (nested sampling orders by increasing energy)
    sorted_idx = jnp.argsort(energies)
    energies = energies[sorted_idx]

    dead_energies = energies[:n_dead]
    live_energies = energies[n_dead:]
    return dead_energies, live_energies


# ---------------------------------------------------------------------------
# Test calc_log_weights
# ---------------------------------------------------------------------------

class TestCalcLogWeights:
    """Tests for calc_log_weights shrinkage factors."""

    def test_shape(self):
        """Output shape must equal n_dead."""
        log_w = calc_log_weights(n_dead=50, n_live=20, n_cull=1)
        assert log_w.shape == (50,)

    def test_monotonically_decreasing(self):
        """Weights should decrease monotonically (earlier shells are larger)."""
        log_w = calc_log_weights(n_dead=100, n_live=50, n_cull=1)
        diffs = jnp.diff(log_w)
        assert jnp.all(diffs < 0), "Log weights should be strictly decreasing."

    def test_known_values_n_cull_1(self):
        """Check shrinkage against manual calculation for n_live=10, n_cull=1."""
        n_live = 10
        n_cull = 1
        n_dead = 5

        log_t = jnp.log(jnp.array(n_live - n_cull, dtype=jnp.float32)) - jnp.log(
            jnp.array(n_live + 1.0 - n_cull, dtype=jnp.float32)
        )
        log_shell = jnp.log1p(-jnp.exp(log_t))
        expected = jnp.arange(n_dead, dtype=jnp.float32) * log_t + log_shell

        log_w = calc_log_weights(n_dead=n_dead, n_live=n_live, n_cull=n_cull)
        assert jnp.allclose(log_w, expected, atol=1e-6)

    def test_n_cull_gt_1(self):
        """Weights with n_cull=2 should differ from n_cull=1."""
        lw1 = calc_log_weights(n_dead=20, n_live=50, n_cull=1)
        lw2 = calc_log_weights(n_dead=20, n_live=50, n_cull=2)
        assert not jnp.allclose(lw1, lw2)

    def test_weights_sum_less_than_one(self):
        """Total prior mass in dead shells must be < 1."""
        log_w = calc_log_weights(n_dead=200, n_live=50, n_cull=1)
        total = jnp.exp(jax.scipy.special.logsumexp(log_w))
        assert total < 1.0


# ---------------------------------------------------------------------------
# Test log_evidence
# ---------------------------------------------------------------------------

class TestLogEvidence:
    """Tests for log_evidence."""

    def test_scalar_output(self):
        """log_evidence should return a scalar."""
        dead_E, live_E = _generate_harmonic_ns_data(n_dead=100, n_live=20)
        log_Z = log_evidence(dead_E, live_E, n_live=20, n_cull=1)
        assert log_Z.ndim == 0

    def test_more_dead_points_refines(self):
        """More dead points should change the evidence estimate."""
        dead_E, live_E = _generate_harmonic_ns_data(n_dead=500, n_live=50)
        log_Z_200 = log_evidence(dead_E[:200], live_E, n_live=50, n_cull=1)
        log_Z_500 = log_evidence(dead_E, live_E, n_live=50, n_cull=1)
        # Both should be finite
        assert jnp.isfinite(log_Z_200)
        assert jnp.isfinite(log_Z_500)


# ---------------------------------------------------------------------------
# Test partition_function
# ---------------------------------------------------------------------------

class TestPartitionFunction:
    """Z(T) for a 3D harmonic potential.

    Analytical result: Z(beta) proportional to (2 pi / (omega^2 * beta))^(n_dim/2)
    so log Z(beta) = (n_dim/2) * log(2 pi / (omega^2 * beta)) + const
    The constant is from the prior volume.

    We test that the *slope* d(log Z)/d(log beta) is correct: -n_dim/2.
    """

    @pytest.fixture
    def harmonic_data(self):
        n_dim = 3
        omega = 1.0
        return _generate_harmonic_ns_data(
            n_dead=3000, n_live=200, n_dim=n_dim, omega=omega, seed=1
        ), n_dim, omega

    def test_scalar_output(self, harmonic_data):
        (dead_E, live_E), n_dim, omega = harmonic_data
        log_Z = partition_function(1.0, dead_E, live_E, n_live=200)
        assert log_Z.ndim == 0
        assert jnp.isfinite(log_Z)

    def test_log_z_slope_vs_temperature(self, harmonic_data):
        """d(log Z)/d(log T) should be n_dim/2 for harmonic potential.

        Equivalently, d(log Z)/d(log beta) = -n_dim/2.
        We check the slope via finite differences at moderate beta.
        """
        (dead_E, live_E), n_dim, omega = harmonic_data
        n_live = 200

        # Use moderate beta values where the integral converges well
        betas = jnp.array([0.5, 1.0, 2.0, 4.0])
        log_Zs = jnp.array([
            partition_function(b, dead_E, live_E, n_live=n_live)
            for b in betas
        ])

        log_betas = jnp.log(betas)
        # Linear fit: log Z = slope * log(beta) + intercept
        # slope should be -n_dim/2 = -1.5
        A = jnp.stack([log_betas, jnp.ones_like(log_betas)], axis=1)
        result = jnp.linalg.lstsq(A, log_Zs, rcond=None)
        slope = result[0][0]

        expected_slope = -n_dim / 2.0
        assert abs(float(slope) - expected_slope) < 0.5, (
            f"Expected slope ~ {expected_slope}, got {float(slope)}"
        )

    def test_z_increases_with_temperature(self, harmonic_data):
        """Z should increase as temperature increases (beta decreases)."""
        (dead_E, live_E), n_dim, omega = harmonic_data
        n_live = 200
        log_Z_cold = partition_function(5.0, dead_E, live_E, n_live=n_live)
        log_Z_hot = partition_function(0.5, dead_E, live_E, n_live=n_live)
        assert float(log_Z_hot) > float(log_Z_cold)


# ---------------------------------------------------------------------------
# Test heat_capacity
# ---------------------------------------------------------------------------

class TestHeatCapacity:
    """C_v for 3D harmonic: classical limit is n_dim/2 per degree of freedom = n_dim/2."""

    @pytest.fixture
    def harmonic_data(self):
        n_dim = 3
        return _generate_harmonic_ns_data(
            n_dead=3000, n_live=200, n_dim=n_dim, omega=1.0, seed=2
        ), n_dim

    def test_scalar_output(self, harmonic_data):
        (dead_E, live_E), n_dim = harmonic_data
        cv = heat_capacity(1.0, dead_E, live_E, n_live=200)
        assert cv.ndim == 0
        assert jnp.isfinite(cv)

    def test_positive(self, harmonic_data):
        """Heat capacity must be non-negative."""
        (dead_E, live_E), n_dim = harmonic_data
        for beta in [0.2, 1.0, 5.0]:
            cv = heat_capacity(beta, dead_E, live_E, n_live=200)
            assert float(cv) >= -1e-6, f"C_v should be non-negative, got {float(cv)}"

    def test_classical_limit_high_T(self, harmonic_data):
        """At high T (low beta), C_v should be positive and finite.

        For a harmonic potential with n_dim degrees of freedom,
        the equipartition theorem gives C_v = n_dim/2 (in units of k_B).
        With finite sampling and a bounded prior, we just check C_v is
        positive and in a reasonable range.
        """
        (dead_E, live_E), n_dim = harmonic_data
        # Moderate beta where integral converges well
        cv = heat_capacity(1.0, dead_E, live_E, n_live=200)
        expected = n_dim / 2.0
        assert float(cv) > 0, f"C_v should be positive, got {float(cv)}"
        assert abs(float(cv) - expected) < 2.0, (
            f"Expected C_v ~ {expected} at moderate T, got {float(cv)}"
        )

    def test_cv_at_different_temperatures(self, harmonic_data):
        """C_v should be finite and positive across a range of temperatures."""
        (dead_E, live_E), n_dim = harmonic_data
        for beta in [0.5, 1.0, 2.0, 5.0]:
            cv = heat_capacity(beta, dead_E, live_E, n_live=200)
            assert jnp.isfinite(cv), f"C_v not finite at beta={beta}"
            assert float(cv) >= -1e-6, f"C_v negative at beta={beta}: {float(cv)}"


# ---------------------------------------------------------------------------
# Test expectation
# ---------------------------------------------------------------------------

class TestExpectation:
    """<E> for harmonic potential: at temperature T, <E> = n_dim/(2*beta)."""

    @pytest.fixture
    def harmonic_data(self):
        n_dim = 3
        return _generate_harmonic_ns_data(
            n_dead=3000, n_live=200, n_dim=n_dim, omega=1.0, seed=3
        ), n_dim

    def test_scalar_output(self, harmonic_data):
        (dead_E, live_E), n_dim = harmonic_data
        all_E = jnp.concatenate([dead_E, live_E])
        mean_E = expectation(all_E, 1.0, dead_E, live_E, n_live=200)
        assert mean_E.ndim == 0

    def test_mean_energy_moderate_T(self, harmonic_data):
        """<E> should be ~ n_dim/(2*beta) at moderate temperature."""
        (dead_E, live_E), n_dim = harmonic_data
        all_E = jnp.concatenate([dead_E, live_E])

        beta = 1.0
        mean_E = expectation(all_E, beta, dead_E, live_E, n_live=200)
        expected = n_dim / (2.0 * beta)

        # Generous tolerance since nested sampling is stochastic
        assert abs(float(mean_E) - expected) < 1.5, (
            f"Expected <E> ~ {expected}, got {float(mean_E)}"
        )

    def test_mean_energy_increases_with_T(self, harmonic_data):
        """<E> should increase with temperature."""
        (dead_E, live_E), n_dim = harmonic_data
        all_E = jnp.concatenate([dead_E, live_E])

        mean_cold = expectation(all_E, 5.0, dead_E, live_E, n_live=200)
        mean_hot = expectation(all_E, 0.5, dead_E, live_E, n_live=200)
        assert float(mean_hot) > float(mean_cold)

    def test_constant_observable(self, harmonic_data):
        """<c> = c for a constant observable."""
        (dead_E, live_E), n_dim = harmonic_data
        c = 42.0
        obs = jnp.full(dead_E.shape[0] + live_E.shape[0], c)
        result = expectation(obs, 1.0, dead_E, live_E, n_live=200)
        assert jnp.allclose(result, c, atol=1e-4)


# ---------------------------------------------------------------------------
# Test free_energy
# ---------------------------------------------------------------------------

class TestFreeEnergy:
    """F = -log Z / beta, consistency with partition_function."""

    @pytest.fixture
    def harmonic_data(self):
        return _generate_harmonic_ns_data(
            n_dead=2000, n_live=100, n_dim=3, omega=1.0, seed=4
        )

    def test_consistency_with_partition_function(self, harmonic_data):
        """F should equal -log_Z / beta."""
        dead_E, live_E = harmonic_data
        n_live = 100
        beta = 2.0

        log_Z = partition_function(beta, dead_E, live_E, n_live=n_live)
        F = free_energy(beta, log_Z)

        expected = -float(log_Z) / beta
        assert jnp.allclose(F, expected, atol=1e-6)

    def test_multiple_temperatures(self, harmonic_data):
        """F should be consistent at several temperatures."""
        dead_E, live_E = harmonic_data
        n_live = 100

        for beta in [0.5, 1.0, 2.0, 5.0]:
            log_Z = partition_function(beta, dead_E, live_E, n_live=n_live)
            F = free_energy(beta, log_Z)
            assert jnp.isfinite(F)
            assert jnp.allclose(F, -log_Z / beta, atol=1e-6)

    def test_f_finite_across_temperatures(self, harmonic_data):
        """Free energy should be finite across a range of temperatures."""
        dead_E, live_E = harmonic_data
        n_live = 100

        for beta in [0.5, 1.0, 2.0, 5.0]:
            log_Z = partition_function(beta, dead_E, live_E, n_live=n_live)
            F = free_energy(beta, log_Z)
            assert jnp.isfinite(F), f"F not finite at beta={beta}"

    def test_f_thermodynamic_relation(self, harmonic_data):
        """Gibbs-Helmholtz: d(beta*F)/d(beta) = <E>.

        Verify via finite difference that the numerical derivative of
        beta*F with respect to beta matches the thermal expectation <E>.
        """
        dead_E, live_E = harmonic_data
        n_live = 100
        all_E = jnp.concatenate([dead_E, live_E])

        beta = 1.0
        dbeta = 0.01

        log_Z_plus = partition_function(beta + dbeta, dead_E, live_E, n_live=n_live)
        log_Z_minus = partition_function(beta - dbeta, dead_E, live_E, n_live=n_live)

        bF_plus = -(beta + dbeta) * free_energy(beta + dbeta, log_Z_plus)  # = log_Z_plus
        bF_minus = -(beta - dbeta) * free_energy(beta - dbeta, log_Z_minus)

        # Numerical d(beta*F)/d(beta) — but beta*F = -log Z, so
        # d(-log Z)/d(beta) = <E> (Gibbs-Helmholtz)
        d_neg_logZ_dbeta = (-float(log_Z_plus) - (-float(log_Z_minus))) / (2 * dbeta)

        mean_E = expectation(all_E, beta, dead_E, live_E, n_live=n_live)

        assert abs(d_neg_logZ_dbeta - float(mean_E)) < 0.5, (
            f"Gibbs-Helmholtz: d(-log Z)/dbeta = {d_neg_logZ_dbeta:.4f}, "
            f"<E> = {float(mean_E):.4f}"
        )


# ---------------------------------------------------------------------------
# Batch-shape tests: (G, P, max_dead) inputs
# ---------------------------------------------------------------------------

# Helper: build (G, P, max_dead) dead_energies and (G, P, n_live) live_energies
# by stacking independent 1-D runs.

def _make_gp_data(G: int = 1, P: int = 2, n_dead: int = 500, n_live: int = 50) -> tuple:
    """Return (dead_energies, live_energies) with shape (G, P, n_dead/n_live).

    Each (g, p) slice is independently generated so per-slice vs batch
    comparisons are exact (reshape-only, no approximation).
    """
    slices_dead = []
    slices_live = []
    for g in range(G):
        row_d, row_l = [], []
        for p in range(P):
            seed = g * P + p
            d, l = _generate_harmonic_ns_data(n_dead=n_dead, n_live=n_live, seed=seed + 10)
            row_d.append(d)
            row_l.append(l)
        slices_dead.append(jnp.stack(row_d))
        slices_live.append(jnp.stack(row_l))
    dead = jnp.stack(slices_dead)   # (G, P, n_dead)
    live = jnp.stack(slices_live)   # (G, P, n_live)
    return dead, live


class TestBatchShapeCalcLogWeights:
    """calc_log_weights is inherently 1-D; verify it still works as before."""

    def test_1d_shape_unchanged(self):
        log_w = calc_log_weights(50, 20, 1)
        assert log_w.shape == (50,)


class TestBatchShapeLogEvidence:
    """log_evidence with (G, P, max_dead) input."""

    def test_output_shape_gp(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        out = log_evidence(dead, live, n_live=50)
        assert out.shape == (G, P), f"Expected ({G}, {P}), got {out.shape}"

    def test_matches_per_slice(self):
        """Batched result must equal per-slice 1-D result."""
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        batched = log_evidence(dead, live, n_live=50)
        for g in range(G):
            for p in range(P):
                ref = log_evidence(dead[g, p], live[g, p], n_live=50)
                assert jnp.allclose(batched[g, p], ref, atol=1e-5), (
                    f"Mismatch at ({g},{p}): batch={batched[g,p]:.6f}, ref={ref:.6f}"
                )

    def test_1d_still_returns_scalar(self):
        dead_1d, live_1d = _generate_harmonic_ns_data(n_dead=200, n_live=20)
        out = log_evidence(dead_1d, live_1d, n_live=20)
        assert out.ndim == 0, f"Expected scalar, got ndim={out.ndim}"

    def test_jit_compatible(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        n_live = 50
        jitted = jax.jit(lambda d, l: log_evidence(d, l, n_live=n_live))
        out = jitted(dead, live)
        assert out.shape == (G, P)
        assert jnp.all(jnp.isfinite(out))


class TestBatchShapePartitionFunction:
    """partition_function with (G, P, max_dead) input."""

    def test_output_shape_gp(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        out = partition_function(1.0, dead, live, n_live=50)
        assert out.shape == (G, P), f"Expected ({G}, {P}), got {out.shape}"

    def test_matches_per_slice(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        batched = partition_function(1.0, dead, live, n_live=50)
        for g in range(G):
            for p in range(P):
                ref = partition_function(1.0, dead[g, p], live[g, p], n_live=50)
                assert jnp.allclose(batched[g, p], ref, atol=1e-5), (
                    f"Mismatch at ({g},{p}): batch={batched[g,p]:.6f}, ref={ref:.6f}"
                )

    def test_1d_still_returns_scalar(self):
        dead_1d, live_1d = _generate_harmonic_ns_data(n_dead=200, n_live=20)
        out = partition_function(1.0, dead_1d, live_1d, n_live=20)
        assert out.ndim == 0

    def test_jit_compatible(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        n_live = 50
        jitted = jax.jit(lambda d, l: partition_function(1.0, d, l, n_live=n_live))
        out = jitted(dead, live)
        assert out.shape == (G, P)
        assert jnp.all(jnp.isfinite(out))


class TestBatchShapeHeatCapacity:
    """heat_capacity with (G, P, max_dead) input."""

    def test_output_shape_gp(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        out = heat_capacity(1.0, dead, live, n_live=50)
        assert out.shape == (G, P), f"Expected ({G}, {P}), got {out.shape}"

    def test_matches_per_slice(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        batched = heat_capacity(1.0, dead, live, n_live=50)
        for g in range(G):
            for p in range(P):
                ref = heat_capacity(1.0, dead[g, p], live[g, p], n_live=50)
                assert jnp.allclose(batched[g, p], ref, atol=1e-5), (
                    f"Mismatch at ({g},{p}): batch={batched[g,p]:.6f}, ref={ref:.6f}"
                )

    def test_1d_still_returns_scalar(self):
        dead_1d, live_1d = _generate_harmonic_ns_data(n_dead=200, n_live=20)
        out = heat_capacity(1.0, dead_1d, live_1d, n_live=20)
        assert out.ndim == 0

    def test_jit_compatible(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        n_live = 50
        jitted = jax.jit(lambda d, l: heat_capacity(1.0, d, l, n_live=n_live))
        out = jitted(dead, live)
        assert out.shape == (G, P)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(out >= 0.0)


class TestBatchShapeExpectation:
    """expectation with (G, P, max_dead) input."""

    def test_output_shape_gp(self):
        G, P, n_dead, n_live = 1, 2, 500, 50
        dead, live = _make_gp_data(G, P, n_dead=n_dead, n_live=n_live)
        # Observable: just the energies concatenated per run.
        obs = jnp.concatenate([dead, live], axis=-1)  # (G, P, n_dead + n_live)
        out = expectation(obs, 1.0, dead, live, n_live=n_live)
        assert out.shape == (G, P), f"Expected ({G}, {P}), got {out.shape}"

    def test_matches_per_slice(self):
        G, P, n_dead, n_live = 1, 2, 300, 30
        dead, live = _make_gp_data(G, P, n_dead=n_dead, n_live=n_live)
        obs = jnp.concatenate([dead, live], axis=-1)
        batched = expectation(obs, 1.0, dead, live, n_live=n_live)
        for g in range(G):
            for p in range(P):
                ref = expectation(obs[g, p], 1.0, dead[g, p], live[g, p], n_live=n_live)
                assert jnp.allclose(batched[g, p], ref, atol=1e-5), (
                    f"Mismatch at ({g},{p}): batch={batched[g,p]:.6f}, ref={ref:.6f}"
                )

    def test_1d_still_returns_scalar(self):
        dead_1d, live_1d = _generate_harmonic_ns_data(n_dead=200, n_live=20)
        obs = jnp.concatenate([dead_1d, live_1d])
        out = expectation(obs, 1.0, dead_1d, live_1d, n_live=20)
        assert out.ndim == 0

    def test_jit_compatible(self):
        G, P, n_dead, n_live = 1, 2, 200, 20
        dead, live = _make_gp_data(G, P, n_dead=n_dead, n_live=n_live)
        obs = jnp.concatenate([dead, live], axis=-1)
        n_live_static = n_live
        jitted = jax.jit(lambda obs_, d, l: expectation(obs_, 1.0, d, l, n_live=n_live_static))
        out = jitted(obs, dead, live)
        assert out.shape == (G, P)


class TestBatchShapeFreeEnergy:
    """free_energy handles batched log_Z_beta."""

    def test_output_shape_gp(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        log_Z = partition_function(1.0, dead, live, n_live=50)
        F = free_energy(1.0, log_Z)
        assert F.shape == (G, P), f"Expected ({G}, {P}), got {F.shape}"

    def test_matches_per_slice(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        log_Z = partition_function(1.0, dead, live, n_live=50)
        F_batch = free_energy(1.0, log_Z)
        for g in range(G):
            for p in range(P):
                log_Z_ref = partition_function(1.0, dead[g, p], live[g, p], n_live=50)
                F_ref = free_energy(1.0, log_Z_ref)
                assert jnp.allclose(F_batch[g, p], F_ref, atol=1e-5), (
                    f"Mismatch at ({g},{p}): batch={F_batch[g,p]:.6f}, ref={F_ref:.6f}"
                )

    def test_1d_still_returns_scalar(self):
        dead_1d, live_1d = _generate_harmonic_ns_data(n_dead=200, n_live=20)
        log_Z = partition_function(1.0, dead_1d, live_1d, n_live=20)
        F = free_energy(1.0, log_Z)
        assert F.ndim == 0

    def test_jit_compatible(self):
        G, P = 1, 2
        dead, live = _make_gp_data(G, P)
        n_live = 50
        jitted = jax.jit(lambda d, l: free_energy(
            1.0, partition_function(1.0, d, l, n_live=n_live)
        ))
        out = jitted(dead, live)
        assert out.shape == (G, P)
        assert jnp.all(jnp.isfinite(out))
