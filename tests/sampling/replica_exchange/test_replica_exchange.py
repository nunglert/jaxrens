"""Tests for the replica exchange driver and the EnsembleSwap kernel.

The tests cover get_swap_pairs, the energy re-basing and acceptance of
EnsembleSwap without ensemble params and with pressures, and
replica_exchange_step, also under jit. Semi-grand RE is tested in
test_semigrand.py and XRENS in test_xrens.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.moves.replica_exchange import (
    EnsembleSwap,
    SwapKernel,
    SwapWalker,
    get_swap_pairs,
    make_swap_kernel,
    replica_exchange_step,
)
from jaxrens.state.config import InterREConfig


class TestGetSwapPairs:
    """Check the even (0,1), (2,3), ... and odd (1,2), (3,4), ... pairs."""

    def test_even_phase_4_runs(self):
        pairs = get_swap_pairs(4, 0)
        expected = jnp.array([[0, 1], [2, 3]])
        assert jnp.array_equal(pairs, expected)

    def test_odd_phase_4_runs(self):
        pairs = get_swap_pairs(4, 1)
        # Runs are 0..3, so (3, 4) does not exist: arange(1, 3, 2) = [1].
        expected = jnp.array([[1, 2]])
        assert jnp.array_equal(pairs, expected)

    def test_even_phase_5_runs(self):
        pairs = get_swap_pairs(5, 0)
        expected = jnp.array([[0, 1], [2, 3]])
        assert jnp.array_equal(pairs, expected)

    def test_odd_phase_5_runs(self):
        pairs = get_swap_pairs(5, 1)
        expected = jnp.array([[1, 2], [3, 4]])
        assert jnp.array_equal(pairs, expected)

    def test_even_phase_2_runs(self):
        pairs = get_swap_pairs(2, 0)
        expected = jnp.array([[0, 1]])
        assert jnp.array_equal(pairs, expected)

    def test_odd_phase_2_runs(self):
        pairs = get_swap_pairs(2, 1)
        assert pairs.shape[0] == 0

    def test_even_phase_3_runs(self):
        pairs = get_swap_pairs(3, 0)
        expected = jnp.array([[0, 1]])
        assert jnp.array_equal(pairs, expected)

    def test_odd_phase_3_runs(self):
        pairs = get_swap_pairs(3, 1)
        expected = jnp.array([[1, 2]])
        assert jnp.array_equal(pairs, expected)

    def test_single_run_no_pairs(self):
        pairs = get_swap_pairs(1, 0)
        assert pairs.shape[0] == 0

    def test_pairs_shape(self):
        pairs = get_swap_pairs(6, 0)
        assert pairs.shape == (3, 2)
        pairs = get_swap_pairs(6, 1)
        assert pairs.shape == (2, 2)


def _walker(energy, volume=None, n_atoms=2):
    cell = None if volume is None else jnp.eye(3) * (volume ** (1.0 / 3.0))
    return SwapWalker(
        positions=jnp.zeros((n_atoms, 3)),
        types=jnp.zeros(n_atoms, dtype=jnp.int32),
        energy=jnp.asarray(energy, dtype=jnp.float32),
        cell=cell,
    )


def _accepts(stored, emax, volumes=None, pressures=None):
    """Run EnsembleSwap for one pair and apply the driver's indicator.

    ``stored`` are the walkers' stored energies in their own run's frame.
    """
    v = volumes if volumes is not None else (None, None)
    w_a, w_b = _walker(stored[0], v[0]), _walker(stored[1], v[1])
    ens = (
        ({}, {})
        if pressures is None
        else tuple({"pressure": jnp.asarray(p)} for p in pressures)
    )
    into_a, into_b = EnsembleSwap().propose(
        w_a, w_b, ens[0], ens[1], jax.random.key(0), None
    )
    return bool((into_a.energy < emax[0]) & (into_b.energy < emax[1]))


class TestEnsembleSwapPlainEnergy:
    """Check EnsembleSwap without ensemble params.

    The energies move unchanged, and a swap is accepted only when each
    incoming walker is strictly below the receiving run's Emax.
    """

    def test_accept_when_energies_within_bounds(self):
        assert _accepts((1.0, 2.0), (3.0, 3.0))

    def test_reject_when_energy_a_exceeds_emax_b(self):
        assert not _accepts((4.0, 1.0), (5.0, 3.0))

    def test_reject_when_energy_b_exceeds_emax_a(self):
        assert not _accepts((1.0, 4.0), (3.0, 5.0))

    def test_reject_when_both_exceed(self):
        assert not _accepts((5.0, 5.0), (3.0, 3.0))

    def test_asymmetric_emax(self):
        # Walker A goes to run B (Emax 2), walker B to run A (Emax 10).
        assert _accepts((1.5, 9.0), (10.0, 2.0))
        assert not _accepts((2.5, 9.0), (10.0, 2.0))

    def test_boundary_energy_equals_emax_rejects(self):
        assert not _accepts((3.0, 1.0), (3.0, 3.0))

    def test_propose_exchanges_walkers(self):
        w_a = _walker(1.0)._replace(positions=jnp.ones((2, 3)))
        w_b = _walker(2.0)
        into_a, into_b = EnsembleSwap().propose(
            w_a, w_b, {}, {}, jax.random.key(0), None
        )
        assert jnp.array_equal(into_a.positions, w_b.positions)
        assert jnp.array_equal(into_b.positions, w_a.positions)
        assert float(into_a.energy) == 2.0
        assert float(into_b.energy) == 1.0


class TestEnsembleSwapPressure:
    """Check the pressure re-basing of EnsembleSwap.

    Stored energies are enthalpies ``U + P_self·V``; the walker entering
    run r carries ``U + P_r·V``.
    """

    @staticmethod
    def _stored(u, p, v):
        return u + p * v

    def test_pressure_accept(self):
        # U = 1, V = 1, P = 1 on both sides: H = 2 < 5.
        s = self._stored(1.0, 1.0, 1.0)
        assert _accepts((s, s), (5.0, 5.0), (1.0, 1.0), (1.0, 1.0))

    def test_pressure_reject_enthalpy_too_high(self):
        # Walker A into run B: 1 + 10·1 = 11 > 5.
        assert not _accepts(
            (self._stored(1.0, 1.0, 1.0), self._stored(1.0, 10.0, 1.0)),
            (5.0, 5.0),
            (1.0, 1.0),
            (1.0, 10.0),
        )

    def test_pressure_asymmetric(self):
        # A into B: 1 + 2·3 = 7 < 10; B into A: 2 + 1·4 = 6 < 10.
        assert _accepts(
            (self._stored(1.0, 1.0, 3.0), self._stored(2.0, 2.0, 4.0)),
            (10.0, 10.0),
            (3.0, 4.0),
            (1.0, 2.0),
        )

    def test_pressure_reject_one_direction(self):
        # B into A: 2 + 1·4 = 6 > Emax_A = 5.
        assert not _accepts(
            (self._stored(1.0, 1.0, 3.0), self._stored(2.0, 2.0, 4.0)),
            (5.0, 10.0),
            (3.0, 4.0),
            (1.0, 2.0),
        )

    def test_equal_pressures_si16_magnitudes_accept(self):
        """Equal U, V and P on both sides, at Si16 magnitudes, is a physical
        no-op, so a loose Emax accepts. A P·V term counted twice rejects it.
        """
        p, v, u = 0.006242, 320.0, -83.0
        s = self._stored(u, p, v)
        assert _accepts((s, s), (u + 5.0, u + 5.0), (v, v), (p, p))

    def test_pv_not_double_counted_at_tight_emax(self):
        """Adjacent pressures, same U: each incoming walker carries exactly
        one PV term (≈ -81 and ≈ -83 eV), below Emax = -80; a spurious extra
        self-PV would lift both to ≈ -79 and reject."""
        p_a, p_b, v, u = 0.006242, 0.01248, 320.0, -85.0
        assert _accepts(
            (self._stored(u, p_a, v), self._stored(u, p_b, v)),
            (-80.0, -80.0),
            (v, v),
            (p_a, p_b),
        )

    def test_rebased_energy_values(self):
        w_a, w_b = _walker(1.0 + 1.0 * 3.0, 3.0), _walker(2.0 + 2.0 * 4.0, 4.0)
        into_a, into_b = EnsembleSwap().propose(
            w_a,
            w_b,
            {"pressure": jnp.asarray(1.0)},
            {"pressure": jnp.asarray(2.0)},
            jax.random.key(0),
            None,
        )
        assert float(into_a.energy) == pytest.approx(2.0 + 1.0 * 4.0)
        assert float(into_b.energy) == pytest.approx(1.0 + 2.0 * 3.0)


class TestSwapKernelABC:
    """Check the SwapKernel base class and the param keys of EnsembleSwap."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SwapKernel()

    def test_param_keys(self):
        assert EnsembleSwap().param_keys == ("pressure", "chemical_potentials")
        sg = EnsembleSwap(required_params=("chemical_potentials",))
        assert sg.required_params == ("chemical_potentials",)
        assert sg.optional_params == ("pressure",)


class TestMakeSwapKernel:
    """Check the kernel make_swap_kernel builds for the pressure flavor and
    the error for an unknown flavor. Semi-grand is in test_semigrand.py,
    XRENS in test_xrens.py.
    """

    def test_pressure_builds_ensemble_swap_without_required_params(self):
        kernel = make_swap_kernel(InterREConfig(flavor="pressure"))
        assert isinstance(kernel, EnsembleSwap)
        assert kernel.required_params == ()

    def test_unknown_flavor_raises(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            make_swap_kernel(InterREConfig(flavor="foo"))


def _make_re_data(n_runs, n_walkers, n_atoms=2, with_cells=True):
    """Dummy population for replica_exchange_step."""
    positions = jax.random.normal(
        jax.random.key(0), (n_runs, n_walkers, n_atoms, 3)
    )
    types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
    energies = jax.random.uniform(
        jax.random.key(1), (n_runs, n_walkers), minval=-5.0, maxval=0.0
    )
    cells = None
    if with_cells:
        cells = jnp.broadcast_to(
            5.0 * jnp.eye(3), (n_runs, n_walkers, 3, 3)
        ).copy()
    return SwapWalker(positions, types, energies, cells)


def _step(walkers, emax, run_params=None, n_swap_cycles=1, seed=0):
    return replica_exchange_step(
        jax.random.key(seed),
        walkers,
        jnp.asarray(emax),
        run_params or {},
        EnsembleSwap(),
        n_swap_cycles=n_swap_cycles,
    )


class TestReplicaExchangeStep:
    """Check the swap schedule, the counters and the write-back of
    replica_exchange_step.
    """

    def test_single_run_no_swap(self):
        walkers = _make_re_data(1, 4)
        new, info = _step(walkers, [0.0])
        assert jnp.array_equal(new.positions, walkers.positions)
        assert jnp.array_equal(new.energy, walkers.energy)
        assert int(info["n_attempted"]) == 0
        assert int(info["n_accepted"]) == 0
        assert info["n_accepted_per_pair"].shape == (0,)

    def test_two_runs_guaranteed_swap(self):
        walkers = _make_re_data(2, 1)
        new, info = _step(walkers, [100.0, 100.0], seed=7)
        # One cycle: the even phase has the only pair for 2 runs.
        assert int(info["n_attempted"]) == 1
        assert int(info["n_accepted"]) == 1
        assert jnp.allclose(new.energy[0, 0], walkers.energy[1, 0])
        assert jnp.allclose(new.energy[1, 0], walkers.energy[0, 0])
        assert jnp.array_equal(new.positions[0, 0], walkers.positions[1, 0])

    def test_two_runs_guaranteed_reject(self):
        walkers = _make_re_data(2, 1)
        new, info = _step(walkers, [-10.0, -10.0], seed=7)
        assert int(info["n_attempted"]) == 1
        assert int(info["n_accepted"]) == 0
        assert jnp.array_equal(new.energy, walkers.energy)
        assert jnp.array_equal(new.positions, walkers.positions)

    def test_returns_correct_shapes(self):
        walkers = _make_re_data(4, 3, 2)
        new, info = _step(walkers, [0.0] * 4)
        assert jax.tree.map(jnp.shape, new) == jax.tree.map(jnp.shape, walkers)
        assert info["n_accepted_per_pair"].shape == (3,)
        assert info["n_attempted_per_pair"].shape == (3,)

    def test_no_cell(self):
        walkers = _make_re_data(3, 2, with_cells=False)
        new, _ = _step(walkers, [100.0] * 3, seed=5)
        assert new.cell is None
        assert new.positions.shape == walkers.positions.shape

    def test_multiple_swap_cycles(self):
        walkers = _make_re_data(3, 2)
        _, info = _step(walkers, [100.0] * 3, n_swap_cycles=3, seed=13)
        # 3 runs: one even and one odd pair => 2 attempts per cycle.
        assert int(info["n_attempted"]) == 6
        assert info["n_attempted_per_pair"].tolist() == [3, 3]
        assert int(info["n_energy_evals"]) == 0

    def test_energy_conservation(self):
        """Without ensemble params swaps only move energies around."""
        walkers = _make_re_data(4, 3)
        new, _ = _step(walkers, [100.0] * 4, n_swap_cycles=5, seed=42)
        assert jnp.allclose(
            jnp.sort(walkers.energy.ravel()), jnp.sort(new.energy.ravel())
        )

    def test_high_pressure_rejects_swaps(self):
        walkers = _make_re_data(2, 1)
        # Stored at P = 0, the receiving runs have P = 1e6: PV >> Emax.
        stored = {"pressure": jnp.array([0.0, 1e6])}
        _, info = _step(walkers, [0.0, 0.0], stored)
        assert int(info["n_accepted"]) == 0

    def test_stored_energy_in_receiving_frame(self):
        """After an accepted pressure swap each stored energy is
        ``U + P_run·V`` of the run the walker is now in."""
        walkers = _make_re_data(2, 1)
        p = jnp.array([0.1, 0.5])
        v = jnp.linalg.det(walkers.cell)
        u = walkers.energy
        walkers = walkers._replace(energy=u + p[:, None] * v)
        new, info = _step(walkers, [1e6, 1e6], {"pressure": p})
        assert int(info["n_accepted"]) == 1
        expected = u[::-1] + p[:, None] * v[::-1]
        assert jnp.allclose(new.energy, expected, atol=1e-5)


class TestJITCompatibility:
    """Check that get_swap_pairs and replica_exchange_step run under jit."""

    def test_get_swap_pairs_jit(self):
        pairs = jax.jit(get_swap_pairs, static_argnums=(0, 1))(4, 0)
        assert jnp.array_equal(pairs, jnp.array([[0, 1], [2, 3]]))

    def test_replica_exchange_step_jit(self):
        walkers = _make_re_data(4, 3)
        p = {"pressure": jnp.array([0.0, 0.1, 0.2, 0.3])}
        kernel = EnsembleSwap()

        @jax.jit
        def jitted(key, walkers, emax, params):
            return replica_exchange_step(
                key, walkers, emax, params, kernel, n_swap_cycles=2
            )

        new, info = jitted(jax.random.key(0), walkers, jnp.zeros(4), p)
        eager, eager_info = replica_exchange_step(
            jax.random.key(0),
            walkers,
            jnp.zeros(4),
            p,
            kernel,
            n_swap_cycles=2,
        )
        assert jnp.array_equal(new.positions, eager.positions)
        assert int(info["n_accepted"]) == int(eager_info["n_accepted"])
