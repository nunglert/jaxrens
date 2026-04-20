"""Test replica exchange moves across parallel NS runs.

Covers: get_swap_pairs, perform_swap, replica_exchange_step,
pressure-based RE, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.moves.replica_exchange import (
    PressureRENSSwap,
    SwapKernel,
    get_swap_pairs,
    perform_swap,
    replica_exchange_step,
)


# ---------------------------------------------------------------------------
# get_swap_pairs
# ---------------------------------------------------------------------------


class TestGetSwapPairs:
    def test_even_phase_4_runs(self):
        pairs = get_swap_pairs(4, 0)
        expected = jnp.array([[0, 1], [2, 3]])
        assert jnp.array_equal(pairs, expected)

    def test_odd_phase_4_runs(self):
        pairs = get_swap_pairs(4, 1)
        expected = jnp.array([[1, 2], [3, 4]])
        # With 4 runs, odd phase starts at 1 with step 2: indices 1, 3
        # But run index 4 doesn't exist (n_runs=4 means indices 0..3)
        # The function uses arange(phase, n_runs-1, 2) => arange(1, 3, 2) => [1]
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
        # arange(1, 1, 2) => empty
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


# ---------------------------------------------------------------------------
# perform_swap
# ---------------------------------------------------------------------------


class TestPerformSwap:
    def test_accept_when_energies_within_bounds(self):
        # E_A=1.0 < Emax_j=5.0 AND E_B=2.0 < Emax_i=5.0 => accept
        energies = jnp.array([1.0, 2.0])
        emax = jnp.array([5.0, 5.0])
        assert perform_swap(energies, emax)

    def test_reject_when_energy_a_exceeds_emax_j(self):
        # E_A=10.0 > Emax_j=5.0 => reject
        energies = jnp.array([10.0, 2.0])
        emax = jnp.array([5.0, 5.0])
        assert not perform_swap(energies, emax)

    def test_reject_when_energy_b_exceeds_emax_i(self):
        # E_B=10.0 > Emax_i=5.0 => reject
        energies = jnp.array([2.0, 10.0])
        emax = jnp.array([5.0, 5.0])
        assert not perform_swap(energies, emax)

    def test_reject_when_both_exceed(self):
        energies = jnp.array([10.0, 10.0])
        emax = jnp.array([5.0, 5.0])
        assert not perform_swap(energies, emax)

    def test_asymmetric_emax_accept(self):
        # E_A=3.0 < Emax_j=4.0 AND E_B=1.0 < Emax_i=2.0 => accept
        energies = jnp.array([3.0, 1.0])
        emax = jnp.array([2.0, 4.0])
        assert perform_swap(energies, emax)

    def test_asymmetric_emax_reject(self):
        # E_A=3.0 < Emax_j=4.0 BUT E_B=3.0 > Emax_i=2.0 => reject
        energies = jnp.array([3.0, 3.0])
        emax = jnp.array([2.0, 4.0])
        assert not perform_swap(energies, emax)

    def test_boundary_energy_equals_emax_rejects(self):
        # E_A == Emax_j uses strict <, so equal should reject
        energies = jnp.array([5.0, 2.0])
        emax = jnp.array([5.0, 5.0])
        assert not perform_swap(energies, emax)


# ---------------------------------------------------------------------------
# perform_swap with pressure (enthalpy-based)
# ---------------------------------------------------------------------------


class TestPerformSwapPressure:
    def test_pressure_accept(self):
        # H_A_in_j = E_A + P_j * V_A = 1.0 + 1.0*1.0 = 2.0 < Emax_j=5.0
        # H_B_in_i = E_B + P_i * V_B = 1.0 + 1.0*1.0 = 2.0 < Emax_i=5.0
        energies = jnp.array([1.0, 1.0])
        emax = jnp.array([5.0, 5.0])
        volumes = jnp.array([1.0, 1.0])
        pressures = jnp.array([1.0, 1.0])
        assert perform_swap(energies, emax, volumes, pressures)

    def test_pressure_reject_enthalpy_too_high(self):
        # H_A_in_j = E_A + P_j * V_A = 1.0 + 10.0*1.0 = 11.0 > Emax_j=5.0
        energies = jnp.array([1.0, 1.0])
        emax = jnp.array([5.0, 5.0])
        volumes = jnp.array([1.0, 1.0])
        pressures = jnp.array([1.0, 10.0])
        assert not perform_swap(energies, emax, volumes, pressures)

    def test_pressure_asymmetric(self):
        # H_A_in_j = 1.0 + 2.0*3.0 = 7.0 < 10.0 => ok
        # H_B_in_i = 2.0 + 1.0*4.0 = 6.0 < 10.0 => ok
        energies = jnp.array([1.0, 2.0])
        emax = jnp.array([10.0, 10.0])
        volumes = jnp.array([3.0, 4.0])
        pressures = jnp.array([1.0, 2.0])
        assert perform_swap(energies, emax, volumes, pressures)

    def test_pressure_reject_one_direction(self):
        # H_A_in_j = 1.0 + 2.0*3.0 = 7.0 < 10.0 => ok
        # H_B_in_i = 2.0 + 1.0*4.0 = 6.0 > 5.0 => reject
        energies = jnp.array([1.0, 2.0])
        emax = jnp.array([5.0, 10.0])
        volumes = jnp.array([3.0, 4.0])
        pressures = jnp.array([1.0, 2.0])
        assert not perform_swap(energies, emax, volumes, pressures)


# ---------------------------------------------------------------------------
# replica_exchange_step
# ---------------------------------------------------------------------------


def _make_re_data(n_runs, n_walkers, n_atoms=2):
    """Helper to create dummy arrays for replica_exchange_step."""
    positions = jax.random.normal(
        jax.random.key(0), (n_runs, n_walkers, n_atoms, 3)
    )
    types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
    energies = jax.random.uniform(
        jax.random.key(1), (n_runs, n_walkers), minval=-5.0, maxval=0.0
    )
    cells = jnp.broadcast_to(
        5.0 * jnp.eye(3), (n_runs, n_walkers, 3, 3)
    ).copy()
    return positions, types, energies, cells


class TestReplicaExchangeStep:
    def test_single_run_no_swap(self):
        """With 1 run, nothing should change."""
        pos, types, ene, cells = _make_re_data(1, 4)
        emax = jnp.array([0.0])
        key = jax.random.key(99)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, cells, emax
        )
        assert jnp.array_equal(new_pos, pos)
        assert jnp.array_equal(new_ene, ene)
        assert int(info["n_attempted"]) == 0
        assert int(info["n_accepted"]) == 0

    def test_two_runs_guaranteed_swap(self):
        """With very high Emax, all swaps should be accepted."""
        n_runs, n_walkers = 2, 1
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        # Very high Emax so both energies are always below both constraints
        emax = jnp.array([100.0, 100.0])
        key = jax.random.key(7)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, cells, emax
        )
        # With 1 cycle, even phase has 1 pair, odd phase has 0 pairs for 2 runs
        assert int(info["n_attempted"]) == 1
        assert int(info["n_accepted"]) == 1
        # Walkers should be swapped: energies exchanged
        assert jnp.allclose(new_ene[0, 0], ene[1, 0])
        assert jnp.allclose(new_ene[1, 0], ene[0, 0])

    def test_two_runs_guaranteed_reject(self):
        """With very low Emax, no swaps should be accepted."""
        n_runs, n_walkers = 2, 1
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        # Emax very low -- energies are in [-5, 0], setting emax to -10 rejects
        emax = jnp.array([-10.0, -10.0])
        key = jax.random.key(7)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, cells, emax
        )
        assert int(info["n_attempted"]) == 1
        assert int(info["n_accepted"]) == 0
        # Nothing should change
        assert jnp.array_equal(new_ene, ene)
        assert jnp.array_equal(new_pos, pos)

    def test_returns_correct_shapes(self):
        n_runs, n_walkers, n_atoms = 4, 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers, n_atoms)
        emax = jnp.array([0.0, 0.0, 0.0, 0.0])
        key = jax.random.key(0)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, cells, emax
        )
        assert new_pos.shape == (n_runs, n_walkers, n_atoms, 3)
        assert new_types.shape == types.shape
        assert new_ene.shape == (n_runs, n_walkers)
        assert new_cells.shape == (n_runs, n_walkers, 3, 3)

    def test_no_cell(self):
        """Works when cells are None."""
        n_runs, n_walkers = 3, 2
        pos, types, ene, _ = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(5)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, None, emax
        )
        assert new_cells is None
        assert new_pos.shape == pos.shape

    def test_multiple_swap_cycles(self):
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(13)
        _, _, _, _, info = replica_exchange_step(
            key, pos, types, ene, cells, emax, n_swap_cycles=3
        )
        # 3 runs: even pairs=1, odd pairs=1 => 2 attempts per cycle, 3 cycles = 6
        assert int(info["n_attempted"]) == 6

    def test_energy_conservation(self):
        """Total energy across all walkers is conserved (swaps only move energy)."""
        n_runs, n_walkers = 4, 3
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0, 100.0])
        key = jax.random.key(42)
        _, _, new_ene, _, _ = replica_exchange_step(
            key, pos, types, ene, cells, emax, n_swap_cycles=5
        )
        # Total energy should be preserved (just rearranged)
        assert jnp.allclose(jnp.sort(ene.ravel()), jnp.sort(new_ene.ravel()))

    def test_types_per_run_not_per_walker(self):
        """When types has shape (P, n_atoms) instead of (P, K, n_atoms)."""
        n_runs, n_walkers, n_atoms = 3, 2, 2
        pos, _, ene, cells = _make_re_data(n_runs, n_walkers, n_atoms)
        types_shared = jnp.zeros((n_runs, n_atoms), dtype=jnp.int32)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(0)
        _, new_types, _, _, _ = replica_exchange_step(
            key, pos, types_shared, ene, cells, emax
        )
        # Types should remain unchanged when not per-walker
        assert jnp.array_equal(new_types, types_shared)


# ---------------------------------------------------------------------------
# Pressure-based replica_exchange_step
# ---------------------------------------------------------------------------


class TestReplicaExchangeStepPressure:
    def test_pressure_re_runs(self):
        """replica_exchange_step with pressures should run without error."""
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        pressures = jnp.array([0.0, 1.0, 2.0])
        key = jax.random.key(0)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, cells, emax, pressures=pressures
        )
        assert new_pos.shape == pos.shape
        assert int(info["n_attempted"]) > 0

    def test_high_pressure_rejects_swaps(self):
        """Very high pressure makes enthalpies exceed Emax, rejecting swaps."""
        n_runs, n_walkers = 2, 1
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        # Set tight emax and very high pressure so enthalpy = E + P*V >> emax
        emax = jnp.array([0.0, 0.0])
        pressures = jnp.array([1e6, 1e6])
        key = jax.random.key(0)
        _, _, new_ene, _, info = replica_exchange_step(
            key, pos, types, ene, cells, emax, pressures=pressures
        )
        assert int(info["n_accepted"]) == 0


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    def test_get_swap_pairs_jit(self):
        pairs = jax.jit(get_swap_pairs, static_argnums=(0, 1))(4, 0)
        expected = jnp.array([[0, 1], [2, 3]])
        assert jnp.array_equal(pairs, expected)

    def test_perform_swap_jit(self):
        energies = jnp.array([1.0, 2.0])
        emax = jnp.array([5.0, 5.0])
        result = jax.jit(perform_swap)(energies, emax)
        assert result

    def test_perform_swap_pressure_jit(self):
        energies = jnp.array([1.0, 1.0])
        emax = jnp.array([5.0, 5.0])
        volumes = jnp.array([1.0, 1.0])
        pressures = jnp.array([1.0, 1.0])
        result = jax.jit(perform_swap)(energies, emax, volumes, pressures)
        assert result

    def test_replica_exchange_step_jit(self):
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(0)
        jitted = jax.jit(replica_exchange_step, static_argnames=("n_swap_cycles",))
        new_pos, _, new_ene, _, info = jitted(
            key, pos, types, ene, cells, emax, n_swap_cycles=1
        )
        assert new_pos.shape == pos.shape
        assert int(info["n_attempted"]) > 0


# ---------------------------------------------------------------------------
# SwapKernel abstraction
# ---------------------------------------------------------------------------


class TestSwapKernelABC:
    """SwapKernel is an ABC; instantiating it directly should fail."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SwapKernel()  # type: ignore[abstract]

    def test_pressure_rens_swap_is_subclass(self):
        assert issubclass(PressureRENSSwap, SwapKernel)


# ---------------------------------------------------------------------------
# PressureRENSSwap.accept — golden equivalence against perform_swap
# ---------------------------------------------------------------------------


class TestPressureRENSSwapAccept:
    """PressureRENSSwap.accept must produce identical results to perform_swap."""

    def _kernel_accept(self, energies, emax, volumes=None, pressures=None):
        """Call PressureRENSSwap.accept with the same arguments as perform_swap."""
        kernel = PressureRENSSwap()
        proposed = {
            "energy_a": energies[0],
            "energy_b": energies[1],
        }
        if volumes is not None and pressures is not None:
            # Build diagonal cell matrices so det == volume (same as perform_swap shim)
            def _vol_to_cell(v):
                side = jnp.cbrt(v)
                return jnp.diag(jnp.array([side, side, side]))

            proposed["cell_a"] = _vol_to_cell(volumes[0])
            proposed["cell_b"] = _vol_to_cell(volumes[1])
            ens_a = {"pressure": pressures[0]}
            ens_b = {"pressure": pressures[1]}
        else:
            ens_a = {}
            ens_b = {}
        return kernel.accept(proposed, emax[0], emax[1], ens_a, ens_b)

    def test_accept_matches_perform_swap_simple(self):
        energies = jnp.array([1.0, 2.0])
        emax = jnp.array([5.0, 5.0])
        assert bool(self._kernel_accept(energies, emax)) == bool(
            perform_swap(energies, emax)
        )

    def test_reject_matches_perform_swap_simple(self):
        energies = jnp.array([10.0, 2.0])
        emax = jnp.array([5.0, 5.0])
        assert bool(self._kernel_accept(energies, emax)) == bool(
            perform_swap(energies, emax)
        )

    def test_accept_matches_perform_swap_pressure(self):
        energies = jnp.array([1.0, 1.0])
        emax = jnp.array([5.0, 5.0])
        volumes = jnp.array([1.0, 1.0])
        pressures = jnp.array([1.0, 1.0])
        assert bool(self._kernel_accept(energies, emax, volumes, pressures)) == bool(
            perform_swap(energies, emax, volumes, pressures)
        )

    def test_reject_matches_perform_swap_pressure(self):
        # H_A_in_j = 1.0 + 10.0*1.0 = 11.0 > 5.0 => reject
        energies = jnp.array([1.0, 1.0])
        emax = jnp.array([5.0, 5.0])
        volumes = jnp.array([1.0, 1.0])
        pressures = jnp.array([1.0, 10.0])
        assert bool(self._kernel_accept(energies, emax, volumes, pressures)) == bool(
            perform_swap(energies, emax, volumes, pressures)
        )

    def test_boundary_matches_perform_swap(self):
        # Boundary: E_A == Emax_j should reject (strict <)
        energies = jnp.array([5.0, 2.0])
        emax = jnp.array([5.0, 5.0])
        assert bool(self._kernel_accept(energies, emax)) == bool(
            perform_swap(energies, emax)
        )

    def test_accept_returns_bool_scalar(self):
        """accept must return a JAX boolean scalar (shape ())."""
        kernel = PressureRENSSwap()
        proposed = {"energy_a": jnp.array(1.0), "energy_b": jnp.array(2.0)}
        result = kernel.accept(proposed, jnp.array(5.0), jnp.array(5.0), {}, {})
        assert result.shape == ()
        assert result.dtype == jnp.bool_

    def test_accept_jit_compatible(self):
        """PressureRENSSwap.accept must be JIT-compatible."""
        kernel = PressureRENSSwap()

        def _accept(e_a, e_b, emax_a, emax_b, cell_a, cell_b, p_a, p_b):
            proposed = {"energy_a": e_a, "energy_b": e_b, "cell_a": cell_a, "cell_b": cell_b}
            ens_a = {"pressure": p_a}
            ens_b = {"pressure": p_b}
            return kernel.accept(proposed, emax_a, emax_b, ens_a, ens_b)

        jitted = jax.jit(_accept)
        cell = jnp.eye(3)
        result = jitted(
            jnp.array(1.0), jnp.array(1.0),
            jnp.array(5.0), jnp.array(5.0),
            cell, cell,
            jnp.array(1.0), jnp.array(1.0),
        )
        assert result.shape == ()


# ---------------------------------------------------------------------------
# replica_exchange_step with explicit swap_kernel parameter
# ---------------------------------------------------------------------------


class TestReplicaExchangeStepWithKernel:
    """Explicit swap_kernel=PressureRENSSwap() must match the default."""

    def test_explicit_kernel_matches_default(self):
        """replica_exchange_step(..., swap_kernel=PressureRENSSwap()) == default."""
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(17)

        # Default (no swap_kernel argument)
        new_pos_default, _, new_ene_default, new_cells_default, info_default = (
            replica_exchange_step(key, pos, types, ene, cells, emax)
        )
        # Explicit kernel
        new_pos_explicit, _, new_ene_explicit, new_cells_explicit, info_explicit = (
            replica_exchange_step(
                key, pos, types, ene, cells, emax,
                swap_kernel=PressureRENSSwap(),
            )
        )

        assert jnp.allclose(new_pos_default, new_pos_explicit)
        assert jnp.allclose(new_ene_default, new_ene_explicit)
        assert jnp.allclose(new_cells_default, new_cells_explicit)
        assert int(info_default["n_accepted"]) == int(info_explicit["n_accepted"])
        assert int(info_default["n_attempted"]) == int(info_explicit["n_attempted"])

    def test_explicit_kernel_pressure_matches_default(self):
        """Same seed + pressures: explicit kernel matches default."""
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        pressures = jnp.array([0.1, 0.5, 1.0])
        key = jax.random.key(31)

        default_result = replica_exchange_step(
            key, pos, types, ene, cells, emax, pressures=pressures
        )
        explicit_result = replica_exchange_step(
            key, pos, types, ene, cells, emax, pressures=pressures,
            swap_kernel=PressureRENSSwap(),
        )

        assert jnp.allclose(default_result[0], explicit_result[0])  # positions
        assert jnp.allclose(default_result[2], explicit_result[2])  # energies
        assert int(default_result[4]["n_accepted"]) == int(
            explicit_result[4]["n_accepted"]
        )

    def test_explicit_kernel_jit(self):
        """replica_exchange_step with explicit swap_kernel is JIT-compatible."""
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(0)
        kernel = PressureRENSSwap()

        jitted = jax.jit(
            replica_exchange_step,
            static_argnames=("n_swap_cycles", "swap_kernel"),
        )
        new_pos, _, new_ene, _, info = jitted(
            key, pos, types, ene, cells, emax,
            n_swap_cycles=1, swap_kernel=kernel,
        )
        assert new_pos.shape == pos.shape
        assert int(info["n_attempted"]) > 0
