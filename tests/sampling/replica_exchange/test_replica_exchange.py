"""Tests for energy and pressure replica exchange across parallel NS runs.

Covers the pair schedule (get_swap_pairs), the acceptance rule
(perform_swap and PressureRENSSwap.accept) and the full exchange step across
parallel runs (replica_exchange_step).
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


class TestGetSwapPairs:
    """Check the even (0,1),(2,3),... and odd (1,2),(3,4),... pair lists."""

    def test_even_phase_4_runs(self):
        pairs = get_swap_pairs(4, 0)
        expected = jnp.array([[0, 1], [2, 3]])
        assert jnp.array_equal(pairs, expected)

    def test_odd_phase_4_runs(self):
        pairs = get_swap_pairs(4, 1)
        # Runs are 0..3, so (3, 4) does not exist.
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


class TestPerformSwap:
    """Check the plain energy rule E_A < Emax_j and E_B < Emax_i."""

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


class TestPerformSwapPressure:
    """Check the enthalpy rule U_A + P_j * V_A < Emax_j (and its mirror).

    perform_swap takes raw energies U, so the cross enthalpy has a single
    PV term.
    """

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


class TestPressureRENSStoredEnthalpyConvention:
    """Check that accept() reads energies as stored enthalpies.

    In production the kernel gets ``state.energy``, which EnsembleBackend sets
    to ``H_self = U + P_self * V``. The kernel must therefore add only the
    cross term ``(P_other - P_self) * V``, not a second ``P_self * V``.
    """

    def test_equal_pressure_accepts_below_emax(self):
        """Equal pressures and equal U: the cross enthalpy equals H_self.

        V = 320 and P = 0.006242 (1 GPa in eV/A^3) give P * V = 2.0, so
        H_self = -83 + 2 = -81. A second P * V would give -79. Emax = -80
        sits between the two.
        """
        v_a = jnp.array(320.0)
        v_b = jnp.array(320.0)
        p_a = jnp.array(0.006242)
        p_b = jnp.array(0.006242)
        u = -83.0
        h_self = u + float(p_a) * float(v_a)
        emax_a = jnp.array(-80.0)
        emax_b = jnp.array(-80.0)

        kernel = PressureRENSSwap()
        proposed = {
            "energy_a": jnp.array(h_self),
            "energy_b": jnp.array(h_self),
            "cell_a": jnp.eye(3) * (float(v_a) ** (1.0 / 3.0)),
            "cell_b": jnp.eye(3) * (float(v_b) ** (1.0 / 3.0)),
        }
        ens_a = {"pressure": p_a}
        ens_b = {"pressure": p_b}

        accepted = kernel.accept(proposed, emax_a, emax_b, ens_a, ens_b)
        assert bool(accepted), (
            "Equal pressure + equal U on both sides should always accept "
            "(a double-counted P·V would reject)."
        )

    def test_pv_not_double_counted_at_tight_emax(self):
        """Tight Emax with adjacent pressures on a staggered P-ladder, same U
        on both sides.  The correct cross-enthalpy is ``h_A_in_B = U_A + P_B ·
        V_A`` (exactly one PV term).  Emax is chosen so this single-PV result
        sits just below the threshold; a spurious extra ``P_A · V_A`` would
        push it above and reject.
        """
        p_a = jnp.array(0.006242)  # 1 GPa in eV/Å³
        p_b = jnp.array(0.01248)  # 2 GPa
        v_a = jnp.array(320.0)
        v_b = jnp.array(320.0)
        u_a = -85.0
        u_b = -85.0
        h_self_a = u_a + float(p_a) * float(v_a)  # ≈ -83.00
        h_self_b = u_b + float(p_b) * float(v_b)  # ≈ -81.01

        # Correct cross-enthalpies (one PV term each):
        #   h_A_in_B = U_A + P_B · V_A = -85 + 0.01248·320 = -81.01
        #   h_B_in_A = U_B + P_A · V_B = -85 + 0.006242·320 = -83.00
        # A spurious extra self-PV would lift both to ≈-79.
        # Emax = -80 → accepts (both ≤ -81); a double-counted P·V rejects.
        emax_a = jnp.array(-80.0)
        emax_b = jnp.array(-80.0)

        kernel = PressureRENSSwap()
        proposed = {
            "energy_a": jnp.array(h_self_a),
            "energy_b": jnp.array(h_self_b),
            "cell_a": jnp.eye(3) * (float(v_a) ** (1.0 / 3.0)),
            "cell_b": jnp.eye(3) * (float(v_b) ** (1.0 / 3.0)),
        }
        ens_a = {"pressure": p_a}
        ens_b = {"pressure": p_b}

        accepted = kernel.accept(proposed, emax_a, emax_b, ens_a, ens_b)
        assert bool(accepted), (
            "Only ΔP·V contributes → both legs sit at ≈-81 eV, below "
            "Emax=-80; a double-counted P·V would reject."
        )


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
    """Check swap counts, outputs and shapes of the full exchange step."""

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
        """With very high Emax the swap is accepted and every array swaps."""
        n_runs, n_walkers = 2, 1
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        # Make types and cells differ between runs so the swap is visible.
        types = types.at[1].set(1)
        cells = cells.at[1].set(6.0 * jnp.eye(3))
        emax = jnp.array([100.0, 100.0])
        key = jax.random.key(7)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, cells, emax
        )
        # 2 runs: the even phase has one pair, the odd phase has none.
        assert int(info["n_attempted"]) == 1
        assert int(info["n_accepted"]) == 1
        assert jnp.allclose(new_ene[0, 0], ene[1, 0])
        assert jnp.allclose(new_ene[1, 0], ene[0, 0])
        assert jnp.array_equal(new_pos[0, 0], pos[1, 0])
        assert jnp.array_equal(new_pos[1, 0], pos[0, 0])
        assert jnp.array_equal(new_types[0, 0], types[1, 0])
        assert jnp.array_equal(new_types[1, 0], types[0, 0])
        assert jnp.array_equal(new_cells[0, 0], cells[1, 0])
        assert jnp.array_equal(new_cells[1, 0], cells[0, 0])

    def test_two_runs_guaranteed_reject(self):
        """With very low Emax, no swaps should be accepted."""
        n_runs, n_walkers = 2, 1
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        # _make_re_data draws energies in [-5, 0].
        emax = jnp.array([-10.0, -10.0])
        key = jax.random.key(7)
        new_pos, new_types, new_ene, new_cells, info = replica_exchange_step(
            key, pos, types, ene, cells, emax
        )
        assert int(info["n_attempted"]) == 1
        assert int(info["n_accepted"]) == 0
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
        # 3 runs: one even and one odd pair, so 2 attempts per cycle.
        assert int(info["n_attempted"]) == 6

    def test_energy_conservation(self):
        """Without pressure, swaps only permute the stored energies."""
        n_runs, n_walkers = 4, 3
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0, 100.0])
        key = jax.random.key(42)
        _, _, new_ene, _, _ = replica_exchange_step(
            key, pos, types, ene, cells, emax, n_swap_cycles=5
        )
        assert jnp.allclose(jnp.sort(ene.ravel()), jnp.sort(new_ene.ravel()))

    def test_types_per_run_not_per_walker(self):
        """Per-run types of shape (P, n_atoms) are returned unchanged."""
        n_runs, n_walkers, n_atoms = 3, 2, 2
        pos, _, ene, cells = _make_re_data(n_runs, n_walkers, n_atoms)
        types_shared = jnp.zeros((n_runs, n_atoms), dtype=jnp.int32)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(0)
        _, new_types, _, _, _ = replica_exchange_step(
            key, pos, types_shared, ene, cells, emax
        )
        assert jnp.array_equal(new_types, types_shared)


class TestReplicaExchangeStepPressure:
    """Check replica_exchange_step with a pressure per run."""

    def test_pressure_re_runs(self):
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

    def test_pressure_gap_rejects_swaps(self):
        """A large pressure gap pushes the cross enthalpy above Emax."""
        n_runs, n_walkers = 2, 1
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        # The walker moving from run 0 to run 1 gets U + 1.0 * 125, about
        # 122, well above Emax = 0.
        emax = jnp.array([0.0, 0.0])
        pressures = jnp.array([0.0, 1.0])
        key = jax.random.key(0)
        *_, info = replica_exchange_step(
            key, pos, types, ene, cells, emax, pressures=pressures
        )
        assert int(info["n_accepted"]) == 0


class TestJITCompatibility:
    """Check that the public functions run under jax.jit."""

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
        jitted = jax.jit(
            replica_exchange_step, static_argnames=("n_swap_cycles",)
        )
        new_pos, _, new_ene, _, info = jitted(
            key, pos, types, ene, cells, emax, n_swap_cycles=1
        )
        assert new_pos.shape == pos.shape
        assert int(info["n_attempted"]) > 0


def test_swap_kernel_cannot_be_instantiated():
    with pytest.raises(TypeError):
        SwapKernel()  # type: ignore[abstract]


class TestPressureRENSSwapAccept:
    """Check the return type of PressureRENSSwap.accept and its JIT use."""

    def test_accept_returns_bool_scalar(self):
        kernel = PressureRENSSwap()
        proposed = {"energy_a": jnp.array(1.0), "energy_b": jnp.array(2.0)}
        result = kernel.accept(
            proposed, jnp.array(5.0), jnp.array(5.0), {}, {}
        )
        assert result.shape == ()
        assert result.dtype == jnp.bool_

    def test_accept_jit_compatible(self):
        kernel = PressureRENSSwap()

        def _accept(e_a, e_b, emax_a, emax_b, cell_a, cell_b, p_a, p_b):
            proposed = {
                "energy_a": e_a,
                "energy_b": e_b,
                "cell_a": cell_a,
                "cell_b": cell_b,
            }
            ens_a = {"pressure": p_a}
            ens_b = {"pressure": p_b}
            return kernel.accept(proposed, emax_a, emax_b, ens_a, ens_b)

        jitted = jax.jit(_accept)
        cell = jnp.eye(3)
        result = jitted(
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(5.0),
            jnp.array(5.0),
            cell,
            cell,
            jnp.array(1.0),
            jnp.array(1.0),
        )
        assert result.shape == ()


class TestReplicaExchangeStepWithKernel:
    """Check that swap_kernel=PressureRENSSwap() matches the default."""

    def test_explicit_kernel_matches_default(self):
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        key = jax.random.key(17)

        (
            new_pos_default,
            _,
            new_ene_default,
            new_cells_default,
            info_default,
        ) = replica_exchange_step(key, pos, types, ene, cells, emax)
        (
            new_pos_explicit,
            _,
            new_ene_explicit,
            new_cells_explicit,
            info_explicit,
        ) = replica_exchange_step(
            key,
            pos,
            types,
            ene,
            cells,
            emax,
            swap_kernel=PressureRENSSwap(),
        )

        assert jnp.allclose(new_pos_default, new_pos_explicit)
        assert jnp.allclose(new_ene_default, new_ene_explicit)
        assert jnp.allclose(new_cells_default, new_cells_explicit)
        assert int(info_default["n_accepted"]) == int(
            info_explicit["n_accepted"]
        )
        assert int(info_default["n_attempted"]) == int(
            info_explicit["n_attempted"]
        )

    def test_explicit_kernel_pressure_matches_default(self):
        n_runs, n_walkers = 3, 2
        pos, types, ene, cells = _make_re_data(n_runs, n_walkers)
        emax = jnp.array([100.0, 100.0, 100.0])
        pressures = jnp.array([0.1, 0.5, 1.0])
        key = jax.random.key(31)

        default_result = replica_exchange_step(
            key, pos, types, ene, cells, emax, pressures=pressures
        )
        explicit_result = replica_exchange_step(
            key,
            pos,
            types,
            ene,
            cells,
            emax,
            pressures=pressures,
            swap_kernel=PressureRENSSwap(),
        )

        assert jnp.allclose(default_result[0], explicit_result[0])  # positions
        assert jnp.allclose(default_result[2], explicit_result[2])  # energies
        assert int(default_result[4]["n_accepted"]) == int(
            explicit_result[4]["n_accepted"]
        )

    def test_explicit_kernel_jit(self):
        """The kernel instance can be a static argument of jax.jit."""
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
            key,
            pos,
            types,
            ene,
            cells,
            emax,
            n_swap_cycles=1,
            swap_kernel=kernel,
        )
        assert new_pos.shape == pos.shape
        assert int(info["n_attempted"]) > 0
