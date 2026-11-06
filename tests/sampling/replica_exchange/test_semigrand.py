"""Tests for SemiGrandSwap, the chemical-potential replica exchange.

Covers the kernel (propose and accept), the step function
semi_grand_replica_exchange_step, short NS runs with flavor="semi_grand"
on VmapRuns and PmapVmapRuns, and the semi_grand checks of InterRESpec.

The stored energy is the grand-canonical energy at the run's own chemical
potentials, Omega = U - mu . N, where N counts the atoms of each species.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.inter_re import InterRESpec
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.moves.replica_exchange import (
    SemiGrandSwap,
    semi_grand_replica_exchange_step,
)
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns_multi_gpu, run_ns_parallel
from jaxrens.sampling.termination import IterationTermination
from jaxrens.state.config import InterREConfig


def _state(positions, types, energy, cell=None):
    """Swap-state dict for one walker, as the step function builds it."""
    return {
        "positions": positions,
        "types": types,
        "energy": energy,
        "cell": cell if cell is not None else jnp.zeros((3, 3)),
    }


def _ens(mu):
    """Ensemble params dict with the chemical potentials of one run."""
    return {"chemical_potentials": jnp.array(mu, dtype=jnp.float32)}


class TestSemiGrandProposeConvention:
    """Check that propose() rebases energies to the partner's mu.

    The new energy of walker A is Omega_A_new = U_A - mu_B . N_A, with U_A
    recovered from the stored Omega_A = U_A - mu_A . N_A.
    """

    def test_hand_calculated_energies_match(self):
        """Hand-computed energies with mu_A = 0.

        Setup:
          run A: μ_A = [0.0, 0.0], walker has types=[0, 0, 1], U_A = 3.0
          run B: μ_B = [0.5, 1.0], walker has types=[1, 1, 0], U_B = 5.0

          N_A = [2, 1]  (2 atoms of species 0, 1 of species 1)
          N_B = [1, 2]  (1 atom of species 0, 2 of species 1)

          stored_A = U_A - μ_A · N_A = 3.0 - 0 = 3.0
          stored_B = U_B - μ_B · N_B = 5.0 - (0.5·1 + 1.0·2) = 5.0 - 2.5 = 2.5

          Ω_A_new = U_A - μ_B · N_A = 3.0 - (0.5·2 + 1.0·1) = 3.0 - 2.0 = 1.0
          Ω_B_new = U_B - μ_A · N_B = 5.0 - (0.0·1 + 0.0·2) = 5.0
        """
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(0)

        mu_a = [0.0, 0.0]
        mu_b = [0.5, 1.0]
        types_a = jnp.array([0, 0, 1], dtype=jnp.int32)  # N_A = [2, 1]
        types_b = jnp.array([1, 1, 0], dtype=jnp.int32)  # N_B = [1, 2]
        stored_a = jnp.array(3.0 - (mu_a[0] * 2 + mu_a[1] * 1))  # = 3.0
        stored_b = jnp.array(5.0 - (mu_b[0] * 1 + mu_b[1] * 2))  # = 2.5

        sa = _state(jnp.zeros((3, 3)), types_a, stored_a)
        sb = _state(jnp.zeros((3, 3)), types_b, stored_b)
        ea = _ens(mu_a)
        eb = _ens(mu_b)

        proposed, n_e, n_g = kernel.propose(sa, sb, ea, eb, key, None)

        expected_omega_a = 3.0 - (0.5 * 2 + 1.0 * 1)  # = 1.0
        expected_omega_b = 5.0 - (0.0 * 1 + 0.0 * 2)  # = 5.0

        assert abs(float(proposed["energy_a"]) - expected_omega_a) < 1e-5
        assert abs(float(proposed["energy_b"]) - expected_omega_b) < 1e-5

    def test_no_backend_calls(self):
        """propose() makes no energy or gradient evaluations."""
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(1)

        types_a = jnp.zeros(4, dtype=jnp.int32)
        types_b = jnp.ones(4, dtype=jnp.int32)
        sa = _state(jnp.zeros((4, 3)), types_a, jnp.array(1.0))
        sb = _state(jnp.zeros((4, 3)), types_b, jnp.array(2.0))

        _, n_e, n_g = kernel.propose(
            sa, sb, _ens([0.0, 0.0]), _ens([1.0, 2.0]), key, None
        )
        assert n_e == 0
        assert n_g == 0

    def test_positions_and_types_unchanged(self):
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(2)

        pos_a = jax.random.normal(key, (3, 3))
        pos_b = jax.random.normal(key, (3, 3)) + 1.0
        types_a = jnp.array([0, 0, 1], dtype=jnp.int32)
        types_b = jnp.array([1, 1, 0], dtype=jnp.int32)
        sa = _state(pos_a, types_a, jnp.array(1.0))
        sb = _state(pos_b, types_b, jnp.array(2.0))

        proposed, _, _ = kernel.propose(
            sa, sb, _ens([0.0, 0.0]), _ens([0.5, 1.0]), key, None
        )

        assert jnp.allclose(proposed["positions_a"], pos_a)
        assert jnp.allclose(proposed["positions_b"], pos_b)
        assert jnp.array_equal(proposed["types_a"], types_a)
        assert jnp.array_equal(proposed["types_b"], types_b)

    def test_hand_calculated_energies_both_mu_nonzero(self):
        """Hand-computed energies with mu_A != 0, so U_A must be recovered.

        Setup:
          μ_A = [1.0, 2.0], μ_B = [0.0, 0.0]
          types_A = [0, 0, 1] → N_A = [2, 1]
          U_A     = 3.0   (raw potential energy)
          state.energy_A = U_A - μ_A · N_A = 3.0 - (1·2 + 2·1) = -1.0

          Expected Ω_A_new = U_A - μ_B · N_A = 3.0 - 0 = 3.0
            kernel: state.energy_A + μ_A·N_A - μ_B·N_A = -1.0 + 4.0 - 0 = 3.0
        """
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(0)

        mu_a = [1.0, 2.0]
        mu_b = [0.0, 0.0]
        types_a = jnp.array([0, 0, 1], dtype=jnp.int32)  # N_A = [2, 1]
        types_b = jnp.array([1, 1, 0], dtype=jnp.int32)  # N_B = [1, 2]
        u_a = 3.0
        u_b = 5.0
        stored_a = u_a - (mu_a[0] * 2 + mu_a[1] * 1)  # = -1.0
        stored_b = u_b - (mu_b[0] * 1 + mu_b[1] * 2)  # = 5.0

        sa = _state(jnp.zeros((3, 3)), types_a, jnp.array(stored_a))
        sb = _state(jnp.zeros((3, 3)), types_b, jnp.array(stored_b))
        proposed, _, _ = kernel.propose(
            sa, sb, _ens(mu_a), _ens(mu_b), key, None
        )

        # Ω_A_new = U_A - μ_B · N_A = 3.0 - 0 = 3.0
        # Ω_B_new = U_B - μ_A · N_B = 5.0 - (1·1 + 2·2) = 0.0
        assert abs(float(proposed["energy_a"]) - 3.0) < 1e-5
        assert abs(float(proposed["energy_b"]) - 0.0) < 1e-5

    def test_symmetric_zero_mu(self):
        """With μ_A = μ_B = 0, propose returns Ω = U (unchanged energies)."""
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(3)

        types_a = jnp.array([0, 1, 0], dtype=jnp.int32)
        types_b = jnp.array([1, 0, 1], dtype=jnp.int32)
        u_a = jnp.array(4.0)
        u_b = jnp.array(7.0)
        sa = _state(jnp.zeros((3, 3)), types_a, u_a)
        sb = _state(jnp.zeros((3, 3)), types_b, u_b)

        proposed, _, _ = kernel.propose(
            sa, sb, _ens([0.0, 0.0]), _ens([0.0, 0.0]), key, None
        )

        assert abs(float(proposed["energy_a"]) - float(u_a)) < 1e-5
        assert abs(float(proposed["energy_b"]) - float(u_b)) < 1e-5


class TestSemiGrandAccept:
    """Check that accept() is Omega_A < Emax_A and Omega_B < Emax_B."""

    def _make_proposed(self, omega_a, omega_b):
        return {
            "energy_a": jnp.array(float(omega_a)),
            "energy_b": jnp.array(float(omega_b)),
            "positions_a": jnp.zeros((3, 3)),
            "positions_b": jnp.zeros((3, 3)),
            "cell_a": jnp.zeros((3, 3)),
            "cell_b": jnp.zeros((3, 3)),
            "types_a": jnp.zeros(3, dtype=jnp.int32),
            "types_b": jnp.zeros(3, dtype=jnp.int32),
        }

    def test_accept_both_below_emax(self):
        kernel = SemiGrandSwap(n_species=2)
        proposed = self._make_proposed(0.5, 0.3)
        result = kernel.accept(
            proposed, jnp.array(1.0), jnp.array(1.0), {}, {}
        )
        assert bool(result)

    def test_reject_a_above_emax(self):
        kernel = SemiGrandSwap(n_species=2)
        proposed = self._make_proposed(2.0, 0.3)  # Ω_A > Emax_A
        result = kernel.accept(
            proposed, jnp.array(1.0), jnp.array(1.0), {}, {}
        )
        assert not bool(result)

    def test_reject_b_above_emax(self):
        kernel = SemiGrandSwap(n_species=2)
        proposed = self._make_proposed(0.3, 2.0)  # Ω_B > Emax_B
        result = kernel.accept(
            proposed, jnp.array(1.0), jnp.array(1.0), {}, {}
        )
        assert not bool(result)

    def test_accept_returns_bool(self):
        kernel = SemiGrandSwap(n_species=2)
        proposed = self._make_proposed(0.1, 0.2)
        result = kernel.accept(
            proposed, jnp.array(1.0), jnp.array(1.0), {}, {}
        )
        assert result.dtype == jnp.bool_

    def test_accept_jit_compatible(self):
        kernel = SemiGrandSwap(n_species=2)
        proposed = self._make_proposed(0.1, 0.2)

        @jax.jit
        def _jit_accept(p, ea, eb):
            return kernel.accept(p, ea, eb, {}, {})

        result = _jit_accept(proposed, jnp.array(1.0), jnp.array(1.0))
        assert result.shape == ()


class TestSemiGrandNSpeciesMismatch:
    """Check the n_species and chemical_potentials input errors."""

    def test_invalid_n_species_zero_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            SemiGrandSwap(n_species=0)

    def test_invalid_n_species_negative_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            SemiGrandSwap(n_species=-1)

    def test_propose_width_mismatch_raises(self):
        kernel = SemiGrandSwap(n_species=3)
        key = jax.random.key(0)
        types = jnp.zeros(4, dtype=jnp.int32)
        sa = _state(jnp.zeros((4, 3)), types, jnp.array(1.0))
        sb = _state(jnp.zeros((4, 3)), types, jnp.array(2.0))
        with pytest.raises(ValueError, match="n_species"):
            kernel.propose(
                sa, sb, _ens([0.0, 1.0]), _ens([1.0, 2.0]), key, None
            )

    def test_propose_missing_chemical_potentials_raises(self):
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(0)
        types = jnp.zeros(4, dtype=jnp.int32)
        sa = _state(jnp.zeros((4, 3)), types, jnp.array(1.0))
        sb = _state(jnp.zeros((4, 3)), types, jnp.array(2.0))
        with pytest.raises(ValueError, match="chemical_potentials"):
            kernel.propose(sa, sb, {}, _ens([0.0, 1.0]), key, None)


class TestSemiGrandStepFunction:
    """Check semi_grand_replica_exchange_step on random populations."""

    @staticmethod
    def _make_inputs(n_runs=2, n_walkers=5, n_atoms=4, n_species=2, seed=0):
        key = jax.random.key(seed)
        k1, k2 = jax.random.split(key)
        positions = (
            jax.random.normal(k1, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
        # Run 1 has half of its atoms in species 1.
        types = types.at[1, :, n_atoms // 2 :].set(1)
        energies = jax.random.normal(k2, (n_runs, n_walkers)) * 2.0
        emax = jnp.max(energies, axis=1)  # (n_runs,)
        chem_pots = jnp.array([[0.0, 0.0], [0.5, 1.0]], dtype=jnp.float32)
        return positions, types, energies, emax, chem_pots

    def test_n_energy_evals_zero(self):
        positions, types, energies, emax, chem_pots = self._make_inputs()
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(7)

        _, _, _, _, swap_info = semi_grand_replica_exchange_step(
            rng_key=key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            chemical_potentials=chem_pots,
            semi_grand_kernel=kernel,
            n_swap_cycles=1,
        )
        assert int(swap_info["n_energy_evals"]) == 0

    def test_jit_compatible(self):
        positions, types, energies, emax, chem_pots = self._make_inputs()
        kernel = SemiGrandSwap(n_species=2)

        @jax.jit
        def _step(rng_key):
            return semi_grand_replica_exchange_step(
                rng_key=rng_key,
                all_positions=positions,
                all_types=types,
                all_energies=energies,
                all_cells=None,
                all_emax=emax,
                chemical_potentials=chem_pots,
                semi_grand_kernel=kernel,
                n_swap_cycles=1,
            )

        key = jax.random.key(42)
        new_pos, new_types, new_ene, new_cells, swap_info = _step(key)

        assert new_pos.shape == positions.shape
        assert new_types.shape == types.shape
        assert new_ene.shape == energies.shape
        assert "n_accepted" in swap_info
        assert "n_attempted" in swap_info
        assert "n_energy_evals" in swap_info

    def test_single_run_no_swaps(self):
        """With one run there are no pairs, so nothing changes."""
        n_runs, n_walkers, n_atoms, n_species = 1, 5, 4, 2
        key = jax.random.key(0)
        positions = (
            jax.random.normal(key, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
        energies = jax.random.normal(key, (n_runs, n_walkers))
        emax = jnp.max(energies, axis=1)
        chem_pots = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        kernel = SemiGrandSwap(n_species=n_species)

        (
            new_pos,
            new_types,
            new_ene,
            _,
            swap_info,
        ) = semi_grand_replica_exchange_step(
            rng_key=key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            chemical_potentials=chem_pots,
            semi_grand_kernel=kernel,
            n_swap_cycles=1,
        )

        assert jnp.allclose(new_pos, positions)
        assert jnp.array_equal(new_types, types)
        assert jnp.allclose(new_ene, energies)
        assert int(swap_info["n_attempted"]) == 0
        assert int(swap_info["n_accepted"]) == 0
        assert int(swap_info["n_energy_evals"]) == 0

    def test_attempted_swaps_positive(self):
        """With 2 runs every cycle attempts a swap."""
        positions, types, energies, emax, chem_pots = self._make_inputs(
            n_runs=2
        )
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(5)

        _, _, _, _, swap_info = semi_grand_replica_exchange_step(
            rng_key=key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            chemical_potentials=chem_pots,
            semi_grand_kernel=kernel,
            n_swap_cycles=2,
        )
        assert int(swap_info["n_attempted"]) > 0

    def test_positions_unchanged_after_swap(self):
        """Semi-grand RE never changes positions or types."""
        positions, types, energies, emax, chem_pots = self._make_inputs()
        kernel = SemiGrandSwap(n_species=2)
        key = jax.random.key(9)

        new_pos, new_types, _, _, _ = semi_grand_replica_exchange_step(
            rng_key=key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            chemical_potentials=chem_pots,
            semi_grand_kernel=kernel,
            n_swap_cycles=1,
        )
        assert jnp.allclose(new_pos, positions)
        assert jnp.array_equal(new_types, types)

    def test_identical_mu_zero_grand_potential_accepts(self):
        """With mu = 0, Omega = U = 0.5 < Emax = 10, so the swap accepts."""
        n_runs, n_walkers, n_atoms = 2, 4, 3
        key = jax.random.key(11)
        positions = jnp.zeros((n_runs, n_walkers, n_atoms, 3))
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)

        energies = jnp.ones((n_runs, n_walkers)) * 0.5
        emax = jnp.ones(n_runs) * 10.0
        chem_pots = jnp.zeros((n_runs, 2), dtype=jnp.float32)

        kernel = SemiGrandSwap(n_species=2)
        *_, swap_info = semi_grand_replica_exchange_step(
            rng_key=key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            chemical_potentials=chem_pots,
            semi_grand_kernel=kernel,
            n_swap_cycles=1,
        )
        # 2 runs and 1 cycle: a single pair in the even phase.
        assert int(swap_info["n_attempted"]) == 1
        assert int(swap_info["n_accepted"]) == 1


def _run_vmap_semi_grand(n_atoms=4, n_walkers=12, n_iters=10, seed=42):
    """Run a short two-run NS with flavor="semi_grand" on VmapRuns.

    Returns the result dict and the inter_re_stats seen by a callback.
    """
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw",
            random_walk.build_kernel,
            step_size=0.3,
            step_size_max=2.0,
        )
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    n_runs = 2
    key = jax.random.key(seed)
    keys = jax.random.split(key, n_runs + 1)
    rng_keys = keys[:n_runs]
    pos_key = keys[-1]

    positions = jax.random.uniform(
        pos_key, (n_runs, n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos_run: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos_run)
    )(positions)

    # All atoms are species 0, so mu[1] has no effect.
    inter_re_cfg = InterREConfig(
        flavor="semi_grand",
        re_interval=1,
        n_swap_cycles=1,
        chemical_potentials=((0.0, 0.0), (0.5, 0.5)),
    )

    re_stats_log = []

    class _StatCb:
        def on_iteration(self, iteration, ns_state, info):
            s = info.get("inter_re_stats")
            if s is not None:
                re_stats_log.append(dict(s))

    result = run_ns_parallel(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        n_walkers=n_walkers,
        max_iterations=n_iters,
        n_mcmc_steps=3,
        termination_criteria=[IterationTermination(n_iters)],
        inter_re_config=inter_re_cfg,
        backend=backend,
        callbacks=[_StatCb()],
    )
    return result, re_stats_log


def test_vmap_semi_grand_run():
    """A two-run NS finishes and passes RE stats to the callbacks."""
    result, re_stats_log = _run_vmap_semi_grand()
    assert result["log_evidence"].shape == (2,)
    assert result["n_dead"].shape == (2,)
    assert jnp.all(jnp.isfinite(result["log_evidence"]))
    assert len(re_stats_log) > 0
    assert all(s["n_energy_evals"] == 0 for s in re_stats_log)


def test_pmap_vmap_semi_grand_run():
    """run_ns_multi_gpu with flavor="semi_grand" runs with one GPU."""
    n_gpu, n_per_gpu, n_walkers, n_atoms = 1, 2, 8, 4
    n_total = n_gpu * n_per_gpu

    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw",
            random_walk.build_kernel,
            step_size=0.3,
            step_size_max=2.0,
        )
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    key = jax.random.key(99)
    keys = jax.random.split(key, n_total + 1)
    rng_keys = keys[:n_total]
    pos_key = keys[-1]

    positions = jax.random.uniform(
        pos_key, (n_total, n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos_run: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos_run)
    )(positions)

    inter_re_cfg = InterREConfig(
        flavor="semi_grand",
        re_interval=1,
        n_swap_cycles=1,
        chemical_potentials=((0.0, 0.0), (0.5, 0.5)),
    )

    result = run_ns_multi_gpu(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
        n_walkers=n_walkers,
        max_iterations=8,
        n_mcmc_steps=3,
        termination_criteria=[IterationTermination(8)],
        inter_re_config=inter_re_cfg,
        backend=backend,
    )

    assert result is not None
    assert result["log_evidence"].shape == (n_gpu, n_per_gpu)
    assert jnp.all(jnp.isfinite(result["log_evidence"]))


class TestInterRESpecSemiGrand:
    """Check the semi_grand validation of the inter_re config section."""

    def test_semi_grand_valid_spec(self):
        spec = InterRESpec(
            flavor="semi_grand",
            re_interval=1,
            n_swap_cycles=1,
            chemical_potentials=[[0.0, 0.0], [0.5, 1.0]],
        )
        cfg = spec.to_inter_re_config()
        assert cfg.flavor == "semi_grand"
        assert cfg.chemical_potentials == ((0.0, 0.0), (0.5, 1.0))

    def test_semi_grand_missing_chemical_potentials_raises(self):
        with pytest.raises(ValueError, match="requires 'chemical_potentials'"):
            InterRESpec(flavor="semi_grand")

    def test_semi_grand_inconsistent_row_lengths_raises(self):
        with pytest.raises(ValueError, match="same"):
            InterRESpec(
                flavor="semi_grand",
                chemical_potentials=[
                    [0.0, 0.0],
                    [0.5, 1.0, 2.0],
                ],
            )

    def test_pressure_flavor_still_valid(self):
        spec = InterRESpec(flavor="pressure", re_interval=2, n_swap_cycles=1)
        cfg = spec.to_inter_re_config()
        assert cfg.flavor == "pressure"
        assert cfg.chemical_potentials is None

    def test_to_inter_re_config_roundtrip(self):
        spec = InterRESpec(
            flavor="semi_grand",
            re_interval=3,
            n_swap_cycles=2,
            chemical_potentials=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        cfg = spec.to_inter_re_config()
        assert cfg.re_interval == 3
        assert cfg.n_swap_cycles == 2
        assert cfg.chemical_potentials == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
