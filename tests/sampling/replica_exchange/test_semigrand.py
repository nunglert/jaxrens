"""Tests for semi-grand (chemical-potential) replica exchange.

Semi-grand RE is ``EnsembleSwap`` with ``chemical_potentials`` required
(see ``make_swap_kernel``). Accepted swaps move walkers between runs, and
each run keeps its own μ: a walker arriving in run r carries
``U + P_r·V − μ_r·N``. The tests cover propose(), make_swap_kernel,
replica_exchange_step with the semi-grand kernel, short NS runs on VmapRuns
and PmapVmapRuns, and the semi_grand checks of InterRESpec.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.inter_re import InterRESpec
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.moves.replica_exchange import (
    EnsembleSwap,
    SwapWalker,
    make_swap_kernel,
    replica_exchange_step,
)
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns_multi_gpu, run_ns_parallel
from jaxrens.sampling.termination import IterationTermination
from jaxrens.state.config import InterREConfig


def _state(positions, types, energy, cell=None):
    """One walker as passed to propose()."""
    return SwapWalker(positions, types, jnp.asarray(energy), cell)


def _semi_grand_kernel():
    """The semi-grand kernel as built by ``make_swap_kernel``."""
    return EnsembleSwap(required_params=("chemical_potentials",))


def _propose(kernel, walker_a, walker_b, ens_a, ens_b, key, backend):
    """Return ``kernel.propose`` as a dict with ``*_a`` / ``*_b`` keys.

    ``*_a`` is the walker entering run A (built from walker B), ``*_b`` the
    one entering run B.
    """
    into_a, into_b = kernel.propose(
        walker_a, walker_b, ens_a, ens_b, key, backend
    )
    proposed = {}
    for suffix, w in (("a", into_a), ("b", into_b)):
        proposed[f"positions_{suffix}"] = w.positions
        proposed[f"types_{suffix}"] = w.types
        proposed[f"energy_{suffix}"] = w.energy
        proposed[f"cell_{suffix}"] = w.cell
    return proposed


def _sg_step(
    rng_key,
    all_positions,
    all_types,
    all_energies,
    all_cells,
    all_emax,
    chemical_potentials,
    semi_grand_kernel,
    pressures=None,
    n_swap_cycles=1,
):
    """Run replica_exchange_step with semi-grand per-run params."""
    run_params = {"chemical_potentials": chemical_potentials}
    if pressures is not None:
        run_params["pressure"] = pressures
    new, info = replica_exchange_step(
        rng_key,
        SwapWalker(all_positions, all_types, all_energies, all_cells),
        all_emax,
        run_params,
        semi_grand_kernel,
        n_swap_cycles=n_swap_cycles,
    )
    return new.positions, new.types, new.energy, new.cell, info


def _ens(mu):
    """Ensemble params with the chemical potentials of one run."""
    return {"chemical_potentials": jnp.array(mu, dtype=jnp.float32)}


class TestSemiGrandProposeConvention:
    """Check that propose() exchanges the walkers and re-bases each energy
    to the receiving run's μ.

    energy_a = U_B − μ_A·N_B (walker B in run A) and
    energy_b = U_A − μ_B·N_A (walker A in run B).
    """

    def test_hand_calculated_energies_match(self):
        """The stored energy is ``U − μ_self·N``; propose() recovers U and
        subtracts the receiving run's ``μ·N``.

        Setup:
          run A: μ_A = [0.0, 0.0], walker has types=[0, 0, 1], U_A = 3.0
          run B: μ_B = [0.5, 1.0], walker has types=[1, 1, 0], U_B = 5.0

          N_A = [2, 1]  (2 atoms of species 0, 1 of species 1)
          N_B = [1, 2]  (1 atom of species 0, 2 of species 1)

          stored_A = U_A - μ_A · N_A = 3.0 - 0 = 3.0
          stored_B = U_B - μ_B · N_B = 5.0 - (0.5·1 + 1.0·2) = 5.0 - 2.5 = 2.5

          walker B in run A: U_B - μ_A · N_B = 5.0 - 0 = 5.0
          walker A in run B: U_A - μ_B · N_A = 3.0 - (0.5·2 + 1.0·1) = 1.0
        """
        kernel = _semi_grand_kernel()
        key = jax.random.key(0)

        mu_a = [0.0, 0.0]
        mu_b = [0.5, 1.0]
        types_a = jnp.array([0, 0, 1], dtype=jnp.int32)  # N_A = [2, 1]
        types_b = jnp.array([1, 1, 0], dtype=jnp.int32)  # N_B = [1, 2]
        # Stored energies in EnsembleBackend convention: Ω_self = U - μ_self·N
        stored_a = jnp.array(3.0 - (mu_a[0] * 2 + mu_a[1] * 1))  # = 3.0
        stored_b = jnp.array(5.0 - (mu_b[0] * 1 + mu_b[1] * 2))  # = 2.5

        sa = _state(jnp.zeros((3, 3)), types_a, stored_a)
        sb = _state(jnp.zeros((3, 3)), types_b, stored_b)
        ea = _ens(mu_a)
        eb = _ens(mu_b)

        proposed = _propose(kernel, sa, sb, ea, eb, key, None)

        assert float(proposed["energy_a"]) == pytest.approx(5.0, abs=1e-5)
        assert float(proposed["energy_b"]) == pytest.approx(1.0, abs=1e-5)

    def test_no_backend_calls(self):
        """propose() never calls the backend, and the kernel reports zero
        energy evaluations per swap."""

        def _backend(*args, **kwargs):
            raise AssertionError("semi-grand propose() called the backend")

        kernel = _semi_grand_kernel()
        types_a = jnp.zeros(4, dtype=jnp.int32)
        types_b = jnp.ones(4, dtype=jnp.int32)
        sa = _state(jnp.zeros((4, 3)), types_a, jnp.array(1.0))
        sb = _state(jnp.zeros((4, 3)), types_b, jnp.array(2.0))

        _propose(
            kernel,
            sa,
            sb,
            _ens([0.0, 0.0]),
            _ens([1.0, 2.0]),
            jax.random.key(1),
            _backend,
        )
        assert kernel.n_energy_evals == 0

    def test_positions_and_types_swapped(self):
        """Side A receives walker B's configuration and side B walker A's."""
        kernel = _semi_grand_kernel()
        key = jax.random.key(2)

        pos_a = jax.random.normal(key, (3, 3))
        pos_b = jax.random.normal(key, (3, 3)) + 1.0
        types_a = jnp.array([0, 0, 1], dtype=jnp.int32)
        types_b = jnp.array([1, 1, 0], dtype=jnp.int32)
        cell_a = jnp.eye(3)
        cell_b = 2.0 * jnp.eye(3)
        sa = _state(pos_a, types_a, jnp.array(1.0), cell_a)
        sb = _state(pos_b, types_b, jnp.array(2.0), cell_b)

        proposed = _propose(
            kernel, sa, sb, _ens([0.0, 0.0]), _ens([0.5, 1.0]), key, None
        )

        assert jnp.allclose(proposed["positions_a"], pos_b)
        assert jnp.allclose(proposed["positions_b"], pos_a)
        assert jnp.array_equal(proposed["types_a"], types_b)
        assert jnp.array_equal(proposed["types_b"], types_a)
        assert jnp.allclose(proposed["cell_a"], cell_b)
        assert jnp.allclose(proposed["cell_b"], cell_a)

    def test_hand_calculated_energies_both_mu_nonzero(self):
        """μ_A ≠ 0, so recovering U from the stored energy matters.

        Setup:
          μ_A = [1.0, 2.0], μ_B = [0.0, 0.0]
          types_A = [0, 0, 1] → N_A = [2, 1]
          U_A     = 3.0   (raw potential energy)
          state.energy_A = U_A - μ_A · N_A = 3.0 - (1·2 + 2·1) = -1.0

          walker A in run B: U_A - μ_B · N_A = 3.0 - 0 = 3.0
            kernel: state.energy_A + μ_A·N_A - μ_B·N_A = -1.0 + 4.0 - 0 = 3.0
          walker B in run A: U_B - μ_A · N_B = 5.0 - (1·1 + 2·2) = 0.0
        """
        kernel = _semi_grand_kernel()
        key = jax.random.key(0)

        mu_a = [1.0, 2.0]
        mu_b = [0.0, 0.0]
        types_a = jnp.array([0, 0, 1], dtype=jnp.int32)  # N_A = [2, 1]
        types_b = jnp.array([1, 1, 0], dtype=jnp.int32)  # N_B = [1, 2]
        # Stored energies in the EnsembleBackend convention.
        u_a = 3.0
        u_b = 5.0
        stored_a = u_a - (mu_a[0] * 2 + mu_a[1] * 1)  # = -1.0
        stored_b = u_b - (mu_b[0] * 1 + mu_b[1] * 2)  # = 5.0

        sa = _state(jnp.zeros((3, 3)), types_a, jnp.array(stored_a))
        sb = _state(jnp.zeros((3, 3)), types_b, jnp.array(stored_b))
        proposed = _propose(kernel, sa, sb, _ens(mu_a), _ens(mu_b), key, None)

        assert float(proposed["energy_a"]) == pytest.approx(0.0, abs=1e-5)
        assert float(proposed["energy_b"]) == pytest.approx(3.0, abs=1e-5)

    def test_symmetric_zero_mu(self):
        """With μ_A = μ_B = 0, each walker keeps Ω = U in the other run."""
        kernel = _semi_grand_kernel()
        key = jax.random.key(3)

        types_a = jnp.array([0, 1, 0], dtype=jnp.int32)
        types_b = jnp.array([1, 0, 1], dtype=jnp.int32)
        u_a = jnp.array(4.0)
        u_b = jnp.array(7.0)
        sa = _state(jnp.zeros((3, 3)), types_a, u_a)
        sb = _state(jnp.zeros((3, 3)), types_b, u_b)

        proposed = _propose(
            kernel, sa, sb, _ens([0.0, 0.0]), _ens([0.0, 0.0]), key, None
        )

        assert float(proposed["energy_a"]) == pytest.approx(float(u_b))
        assert float(proposed["energy_b"]) == pytest.approx(float(u_a))

    def test_pressure_rebased_with_cells(self):
        """μPT: the P·V term is also re-based to the receiving run.

        Setup: μ_A = μ_B = 0, P_A = 1.0, P_B = 3.0.
          walker A: U_A = 2.0, cell 1·I (V = 1),  stored 2.0 + 1·1 = 3.0
          walker B: U_B = 4.0, cell 2·I (V = 8),  stored 4.0 + 3·8 = 28.0
          walker B in run A: 4.0 + 1·8 = 12.0
          walker A in run B: 2.0 + 3·1 = 5.0
        """
        kernel = _semi_grand_kernel()
        types = jnp.zeros(3, dtype=jnp.int32)
        sa = _state(jnp.zeros((3, 3)), types, jnp.array(3.0), jnp.eye(3))
        sb = _state(
            jnp.zeros((3, 3)), types, jnp.array(28.0), 2.0 * jnp.eye(3)
        )
        ens_a = dict(_ens([0.0, 0.0]), pressure=jnp.array(1.0))
        ens_b = dict(_ens([0.0, 0.0]), pressure=jnp.array(3.0))

        proposed = _propose(
            kernel, sa, sb, ens_a, ens_b, jax.random.key(0), None
        )

        assert jnp.allclose(proposed["energy_a"], 12.0)
        assert jnp.allclose(proposed["energy_b"], 5.0)


class TestMakeSwapKernelSemiGrand:
    """Check the semi-grand kernel that make_swap_kernel builds."""

    def test_requires_chemical_potentials(self):
        cfg = InterREConfig(flavor="semi_grand", chemical_potentials=None)
        with pytest.raises(ValueError, match="chemical_potentials"):
            make_swap_kernel(cfg)

    def test_builds_ensemble_swap(self):
        cfg = InterREConfig(
            flavor="semi_grand", chemical_potentials=((0.0, 0.0), (0.5, 1.0))
        )
        kernel = make_swap_kernel(cfg)
        assert isinstance(kernel, EnsembleSwap)
        assert kernel.required_params == ("chemical_potentials",)


class TestSemiGrandStepFunction:
    """Check replica_exchange_step with the semi-grand kernel."""

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
        kernel = _semi_grand_kernel()
        key = jax.random.key(7)

        _, _, _, _, swap_info = _sg_step(
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
        kernel = _semi_grand_kernel()

        @jax.jit
        def _step(rng_key):
            return _sg_step(
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
        n_runs, n_walkers, n_atoms = 1, 5, 4
        key = jax.random.key(0)
        positions = (
            jax.random.normal(key, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
        energies = jax.random.normal(key, (n_runs, n_walkers))
        emax = jnp.max(energies, axis=1)
        chem_pots = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        kernel = _semi_grand_kernel()

        (
            new_pos,
            new_types,
            new_ene,
            _,
            swap_info,
        ) = _sg_step(
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
        """With 2 runs each cycle attempts the one even pair, so 2 cycles
        give 2 attempts."""
        positions, types, energies, emax, chem_pots = self._make_inputs(
            n_runs=2
        )
        kernel = _semi_grand_kernel()
        key = jax.random.key(5)

        _, _, _, _, swap_info = _sg_step(
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
        assert int(swap_info["n_attempted"]) == 2

    @staticmethod
    def _two_walkers(u_a, u_b, emax, cells=None, pressures=None):
        """One walker per run, 2 atoms, 2 species.

        run 0: μ = [0, 0],     walker A types [0, 0] -> N_A = [2, 0]
        run 1: μ = [0.5, 1.0], walker B types [1, 1] -> N_B = [0, 2]
        """
        chem_pots = jnp.array([[0.0, 0.0], [0.5, 1.0]], dtype=jnp.float32)
        types = jnp.array([[[0, 0]], [[1, 1]]], dtype=jnp.int32)
        positions = jnp.stack([jnp.zeros((1, 2, 3)), jnp.ones((1, 2, 3))])
        counts = jax.vmap(lambda t: jnp.bincount(t[0], length=2))(types)
        energies = (
            jnp.array([u_a, u_b]) - jnp.sum(chem_pots * counts, axis=1)
        )[:, None]
        out = _sg_step(
            rng_key=jax.random.key(0),
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=cells,
            all_emax=jnp.array(emax),
            chemical_potentials=chem_pots,
            semi_grand_kernel=_semi_grand_kernel(),
            pressures=pressures,
            n_swap_cycles=1,
        )
        return positions, types, energies, chem_pots, out

    def test_accepted_swap_moves_walkers(self):
        """An accepted swap exchanges positions and types between runs."""
        positions, types, _, _, out = self._two_walkers(
            0.0, 0.0, emax=[1e6, 1e6]
        )
        new_pos, new_types, _, _, swap_info = out
        assert int(swap_info["n_accepted"]) == 1
        assert jnp.allclose(new_pos[0], positions[1])
        assert jnp.allclose(new_pos[1], positions[0])
        assert jnp.array_equal(new_types[0], types[1])
        assert jnp.array_equal(new_types[1], types[0])

    def test_stored_energy_matches_run_mu_after_swap(self):
        """After a swap every stored energy is U - μ_run · N of the run the
        walker is in.

        U = 0: before the swap the stored energies are [0, -2]; walker B
        arrives in run 0 with -μ_0·N_B = 0 and walker A in run 1 with
        -μ_1·N_A = -1, so the result is [0, -1].
        """
        _, _, _, chem_pots, out = self._two_walkers(0.0, 0.0, emax=[1e6, 1e6])
        _, new_types, new_ene, _, swap_info = out
        assert int(swap_info["n_accepted"]) == 1
        counts = jax.vmap(lambda t: jnp.bincount(t[0], length=2))(new_types)
        expected = -jnp.sum(chem_pots * counts, axis=1)
        assert jnp.allclose(new_ene[:, 0], expected)
        assert jnp.allclose(new_ene[:, 0], jnp.array([0.0, -1.0]))

    def test_acceptance_uses_receiving_emax(self):
        """Each walker is checked against the Emax of the run it enters.

        U_A = 0, U_B = 0.5: walker A in run 1 has -1, walker B in run 0
        has 0.5. With Emax = [0.6, -0.9] both fit their new run (the old
        rule, A against Emax_0 and B against Emax_1, rejected this swap).
        With Emax = [0.4, -0.9] walker B does not fit run 0.
        """
        *_, out = self._two_walkers(0.0, 0.5, emax=[0.6, -0.9])
        assert int(out[4]["n_accepted"]) == 1
        *_, out = self._two_walkers(0.0, 0.5, emax=[0.4, -0.9])
        assert int(out[4]["n_accepted"]) == 0

    def test_swap_moves_cells_and_rebases_pressure(self):
        """μPT: cells move with the walkers and P·V uses the new run's P.

        U = 0, cells 1·I (V = 1) in run 0 and 2·I (V = 8) in run 1,
        P = [1, 3]. Walker B in run 0: 1·8 - μ_0·N_B = 8.
        Walker A in run 1: 3·1 - μ_1·N_A = 3 - 1 = 2.
        """
        cells = jnp.stack([jnp.eye(3)[None], 2.0 * jnp.eye(3)[None]])
        pressures = jnp.array([1.0, 3.0])
        # Stored U + P·V - μ·N: run 0 is 0 + 1 - 0, run 1 is 0 + 24 - 2.
        energies = jnp.array([[1.0], [22.0]])
        out = _sg_step(
            rng_key=jax.random.key(0),
            all_positions=jnp.zeros((2, 1, 2, 3)),
            all_types=jnp.array([[[0, 0]], [[1, 1]]], dtype=jnp.int32),
            all_energies=energies,
            all_cells=cells,
            all_emax=jnp.array([1e6, 1e6]),
            chemical_potentials=jnp.array([[0.0, 0.0], [0.5, 1.0]]),
            semi_grand_kernel=_semi_grand_kernel(),
            pressures=pressures,
            n_swap_cycles=1,
        )
        _, _, new_ene, new_cells, swap_info = out
        assert int(swap_info["n_accepted"]) == 1
        assert jnp.allclose(new_cells[0], cells[1])
        assert jnp.allclose(new_cells[1], cells[0])
        assert jnp.allclose(new_ene[:, 0], jnp.array([8.0, 2.0]))

    def test_identical_mu_zero_grand_potential_accepts(self):
        """With μ = 0 the stored energy is U, so U = 0.5 < Emax = 10 always
        accepts."""
        n_runs, n_walkers, n_atoms = 2, 4, 3
        key = jax.random.key(11)
        positions = jnp.zeros((n_runs, n_walkers, n_atoms, 3))
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)

        energies = jnp.ones((n_runs, n_walkers)) * 0.5
        emax = jnp.ones(n_runs) * 10.0
        chem_pots = jnp.zeros((n_runs, 2), dtype=jnp.float32)

        kernel = _semi_grand_kernel()
        _, _, new_ene, _, swap_info = _sg_step(
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
        assert int(swap_info["n_attempted"]) == 1
        assert int(swap_info["n_accepted"]) == 1


def _run_vmap_semi_grand(n_atoms=4, n_walkers=12, n_iters=10, seed=42):
    """Run a two-run NS with flavor="semi_grand" and two μ vectors."""
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

    # Only species 0 is present, so μ[1] has no effect.
    inter_re_cfg = InterREConfig(
        flavor="semi_grand",
        re_interval=1,
        n_swap_cycles=1,
        chemical_potentials=((0.0, 0.0), (0.5, 0.5)),
    )

    return run_ns_parallel(
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
    )


def test_vmap_semi_grand_run():
    result = _run_vmap_semi_grand()
    assert result["log_evidence"].shape == (2,)
    assert result["n_dead"].shape == (2,)
    assert jnp.all(jnp.isfinite(result["log_evidence"]))


def test_pmap_vmap_semi_grand_run():
    """run_ns_multi_gpu with flavor="semi_grand" on n_gpu=1."""
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

    assert result["log_evidence"].shape == (n_gpu, n_per_gpu)
    assert jnp.all(jnp.isfinite(result["log_evidence"]))


class TestInterRESpecSemiGrand:
    """Check the semi_grand checks of the InterRESpec schema."""

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
        with pytest.raises(
            ValidationError, match="requires 'chemical_potentials'"
        ):
            InterRESpec(flavor="semi_grand")

    def test_semi_grand_inconsistent_row_lengths_raises(self):
        with pytest.raises(ValidationError, match="same length"):
            InterRESpec(
                flavor="semi_grand",
                chemical_potentials=[[0.0, 0.0], [0.5, 1.0, 2.0]],
            )

    def test_pressure_flavor_has_no_chemical_potentials(self):
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
