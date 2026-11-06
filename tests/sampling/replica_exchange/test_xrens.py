"""Tests for XRENSSwap, the composition-morphing replica exchange.

A swap moves a walker to the partner run, relabels its atom types to the
partner's target composition and evaluates the energy again. The tests cover
the kernel (propose and accept), the step function xrens_replica_exchange_step,
a short two-run NS with flavor="xrens", and the xrens checks of InterRESpec.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.cli.schema.inter_re import InterRESpec
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.moves.replica_exchange import (
    XRENSSwap,
    xrens_replica_exchange_step,
)
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns_parallel
from jaxrens.sampling.termination import IterationTermination
from jaxrens.state.config import InterREConfig


class SpeciesHarmonicBackend:
    """Harmonic backend with a weight per species.

    E = sum_i species_weight[types[i]] * 0.5 * ||pos_i||^2, so the energy
    changes when atom types are relabeled.
    """

    def __init__(self, species_weights: list[float]):
        self.species_weights = jnp.array(species_weights)
        self.r_cutoff = 0.0

    def __call__(
        self,
        positions: jnp.ndarray,
        types: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        weights = self.species_weights[types]
        energy = jnp.sum(weights * 0.5 * jnp.sum(positions**2, axis=-1))
        return BackendResult(energy=energy)


def _make_state(positions, types, energy, cell=None):
    """Swap-state dict for one walker, as the step function builds it."""
    return {
        "positions": positions,
        "types": types,
        "energy": energy,
        "cell": cell if cell is not None else jnp.zeros((3, 3)),
    }


class TestXRENSProposeMorphedTypes:
    """Check what XRENSSwap.propose() returns for each side of the swap."""

    def _run_propose(self, n_atoms, n_species, target_a, target_b, seed=0):
        backend = SpeciesHarmonicBackend(
            species_weights=[1.0, 2.0][:n_species]
        )
        kernel = XRENSSwap(n_species=n_species)
        key = jax.random.key(seed)
        k1, k2, morph_key = jax.random.split(key, 3)

        # With 2 species, A is all species 0 and B all species 1.
        if n_species == 2:
            types_a = jnp.zeros(n_atoms, dtype=jnp.int32)
            types_b = jnp.ones(n_atoms, dtype=jnp.int32)
        else:
            types_a = jax.random.randint(k1, (n_atoms,), 0, n_species)
            types_b = jax.random.randint(k2, (n_atoms,), 0, n_species)

        pos_a = jax.random.normal(k1, (n_atoms, 3)) * 0.1
        pos_b = jax.random.normal(k2, (n_atoms, 3)) * 0.1

        e_a = backend(pos_a, types_a, jnp.zeros((3, 3))).energy
        e_b = backend(pos_b, types_b, jnp.zeros((3, 3))).energy

        state_a = _make_state(pos_a, types_a, e_a)
        state_b = _make_state(pos_b, types_b, e_b)
        ens_a = {"target_composition": jnp.array(target_a, dtype=jnp.int32)}
        ens_b = {"target_composition": jnp.array(target_b, dtype=jnp.int32)}

        proposed, n_e, n_g = kernel.propose(
            state_a, state_b, ens_a, ens_b, morph_key, backend
        )
        return proposed, n_e, n_g

    def test_energies_match_morphed_configs(self):
        """Each proposed energy is the backend energy of the moved walker."""
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=2)
        k1, k2, mkey = jax.random.split(jax.random.key(3), 3)

        pos_a = jax.random.normal(k1, (8, 3))
        pos_b = jax.random.normal(k2, (8, 3))
        types_a = jnp.zeros(8, dtype=jnp.int32)
        types_b = jnp.ones(8, dtype=jnp.int32)
        state_a = _make_state(pos_a, types_a, jnp.array(0.0))
        state_b = _make_state(pos_b, types_b, jnp.array(0.0))
        ens_a = {"target_composition": jnp.array([8, 0])}
        ens_b = {"target_composition": jnp.array([4, 4])}

        proposed, n_e, n_g = kernel.propose(
            state_a, state_b, ens_a, ens_b, mkey, backend
        )
        cell = jnp.zeros((3, 3))
        expected_a = backend(pos_b, proposed["types_a"], cell).energy
        expected_b = backend(pos_a, proposed["types_b"], cell).energy
        assert n_e == 2
        assert n_g == 0
        assert jnp.allclose(proposed["energy_a"], expected_a)
        assert jnp.allclose(proposed["energy_b"], expected_b)

    def test_receiving_run_params_reach_backend(self):
        """Each energy uses the receiving run's pressure.

        The backend adds ensemble_params["pressure"] to the energy, so a
        swap of the two parameter sets would show up as a 1.0 offset.
        target_composition is stripped before the backend call.
        """
        harmonic = SpeciesHarmonicBackend([1.0, 2.0])

        def backend(
            positions, types, cell, max_neighbors=0, ensemble_params=None
        ):
            assert "target_composition" not in ensemble_params
            energy = harmonic(positions, types, cell).energy
            return BackendResult(energy=energy + ensemble_params["pressure"])

        kernel = XRENSSwap(n_species=2)
        k1, k2, mkey = jax.random.split(jax.random.key(5), 3)
        pos_a = jax.random.normal(k1, (4, 3))
        pos_b = jax.random.normal(k2, (4, 3))
        state_a = _make_state(
            pos_a, jnp.zeros(4, dtype=jnp.int32), jnp.array(0.0)
        )
        state_b = _make_state(
            pos_b, jnp.ones(4, dtype=jnp.int32), jnp.array(0.0)
        )
        ens_a = {
            "target_composition": jnp.array([4, 0]),
            "pressure": jnp.array(1.0),
        }
        ens_b = {
            "target_composition": jnp.array([2, 2]),
            "pressure": jnp.array(2.0),
        }

        proposed, _, _ = kernel.propose(
            state_a, state_b, ens_a, ens_b, mkey, backend
        )
        cell = jnp.zeros((3, 3))
        base_a = harmonic(pos_b, proposed["types_a"], cell).energy
        base_b = harmonic(pos_a, proposed["types_b"], cell).energy
        assert jnp.allclose(proposed["energy_a"], base_a + 1.0)
        assert jnp.allclose(proposed["energy_b"], base_b + 2.0)

    def test_positions_a_are_from_b(self):
        """Side A receives B's positions and side B receives A's."""
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=2)
        key = jax.random.key(7)
        k1, k2, mkey = jax.random.split(key, 3)

        pos_a = jax.random.normal(k1, (4, 3))
        pos_b = jax.random.normal(k2, (4, 3))
        types_a = jnp.zeros(4, dtype=jnp.int32)
        types_b = jnp.ones(4, dtype=jnp.int32)
        e_a = backend(pos_a, types_a, jnp.zeros((3, 3))).energy
        e_b = backend(pos_b, types_b, jnp.zeros((3, 3))).energy

        state_a = _make_state(pos_a, types_a, e_a)
        state_b = _make_state(pos_b, types_b, e_b)
        ens_a = {"target_composition": jnp.array([4, 0])}
        ens_b = {"target_composition": jnp.array([2, 2])}

        proposed, _, _ = kernel.propose(
            state_a, state_b, ens_a, ens_b, mkey, backend
        )
        assert jnp.allclose(proposed["positions_a"], pos_b)
        assert jnp.allclose(proposed["positions_b"], pos_a)

    def test_propose_many_seeds(self):
        """Each side gets its own target composition, for several seeds."""
        n_atoms = 12
        target_a = [12, 0]
        target_b = [6, 6]
        for seed in range(5):
            proposed, _, _ = self._run_propose(
                n_atoms, 2, target_a, target_b, seed=seed
            )
            got_a = jnp.bincount(proposed["types_a"], length=2)
            got_b = jnp.bincount(proposed["types_b"], length=2)
            assert jnp.array_equal(got_a, jnp.array(target_a)), f"seed={seed}"
            assert jnp.array_equal(got_b, jnp.array(target_b)), f"seed={seed}"


class TestXRENSAccept:
    """Check that accept() is E_A_new < Emax_A and E_B_new < Emax_B."""

    def _make_proposed(self, e_a, e_b, cell_a=None, cell_b=None):
        cell = jnp.zeros((3, 3))
        return {
            "energy_a": jnp.array(e_a),
            "energy_b": jnp.array(e_b),
            "cell_a": cell_a if cell_a is not None else cell,
            "cell_b": cell_b if cell_b is not None else cell,
            "types_a": jnp.zeros(4, dtype=jnp.int32),
            "types_b": jnp.zeros(4, dtype=jnp.int32),
            "positions_a": jnp.zeros((4, 3)),
            "positions_b": jnp.zeros((4, 3)),
        }

    def test_accept_returns_bool(self):
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.5, 0.3)
        result = kernel.accept(
            proposed,
            emax_a=jnp.array(1.0),
            emax_b=jnp.array(1.0),
            ensemble_params_a={},
            ensemble_params_b={},
        )
        assert result.dtype == jnp.bool_

    def test_accept_low_energy_accepts(self):
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.1, 0.2)
        result = kernel.accept(
            proposed,
            emax_a=jnp.array(5.0),
            emax_b=jnp.array(5.0),
            ensemble_params_a={},
            ensemble_params_b={},
        )
        assert bool(result)

    def test_accept_high_energy_rejects(self):
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(10.0, 0.2)
        result = kernel.accept(
            proposed,
            emax_a=jnp.array(1.0),
            emax_b=jnp.array(1.0),
            ensemble_params_a={},
            ensemble_params_b={},
        )
        assert not bool(result)

    def test_accept_jit_compatible(self):
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.5, 0.3)

        @jax.jit
        def jitted_accept(prop, ea, eb):
            return kernel.accept(prop, ea, eb, {}, {})

        result = jitted_accept(proposed, jnp.array(1.0), jnp.array(1.0))
        assert result.shape == ()

    def test_accept_ignores_pressure(self):
        """accept() adds no PV term: propose() already used the receiving
        run's pressure."""
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.5, 0.4)
        ens_a = {"pressure": jnp.array(0.1)}
        ens_b = {"pressure": jnp.array(0.2)}
        proposed["cell_a"] = jnp.eye(3) * 2.0
        proposed["cell_b"] = jnp.eye(3) * 2.0

        emax_a = jnp.array(1.0)
        emax_b = jnp.array(1.0)
        assert bool(kernel.accept(proposed, emax_a, emax_b, ens_a, ens_b))

        # Rejects on E_A = 0.5 > 0.45; a PV term would only raise E_A.
        emax_a_tight = jnp.array(0.45)
        assert not bool(
            kernel.accept(proposed, emax_a_tight, emax_b, ens_a, ens_b)
        )


class TestNSpeciesMismatch:
    """Check the errors for a bad n_species."""

    def test_invalid_n_species_zero_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            XRENSSwap(n_species=0)

    def test_invalid_n_species_negative_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            XRENSSwap(n_species=-1)

    def test_propose_composition_target_width_mismatch(self):
        """A target of the wrong width fails inside morph with a shape
        error; XRENSSwap has no explicit check."""
        kernel = XRENSSwap(n_species=3)
        backend = SpeciesHarmonicBackend([1.0, 2.0, 3.0])
        key = jax.random.key(0)
        types_a = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
        types_b = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        pos = jax.random.normal(key, (6, 3)) * 0.1
        e = backend(pos, types_a, jnp.zeros((3, 3))).energy

        state_a = _make_state(pos, types_a, e)
        state_b = _make_state(pos, types_b, e)
        ens_a = {"target_composition": jnp.array([3, 3], dtype=jnp.int32)}
        ens_b = {"target_composition": jnp.array([6, 0], dtype=jnp.int32)}

        with pytest.raises(TypeError):
            kernel.propose(state_a, state_b, ens_a, ens_b, key, backend)


class TestXRENSEndToEnd:
    """Check a short two-run NS with flavor="xrens" and the step function."""

    @classmethod
    def _run(cls, n_atoms=8, n_walkers=15, n_iters=15, seed=42):
        """Run XRENS with two different compositions and collect stats."""

        # Run 0 is all species 0, run 1 is half and half.
        target_a = [n_atoms, 0]
        target_b = [n_atoms // 2, n_atoms // 2]

        backend = SpeciesHarmonicBackend([1.0, 1.5])

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
            pos_key, (n_runs, n_walkers, n_atoms, 3), minval=-0.5, maxval=0.5
        )

        types_0 = jnp.zeros(n_atoms, dtype=jnp.int32)
        half = n_atoms // 2
        types_1 = jnp.concatenate(
            [
                jnp.zeros(half, dtype=jnp.int32),
                jnp.ones(n_atoms - half, dtype=jnp.int32),
            ]
        )
        types_stack = jnp.stack([types_0, types_1])

        def _eval_run(pos_run, types_run):
            return jax.vmap(
                lambda p: backend(p, types_run, jnp.zeros((3, 3)), 0)[0]
            )(pos_run)

        energies = jnp.stack(
            [
                _eval_run(positions[0], types_0),
                _eval_run(positions[1], types_1),
            ]
        )

        inter_re_cfg = InterREConfig(
            flavor="xrens",
            re_interval=1,
            n_swap_cycles=1,
            composition_targets=(tuple(target_a), tuple(target_b)),
        )

        re_stats_log = []

        class _StatCb:
            def on_iteration(self, iteration, ns_state, info):
                s = info.get("inter_re_stats")
                if s is not None:
                    re_stats_log.append(dict(s))

        result = run_ns_parallel(
            positions=positions,
            types=types_stack,
            energies=energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            n_walkers=n_walkers,
            max_iterations=n_iters,
            n_mcmc_steps=5,
            termination_criteria=[IterationTermination(n_iters)],
            inter_re_config=inter_re_cfg,
            backend=backend,
            callbacks=[_StatCb()],
        )
        return result, re_stats_log

    def test_result_shapes(self):
        result, re_stats_log = self._run()
        assert result["log_evidence"].shape == (2,)
        assert result["n_dead"].shape == (2,)
        assert jnp.all(jnp.isfinite(result["log_evidence"]))
        assert len(re_stats_log) > 0
        assert all(s["n_energy_evals"] == 2 for s in re_stats_log)

    def test_energy_evals_per_pair(self):
        """With 2 runs and 1 cycle there is one pair, and each attempted
        pair costs 2 energy evaluations."""
        n_runs, n_walkers, n_atoms, n_species = 2, 5, 8, 2
        backend = SpeciesHarmonicBackend([1.0, 1.5])
        kernel = XRENSSwap(n_species=n_species)

        key = jax.random.key(99)
        k1, k2 = jax.random.split(key)
        positions = (
            jax.random.normal(k1, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)

        # Run 0: all species 0; run 1: half-half.
        types = types.at[1, :, n_atoms // 2 :].set(1)

        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)

        comp_targets = jnp.array(
            [[n_atoms, 0], [n_atoms // 2, n_atoms // 2]], dtype=jnp.int32
        )
        emax = jnp.max(energies, axis=1)

        _, _, _, _, swap_info = xrens_replica_exchange_step(
            rng_key=k2,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            composition_targets=comp_targets,
            backend=backend,
            xrens_kernel=kernel,
            n_swap_cycles=1,
        )
        assert int(swap_info["n_attempted"]) == 1
        assert int(swap_info["n_energy_evals"]) == 2

    def test_accepted_swap_is_written_back(self):
        """With Emax 1e6 the only pair always swaps: each run gets the
        partner's positions, its own composition and a matching energy."""
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=2)
        k1, k2 = jax.random.split(jax.random.key(11))
        positions = jax.random.normal(k1, (2, 1, 4, 3))
        types = jnp.zeros((2, 1, 4), dtype=jnp.int32)
        types = types.at[1, :, 2:].set(1)

        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)
        comp_targets = jnp.array([[4, 0], [2, 2]], dtype=jnp.int32)

        (
            new_pos,
            new_types,
            new_ene,
            _,
            swap_info,
        ) = xrens_replica_exchange_step(
            rng_key=k2,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=jnp.array([1e6, 1e6]),
            composition_targets=comp_targets,
            backend=backend,
            xrens_kernel=kernel,
            n_swap_cycles=1,
        )

        assert int(swap_info["n_accepted"]) == 1
        assert jnp.allclose(new_pos[0, 0], positions[1, 0])
        assert jnp.allclose(new_pos[1, 0], positions[0, 0])
        for r in range(2):
            counts = jnp.bincount(new_types[r, 0], length=2)
            assert jnp.array_equal(counts, comp_targets[r])
            expected = _eval(new_pos[r, 0], new_types[r, 0])
            assert jnp.allclose(new_ene[r, 0], expected)

    def test_jit_xrens_step(self):
        n_runs, n_walkers, n_atoms, n_species = 2, 4, 6, 2
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=n_species)

        key = jax.random.key(10)
        k1, k2 = jax.random.split(key)
        positions = (
            jax.random.normal(k1, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
        types = types.at[1, :, n_atoms // 2 :].set(1)

        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)

        comp_targets = jnp.array(
            [[n_atoms, 0], [n_atoms // 2, n_atoms // 2]], dtype=jnp.int32
        )
        emax = jnp.max(energies, axis=1)

        @jax.jit
        def jitted_step(rng_key):
            return xrens_replica_exchange_step(
                rng_key=rng_key,
                all_positions=positions,
                all_types=types,
                all_energies=energies,
                all_cells=None,
                all_emax=emax,
                composition_targets=comp_targets,
                backend=backend,
                xrens_kernel=kernel,
                n_swap_cycles=1,
            )

        result = jitted_step(k2)
        new_pos, new_types, new_ene, _, swap_info = result

        assert new_pos.shape == positions.shape
        assert new_types.shape == types.shape
        assert new_ene.shape == energies.shape
        assert "n_accepted" in swap_info
        assert "n_attempted" in swap_info
        assert "n_energy_evals" in swap_info


class TestXRENSSingleRunSkip:
    """Check that a single run has no pairs and leaves the state alone."""

    def test_single_run_no_swaps(self):
        n_runs, n_walkers, n_atoms, n_species = 1, 5, 6, 2
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=n_species)

        key = jax.random.key(0)
        positions = (
            jax.random.normal(key, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)

        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)

        comp_targets = jnp.array([[n_atoms, 0]], dtype=jnp.int32)
        emax = jnp.max(energies, axis=1)

        (
            new_pos,
            new_types,
            new_ene,
            new_cells,
            swap_info,
        ) = xrens_replica_exchange_step(
            rng_key=key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            composition_targets=comp_targets,
            backend=backend,
            xrens_kernel=kernel,
            n_swap_cycles=1,
        )

        assert jnp.allclose(new_pos, positions)
        assert jnp.array_equal(new_types, types)
        assert jnp.allclose(new_ene, energies)
        assert int(swap_info["n_attempted"]) == 0
        assert int(swap_info["n_accepted"]) == 0
        assert int(swap_info["n_energy_evals"]) == 0


class TestInterRESpecXRENS:
    """Check the xrens validation of the inter_re config section."""

    def test_xrens_valid_spec(self):
        spec = InterRESpec(
            flavor="xrens",
            re_interval=1,
            n_swap_cycles=1,
            composition_targets=[[8, 0], [4, 4]],
        )
        cfg = spec.to_inter_re_config()
        assert cfg.flavor == "xrens"
        assert cfg.composition_targets == ((8, 0), (4, 4))

    def test_xrens_missing_composition_targets_raises(self):
        with pytest.raises(ValueError, match="requires 'composition_targets'"):
            InterRESpec(flavor="xrens")

    def test_xrens_inconsistent_row_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            InterRESpec(
                flavor="xrens",
                composition_targets=[[8, 0], [4, 4, 2]],
            )

    def test_xrens_inconsistent_sums_raises(self):
        with pytest.raises(ValueError, match="sum to the same"):
            InterRESpec(
                flavor="xrens",
                composition_targets=[[8, 0], [3, 4]],
            )
