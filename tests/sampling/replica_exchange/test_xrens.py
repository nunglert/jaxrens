"""Tests for XRENSSwap, the composition-morphing replica exchange.

A walker that enters another run gets its atom types morphed to that run's
target composition, and its energy is evaluated again under that run's
ensemble params. The tests cover propose(), replica_exchange_step with
XRENSSwap, a short two-run NS with flavor="xrens", and the xrens checks of
InterRESpec.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from jaxrens.backends.base import BackendResult
from jaxrens.cli.schema.inter_re import InterRESpec
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.moves.replica_exchange import (
    SwapWalker,
    XRENSSwap,
    make_swap_kernel,
    replica_exchange_step,
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
        weights = self.species_weights[types]  # (n_atoms,)
        energy = jnp.sum(weights * 0.5 * jnp.sum(positions**2, axis=-1))
        return BackendResult(energy=energy)


def _make_state(positions, types, energy, cell=None):
    """One walker as passed to XRENSSwap.propose."""
    return SwapWalker(positions, types, jnp.asarray(energy), cell)


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


def _xrens_step(
    rng_key,
    all_positions,
    all_types,
    all_energies,
    all_cells,
    all_emax,
    composition_targets,
    backend,
    xrens_kernel,
    pressures=None,
    n_swap_cycles=1,
):
    """Run replica_exchange_step with XRENS per-run params."""
    run_params = {"target_composition": composition_targets}
    if pressures is not None:
        run_params["pressure"] = pressures
    new, info = replica_exchange_step(
        rng_key,
        SwapWalker(all_positions, all_types, all_energies, all_cells),
        all_emax,
        run_params,
        xrens_kernel,
        backend=backend,
        n_swap_cycles=n_swap_cycles,
    )
    return new.positions, new.types, new.energy, new.cell, info


class TestXRENSProposeMorphedTypes:
    """Check that propose() gives each incoming walker the receiving run's
    composition, the partner's positions, and the matching energy.
    """

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

        proposed = _propose(
            kernel, state_a, state_b, ens_a, ens_b, morph_key, backend
        )
        return proposed, backend

    def test_morphed_types_a_match_target(self):
        proposed, _ = self._run_propose(8, 2, [8, 0], [4, 4])
        got_a = jnp.bincount(proposed["types_a"], length=2)
        assert jnp.array_equal(got_a, jnp.array([8, 0]))

    def test_morphed_types_b_match_target(self):
        proposed, _ = self._run_propose(8, 2, [8, 0], [4, 4])
        got_b = jnp.bincount(proposed["types_b"], length=2)
        assert jnp.array_equal(got_b, jnp.array([4, 4]))

    def test_energies_match_morphed_configs(self):
        """Each proposed energy is the backend energy of the proposed
        positions with the morphed types."""
        proposed, backend = self._run_propose(8, 2, [8, 0], [4, 4])
        for side in ("a", "b"):
            expected = backend(
                proposed[f"positions_{side}"],
                proposed[f"types_{side}"],
                jnp.zeros((3, 3)),
            ).energy
            assert jnp.allclose(proposed[f"energy_{side}"], expected)

    def test_positions_a_are_from_b(self):
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

        proposed = _propose(
            kernel, state_a, state_b, ens_a, ens_b, mkey, backend
        )
        assert jnp.allclose(proposed["positions_a"], pos_b)
        assert jnp.allclose(proposed["positions_b"], pos_a)

    def test_propose_many_seeds(self):
        """The compositions match the targets for several seeds."""
        target_a = [12, 0]
        target_b = [6, 6]
        for seed in range(5):
            proposed, _ = self._run_propose(
                12, 2, target_a, target_b, seed=seed
            )
            got_a = jnp.bincount(proposed["types_a"], length=2)
            got_b = jnp.bincount(proposed["types_b"], length=2)
            assert jnp.array_equal(got_a, jnp.array(target_a)), seed
            assert jnp.array_equal(got_b, jnp.array(target_b)), seed


class _PressureAwareBackend(SpeciesHarmonicBackend):
    """SpeciesHarmonicBackend + P·V from ensemble_params, like EnsembleTerm."""

    def __call__(
        self, positions, types, cell, max_neighbors=0, ensemble_params=None
    ):
        result = super().__call__(positions, types, cell, max_neighbors)
        p = (ensemble_params or {}).get("pressure", 0.0)
        return BackendResult(energy=result.energy + p * jnp.linalg.det(cell))


def test_energy_uses_receiving_pressure():
    """The walker entering run A is evaluated at P_A, not at its home P."""
    backend = _PressureAwareBackend([1.0, 1.0])
    kernel = XRENSSwap(n_species=2)
    pos = jnp.ones((4, 3))
    types = jnp.zeros(4, dtype=jnp.int32)
    cell_a, cell_b = jnp.eye(3), 2.0 * jnp.eye(3)  # V = 1, 8
    u = float(SpeciesHarmonicBackend([1.0, 1.0])(pos, types, None).energy)
    wa = _make_state(pos, types, 0.0, cell_a)
    wb = _make_state(pos, types, 0.0, cell_b)
    target = jnp.array([4, 0])
    ens_a = {"target_composition": target, "pressure": jnp.array(0.1)}
    ens_b = {"target_composition": target, "pressure": jnp.array(0.5)}

    proposed = _propose(
        kernel, wa, wb, ens_a, ens_b, jax.random.key(0), backend
    )

    # Walker B (V = 8) enters run A (P = 0.1); walker A (V = 1) enters
    # run B (P = 0.5).
    assert float(proposed["energy_a"]) == pytest.approx(u + 0.8, rel=1e-5)
    assert float(proposed["energy_b"]) == pytest.approx(u + 0.5, rel=1e-5)


class TestNSpeciesMismatch:
    """Check that a bad n_species or a target of the wrong width raises."""

    def test_invalid_n_species_zero_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            XRENSSwap(n_species=0)

    def test_invalid_n_species_negative_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            XRENSSwap(n_species=-1)

    def test_propose_composition_target_width_mismatch(self):
        """A 2-element target for XRENSSwap(n_species=3) raises."""
        kernel = XRENSSwap(n_species=3)
        backend = SpeciesHarmonicBackend([1.0, 2.0, 3.0])
        key = jax.random.key(0)
        types = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
        pos = jax.random.normal(key, (6, 3)) * 0.1
        state = _make_state(pos, types, 0.0)
        ens_a = {"target_composition": jnp.array([3, 3], dtype=jnp.int32)}
        ens_b = {"target_composition": jnp.array([6, 0], dtype=jnp.int32)}
        with pytest.raises(ValueError, match="n_species"):
            kernel.propose(state, state, ens_a, ens_b, key, backend)


def test_make_swap_kernel_takes_n_species_from_targets():
    """make_swap_kernel sets n_species to the width of the targets."""
    cfg = InterREConfig(
        flavor="xrens", composition_targets=((6, 0, 0), (2, 2, 2))
    )
    kernel = make_swap_kernel(cfg)
    assert isinstance(kernel, XRENSSwap)
    assert kernel.n_species == 3


class TestXRENSEndToEnd:
    """Check a short two-run NS with flavor="xrens" and replica_exchange_step
    with XRENSSwap.
    """

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
        """The run finishes, and every swap pass reports two energy
        evaluations (one pair, two morphed walkers)."""
        result, re_stats_log = self._run()
        assert result["log_evidence"].shape == (2,)
        assert result["n_dead"].shape == (2,)
        assert jnp.all(jnp.isfinite(result["log_evidence"]))
        assert len(re_stats_log) > 0
        assert all(s["n_energy_evals"] == 2 for s in re_stats_log)

    def test_energy_evals_per_pair(self):
        """With 2 runs and one cycle there is one attempt, which costs two
        energy evaluations."""
        n_runs, n_walkers, n_atoms, n_species = 2, 5, 8, 2
        backend = SpeciesHarmonicBackend([1.0, 1.5])
        kernel = XRENSSwap(n_species=n_species)

        key = jax.random.key(99)
        k1, k2 = jax.random.split(key)
        positions = (
            jax.random.normal(k1, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)

        # Run 1 has half of its atoms in species 1.
        types = types.at[1, :, n_atoms // 2 :].set(1)

        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)

        comp_targets = jnp.array(
            [[n_atoms, 0], [n_atoms // 2, n_atoms // 2]], dtype=jnp.int32
        )
        emax = jnp.max(energies, axis=1)

        _, _, _, _, swap_info = _xrens_step(
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

    def test_jit_xrens_step(self):
        """replica_exchange_step with XRENSSwap runs under jit."""
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
            return _xrens_step(
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


def test_single_run_no_swaps():
    """With one run there are no pairs, so nothing changes."""
    n_runs, n_walkers, n_atoms, n_species = 1, 5, 6, 2
    backend = SpeciesHarmonicBackend([1.0, 2.0])
    kernel = XRENSSwap(n_species=n_species)

    key = jax.random.key(0)
    positions = jax.random.normal(key, (n_runs, n_walkers, n_atoms, 3)) * 0.1
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
    ) = _xrens_step(
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
    """Check the xrens checks of the InterRESpec schema."""

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
        with pytest.raises(
            ValidationError, match="requires 'composition_targets'"
        ):
            InterRESpec(flavor="xrens")

    def test_xrens_inconsistent_row_lengths_raises(self):
        with pytest.raises(ValidationError, match="same length"):
            InterRESpec(
                flavor="xrens",
                composition_targets=[[8, 0], [4, 4, 2]],
            )

    def test_xrens_inconsistent_sums_raises(self):
        """Rows that sum to 8 and 7 atoms raise."""
        with pytest.raises(ValidationError, match="same n_atoms"):
            InterRESpec(
                flavor="xrens",
                composition_targets=[[8, 0], [3, 4]],
            )
