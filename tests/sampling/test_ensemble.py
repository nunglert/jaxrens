"""Tests for the NPT and muPT corrections of EnsembleBackend."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.ensemble import EnsembleBackend, make_ensemble_params
from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.resolve import (
    _finalise_initial_energies_and_counts,
    _stack_ensemble_params,
)
from jaxrens.postprocess.thermodynamics import partition_function
from jaxrens.sampling.batch_descriptor import VmapRuns
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns


def _make_periodic_setup(
    n_walkers=20, n_atoms=2, cell_size=5.0, seed=0, pressure=None
):
    """Create a periodic harmonic system, with PV added if pressure is set."""
    base_backend = create_harmonic(k=1.0)
    if pressure:
        backend = EnsembleBackend(base_backend, pressure=pressure)
    else:
        backend = base_backend

    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("random_walk", random_walk.build_kernel),
        ],
    )

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.tile(cell_size * jnp.eye(3), (n_walkers, 1, 1))

    energies = jax.vmap(lambda pos, cell: backend(pos, types, cell, 0)[0])(
        positions, cells
    )

    return {
        "backend": backend,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": cells,
        "key": key,
        "n_walkers": n_walkers,
    }


@pytest.fixture
def periodic_setup():
    return _make_periodic_setup()


@pytest.fixture
def periodic_setup_npt():
    return _make_periodic_setup(pressure=0.01)


class TestEnsembleBackend:
    """Check the P*V and -mu*N terms that EnsembleBackend adds to U."""

    def test_no_pressure_returns_raw_energy(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.0)
        pos = jnp.array([[1.0, 0.0, 0.0]])
        types = jnp.array([0])
        cell = 5.0 * jnp.eye(3)
        e_raw = base(pos, types, cell, 0).energy
        e_ens = backend(pos, types, cell, 0).energy
        assert jnp.allclose(e_raw, e_ens)

    def test_finite_pressure_adds_pv(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.01)
        pos = jnp.array([[1.0, 0.0, 0.0]])
        types = jnp.array([0])
        cell = 5.0 * jnp.eye(3)
        e_raw = base(pos, types, cell, 0).energy
        e_ens = backend(pos, types, cell, 0).energy
        V = 5.0**3
        assert jnp.allclose(e_ens, e_raw + 0.01 * V, atol=1e-4)

    def test_different_volumes(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.1)
        pos = jnp.array([[0.0, 0.0, 0.0]])
        types = jnp.array([0])
        cell_a = 4.0 * jnp.eye(3)
        cell_b = 6.0 * jnp.eye(3)
        e_a = backend(pos, types, cell_a, 0).energy
        e_b = backend(pos, types, cell_b, 0).energy
        assert jnp.allclose(e_b - e_a, 0.1 * (6.0**3 - 4.0**3), atol=1e-4)

    def test_chemical_potentials_subtract_muN(self):
        # Per-call ensemble_params["chemical_potentials"] gives H = U - mu*N.
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.0)
        pos = jnp.zeros((3, 3))
        types = jnp.array([0, 0, 1])  # N = [2, 1]
        cell = 5.0 * jnp.eye(3)
        mu = jnp.array([1.0, 2.0], dtype=jnp.float32)
        e_raw = base(pos, types, cell, 0).energy
        e_ens = backend(
            pos,
            types,
            cell,
            0,
            ensemble_params={"chemical_potentials": mu},
        ).energy
        # mu.N = 1*2 + 2*1 = 4
        assert jnp.allclose(e_ens, e_raw - 4.0, atol=1e-5)

    def test_chemical_potentials_closured_default(self):
        # A mu given to the constructor is the default.
        base = create_harmonic(k=1.0)
        mu = jnp.array([1.0, 2.0], dtype=jnp.float32)
        backend = EnsembleBackend(base, chemical_potentials=mu)
        pos = jnp.zeros((3, 3))
        types = jnp.array([0, 0, 1])
        cell = 5.0 * jnp.eye(3)
        e_raw = base(pos, types, cell, 0).energy
        e_ens = backend(pos, types, cell, 0).energy
        assert jnp.allclose(e_ens, e_raw - 4.0, atol=1e-5)

    def test_pressure_and_chemical_potentials_combined(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.01)
        pos = jnp.zeros((3, 3))
        types = jnp.array([0, 0, 1])
        cell = 5.0 * jnp.eye(3)
        mu = jnp.array([1.0, 2.0], dtype=jnp.float32)
        e_raw = base(pos, types, cell, 0).energy
        e_ens = backend(
            pos,
            types,
            cell,
            0,
            ensemble_params={"chemical_potentials": mu},
        ).energy
        V = 5.0**3
        assert jnp.allclose(e_ens, e_raw + 0.01 * V - 4.0, atol=1e-4)


class TestInitialEnergyEnsembleThreading:
    """Check that initial energies use the per-replica ensemble params.

    The initial energies must include P*V and -mu*N exactly as the NS loop
    does, so the first contour matches the running chain.
    """

    def test_finalise_threads_chemical_potentials_per_replica(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.0)

        n_runs, K, n_atoms = 2, 3, 2
        positions = jax.random.uniform(
            jax.random.key(0), (n_runs, K, n_atoms, 3)
        )
        cells = jnp.tile(5.0 * jnp.eye(3), (n_runs, K, 1, 1))
        types = jnp.array([0, 1], dtype=jnp.int32)  # N = [1, 1]
        # The seam maps types over the run and walker axes, so it needs the
        # full (n_runs, K, n_atoms) shape.
        types_batched = jnp.broadcast_to(types, (n_runs, K, n_atoms))

        # A distinct mu per replica gives a distinct -mu.N shift per run.
        mu = jnp.array([[1.0, 2.0], [0.5, 0.5]], dtype=jnp.float32)
        ep_batched = {"chemical_potentials": mu}

        energies, counts = _finalise_initial_energies_and_counts(
            backend,
            positions,
            types_batched,
            cells,
            batcher=VmapRuns(n_runs),
            ensemble_params_batched=ep_batched,
        )
        assert counts is None
        assert energies.shape == (n_runs, K)

        U = jax.vmap(jax.vmap(lambda p, c: base(p, types, c, 0).energy))(
            positions, cells
        )
        N = jnp.array([1.0, 1.0])
        expected = U - (mu @ N)[:, None]
        assert jnp.allclose(energies, expected, atol=1e-4)

    def test_finalise_without_ensemble_params_is_raw_energy(self):
        base = create_harmonic(k=1.0)
        n_runs, K, n_atoms = 2, 3, 2
        positions = jax.random.uniform(
            jax.random.key(1), (n_runs, K, n_atoms, 3)
        )
        cells = jnp.tile(5.0 * jnp.eye(3), (n_runs, K, 1, 1))
        types = jnp.array([0, 1], dtype=jnp.int32)
        types_batched = jnp.broadcast_to(types, (n_runs, K, n_atoms))

        energies, counts = _finalise_initial_energies_and_counts(
            base,
            positions,
            types_batched,
            cells,
            batcher=VmapRuns(n_runs),
            ensemble_params_batched=None,
        )
        U = jax.vmap(jax.vmap(lambda p, c: base(p, types, c, 0).energy))(
            positions, cells
        )
        assert jnp.allclose(energies, U, atol=1e-5)

    def test_stack_ensemble_params_shapes_and_keys(self):
        params = [
            {"pressure": 0.01, "chemical_potentials": jnp.array([1.0, 2.0])},
            {"pressure": 0.02, "chemical_potentials": jnp.array([3.0, 4.0])},
        ]
        out = _stack_ensemble_params(
            params, ("pressure", "chemical_potentials"), (2,)
        )
        assert set(out) == {"pressure", "chemical_potentials"}
        assert out["pressure"].shape == (2,)
        assert out["chemical_potentials"].shape == (2, 2)
        assert jnp.allclose(
            out["chemical_potentials"], jnp.array([[1.0, 2.0], [3.0, 4.0]])
        )
        # A 2-D (G, P) prefix reshapes the leaves.
        out2 = _stack_ensemble_params(params, ("chemical_potentials",), (1, 2))
        assert out2["chemical_potentials"].shape == (1, 2, 2)

    def test_stack_ensemble_params_none_when_no_energy_keys(self):
        # target_composition is an XRENS morph target, not an energy term.
        params = [
            {"target_composition": jnp.array([1, 1])},
            {"target_composition": jnp.array([1, 1])},
        ]
        assert (
            _stack_ensemble_params(
                params, ("pressure", "chemical_potentials"), (2,)
            )
            is None
        )

    def test_stack_ensemble_params_skips_partial_keys(self):
        # A key present in only some replicas is skipped (no ragged stack).
        params = [{"pressure": 0.01}, {}]
        assert _stack_ensemble_params(params, ("pressure",), (2,)) is None


class TestNSStepDeadWalker:
    """Check info['dead_walker'], the pre-cull WalkerState of the dead point.

    It holds positions, types, energy and cell of the walker at ``worst_idx``.
    """

    def test_dead_walker_has_walker_state_fields(self, periodic_setup_npt):
        s = periodic_setup_npt
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=s["cells"],
            rng_key=s["key"],
        )
        _, info = ns_step(state, s["step_fn"], n_mcmc_steps=5)

        assert "dead_walker" in info
        dw = info["dead_walker"]
        n_atoms = s["positions"].shape[-2]
        assert dw.positions.shape == (n_atoms, 3)
        assert dw.types.shape == (n_atoms,)
        assert dw.cell.shape == (3, 3)
        assert dw.energy.shape == ()

    def test_dead_walker_cell_matches_pre_cull_slot(self, periodic_setup_npt):
        """The cell is the pre-cull one at ``worst_idx``, not the clone's."""
        s = periodic_setup_npt
        state = init_ns(
            s["init_fn"],
            s["positions"],
            s["types"],
            s["energies"],
            cells=s["cells"],
            rng_key=s["key"],
        )
        _, info = ns_step(state, s["step_fn"], n_mcmc_steps=5)

        worst_idx = int(info["worst_idx"])
        np.testing.assert_allclose(
            np.asarray(info["dead_walker"].cell),
            np.asarray(s["cells"][worst_idx]),
            atol=1e-6,
        )


@pytest.mark.heavy
class TestRunNSPressure:
    """Check run_ns with and without a pressure term."""

    def test_nvt_no_volumes(self, periodic_setup):
        s = periodic_setup
        result = run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=s["cells"],
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=30,
            n_mcmc_steps=5,
            initial_step_size=0.3,
        )
        # Without a pressure or ensemble term there are no live volumes.
        assert result["live_volumes"] is None

    def test_npt_converges(self, periodic_setup_npt):
        s = periodic_setup_npt

        result = run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=s["cells"],
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=100,
            n_mcmc_steps=5,
            initial_step_size=0.3,
            ensemble_params=make_ensemble_params(pressure=0.01),
        )
        assert jnp.isfinite(result["log_evidence"])
        assert result["n_dead"] > 0

    def test_chemical_potential_converges(self, periodic_setup_npt):
        s = periodic_setup_npt
        params = make_ensemble_params(
            pressure=0.01, chemical_potentials=jnp.array([1.0])
        )
        energies = jax.vmap(
            lambda pos, cell: s["backend"](
                pos, s["types"], cell, 0, ensemble_params=params
            )[0]
        )(s["positions"], s["cells"])

        result = run_ns(
            s["positions"],
            s["types"],
            energies,
            cells=s["cells"],
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=100,
            n_mcmc_steps=5,
            initial_step_size=0.3,
            ensemble_params=params,
        )
        assert jnp.isfinite(result["log_evidence"])
        assert result["n_dead"] > 0


class TestPartitionFunctionWithVolumes:
    """Check how dead and live volumes enter partition_function."""

    def test_pv_changes_partition_function(self):
        dead_E = jnp.linspace(0.5, 10.0, 100)
        live_E = jnp.linspace(10.0, 12.0, 20)
        n_live = 20

        log_Z_nvt = partition_function(1.0, dead_E, live_E, n_live=n_live)

        dead_vols = jnp.full(100, 1.25)
        live_vols = jnp.full(20, 1.25)
        log_Z_npt = partition_function(
            1.0,
            dead_E,
            live_E,
            n_live=n_live,
            dead_volumes=dead_vols,
            live_volumes=live_vols,
        )

        # A constant contribution c shifts log Z by -beta * c.
        assert jnp.allclose(log_Z_npt, log_Z_nvt - 1.25, atol=1e-5)

    def test_zero_volumes_unchanged(self):
        dead_E = jnp.linspace(0.5, 10.0, 100)
        live_E = jnp.linspace(10.0, 12.0, 20)
        n_live = 20

        log_Z_base = partition_function(1.0, dead_E, live_E, n_live=n_live)
        log_Z_zero = partition_function(
            1.0,
            dead_E,
            live_E,
            n_live=n_live,
            dead_volumes=jnp.zeros(100),
            live_volumes=jnp.zeros(20),
        )

        assert jnp.allclose(log_Z_base, log_Z_zero, atol=1e-5)
