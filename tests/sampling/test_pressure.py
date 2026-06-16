"""Tests for pressure/enthalpy (NPT) support via EnsembleBackend.

Verifies:
- EnsembleBackend correctly adds PV to energy
- NS state tracking of dead_volumes
- NPT NS runs converge correctly
- Post-processing with volumes
- Checkpoint round-trip of volumes
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns,
    ns_step,
    run_ns,
)
from jaxrens.postprocess.thermodynamics import partition_function
from jaxrens.utils.cell import get_volume


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_periodic_setup(n_walkers=20, n_atoms=2, cell_size=5.0, seed=0, pressure=None):
    """Create a periodic harmonic system with cells."""
    base_backend = create_harmonic(k=1.0)
    if pressure:
        backend = EnsembleBackend(base_backend, pressure=pressure)
    else:
        backend = base_backend

    init_fn, step_fn, _ = build_mwg(backend, [
        MoveKernel("random_walk", random_walk.build_kernel),
    ])

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.tile(cell_size * jnp.eye(3), (n_walkers, 1, 1))

    # Compute energies through the backend (includes PV if NPT)
    energies = jax.vmap(
        lambda pos, cell: backend(pos, types, cell, 0)[0]
    )(positions, cells)

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


# ---------------------------------------------------------------------------
# EnsembleBackend correctness
# ---------------------------------------------------------------------------

class TestEnsembleBackend:
    def test_no_pressure_returns_raw_energy(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.0)
        pos = jnp.array([[1.0, 0.0, 0.0]])
        types = jnp.array([0])
        cell = 5.0 * jnp.eye(3)
        e_raw, _, _ = base(pos, types, cell, 0).legacy()
        e_ens, _, _ = backend(pos, types, cell, 0).legacy()
        assert jnp.allclose(e_raw, e_ens)

    def test_finite_pressure_adds_pv(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.01)
        pos = jnp.array([[1.0, 0.0, 0.0]])
        types = jnp.array([0])
        cell = 5.0 * jnp.eye(3)
        e_raw, _, _ = base(pos, types, cell, 0).legacy()
        e_ens, _, _ = backend(pos, types, cell, 0).legacy()
        V = 5.0**3
        assert jnp.allclose(e_ens, e_raw + 0.01 * V, atol=1e-4)

    def test_different_volumes(self):
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.1)
        pos = jnp.array([[0.0, 0.0, 0.0]])
        types = jnp.array([0])
        cell_a = 4.0 * jnp.eye(3)
        cell_b = 6.0 * jnp.eye(3)
        e_a, _, _ = backend(pos, types, cell_a, 0).legacy()
        e_b, _, _ = backend(pos, types, cell_b, 0).legacy()
        # Different volumes → different PV → different energies
        assert float(e_b) > float(e_a)


# ---------------------------------------------------------------------------
# ns_step with pressure
# ---------------------------------------------------------------------------

class TestNSStepDeadWalker:
    """``info['dead_walker']`` is a ``WalkerState`` pytree (positions /
    types / energy / cell) holding the dead walker's pre-cull state.  This
    replaces the older parallel ``dead_position`` / ``dead_volume`` /
    ``dead_cell`` / ``dead_energy`` keys — see WORKLOG 2026-05-28.
    """

    def test_dead_walker_has_walker_state_fields(self, periodic_setup_npt):
        s = periodic_setup_npt
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=s["cells"], rng_key=s["key"],
        )
        _, info = ns_step(state, s["step_fn"], n_mcmc_steps=5)

        assert "dead_walker" in info
        dw = info["dead_walker"]
        n_atoms = s["positions"].shape[-2]
        assert dw.positions.shape == (n_atoms, 3)
        assert dw.types.shape == (n_atoms,)
        assert dw.cell.shape == (3, 3)
        assert dw.energy.shape == ()

    def test_dead_walker_volume_from_cell_matches_fixture(
        self, periodic_setup_npt,
    ):
        """Volume must derive correctly from ``dead_walker.cell``."""
        s = periodic_setup_npt
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=s["cells"], rng_key=s["key"],
        )
        _, info = ns_step(state, s["step_fn"], n_mcmc_steps=5)
        vol = float(jnp.abs(jnp.linalg.det(info["dead_walker"].cell)))
        assert vol == pytest.approx(125.0, abs=1e-3)

    def test_dead_walker_cell_matches_pre_cull_slot(self, periodic_setup_npt):
        """Dead walker's cell is the pre-cull cell at ``worst_idx``, not the
        clone's post-MCMC cell that now occupies that slot.
        """
        s = periodic_setup_npt
        state = init_ns(
            s["init_fn"],
            s["positions"], s["types"], s["energies"],
            cells=s["cells"], rng_key=s["key"],
        )
        _, info = ns_step(state, s["step_fn"], n_mcmc_steps=5)

        worst_idx = int(info["worst_idx"])
        np.testing.assert_allclose(
            np.asarray(info["dead_walker"].cell),
            np.asarray(s["cells"][worst_idx]),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# run_ns with pressure
# ---------------------------------------------------------------------------

@pytest.mark.heavy
class TestRunNSPressure:
    def test_nvt_no_volumes(self, periodic_setup):
        s = periodic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            cells=s["cells"], init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"], max_iterations=30,
            n_mcmc_steps=5, initial_step_size=0.3,
        )
        # Dead-point history (incl. dead_volumes) is no longer returned in the
        # result dict — it is persisted to disk by callbacks.  For an NVT run
        # the live-volume array is None because no pressure/ensemble term is set.
        assert result["live_volumes"] is None

    def test_npt_converges(self, periodic_setup_npt):
        s = periodic_setup_npt
        from jaxrens.backends.ensemble import make_ensemble_params
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            cells=s["cells"], init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"], max_iterations=100,
            n_mcmc_steps=5, initial_step_size=0.3,
            ensemble_params=make_ensemble_params(pressure=0.01),
        )
        assert jnp.isfinite(result["log_evidence"])
        assert result["n_dead"] > 0


# ---------------------------------------------------------------------------
# Post-processing with volumes
# ---------------------------------------------------------------------------

class TestPartitionFunctionWithVolumes:
    def test_pv_changes_partition_function(self):
        dead_E = jnp.linspace(0.5, 10.0, 100)
        live_E = jnp.linspace(10.0, 12.0, 20)
        n_live = 20

        log_Z_nvt = partition_function(1.0, dead_E, live_E, n_live=n_live)

        dead_vols = jnp.full(100, 1.25)
        live_vols = jnp.full(20, 1.25)
        log_Z_npt = partition_function(
            1.0, dead_E, live_E, n_live=n_live,
            dead_volumes=dead_vols, live_volumes=live_vols,
        )

        assert not jnp.allclose(log_Z_nvt, log_Z_npt)
        assert float(log_Z_npt) < float(log_Z_nvt)

    def test_zero_volumes_unchanged(self):
        dead_E = jnp.linspace(0.5, 10.0, 100)
        live_E = jnp.linspace(10.0, 12.0, 20)
        n_live = 20

        log_Z_base = partition_function(1.0, dead_E, live_E, n_live=n_live)
        log_Z_zero = partition_function(
            1.0, dead_E, live_E, n_live=n_live,
            dead_volumes=jnp.zeros(100), live_volumes=jnp.zeros(20),
        )

        assert jnp.allclose(log_Z_base, log_Z_zero, atol=1e-5)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigPressure:
    def test_pressure_parsed_from_input(self, tmp_path):
        from jaxrens.cli.parser import parse_input_file
        from jaxrens.cli.migrate import migrate_ns_inp

        inp = tmp_path / "ns.inp"
        inp.write_text("n_walkers = 100\npressure = 0.05\n")
        raw = parse_input_file(inp)
        result = migrate_ns_inp(raw)
        assert result["config"]["ensemble"]["pressure"] == pytest.approx(0.05)

    def test_no_pressure_in_input(self, tmp_path):
        from jaxrens.cli.parser import parse_input_file
        from jaxrens.cli.migrate import migrate_ns_inp

        inp = tmp_path / "ns.inp"
        inp.write_text("n_walkers = 100\n")
        raw = parse_input_file(inp)
        result = migrate_ns_inp(raw)
        assert "pressure" not in result["config"].get("ensemble", {})
