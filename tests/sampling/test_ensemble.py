"""Tests for ensemble corrections (NPT/μPT) via EnsembleBackend.

Verifies:
- EnsembleBackend correctly adds the P·V (NPT) and −μ·N (μPT) terms
- the resolver threads per-replica ensemble params into the initial energy
  so resolved energies match the runtime NS loop by construction
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

from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.backends.toy import create_harmonic
from jaxrens.postprocess.thermodynamics import partition_function
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step, run_ns
from jaxrens.utils.cell import get_volume

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_periodic_setup(
    n_walkers=20, n_atoms=2, cell_size=5.0, seed=0, pressure=None
):
    """Create a periodic harmonic system with cells."""
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

    # Compute energies through the backend (includes PV if NPT)
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
        # Different volumes → different PV → different energies
        assert float(e_b) > float(e_a)

    def test_chemical_potentials_subtract_muN(self):
        # Grand-canonical: H = U - μ·N, applied via the per-call
        # ensemble_params key "chemical_potentials" (regression: the backend
        # previously read key "mu", so this term was silently dropped).
        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.0)
        pos = jnp.zeros((3, 3))
        types = jnp.array([0, 0, 1])  # counts: species 0 → 2, species 1 → 1
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
        # U - (1*2 + 2*1) = U - 4
        assert jnp.allclose(e_ens, e_raw - 4.0, atol=1e-5)

    def test_chemical_potentials_closured_default(self):
        # μ supplied to the constructor is used when no per-call override.
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
        # H = U + P*V - μ·N
        assert jnp.allclose(e_ens, e_raw + 0.01 * V - 4.0, atol=1e-4)


class TestInitialEnergyEnsembleThreading:
    """The resolver's initial-energy compute must thread the SAME per-replica
    ensemble params the runtime NS loop uses, so resolved energies include
    P·V and −μ·N by construction.

    Regression for the semi-grand bug where ``chemical_potentials`` were
    silently dropped from the initial energy (the resolver extracted only
    ``pressure``), so the first NS contour disagreed with the running chain
    and with ``SemiGrandSwap``'s ``Ω = U − μ·N`` assumption.
    """

    def test_finalise_threads_chemical_potentials_per_replica(self):
        import jax

        from jaxrens.cli.resolve import _finalise_initial_energies_and_counts
        from jaxrens.sampling.batch_descriptor import VmapRuns

        base = create_harmonic(k=1.0)
        backend = EnsembleBackend(base, pressure=0.0)

        n_runs, K, n_atoms = 2, 3, 2
        positions = jax.random.uniform(
            jax.random.key(0), (n_runs, K, n_atoms, 3)
        )
        cells = jnp.tile(5.0 * jnp.eye(3), (n_runs, K, 1, 1))
        types = jnp.array([0, 1], dtype=jnp.int32)  # N = [1, 1]

        # Distinct μ per replica → distinct −μ·N shift per run.
        mu = jnp.array([[1.0, 2.0], [0.5, 0.5]], dtype=jnp.float32)
        ep_batched = {"chemical_potentials": mu}

        energies, counts = _finalise_initial_energies_and_counts(
            backend,
            positions,
            types,
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
        expected = U - (mu @ N)[:, None]  # per-run μ·N, broadcast over K
        assert jnp.allclose(energies, expected, atol=1e-4)

    def test_finalise_without_ensemble_params_is_raw_energy(self):
        import jax

        from jaxrens.cli.resolve import _finalise_initial_energies_and_counts
        from jaxrens.sampling.batch_descriptor import VmapRuns

        base = create_harmonic(k=1.0)
        n_runs, K, n_atoms = 2, 3, 2
        positions = jax.random.uniform(
            jax.random.key(1), (n_runs, K, n_atoms, 3)
        )
        cells = jnp.tile(5.0 * jnp.eye(3), (n_runs, K, 1, 1))
        types = jnp.array([0, 1], dtype=jnp.int32)

        energies, counts = _finalise_initial_energies_and_counts(
            base,
            positions,
            types,
            cells,
            batcher=VmapRuns(n_runs),
            ensemble_params_batched=None,
        )
        U = jax.vmap(jax.vmap(lambda p, c: base(p, types, c, 0).energy))(
            positions, cells
        )
        assert jnp.allclose(energies, U, atol=1e-5)

    def test_stack_ensemble_params_shapes_and_keys(self):
        from jaxrens.cli.resolve import _stack_ensemble_params

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
        # 2-D (G, P) prefix reshapes leaves correctly.
        out2 = _stack_ensemble_params(params, ("chemical_potentials",), (1, 2))
        assert out2["chemical_potentials"].shape == (1, 2, 2)

    def test_stack_ensemble_params_none_when_no_energy_keys(self):
        from jaxrens.cli.resolve import _stack_ensemble_params

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
        from jaxrens.cli.resolve import _stack_ensemble_params

        params = [{"pressure": 0.01}, {}]
        assert _stack_ensemble_params(params, ("pressure",), (2,)) is None


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

    def test_dead_walker_volume_from_cell_matches_fixture(
        self,
        periodic_setup_npt,
    ):
        """Volume must derive correctly from ``dead_walker.cell``."""
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
        vol = float(jnp.abs(jnp.linalg.det(info["dead_walker"].cell)))
        assert vol == pytest.approx(125.0, abs=1e-3)

    def test_dead_walker_cell_matches_pre_cull_slot(self, periodic_setup_npt):
        """Dead walker's cell is the pre-cull cell at ``worst_idx``, not the
        clone's post-MCMC cell that now occupies that slot.
        """
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


# ---------------------------------------------------------------------------
# run_ns with pressure
# ---------------------------------------------------------------------------


@pytest.mark.heavy
class TestRunNSPressure:
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
        # Dead-point history (incl. dead_volumes) is no longer returned in the
        # result dict — it is persisted to disk by callbacks.  For an NVT run
        # the live-volume array is None because no pressure/ensemble term is set.
        assert result["live_volumes"] is None

    def test_npt_converges(self, periodic_setup_npt):
        s = periodic_setup_npt
        from jaxrens.backends.ensemble import make_ensemble_params

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
            1.0,
            dead_E,
            live_E,
            n_live=n_live,
            dead_volumes=dead_vols,
            live_volumes=live_vols,
        )

        assert not jnp.allclose(log_Z_nvt, log_Z_npt)
        assert float(log_Z_npt) < float(log_Z_nvt)

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
