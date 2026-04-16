"""Tests for pressure/enthalpy (NPT) support.

Verifies:
- Enthalpy computation H = E + PV
- NS state tracking of dead_volumes
- Pressure=None/0 is backward-compatible with NVT
- NPT NS runs converge correctly
- Post-processing with volumes
- Checkpoint round-trip of volumes
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.move_descriptor import MoveDescriptor
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    _compute_enthalpies,
    init_ns,
    ns_step,
    run_ns,
)
from jaxrens.sampling.adaptation.step_size import init_adaptation
from jaxrens.postprocess.thermodynamics import partition_function
from jaxrens.utils.cell import get_volume


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_periodic_setup(n_walkers=20, n_atoms=2, box_size=5.0, seed=0):
    """Create a periodic harmonic system with boxes."""
    energy_fn, params = create_harmonic(k=1.0)
    init_fn, step_fn = build_mwg(energy_fn, params, [
        MoveDescriptor("random_walk", random_walk.build_kernel),
    ])

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    boxes = jnp.tile(box_size * jnp.eye(3), (n_walkers, 1, 1))
    energies = jax.vmap(energy_fn, in_axes=(None, 0, None))(
        params, positions, types
    )
    return {
        "energy_fn": energy_fn,
        "params": params,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "boxes": boxes,
        "key": key,
        "n_walkers": n_walkers,
    }


@pytest.fixture
def periodic_setup():
    return _make_periodic_setup()


# ---------------------------------------------------------------------------
# Enthalpy computation
# ---------------------------------------------------------------------------

class TestComputeEnthalpies:
    def test_no_pressure_returns_energies(self):
        energies = jnp.array([1.0, 2.0, 3.0])
        result = _compute_enthalpies(energies, boxes=None, pressure=None)
        assert jnp.allclose(result, energies)

    def test_zero_pressure_returns_energies(self):
        energies = jnp.array([1.0, 2.0, 3.0])
        boxes = jnp.tile(5.0 * jnp.eye(3), (3, 1, 1))
        result = _compute_enthalpies(energies, boxes, pressure=0.0)
        assert jnp.allclose(result, energies)

    def test_finite_pressure_adds_pv(self):
        energies = jnp.array([1.0, 2.0, 3.0])
        box_size = 5.0
        boxes = jnp.tile(box_size * jnp.eye(3), (3, 1, 1))
        pressure = 0.01
        V = box_size**3  # 125.0

        result = _compute_enthalpies(energies, boxes, pressure)
        expected = energies + pressure * V
        assert jnp.allclose(result, expected)

    def test_pressure_without_boxes_raises(self):
        energies = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="pressure is set but boxes are None"):
            _compute_enthalpies(energies, boxes=None, pressure=0.01)

    def test_different_volumes(self):
        """Walkers with different cell sizes get different PV terms."""
        energies = jnp.array([1.0, 1.0])
        box_a = 4.0 * jnp.eye(3)
        box_b = 6.0 * jnp.eye(3)
        boxes = jnp.stack([box_a, box_b])
        pressure = 0.1

        result = _compute_enthalpies(energies, boxes, pressure)
        # V_a = 64, V_b = 216; H = E + PV
        assert float(result[0]) == pytest.approx(1.0 + 0.1 * 64.0, abs=1e-4)
        assert float(result[1]) == pytest.approx(1.0 + 0.1 * 216.0, abs=1e-4)


# ---------------------------------------------------------------------------
# init_ns with pressure
# ---------------------------------------------------------------------------

class TestInitNSPressure:
    def test_no_pressure_no_dead_volumes(self, periodic_setup):
        s = periodic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], rng_key=s["key"],
        )
        assert state["dead_volumes"] is None

    def test_pressure_creates_dead_volumes(self, periodic_setup):
        s = periodic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], rng_key=s["key"],
            max_dead=100, pressure=0.01,
        )
        assert state["dead_volumes"] is not None
        assert state["dead_volumes"].shape == (100,)
        # Initially all zeros
        assert jnp.allclose(state["dead_volumes"], 0.0)


# ---------------------------------------------------------------------------
# ns_step with pressure
# ---------------------------------------------------------------------------

class TestNSStepPressure:
    def test_step_records_dead_volume(self, periodic_setup):
        s = periodic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], rng_key=s["key"],
            max_dead=100, pressure=0.01,
        )
        adapt = init_adaptation(initial_step_size=0.3)

        new_state, info, _ = ns_step(
            state, s["init_fn"], s["step_fn"], n_mcmc_steps=5,
            adapt_state=adapt, pressure=0.01,
        )

        # dead_volumes[0] should have the volume of the worst walker's cell
        dv = float(new_state["dead_volumes"][0])
        assert dv > 0, "Dead volume should be recorded"
        # For 5.0*eye(3) boxes, volume = 125.0
        assert dv == pytest.approx(125.0, abs=1e-3)

    def test_hmax_differs_from_emax_with_pressure(self, periodic_setup):
        s = periodic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], rng_key=s["key"],
            max_dead=100, pressure=0.01,
        )
        adapt = init_adaptation(initial_step_size=0.3)

        _, info, _ = ns_step(
            state, s["init_fn"], s["step_fn"], n_mcmc_steps=5,
            adapt_state=adapt, pressure=0.01,
        )

        # hmax = emax + P*V, so hmax > emax when P > 0
        assert float(info["hmax"]) > float(info["emax"])

    def test_zero_pressure_hmax_equals_emax(self, periodic_setup):
        s = periodic_setup
        state = init_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], rng_key=s["key"],
        )
        adapt = init_adaptation(initial_step_size=0.3)

        _, info, _ = ns_step(
            state, s["init_fn"], s["step_fn"], n_mcmc_steps=5,
            adapt_state=adapt, pressure=None,
        )

        assert float(info["hmax"]) == float(info["emax"])


# ---------------------------------------------------------------------------
# run_ns with pressure
# ---------------------------------------------------------------------------

@pytest.mark.heavy
class TestRunNSPressure:
    def test_nvt_no_volumes(self, periodic_setup):
        """NVT run (no pressure) should have no dead_volumes or live_volumes."""
        s = periodic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"], max_iterations=30,
            n_mcmc_steps=5, initial_step_size=0.3,
        )
        assert result["dead_volumes"] is None
        assert result["live_volumes"] is None

    def test_npt_records_volumes(self, periodic_setup):
        """NPT run should populate dead_volumes and live_volumes."""
        s = periodic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"], max_iterations=30,
            n_mcmc_steps=5, initial_step_size=0.3,
            pressure=0.01,
        )
        assert result["dead_volumes"] is not None
        n_dead = result["n_dead"]
        # Recorded volumes for dead points should be > 0
        recorded = result["dead_volumes"][:n_dead]
        assert jnp.all(recorded > 0)

        assert result["live_volumes"] is not None
        assert result["live_volumes"].shape == (s["n_walkers"],)
        assert jnp.all(result["live_volumes"] > 0)

    def test_npt_converges(self, periodic_setup):
        """NPT run should converge to finite evidence."""
        s = periodic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"], max_iterations=100,
            n_mcmc_steps=5, initial_step_size=0.3,
            pressure=0.01,
        )
        assert jnp.isfinite(result["log_evidence"])
        assert result["n_dead"] > 0


# ---------------------------------------------------------------------------
# Post-processing with volumes
# ---------------------------------------------------------------------------

class TestPartitionFunctionWithVolumes:
    def test_pv_changes_partition_function(self):
        """partition_function with volumes should differ from without."""
        dead_E = jnp.linspace(0.5, 10.0, 100)
        live_E = jnp.linspace(10.0, 12.0, 20)
        n_live = 20

        log_Z_nvt = partition_function(1.0, dead_E, live_E, n_live=n_live)

        # Add PV contribution (pressure * volume)
        dead_vols = jnp.full(100, 1.25)  # pressure * volume terms
        live_vols = jnp.full(20, 1.25)
        log_Z_npt = partition_function(
            1.0, dead_E, live_E, n_live=n_live,
            dead_volumes=dead_vols, live_volumes=live_vols,
        )

        assert not jnp.allclose(log_Z_nvt, log_Z_npt)
        # Adding positive PV shifts energies up, reducing Z
        assert float(log_Z_npt) < float(log_Z_nvt)

    def test_zero_volumes_unchanged(self):
        """Zero PV contributions should give same result as no volumes."""
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
# Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpointPressure:
    def test_checkpoint_roundtrip_with_volumes(self, periodic_setup):
        """Checkpoint save/load should preserve dead_volumes."""
        from jaxrens.io.checkpoint import save_checkpoint, load_checkpoint

        s = periodic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"], max_iterations=30,
            n_mcmc_steps=5, initial_step_size=0.3,
            pressure=0.01,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.checkpoint.h5"
            save_checkpoint(path, result)
            loaded = load_checkpoint(path, rng_key=jax.random.key(99))

        n_dead = result["n_dead"]
        assert loaded["dead_volumes"] is not None
        assert jnp.allclose(
            loaded["dead_volumes"][:n_dead],
            result["dead_volumes"][:n_dead],
            atol=1e-5,
        )
        assert loaded["live_volumes"] is not None
        assert jnp.allclose(
            loaded["live_volumes"],
            result["live_volumes"],
            atol=1e-5,
        )

    def test_checkpoint_roundtrip_without_volumes(self, periodic_setup):
        """NVT checkpoint should load cleanly without volumes."""
        from jaxrens.io.checkpoint import save_checkpoint, load_checkpoint

        s = periodic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
            boxes=s["boxes"], init_fn=s["init_fn"], step_fn=s["step_fn"],
            rng_key=s["key"], max_iterations=30,
            n_mcmc_steps=5, initial_step_size=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.checkpoint.h5"
            save_checkpoint(path, result)
            loaded = load_checkpoint(path, rng_key=jax.random.key(99))

        assert loaded["dead_volumes"] is None
        assert loaded["live_volumes"] is None


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigPressure:
    def test_pressure_parsed_from_input(self, tmp_path):
        from jaxrens.cli.parser import parse_input_file, raw_to_configs

        inp = tmp_path / "ns.inp"
        inp.write_text("n_walkers = 100\npressure = 0.05\n")
        raw = parse_input_file(inp)
        ns_config, _, _, _ = raw_to_configs(raw)
        assert ns_config.pressure == pytest.approx(0.05)

    def test_no_pressure_in_input(self, tmp_path):
        from jaxrens.cli.parser import parse_input_file, raw_to_configs

        inp = tmp_path / "ns.inp"
        inp.write_text("n_walkers = 100\n")
        raw = parse_input_file(inp)
        ns_config, _, _, _ = raw_to_configs(raw)
        assert ns_config.pressure is None
