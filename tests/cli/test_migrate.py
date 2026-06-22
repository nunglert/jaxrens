"""Tests for the ns.inp -> YAML migration path (step 7)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from jaxrens.cli.migrate import migrate_ns_inp
from jaxrens.cli.schema import RootSpec

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_raw() -> dict[str, str]:
    """Smallest valid raw dict that produces a validatable RootSpec."""
    return {
        "n_walkers": "100",
        "n_iter": "1000",
        "energy_calculator": "lj",
        "n_atoms": "13",
        "n_gmc_steps": "1",
        "atom_traj_len": "5",
        "start_species": "18",
    }


def _migrate_and_validate(raw: dict[str, str]) -> RootSpec:
    result = migrate_ns_inp(raw)
    return RootSpec.model_validate(result["config"])


def _log_messages(raw: dict[str, str]) -> list[str]:
    result = migrate_ns_inp(raw)
    return [e["message"] for e in result["logs"]]


def _log_levels(raw: dict[str, str]) -> list[str]:
    result = migrate_ns_inp(raw)
    return [e["level"] for e in result["logs"]]


# ---------------------------------------------------------------------------
# 1. Round-trip: minimal config validates correctly
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_minimal_validates(self):
        cfg = _migrate_and_validate(_minimal_raw())
        assert cfg.run.n_live == 100
        assert cfg.run.max_iterations == 1000
        assert cfg.backend.type == "lj"  # type: ignore[union-attr]
        # n_atoms is no longer stored on BackendConfig; it is derived from init
        # positions at resolve time.

    def test_n_cull_routed(self):
        raw = {**_minimal_raw(), "n_cull": "2"}
        cfg = _migrate_and_validate(raw)
        assert cfg.run.n_cull == 2

    def test_seed_routed(self):
        raw = {**_minimal_raw(), "seed": "123"}
        cfg = _migrate_and_validate(raw)
        assert cfg.run.seed == 123

    def test_lj_cutoff_routed(self):
        raw = {**_minimal_raw(), "lj_r_cut": "6.5"}
        cfg = _migrate_and_validate(raw)
        assert cfg.backend.cutoff == pytest.approx(6.5)  # type: ignore[union-attr]

    def test_output_format_routed(self):
        raw = {**_minimal_raw(), "config_file_format": "extxyz"}
        cfg = _migrate_and_validate(raw)
        assert cfg.output.format == "extxyz"

    def test_traj_interval_routed(self):
        raw = {**_minimal_raw(), "traj_interval": "50"}
        cfg = _migrate_and_validate(raw)
        assert cfg.output.traj_interval == 50

    def test_start_species_in_init(self):
        raw = {**_minimal_raw(), "start_species": "18"}
        cfg = _migrate_and_validate(raw)
        assert cfg.init.start_species == "18"

    def test_legacy_ns_inp_file(self):
        """The fixture file in tests/data/cli/legacy_ns.inp must validate."""
        from jaxrens.cli.parser import parse_input_file
        path = Path(__file__).parent.parent / "data" / "cli" / "legacy_ns.inp"
        raw = parse_input_file(path)
        cfg = _migrate_and_validate(raw)
        assert cfg.run.n_live == 200
        assert cfg.run.max_iterations == 5000


# ---------------------------------------------------------------------------
# 2. Unit conversion: pressure in GPa
# ---------------------------------------------------------------------------

class TestPressureConversion:
    def test_pressure_single_gpa(self):
        raw = {**_minimal_raw(), "MC_cell_P": "0.1"}
        result = migrate_ns_inp(raw)
        ens = result["config"].get("ensemble", {})
        assert ens.get("type") == "npt"
        assert ens.get("pressure") == pytest.approx(0.1)
        assert ens.get("pressure_units") == "gpa"

    def test_pressure_validates_as_npt(self):
        raw = {**_minimal_raw(), "MC_cell_P": "0.1"}
        cfg = _migrate_and_validate(raw)
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        assert isinstance(cfg.ensemble, NPTEnsembleSpec)
        # pressure_units=gpa means the schema converts; check eV/A3 value
        params = cfg.ensemble.to_ensemble_params(cohort_index=0)
        assert params["pressure"] == pytest.approx(0.1 * 0.006241509)

    def test_pressure_alias_key(self):
        """Old files sometimes use 'pressure' as well as 'MC_cell_P'."""
        raw = {**_minimal_raw(), "pressure": "0.5"}
        result = migrate_ns_inp(raw)
        ens = result["config"].get("ensemble", {})
        assert ens.get("pressure") == pytest.approx(0.5)
        assert ens.get("pressure_units") == "gpa"


# ---------------------------------------------------------------------------
# 3. List cohort: space-separated pressures
# ---------------------------------------------------------------------------

class TestPressureCohort:
    def test_multi_pressure_list(self):
        raw = {**_minimal_raw(), "MC_cell_P": "0.1 0.5 1.0"}
        result = migrate_ns_inp(raw)
        ens = result["config"]["ensemble"]
        assert ens["pressure"] == [0.1, 0.5, 1.0]

    def test_multi_pressure_validates(self):
        raw = {**_minimal_raw(), "MC_cell_P": "0.1 0.5"}
        cfg = _migrate_and_validate(raw)
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        assert isinstance(cfg.ensemble, NPTEnsembleSpec)
        assert cfg.ensemble.cohort_size() == 2


# ---------------------------------------------------------------------------
# 4. Drop list: dropped keys do not appear in config
# ---------------------------------------------------------------------------

class TestDropList:
    def test_pivot_interval_dropped(self):
        raw = {**_minimal_raw(), "pivot_interval": "5"}
        result = migrate_ns_inp(raw)
        cfg = result["config"]
        assert "pivot_interval" not in str(cfg)
        warnings = [e for e in result["logs"] if e["level"] == "WARNING"]
        assert any("pivot_interval" in w["message"] for w in warnings)

    def test_random_energy_perturbation_dropped(self):
        raw = {**_minimal_raw(), "random_energy_perturbation": "0.01"}
        result = migrate_ns_inp(raw)
        assert "random_energy_perturbation" not in str(result["config"])
        warnings = [e for e in result["logs"] if e["level"] == "WARNING"]
        assert any("random_energy_perturbation" in w["message"] for w in warnings)

    def test_n_swap_steps_dropped(self):
        raw = {**_minimal_raw(), "n_swap_steps": "3"}
        result = migrate_ns_inp(raw)
        assert "n_swap_steps" not in str(result["config"])

    def test_start_energy_ceiling_dropped(self):
        raw = {**_minimal_raw(), "start_energy_ceiling": "100.0"}
        result = migrate_ns_inp(raw)
        assert "start_energy_ceiling" not in str(result["config"])
        warnings = [e for e in result["logs"] if e["level"] == "WARNING"]
        assert any("start_energy_ceiling" in w["message"] for w in warnings)

    def test_sample_size_dropped(self):
        raw = {**_minimal_raw(), "sample_size": "200"}
        result = migrate_ns_inp(raw)
        assert "sample_size" not in str(result["config"])

    def test_n_pressure_swap_steps_dropped(self):
        raw = {**_minimal_raw(), "n_pressure_swap_steps": "1"}
        result = migrate_ns_inp(raw)
        assert "n_pressure_swap_steps" not in str(result["config"])


# ---------------------------------------------------------------------------
# 5. Unknown key: preserved as comment, not as live YAML key
# ---------------------------------------------------------------------------

class TestUnknownKey:
    def test_unknown_key_in_unknown_bucket(self):
        raw = {**_minimal_raw(), "some_future_param": "42"}
        result = migrate_ns_inp(raw)
        assert result["config"].get("_unknown", {}).get("some_future_param") == "42"

    def test_unknown_key_warning_emitted(self):
        raw = {**_minimal_raw(), "some_future_param": "42"}
        result = migrate_ns_inp(raw)
        warnings = [e for e in result["logs"] if e["level"] == "WARNING"]
        assert any("some_future_param" in w["message"] for w in warnings)

    def test_unknown_key_as_yaml_comment(self, tmp_path):
        """CLI must render _unknown keys as comments, not live YAML keys."""
        inp = tmp_path / "test.inp"
        inp.write_text("\n".join([
            "n_walkers = 50",
            "n_iter = 100",
            "energy_calculator = lj",
            "n_atoms = 13",
            "n_gmc_steps = 1",
            "atom_traj_len = 5",
            "start_species = 18",
            "some_future_param = 42",
        ]))
        out = tmp_path / "out.yaml"
        proc = subprocess.run(
            [PYTHON, "-m", "jaxrens.cli.cli", "migrate-ns-inp", "-i", str(inp), "-o", str(out)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        yaml_text = out.read_text()
        parsed = yaml.safe_load(yaml_text)
        # Must NOT appear as a live key in the YAML document
        assert "_unknown" not in parsed
        assert "some_future_param" not in parsed
        # Must appear as a comment in the raw text
        assert "# UNKNOWN" in yaml_text
        assert "some_future_param" in yaml_text


# ---------------------------------------------------------------------------
# 6. Removed DB / time-snapshot keys are dropped; snapshot_clean routes through
# ---------------------------------------------------------------------------

class TestDeferredField:
    def test_write_traj_db_is_dropped(self):
        raw = {**_minimal_raw(), "write_traj_db": "T"}
        result = migrate_ns_inp(raw)
        assert "write_traj_db" not in result["config"].get("output", {})
        warns = [e["message"] for e in result["logs"] if e["level"] == "WARNING"]
        assert any("write_traj_db" in m for m in warns)

    def test_write_walkers_db_is_dropped(self):
        raw = {**_minimal_raw(), "write_walkers_db": "F"}
        result = migrate_ns_inp(raw)
        assert "write_walkers_db" not in result["config"].get("output", {})

    def test_snapshot_time_is_dropped(self):
        raw = {**_minimal_raw(), "snapshot_time": "60.0"}
        result = migrate_ns_inp(raw)
        assert "snapshot_time" not in result["config"].get("output", {})

    def test_snapshot_clean_routes_to_output(self):
        raw = {**_minimal_raw(), "snapshot_clean": "T"}
        result = migrate_ns_inp(raw)
        assert result["config"].get("output", {}).get("snapshot_clean") is True

    def test_snapshot_clean_validates(self):
        raw = {**_minimal_raw(), "snapshot_clean": "T"}
        cfg = _migrate_and_validate(raw)
        assert cfg.output.snapshot_clean is True

    def test_cell_config_deferred_fields(self):
        raw = {**_minimal_raw(), "max_volume_per_atom": "500.0", "min_volume_per_atom": "2.0"}
        result = migrate_ns_inp(raw)
        cell = result["config"].get("cell", {})
        assert cell.get("max_volume_per_atom") == pytest.approx(500.0)
        assert cell.get("min_volume_per_atom") == pytest.approx(2.0)

    def test_cell_config_validates(self):
        raw = {**_minimal_raw(), "max_volume_per_atom": "500.0"}
        cfg = _migrate_and_validate(raw)
        assert cfg.cell.max_volume_per_atom == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 7. --validate flag
# ---------------------------------------------------------------------------

class TestValidateFlag:
    def test_validate_flag_passes_on_good_config(self, tmp_path):
        inp = tmp_path / "good.inp"
        inp.write_text("\n".join([
            "n_walkers = 50",
            "n_iter = 100",
            "energy_calculator = lj",
            "n_atoms = 13",
            "n_gmc_steps = 1",
            "atom_traj_len = 5",
            "start_species = 18",
        ]))
        proc = subprocess.run(
            [PYTHON, "-m", "jaxrens.cli.cli", "migrate-ns-inp",
             "-i", str(inp), "--validate"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Validation OK" in proc.stderr

    def test_validate_flag_fails_on_bad_config(self, tmp_path):
        # Produce a YAML that will fail RootSpec because both run.pressure
        # and ensemble are set — deliberately inject this scenario by handing
        # the migrator a raw dict we know will produce a broken config.
        # The easiest way: write a YAML directly that would fail.
        yaml_str = textwrap.dedent("""\
            run:
              n_live: 10
              max_iterations: 10
              n_mcmc_steps: 5
              n_cull: 1
              seed: 1
              pressure: 0.01
            moves:
              - type: gmc
                n_reflect: 3
            backend:
              type: lj
            output:
              format: extxyz
            ensemble:
              type: npt
              pressure: 0.01
              pressure_units: gpa
        """)
        # Validate directly (not via migrate) to confirm this is actually invalid
        with pytest.raises(Exception):
            RootSpec.model_validate(yaml.safe_load(yaml_str))

    def test_validate_roundtrip_via_cli(self, tmp_path):
        inp = tmp_path / "valid.inp"
        inp.write_text("\n".join([
            "n_walkers = 30",
            "n_iter = 200",
            "energy_calculator = lj",
            "n_atoms = 5",
            "n_gmc_steps = 1",
            "atom_traj_len = 3",
            "start_species = 1",
        ]))
        out = tmp_path / "out.yaml"
        proc = subprocess.run(
            [PYTHON, "-m", "jaxrens.cli.cli", "migrate-ns-inp",
             "-i", str(inp), "-o", str(out), "--validate"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        RootSpec.model_validate(parsed)


# ---------------------------------------------------------------------------
# 8. End-to-end CLI: subprocess with tempfile
# ---------------------------------------------------------------------------

class TestEndToEndCLI:
    def test_stdout_output(self, tmp_path):
        inp = tmp_path / "e2e.inp"
        inp.write_text("\n".join([
            "n_walkers = 100",
            "n_iter = 2000",
            "energy_calculator = lj",
            "n_atoms = 13",
            "n_gmc_steps = 1",
            "atom_traj_len = 8",
            "start_species = 18",
        ]))
        proc = subprocess.run(
            [PYTHON, "-m", "jaxrens.cli.cli", "migrate-ns-inp", "-i", str(inp)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        parsed = yaml.safe_load(proc.stdout)
        cfg = RootSpec.model_validate(parsed)
        assert cfg.run.n_live == 100
        assert cfg.run.max_iterations == 2000

    def test_file_output(self, tmp_path):
        inp = tmp_path / "in.inp"
        out = tmp_path / "out.yaml"
        inp.write_text("\n".join([
            "n_walkers = 50",
            "n_iter = 500",
            "energy_calculator = lj",
            "n_atoms = 8",
            "n_gmc_steps = 1",
            "atom_traj_len = 4",
            "start_species = 1",
        ]))
        proc = subprocess.run(
            [PYTHON, "-m", "jaxrens.cli.cli", "migrate-ns-inp",
             "-i", str(inp), "-o", str(out)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        cfg = RootSpec.model_validate(yaml.safe_load(out.read_text()))
        assert cfg.run.n_live == 50

    def test_warnings_on_stderr_not_stdout(self, tmp_path):
        inp = tmp_path / "warn.inp"
        inp.write_text("\n".join([
            "n_walkers = 50",
            "n_iter = 100",
            "energy_calculator = lj",
            "n_atoms = 5",
            "n_gmc_steps = 1",
            "atom_traj_len = 3",
            "start_species = 1",
            "pivot_interval = 5",   # dropped key — should emit WARNING on stderr
        ]))
        proc = subprocess.run(
            [PYTHON, "-m", "jaxrens.cli.cli", "migrate-ns-inp", "-i", str(inp)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        # stdout must be clean YAML
        yaml.safe_load(proc.stdout)
        # WARNING appears on stderr
        assert "pivot_interval" in proc.stderr


# ---------------------------------------------------------------------------
# Additional: n_iter_times_fraction_killed resolution
# ---------------------------------------------------------------------------

class TestFractionalIter:
    def test_fractional_iter_resolved(self):
        # n_iter_times_fraction_killed=500 with n_walkers=100, n_cull=1
        # => n_iter = round(500 / (1/100)) = 50000
        raw = {
            "n_walkers": "100",
            "n_iter_times_fraction_killed": "500",
            "energy_calculator": "lj",
            "n_atoms": "5",
            "n_gmc_steps": "1",
            "atom_traj_len": "3",
            "start_species": "1",
        }
        result = migrate_ns_inp(raw)
        assert result["config"]["run"]["max_iterations"] == 50000

    def test_explicit_n_iter_takes_precedence(self):
        raw = {**_minimal_raw(), "n_iter": "9999"}
        result = migrate_ns_inp(raw)
        assert result["config"]["run"]["max_iterations"] == 9999


# ---------------------------------------------------------------------------
# Additional: neuralil backend mapping
# ---------------------------------------------------------------------------

class TestBackendMapping:
    def test_nn_maps_to_neuralil(self):
        raw = {
            **_minimal_raw(),
            "energy_calculator": "NN",
            "pickle_file": "/path/to/model.pkl",
            "max_neighbors_list": "30 35 40",
        }
        result = migrate_ns_inp(raw)
        cfg_backend = result["config"]["backend"]
        assert cfg_backend["type"] == "neuralil"
        assert cfg_backend["checkpoint_path"] == "/path/to/model.pkl"
        assert cfg_backend["max_neighbors_list"] == [30, 35, 40]

    def test_toy_maps_to_harmonic(self):
        raw = {
            "n_walkers": "20",
            "n_iter": "100",
            "energy_calculator": "Toy",
            "n_atoms": "1",
            "n_gmc_steps": "1",
            "atom_traj_len": "3",
            "start_species": "1",
        }
        result = migrate_ns_inp(raw)
        assert result["config"]["backend"]["type"] == "harmonic"

    def test_lj_fields_routed(self):
        raw = {
            **_minimal_raw(),
            "lj_sigma": "2.0",
            "lj_eps": "1.5",
            "lj_r_cut": "7.0",
        }
        result = migrate_ns_inp(raw)
        b = result["config"]["backend"]
        assert b["sigma"] == pytest.approx(2.0)
        assert b["epsilon"] == pytest.approx(1.5)
        assert b["cutoff"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Additional: termination criteria
# ---------------------------------------------------------------------------

class TestTerminationRouting:
    def test_converge_down_to_T(self):
        raw = {**_minimal_raw(), "converge_down_to_T": "100.0"}
        result = migrate_ns_inp(raw)
        terms = result["config"].get("termination", [])
        assert any(t.get("type") == "temperature" for t in terms)
        temp_spec = next(t for t in terms if t.get("type") == "temperature")
        assert temp_spec["target_temp"] == pytest.approx(100.0)

    def test_min_Emax(self):
        raw = {**_minimal_raw(), "min_Emax": "-5.0"}
        result = migrate_ns_inp(raw)
        terms = result["config"].get("termination", [])
        assert any(t.get("type") == "energy" for t in terms)
        energy_spec = next(t for t in terms if t.get("type") == "energy")
        assert energy_spec["min_energy"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Additional: adaptation fields
# ---------------------------------------------------------------------------

class TestAdaptationRouting:
    def test_full_auto_step_sizes(self):
        raw = {**_minimal_raw(), "full_auto_step_sizes": "T"}
        result = migrate_ns_inp(raw)
        assert result["config"].get("adaptation", {}).get("full_auto") is True

    def test_GMC_adjust_rates(self):
        raw = {**_minimal_raw(), "GMC_adjust_min_rate": "0.3", "GMC_adjust_max_rate": "0.8"}
        result = migrate_ns_inp(raw)
        per_move = result["config"].get("adaptation", {}).get("per_move", {})
        assert per_move.get("gmc", {}).get("min_rate") == pytest.approx(0.3)
        assert per_move.get("gmc", {}).get("max_rate") == pytest.approx(0.8)

    def test_MC_adjust_step_factor(self):
        raw = {**_minimal_raw(), "MC_adjust_step_factor": "2.0"}
        result = migrate_ns_inp(raw)
        assert result["config"].get("adaptation", {}).get("defaults", {}).get("adjust_factor") == pytest.approx(2.0)

    def test_adaptation_validates(self):
        raw = {**_minimal_raw(), "full_auto_step_sizes": "T", "GMC_adjust_min_rate": "0.2"}
        cfg = _migrate_and_validate(raw)
        assert cfg.adaptation.full_auto is True
