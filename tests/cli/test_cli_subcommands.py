"""Coverage-focused tests for ``jaxrens.cli.cli`` subcommand handlers
and main() exception plumbing.

Pure argparse / formatting tests — no JAX execution.  Complements
``test_schema.py`` (which exercises ``main(["validate", ...])`` on the
happy path) and ``test_cli.py`` (which exercises ``run_from_config``
directly).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxrens.cli.cli import main

_DATA = Path(__file__).parent.parent / "data" / "cli"


# ---------------------------------------------------------------------------
# dump-schema
# ---------------------------------------------------------------------------


class TestDumpSchema:
    def test_prints_valid_json_schema(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["dump-schema"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        schema = json.loads(out)
        # RootSpec is a pydantic model; its JSON schema always has these.
        assert "properties" in schema
        assert "run" in schema["properties"]
        assert "moves" in schema["properties"]
        assert "backend" in schema["properties"]

    def test_explicit_format_json(self, capsys):
        """``--format json`` is the only currently-accepted format."""
        with pytest.raises(SystemExit) as exc_info:
            main(["dump-schema", "--format", "json"])
        assert exc_info.value.code == 0
        json.loads(capsys.readouterr().out)  # parses


# ---------------------------------------------------------------------------
# annotate-steinhardt
# ---------------------------------------------------------------------------


class TestAnnotateSteinhardt:
    def test_annotates_extxyz(self, tmp_path, capsys):
        from ase.io import read

        src = Path(__file__).parent.parent / "data" / "postprocess"
        frames = read(str(src / "steinhardt" / "fcc_Cu.xyz"), index=":")
        traj = tmp_path / "run.traj.extxyz"
        from ase.io import write

        write(str(traj), frames)

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "annotate-steinhardt",
                    "--traj",
                    str(traj),
                    "--l",
                    "4",
                    "6",
                    "--r-cut",
                    "3.0",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Steinhardt annotation" in out

        annotated = read(
            str(tmp_path / "run.traj.annotated.extxyz"), index=":"
        )
        for atoms in annotated:
            assert "q6" in atoms.arrays and "w4" in atoms.arrays

    def test_non_extxyz_returns_2(self, tmp_path, capsys):
        bogus = tmp_path / "run.traj.h5"
        bogus.write_bytes(b"nope")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "annotate-steinhardt",
                    "--traj",
                    str(bogus),
                    "--l",
                    "6",
                    "--r-cut",
                    "3.0",
                ]
            )
        assert exc_info.value.code == 2
        assert "jaxrens annotate-steinhardt:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# plot (error paths)
# ---------------------------------------------------------------------------


class TestPlotErrors:
    def test_missing_file_returns_2(self, tmp_path, capsys):
        nonexistent = tmp_path / "nope.adaptation.h5"
        with pytest.raises(SystemExit) as exc_info:
            main(["plot", str(nonexistent)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "jaxrens plot:" in err

    def test_unknown_suffix_returns_2(self, tmp_path, capsys):
        bogus = tmp_path / "data.weirdext"
        bogus.write_text("garbage")
        with pytest.raises(SystemExit) as exc_info:
            main(["plot", str(bogus)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "jaxrens plot:" in err


# ---------------------------------------------------------------------------
# Validation-error formatting + did-you-mean
# ---------------------------------------------------------------------------


class TestValidationErrorFormatting:
    def test_extra_forbidden_emits_did_you_mean(self, tmp_path, capsys):
        """Typo in a known field name triggers the fuzzy-match hint."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            "run:\n"
            "  n_lvie: 20\n"  # typo of n_live
            "  max_iterations: 10\n"
            "  n_mcmc_steps: 2\n"
            "  seed: 0\n"
            "moves:\n"
            "  move_type: random_walk\n"
            "  step_size: 0.3\n"
            "backend:\n"
            "  backend_type: harmonic\n"
            "output:\n"
            "  format: none\n"
            "  working_dir: '.'\n"
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(bad_yaml)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "invalid configuration" in err
        assert "did you mean" in err
        assert "n_live" in err

    def test_extra_forbidden_at_root_exact_leaf(self, tmp_path, capsys):
        """An extra top-level field that exists deeper in the schema
        triggers the exact-leaf-match branch."""
        bad_yaml = tmp_path / "bad.yaml"
        # ``step_size`` is a valid leaf under ``moves`` but not at root.
        bad_yaml.write_text(
            "step_size: 0.5\n"
            "run:\n"
            "  n_live: 20\n"
            "  max_iterations: 10\n"
            "  n_mcmc_steps: 2\n"
            "  seed: 0\n"
            "moves:\n"
            "  move_type: random_walk\n"
            "  step_size: 0.3\n"
            "backend:\n"
            "  backend_type: harmonic\n"
            "output:\n"
            "  format: none\n"
            "  working_dir: '.'\n"
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(bad_yaml)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "did you mean" in err
        assert "step_size" in err


# ---------------------------------------------------------------------------
# main() exception handlers
# ---------------------------------------------------------------------------


class TestMainExceptionHandlers:
    def test_missing_config_file_returns_2(self, tmp_path, capsys):
        nonexistent = tmp_path / "no_such_file.yaml"
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(nonexistent)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "jaxrens:" in err

    def test_malformed_yaml_returns_2(self, tmp_path, capsys):
        bad_yaml = tmp_path / "broken.yaml"
        bad_yaml.write_text("run:\n  n_live: [unbalanced\n")
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(bad_yaml)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "YAML parse error" in err


# ---------------------------------------------------------------------------
# validate --parse-only
# ---------------------------------------------------------------------------


class TestValidateParseOnly:
    def test_parse_only_skips_resolver(self, capsys):
        """``--parse-only`` returns 0 and prints the parse-only marker
        without touching the resolver."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "validate",
                    "-c",
                    str(_DATA / "minimal.yaml"),
                    "--parse-only",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "parse-only" in out
        assert "n_live=20" in out

    def test_parse_only_set_override(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "validate",
                    "-c",
                    str(_DATA / "minimal.yaml"),
                    "--set",
                    "run.n_live=128",
                    "--parse-only",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "n_live=128" in out


# ---------------------------------------------------------------------------
# validate topology lines (the 3 batcher branches of _cmd_validate)
# ---------------------------------------------------------------------------


class TestValidateTopologyLines:
    @pytest.fixture(autouse=True)
    def _single_device(self, monkeypatch):
        """Force ``_local_device_count`` to 1 so SingleRun fixtures
        actually resolve to SingleRun regardless of the host's GPU count."""
        from jaxrens.cli import resolve as _r

        monkeypatch.setattr(_r, "_local_device_count", lambda: 1)

    def test_single_run_topology(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "minimal.yaml")])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "SingleRun" in out

    # NOTE: PmapVmapRuns topology line is exercised by
    # ``tests/cli/test_schema.py::TestValidateTopologyLine::
    # test_validate_reports_multi_replica_topology`` which monkeypatches
    # the device count and hits the resolver's multi-replica branch
    # through ``main(["validate", ...])`` end-to-end.
