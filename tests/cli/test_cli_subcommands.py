"""Tests for the ``jaxrens`` subcommands other than ``run``.

They cover ``--version``, ``dump-schema``, the ``plot`` error paths, the
three ``validate`` tiers and the error messages ``main`` prints for bad
config files. No test runs NS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from jaxrens import __version__
from jaxrens.cli import resolve as _r
from jaxrens.cli.cli import main

_DATA = Path(__file__).parent.parent / "_assets" / "data" / "cli"


class TestVersion:
    """Check the ``--version`` report."""

    @pytest.mark.parametrize("flag", ["--version", "-V"])
    def test_prints_version_and_exits_zero(self, flag, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([flag])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "jaxrens" in out
        assert __version__ in out
        # Runtime deps are reported too (read from metadata, JAX not imported).
        assert "jax" in out
        assert "python" in out

    def test_version_output_is_plain_when_not_a_tty(self, capsys):
        """capsys is not a TTY, so no ANSI escape codes should leak."""
        with pytest.raises(SystemExit):
            main(["--version"])
        assert "\x1b[" not in capsys.readouterr().out


class TestDumpSchema:
    """Check that ``dump-schema`` prints the RootSpec JSON schema."""

    def test_prints_valid_json_schema(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["dump-schema"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        schema = json.loads(out)
        assert "properties" in schema
        assert "run" in schema["properties"]
        assert "moves" in schema["properties"]
        assert "backend" in schema["properties"]

    def test_explicit_format_json(self, capsys):
        """``--format json`` is the only currently-accepted format."""
        with pytest.raises(SystemExit) as exc_info:
            main(["dump-schema", "--format", "json"])
        assert exc_info.value.code == 0
        json.loads(capsys.readouterr().out)


class TestPlotErrors:
    """Check that ``plot`` turns input errors into exit code 2."""

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


class TestValidationErrorFormatting:
    """Check the "did you mean" hint for unknown config keys."""

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
        """A root key that is a valid field deeper down is pointed there."""
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


class TestMainExceptionHandlers:
    """Check that ``main`` reports file, YAML and ``--set`` errors with exit
    code 2 instead of a traceback.
    """

    def test_missing_config_file_returns_2(self, tmp_path, capsys):
        nonexistent = tmp_path / "no_such_file.yaml"
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(nonexistent)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "jaxrens:" in err

    @pytest.mark.parametrize(
        "spec, message",
        [
            ("run.n_live", "expects KEY=VALUE"),
            ("moves[5].step_size=1", "key path does not exist"),
            ("run.n_live.x=1", "key path does not exist"),
        ],
        ids=["no-equals", "missing-index", "below-scalar"],
    )
    def test_bad_set_override_returns_2(self, capsys, spec, message):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "validate",
                    "-c",
                    str(_DATA / "minimal.yaml"),
                    "--set",
                    spec,
                    "--parse-only",
                ]
            )
        assert exc_info.value.code == 2
        assert message in capsys.readouterr().err

    def test_malformed_yaml_returns_2(self, tmp_path, capsys):
        bad_yaml = tmp_path / "broken.yaml"
        bad_yaml.write_text("run:\n  n_live: [unbalanced\n")
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(bad_yaml)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "YAML parse error" in err


class TestValidateParseOnly:
    """Check that ``validate --parse-only`` reports the parsed config."""

    def test_parse_only_skips_resolver(self, capsys):
        """``--parse-only`` returns 0 and says it only parsed the config."""
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


def test_validate_single_run_topology(capsys, monkeypatch):
    monkeypatch.setattr(_r, "_local_device_count", lambda: 1)
    with pytest.raises(SystemExit) as exc_info:
        main(["validate", "-c", str(_DATA / "minimal.yaml")])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "SingleRun" in out


def test_validate_multi_replica_topology(capsys, monkeypatch):
    """Two pressures on one device give n_gpu=1 x n_per_gpu=2."""
    monkeypatch.setattr(_r, "_local_device_count", lambda: 1)
    with pytest.raises(SystemExit) as exc_info:
        main(["validate", "-c", str(_DATA / "npt_sweep.yaml")])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "n_gpu=" in out
    assert "2 replica" in out


class TestValidateTiers:
    """Check that default ``validate`` stops at the plan and ``--full``
    resolves everything.

    The default tier must not build a backend or place walkers: the full
    resolve takes seconds on a toy backend and minutes on an MLIP, nearly
    all of it spent on an energy model that validation never calls.
    """

    @pytest.fixture(autouse=True)
    def _single_device(self, monkeypatch):
        monkeypatch.setattr(_r, "_local_device_count", lambda: 1)

    def test_default_tier_does_not_materialise(self, capsys, monkeypatch):
        def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError(
                "default `validate` resolved the full config; it must stop "
                "at the plan phase"
            )

        monkeypatch.setattr(_r, "resolve", _boom)
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "minimal.yaml")])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "plan valid" in out
        # The report must say what it did not check, so a passing run is
        # never mistaken for a full startup rehearsal.
        assert "skipped" in out
        assert "--full" in out

    def test_full_tier_materialises(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "minimal.yaml"), "--full"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "configuration valid" in out
        assert "plan valid" not in out

    def test_plan_reports_n_atoms_from_species(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "minimal.yaml")])
        assert exc_info.value.code == 0
        # n_atoms comes from the species string, not from placed walkers.
        assert "n_atoms=" in capsys.readouterr().out

    def test_plan_rejects_missing_path(self, tmp_path, capsys):
        raw = yaml.safe_load((_DATA / "minimal.yaml").read_text())
        raw["init"] = {"start_config_file": str(tmp_path / "nope.xyz")}
        cfg = tmp_path / "missing_path.yaml"
        cfg.write_text(yaml.safe_dump(raw))
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(cfg)])
        assert exc_info.value.code == 2
        assert "does not exist" in capsys.readouterr().err
