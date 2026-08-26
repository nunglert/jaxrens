"""Tests for ``jaxrens.cli.analyze`` and the ``jaxrens analyze`` subcommand."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, must be before any other matplotlib import
import numpy as np
import pytest

from jaxrens.cli.analyze import (
    _default_output,
    _prefix_from_checkpoint,
    analyze_file,
)
from jaxrens.cli.cli import main

_LJ8_NPT_OUTPUT = (
    Path(__file__).parent.parent / "_assets" / "data" / "lj8_npt" / "output"
)


# ---------------------------------------------------------------------------
# Checkpoint-filename parsing
# ---------------------------------------------------------------------------


class TestPrefixFromCheckpoint:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("ns.checkpoint.h5", "ns"),
            ("ns.final.checkpoint.h5", "ns"),
            ("lj8.final.checkpoint.h5", "lj8"),
            ("run00.checkpoint.h5", "run00"),
        ],
    )
    def test_recognises(self, tmp_path, name, expected):
        assert _prefix_from_checkpoint(tmp_path / name) == expected

    def test_unrecognised_suffix_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unrecognised checkpoint"):
            _prefix_from_checkpoint(tmp_path / "ns.energies")


class TestDefaultOutput:
    @pytest.mark.parametrize("suffix", ["csv", "json", "png"])
    def test_names_by_prefix_observable_and_suffix(self, tmp_path, suffix):
        out = _default_output(
            tmp_path / "lj8.final.checkpoint.h5", "lj8", "heat_capacity", suffix
        )
        assert out.name == f"lj8.heat_capacity.{suffix}"
        assert out.parent == tmp_path


class TestWriteCsvGuard:
    """``_write_csv`` refuses a non-scalar observable rather than mangling it
    into a shape CSV cannot represent -- ``--format json`` is the escape
    hatch, exercised in ``TestAnalyzeFileEndToEnd`` below.
    """

    def test_non_scalar_values_raise(self, tmp_path):
        from jaxrens.cli.analyze import _write_csv

        T = np.linspace(0.0, 1.0, 4)
        values = np.zeros((4, 2))  # two components per T -- not scalar
        with pytest.raises(ValueError, match="not one scalar per"):
            _write_csv(tmp_path / "out.csv", T, "Cv", values)


# ---------------------------------------------------------------------------
# End-to-end: synthetic run directory + analyze_file
# ---------------------------------------------------------------------------


def _write_run_dir(tmp_path, prefix="ns", n_live=10, n_dead=20, seed=0):
    """Synthesise a minimal run directory analyze_file can load.

    Mirrors the pattern in ``tests/postprocess/test_monitor.py``'s
    ``TestMonitorFromDirectory``: a checkpoint via ``save_checkpoint`` plus
    a matching ``.energies`` log via ``EnergyLogger``.
    """
    import jax.numpy as jnp

    from jaxrens.io.checkpoint import save_checkpoint
    from jaxrens.io.energy_log import EnergyLogger

    rng = np.random.default_rng(seed)
    dead_e = np.sort(rng.uniform(0.0, 5.0, n_dead))
    live_e = rng.uniform(4.0, 6.0, n_live)
    positions = rng.uniform(0.0, 1.0, (n_live, 4, 3))
    dead_positions = rng.uniform(0.0, 1.0, (n_dead, 4, 3))

    ns_state = {
        "positions": jnp.asarray(positions),
        "types": jnp.zeros(n_live, dtype=int),
        "energies": jnp.asarray(live_e),
        "cells": None,
        "dead_energies": jnp.asarray(
            np.concatenate([dead_e, np.full(1000, np.inf)])
        ),
        "dead_positions": jnp.asarray(
            np.concatenate([dead_positions, np.zeros((1000, 4, 3))], axis=0)
        ),
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": 12.5,
        "iteration": n_dead,
        "emax": float(live_e.max()),
        "n_dead": n_dead,
        "n_walkers": n_live,
    }
    ckpt_path = tmp_path / f"{prefix}.final.checkpoint.h5"
    save_checkpoint(ckpt_path, ns_state, symbol_map={0: "Ar"})

    energy_logger = EnergyLogger(
        tmp_path / f"{prefix}.energies", n_walkers=n_live, n_cull=1
    )
    energy_logger.write_header()
    for i, e in enumerate(dead_e):
        energy_logger.write_entry(i, float(e))
    energy_logger.close()

    return ckpt_path


class TestAnalyzeFileEndToEnd:
    @pytest.mark.parametrize(
        "observable, column",
        [
            ("heat_capacity", "Cv"),
            ("partition_function", "log_Z"),
            ("free_energy", "F"),
        ],
    )
    def test_each_observable_writes_csv(self, tmp_path, observable, column):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        csv_path, png_path = analyze_file(
            ckpt_path, observable=observable, t_min=0.5, t_max=5.0, n_t=10
        )
        assert png_path is None
        assert csv_path.exists()
        assert csv_path.name == f"ns.{observable}.csv"

        with csv_path.open() as fh:
            rows = [[c.strip() for c in row] for row in csv.reader(fh)]
        assert rows[0] == ["T", column]
        assert len(rows) == 11  # header + n_t rows
        # Every data row is two finite floats, and (being fixed-width,
        # right-aligned) the same width as the header.
        header_line_len = len(f"{'T':>16},{column:>16}")
        with csv_path.open() as fh:
            lines = fh.read().splitlines()
        for row, line in zip(rows[1:], lines[1:], strict=True):
            assert len(row) == 2
            assert len(line) == header_line_len
            t, value = float(row[0]), float(row[1])
            assert np.isfinite(t)
            assert np.isfinite(value)
        # T column matches the requested linspace, to the 8-sig-fig
        # precision the writer keeps.
        t_col = [float(row[0]) for row in rows[1:]]
        np.testing.assert_allclose(
            t_col, np.linspace(0.5, 5.0, 10), rtol=1e-7
        )

    @pytest.mark.parametrize(
        "observable, column",
        [
            ("heat_capacity", "Cv"),
            ("partition_function", "log_Z"),
            ("free_energy", "F"),
        ],
    )
    def test_each_observable_writes_json(self, tmp_path, observable, column):
        import json

        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        json_path, png_path = analyze_file(
            ckpt_path,
            observable=observable,
            t_min=0.5,
            t_max=5.0,
            n_t=10,
            fmt="json",
        )
        assert png_path is None
        assert json_path.exists()
        assert json_path.name == f"ns.{observable}.json"

        payload = json.loads(json_path.read_text())
        assert payload["observable"] == observable
        assert payload["column"] == column
        assert payload["prefix"] == "ns"
        assert payload["k_b"] == 1.0
        assert len(payload["T"]) == 10
        assert len(payload[column]) == 10
        np.testing.assert_allclose(
            payload["T"], np.linspace(0.5, 5.0, 10), rtol=1e-10
        )
        assert all(np.isfinite(v) for v in payload[column])

    def test_unknown_format_raises(self, tmp_path):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(ValueError, match="Unknown format"):
            analyze_file(ckpt_path, t_min=0.5, t_max=5.0, fmt="bogus")

    def test_plot_also_writes_png(self, tmp_path):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        csv_path, png_path = analyze_file(
            ckpt_path, t_min=0.5, t_max=5.0, plot=True
        )
        assert csv_path.exists()
        assert csv_path.name == "ns.heat_capacity.csv"
        assert png_path is not None
        assert png_path.exists()
        assert png_path.name == "ns.heat_capacity.png"
        assert png_path.stat().st_size > 0

    def test_no_plot_writes_no_png(self, tmp_path):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        csv_path, png_path = analyze_file(ckpt_path, t_min=0.5, t_max=5.0)
        assert png_path is None
        assert not (tmp_path / "ns.heat_capacity.png").exists()

    def test_explicit_output_path_honoured(self, tmp_path):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        out_path = tmp_path / "elsewhere" / "custom.csv"
        csv_path, _ = analyze_file(
            ckpt_path, t_min=0.5, t_max=5.0, output_path=out_path
        )
        assert csv_path == out_path
        assert csv_path.exists()

    def test_explicit_plot_path_honoured(self, tmp_path):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        plot_path = tmp_path / "elsewhere" / "custom.png"
        _, png_path = analyze_file(
            ckpt_path, t_min=0.5, t_max=5.0, plot=True, plot_path=plot_path
        )
        assert png_path == plot_path
        assert png_path.exists()

    def test_prefers_final_checkpoint(self, tmp_path):
        """Both a periodic and a final checkpoint exist; final wins."""
        import jax.numpy as jnp

        from jaxrens.io.checkpoint import save_checkpoint

        _write_run_dir(tmp_path, prefix="ns")
        # A stale periodic checkpoint with a different n_walkers -- if
        # analyze_file picked this one up instead, n_live below would be 3.
        stale_state = {
            "positions": jnp.zeros((3, 4, 3)),
            "types": jnp.zeros(3, dtype=int),
            "energies": jnp.zeros(3),
            "cells": None,
            "dead_energies": jnp.full(1000, np.inf),
            "dead_positions": jnp.zeros((1000, 4, 3)),
            "dead_volumes": None,
            "live_volumes": None,
            "log_evidence": 0.0,
            "iteration": 0,
            "emax": 0.0,
            "n_dead": 0,
            "n_walkers": 3,
        }
        save_checkpoint(tmp_path / "ns.checkpoint.h5", stale_state)

        ckpt_path = tmp_path / "ns.final.checkpoint.h5"
        csv_path, _ = analyze_file(ckpt_path, t_min=0.5, t_max=5.0)
        assert csv_path.exists()

    def test_unknown_observable_raises(self, tmp_path):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(ValueError, match="Unknown observable"):
            analyze_file(
                ckpt_path, observable="bogus", t_min=0.5, t_max=5.0
            )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_file(
                tmp_path / "does_not_exist.final.checkpoint.h5",
                t_min=0.5,
                t_max=5.0,
            )

    def test_bad_suffix_raises(self, tmp_path):
        bogus = tmp_path / "ns.energies"
        bogus.write_text("not a checkpoint")
        with pytest.raises(ValueError, match="Unrecognised checkpoint"):
            analyze_file(bogus, t_min=0.5, t_max=5.0)


# ---------------------------------------------------------------------------
# Smoke test against the committed lj8_npt example output
# ---------------------------------------------------------------------------


class TestAnalyzeSmokeRealRun:
    """Same fixture as ``TestSmokeRealRun`` in test_monitor.py."""

    def test_heat_capacity_on_real_run(self, tmp_path):
        ckpt = _LJ8_NPT_OUTPUT / "lj8_npt.final.checkpoint.h5"
        csv_path, _ = analyze_file(
            ckpt,
            t_min=0.1,
            t_max=5.0,
            output_path=tmp_path / "cv.csv",
        )
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# CLI subcommand: main(["analyze", ...])
# ---------------------------------------------------------------------------


class TestAnalyzeCLI:
    def test_writes_csv_and_prints_path(self, tmp_path, capsys):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(ckpt_path),
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        expected = tmp_path / "ns.heat_capacity.csv"
        assert str(expected) in out
        assert expected.exists()
        # No --plot: no PNG, and only one "Wrote" line.
        assert not (tmp_path / "ns.heat_capacity.png").exists()
        assert out.count("Wrote ") == 1

    def test_format_json_writes_json(self, tmp_path, capsys):
        import json

        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(ckpt_path),
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                    "--format",
                    "json",
                ]
            )
        assert exc_info.value.code == 0
        expected = tmp_path / "ns.heat_capacity.json"
        out = capsys.readouterr().out
        assert str(expected) in out
        assert expected.exists()
        payload = json.loads(expected.read_text())
        assert payload["column"] == "Cv"

    def test_unknown_format_is_argparse_error(self, tmp_path, capsys):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(ckpt_path),
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                    "--format",
                    "bogus",
                ]
            )
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--format" in err

    def test_plot_flag_writes_png_too(self, tmp_path, capsys):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(ckpt_path),
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                    "--plot",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        csv_expected = tmp_path / "ns.heat_capacity.csv"
        png_expected = tmp_path / "ns.heat_capacity.png"
        assert str(csv_expected) in out
        assert str(png_expected) in out
        assert csv_expected.exists()
        assert png_expected.exists()
        assert out.count("Wrote ") == 2

    def test_plot_output_flag_honoured(self, tmp_path, capsys):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        plot_out = tmp_path / "custom_plot.png"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(ckpt_path),
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                    "--plot",
                    "--plot-output",
                    str(plot_out),
                ]
            )
        assert exc_info.value.code == 0
        assert plot_out.exists()

    def test_missing_t_bounds_is_argparse_error(self, tmp_path, capsys):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", str(ckpt_path)])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--t-min" in err

    def test_missing_file_returns_2(self, tmp_path, capsys):
        nonexistent = tmp_path / "nope.final.checkpoint.h5"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(nonexistent),
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                ]
            )
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "jaxrens analyze:" in err

    def test_bad_suffix_returns_2(self, tmp_path, capsys):
        bogus = tmp_path / "ns.energies"
        bogus.write_text("not a checkpoint")
        with pytest.raises(SystemExit) as exc_info:
            main(
                ["analyze", str(bogus), "--t-min", "0.5", "--t-max", "5.0"]
            )
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "jaxrens analyze:" in err

    def test_unknown_observable_is_argparse_error(self, tmp_path, capsys):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(ckpt_path),
                    "--observable",
                    "bogus",
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                ]
            )
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--observable" in err

    def test_explicit_output_and_observable(self, tmp_path, capsys):
        ckpt_path = _write_run_dir(tmp_path, prefix="ns")
        out_path = tmp_path / "custom.csv"
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "analyze",
                    str(ckpt_path),
                    "--observable",
                    "free_energy",
                    "--t-min",
                    "0.5",
                    "--t-max",
                    "5.0",
                    "-o",
                    str(out_path),
                ]
            )
        assert exc_info.value.code == 0
        assert out_path.exists()
        with out_path.open() as fh:
            header = [c.strip() for c in next(csv.reader(fh))]
        assert header == ["T", "F"]
