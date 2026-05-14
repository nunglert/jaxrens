"""Tests for ``jaxrens.cli.plot`` and the ``jaxrens plot`` subcommand."""

from __future__ import annotations

import json

import h5py
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from jaxrens.cli.plot import (
    _default_output,
    _suffix_kind,
    plot_file,
)


# ---------------------------------------------------------------------------
# Suffix dispatch
# ---------------------------------------------------------------------------


class TestSuffixKind:
    @pytest.mark.parametrize("name, expected", [
        ("foo.adaptation.h5", "adaptation"),
        ("run123.adaptation.h5", "adaptation"),
        ("ns.re_stats.h5", "re_stats"),
        ("sweep.re_stats.h5", "re_stats"),
        ("any.energies", "energies"),
        ("ns.run00.energies", "energies"),
    ])
    def test_recognises(self, tmp_path, name, expected):
        assert _suffix_kind(tmp_path / name) == expected

    def test_unknown_suffix_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unrecognised"):
            _suffix_kind(tmp_path / "foo.checkpoint.h5")


class TestDefaultOutput:
    def test_strips_full_suffix(self, tmp_path):
        # The whole ``.adaptation.h5`` suffix is stripped, not just ``.h5``.
        out = _default_output(tmp_path / "myrun.adaptation.h5", "adaptation")
        assert out.name == "myrun.adaptation.png"

    def test_re_stats(self, tmp_path):
        out = _default_output(tmp_path / "sweep.re_stats.h5", "re_stats")
        assert out.name == "sweep.re_stats.png"

    def test_energies(self, tmp_path):
        out = _default_output(tmp_path / "ns.run00.energies", "energies")
        assert out.name == "ns.run00.energies.png"


# ---------------------------------------------------------------------------
# End-to-end: synthetic fixtures + plot_file
# ---------------------------------------------------------------------------


def _write_adaptation_h5(path, n_entries=4, n_runs=2, n_moves=3,
                         move_names=("a", "b", "c")):
    with h5py.File(path, "w") as f:
        f.attrs["adaptation_log_schema_version"] = 3
        f.attrs["n_moves"] = n_moves
        f.attrs["n_runs"] = n_runs
        f.attrs["move_names"] = json.dumps(list(move_names))
        f.create_dataset("iterations", data=np.arange(n_entries, dtype=np.int64))
        f.create_dataset(
            "step_sizes",
            data=np.full((n_entries, n_runs, n_moves), 0.05, dtype=np.float32),
        )
        f.create_dataset(
            "acceptance_rates",
            data=np.full((n_entries, n_runs, n_moves), 0.4, dtype=np.float32),
        )
        f.create_dataset(
            "n_evaluations",
            data=np.full((n_entries, n_runs, n_moves), 100, dtype=np.int64),
        )
        f.create_dataset(
            "n_grad_evaluations",
            data=np.full((n_entries, n_runs, n_moves), 50, dtype=np.int64),
        )
        s = f.create_group("adjustment_stats")
        zeros = np.zeros((n_entries, n_runs, n_moves), dtype=np.int32)
        s.create_dataset("n_rounds", data=np.ones_like(zeros))
        s.create_dataset("converged", data=zeros.astype(bool))
        s.create_dataset("cap_hits", data=zeros)
        s.create_dataset("floor_hits", data=zeros)
        s.create_dataset("bracket_detected", data=zeros.astype(bool))
        s.create_dataset(
            "reject_reason_counts",
            data=np.zeros((n_entries, n_runs, n_moves, 4), dtype=np.int32),
        )


def _write_re_stats_h5(path, n_entries=3, n_pairs=2, flavor="pressure"):
    with h5py.File(path, "w") as f:
        f.attrs["n_pairs"] = n_pairs
        f.attrs["flavor"] = flavor
        f.create_dataset("iterations", data=np.arange(n_entries, dtype=np.int64))
        f.create_dataset(
            "n_accepted_per_pair",
            data=np.full((n_entries, n_pairs), 3, dtype=np.int32),
        )
        f.create_dataset(
            "n_attempted_per_pair",
            data=np.full((n_entries, n_pairs), 10, dtype=np.int32),
        )


def _write_energies(path, n_entries=20, n_atoms=4, with_volume=True):
    from jaxrens.io.energy_log import EnergyLogger

    logger = EnergyLogger(
        path=path, n_walkers=5, n_cull=1, n_dof=0, n_atoms=n_atoms,
    )
    logger.write_header()
    rng = np.random.default_rng(0)
    for i in range(n_entries):
        v = float(20.0 + rng.normal()) if with_volume else 0.0
        logger.write_entry(i, energy=-float(i) * 0.5, volume=v)
    logger.close()


class TestPlotFileEndToEnd:
    def test_adaptation_round_trip(self, tmp_path):
        in_path = tmp_path / "myrun.adaptation.h5"
        _write_adaptation_h5(in_path)
        out = plot_file(in_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_re_stats_round_trip(self, tmp_path):
        in_path = tmp_path / "sweep.re_stats.h5"
        _write_re_stats_h5(in_path)
        out = plot_file(in_path)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_energies_round_trip_npt(self, tmp_path):
        in_path = tmp_path / "ns.run00.energies"
        _write_energies(in_path, with_volume=True)
        out = plot_file(in_path)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_energies_round_trip_nvt(self, tmp_path):
        """Volumes all-zero → only the energy panel is drawn."""
        in_path = tmp_path / "ns.run00.energies"
        _write_energies(in_path, with_volume=False)
        out = plot_file(in_path)
        assert out.exists()

    def test_explicit_output_path_honoured(self, tmp_path):
        in_path = tmp_path / "myrun.adaptation.h5"
        out_path = tmp_path / "elsewhere" / "custom.png"
        _write_adaptation_h5(in_path)
        out = plot_file(in_path, out_path)
        assert out == out_path
        assert out.exists()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            plot_file(tmp_path / "does_not_exist.adaptation.h5")
