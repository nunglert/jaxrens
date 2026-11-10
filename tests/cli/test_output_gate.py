"""Tests for ``jaxrens.cli.output_gate``.

The output-dir gate refuses a dirty ``working_dir`` and ``--force`` wipes
it. ``discover_checkpoint`` picks the checkpoint for ``--resume``, and the
snapshot helpers write, locate and read the config snapshot.
"""

from __future__ import annotations

import os
import time

import pytest

from jaxrens.cli.output_gate import (
    discover_checkpoint,
    enforce_clean_output_dir,
    read_config_snapshot,
    snapshot_filename,
    snapshot_path_for_checkpoint,
    write_config_snapshot,
)
from jaxrens.cli.schema import RootSpec

PREFIX = "ns"


def _touch(d, name):
    (d / name).write_text("stale content")


def test_gate_passes_on_empty_dir(tmp_path):
    enforce_clean_output_dir(tmp_path, PREFIX, force=False)


def test_gate_passes_when_dir_missing(tmp_path):
    enforce_clean_output_dir(tmp_path / "does_not_exist", PREFIX, force=False)


def test_gate_fails_on_energies(tmp_path):
    _touch(tmp_path, f"{PREFIX}.energies")
    with pytest.raises(SystemExit) as ex:
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)
    assert ex.value.code == 2


def test_gate_fails_on_run_suffixed_energies(tmp_path):
    _touch(tmp_path, f"{PREFIX}.run00.energies")
    _touch(tmp_path, f"{PREFIX}.run01.energies")
    with pytest.raises(SystemExit):
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)


def test_gate_fails_on_traj_extxyz(tmp_path):
    _touch(tmp_path, f"{PREFIX}.traj.extxyz")
    with pytest.raises(SystemExit):
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)


def test_gate_fails_on_snapshot(tmp_path):
    _touch(tmp_path, f"{PREFIX}.traj.snap.1250.extxyz")
    with pytest.raises(SystemExit):
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)


def test_gate_fails_on_hdf5_artifacts(tmp_path):
    _touch(tmp_path, f"{PREFIX}.adaptation.h5")
    _touch(tmp_path, f"{PREFIX}.re_stats.h5")
    _touch(tmp_path, f"{PREFIX}.max_neighbors.h5")
    _touch(tmp_path, f"{PREFIX}.acc_rates.h5")
    _touch(tmp_path, f"{PREFIX}.checkpoint.h5")
    _touch(tmp_path, f"{PREFIX}.final.checkpoint.h5")
    _touch(tmp_path, f"{PREFIX}.initial.checkpoint.h5")
    with pytest.raises(SystemExit):
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)


def test_gate_force_deletes_everything(tmp_path):
    files = [
        f"{PREFIX}.energies",
        f"{PREFIX}.run00.energies",
        f"{PREFIX}.run01.energies",
        f"{PREFIX}.traj.extxyz",
        f"{PREFIX}.run00.traj.extxyz",
        f"{PREFIX}.traj.snap.42.extxyz",
        f"{PREFIX}.adaptation.h5",
        f"{PREFIX}.re_stats.h5",
        f"{PREFIX}.max_neighbors.h5",
        f"{PREFIX}.acc_rates.h5",
        f"{PREFIX}.checkpoint.h5",
        f"{PREFIX}.final.checkpoint.h5",
        f"{PREFIX}.initial.checkpoint.h5",
    ]
    for name in files:
        _touch(tmp_path, name)

    enforce_clean_output_dir(tmp_path, PREFIX, force=True)

    for name in files:
        assert not (tmp_path / name).exists()


def test_gate_ignores_foreign_prefix(tmp_path):
    _touch(tmp_path, "other.energies")
    _touch(tmp_path, "other.traj.extxyz")
    enforce_clean_output_dir(tmp_path, PREFIX, force=False)
    assert (tmp_path / "other.energies").exists()


def test_gate_force_idempotent(tmp_path):
    _touch(tmp_path, f"{PREFIX}.energies")
    enforce_clean_output_dir(tmp_path, PREFIX, force=True)
    enforce_clean_output_dir(tmp_path, PREFIX, force=True)


def test_gate_error_message_lists_files(tmp_path, capsys):
    _touch(tmp_path, f"{PREFIX}.energies")
    _touch(tmp_path, f"{PREFIX}.traj.extxyz")
    with pytest.raises(SystemExit):
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)
    err = capsys.readouterr().err
    assert "ns.energies" in err
    assert "ns.traj.extxyz" in err
    assert "--force" in err


def test_gate_error_truncates_long_listing(tmp_path, capsys):
    """The listing shows the first 10 files, so 15 files leave "+5 more"."""
    for i in range(15):
        _touch(tmp_path, f"{PREFIX}.traj.snap.{i}.extxyz")
    with pytest.raises(SystemExit):
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)
    err = capsys.readouterr().err
    assert "+5 more" in err


def test_gate_includes_config_snapshot(tmp_path):
    _touch(tmp_path, snapshot_filename(PREFIX))
    with pytest.raises(SystemExit):
        enforce_clean_output_dir(tmp_path, PREFIX, force=False)
    enforce_clean_output_dir(tmp_path, PREFIX, force=True)
    assert not (tmp_path / snapshot_filename(PREFIX)).exists()


def test_gate_skips_matching_directories(tmp_path):
    (tmp_path / f"{PREFIX}.traj.frames").mkdir()
    enforce_clean_output_dir(tmp_path, PREFIX, force=False)
    enforce_clean_output_dir(tmp_path, PREFIX, force=True)
    assert (tmp_path / f"{PREFIX}.traj.frames").is_dir()


def test_discover_errors_when_no_checkpoint(tmp_path, capsys):
    with pytest.raises(SystemExit) as ex:
        discover_checkpoint(tmp_path, PREFIX)
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "no checkpoint" in err
    assert PREFIX in err


def test_discover_picks_final_when_only_final(tmp_path):
    _touch(tmp_path, f"{PREFIX}.final.checkpoint.h5")
    chosen = discover_checkpoint(tmp_path, PREFIX)
    assert chosen.name == f"{PREFIX}.final.checkpoint.h5"


def test_discover_picks_rolling_when_only_rolling(tmp_path):
    _touch(tmp_path, f"{PREFIX}.checkpoint.h5")
    chosen = discover_checkpoint(tmp_path, PREFIX)
    assert chosen.name == f"{PREFIX}.checkpoint.h5"


def test_discover_picks_higher_mtime(tmp_path):
    rolling = tmp_path / f"{PREFIX}.checkpoint.h5"
    final = tmp_path / f"{PREFIX}.final.checkpoint.h5"
    rolling.write_text("rolling")
    final.write_text("final")
    # Make rolling newer than final.
    old = time.time() - 100
    os.utime(final, (old, old))
    chosen = discover_checkpoint(tmp_path, PREFIX)
    assert chosen.name == rolling.name


def test_discover_prefers_final_on_mtime_tie(tmp_path):
    rolling = tmp_path / f"{PREFIX}.checkpoint.h5"
    final = tmp_path / f"{PREFIX}.final.checkpoint.h5"
    rolling.write_text("rolling")
    final.write_text("final")
    same = time.time()
    os.utime(rolling, (same, same))
    os.utime(final, (same, same))
    chosen = discover_checkpoint(tmp_path, PREFIX)
    assert chosen.name == final.name


def test_discover_ignores_initial(tmp_path):
    # initial.checkpoint.h5 is written at run start; not a restart source.
    _touch(tmp_path, f"{PREFIX}.initial.checkpoint.h5")
    with pytest.raises(SystemExit):
        discover_checkpoint(tmp_path, PREFIX)


def test_snapshot_path_for_final_checkpoint(tmp_path):
    p = snapshot_path_for_checkpoint(
        tmp_path / f"{PREFIX}.final.checkpoint.h5"
    )
    assert p == tmp_path / snapshot_filename(PREFIX)


def test_snapshot_path_for_rolling_checkpoint(tmp_path):
    p = snapshot_path_for_checkpoint(tmp_path / f"{PREFIX}.checkpoint.h5")
    assert p == tmp_path / snapshot_filename(PREFIX)


def test_snapshot_path_for_unknown_returns_none(tmp_path):
    assert snapshot_path_for_checkpoint(tmp_path / "random.h5") is None


def test_snapshot_write_read_roundtrip(tmp_path):
    """The snapshot holds the JSON-mode dump of the validated config.

    ``working_dir`` is a ``Path`` in the model, so this also checks that
    the dump is plain enough for ``yaml.safe_dump``.
    """
    root = RootSpec.model_validate(
        {
            "run": {"n_live": 8, "max_iterations": 5, "seed": 1},
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"backend_type": "harmonic"},
            "output": {"format": "none", "working_dir": str(tmp_path)},
        }
    )
    path = write_config_snapshot(tmp_path, PREFIX, root)
    assert path == tmp_path / snapshot_filename(PREFIX)
    assert read_config_snapshot(path) == root.model_dump(mode="json")
