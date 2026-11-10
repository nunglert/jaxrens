"""Tests for ``validate_restart_compatibility`` (restart_validate.py).

On restart the current config is compared with the snapshot of the first
run: a change to an immutable field exits with code 2, a change to a soft
field only logs a warning. Both sides are scaled to absolute intervals
first, and a field missing from an older snapshot matches None.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from jaxrens.cli.output_gate import snapshot_filename
from jaxrens.cli.restart_validate import validate_restart_compatibility

PREFIX = "ns"


class _FakeRoot:
    """Stand-in for ``RootSpec``: ``model_dump`` returns the given dict."""

    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, mode: str = "json") -> dict:
        return self._payload


def _baseline_config() -> dict:
    """A minimal dumped config with every path the validator reads."""
    return {
        "interval_units": "absolute",
        "run": {
            "n_live": 64,
            "n_gpu": 1,
            "n_per_gpu": 1,
            "n_mcmc_steps": 5,
            "max_iterations": 10000,
            "seed": 1,
        },
        "moves": [{"type": "random_walk", "step_size": 0.1, "weight": 1.0}],
        "backend": {"type": "lj", "epsilon": 1.0, "sigma": 1.0},
        "ensemble": {"type": "npt", "pressure": 0.001},
        "inter_re": None,
        "cell": {"ndim": 3, "pbc": [True, True, True]},
        "output": {
            "out_file_prefix": PREFIX,
            "format": "extxyz",
            "traj_interval": 1,
            "snapshot_interval": 100,
            "checkpoint_interval": 1000,
            "info_interval": 10,
        },
        "termination": [{"type": "iteration", "max_iterations": 10000}],
        "adaptation": {"full_auto": False},
    }


def _write_snapshot(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / snapshot_filename(PREFIX)
    with open(path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    return path


def _checkpoint_path(tmp_path: Path) -> Path:
    p = tmp_path / f"{PREFIX}.final.checkpoint.h5"
    p.write_text("not really h5 — validator doesn't open it")
    return p


def test_identical_config_passes(tmp_path, caplog):
    payload = _baseline_config()
    snap = _write_snapshot(tmp_path, payload)
    ckpt = _checkpoint_path(tmp_path)

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(payload),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    # The unchanged seed is the only warning on an identical config.
    assert len(caplog.records) == 1
    assert "seed" in caplog.records[0].getMessage()


def test_n_live_change_refused(tmp_path, capsys):
    snap_payload = _baseline_config()
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["n_live"] = 128

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "run.n_live" in err
    assert "64" in err
    assert "128" in err


def test_pressure_change_refused(tmp_path, capsys):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["ensemble"]["pressure"] = 0.005

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    err = capsys.readouterr().err
    assert "ensemble" in err


def test_backend_change_refused(tmp_path, capsys):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["backend"] = {"type": "neuralil", "model_path": "/tmp/m.pkl"}

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    err = capsys.readouterr().err
    assert "backend" in err


def test_inter_re_change_refused(tmp_path, capsys):
    def config(n_swap_cycles):
        c = _baseline_config()
        c["ensemble"] = {"type": "npt", "pressure": [0.001, 0.01]}
        c["inter_re"] = {
            "flavor": "pressure",
            "re_interval": 1,
            "n_swap_cycles": n_swap_cycles,
        }
        return c

    snap = _write_snapshot(tmp_path, config(50))
    ckpt = _checkpoint_path(tmp_path)

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(config(10)),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    assert "inter_re" in capsys.readouterr().err


def test_replica_count_change_refused_via_pressure_list(tmp_path, capsys):
    """A longer ``ensemble.pressure`` list gets the replica-count message,
    which comes before the generic ensemble diff.
    """
    snap_payload = _baseline_config()
    snap_payload["ensemble"] = {"type": "npt", "pressure": [0.001, 0.005]}
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["ensemble"] = {
        "type": "npt",
        "pressure": [0.001, 0.005, 0.01, 0.05],
    }

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "replica count changed" in err
    assert "n_total = 2" in err
    assert "n_total = 4" in err


def test_replica_count_change_refused_via_composition_targets(
    tmp_path, capsys
):
    snap_payload = _baseline_config()
    snap_payload["inter_re"] = {
        "flavor": "xrens",
        "composition_targets": [[8, 0], [4, 4]],
    }
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["inter_re"] = {
        "flavor": "xrens",
        "composition_targets": [[8, 0], [4, 4], [0, 8]],
    }

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "replica count changed" in err
    assert "n_total = 2" in err
    assert "n_total = 3" in err


def test_replica_count_change_refused_via_semi_grand_mu(tmp_path, capsys):
    """A list of μ-vectors sets the replica count, like a pressure list."""
    snap_payload = _baseline_config()
    snap_payload["ensemble"] = {
        "type": "semi_grand",
        "chemical_potentials": [[0.0], [0.5]],
    }
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["ensemble"] = {
        "type": "semi_grand",
        "chemical_potentials": [[0.0], [0.5], [1.0]],
    }

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "replica count changed" in err
    assert "n_total = 2" in err
    assert "n_total = 3" in err


def test_single_mu_vector_is_one_replica(tmp_path, capsys):
    """A flat μ list has one entry per species, not one per replica.

    Going from a 2-species μ-vector to an NPT pressure list of 2 changes
    the replica count from 1 to 2.
    """
    snap_payload = _baseline_config()
    snap_payload["ensemble"] = {
        "type": "semi_grand",
        "chemical_potentials": [0.0, 0.5],
    }
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["ensemble"] = {"type": "npt", "pressure": [0.001, 0.005]}

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    err = capsys.readouterr().err
    assert "n_total = 1" in err
    assert "n_total = 2" in err


def test_replica_count_unchanged_but_values_changed_still_refused(
    tmp_path, capsys
):
    """Same number of pressures, one value changed: the generic ensemble
    diff refuses it, not the replica-count check.
    """
    snap_payload = _baseline_config()
    snap_payload["ensemble"] = {
        "type": "npt",
        "pressure": [0.001, 0.005, 0.01, 0.05],
    }
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["ensemble"] = {
        "type": "npt",
        "pressure": [0.002, 0.005, 0.01, 0.05],
    }

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "replica count changed" not in err
    assert "ensemble" in err


def test_cell_change_refused(tmp_path, capsys):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["cell"]["pbc"] = [True, True, False]

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    assert "cell" in capsys.readouterr().err


def test_topology_change_refused(tmp_path, capsys):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["n_gpu"] = 2
    current["run"]["n_per_gpu"] = 4

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "run.n_gpu" in err
    assert "run.n_per_gpu" in err


def test_multiple_hard_diffs_listed(tmp_path, capsys):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["n_live"] = 128
    current["ensemble"]["pressure"] = 0.01
    current["backend"]["type"] = "harmonic"

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    err = capsys.readouterr().err
    assert "run.n_live" in err
    assert "ensemble" in err
    assert "backend" in err


def test_move_retuning_warns_not_refuses(tmp_path, caplog):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["moves"] = [
        {"type": "random_walk", "step_size": 0.2, "weight": 1.0},
        {"type": "galilean", "step_size": 0.05, "weight": 1.0},
    ]

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert any("moves" in r.getMessage() for r in caplog.records)


def test_logging_cadence_warns_not_refuses(tmp_path, caplog):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["output"]["traj_interval"] = 5
    current["output"]["snapshot_interval"] = 500

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("traj_interval" in m for m in msgs)
    assert any("snapshot_interval" in m for m in msgs)


def test_seed_changed_does_not_warn(tmp_path, caplog):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["seed"] = 42

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    seed_warnings = [
        r
        for r in caplog.records
        if "seed" in r.getMessage() and "unchanged" in r.getMessage()
    ]
    assert seed_warnings == []


def test_missing_snapshot_warns_and_proceeds(tmp_path, caplog):
    ckpt = _checkpoint_path(tmp_path)
    nonexistent = tmp_path / snapshot_filename(PREFIX)
    assert not nonexistent.exists()

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(_baseline_config()),
            checkpoint_path=ckpt,
            snapshot_path=nonexistent,
        )
    assert any("snapshot not found" in r.getMessage() for r in caplog.records)


def test_none_snapshot_path_warns_and_proceeds(tmp_path, caplog):
    ckpt = _checkpoint_path(tmp_path)
    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(_baseline_config()),
            checkpoint_path=ckpt,
            snapshot_path=None,
        )
    assert any("snapshot not found" in r.getMessage() for r in caplog.records)


def test_snapshot_missing_optional_backend_field_not_refused(tmp_path):
    """Snapshot predates ``backend.softcore_repulsion``; current fills None."""
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["backend"]["softcore_repulsion"] = None

    # Missing versus None is not a change, so this must not raise.
    validate_restart_compatibility(
        _FakeRoot(current),
        checkpoint_path=ckpt,
        snapshot_path=snap,
    )


def test_added_backend_field_with_nondefault_value_still_refused(
    tmp_path, capsys
):
    """A newly-set non-default value for an added field IS a real change."""
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["backend"]["softcore_repulsion"] = {"a0": 4.0, "d0": 1000.0}

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    assert "backend" in capsys.readouterr().err


def _config_with_inter_re(units: str, re_interval, n_live: int) -> dict:
    c = _baseline_config()
    c["interval_units"] = units
    c["run"]["n_live"] = n_live
    c["ensemble"] = {"type": "npt", "pressure": [0.001, 0.002]}
    c["inter_re"] = {
        "flavor": "pressure",
        "re_interval": re_interval,
        "n_swap_cycles": 50,
    }
    return c


def test_per_walker_inter_re_interval_scaled_not_refused(tmp_path):
    """0.002 per_walker sweeps x 500 walkers = 1, the snapshot's value."""
    snap = _write_snapshot(
        tmp_path,
        _config_with_inter_re("absolute", 1, n_live=500),
    )
    ckpt = _checkpoint_path(tmp_path)
    current = _config_with_inter_re("per_walker", 0.002, n_live=500)

    validate_restart_compatibility(
        _FakeRoot(current),
        checkpoint_path=ckpt,
        snapshot_path=snap,
    )


def test_inter_re_interval_genuine_change_still_refused(tmp_path, capsys):
    """0.01 per_walker sweeps x 500 walkers = 5, not the snapshot's 1."""
    snap = _write_snapshot(
        tmp_path,
        _config_with_inter_re("absolute", 1, n_live=500),
    )
    ckpt = _checkpoint_path(tmp_path)
    current = _config_with_inter_re("per_walker", 0.01, n_live=500)

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    assert "inter_re" in capsys.readouterr().err


def test_per_walker_inter_re_interval_scaled_by_n_cull(tmp_path):
    """0.02 x 500 walkers / n_cull 10 = 1, the snapshot's value."""
    snap_payload = _config_with_inter_re("absolute", 1, n_live=500)
    snap_payload["run"]["n_cull"] = 10
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)
    current = _config_with_inter_re("per_walker", 0.02, n_live=500)
    current["run"]["n_cull"] = 10

    validate_restart_compatibility(
        _FakeRoot(current),
        checkpoint_path=ckpt,
        snapshot_path=snap,
    )


def test_per_walker_n_cull_change_of_scaled_interval_refused(tmp_path, capsys):
    """0.1 x 500 walkers / n_cull 10 = 5, not the snapshot's 1."""
    snap_payload = _config_with_inter_re("absolute", 1, n_live=500)
    snap_payload["run"]["n_cull"] = 10
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)
    current = _config_with_inter_re("per_walker", 0.1, n_live=500)
    current["run"]["n_cull"] = 10

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    assert ex.value.code == 2
    assert "inter_re" in capsys.readouterr().err


def test_per_walker_output_intervals_do_not_warn(tmp_path, caplog):
    """per_walker intervals equal to the snapshot's absolute ones after
    scaling do not warn.
    """
    snap_payload = _baseline_config()
    snap_payload["run"]["n_live"] = 100
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["interval_units"] = "per_walker"
    current["run"]["n_live"] = 100
    # Each value times 100 walkers gives the snapshot's absolute value.
    current["output"]["traj_interval"] = 0.01  # 1
    current["output"]["snapshot_interval"] = 1.0
    current["output"]["checkpoint_interval"] = 10.0
    current["output"]["info_interval"] = 0.1
    current["termination"][0]["max_iterations"] = 100.0
    current["run"]["max_iterations"] = 100.0

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(current),
            checkpoint_path=ckpt,
            snapshot_path=snap,
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("traj_interval" in m or "interval_units" in m for m in msgs)
