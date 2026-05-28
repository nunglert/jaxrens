"""Strict restart-compatibility validator: refuse on immutables, warn on soft diffs."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from jaxrens.cli.output_gate import snapshot_filename
from jaxrens.cli.restart_validate import validate_restart_compatibility


PREFIX = "ns"


# ---------------------------------------------------------------------------
# Test scaffolding — fake "root" object that has ``model_dump(mode="json")``
# ---------------------------------------------------------------------------


class _FakeRoot:
    """Stand-in for ``RootSpec``: holds a payload dict and dumps it back."""

    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, mode: str = "json") -> dict:
        return self._payload


def _baseline_config() -> dict:
    """A complete minimal payload covering every path the validator looks at."""
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


# ---------------------------------------------------------------------------
# Happy path: identical snapshot and current config → no refusal, no warnings
# ---------------------------------------------------------------------------


def test_identical_config_passes(tmp_path, caplog):
    payload = _baseline_config()
    snap = _write_snapshot(tmp_path, payload)
    ckpt = _checkpoint_path(tmp_path)

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(payload), checkpoint_path=ckpt, snapshot_path=snap,
        )
    # Only the "seed unchanged" warning is expected on an identical config.
    seed_warnings = [r for r in caplog.records if "seed" in r.getMessage()]
    assert len(seed_warnings) == 1


# ---------------------------------------------------------------------------
# Hard refusals
# ---------------------------------------------------------------------------


def test_n_live_change_refused(tmp_path, capsys):
    snap_payload = _baseline_config()
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["n_live"] = 128

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
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
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
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
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    err = capsys.readouterr().err
    assert "backend" in err


def test_inter_re_change_refused(tmp_path):
    snap_payload = _baseline_config()
    snap_payload["inter_re"] = {"flavor": "pressure", "pressures": [0.001, 0.01]}
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["inter_re"] = {"flavor": "pressure", "pressures": [0.001, 0.005]}

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )


def test_replica_count_change_refused_via_pressure_list(tmp_path, capsys):
    """Growing/shrinking ``ensemble.pressure`` list across restart is refused
    with a targeted replica-count message before the generic ensemble diff."""
    snap_payload = _baseline_config()
    snap_payload["ensemble"] = {"type": "npt", "pressure": [0.001, 0.005]}
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["ensemble"] = {"type": "npt", "pressure": [0.001, 0.005, 0.01, 0.05]}

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "replica count changed" in err
    assert "n_total = 2" in err
    assert "n_total = 4" in err


def test_replica_count_change_refused_via_composition_targets(tmp_path, capsys):
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
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "replica count changed" in err
    assert "n_total = 2" in err
    assert "n_total = 3" in err


def test_replica_count_unchanged_but_values_changed_still_refused(tmp_path, capsys):
    """Length-preserved value changes fall through to the generic ensemble
    subtree diff and are still refused — just via the broader path."""
    snap_payload = _baseline_config()
    snap_payload["ensemble"] = {"type": "npt", "pressure": [0.001, 0.005, 0.01, 0.05]}
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["ensemble"] = {"type": "npt", "pressure": [0.002, 0.005, 0.01, 0.05]}

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    assert ex.value.code == 2
    err = capsys.readouterr().err
    # The replica-count check must NOT fire (lengths match);
    # the generic ensemble immutable-subtree check must fire instead.
    assert "replica count changed" not in err
    assert "ensemble" in err


def test_cell_change_refused(tmp_path):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["cell"]["pbc"] = [True, True, False]

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )


def test_topology_change_refused(tmp_path):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["n_gpu"] = 2
    current["run"]["n_per_gpu"] = 4

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )


def test_multiple_hard_diffs_listed(tmp_path, capsys):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["n_live"] = 128
    current["ensemble"]["pressure"] = 0.01
    current["backend"]["type"] = "harmonic"

    with pytest.raises(SystemExit):
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    err = capsys.readouterr().err
    assert "run.n_live" in err
    assert "ensemble" in err
    assert "backend" in err


# ---------------------------------------------------------------------------
# Soft warnings (run proceeds)
# ---------------------------------------------------------------------------


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
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
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
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("traj_interval" in m for m in msgs)
    assert any("snapshot_interval" in m for m in msgs)


def test_seed_unchanged_warns(tmp_path, caplog):
    payload = _baseline_config()
    snap = _write_snapshot(tmp_path, payload)
    ckpt = _checkpoint_path(tmp_path)

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(payload), checkpoint_path=ckpt, snapshot_path=snap,
        )
    assert any("seed" in r.getMessage() for r in caplog.records)


def test_seed_changed_does_not_warn(tmp_path, caplog):
    snap = _write_snapshot(tmp_path, _baseline_config())
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["run"]["seed"] = 42

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    seed_warnings = [
        r for r in caplog.records
        if "seed" in r.getMessage() and "unchanged" in r.getMessage()
    ]
    assert seed_warnings == []


# ---------------------------------------------------------------------------
# Missing snapshot — legacy artifact dirs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fix 1: schema-evolution tolerance — a newly-added Optional field (absent in
# an older snapshot, None in the current dump) must not read as a change.
# ---------------------------------------------------------------------------


def test_snapshot_missing_optional_backend_field_not_refused(tmp_path):
    """Snapshot predates ``backend.softcore_repulsion``; current fills None."""
    snap = _write_snapshot(tmp_path, _baseline_config())  # no softcore_repulsion
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["backend"]["softcore_repulsion"] = None  # schema default

    # Must NOT raise — missing-vs-None is not a real change.
    validate_restart_compatibility(
        _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
    )


def test_added_backend_field_with_nondefault_value_still_refused(tmp_path, capsys):
    """A newly-set non-default value for an added field IS a real change."""
    snap = _write_snapshot(tmp_path, _baseline_config())  # no softcore_repulsion
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["backend"]["softcore_repulsion"] = {"a0": 4.0, "d0": 1000.0}

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    assert ex.value.code == 2
    assert "backend" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Fix 2: interval-unit normalisation — per_walker vs absolute must compare on
# scaled (absolute) values, not raw, for the immutable inter_re.re_interval.
# ---------------------------------------------------------------------------


def _config_with_inter_re(units: str, re_interval, n_live: int) -> dict:
    c = _baseline_config()
    c["interval_units"] = units
    c["run"]["n_live"] = n_live
    c["ensemble"] = {"type": "npt", "pressure": [0.001, 0.002]}  # 2 replicas
    c["inter_re"] = {
        "flavor": "pressure", "re_interval": re_interval, "n_swap_cycles": 50,
    }
    return c


def test_per_walker_inter_re_interval_scaled_not_refused(tmp_path):
    """re_interval 0.002 per_walker × n_live 500 = 1 == snapshot's absolute 1."""
    snap = _write_snapshot(
        tmp_path, _config_with_inter_re("absolute", 1, n_live=500),
    )
    ckpt = _checkpoint_path(tmp_path)
    current = _config_with_inter_re("per_walker", 0.002, n_live=500)

    # Must NOT raise — both normalise to re_interval=1.
    validate_restart_compatibility(
        _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
    )


def test_inter_re_interval_genuine_change_still_refused(tmp_path, capsys):
    """0.01 per_walker × 500 = 5 ≠ snapshot's 1 → genuine change, refused."""
    snap = _write_snapshot(
        tmp_path, _config_with_inter_re("absolute", 1, n_live=500),
    )
    ckpt = _checkpoint_path(tmp_path)
    current = _config_with_inter_re("per_walker", 0.01, n_live=500)

    with pytest.raises(SystemExit) as ex:
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    assert ex.value.code == 2
    assert "inter_re" in capsys.readouterr().err


def test_per_walker_output_intervals_do_not_warn(tmp_path, caplog):
    """Soft output.*_interval fields also normalise → no spurious warning."""
    snap_payload = _baseline_config()  # absolute: traj=1, snapshot=100, ...
    snap_payload["run"]["n_live"] = 100  # n_live is immutable — match both sides
    snap = _write_snapshot(tmp_path, snap_payload)
    ckpt = _checkpoint_path(tmp_path)

    current = _baseline_config()
    current["interval_units"] = "per_walker"
    current["run"]["n_live"] = 100
    # per_walker values that scale back to the snapshot's absolute ones.
    current["output"]["traj_interval"] = 0.01      # *100 = 1
    current["output"]["snapshot_interval"] = 1.0   # *100 = 100
    current["output"]["checkpoint_interval"] = 10.0 # *100 = 1000
    current["output"]["info_interval"] = 0.1       # *100 = 10
    current["termination"][0]["max_iterations"] = 100.0  # *100 = 10000
    current["run"]["max_iterations"] = 100.0       # *100 = 10000

    with caplog.at_level(logging.WARNING):
        validate_restart_compatibility(
            _FakeRoot(current), checkpoint_path=ckpt, snapshot_path=snap,
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("traj_interval" in m or "interval_units" in m for m in msgs)
