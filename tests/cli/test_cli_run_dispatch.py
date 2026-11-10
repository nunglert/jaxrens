"""Tests for ``_cmd_run`` in ``jaxrens.cli.cli`` (``jaxrens run``).

The three orchestrators (``_run_single``, ``run_sharded_from_config`` and
``run_multi_gpu_from_config``) are replaced by stubs that record their
calls. The tests then check flag handling, the restart-conflict guards and
which orchestrator the resolved batcher selects, without running NS.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import pytest
import yaml

from jaxrens.cli import resolve as _r
from jaxrens.cli.cli import _package_version, main
from jaxrens.cli.output_gate import snapshot_filename


@pytest.fixture
def single_device(monkeypatch):
    """Make the resolver see one device, whatever the host has."""
    monkeypatch.setattr(_r, "_local_device_count", lambda: 1)


@pytest.fixture
def stub_orchestrators(monkeypatch):
    """Replace the three orchestrators with stubs that record their calls.

    Returns ``{name: [(args, kwargs), ...]}`` for ``"single"``,
    ``"sharded"`` and ``"multi_gpu"``.
    """
    calls: dict[str, list] = {
        "single": [],
        "sharded": [],
        "multi_gpu": [],
    }

    def _stub(label):
        def _f(*args, **kwargs):
            calls[label].append((args, kwargs))

        return _f

    monkeypatch.setattr(
        "jaxrens.cli.cli._run_single",
        _stub("single"),
    )
    monkeypatch.setattr(
        "jaxrens.cli.run.run_sharded_from_config",
        _stub("sharded"),
    )
    monkeypatch.setattr(
        "jaxrens.cli.run.run_multi_gpu_from_config",
        _stub("multi_gpu"),
    )
    return calls


def _write_minimal_yaml(path: Path, working_dir: Path, **extras) -> Path:
    """Write a small valid YAML config whose output goes to ``working_dir``.

    Each keyword in ``extras`` replaces a top-level section, for example
    ``init={"restart_file": "..."}``.
    """
    cfg = {
        "run": {
            "n_live": 8,
            "max_iterations": 5,
            "n_mcmc_steps": 2,
            "seed": 0,
        },
        "moves": [{"type": "random_walk", "step_size": 0.3}],
        "backend": {"type": "harmonic"},
        "output": {
            "format": "none",
            "working_dir": str(working_dir),
            "info_interval": 999,
        },
    }
    for k, v in extras.items():
        cfg[k] = v
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestRestartConflictGuards:
    """Check that any two of ``--force``, ``--resume`` and
    ``init.restart_file`` together give exit code 2 before any I/O.
    """

    def test_force_and_resume_conflict(
        self,
        tmp_path,
        single_device,
        capsys,
    ):
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--force", "--resume"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--force and --resume" in err

    def test_force_and_restart_file_conflict(
        self,
        tmp_path,
        single_device,
        capsys,
    ):
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
            init={"restart_file": str(tmp_path / "fake.h5")},
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--force"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--force is incompatible with init.restart_file" in err

    def test_resume_and_restart_file_conflict(
        self,
        tmp_path,
        single_device,
        capsys,
    ):
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
            init={"restart_file": str(tmp_path / "fake.h5")},
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--resume"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--resume and init.restart_file" in err


class TestRunDispatch:
    """Check that each batcher type reaches its own orchestrator."""

    def test_single_run_dispatch(
        self,
        tmp_path,
        single_device,
        stub_orchestrators,
    ):
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg)])
        assert exc_info.value.code == 0
        assert len(stub_orchestrators["single"]) == 1
        assert len(stub_orchestrators["sharded"]) == 0
        assert len(stub_orchestrators["multi_gpu"]) == 0

    def test_logs_package_version_for_reproducibility(
        self,
        tmp_path,
        single_device,
        stub_orchestrators,
        caplog,
    ):
        """The log records the jaxrens version, so it shows which build ran."""
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
        )
        with caplog.at_level(logging.INFO, logger="jaxrens.cli.cli"):
            with pytest.raises(SystemExit) as exc_info:
                main(["run", "-c", str(cfg)])
        assert exc_info.value.code == 0
        expected = f"jaxrens version: {_package_version()}"
        assert any(expected in r.message for r in caplog.records)

    def test_multi_replica_dispatch(
        self,
        tmp_path,
        single_device,
        stub_orchestrators,
    ):
        """Two pressures on one device resolve to PmapVmapRuns(n_gpu=1,
        n_per_gpu=2) and reach ``run_multi_gpu_from_config``.

        With one device the resolver does not try a real pmap over GPUs
        when it computes the initial energies.
        """
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
            ensemble={"type": "npt", "pressure": [0.01, 0.02]},
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg)])
        assert exc_info.value.code == 0
        assert len(stub_orchestrators["multi_gpu"]) == 1
        assert len(stub_orchestrators["single"]) == 0
        assert len(stub_orchestrators["sharded"]) == 0

    def test_sharded_single_dispatch(
        self,
        tmp_path,
        single_device,
        stub_orchestrators,
    ):
        """``run.shard_n_gpu=2`` resolves to ShardedSingleRun and reaches
        ``run_sharded_from_config``.
        """
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "run",
                    "-c",
                    str(cfg),
                    "--set",
                    "run.shard_n_gpu=2",
                ]
            )
        assert exc_info.value.code == 0
        assert len(stub_orchestrators["sharded"]) == 1
        assert len(stub_orchestrators["single"]) == 0
        assert len(stub_orchestrators["multi_gpu"]) == 0


class TestFreshRunOutput:
    """Check the output directory handling of a fresh run.

    The run refuses a directory that already holds artifacts unless
    ``--force`` is given, writes the config snapshot, and scales
    ``per_walker`` intervals only once.
    """

    def test_writes_snapshot_and_log(
        self, tmp_path, single_device, stub_orchestrators
    ):
        cfg = _write_minimal_yaml(tmp_path / "cfg.yaml", tmp_path / "out")
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg)])
        assert exc_info.value.code == 0
        assert (tmp_path / "out" / snapshot_filename("ns")).exists()
        # The log goes next to the output directory, not inside it.
        assert (tmp_path / "ns.log").exists()

    def test_dirty_output_dir_refused(
        self, tmp_path, single_device, stub_orchestrators
    ):
        cfg = _write_minimal_yaml(tmp_path / "cfg.yaml", tmp_path / "out")
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "ns.energies").write_text("stale")
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg)])
        assert exc_info.value.code == 2
        assert stub_orchestrators["single"] == []

    def test_force_wipes_dirty_output_dir(
        self, tmp_path, single_device, stub_orchestrators
    ):
        cfg = _write_minimal_yaml(tmp_path / "cfg.yaml", tmp_path / "out")
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "ns.energies").write_text("stale")
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--force"])
        assert exc_info.value.code == 0
        assert not (tmp_path / "out" / "ns.energies").exists()
        assert len(stub_orchestrators["single"]) == 1

    def test_per_walker_intervals_scaled_once(
        self, tmp_path, single_device, stub_orchestrators
    ):
        """With n_live = 8, 0.5 sweeps is 4 iterations (32 if scaled twice)."""
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
            interval_units="per_walker",
        )
        data = yaml.safe_load(cfg.read_text())
        data["output"]["info_interval"] = 0.5
        # The default adjust_interval (100) would warn in per_walker mode.
        data["adaptation"] = {"adjust_interval": 0.5}
        cfg.write_text(yaml.safe_dump(data))
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg)])
        assert exc_info.value.code == 0
        ((args, _),) = stub_orchestrators["single"]
        assert args[0].output.info_interval == 4


class TestAssertNGpus:
    """Check that ``--n-gpus=N`` gives exit code 2 unless JAX sees N
    devices.
    """

    def test_mismatch_returns_2(self, tmp_path, monkeypatch, capsys):
        # JAX sees 1 device, the user asserts 4.
        monkeypatch.setattr(jax, "local_devices", lambda: [object()])
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--n-gpus", "4"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "but JAX sees 1 local" in err
        assert "SLURM" in err

    def test_match_proceeds(
        self,
        tmp_path,
        single_device,
        stub_orchestrators,
        monkeypatch,
    ):
        monkeypatch.setattr(jax, "local_devices", lambda: [object()])
        cfg = _write_minimal_yaml(
            tmp_path / "cfg.yaml",
            tmp_path / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--n-gpus", "1"])
        assert exc_info.value.code == 0
        assert len(stub_orchestrators["single"]) == 1
