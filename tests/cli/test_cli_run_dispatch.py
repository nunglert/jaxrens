"""Coverage tests for ``_cmd_run`` in ``jaxrens.cli.cli``.

Strategy: monkeypatch the three orchestrators (``_run_single``,
``run_sharded_from_config``, ``run_multi_gpu_from_config``) to no-op
stubs that just record they were called.  This isolates ``_cmd_run``
itself — flag parsing, restart-conflict guards, batcher-based
dispatch — from the heavy NS execution that lives in ``cli.run``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxrens.cli.cli import main

_DATA = Path(__file__).parent.parent / "_assets" / "data" / "cli"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_device(monkeypatch):
    """Force ``_local_device_count`` → 1 so minimal.yaml resolves to SingleRun
    regardless of the host's actual GPU count."""
    from jaxrens.cli import resolve as _r

    monkeypatch.setattr(_r, "_local_device_count", lambda: 1)


@pytest.fixture
def stub_orchestrators(monkeypatch):
    """Replace the three NS dispatch functions with no-op recorders.

    Returns a dict ``{name: list_of_calls}`` where each ``list_of_calls``
    is populated with the ``(args, kwargs)`` of every invocation.  Lets
    tests assert which orchestrator ran without paying for the actual NS
    loop or JIT compilation.
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


@pytest.fixture
def fresh_workdir(tmp_path, monkeypatch):
    """Run with ``output.working_dir`` rebased into ``tmp_path`` so
    ``configure_file_logging`` writes to a clean directory and the
    output-dir gate doesn't trip on prior runs."""
    return tmp_path


def _write_minimal_yaml(path: Path, working_dir: Path, **extras) -> Path:
    """Materialise a tiny valid YAML config rooted at ``working_dir``.

    ``extras`` may include nested overrides:
    ``init={"restart_file": "..."}`` etc.
    """
    import yaml as _yaml

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
    path.write_text(_yaml.safe_dump(cfg))
    return path


# ---------------------------------------------------------------------------
# Restart-conflict guards (cli.py 220-236)
# ---------------------------------------------------------------------------


class TestRestartConflictGuards:
    """The three mutually-exclusive restart triggers (``--force``,
    ``--resume``, ``init.restart_file``) form three pairwise conflicts.
    All three are caught before any I/O and return exit code 2."""

    def test_force_and_resume_conflict(
        self,
        fresh_workdir,
        single_device,
        capsys,
    ):
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--force", "--resume"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--force and --resume" in err

    def test_force_and_restart_file_conflict(
        self,
        fresh_workdir,
        single_device,
        capsys,
    ):
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
            init={"restart_file": str(fresh_workdir / "fake.h5")},
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--force"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--force is incompatible with init.restart_file" in err

    def test_resume_and_restart_file_conflict(
        self,
        fresh_workdir,
        single_device,
        capsys,
    ):
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
            init={"restart_file": str(fresh_workdir / "fake.h5")},
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--resume"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--resume and init.restart_file" in err


# ---------------------------------------------------------------------------
# Dispatch by batcher type (cli.py 309-340)
# ---------------------------------------------------------------------------


class TestRunDispatch:
    """``_cmd_run`` picks one of three orchestrators based on
    ``resolved.batcher`` — verify each branch reaches the right stub."""

    def test_single_run_dispatch(
        self,
        fresh_workdir,
        single_device,
        stub_orchestrators,
    ):
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg)])
        assert exc_info.value.code == 0
        assert len(stub_orchestrators["single"]) == 1
        assert len(stub_orchestrators["sharded"]) == 0
        assert len(stub_orchestrators["multi_gpu"]) == 0

    def test_multi_replica_dispatch(
        self,
        fresh_workdir,
        single_device,
        stub_orchestrators,
    ):
        """NPT pressure list of length 2 with ``_local_device_count → 1``
        → PmapVmapRuns(n_gpu=1, n_per_gpu=2) → ``run_multi_gpu_from_config``.

        Forcing 1 device avoids the real pmap fan-out that the resolver's
        consolidated initial-energy compute would otherwise attempt.
        Same pattern as ``test_multi_run.py::TestRunMultiGpuFromConfig``.
        """
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
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
        fresh_workdir,
        single_device,
        stub_orchestrators,
    ):
        """``run.shard_n_gpu=2`` (with the resolver's single-replica
        initial-energy compute pinned to ``batcher=SingleRun()``) →
        ShardedSingleRun → ``run_sharded_from_config``.  Stubbed
        orchestrator means no actual sharded pmap runs."""
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
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


# ---------------------------------------------------------------------------
# _assert_n_gpus (cli.py 137-170)
# ---------------------------------------------------------------------------


class TestAssertNGpus:
    """``--n-gpus=N`` aborts early with exit code 2 when the visible
    device count doesn't match."""

    def test_mismatch_returns_2(self, fresh_workdir, monkeypatch, capsys):
        # Pretend JAX only sees 1 device, while the user asserted 4.
        import jax

        monkeypatch.setattr(jax, "local_devices", lambda: [object()])
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--n-gpus", "4"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "but JAX sees 1 local" in err
        assert "SLURM" in err  # diagnostic text mentions SLURM

    def test_match_proceeds(
        self,
        fresh_workdir,
        single_device,
        stub_orchestrators,
        monkeypatch,
    ):
        import jax

        monkeypatch.setattr(jax, "local_devices", lambda: [object()])
        cfg = _write_minimal_yaml(
            fresh_workdir / "cfg.yaml",
            fresh_workdir / "out",
        )
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-c", str(cfg), "--n-gpus", "1"])
        assert exc_info.value.code == 0
        assert len(stub_orchestrators["single"]) == 1
