"""Each move starts from its own configured ``step_size``.

The CLI used to broadcast the first move's step size to every move.  These
tests resolve a config whose moves have different step sizes, stop the run
at the walker initialisation, and check the step sizes it received.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxrens.sampling.nested_sampling as ns_mod
from jaxrens.cli.resolve import resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.nested_sampling import initial_step_sizes

STEP_SIZES = [0.05, 0.5]


class _Stop(Exception):
    pass


def _config(tmp_path, **extra) -> dict:
    cfg = {
        "run": {"n_live": 8, "max_iterations": 5, "n_mcmc_steps": 2},
        "moves": [
            {"type": "random_walk", "name": "rw_small", "step_size": 0.05},
            {"type": "random_walk", "name": "rw_large", "step_size": 0.5},
        ],
        "backend": {"type": "harmonic"},
        "output": {
            "format": "none",
            "working_dir": str(tmp_path),
            "info_interval": 999,
        },
    }
    cfg.update(extra)
    return cfg


@pytest.fixture
def single_device(monkeypatch):
    from jaxrens.cli import resolve as _r

    monkeypatch.setattr(_r, "_local_device_count", lambda: 1)


@pytest.fixture
def captured(monkeypatch):
    """Replace the NS initialisers with a spy that records and stops."""
    seen: dict[str, np.ndarray] = {}

    def spy(name):
        def f(*args, **kwargs):
            seen[name] = np.asarray(kwargs["step_sizes"])
            raise _Stop

        return f

    for name in ("init_ns", "init_ns_multi_gpu"):
        monkeypatch.setattr(ns_mod, name, spy(name))
    return seen


def test_single_run_starts_each_move_from_its_step_size(
    tmp_path, single_device, captured
):
    from jaxrens.cli.cli import _run_single

    resolved = resolve(RootSpec.model_validate(_config(tmp_path)))
    with pytest.raises(_Stop):
        _run_single(resolved)
    np.testing.assert_allclose(captured["init_ns"], STEP_SIZES)


def test_replica_run_starts_each_move_from_its_step_size(
    tmp_path, single_device, captured
):
    from jaxrens.cli.run import run_multi_gpu_from_config

    cfg = _config(
        tmp_path,
        ensemble={
            "type": "npt",
            "pressure": [0.01, 0.02],
            "pressure_units": "eva3",
        },
    )
    resolved = resolve(RootSpec.model_validate(cfg))
    with pytest.raises(_Stop):
        run_multi_gpu_from_config(resolved)
    np.testing.assert_allclose(captured["init_ns_multi_gpu"], STEP_SIZES)


def _descriptor(step_size):
    return MoveKernel(
        name="m", build_kernel=lambda *a, **k: None, step_size=step_size
    )


class TestInitialStepSizes:
    def test_from_descriptors(self):
        ss = initial_step_sizes(2, None, [_descriptor(0.05), _descriptor(0.5)])
        np.testing.assert_allclose(ss, STEP_SIZES)

    def test_explicit_value_wins(self):
        ss = initial_step_sizes(2, 0.3, [_descriptor(0.05), _descriptor(0.5)])
        np.testing.assert_allclose(ss, [0.3, 0.3])

    def test_default_without_descriptors(self):
        np.testing.assert_allclose(initial_step_sizes(3), [0.1, 0.1, 0.1])

    def test_mismatched_descriptors_raise(self):
        with pytest.raises(ValueError, match="2 move descriptors for 3"):
            initial_step_sizes(3, None, [_descriptor(0.1), _descriptor(0.2)])

    def test_dtype(self):
        assert initial_step_sizes(2).dtype == np.float32
