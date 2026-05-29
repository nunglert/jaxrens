"""Writer mode contract: mode='w' truncates a pre-existing file, mode='a' preserves it.

Regression for the bug where ``EnergyLogger`` opened with ``mode='w'``
unconditionally and silently destroyed a prior run's ``.energies`` file
when ``jaxrens run`` was invoked again against the same output dir.
"""

from __future__ import annotations

import numpy as np

from jaxrens.io.acc_rates_log import AccRatesLogger
from jaxrens.io.adaptation_log import AdaptationLogger
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.max_neighbors_log import MaxNeighborsLogger
from jaxrens.io.re_stats_log import RELogger
from jaxrens.io.trajectory import (
    ExtxyzTrajectoryWriter,
    H5TrajectoryWriter,
)


# ---------------------------------------------------------------------------
# EnergyLogger
# ---------------------------------------------------------------------------


def _write_energies(path, *, mode, n, base_iter=0):
    lg = EnergyLogger(path, n_walkers=4, n_atoms=2, mode=mode)
    for i in range(n):
        lg.write_entry(base_iter + i, energy=-float(base_iter + i), volume=1.0)
    lg.close()


def test_energy_logger_mode_w_truncates(tmp_path):
    path = tmp_path / "ns.energies"
    _write_energies(path, mode="w", n=10)
    _write_energies(path, mode="w", n=3, base_iter=100)

    log = EnergyLogger.read(path)
    assert len(log.energies) == 3
    assert int(log.iterations[0]) == 100


def test_energy_logger_mode_a_appends(tmp_path):
    path = tmp_path / "ns.energies"
    _write_energies(path, mode="w", n=10)
    _write_energies(path, mode="a", n=3, base_iter=100)

    log = EnergyLogger.read(path)
    assert len(log.energies) == 13
    assert int(log.iterations[0]) == 0
    assert int(log.iterations[-1]) == 102


# ---------------------------------------------------------------------------
# ExtxyzTrajectoryWriter
# ---------------------------------------------------------------------------


def _walker(i):
    return {
        "positions": np.array([[float(i), 0.0, 0.0]]),
        "types": np.array([0]),
        "energy": float(-i),
    }


def _count_frames(path):
    from ase.io import read as ase_read
    return len(ase_read(str(path), ":"))


def test_extxyz_writer_mode_w_truncates(tmp_path):
    path = tmp_path / "traj.extxyz"
    w1 = ExtxyzTrajectoryWriter(path, symbol_map={0: "H"}, wrap=False, mode="w")
    for i in range(5):
        w1.write_dead_point(i, _walker(i), float(-i))
    w1.close()
    assert _count_frames(path) == 5

    w2 = ExtxyzTrajectoryWriter(path, symbol_map={0: "H"}, wrap=False, mode="w")
    for i in range(2):
        w2.write_dead_point(100 + i, _walker(i), float(-i))
    w2.close()
    assert _count_frames(path) == 2


def test_extxyz_writer_mode_a_appends(tmp_path):
    path = tmp_path / "traj.extxyz"
    w1 = ExtxyzTrajectoryWriter(path, symbol_map={0: "H"}, wrap=False, mode="w")
    for i in range(5):
        w1.write_dead_point(i, _walker(i), float(-i))
    w1.close()

    w2 = ExtxyzTrajectoryWriter(path, symbol_map={0: "H"}, wrap=False, mode="a")
    for i in range(2):
        w2.write_dead_point(100 + i, _walker(i), float(-i))
    w2.close()
    assert _count_frames(path) == 7


def test_extxyz_writer_accumulates_within_single_run(tmp_path):
    """First write of a mode='w' run overwrites; subsequent writes append."""
    path = tmp_path / "traj.extxyz"
    w = ExtxyzTrajectoryWriter(path, symbol_map={0: "H"}, wrap=False, mode="w")
    for i in range(4):
        w.write_dead_point(i, _walker(i), float(-i))
    w.close()
    assert _count_frames(path) == 4


# ---------------------------------------------------------------------------
# H5TrajectoryWriter
# ---------------------------------------------------------------------------


def test_h5_trajectory_writer_mode_w_truncates(tmp_path):
    import h5py

    path = tmp_path / "traj.h5"
    w1 = H5TrajectoryWriter(path, {0: "H"}, mode="w")
    for i in range(3):
        w1.write_dead_point(i, _walker(i), float(-i))
    w1.close()

    w2 = H5TrajectoryWriter(path, {0: "H"}, mode="w")
    w2.write_dead_point(100, _walker(0), 0.0)
    w2.close()

    with h5py.File(path, "r") as f:
        # Only the iter=100 group should be present.
        assert "100" in f
        assert "0" not in f
        assert "1" not in f


def test_h5_trajectory_writer_mode_a_preserves(tmp_path):
    import h5py

    path = tmp_path / "traj.h5"
    w1 = H5TrajectoryWriter(path, {0: "H"}, mode="w")
    for i in range(3):
        w1.write_dead_point(i, _walker(i), float(-i))
    w1.close()

    w2 = H5TrajectoryWriter(path, {0: "H"}, mode="a")
    w2.write_dead_point(100, _walker(0), 0.0)
    w2.close()

    with h5py.File(path, "r") as f:
        assert "0" in f
        assert "1" in f
        assert "2" in f
        assert "100" in f


# ---------------------------------------------------------------------------
# AdaptationLogger
# ---------------------------------------------------------------------------


def _write_adaptation(path, *, mode, n, base_iter=0):
    lg = AdaptationLogger(path=path, move_names=["a", "b"], n_runs=1, mode=mode)
    for i in range(n):
        lg.write_entry(
            base_iter + i,
            step_sizes=np.array([0.1, 0.2]),
            acceptance_rates=np.array([0.5, 0.5]),
        )
    lg.close()


def test_adaptation_logger_mode_w_truncates(tmp_path):
    path = tmp_path / "adapt.h5"
    _write_adaptation(path, mode="w", n=10)
    _write_adaptation(path, mode="w", n=3, base_iter=100)

    log = AdaptationLogger.read(path)
    assert log.iterations.shape == (3,)
    assert int(log.iterations[0]) == 100


def test_adaptation_logger_mode_a_appends(tmp_path):
    path = tmp_path / "adapt.h5"
    _write_adaptation(path, mode="w", n=10)
    _write_adaptation(path, mode="a", n=3, base_iter=100)

    log = AdaptationLogger.read(path)
    assert log.iterations.shape == (13,)


# ---------------------------------------------------------------------------
# RELogger
# ---------------------------------------------------------------------------


def _write_re(path, *, mode, n, base_iter=0):
    lg = RELogger(
        path=path, n_pairs=2, flavor="pressure", flush_interval=1, mode=mode,
    )
    for i in range(n):
        lg.write_entry(
            base_iter + i,
            n_accepted_per_pair=np.array([1, 2], dtype=np.int32),
            n_attempted_per_pair=np.array([5, 5], dtype=np.int32),
        )
    lg.close()


def test_re_logger_mode_w_truncates(tmp_path):
    path = tmp_path / "re.h5"
    _write_re(path, mode="w", n=10)
    _write_re(path, mode="w", n=3, base_iter=100)

    log = RELogger.read(path)
    assert log.iterations.shape == (3,)
    assert int(log.iterations[0]) == 100


def test_re_logger_mode_a_appends(tmp_path):
    path = tmp_path / "re.h5"
    _write_re(path, mode="w", n=10)
    _write_re(path, mode="a", n=3, base_iter=100)

    log = RELogger.read(path)
    assert log.iterations.shape == (13,)


# ---------------------------------------------------------------------------
# MaxNeighborsLogger
# ---------------------------------------------------------------------------


def _write_mn(path, *, mode, n, base_iter=0):
    lg = MaxNeighborsLogger(
        path=path, n_runs=1, n_walkers=2, flush_interval=1, mode=mode,
    )
    for i in range(n):
        lg.write_entry(
            base_iter + i,
            max_neighbor_count=np.array([[5, 6]], dtype=np.int32),
            bucket_size=np.array([8], dtype=np.int32),
            overflow=np.array([False]),
        )
    lg.close()


def test_max_neighbors_logger_mode_w_truncates(tmp_path):
    path = tmp_path / "mn.h5"
    _write_mn(path, mode="w", n=10)
    _write_mn(path, mode="w", n=3, base_iter=100)

    log = MaxNeighborsLogger.read(path)
    assert log.iterations.shape == (3,)
    assert int(log.iterations[0]) == 100


def test_max_neighbors_logger_mode_a_appends(tmp_path):
    path = tmp_path / "mn.h5"
    _write_mn(path, mode="w", n=10)
    _write_mn(path, mode="a", n=3, base_iter=100)

    log = MaxNeighborsLogger.read(path)
    assert log.iterations.shape == (13,)


# ---------------------------------------------------------------------------
# AccRatesLogger
# ---------------------------------------------------------------------------


def _write_acc(path, *, mode, n, base_iter=0):
    lg = AccRatesLogger(
        path=path, move_names=["a", "b"], n_runs=1, flush_interval=1, mode=mode,
    )
    for i in range(n):
        lg.write_entry(
            base_iter + i,
            n_accepted=np.array([1, 2], dtype=np.int64),
            n_proposed=np.array([5, 5], dtype=np.int64),
        )
    lg.close()


def test_acc_rates_logger_mode_w_truncates(tmp_path):
    path = tmp_path / "acc.h5"
    _write_acc(path, mode="w", n=10)
    _write_acc(path, mode="w", n=3, base_iter=100)

    log = AccRatesLogger.read(path)
    assert log.iterations.shape == (3,)
    assert int(log.iterations[0]) == 100


def test_acc_rates_logger_mode_a_appends(tmp_path):
    path = tmp_path / "acc.h5"
    _write_acc(path, mode="w", n=10)
    _write_acc(path, mode="a", n=3, base_iter=100)

    log = AccRatesLogger.read(path)
    assert log.iterations.shape == (13,)
