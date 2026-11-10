"""Tests for the dead-point output when ``n_cull > 1``.

The trajectory callbacks write one dead point and one ``.energies`` row per
culled walker, in descending energy. The HDF5 writer keys them
``"{iter}_{rank}"`` (``"{iter}"`` with a single cull), restart truncation
handles both keys, and the ``.energies`` header stores ``n_cull``.
"""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.cli.monitor import BatchedTrajectoryCallback, TrajectoryCallback
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.restart_truncate import truncate_h5_traj
from jaxrens.io.trajectory import create_trajectory_writer
from jaxrens.sampling.batch_descriptor import VmapRuns
from jaxrens.state.walker import WalkerState


class _RecWriter:
    """Trajectory writer that records each dead point it is given."""

    def __init__(self):
        self.rows = []

    def write_dead_point(self, iteration, walker, energy, cull_rank=None):
        self.rows.append((iteration, float(energy), cull_rank, walker))

    def write_walker_snapshot(self, iteration, walkers):
        pass

    def close(self):
        pass


def _dead_walkers(energies, n_atoms=2, prefix=()):
    energies = jnp.asarray(energies, dtype=jnp.float32)
    shape = prefix + energies.shape[len(prefix) :]
    return WalkerState(
        positions=jnp.zeros(shape + (n_atoms, 3)),
        types=jnp.zeros(shape + (n_atoms,), dtype=jnp.int32),
        energy=energies,
        cell=jnp.broadcast_to(2.0 * jnp.eye(3), shape + (3, 3)),
    )


def test_single_run_writes_every_cull(tmp_path):
    writer = _RecWriter()
    elog = EnergyLogger(tmp_path / "ns.energies", n_walkers=10, n_cull=3)
    cb = TrajectoryCallback(writer, energy_logger=elog, snapshot_interval=0)
    dw = _dead_walkers([5.0, 4.0, 3.5])
    cb.on_iteration(7, None, {"dead_walkers": dw, "emax": dw.energy[-1]})
    elog.close()

    assert [(r[0], r[1], r[2]) for r in writer.rows] == [
        (7, 5.0, 0),
        (7, 4.0, 1),
        (7, 3.5, 2),
    ]
    log = EnergyLogger.read(tmp_path / "ns.energies")
    assert log.n_cull == 3
    assert log.iterations.tolist() == [7, 7, 7]
    assert log.energies.tolist() == [5.0, 4.0, 3.5]
    np.testing.assert_allclose(log.volumes, 8.0)


def test_traj_interval_skips_writer_but_logs_every_energy(tmp_path):
    """Off a ``traj_interval`` multiple, ``.energies`` still gets all culls.

    NS analysis needs every dead point, so only the trajectory writer is
    gated.
    """
    writer = _RecWriter()
    elog = EnergyLogger(tmp_path / "ns.energies", n_walkers=10, n_cull=3)
    cb = TrajectoryCallback(
        writer, energy_logger=elog, traj_interval=2, snapshot_interval=0
    )
    cb.on_iteration(3, None, {"dead_walkers": _dead_walkers([5.0, 4.0, 3.5])})
    elog.close()

    assert writer.rows == []
    log = EnergyLogger.read(tmp_path / "ns.energies")
    assert log.energies.tolist() == [5.0, 4.0, 3.5]


def test_single_run_legacy_info_unchanged(tmp_path):
    """An info with only ``dead_walker`` (n_cull = 1) writes one row
    without a cull rank and records ``emax``.
    """
    writer = _RecWriter()
    cb = TrajectoryCallback(writer, snapshot_interval=0)
    dw = _dead_walkers(2.5)
    cb.on_iteration(3, None, {"dead_walker": dw, "emax": jnp.float32(2.7)})
    assert [(r[0], r[1], r[2]) for r in writer.rows] == [
        (3, pytest.approx(2.7), None)
    ]


def test_batched_writes_every_cull_per_replica(tmp_path):
    n_runs, c = 2, 2
    writers = [_RecWriter() for _ in range(n_runs)]
    elogs = [
        EnergyLogger(tmp_path / f"r{r}.energies", n_walkers=10, n_cull=c)
        for r in range(n_runs)
    ]
    cb = BatchedTrajectoryCallback(writers, elogs, snapshot_interval=0)
    e = jnp.array([[9.0, 8.0], [7.0, 6.0]])
    dw = _dead_walkers(e, prefix=(n_runs,))
    info = {
        "dead_walkers": dw,
        "emax": e[:, -1],
        "_batcher": VmapRuns(n_runs=n_runs),
    }
    cb.on_iteration(4, SimpleNamespace(population=None), info)
    for el in elogs:
        el.close()

    assert [(r[1], r[2]) for r in writers[0].rows] == [(9.0, 0), (8.0, 1)]
    assert [(r[1], r[2]) for r in writers[1].rows] == [(7.0, 0), (6.0, 1)]
    assert EnergyLogger.read(tmp_path / "r1.energies").energies.tolist() == [
        7.0,
        6.0,
    ]


def test_h5_writer_keys_and_truncation(tmp_path):
    path = tmp_path / "ns.traj.h5"
    w = create_trajectory_writer("h5", path, {0: "C"})
    one = WalkerState(
        positions=jnp.zeros((2, 3)),
        types=jnp.zeros((2,), dtype=jnp.int32),
        energy=jnp.float32(1.0),
        cell=2.0 * jnp.eye(3),
    )
    for it in range(3):
        for k in range(2):
            w.write_dead_point(it, one, 1.0, cull_rank=k)
    w.write_dead_point(3, one, 1.0)  # no cull_rank: key "3"
    w.close()

    with h5py.File(path, "r") as f:
        keys = set(k for k in f.keys() if not k.startswith("snapshot_"))
        assert {"0_0", "0_1", "2_1", "3"} <= keys
        assert int(f["2_1"].attrs["cull_rank"]) == 1
        assert int(f["2_1"].attrs["iteration"]) == 2

    truncate_h5_traj(path, 2)
    with h5py.File(path, "r") as f:
        keys = set(k for k in f.keys() if not k.startswith("snapshot_"))
    assert keys == {"0_0", "0_1", "1_0", "1_1"}
