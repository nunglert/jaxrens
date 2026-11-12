"""Truncate-on-restart: every streamed artifact rewinds to the checkpoint.

On resume from a checkpoint at global iteration ``M``, each writer opened in
append mode must drop records with iteration label ``>= M`` (whatever the
previous process flushed past the last checkpoint), then continue appending.
The boundary is "keep iteration < M".
"""

from __future__ import annotations

import h5py
import jax
import numpy as np
import pytest
from ase.io import read as ase_read

import jaxrens.sampling.moves.random_walk as rw_mod
from jaxrens.backends.toy import create_harmonic
from jaxrens.init.restart import load_restart
from jaxrens.io.adaptation_log import AdaptationLogger
from jaxrens.io.checkpoint import save_checkpoint
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.restart_truncate import (
    truncate_energies,
    truncate_extxyz,
    truncate_h5_iterations,
    truncate_h5_traj,
)
from jaxrens.io.trajectory import ExtxyzTrajectoryWriter, H5TrajectoryWriter
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns
from jaxrens.sampling.termination import IterationTermination


class TestTruncateEnergies:
    """Check that truncate_energies keeps the header and the rows with
    iteration < restart_iteration."""

    def _write(self, path, n_rows):
        log = EnergyLogger(path, n_walkers=4, n_atoms=2, mode="w")
        for i in range(n_rows):
            log.write_entry(i, energy=-float(i), volume=float(i))
        log.close()

    def test_drops_rows_at_and_after_cut(self, tmp_path):
        p = tmp_path / "run.energies"
        self._write(p, 10)  # labels 0..9
        truncate_energies(p, restart_iteration=6)  # keep 0..5
        log = EnergyLogger.read(p)
        assert list(log.iterations) == [0, 1, 2, 3, 4, 5]
        # Header preserved.
        assert log.n_walkers == 4 and log.n_atoms == 2

    def test_noop_when_fresh(self, tmp_path):
        p = tmp_path / "run.energies"
        self._write(p, 5)
        truncate_energies(p, restart_iteration=0)  # fresh run: no-op
        assert len(EnergyLogger.read(p).iterations) == 5

    def test_append_after_truncate_is_continuous(self, tmp_path):
        """Full round-trip: write 0..9, crash, rewind to 6, append 6..8."""
        p = tmp_path / "run.energies"
        self._write(p, 10)
        # Resume: mode="a", restart_iteration=6 truncates then appends.
        log = EnergyLogger(
            p, n_walkers=4, n_atoms=2, mode="a", restart_iteration=6
        )
        for i in (6, 7, 8):
            log.write_entry(i, energy=-float(i), volume=float(i))
        log.close()
        out = EnergyLogger.read(p)
        # No duplicates: a clean 0..8 sequence.
        assert list(out.iterations) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


class TestTruncateExtxyz:
    """Check that extxyz frames with iter >= restart_iteration are dropped,
    directly and when the writer reopens in append mode."""

    def _write(self, path, n_frames):
        w = ExtxyzTrajectoryWriter(path, symbol_map={0: "H"}, mode="w")
        for i in range(n_frames):
            walker = {
                "positions": np.zeros((2, 3), dtype=np.float32),
                "types": np.zeros(2, dtype=np.int32),
                "energy": -float(i),
            }
            w.write_dead_point(i, walker, -float(i))
        w.close()

    def test_drops_frames_at_and_after_cut(self, tmp_path):
        p = tmp_path / "run.traj.extxyz"
        self._write(p, 8)
        truncate_extxyz(p, restart_iteration=5)
        frames = ase_read(str(p), index=":")
        iters = [int(a.info["iter"]) for a in frames]
        assert iters == [0, 1, 2, 3, 4]

    def test_file_removed_when_no_frame_kept(self, tmp_path):
        p = tmp_path / "run.traj.extxyz"
        w = ExtxyzTrajectoryWriter(p, symbol_map={0: "H"}, mode="w")
        for i in (3, 4, 5):
            w.write_dead_point(i, _walker(i), -float(i))
        w.close()
        truncate_extxyz(p, restart_iteration=2)
        assert not p.exists()

    def test_writer_truncates_on_append_open(self, tmp_path):
        p = tmp_path / "run.traj.extxyz"
        self._write(p, 8)
        # Resume: append-mode construction with restart_iteration=5 rewinds.
        w = ExtxyzTrajectoryWriter(
            p,
            symbol_map={0: "H"},
            mode="a",
            restart_iteration=5,
        )
        for i in (5, 6):
            walker = {
                "positions": np.zeros((2, 3), dtype=np.float32),
                "types": np.zeros(2, dtype=np.int32),
                "energy": -float(i),
            }
            w.write_dead_point(i, walker, -float(i))
        w.close()
        iters = [int(a.info["iter"]) for a in ase_read(str(p), index=":")]
        assert iters == [0, 1, 2, 3, 4, 5, 6]


class TestTruncateH5Iterations:
    """Check that truncate_h5_iterations shrinks every per-iteration
    dataset of an HDF5 log, subgroups included."""

    def _write_adaptation(self, path, n_rows):
        log = AdaptationLogger(path, move_names=["a", "b"], n_runs=1, mode="w")
        for i in range(n_rows):
            log.write_entry(
                iteration=i,
                step_sizes=np.array([0.1, 0.2], dtype=np.float32),
                acceptance_rates=np.array([0.5, 0.5], dtype=np.float32),
            )
        log.close()

    def test_resizes_all_datasets(self, tmp_path):
        p = tmp_path / "run.adaptation.h5"
        self._write_adaptation(p, 10)
        truncate_h5_iterations(p, restart_iteration=4)
        log = AdaptationLogger.read(p)
        assert list(log.iterations) == [0, 1, 2, 3]
        assert log.step_sizes.shape[0] == 4
        assert log.acceptance_rates.shape[0] == 4

    def test_append_after_truncate_continuous(self, tmp_path):
        p = tmp_path / "run.adaptation.h5"
        self._write_adaptation(p, 10)
        log = AdaptationLogger(
            p,
            move_names=["a", "b"],
            n_runs=1,
            mode="a",
            restart_iteration=4,
        )
        for i in (4, 5):
            log.write_entry(
                iteration=i,
                step_sizes=np.array([0.1, 0.2], dtype=np.float32),
                acceptance_rates=np.array([0.5, 0.5], dtype=np.float32),
            )
        log.close()
        assert list(AdaptationLogger.read(p).iterations) == [0, 1, 2, 3, 4, 5]

    def test_truncates_subgroup_datasets(self, tmp_path):
        """The adjustment_stats/ datasets share the leading axis and shrink
        too."""
        p = tmp_path / "run.adaptation.h5"
        log = AdaptationLogger(p, move_names=["a"], n_runs=1, mode="w")
        for i in range(6):
            log.write_entry(
                iteration=i,
                step_sizes=np.array([0.1], dtype=np.float32),
                acceptance_rates=np.array([0.5], dtype=np.float32),
                adjustment_stats={
                    "n_rounds": np.array([2], dtype=np.int32),
                    "converged": np.array([True]),
                    "cap_hits": np.array([0], dtype=np.int32),
                    "floor_hits": np.array([0], dtype=np.int32),
                    "bracket_detected": np.array([False]),
                    "reject_reason_counts": np.zeros((1, 4), dtype=np.int32),
                },
            )
        log.close()
        truncate_h5_iterations(p, restart_iteration=3)
        with h5py.File(p, "r") as f:
            assert f["iterations"].shape[0] == 3
            assert f["adjustment_stats"]["n_rounds"].shape[0] == 3
            assert f["adjustment_stats"]["reject_reason_counts"].shape[0] == 3


def _make_checkpoint(tmp_path, n_dead, n_walkers=4, n_atoms=1):
    rng = np.random.default_rng(0)
    state = {
        "positions": rng.uniform(-2, 2, (n_walkers, n_atoms, 3)).astype(
            np.float32
        ),
        "types": np.zeros((n_walkers, n_atoms), dtype=np.int32),
        "energies": rng.uniform(1, 10, n_walkers).astype(np.float32),
        "cells": np.stack([np.eye(3, dtype=np.float32) * 6.0] * n_walkers),
        "dead_energies": rng.uniform(10, 20, n_dead).astype(np.float32),
        "dead_positions": rng.uniform(-2, 2, (n_dead, n_atoms, 3)).astype(
            np.float32
        ),
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": -7.3,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(1),
    }
    p = tmp_path / "ns.checkpoint.h5"
    save_checkpoint(p, state, symbol_map={0: "Si"})
    return p


def test_dead_point_labels_continue_from_checkpoint(tmp_path):
    """After a restart the loop passes callbacks the global iteration,
    continuing from the checkpoint, not a counter that restarts at 0.
    Truncation needs this to have a stable cut point."""
    n_dead = 5
    n_extra = 3
    walker_set, bundle = load_restart(_make_checkpoint(tmp_path, n_dead))

    backend = create_harmonic()
    desc = MoveKernel(
        name="random_walk",
        build_kernel=rw_mod.build_kernel,
        step_size=0.3,
        weight=1.0,
        kernel_kwargs={},
        extra_state_fields={},
    )
    init_fn, step_fn, _ = build_mwg(backend, [desc])

    # Energies are re-evaluated on restart (WalkerSet carries no energy).
    types0 = walker_set.types[0]
    energies = jax.vmap(lambda p, c: backend(p, types0, c, 0)[0])(
        walker_set.positions, walker_set.cells
    )

    seen: list[int] = []

    class _Capture:
        def on_iteration(self, iteration, ns_state, info):
            seen.append(int(iteration))

    run_ns(
        positions=walker_set.positions,
        types=walker_set.types,
        energies=energies,
        cells=walker_set.cells,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_key=jax.random.key(11),
        max_iterations=n_extra,
        n_mcmc_steps=3,
        termination_criteria=[IterationTermination(n_extra)],
        callbacks=[_Capture()],
        restart_state=bundle,
    )

    # Labels continue from the checkpoint (5, 6, 7), not reset to 0.
    assert seen == [n_dead, n_dead + 1, n_dead + 2]


def _walker(i):
    return {
        "positions": np.zeros((2, 3), dtype=np.float32),
        "types": np.zeros(2, dtype=np.int32),
        "energy": -float(i),
    }


_SNAPSHOT = {
    "positions": np.zeros((1, 2, 3), dtype=np.float32),
    "types": np.zeros((1, 2), dtype=np.int32),
    "energies": np.zeros(1, dtype=np.float32),
    "cells": None,
}


class TestTruncateH5Traj:
    """Check that dead-point and snapshot groups at and after the cut are
    deleted, and that a writer reopened for a restart can write them
    again."""

    def test_deletes_groups_at_and_after_cut(self, tmp_path):
        p = tmp_path / "run.traj.h5"
        w = H5TrajectoryWriter(p, symbol_map={0: "H"}, mode="w")
        for i in range(6):
            w.write_dead_point(i, _walker(i), -float(i))
            if i in (2, 5):
                w.write_walker_snapshot(i, _SNAPSHOT)
        w.close()
        truncate_h5_traj(p, restart_iteration=4)
        with h5py.File(p, "r") as f:
            keys = sorted(f.keys())
        assert keys == ["0", "1", "2", "3", "snapshot_2"]

    def test_writer_resume_rewrites_groups(self, tmp_path):
        """Checkpoint at 4, crash after 5: the resumed writer rewinds and
        writes dead points 4-5 and snapshot 5 again without a name
        clash."""
        p = tmp_path / "run.traj.h5"
        w = H5TrajectoryWriter(p, symbol_map={0: "H"}, mode="w")
        for i in range(6):
            w.write_dead_point(i, _walker(i), -float(i))
        w.write_walker_snapshot(5, _SNAPSHOT)
        w.close()

        w = H5TrajectoryWriter(
            p, symbol_map={0: "H"}, mode="a", restart_iteration=4
        )
        for i in (4, 5):
            w.write_dead_point(i, _walker(i), -float(i))
        w.write_walker_snapshot(5, _SNAPSHOT)
        w.close()
        with h5py.File(p, "r") as f:
            keys = sorted(f.keys())
        assert keys == ["0", "1", "2", "3", "4", "5", "snapshot_5"]


@pytest.mark.parametrize(
    "truncate",
    [
        truncate_energies,
        truncate_extxyz,
        truncate_h5_iterations,
        truncate_h5_traj,
    ],
)
def test_missing_file_is_noop(tmp_path, truncate):
    truncate(tmp_path / "missing", restart_iteration=3)
    assert list(tmp_path.iterdir()) == []
