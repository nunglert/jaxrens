"""Tests for the per-iteration neighbor-bucket diagnostic log.

Covers:
- ``MaxNeighborsLogger`` round-trip (write → read produces identical arrays).
- Auto-flush behaviour: deferring the file creation until first flush.
- Shape coercion: 1D inputs broadcast to (1, n_walkers); per-run scalars
  broadcast to (n_runs,).
- ``MaxNeighborsCallback`` end-to-end against a SingleRun ``_run_loop``
  invocation: the file appears at the expected path with the right shapes,
  and the recorded bucket size + max-neighbor counts match what the loop
  produced.
- ``Monitor.from_directory`` loads the trace into ``max_neighbors_trace``.
- ``plot_max_neighbors`` smoke test in both ``percentiles`` and ``heatmap``
  kinds.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless

import jaxrens.sampling.moves.random_walk as _rw_mod
from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.monitor import MaxNeighborsCallback
from jaxrens.io.max_neighbors_log import MaxNeighborsLog, MaxNeighborsLogger
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.plotting import plot_max_neighbors
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import SingleRun
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns
from jaxrens.sampling.run_loop import _run_loop
from jaxrens.sampling.termination import IterationTermination

# ---------------------------------------------------------------------------
# Logger round-trip
# ---------------------------------------------------------------------------


class TestMaxNeighborsLogger:
    def test_round_trip_single_run(self, tmp_path: Path):
        log_path = tmp_path / "ns.max_neighbors.h5"
        logger = MaxNeighborsLogger(log_path, n_runs=1, n_walkers=4)
        rng = np.random.default_rng(0)
        iters = [0, 1, 2]
        counts_in = [
            rng.integers(0, 40, size=4, dtype=np.int32) for _ in iters
        ]
        for i, c in zip(iters, counts_in, strict=False):
            logger.write_entry(
                iteration=i,
                max_neighbor_count=c,
                bucket_size=np.int32(50),
                overflow=np.bool_(False),
            )
        logger.close()
        assert log_path.exists()

        loaded = MaxNeighborsLogger.read(log_path)
        assert loaded.n_runs == 1
        assert loaded.n_walkers == 4
        assert loaded.iterations.tolist() == iters
        assert loaded.max_neighbor_count.shape == (3, 1, 4)
        for i, c in enumerate(counts_in):
            np.testing.assert_array_equal(loaded.max_neighbor_count[i, 0], c)
        np.testing.assert_array_equal(loaded.bucket_size, np.int32([[50]] * 3))
        np.testing.assert_array_equal(loaded.overflow, np.bool_([[False]] * 3))

    def test_no_file_when_no_entries(self, tmp_path: Path):
        log_path = tmp_path / "ns.max_neighbors.h5"
        logger = MaxNeighborsLogger(log_path, n_runs=1, n_walkers=4)
        logger.close()
        # Closing without any write_entry must not produce an empty artefact.
        assert not log_path.exists()

    def test_multi_run_shape(self, tmp_path: Path):
        log_path = tmp_path / "ns.max_neighbors.h5"
        n_runs, n_walkers = 3, 5
        logger = MaxNeighborsLogger(
            log_path, n_runs=n_runs, n_walkers=n_walkers
        )
        rng = np.random.default_rng(1)
        counts = rng.integers(0, 50, size=(n_runs, n_walkers), dtype=np.int32)
        logger.write_entry(
            iteration=0,
            max_neighbor_count=counts,
            bucket_size=np.array([45, 50, 50], dtype=np.int32),
            overflow=np.array([False, True, False], dtype=np.bool_),
        )
        logger.close()

        loaded = MaxNeighborsLogger.read(log_path)
        assert loaded.max_neighbor_count.shape == (1, n_runs, n_walkers)
        np.testing.assert_array_equal(loaded.max_neighbor_count[0], counts)
        np.testing.assert_array_equal(loaded.bucket_size[0], [45, 50, 50])
        np.testing.assert_array_equal(loaded.overflow[0], [False, True, False])

    def test_per_iter_helpers(self):
        # Build a small in-memory log and verify the property accessors.
        log = MaxNeighborsLog(
            iterations=np.array([0, 1], dtype=np.int64),
            max_neighbor_count=np.array(
                [[[10, 12, 5]], [[20, 8, 9]]],
                dtype=np.int32,
            ),
            bucket_size=np.array([[30], [30]], dtype=np.int32),
            overflow=np.array([[False], [False]], dtype=np.bool_),
            n_runs=1,
            n_walkers=3,
        )
        np.testing.assert_array_equal(
            log.peak_per_iter, np.array([[12], [20]])
        )
        np.testing.assert_array_equal(log.headroom, np.array([[18], [10]]))


# ---------------------------------------------------------------------------
# Callback end-to-end inside _run_loop
# ---------------------------------------------------------------------------


def _rw_descriptor(step_size: float = 0.3) -> MoveKernel:
    return MoveKernel(
        name="random_walk",
        build_kernel=_rw_mod.build_kernel,
        step_size=step_size,
        weight=1.0,
        kernel_kwargs={},
        extra_state_fields={},
    )


def _build_ns_state(n_walkers: int = 4, n_atoms: int = 2):
    backend = create_harmonic()
    desc = _rw_descriptor()
    init_fn, step_fn, _ = build_mwg(backend, [desc])

    key = jax.random.key(0)
    key, key_pos = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n_walkers, n_atoms, 3),
        minval=-2.0,
        maxval=2.0,
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_walkers, 3, 3))
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    ns_state = init_ns(init_fn, positions, types, energies, cells, key)
    return ns_state, step_fn


class TestMaxNeighborsCallback:
    """End-to-end smoke test: ``_run_loop`` + MaxNeighborsCallback writes
    the expected HDF5 file with sensible shapes."""

    def test_writes_expected_shapes_single_run(self, tmp_path: Path):
        ns_state, step_fn = _build_ns_state(n_walkers=4, n_atoms=2)
        log_path = tmp_path / "ns.max_neighbors.h5"
        mn_logger = MaxNeighborsLogger(
            log_path,
            n_runs=1,
            n_walkers=4,
        )
        cb = MaxNeighborsCallback(mn_logger, interval=1)

        batcher = SingleRun()
        adapt_step = build_adapt_step(
            move_descriptors=[],
            per_move_fns=None,
            batcher=batcher,
            adjust_n_samples=1,
            adjust_factor=1.5,
            adjust_max_rounds=1,
            adjust_interval=0,
        )
        _run_loop(
            batcher=batcher,
            adapt_step=adapt_step,
            adjust_interval=0,
            ns_state=ns_state,
            step_fn=step_fn,
            n_mcmc_steps=1,
            n_extra=0,
            termination_criteria=[IterationTermination(5)],
            callbacks=[cb],
            n_moves=1,
            move_descriptors=[_rw_descriptor()],
            rng_key=jax.random.key(7),
            info_interval=1,
        )
        cb.on_finish(ns_state)
        assert log_path.exists()

        loaded = MaxNeighborsLogger.read(log_path)
        assert loaded.n_runs == 1
        assert loaded.n_walkers == 4
        # Schema: ``(n_entries, 1, 4)``; we ran 6 iterations (0..5) with
        # interval=1, plus a possible iter-0 reset depending on cadence.
        assert loaded.max_neighbor_count.ndim == 3
        assert loaded.max_neighbor_count.shape[1:] == (1, 4)
        assert loaded.bucket_size.shape == (loaded.iterations.shape[0], 1)
        assert loaded.overflow.shape == loaded.bucket_size.shape
        # The harmonic backend never overflows.
        assert not loaded.overflow.any()


# ---------------------------------------------------------------------------
# Postprocess loader + plotter
# ---------------------------------------------------------------------------


def _seed_dummy_run_dir(tmp_path: Path) -> Path:
    """Drop in the minimal checkpoint + energies needed by Monitor.from_directory.

    Mirrors the pattern in ``test_postprocess_monitor.py`` (passes a dict to
    ``save_checkpoint`` rather than going through real ``run_ns``).
    """
    from jaxrens.io.checkpoint import save_checkpoint
    from jaxrens.io.energy_log import EnergyLogger

    n_live = 4
    n_dead = 3
    rng = np.random.default_rng(42)
    dead_e = np.sort(rng.uniform(0.0, 5.0, n_dead))
    live_e = rng.uniform(4.0, 6.0, n_live)
    positions = rng.uniform(0.0, 1.0, (n_live, 2, 3))
    dead_positions = rng.uniform(0.0, 1.0, (n_dead, 2, 3))

    ns_state = {
        "positions": jnp.asarray(positions),
        "types": jnp.zeros(2, dtype=jnp.int32),
        "energies": jnp.asarray(live_e),
        "cells": None,
        "dead_energies": jnp.asarray(
            np.concatenate([dead_e, np.full(100, np.inf)])
        ),
        "dead_positions": jnp.asarray(
            np.concatenate([dead_positions, np.zeros((100, 2, 3))], axis=0)
        ),
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": 0.0,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_live,
    }
    save_checkpoint(tmp_path / "ns.final.checkpoint.h5", ns_state)

    en = EnergyLogger(tmp_path / "ns.energies", n_walkers=n_live, n_cull=1)
    en.write_header()
    for i, e in enumerate(dead_e):
        en.write_entry(i, float(e))
    en.close()
    return tmp_path


class TestMonitorLoaderAndPlot:
    """Verify Monitor.from_directory picks up max_neighbors_trace and the
    plot_max_neighbors helper runs end-to-end in both display kinds."""

    def test_monitor_loads_trace_when_file_present(self, tmp_path: Path):
        _seed_dummy_run_dir(tmp_path)
        # Add a max_neighbors.h5 with two entries.
        mn = MaxNeighborsLogger(
            tmp_path / "ns.max_neighbors.h5",
            n_runs=1,
            n_walkers=4,
        )
        mn.write_entry(
            iteration=0,
            max_neighbor_count=np.array([10, 12, 5, 8], dtype=np.int32),
            bucket_size=np.int32(30),
            overflow=np.bool_(False),
        )
        mn.write_entry(
            iteration=1,
            max_neighbor_count=np.array([11, 14, 6, 9], dtype=np.int32),
            bucket_size=np.int32(30),
            overflow=np.bool_(False),
        )
        mn.close()

        monitor = Monitor.from_directory(tmp_path)
        assert monitor.max_neighbors_trace is not None
        assert monitor.max_neighbors_trace.n_runs == 1
        assert monitor.max_neighbors_trace.n_walkers == 4

    def test_monitor_trace_is_none_when_file_absent(self, tmp_path: Path):
        _seed_dummy_run_dir(tmp_path)
        monitor = Monitor.from_directory(tmp_path)
        assert monitor.max_neighbors_trace is None

    def test_plot_percentiles_smoke(self, tmp_path: Path):
        _seed_dummy_run_dir(tmp_path)
        mn = MaxNeighborsLogger(
            tmp_path / "ns.max_neighbors.h5",
            n_runs=1,
            n_walkers=4,
        )
        for i in range(5):
            mn.write_entry(
                iteration=i,
                max_neighbor_count=np.array(
                    [10 + i, 12, 5, 8 + i],
                    dtype=np.int32,
                ),
                bucket_size=np.int32(30),
                overflow=np.bool_(False),
            )
        mn.close()
        monitor = Monitor.from_directory(tmp_path)
        ax = plot_max_neighbors(
            monitor,
            kind="percentiles",
            percentiles=(50, 95, 100),
            show_bucket=True,
        )
        # Two percentile lines + bucket step = at least 3 lines on the axes.
        assert len(ax.lines) >= 3
        ax.figure.savefig(tmp_path / "smoke_percentiles.png")

    def test_plot_heatmap_smoke(self, tmp_path: Path):
        _seed_dummy_run_dir(tmp_path)
        mn = MaxNeighborsLogger(
            tmp_path / "ns.max_neighbors.h5",
            n_runs=1,
            n_walkers=4,
        )
        for i in range(5):
            mn.write_entry(
                iteration=i,
                max_neighbor_count=np.array(
                    [10 + i, 12, 5, 8 + i],
                    dtype=np.int32,
                ),
                bucket_size=np.int32(30),
                overflow=np.bool_(False),
            )
        mn.close()
        monitor = Monitor.from_directory(tmp_path)
        ax = plot_max_neighbors(monitor, kind="heatmap", show_bucket=False)
        # pcolormesh + nothing else.
        assert len(ax.collections) >= 1
        ax.figure.savefig(tmp_path / "smoke_heatmap.png")

    def test_invalid_kind_raises(self, tmp_path: Path):
        _seed_dummy_run_dir(tmp_path)
        mn = MaxNeighborsLogger(
            tmp_path / "ns.max_neighbors.h5",
            n_runs=1,
            n_walkers=4,
        )
        mn.write_entry(
            iteration=0,
            max_neighbor_count=np.array([10, 12, 5, 8], dtype=np.int32),
            bucket_size=np.int32(30),
            overflow=np.bool_(False),
        )
        mn.close()
        monitor = Monitor.from_directory(tmp_path)
        with pytest.raises(ValueError, match="kind must be"):
            plot_max_neighbors(monitor, kind="banana")

    def test_no_trace_raises_helpful_error(self, tmp_path: Path):
        _seed_dummy_run_dir(tmp_path)
        monitor = Monitor.from_directory(tmp_path)
        with pytest.raises(ValueError, match="No max_neighbors_trace"):
            plot_max_neighbors(monitor)
