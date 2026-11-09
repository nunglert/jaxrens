"""Tests for AdaptationLogger, the step-size adaptation log.

Covers the .adaptation.h5 round-trip for schema v1, v2 (adjustment_stats)
and v3 (evaluation counts), the empty/closed logger, loading the trace in
Monitor.from_directory, the step-size and acceptance-rate plots, and
AdaptationCallback in cli.monitor (PmapVmapRuns flattening and adapter
health warnings).
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")  # headless before any other matplotlib import

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxrens.cli.monitor import AdaptationCallback
from jaxrens.io.adaptation_log import (
    _FLUSH_INTERVAL,
    AdaptationLog,
    AdaptationLogger,
)
from jaxrens.postprocess.collection import MonitorCollection
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.plotting import plot_acceptance_rates, plot_step_sizes
from jaxrens.sampling.batch_descriptor import PmapVmapRuns


def _make_trace(
    n_entries: int = 10,
    n_runs: int = 1,
    n_moves: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (iterations, step_sizes, acceptance_rates, move_names)."""
    rng = np.random.default_rng(seed)
    iters = np.arange(n_entries, dtype=np.int64) * 10
    ss = rng.uniform(0.001, 5.0, (n_entries, n_runs, n_moves)).astype(
        np.float32
    )
    acc = rng.uniform(0.0, 1.0, (n_entries, n_runs, n_moves)).astype(
        np.float32
    )
    names = [f"move_{k}" for k in range(n_moves)]
    return iters, ss, acc, names


def _make_monitor_with_trace(
    n_runs: int = 1,
    n_moves: int = 2,
    n_entries: int = 5,
    label: str = "test",
) -> Monitor:
    """Build a Monitor with a synthetic AdaptationLog attached."""
    rng = np.random.default_rng(42)
    n_dead, n_live = 50, 10
    iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)
    trace = AdaptationLog(
        iterations=iters,
        step_sizes=ss,
        acceptance_rates=acc,
        move_names=names,
        n_runs=n_runs,
        n_moves=n_moves,
    )
    return Monitor(
        dead_energies=np.sort(rng.uniform(0, 10, n_dead)),
        dead_volumes=None,
        live_energies=rng.uniform(8, 12, n_live),
        live_volumes=None,
        log_evidence=42.0,
        iteration=n_dead,
        n_live=n_live,
        n_cull=1,
        label=label,
        adaptation_trace=trace,
    )


class TestAdaptationLoggerRoundTrip:
    """Check that step sizes and acceptance rates read back unchanged."""

    def test_round_trip_10_entries_single_run(self, tmp_path):
        n_entries, n_runs, n_moves = 10, 1, 3
        iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)
        path = tmp_path / "adapt.h5"

        logger = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            logger.write_entry(
                int(iters[idx]),
                ss[idx],  # (1, n_moves)
                acc[idx],  # (1, n_moves)
            )
        logger.close()

        assert path.exists()
        loaded = AdaptationLogger.read(path)

        assert loaded.n_runs == n_runs
        assert loaded.n_moves == n_moves
        assert loaded.move_names == names
        assert loaded.iterations.shape == (n_entries,)
        assert loaded.step_sizes.shape == (n_entries, n_runs, n_moves)
        assert loaded.acceptance_rates.shape == (n_entries, n_runs, n_moves)

        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_allclose(loaded.step_sizes, ss, rtol=1e-5)
        np.testing.assert_allclose(loaded.acceptance_rates, acc, rtol=1e-5)

    def test_round_trip_3_runs_4_moves(self, tmp_path):
        n_entries, n_runs, n_moves = 8, 3, 4
        iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)
        path = tmp_path / "adapt_batched.h5"

        logger = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            logger.write_entry(int(iters[idx]), ss[idx], acc[idx])
        logger.close()

        loaded = AdaptationLogger.read(path)
        assert loaded.step_sizes.shape == (n_entries, n_runs, n_moves)
        np.testing.assert_allclose(loaded.step_sizes, ss, rtol=1e-5)

    def test_1d_input_coerced_to_2d(self, tmp_path):
        """A (n_moves,) input is stored as (1, n_moves)."""
        n_moves = 2
        names = ["a", "b"]
        path = tmp_path / "coerce.h5"
        logger = AdaptationLogger(path=path, move_names=names, n_runs=1)
        ss_1d = np.array([0.5, 1.2], dtype=np.float32)
        acc_1d = np.array([0.6, 0.4], dtype=np.float32)
        logger.write_entry(0, ss_1d, acc_1d)
        logger.close()

        loaded = AdaptationLogger.read(path)
        assert loaded.step_sizes.shape == (1, 1, n_moves)
        np.testing.assert_allclose(loaded.step_sizes[0, 0], ss_1d, rtol=1e-5)

    def test_each_entry_flushed_before_close(self, tmp_path):
        """_FLUSH_INTERVAL is 1, so every entry is on disk before close()
        and a crashed run keeps its full trace."""
        assert _FLUSH_INTERVAL == 1
        n_entries = 6
        path = tmp_path / "flush.h5"
        logger = AdaptationLogger(path=path, move_names=["x", "y"], n_runs=1)
        for idx in range(n_entries):
            ss = np.array([[float(idx), float(idx) * 2]], dtype=np.float32)
            acc = np.array([[0.5, 0.5]], dtype=np.float32)
            logger.write_entry(idx, ss, acc)

        loaded = AdaptationLogger.read(path)
        np.testing.assert_array_equal(loaded.iterations, np.arange(n_entries))
        np.testing.assert_allclose(
            loaded.step_sizes[:, 0, 1], 2.0 * np.arange(n_entries)
        )
        logger.close()

    def test_1d_stats_and_counts_coerced(self, tmp_path):
        """(n_moves,) stats and counts and (n_moves, 4) reject counts gain
        a leading n_runs=1 axis."""
        path = tmp_path / "coerce_stats.h5"
        logger = AdaptationLogger(path=path, move_names=["a", "b"], n_runs=1)
        stats = {
            "n_rounds": np.array([1, 2]),
            "converged": np.array([True, False]),
            "cap_hits": np.array([0, 1]),
            "floor_hits": np.array([0, 0]),
            "bracket_detected": np.array([True, True]),
            "reject_reason_counts": np.arange(8).reshape(2, 4),
        }
        logger.write_entry(
            0,
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            adjustment_stats=stats,
            n_evaluations=np.array([3, 4]),
            n_grad_evaluations=np.array([1, 2]),
        )
        logger.close()

        loaded = AdaptationLogger.read(path)
        adj = loaded.adjustment_stats
        assert adj["n_rounds"].shape == (1, 1, 2)
        np.testing.assert_array_equal(adj["n_rounds"][0, 0], [1, 2])
        assert adj["reject_reason_counts"].shape == (1, 1, 2, 4)
        np.testing.assert_array_equal(
            adj["reject_reason_counts"][0, 0], np.arange(8).reshape(2, 4)
        )
        np.testing.assert_array_equal(loaded.n_evaluations[0, 0], [3, 4])
        np.testing.assert_array_equal(loaded.n_grad_evaluations[0, 0], [1, 2])

    @pytest.mark.parametrize(
        "with_stats, with_counts, version",
        [(False, False, None), (True, False, 2), (True, True, 3)],
    )
    def test_schema_version_attr(
        self, tmp_path, with_stats, with_counts, version
    ):
        """The schema version attr is 2 with adjustment_stats, 3 with
        evaluation counts, and absent with neither (v1)."""
        iters, ss, acc, names = _make_trace(1, 1, 2)
        kwargs = {}
        if with_stats:
            stats = _make_adjustment_stats(1, 1, 2)
            kwargs["adjustment_stats"] = {k: v[0] for k, v in stats.items()}
        if with_counts:
            kwargs["n_evaluations"] = np.array([[3, 4]])
            kwargs["n_grad_evaluations"] = np.array([[1, 2]])
        path = tmp_path / "version.h5"
        logger = AdaptationLogger(path=path, move_names=names, n_runs=1)
        logger.write_entry(int(iters[0]), ss[0], acc[0], **kwargs)
        logger.close()

        with h5py.File(path, "r") as f:
            assert f.attrs.get("adaptation_log_schema_version") == version


class TestAdaptationLoggerEmpty:
    """Check the logger lifecycle: no file without entries, no writes
    after close."""

    def test_empty_logger_creates_no_file(self, tmp_path):
        path = tmp_path / "empty.h5"
        logger = AdaptationLogger(path=path, move_names=["m0"], n_runs=1)
        logger.close()
        assert not path.exists()

    def test_write_after_close_raises(self, tmp_path):
        path = tmp_path / "closed.h5"
        logger = AdaptationLogger(path=path, move_names=["m0"], n_runs=1)
        logger.close()
        with pytest.raises(RuntimeError, match="closed"):
            logger.write_entry(0, np.array([[0.5]]), np.array([[0.5]]))


@pytest.mark.usefixtures("minimal_run_dir")
class TestMonitorFromDirectoryAdaptation:
    """Check that Monitor.from_directory loads <prefix>.adaptation.h5 into
    adaptation_trace when the file exists."""

    def test_loads_adaptation_trace_when_present(self, tmp_path):
        prefix = "ns"

        names = ["galilean", "volume"]
        adapt_path = tmp_path / f"{prefix}.adaptation.h5"
        log = AdaptationLogger(path=adapt_path, move_names=names, n_runs=1)
        rng = np.random.default_rng(0)
        for i in range(5):
            ss = rng.uniform(0.01, 1.0, (1, 2)).astype(np.float32)
            acc = rng.uniform(0.1, 0.9, (1, 2)).astype(np.float32)
            log.write_entry(i * 10, ss, acc)
        log.close()

        m = Monitor.from_directory(tmp_path, prefix=prefix)
        assert m.adaptation_trace is not None
        assert isinstance(m.adaptation_trace, AdaptationLog)
        assert m.adaptation_trace.n_moves == 2
        assert m.adaptation_trace.move_names == names
        assert m.adaptation_trace.iterations.shape == (5,)

    def test_adaptation_trace_none_when_missing(self, tmp_path):
        prefix = "ns"
        m = Monitor.from_directory(tmp_path, prefix=prefix)
        assert m.adaptation_trace is None


class TestPlotStepSizes:
    """Check the lines drawn by plot_step_sizes and its errors."""

    def test_per_run_true_draws_n_moves_lines_for_single_run(self):
        n_moves = 3
        m = _make_monitor_with_trace(n_runs=1, n_moves=n_moves, n_entries=6)
        ax = plot_step_sizes(m, per_run=True)
        assert len(ax.get_lines()) == n_moves
        plt.close("all")

    def test_per_run_true_draws_n_runs_times_n_moves_lines(self):
        n_runs, n_moves = 3, 2
        m = _make_monitor_with_trace(
            n_runs=n_runs, n_moves=n_moves, n_entries=5
        )
        ax = plot_step_sizes(m, per_run=True)
        assert len(ax.get_lines()) == n_runs * n_moves
        plt.close("all")

    def test_per_run_false_mean_plus_fill(self):
        """per_run=False draws one mean line per move with a fill_between
        band."""
        n_runs, n_moves = 3, 2
        m = _make_monitor_with_trace(
            n_runs=n_runs, n_moves=n_moves, n_entries=5
        )
        ax = plot_step_sizes(m, per_run=False)
        assert len(ax.get_lines()) == n_moves
        assert len(ax.collections) == n_moves
        plt.close("all")

    def test_raises_when_no_trace(self):
        rng = np.random.default_rng(0)
        m = Monitor(
            dead_energies=rng.uniform(0, 10, 20),
            dead_volumes=None,
            live_energies=rng.uniform(8, 12, 5),
            live_volumes=None,
            log_evidence=0.0,
            iteration=20,
            n_live=5,
            n_cull=1,
        )
        with pytest.raises(ValueError, match="adaptation_trace"):
            plot_step_sizes(m)

    def test_reuses_existing_ax(self):
        m = _make_monitor_with_trace()
        _, existing_ax = plt.subplots()
        returned = plot_step_sizes(m, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")


class TestPlotAcceptanceRates:
    """Check the lines drawn by plot_acceptance_rates and its errors."""

    def test_per_run_true_3_runs_2_moves(self):
        n_runs, n_moves = 3, 2
        m = _make_monitor_with_trace(
            n_runs=n_runs, n_moves=n_moves, n_entries=5
        )
        ax = plot_acceptance_rates(m, per_run=True)
        assert len(ax.get_lines()) == n_runs * n_moves
        plt.close("all")

    def test_raises_when_no_trace(self):
        rng = np.random.default_rng(0)
        m = Monitor(
            dead_energies=rng.uniform(0, 10, 20),
            dead_volumes=None,
            live_energies=rng.uniform(8, 12, 5),
            live_volumes=None,
            log_evidence=0.0,
            iteration=20,
            n_live=5,
            n_cull=1,
        )
        with pytest.raises(ValueError, match="adaptation_trace"):
            plot_acceptance_rates(m)

    def test_reuses_existing_ax(self):
        m = _make_monitor_with_trace()
        _, existing_ax = plt.subplots()
        returned = plot_acceptance_rates(m, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")


class TestMonitorCollectionAdaptationPlots:
    """Check that MonitorCollection.plot_step_sizes skips monitors without
    an adaptation trace."""

    def test_collection_skips_monitors_without_trace(self):
        m_with = _make_monitor_with_trace(n_moves=2, n_entries=4, label="with")
        rng = np.random.default_rng(0)
        m_without = Monitor(
            dead_energies=rng.uniform(0, 10, 20),
            dead_volumes=None,
            live_energies=rng.uniform(8, 12, 5),
            live_volumes=None,
            log_evidence=0.0,
            iteration=20,
            n_live=5,
            n_cull=1,
            label="without",
        )
        coll = MonitorCollection([m_with, m_without])
        ax = coll.plot_step_sizes()
        # Only the 2 moves of the monitor with a trace are drawn.
        assert len(ax.get_lines()) == 2
        plt.close("all")

    def test_collection_returns_none_ax_when_all_traces_missing(self):
        rng = np.random.default_rng(0)

        def _bare():
            return Monitor(
                dead_energies=rng.uniform(0, 10, 20),
                dead_volumes=None,
                live_energies=rng.uniform(8, 12, 5),
                live_volumes=None,
                log_evidence=0.0,
                iteration=20,
                n_live=5,
                n_cull=1,
            )

        coll = MonitorCollection([_bare(), _bare()])
        ax = coll.plot_step_sizes()
        assert ax is None


def _make_adjustment_stats(
    n_entries: int, n_runs: int, n_moves: int, seed: int = 0
) -> dict[str, np.ndarray]:
    """Build synthetic per-adjust-call stats arrays."""
    rng = np.random.default_rng(seed)
    return {
        "n_rounds": rng.integers(0, 10, (n_entries, n_runs, n_moves)).astype(
            np.int32
        ),
        "converged": rng.integers(0, 2, (n_entries, n_runs, n_moves)).astype(
            bool
        ),
        "cap_hits": rng.integers(0, 5, (n_entries, n_runs, n_moves)).astype(
            np.int32
        ),
        "floor_hits": rng.integers(0, 5, (n_entries, n_runs, n_moves)).astype(
            np.int32
        ),
        "bracket_detected": rng.integers(
            0, 2, (n_entries, n_runs, n_moves)
        ).astype(bool),
        "reject_reason_counts": rng.integers(
            0, 30, (n_entries, n_runs, n_moves, 4)
        ).astype(np.int32),
    }


class TestAdjustmentStatsRoundtrip:
    """Check schema v2: adjustment_stats round-trip, and files without the
    group read back with adjustment_stats=None."""

    def test_v2_roundtrip_single_run(self, tmp_path):
        n_entries, n_runs, n_moves = 5, 1, 2
        iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)
        adj_stats = _make_adjustment_stats(n_entries, n_runs, n_moves)

        path = tmp_path / "v2_single.h5"
        logger = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            entry_stats = {k: v[idx] for k, v in adj_stats.items()}
            logger.write_entry(
                int(iters[idx]),
                ss[idx],
                acc[idx],
                adjustment_stats=entry_stats,
            )
        logger.close()

        loaded = AdaptationLogger.read(path)
        assert loaded.adjustment_stats is not None
        for key in adj_stats:
            np.testing.assert_array_equal(
                loaded.adjustment_stats[key], adj_stats[key], err_msg=key
            )
        # Writing adjustment_stats leaves the core fields intact.
        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_allclose(loaded.step_sizes, ss, rtol=1e-5)
        np.testing.assert_allclose(loaded.acceptance_rates, acc, rtol=1e-5)

    def test_v2_roundtrip_multi_run(self, tmp_path):
        n_entries, n_runs, n_moves = 8, 3, 4
        iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)
        adj_stats = _make_adjustment_stats(n_entries, n_runs, n_moves, seed=7)

        path = tmp_path / "v2_multi.h5"
        logger = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            entry_stats = {k: v[idx] for k, v in adj_stats.items()}
            logger.write_entry(
                int(iters[idx]),
                ss[idx],
                acc[idx],
                adjustment_stats=entry_stats,
            )
        logger.close()

        loaded = AdaptationLogger.read(path)
        assert loaded.adjustment_stats is not None
        assert loaded.adjustment_stats["n_rounds"].shape == (
            n_entries,
            n_runs,
            n_moves,
        )
        assert loaded.adjustment_stats["reject_reason_counts"].shape == (
            n_entries,
            n_runs,
            n_moves,
            4,
        )
        np.testing.assert_array_equal(
            loaded.adjustment_stats["n_rounds"], adj_stats["n_rounds"]
        )

    def test_v1_backward_compat(self, tmp_path):
        """Entries written without adjustment_stats create no group and
        read back with adjustment_stats=None."""
        n_entries, n_runs, n_moves = 5, 1, 2
        iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)

        path = tmp_path / "v1.h5"
        logger = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            # No adjustment_stats passed — v1 behaviour
            logger.write_entry(int(iters[idx]), ss[idx], acc[idx])
        logger.close()

        with h5py.File(path, "r") as f:
            assert "adjustment_stats" not in f

        loaded = AdaptationLogger.read(path)
        assert loaded.adjustment_stats is None

    def test_v1_written_explicitly_without_schema_version(self, tmp_path):
        """A hand-written v1 file (no version attr) reads back with
        adjustment_stats and both evaluation counts set to None."""
        path = tmp_path / "hand_v1.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "iterations", data=np.array([0, 10, 20], dtype=np.int64)
            )
            f.create_dataset(
                "step_sizes", data=np.ones((3, 1, 2), dtype=np.float32)
            )
            f.create_dataset(
                "acceptance_rates",
                data=np.full((3, 1, 2), 0.5, dtype=np.float32),
            )
            f.attrs["move_names"] = json.dumps(["a", "b"])
            f.attrs["n_runs"] = 1
            f.attrs["n_moves"] = 2
            # No adjustment_stats group and no schema version attr.

        loaded = AdaptationLogger.read(path)
        assert loaded.adjustment_stats is None
        assert loaded.n_evaluations is None
        assert loaded.n_grad_evaluations is None


class TestAdaptationLogV3:
    """Check schema v3: n_evaluations and n_grad_evaluations round-trip,
    and older files read back with both set to None."""

    def _make_v3_data(self, n_entries=5, n_runs=1, n_moves=3, seed=42):
        rng = np.random.default_rng(seed)
        iters = np.arange(n_entries, dtype=np.int64) * 10
        ss = rng.uniform(0.01, 2.0, (n_entries, n_runs, n_moves)).astype(
            np.float32
        )
        acc = rng.uniform(0.1, 0.9, (n_entries, n_runs, n_moves)).astype(
            np.float32
        )
        n_evals = rng.integers(1, 100, (n_entries, n_runs, n_moves)).astype(
            np.int64
        )
        n_grad_evals = (n_evals * rng.uniform(0, 1, n_evals.shape)).astype(
            np.int64
        )
        names = [f"move_{k}" for k in range(n_moves)]
        return iters, ss, acc, n_evals, n_grad_evals, names

    def test_v3_roundtrip_single_run(self, tmp_path):
        n_entries, n_runs, n_moves = 5, 1, 3
        iters, ss, acc, n_evals, n_grad_evals, names = self._make_v3_data(
            n_entries,
            n_runs,
            n_moves,
        )

        path = tmp_path / "v3_single.h5"
        log = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            log.write_entry(
                int(iters[idx]),
                ss[idx],
                acc[idx],
                n_evaluations=n_evals[idx],
                n_grad_evaluations=n_grad_evals[idx],
            )
        log.close()

        loaded = AdaptationLogger.read(path)
        assert loaded.n_evaluations is not None
        assert loaded.n_grad_evaluations is not None
        assert loaded.n_evaluations.shape == (n_entries, n_runs, n_moves)
        np.testing.assert_array_equal(loaded.n_evaluations, n_evals)
        np.testing.assert_array_equal(loaded.n_grad_evaluations, n_grad_evals)
        # Writing evaluation counts leaves the core fields intact.
        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_allclose(loaded.step_sizes, ss, rtol=1e-5)
        np.testing.assert_allclose(loaded.acceptance_rates, acc, rtol=1e-5)

    def test_v3_roundtrip_multi_run(self, tmp_path):
        n_entries, n_runs, n_moves = 4, 2, 3
        iters, ss, acc, n_evals, n_grad_evals, names = self._make_v3_data(
            n_entries,
            n_runs,
            n_moves,
            seed=7,
        )

        path = tmp_path / "v3_multi.h5"
        log = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            log.write_entry(
                int(iters[idx]),
                ss[idx],
                acc[idx],
                n_evaluations=n_evals[idx],
                n_grad_evaluations=n_grad_evals[idx],
            )
        log.close()

        loaded = AdaptationLogger.read(path)
        assert loaded.n_evaluations.shape == (n_entries, n_runs, n_moves)
        np.testing.assert_array_equal(loaded.n_evaluations, n_evals)

    def test_v2_file_reads_with_none_fields(self, tmp_path):
        """A v2 file has no evaluation counts; both read back as None."""
        path = tmp_path / "hand_v2.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "iterations", data=np.array([0, 10, 20], dtype=np.int64)
            )
            f.create_dataset(
                "step_sizes", data=np.ones((3, 1, 2), dtype=np.float32)
            )
            f.create_dataset(
                "acceptance_rates",
                data=np.full((3, 1, 2), 0.5, dtype=np.float32),
            )
            f.attrs["move_names"] = json.dumps(["a", "b"])
            f.attrs["n_runs"] = 1
            f.attrs["n_moves"] = 2
            f.attrs["adaptation_log_schema_version"] = 2

        loaded = AdaptationLogger.read(path)
        assert loaded.n_evaluations is None
        assert loaded.n_grad_evaluations is None


class TestAdaptationCallbackPmapVmapShape:
    """Check that AdaptationCallback flattens PmapVmapRuns payloads.

    PmapVmapRuns(G=2, P=8) gives per-move arrays shaped
    ``(G, P, n_moves[, 4])``, while the HDF5 datasets are
    ``(n_runs=G*P, n_moves[, 4])``. Every per-replica array, not only
    step sizes and acceptance rates, must go through ``batcher.flatten``.
    """

    def test_pmap_vmap_shape_coerced_to_flat_n_runs(self, tmp_path):
        """The iter-0 baseline row and a (G, P)-shaped adjust-event row
        both write without broadcast errors."""
        n_gpu, n_per_gpu, n_moves = 2, 8, 4
        n_runs = n_gpu * n_per_gpu
        batcher = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)

        path = tmp_path / "shape_regress.adaptation.h5"
        adapt_logger = AdaptationLogger(
            path=path,
            move_names=[f"m{k}" for k in range(n_moves)],
            n_runs=n_runs,
        )
        cb = AdaptationCallback(adapt_logger)

        # on_start writes the iter-0 baseline.
        ss_initial = np.full(
            (n_gpu, n_per_gpu, n_moves), 0.2, dtype=np.float32
        )
        cb.on_start(
            ns_state=None,
            start_info={
                "_batcher": batcher,
                "step_sizes_per_move": ss_initial,
            },
        )

        info = {
            "_batcher": batcher,
            "step_sizes_per_move": ss_initial,
            "acceptance_rates_per_move": np.full(
                (n_gpu, n_per_gpu, n_moves),
                0.5,
                dtype=np.float32,
            ),
            "reject_counts_per_move": np.zeros(
                (n_gpu, n_per_gpu, n_moves, 4),
                dtype=np.int32,
            ),
            "adjustment_n_rounds": np.ones(
                (n_gpu, n_per_gpu, n_moves),
                dtype=np.int32,
            ),
            "adjustment_converged": np.ones(
                (n_gpu, n_per_gpu, n_moves),
                dtype=bool,
            ),
            "adjustment_cap_hits": np.zeros(
                (n_gpu, n_per_gpu, n_moves),
                dtype=np.int32,
            ),
            "adjustment_floor_hits": np.zeros(
                (n_gpu, n_per_gpu, n_moves),
                dtype=np.int32,
            ),
            "adjustment_bracket_detected": np.zeros(
                (n_gpu, n_per_gpu, n_moves),
                dtype=bool,
            ),
            "n_evaluations_per_move": np.full(
                (n_gpu, n_per_gpu, n_moves),
                100,
                dtype=np.int64,
            ),
            "n_grad_evaluations_per_move": np.full(
                (n_gpu, n_per_gpu, n_moves),
                20,
                dtype=np.int64,
            ),
        }
        # Unflattened, this fails with "Can't broadcast (1, 2, 8, 4) ->
        # (1, 16, 4)".
        cb.on_iteration(iteration=150, ns_state=None, info=info)
        cb.on_finish(ns_state=None)

        log = AdaptationLogger.read(path)
        assert log.n_runs == n_runs
        assert log.n_moves == n_moves
        # Rows: iter-0 baseline and the iter-150 adjust event.
        assert log.iterations.shape == (2,)
        assert log.step_sizes.shape == (2, n_runs, n_moves)
        assert log.adjustment_stats is not None
        assert log.adjustment_stats["reject_reason_counts"].shape == (
            2,
            n_runs,
            n_moves,
            4,
        )
        assert log.adjustment_stats["n_rounds"].shape == (2, n_runs, n_moves)
        assert log.n_evaluations is not None
        assert log.n_evaluations.shape == (2, n_runs, n_moves)


class TestAdaptationCallbackHealthWarnings:
    """Check the floor-hit and stall warnings of
    AdaptationCallback._check_adapter_health.

    The info dicts have no ``_batcher``, so each per-move array gets a
    leading n_runs=1 axis and the tests target a single replica.
    """

    @staticmethod
    def _make_callback(tmp_path, n_moves=3):
        path = tmp_path / "health.adaptation.h5"
        adapt_logger = AdaptationLogger(
            path=path,
            move_names=[f"m{k}" for k in range(n_moves)],
            n_runs=1,
        )
        # Seed the iter-0 baseline so on_iteration writes append cleanly.
        ss0 = np.full((1, n_moves), 0.1, dtype=np.float32)
        cb = AdaptationCallback(adapt_logger)
        cb.on_start(ns_state=None, start_info={"step_sizes_per_move": ss0})
        return cb

    @staticmethod
    def _event_info(n_moves, *, ss, acc, floor_hits, bracket_detected):
        return {
            "step_sizes_per_move": np.asarray(ss, dtype=np.float32),
            "acceptance_rates_per_move": np.asarray(acc, dtype=np.float32),
            "adjustment_floor_hits": np.asarray(floor_hits, dtype=np.int32),
            "adjustment_bracket_detected": np.asarray(
                bracket_detected,
                dtype=bool,
            ),
            "adjustment_n_rounds": np.zeros(n_moves, dtype=np.int32),
            "adjustment_converged": np.ones(n_moves, dtype=bool),
            "adjustment_cap_hits": np.zeros(n_moves, dtype=np.int32),
            "reject_counts_per_move": np.zeros((n_moves, 4), dtype=np.int32),
        }

    def test_floor_hit_warns_once_per_move(self, tmp_path, caplog):
        """The first event with floor_hits > 0 for a move warns once; later
        events for the same move stay silent."""
        n_moves = 3
        cb = self._make_callback(tmp_path, n_moves=n_moves)
        # Move 0 hits the floor on iter 100 and again on iter 200.
        info_floor = self._event_info(
            n_moves,
            ss=[1e-7, 0.1, 0.1],
            acc=[0.0, 0.4, 0.4],
            floor_hits=[3, 0, 0],
            bracket_detected=[False, True, True],
        )

        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.monitor"):
            cb.on_iteration(iteration=100, ns_state=None, info=info_floor)
            cb.on_iteration(iteration=200, ns_state=None, info=info_floor)
        cb.on_finish(ns_state=None)

        floor_msgs = [
            r
            for r in caplog.records
            if "hit step-size floor" in r.getMessage()
        ]
        assert len(floor_msgs) == 1
        assert "move=m0" in floor_msgs[0].getMessage()

    def test_stall_guard_fires_after_K_consecutive(self, tmp_path, caplog):
        """K = _STALL_HISTORY_LEN consecutive pathological events on one
        move give exactly one STALLED warning."""
        n_moves = 2
        K = AdaptationCallback._STALL_HISTORY_LEN
        cb = self._make_callback(tmp_path, n_moves=n_moves)
        pathological = self._event_info(
            n_moves,
            ss=[1e-7, 0.05],
            acc=[0.0, 0.35],
            floor_hits=[5, 0],
            bracket_detected=[False, True],
        )

        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.monitor"):
            for k in range(K):
                cb.on_iteration(
                    iteration=1000 + 100 * k,
                    ns_state=None,
                    info=pathological,
                )
        cb.on_finish(ns_state=None)

        stall_msgs = [r for r in caplog.records if "STALLED" in r.getMessage()]
        assert len(stall_msgs) == 1
        assert "move=m0" in stall_msgs[0].getMessage()
        # The floor warning fires too, once, on the first event.
        floor_msgs = [
            r
            for r in caplog.records
            if "hit step-size floor" in r.getMessage()
        ]
        assert len(floor_msgs) == 1

    def test_stall_guard_silent_when_bracket_detected(self, tmp_path, caplog):
        """A floor-hit event with bracket_detected=True on every replica
        gives no STALLED warning."""
        n_moves = 1
        K = AdaptationCallback._STALL_HISTORY_LEN
        cb = self._make_callback(tmp_path, n_moves=n_moves)
        bracketed = self._event_info(
            n_moves,
            ss=[1e-7],
            acc=[0.0],
            floor_hits=[1],
            bracket_detected=[True],
        )
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.monitor"):
            for k in range(K + 2):
                cb.on_iteration(
                    iteration=10 + k,
                    ns_state=None,
                    info=bracketed,
                )
        cb.on_finish(ns_state=None)
        assert not any("STALLED" in r.getMessage() for r in caplog.records)

    def test_no_warnings_when_healthy(self, tmp_path, caplog):
        """No floor hits and acceptance near target: no warnings."""
        n_moves = 2
        cb = self._make_callback(tmp_path, n_moves=n_moves)
        healthy = self._event_info(
            n_moves,
            ss=[0.05, 0.05],
            acc=[0.32, 0.36],
            floor_hits=[0, 0],
            bracket_detected=[True, True],
        )
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.monitor"):
            for k in range(10):
                cb.on_iteration(
                    iteration=10 + k,
                    ns_state=None,
                    info=healthy,
                )
        cb.on_finish(ns_state=None)
        assert caplog.records == []
