"""Tests for AdaptationLogger, AdaptationLog, and related postprocess integration.

All I/O tests use tmp_path (pytest fixture) — no real NS run required.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless before any other matplotlib import

import numpy as np
import pytest

from jaxrens.io.adaptation_log import AdaptationLog, AdaptationLogger
from jaxrens.postprocess.collection import MonitorCollection
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.plotting import plot_acceptance_rates, plot_step_sizes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 1. AdaptationLogger round-trip
# ---------------------------------------------------------------------------


class TestAdaptationLoggerRoundTrip:
    def test_round_trip_10_entries_single_run(self, tmp_path):
        """Write 10 entries, close, read back: shapes and values match."""
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

        assert path.exists(), "File must be created on close()"
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
        """Batched run: n_runs=3, n_moves=4."""
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
        """Caller passes (n_moves,) 1D; should be reshaped to (1, n_moves)."""
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

    def test_values_preserved_over_flush_boundary(self, tmp_path):
        """Write more than _FLUSH_INTERVAL entries; data survives multi-flush."""
        from jaxrens.io.adaptation_log import _FLUSH_INTERVAL

        n_entries = _FLUSH_INTERVAL + 5
        n_moves = 2
        names = ["x", "y"]
        path = tmp_path / "flush.h5"
        logger = AdaptationLogger(path=path, move_names=names, n_runs=1)

        iters_list = []
        ss_list = []
        for idx in range(n_entries):
            ss = np.array([[float(idx), float(idx) * 2]], dtype=np.float32)
            acc = np.array([[0.5, 0.5]], dtype=np.float32)
            logger.write_entry(idx, ss, acc)
            iters_list.append(idx)
            ss_list.append(ss)
        logger.close()

        loaded = AdaptationLogger.read(path)
        assert loaded.iterations.shape == (n_entries,)
        np.testing.assert_array_equal(loaded.iterations, np.array(iters_list))
        expected_ss = np.stack(ss_list, axis=0)  # (n_entries, 1, 2)
        np.testing.assert_allclose(loaded.step_sizes, expected_ss, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Empty logger behaviour
# ---------------------------------------------------------------------------


class TestAdaptationLoggerEmpty:
    def test_empty_logger_creates_no_file(self, tmp_path):
        """No write_entry calls: close() must NOT create the file."""
        path = tmp_path / "empty.h5"
        logger = AdaptationLogger(path=path, move_names=["m0"], n_runs=1)
        logger.close()
        assert not path.exists(), "Empty logger must not create a file"

    def test_write_after_close_raises(self, tmp_path):
        path = tmp_path / "closed.h5"
        logger = AdaptationLogger(path=path, move_names=["m0"], n_runs=1)
        logger.close()
        with pytest.raises(RuntimeError, match="closed"):
            logger.write_entry(0, np.array([[0.5]]), np.array([[0.5]]))


# ---------------------------------------------------------------------------
# 3 & 4. Monitor.from_directory integration
# ---------------------------------------------------------------------------


def _write_minimal_checkpoint(tmp_path: Path, prefix: str = "ns") -> None:
    """Write the bare minimum files for Monitor.from_directory to succeed."""
    import jax.numpy as jnp

    from jaxrens.io.checkpoint import save_checkpoint
    from jaxrens.io.energy_log import EnergyLogger

    rng = np.random.default_rng(7)
    n_live, n_dead = 5, 10
    dead_e = np.sort(rng.uniform(0, 5, n_dead))
    live_e = rng.uniform(4, 6, n_live)
    pos = rng.uniform(0, 1, (n_live, 2, 3))
    dp = rng.uniform(0, 1, (n_dead, 2, 3))
    ns_state = {
        "positions": jnp.asarray(pos),
        "types": jnp.zeros(n_live, dtype=int),
        "energies": jnp.asarray(live_e),
        "cells": None,
        "dead_energies": jnp.asarray(
            np.concatenate([dead_e, np.full(100, np.inf)])
        ),
        "dead_positions": jnp.asarray(
            np.concatenate([dp, np.zeros((100, 2, 3))], axis=0)
        ),
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": 10.0,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_live,
    }
    save_checkpoint(tmp_path / f"{prefix}.final.checkpoint.h5", ns_state)

    energy_logger = EnergyLogger(
        tmp_path / f"{prefix}.energies",
        n_walkers=n_live,
        n_cull=1,
    )
    energy_logger.write_header()
    for i, e in enumerate(dead_e):
        energy_logger.write_entry(i, float(e))
    energy_logger.close()


class TestMonitorFromDirectoryAdaptation:
    def test_loads_adaptation_trace_when_present(self, tmp_path):
        """from_directory with a .adaptation.h5 sets adaptation_trace."""
        prefix = "ns"
        _write_minimal_checkpoint(tmp_path, prefix)

        # Write an adaptation trace
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
        """from_directory without .adaptation.h5 sets adaptation_trace=None."""
        prefix = "ns"
        _write_minimal_checkpoint(tmp_path, prefix)
        m = Monitor.from_directory(tmp_path, prefix=prefix)
        assert m.adaptation_trace is None


# ---------------------------------------------------------------------------
# 5–7. plot_step_sizes / plot_acceptance_rates
# ---------------------------------------------------------------------------


class TestPlotStepSizes:
    def test_per_run_true_draws_n_moves_lines_for_single_run(self):
        import matplotlib.pyplot as plt

        n_moves = 3
        m = _make_monitor_with_trace(n_runs=1, n_moves=n_moves, n_entries=6)
        ax = plot_step_sizes(m, per_run=True)
        assert len(ax.get_lines()) == n_moves
        plt.close("all")

    def test_per_run_true_draws_n_runs_times_n_moves_lines(self):
        """With 3 runs and 2 moves, per_run=True => 6 lines."""
        import matplotlib.pyplot as plt

        n_runs, n_moves = 3, 2
        m = _make_monitor_with_trace(
            n_runs=n_runs, n_moves=n_moves, n_entries=5
        )
        ax = plot_step_sizes(m, per_run=True)
        assert len(ax.get_lines()) == n_runs * n_moves
        plt.close("all")

    def test_per_run_false_mean_plus_fill(self):
        """per_run=False: n_moves lines, collections for fill_between."""
        import matplotlib.pyplot as plt

        n_runs, n_moves = 3, 2
        m = _make_monitor_with_trace(
            n_runs=n_runs, n_moves=n_moves, n_entries=5
        )
        ax = plot_step_sizes(m, per_run=False)
        # One Line2D per move (mean), plus PolyCollection(s) for fill_between
        assert len(ax.get_lines()) == n_moves
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
        import matplotlib.pyplot as plt

        m = _make_monitor_with_trace()
        _, existing_ax = plt.subplots()
        returned = plot_step_sizes(m, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")


class TestPlotAcceptanceRates:
    def test_per_run_true_3_runs_2_moves(self):
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

        m = _make_monitor_with_trace()
        _, existing_ax = plt.subplots()
        returned = plot_acceptance_rates(m, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")


# ---------------------------------------------------------------------------
# 8. MonitorCollection.plot_step_sizes
# ---------------------------------------------------------------------------


class TestMonitorCollectionAdaptationPlots:
    def test_collection_skips_monitors_without_trace(self):
        """MonitorCollection.plot_step_sizes skips monitors with no adaptation_trace."""
        import matplotlib.pyplot as plt

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
        ax = coll.plot_step_sizes()  # must not raise
        assert ax is not None
        plt.close("all")

    def test_collection_returns_none_ax_when_all_traces_missing(self):
        """Returns None (no ax created) when no monitor has a trace."""
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


# ---------------------------------------------------------------------------
# 9. Schema v2 — adjustment_stats roundtrip and v1 backward compat
# ---------------------------------------------------------------------------


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
    def test_v2_roundtrip_single_run(self, tmp_path):
        """Write v2 file with adjustment_stats; read back and check field equality."""
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
        assert (
            loaded.adjustment_stats is not None
        ), "v2 file must have adjustment_stats"
        for key in adj_stats:
            assert key in loaded.adjustment_stats, f"missing key: {key}"
            np.testing.assert_array_equal(
                loaded.adjustment_stats[key],
                adj_stats[key],
                err_msg=f"mismatch for key {key}",
            )

    def test_v2_roundtrip_multi_run(self, tmp_path):
        """n_runs=3, n_moves=4 roundtrip."""
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
        """v1-style file (no adjustment_stats group) loads with adjustment_stats=None."""
        import h5py

        n_entries, n_runs, n_moves = 5, 1, 2
        iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)

        path = tmp_path / "v1.h5"
        logger = AdaptationLogger(path=path, move_names=names, n_runs=n_runs)
        for idx in range(n_entries):
            # No adjustment_stats passed — v1 behaviour
            logger.write_entry(int(iters[idx]), ss[idx], acc[idx])
        logger.close()

        # Verify file has no adjustment_stats group
        with h5py.File(path, "r") as f:
            assert "adjustment_stats" not in f

        loaded = AdaptationLogger.read(path)
        assert (
            loaded.adjustment_stats is None
        ), "v1 file must give adjustment_stats=None"

    def test_v1_written_explicitly_without_schema_version(self, tmp_path):
        """Write a raw v1 file by hand (no version attr); read gives adjustment_stats=None."""
        import json

        import h5py

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
            # deliberately no "adjustment_stats" group and no schema_version attr

        loaded = AdaptationLogger.read(path)
        assert loaded.adjustment_stats is None

    def test_v2_existing_field_basic_shapes_correct(self, tmp_path):
        """After roundtrip, core fields (iterations, step_sizes, acceptance_rates) still correct."""
        n_entries, n_runs, n_moves = 4, 2, 3
        iters, ss, acc, names = _make_trace(n_entries, n_runs, n_moves)
        adj_stats = _make_adjustment_stats(n_entries, n_runs, n_moves)

        path = tmp_path / "v2_core.h5"
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
        assert loaded.iterations.shape == (n_entries,)
        assert loaded.step_sizes.shape == (n_entries, n_runs, n_moves)
        assert loaded.acceptance_rates.shape == (n_entries, n_runs, n_moves)
        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_allclose(loaded.step_sizes, ss, rtol=1e-5)
        np.testing.assert_allclose(loaded.acceptance_rates, acc, rtol=1e-5)


# ---------------------------------------------------------------------------
# v3 schema (n_evaluations / n_grad_evaluations) — moved from
# test_evaluation_counter.py so all schema-version round-trip tests live here.
# ---------------------------------------------------------------------------


class TestAdaptationLogV3:
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
        """v2 file (no n_evaluations / n_grad_evaluations) loads cleanly with None."""
        import h5py

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

    def test_v1_file_reads_with_none_fields(self, tmp_path):
        """v1 file (no schema attr) loads cleanly with None counters."""
        import h5py

        path = tmp_path / "hand_v1.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("iterations", data=np.array([0], dtype=np.int64))
            f.create_dataset(
                "step_sizes", data=np.ones((1, 1, 1), dtype=np.float32)
            )
            f.create_dataset(
                "acceptance_rates",
                data=np.full((1, 1, 1), 0.5, dtype=np.float32),
            )
            f.attrs["move_names"] = json.dumps(["m0"])
            f.attrs["n_runs"] = 1
            f.attrs["n_moves"] = 1

        loaded = AdaptationLogger.read(path)
        assert loaded.n_evaluations is None
        assert loaded.n_grad_evaluations is None

    def test_v3_core_fields_unchanged(self, tmp_path):
        """v3 write/read leaves iterations / step_sizes / acceptance_rates intact."""
        n_entries, n_runs, n_moves = 4, 1, 2
        iters, ss, acc, n_evals, n_grad_evals, names = self._make_v3_data(
            n_entries,
            n_runs,
            n_moves,
        )

        path = tmp_path / "v3_core.h5"
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
        assert loaded.iterations.shape == (n_entries,)
        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_allclose(loaded.step_sizes, ss, rtol=1e-5)
        np.testing.assert_allclose(loaded.acceptance_rates, acc, rtol=1e-5)
