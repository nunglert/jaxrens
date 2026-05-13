"""Tests for RELogger / RELog round-trips.

Mirrors ``test_acc_rates_log.py``; pure I/O — no JAX or NS run required.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless before any other matplotlib import

import numpy as np
import pytest

from jaxrens.io.re_stats_log import RELog, RELogger
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.plotting import plot_re_acceptance


def _make_counts(n_entries=8, n_pairs=3, seed=0):
    rng = np.random.default_rng(seed)
    iters = np.arange(n_entries, dtype=np.int64) * 5  # arbitrary cadence
    n_att = rng.integers(1, 10, (n_entries, n_pairs)).astype(np.int32)
    # n_accepted <= n_attempted component-wise
    n_acc = (n_att * rng.uniform(0.0, 1.0, n_att.shape)).astype(np.int32)
    return iters, n_acc, n_att


class TestRELoggerRoundTrip:
    def test_basic_roundtrip(self, tmp_path):
        n_entries, n_pairs = 8, 3
        iters, n_acc, n_att = _make_counts(n_entries, n_pairs)
        path = tmp_path / "re.h5"

        log = RELogger(path=path, n_pairs=n_pairs, flavor="pressure")
        for i in range(n_entries):
            log.write_entry(int(iters[i]), n_acc[i], n_att[i])
        log.close()

        assert path.exists()
        loaded = RELogger.read(path)
        assert loaded.n_pairs == n_pairs
        assert loaded.flavor == "pressure"
        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_array_equal(loaded.n_accepted_per_pair, n_acc)
        np.testing.assert_array_equal(loaded.n_attempted_per_pair, n_att)

    def test_buffer_flush_threshold(self, tmp_path):
        # Force at least one buffer flush mid-run (>_FLUSH_INTERVAL=256).
        n_entries, n_pairs = 300, 2
        iters, n_acc, n_att = _make_counts(n_entries, n_pairs, seed=1)
        path = tmp_path / "re.h5"

        log = RELogger(path=path, n_pairs=n_pairs, flavor="xrens")
        for i in range(n_entries):
            log.write_entry(int(iters[i]), n_acc[i], n_att[i])
        # Mid-run, the file should already exist after the first auto-flush.
        assert path.exists()
        log.close()

        loaded = RELogger.read(path)
        assert loaded.iterations.shape == (n_entries,)
        assert loaded.n_accepted_per_pair.shape == (n_entries, n_pairs)
        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_array_equal(loaded.n_accepted_per_pair, n_acc)
        np.testing.assert_array_equal(loaded.n_attempted_per_pair, n_att)

    def test_no_file_when_no_entries(self, tmp_path):
        path = tmp_path / "re_empty.h5"
        log = RELogger(path=path, n_pairs=3, flavor="pressure")
        log.close()
        assert not path.exists()

    def test_flavor_attr_preserved(self, tmp_path):
        for flavor in ("pressure", "xrens", "semi_grand"):
            path = tmp_path / f"re_{flavor}.h5"
            log = RELogger(path=path, n_pairs=2, flavor=flavor)
            log.write_entry(0, np.array([1, 0], dtype=np.int32),
                            np.array([1, 1], dtype=np.int32))
            log.close()
            assert RELogger.read(path).flavor == flavor

    def test_close_idempotent(self, tmp_path):
        path = tmp_path / "re.h5"
        log = RELogger(path=path, n_pairs=2, flavor="pressure")
        log.write_entry(0, np.array([1, 0], dtype=np.int32),
                        np.array([1, 1], dtype=np.int32))
        log.close()
        log.close()  # second close must not raise

    def test_write_after_close_raises(self, tmp_path):
        log = RELogger(path=tmp_path / "re.h5", n_pairs=2, flavor="pressure")
        log.close()
        with pytest.raises(RuntimeError):
            log.write_entry(0, np.array([0, 0], dtype=np.int32),
                            np.array([0, 0], dtype=np.int32))

    def test_shape_mismatch_raises(self, tmp_path):
        log = RELogger(path=tmp_path / "re.h5", n_pairs=3, flavor="pressure")
        with pytest.raises(ValueError, match="expected per-pair array of shape"):
            log.write_entry(0, np.array([0, 0], dtype=np.int32),
                            np.array([0, 0, 0], dtype=np.int32))


class TestRELogAcceptanceRates:
    def test_rate_computation(self):
        log = RELog(
            iterations=np.array([0, 5], dtype=np.int64),
            n_accepted_per_pair=np.array([[3, 0], [4, 2]], dtype=np.int32),
            n_attempted_per_pair=np.array([[6, 0], [8, 4]], dtype=np.int32),
            n_pairs=2,
            flavor="pressure",
        )
        rates = log.acceptance_rates
        # Pair 0: 3/6 = 0.5, then 4/8 = 0.5.  Pair 1: 0/0 → 0 (guarded), then 2/4 = 0.5.
        np.testing.assert_allclose(
            rates,
            np.array([[0.5, 0.0], [0.5, 0.5]], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Monitor.from_directory integration + plot_re_acceptance
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
    ns_state = {
        "positions": jnp.asarray(pos),
        "types": jnp.zeros(n_live, dtype=int),
        "energies": jnp.asarray(live_e),
        "cells": None,
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": 10.0,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_live,
    }
    save_checkpoint(tmp_path / f"{prefix}.final.checkpoint.h5", ns_state)

    # Monitor.from_directory needs .energies for the dead-energy fallback.
    e_logger = EnergyLogger(path=tmp_path / f"{prefix}.energies", n_walkers=n_live)
    for i, e in enumerate(dead_e):
        e_logger.write_entry(i, float(e), 0.0)
    e_logger.close()


def _make_monitor_with_re_trace(
    n_entries: int = 20,
    n_pairs: int = 3,
    flavor: str = "pressure",
    label: str = "test-re",
    seed: int = 1,
) -> Monitor:
    """Build a Monitor with a synthetic RELog attached."""
    rng = np.random.default_rng(seed)
    n_dead, n_live = 50, 10
    iters = np.arange(n_entries, dtype=np.int64) * 5
    n_att = rng.integers(2, 12, (n_entries, n_pairs)).astype(np.int32)
    n_acc = (n_att * rng.uniform(0.0, 1.0, n_att.shape)).astype(np.int32)
    trace = RELog(
        iterations=iters,
        n_accepted_per_pair=n_acc,
        n_attempted_per_pair=n_att,
        n_pairs=n_pairs,
        flavor=flavor,
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
        re_trace=trace,
    )


class TestMonitorFromDirectoryRETrace:
    def test_loads_re_trace_when_present(self, tmp_path):
        """from_directory with a .re_stats.h5 file sets ``re_trace``."""
        prefix = "ns"
        _write_minimal_checkpoint(tmp_path, prefix)

        n_pairs = 2
        re_logger = RELogger(
            path=tmp_path / f"{prefix}.re_stats.h5",
            n_pairs=n_pairs,
            flavor="pressure",
        )
        rng = np.random.default_rng(0)
        for i in range(6):
            n_att = rng.integers(1, 5, n_pairs).astype(np.int32)
            n_acc = (n_att * rng.uniform(0.0, 1.0, n_pairs)).astype(np.int32)
            re_logger.write_entry(i * 3, n_acc, n_att)
        re_logger.close()

        m = Monitor.from_directory(tmp_path, prefix=prefix)
        assert m.re_trace is not None
        assert isinstance(m.re_trace, RELog)
        assert m.re_trace.n_pairs == n_pairs
        assert m.re_trace.flavor == "pressure"
        assert m.re_trace.iterations.shape == (6,)

    def test_re_trace_none_when_missing(self, tmp_path):
        prefix = "ns"
        _write_minimal_checkpoint(tmp_path, prefix)
        m = Monitor.from_directory(tmp_path, prefix=prefix)
        assert m.re_trace is None


class TestPlotREAcceptance:
    def test_per_pair_draws_n_pairs_lines(self):
        import matplotlib.pyplot as plt

        m = _make_monitor_with_re_trace(n_entries=10, n_pairs=4)
        ax = plot_re_acceptance(m, per_pair=True)
        assert len(ax.get_lines()) == 4
        plt.close("all")

    def test_per_pair_false_one_line_plus_fill(self):
        """per_pair=False: one Line2D for the mean, plus fill_between collection."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        m = _make_monitor_with_re_trace(n_entries=10, n_pairs=4)
        ax = plot_re_acceptance(m, per_pair=False)
        assert len(ax.get_lines()) == 1
        assert any(isinstance(c, PolyCollection) for c in ax.collections)
        plt.close("all")

    def test_window_smoothing_preserves_shape(self):
        """``window=W`` smooths each per-pair trace; line count is unchanged."""
        import matplotlib.pyplot as plt

        m = _make_monitor_with_re_trace(n_entries=20, n_pairs=3)
        ax = plot_re_acceptance(m, per_pair=True, window=5)
        assert len(ax.get_lines()) == 3
        # Each smoothed line has one y-value per entry.
        for line in ax.get_lines():
            assert line.get_ydata().shape == (20,)
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
        with pytest.raises(ValueError, match="re_trace"):
            plot_re_acceptance(m)

    def test_raises_when_zero_pairs(self):
        rng = np.random.default_rng(0)
        empty_trace = RELog(
            iterations=np.array([], dtype=np.int64),
            n_accepted_per_pair=np.zeros((0, 0), dtype=np.int32),
            n_attempted_per_pair=np.zeros((0, 0), dtype=np.int32),
            n_pairs=0,
            flavor="pressure",
        )
        m = Monitor(
            dead_energies=rng.uniform(0, 10, 20),
            dead_volumes=None,
            live_energies=rng.uniform(8, 12, 5),
            live_volumes=None,
            log_evidence=0.0,
            iteration=20,
            n_live=5,
            n_cull=1,
            re_trace=empty_trace,
        )
        with pytest.raises(ValueError, match="zero pairs"):
            plot_re_acceptance(m)

    def test_reuses_existing_ax(self):
        import matplotlib.pyplot as plt

        m = _make_monitor_with_re_trace()
        _, existing_ax = plt.subplots()
        returned = plot_re_acceptance(m, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")

    def test_flavor_appears_in_ylabel(self):
        import matplotlib.pyplot as plt

        m = _make_monitor_with_re_trace(flavor="xrens")
        ax = plot_re_acceptance(m)
        assert "xrens" in ax.get_ylabel()
        plt.close("all")
