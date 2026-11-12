"""Tests for AccRatesLogger, the per-iteration acceptance-count log.

Covers the .acc_rates.h5 round-trip, input shape coercion, the zero
guard of acceptance_rates, flushing and the empty/closed logger.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxrens.io.acc_rates_log import AccRatesLogger


def _make_counts(n_entries=10, n_runs=1, n_moves=3, seed=0):
    rng = np.random.default_rng(seed)
    iters = np.arange(n_entries, dtype=np.int64)
    n_prop = rng.integers(50, 200, (n_entries, n_runs, n_moves)).astype(
        np.int64
    )
    n_acc = (n_prop * rng.uniform(0.0, 1.0, n_prop.shape)).astype(np.int64)
    return iters, n_acc, n_prop


class TestAccRatesLoggerRoundTrip:
    """Check that counts written with write_entry read back unchanged."""

    def test_single_run(self, tmp_path):
        n_entries, n_runs, n_moves = 8, 1, 3
        iters, n_acc, n_prop = _make_counts(n_entries, n_runs, n_moves)
        path = tmp_path / "acc.h5"

        names = [f"m{k}" for k in range(n_moves)]
        log = AccRatesLogger(path=path, move_names=names, n_runs=n_runs)
        for i in range(n_entries):
            log.write_entry(int(iters[i]), n_acc[i], n_prop[i])
        log.close()

        assert path.exists()
        loaded = AccRatesLogger.read(path)
        assert loaded.n_runs == n_runs
        assert loaded.n_moves == n_moves
        assert loaded.move_names == names
        assert loaded.n_accepted.shape == (n_entries, n_runs, n_moves)
        assert loaded.n_proposed.shape == (n_entries, n_runs, n_moves)
        np.testing.assert_array_equal(loaded.iterations, iters)
        np.testing.assert_array_equal(loaded.n_accepted, n_acc)
        np.testing.assert_array_equal(loaded.n_proposed, n_prop)

    def test_multi_run(self, tmp_path):
        n_entries, n_runs, n_moves = 12, 3, 2
        iters, n_acc, n_prop = _make_counts(n_entries, n_runs, n_moves)
        path = tmp_path / "acc_multi.h5"
        log = AccRatesLogger(
            path=path,
            move_names=["a", "b"],
            n_runs=n_runs,
        )
        for i in range(n_entries):
            log.write_entry(int(iters[i]), n_acc[i], n_prop[i])
        log.close()

        loaded = AccRatesLogger.read(path)
        assert loaded.n_accepted.shape == (n_entries, n_runs, n_moves)
        np.testing.assert_array_equal(loaded.n_accepted, n_acc)

    def test_derived_acceptance_rates_property(self, tmp_path):
        n_entries, n_runs, n_moves = 5, 1, 2
        iters, n_acc, n_prop = _make_counts(n_entries, n_runs, n_moves, seed=7)
        path = tmp_path / "acc_rates.h5"
        log = AccRatesLogger(path=path, move_names=["x", "y"], n_runs=1)
        for i in range(n_entries):
            log.write_entry(int(iters[i]), n_acc[i], n_prop[i])
        log.close()

        loaded = AccRatesLogger.read(path)
        rates = loaded.acceptance_rates
        # _make_counts draws n_proposed in [50, 200), so it is never 0.
        np.testing.assert_allclose(
            rates,
            n_acc / n_prop,
            rtol=1e-5,
        )

    def test_acceptance_rates_zero_proposed(self, tmp_path):
        """A move with no proposals gets rate 0, not NaN."""
        path = tmp_path / "zero.h5"
        log = AccRatesLogger(path=path, move_names=["x", "y"], n_runs=1)
        log.write_entry(0, np.array([0, 5]), np.array([0, 10]))
        log.close()
        rates = AccRatesLogger.read(path).acceptance_rates
        np.testing.assert_array_equal(rates[0, 0], [0.0, 0.5])

    def test_1d_input_coerced(self, tmp_path):
        path = tmp_path / "coerce.h5"
        log = AccRatesLogger(path=path, move_names=["a", "b"], n_runs=1)
        log.write_entry(
            0,
            np.array([10, 20], dtype=np.int64),
            np.array([30, 40], dtype=np.int64),
        )
        log.close()
        loaded = AccRatesLogger.read(path)
        assert loaded.n_accepted.shape == (1, 1, 2)
        np.testing.assert_array_equal(loaded.n_accepted[0, 0], [10, 20])

    def test_flush_boundary(self, tmp_path):
        # flush_interval=50 over 130 iterations: flushes at iterations 50
        # and 100, then close() writes the rest.
        flush_interval = 50
        n_entries = 130
        path = tmp_path / "flush.h5"
        log = AccRatesLogger(
            path=path,
            move_names=["x", "y"],
            n_runs=1,
            flush_interval=flush_interval,
        )
        for i in range(n_entries):
            log.write_entry(
                i,
                np.array([i, 2 * i], dtype=np.int64),
                np.array([100, 100], dtype=np.int64),
            )
            if i == 59:
                # The flush at iteration 50 wrote rows 0-50 before close().
                assert AccRatesLogger.read(path).iterations.shape == (51,)
        log.close()
        loaded = AccRatesLogger.read(path)
        np.testing.assert_array_equal(
            loaded.iterations,
            np.arange(n_entries, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            loaded.n_accepted[:, 0, 1], 2 * np.arange(n_entries)
        )


class TestAccRatesLoggerEmpty:
    """Check the logger lifecycle: no file without entries, no writes
    after close."""

    def test_empty_creates_no_file(self, tmp_path):
        path = tmp_path / "empty.h5"
        log = AccRatesLogger(path=path, move_names=["m0"], n_runs=1)
        log.close()
        assert not path.exists()

    def test_write_after_close_raises(self, tmp_path):
        path = tmp_path / "after_close.h5"
        log = AccRatesLogger(path=path, move_names=["m0"], n_runs=1)
        log.close()
        with pytest.raises(RuntimeError):
            log.write_entry(
                0, np.array([1], dtype=np.int64), np.array([2], dtype=np.int64)
            )
