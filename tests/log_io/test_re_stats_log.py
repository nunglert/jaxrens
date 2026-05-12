"""Tests for RELogger / RELog round-trips.

Mirrors ``test_acc_rates_log.py``; pure I/O — no JAX or NS run required.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxrens.io.re_stats_log import RELog, RELogger


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
