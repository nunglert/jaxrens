"""Tests for AccRatesLogger / AccRatesLog round-trips.

Mirrors the structure of ``test_adaptation_log.py``; pure I/O — no JAX
or NS run required.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxrens.io.acc_rates_log import AccRatesLog, AccRatesLogger


def _make_counts(n_entries=10, n_runs=1, n_moves=3, seed=0):
    rng = np.random.default_rng(seed)
    iters = np.arange(n_entries, dtype=np.int64)
    n_prop = rng.integers(50, 200, (n_entries, n_runs, n_moves)).astype(np.int64)
    # n_accepted <= n_proposed component-wise
    n_acc = (n_prop * rng.uniform(0.0, 1.0, n_prop.shape)).astype(np.int64)
    return iters, n_acc, n_prop


class TestAccRatesLoggerRoundTrip:
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
            path=path, move_names=["a", "b"], n_runs=n_runs,
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
        # n_proposed is always > 0 from _make_counts (50..200), so no
        # zero-guard surprise here.
        np.testing.assert_allclose(
            rates, n_acc / n_prop, rtol=1e-5,
        )

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
        # Force at least two mid-run flushes: with flush_interval=50 the
        # auto-flush condition fires every 50 iterations, so iterating
        # over 130 absolute iters yields 2 mid-run flushes + close().
        flush_interval = 50
        n_entries = 130
        n_moves = 2
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
        log.close()
        loaded = AccRatesLogger.read(path)
        assert loaded.iterations.shape == (n_entries,)
        np.testing.assert_array_equal(
            loaded.iterations, np.arange(n_entries, dtype=np.int64),
        )


class TestAdaptationCallbackPmapVmapShapeRegression:
    """Reproduces the production crash from SLURM job 453042:
    ``TypeError: Can't broadcast (1, 2, 8, 4) -> (1, 16, 4)``.

    PmapVmapRuns(G=2, P=8) produces adjust_info arrays shaped
    ``(G, P, n_moves[, 4])``.  ``AdaptationCallback.on_iteration``
    previously forwarded these raw to the HDF5 writer, whose dataset
    was created at ``(n_runs=G*P, n_moves[, 4])`` from the iter-0 row.
    The fix is to apply ``batcher.flatten`` to all per-replica payload
    arrays, not just ss/acc.
    """

    def test_pmap_vmap_shape_coerced_to_flat_n_runs(self, tmp_path):
        """Synthetic test: simulate the on_iteration code path with
        a PmapVmapRuns-shaped info dict and confirm the HDF5 writer
        accepts both the iter-0 baseline row and the subsequent
        adjust-event row without broadcast errors.
        """
        from jaxrens.cli.monitor import AdaptationCallback
        from jaxrens.io.adaptation_log import AdaptationLogger
        from jaxrens.sampling.batch_descriptor import PmapVmapRuns

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

        # ---- on_start writes the iter-0 baseline (already-flat shape) ----
        ss_initial = np.full((n_gpu, n_per_gpu, n_moves), 0.2, dtype=np.float32)
        cb.on_start(
            ns_state=None,
            start_info={
                "_batcher": batcher,
                "step_sizes_per_move": ss_initial,
            },
        )

        # ---- on_iteration with PmapVmapRuns-shaped adjustment_stats ----
        info = {
            "_batcher": batcher,
            "step_sizes_per_move": ss_initial,
            "acceptance_rates_per_move": np.full(
                (n_gpu, n_per_gpu, n_moves), 0.5, dtype=np.float32,
            ),
            "reject_counts_per_move": np.zeros(
                (n_gpu, n_per_gpu, n_moves, 4), dtype=np.int32,
            ),
            "adjustment_n_rounds": np.ones(
                (n_gpu, n_per_gpu, n_moves), dtype=np.int32,
            ),
            "adjustment_converged": np.ones(
                (n_gpu, n_per_gpu, n_moves), dtype=bool,
            ),
            "adjustment_cap_hits": np.zeros(
                (n_gpu, n_per_gpu, n_moves), dtype=np.int32,
            ),
            "adjustment_floor_hits": np.zeros(
                (n_gpu, n_per_gpu, n_moves), dtype=np.int32,
            ),
            "adjustment_bracket_detected": np.zeros(
                (n_gpu, n_per_gpu, n_moves), dtype=bool,
            ),
            "n_evaluations_per_move": np.full(
                (n_gpu, n_per_gpu, n_moves), 100, dtype=np.int64,
            ),
            "n_grad_evaluations_per_move": np.full(
                (n_gpu, n_per_gpu, n_moves), 20, dtype=np.int64,
            ),
        }
        # This used to crash with "Can't broadcast (1, 2, 8, 4) -> (1, 16, 4)".
        cb.on_iteration(iteration=150, ns_state=None, info=info)
        cb.on_finish(ns_state=None)

        log = AdaptationLogger.read(path)
        assert log.n_runs == n_runs
        assert log.n_moves == n_moves
        # 2 rows: iter-0 baseline + iter-150 adjust event.
        assert log.iterations.shape == (2,)
        assert log.step_sizes.shape == (2, n_runs, n_moves)
        assert log.adjustment_stats is not None
        assert log.adjustment_stats["reject_reason_counts"].shape == (
            2, n_runs, n_moves, 4,
        )
        assert log.adjustment_stats["n_rounds"].shape == (2, n_runs, n_moves)
        assert log.n_evaluations is not None
        assert log.n_evaluations.shape == (2, n_runs, n_moves)


class TestAccRatesLoggerEmpty:
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
            log.write_entry(0, np.array([1], dtype=np.int64),
                            np.array([2], dtype=np.int64))
