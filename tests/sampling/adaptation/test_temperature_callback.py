"""Tests for ``TemperatureCallback`` (Baldock-style finite-difference T).

Covers:
- Recovery of a known constant temperature from a synthetic Emax sequence
  built as the analytical inverse of the estimator.
- NaN guard for short history (< 2 samples) and for constant Emax.
- SingleRun shape (scalar emax).
- VmapRuns shape (n_runs,).
- PmapVmapRuns shape (G, P) via the batcher's ``flatten``.
- Cadence: the ``interval`` argument controls log emission but the deque
  is updated every iteration.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.cli.monitor import TemperatureCallback
from jaxrens.sampling.batch_descriptor import (
    PmapVmapRuns,
    SingleRun,
    VmapRuns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_emax_sequence(
    n_live: int, n_cull: int, kB: float, T: float, n_iters: int, E0: float = 0.0
) -> np.ndarray:
    """Build an Emax sequence that should yield exactly temperature ``T``.

    Inverts ``beta = (L-1) log_alpha / dE``: for constant T across the
    deque, ``dE = (L-1) * log_alpha / beta`` with ``beta = 1/(kB*T)``,
    so ``E_i - E_0 = i * (log_alpha / beta) = i * kB * T * log_alpha``.
    Since ``log_alpha < 0`` and ``T > 0``, the increments are negative
    (Emax decreases monotonically — the correct NS behaviour).
    """
    log_alpha = np.log((n_live + 1 - n_cull) / (n_live + 1))
    step = kB * T * log_alpha  # negative
    return E0 + step * np.arange(n_iters, dtype=np.float64)


# ---------------------------------------------------------------------------
# Unit: pure formula via a SingleRun batcher
# ---------------------------------------------------------------------------


class TestSingleRunFormula:
    """SingleRun: scalar emax fed through the callback's full path."""

    def test_recovers_known_temperature(self, caplog):
        n_live, n_cull = 500, 1
        kB = 1.0
        T_true = 0.7
        lag = 50
        seq = _synthetic_emax_sequence(n_live, n_cull, kB, T_true, n_iters=200)

        cb = TemperatureCallback(
            n_live=n_live, n_cull=n_cull, lag=lag, interval=1, kB=kB,
        )
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})

        with caplog.at_level(logging.INFO, logger="jaxrens.cli.monitor"):
            for i, e in enumerate(seq):
                cb.on_iteration(
                    iteration=i,
                    ns_state=None,
                    info={"emax": jnp.asarray(e), "_batcher": SingleRun()},
                )

        # After the deque fills (i >= lag-1), the recovered T must match
        # the true T to within numerical precision.
        history = cb._histories[0]
        assert len(history) == lag
        T_rec = cb._compute_T(history)
        assert np.isclose(T_rec, T_true, rtol=1e-3), (
            f"Recovered T={T_rec} differs from injected T={T_true}"
        )

        # And the log message should have been emitted with the right value.
        last_log = [r for r in caplog.records if "T_est" in r.getMessage()][-1]
        assert f"{T_true:.3f}" in last_log.getMessage()

    def test_short_history_returns_nan(self):
        cb = TemperatureCallback(n_live=500, lag=50, interval=1, kB=1.0)
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})

        cb.on_iteration(
            iteration=0,
            ns_state=None,
            info={"emax": jnp.asarray(0.0), "_batcher": SingleRun()},
        )
        # Only one sample — formula needs ≥ 2.
        T = cb._compute_T(cb._histories[0])
        assert np.isnan(T)

    def test_constant_emax_returns_nan(self):
        cb = TemperatureCallback(n_live=500, lag=10, interval=1, kB=1.0)
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})

        for i in range(20):
            cb.on_iteration(
                iteration=i,
                ns_state=None,
                info={"emax": jnp.asarray(-1.0), "_batcher": SingleRun()},
            )
        T = cb._compute_T(cb._histories[0])
        assert np.isnan(T), "Constant Emax must yield NaN (dE=0)"

    def test_interval_controls_log_cadence_but_not_deque(self, caplog):
        cb = TemperatureCallback(n_live=500, lag=5, interval=10, kB=1.0)
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})

        with caplog.at_level(logging.INFO, logger="jaxrens.cli.monitor"):
            for i in range(25):
                cb.on_iteration(
                    iteration=i,
                    ns_state=None,
                    info={
                        "emax": jnp.asarray(-float(i)),
                        "_batcher": SingleRun(),
                    },
                )

        # Deque receives every sample regardless of interval.
        assert len(cb._histories[0]) == 5
        # Only iterations 0, 10, 20 emit a log line (interval=10).
        t_lines = [r for r in caplog.records if "T_est" in r.getMessage()]
        assert len(t_lines) == 3


# ---------------------------------------------------------------------------
# Batched shapes — VmapRuns and PmapVmapRuns
# ---------------------------------------------------------------------------


class TestBatchedShapes:
    def test_vmap_runs_independent_deques(self):
        n_live, n_cull = 500, 1
        kB = 1.0
        n_runs = 4
        T_per_run = np.array([0.5, 1.0, 1.5, 2.0])
        lag = 30

        # Build per-run synthetic sequences and stack to (n_iters, n_runs).
        n_iters = 100
        seqs = np.stack(
            [
                _synthetic_emax_sequence(
                    n_live, n_cull, kB, T, n_iters, E0=10.0 * k,
                )
                for k, T in enumerate(T_per_run)
            ],
            axis=1,
        )  # shape (n_iters, n_runs)

        batcher = VmapRuns(n_runs=n_runs)
        cb = TemperatureCallback(
            n_live=n_live, n_cull=n_cull, lag=lag, interval=1, kB=kB,
        )
        cb.on_start(ns_state=None, start_info={"_batcher": batcher})

        for i in range(n_iters):
            cb.on_iteration(
                iteration=i,
                ns_state=None,
                info={
                    "emax": jnp.asarray(seqs[i]),  # shape (n_runs,)
                    "_batcher": batcher,
                },
            )

        assert len(cb._histories) == n_runs
        for k, T_true in enumerate(T_per_run):
            T_rec = cb._compute_T(cb._histories[k])
            assert np.isclose(T_rec, T_true, rtol=1e-3), (
                f"Run {k}: recovered T={T_rec}, expected {T_true}"
            )

    def test_pmap_vmap_runs_flattens_to_n_runs(self):
        n_live, n_cull = 500, 1
        kB = 1.0
        G, P = 2, 3
        n_runs = G * P
        T_per_run = np.linspace(0.5, 2.0, n_runs)
        lag = 30
        n_iters = 80

        # Per-replica synthetic sequences shaped (n_iters, G, P).
        flat_seqs = np.stack(
            [
                _synthetic_emax_sequence(
                    n_live, n_cull, kB, T, n_iters, E0=10.0 * k,
                )
                for k, T in enumerate(T_per_run)
            ],
            axis=1,
        )  # (n_iters, n_runs)
        # Reshape to (n_iters, G, P) matching the batcher's shape_prefix.
        seqs = flat_seqs.reshape(n_iters, G, P)

        batcher = PmapVmapRuns(n_gpu=G, n_per_gpu=P)
        cb = TemperatureCallback(
            n_live=n_live, n_cull=n_cull, lag=lag, interval=1, kB=kB,
        )
        cb.on_start(ns_state=None, start_info={"_batcher": batcher})

        for i in range(n_iters):
            cb.on_iteration(
                iteration=i,
                ns_state=None,
                info={
                    "emax": jnp.asarray(seqs[i]),  # shape (G, P)
                    "_batcher": batcher,
                },
            )

        assert len(cb._histories) == n_runs
        for k, T_true in enumerate(T_per_run):
            T_rec = cb._compute_T(cb._histories[k])
            assert np.isclose(T_rec, T_true, rtol=1e-3), (
                f"Flat run {k}: recovered T={T_rec}, expected {T_true}"
            )

    def test_batched_log_line_reports_range(self, caplog):
        """When n_runs > 1 the log message uses the [min..max] median form."""
        batcher = VmapRuns(n_runs=3)
        cb = TemperatureCallback(n_live=500, lag=10, interval=1, kB=1.0)
        cb.on_start(ns_state=None, start_info={"_batcher": batcher})

        seqs = np.stack(
            [
                _synthetic_emax_sequence(500, 1, 1.0, T, n_iters=20)
                for T in (0.5, 1.0, 1.5)
            ],
            axis=1,
        )

        with caplog.at_level(logging.INFO, logger="jaxrens.cli.monitor"):
            for i in range(20):
                cb.on_iteration(
                    iteration=i,
                    ns_state=None,
                    info={"emax": jnp.asarray(seqs[i]), "_batcher": batcher},
                )

        t_lines = [r for r in caplog.records if "T_est=[" in r.getMessage()]
        assert len(t_lines) >= 1
        msg = t_lines[-1].getMessage()
        assert "median=" in msg


# ---------------------------------------------------------------------------
# Lazy init: callbacks called without on_start should still work
# ---------------------------------------------------------------------------


class TestLazyInit:
    def test_no_on_start_initialises_from_info(self):
        cb = TemperatureCallback(n_live=500, lag=10, interval=1, kB=1.0)
        # Skipping on_start: histories is None until first on_iteration.
        assert cb._histories is None
        batcher = VmapRuns(n_runs=2)
        cb.on_iteration(
            iteration=0,
            ns_state=None,
            info={
                "emax": jnp.asarray([-1.0, -2.0]),
                "_batcher": batcher,
            },
        )
        assert cb._histories is not None
        assert len(cb._histories) == 2

    def test_no_batcher_in_info_defaults_to_single_run(self):
        cb = TemperatureCallback(n_live=500, lag=10, interval=1, kB=1.0)
        # No batcher anywhere — emulates a minimal scripted caller.
        cb.on_iteration(
            iteration=0,
            ns_state=None,
            info={"emax": jnp.asarray(-1.0)},
        )
        assert cb._histories is not None
        assert len(cb._histories) == 1
