"""Tests for TemperatureCallback, the finite-difference temperature."""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.cli.monitor import TemperatureCallback
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun, VmapRuns


def _synthetic_emax_sequence(
    n_live: int,
    n_cull: int,
    kB: float,
    T: float,
    n_iters: int,
    E0: float = 0.0,
) -> np.ndarray:
    """Build an Emax sequence whose estimated temperature is exactly ``T``.

    It inverts ``beta = (L-1) log_alpha / dE``, which gives
    ``E_i - E_0 = i * kB * T * log_alpha``.  Since ``log_alpha < 0`` the
    sequence decreases, as Emax does in NS.
    """
    log_alpha = np.log((n_live + 1 - n_cull) / (n_live + 1))
    step = kB * T * log_alpha  # negative
    return E0 + step * np.arange(n_iters, dtype=np.float64)


class TestSingleRunFormula:
    """Check the temperature estimate and logging for a single run.

    The tests cover recovery of a known T, NaN for short or constant
    histories, and the log cadence set by ``interval``.
    """

    def test_recovers_known_temperature(self, caplog):
        n_live, n_cull = 500, 1
        kB = 1.0
        T_true = 0.7
        lag = 50
        seq = _synthetic_emax_sequence(n_live, n_cull, kB, T_true, n_iters=200)

        cb = TemperatureCallback(
            n_live=n_live,
            n_cull=n_cull,
            lag=lag,
            interval=1,
            kB=kB,
        )
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})

        with caplog.at_level(logging.INFO, logger="jaxrens.cli.monitor"):
            for i, e in enumerate(seq):
                cb.on_iteration(
                    iteration=i,
                    ns_state=None,
                    info={"emax": jnp.asarray(e), "_batcher": SingleRun()},
                )

        history = cb._histories[0]
        assert len(history) == lag
        T_rec = cb._compute_T(history)
        assert np.isclose(T_rec, T_true, rtol=1e-3)
        last_log = [r for r in caplog.records if "T_est" in r.getMessage()][-1]
        assert f"{T_true:.3f}" in last_log.getMessage()

    def test_short_history_returns_nan(self, caplog):
        cb = TemperatureCallback(n_live=500, lag=50, interval=1, kB=1.0)
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})

        with caplog.at_level(logging.INFO, logger="jaxrens.cli.monitor"):
            cb.on_iteration(
                iteration=0,
                ns_state=None,
                info={"emax": jnp.asarray(0.0), "_batcher": SingleRun()},
            )
        # The formula needs at least two samples.
        assert np.isnan(cb._compute_T(cb._histories[0]))
        assert "T_est=pending" in caplog.text

    def test_constant_emax_returns_nan(self, caplog):
        cb = TemperatureCallback(n_live=500, lag=10, interval=1, kB=1.0)
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})

        with caplog.at_level(logging.INFO, logger="jaxrens.cli.monitor"):
            for i in range(20):
                cb.on_iteration(
                    iteration=i,
                    ns_state=None,
                    info={"emax": jnp.asarray(-1.0), "_batcher": SingleRun()},
                )
        assert np.isnan(cb._compute_T(cb._histories[0]))
        assert "T_est=pending" in caplog.messages[-1]

    def test_window_follows_a_change_of_temperature(self):
        n_live, lag = 500, 50
        phase1 = _synthetic_emax_sequence(n_live, 1, 1.0, 0.5, n_iters=100)
        # The second phase continues from the last Emax with T = 2.
        step2 = _synthetic_emax_sequence(n_live, 1, 1.0, 2.0, n_iters=101)
        phase2 = phase1[-1] + step2[1:]

        cb = TemperatureCallback(n_live=n_live, lag=lag, interval=1, kB=1.0)
        cb.on_start(ns_state=None, start_info={"_batcher": SingleRun()})
        temperatures = []
        for i, e in enumerate(np.concatenate([phase1, phase2])):
            cb.on_iteration(
                iteration=i,
                ns_state=None,
                info={"emax": jnp.asarray(e), "_batcher": SingleRun()},
            )
            temperatures.append(cb._compute_T(cb._histories[0]))

        assert np.isclose(temperatures[99], 0.5, rtol=1e-3)
        assert np.isclose(temperatures[-1], 2.0, rtol=1e-3)
        # Halfway through the change the window mixes both phases.
        assert 0.5 < temperatures[124] < 2.0

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

        assert len(cb._histories[0]) == 5
        # Only iterations 0, 10 and 20 log.
        t_lines = [r for r in caplog.records if "T_est" in r.getMessage()]
        assert len(t_lines) == 3


class TestBatchedShapes:
    """Check that VmapRuns and PmapVmapRuns keep one history per run."""

    @pytest.mark.parametrize(
        "batcher",
        [VmapRuns(n_runs=4), PmapVmapRuns(n_gpu=2, n_per_gpu=3)],
        ids=["vmap_4", "pmap_vmap_2x3"],
    )
    def test_one_deque_per_run_recovers_each_temperature(self, batcher):
        n_live, n_cull, kB, lag, n_iters = 500, 1, 1.0, 30, 100
        n_runs = batcher.n_runs
        T_per_run = np.linspace(0.5, 2.0, n_runs)
        flat_seqs = np.stack(
            [
                _synthetic_emax_sequence(
                    n_live, n_cull, kB, T, n_iters, E0=10.0 * k
                )
                for k, T in enumerate(T_per_run)
            ],
            axis=1,
        )
        seqs = flat_seqs.reshape((n_iters,) + batcher.shape_prefix)

        cb = TemperatureCallback(
            n_live=n_live, n_cull=n_cull, lag=lag, interval=1, kB=kB
        )
        cb.on_start(ns_state=None, start_info={"_batcher": batcher})
        for i in range(n_iters):
            cb.on_iteration(
                iteration=i,
                ns_state=None,
                info={"emax": jnp.asarray(seqs[i]), "_batcher": batcher},
            )

        assert len(cb._histories) == n_runs
        for k, T_true in enumerate(T_per_run):
            T_rec = cb._compute_T(cb._histories[k])
            assert np.isclose(T_rec, T_true, rtol=1e-3)

    def test_batched_log_line_reports_range(self, caplog):
        """With n_runs > 1 the log line reports a range and a median."""
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
        msg = t_lines[-1].getMessage()
        assert "T_est=[0.500..1.500]" in msg
        assert "median=1.000" in msg


class TestLazyInit:
    """Check that the callback works without on_start."""

    def test_no_on_start_initialises_from_info(self):
        cb = TemperatureCallback(n_live=500, lag=10, interval=1, kB=1.0)
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
        cb.on_iteration(
            iteration=0,
            ns_state=None,
            info={"emax": jnp.asarray(-1.0)},
        )
        assert cb._histories is not None
        assert len(cb._histories) == 1
