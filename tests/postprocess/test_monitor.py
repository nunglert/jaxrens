"""Tests for postprocess.monitor, postprocess.plotting, and postprocess.collection.

All Monitor tests use synthetic data (no real run required).
The Monitor.from_directory test synthesises a run directory via
save_checkpoint + EnergyLogger.

The smoke test (real lj8_npt run) is skipped when GPU/hardware
is unavailable: it tests that Monitor.from_directory works with
the pre-existing example output that ships in the repo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, must be before any other matplotlib import

import numpy as np
import pytest

from jaxrens.cli.monitor import (
    _format_reject_breakdown,
    EnergyCheckCallback,
    ProgressCallback,
)
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.collection import MonitorCollection
from jaxrens.postprocess import (
    plot_energy_trace,
    plot_free_energy,
    plot_heat_capacity,
    plot_log_evidence_trace,
    plot_partition_function,
)
from jaxrens.postprocess.thermodynamics import (
    calc_log_weights,
    heat_capacity as thermo_heat_capacity,
    expectation as thermo_expectation,
    partition_function as thermo_partition_function,
    free_energy as thermo_free_energy,
)


# ---------------------------------------------------------------------------
# _format_reject_breakdown unit tests
# ---------------------------------------------------------------------------

class TestFormatRejectBreakdown:
    """Unit tests for the reject-reason column formatter in monitor.py.

    Percentages use fixed-width format ``{pct:>3.0f}%`` (4 chars incl. ``%``):
    - 100% -> ``"100%"`` (no leading space)
    - 60%  -> ``" 60%"`` (one leading space)
    - 7%   -> ``"  7%"`` (two leading spaces)
    """

    def test_backward_compat_none_reasons_used(self):
        """counts=[10,3,2,0] with reasons_used=None -> all three columns (backward compat).

        Each percentage token is 4 chars: ``{label}={pct:>3.0f}%``.
        """
        result = _format_reject_breakdown([10, 3, 2, 0], reasons_used=None)
        # E= 60%, C= 40%, P=  0%  (fixed-width, right-aligned in 3 chars)
        assert "E= 60%" in result
        assert "C= 40%" in result
        assert "P=  0%" in result

    def test_all_accepted_returns_empty(self):
        """counts=[10,0,0,0] with any reasons_used -> empty string."""
        assert _format_reject_breakdown([10, 0, 0, 0], reasons_used=frozenset({"energy"})) == ""
        assert _format_reject_breakdown([10, 0, 0, 0], reasons_used=None) == ""
        assert _format_reject_breakdown([10, 0, 0, 0], reasons_used=frozenset({"energy", "cell", "prior"})) == ""

    def test_all_three_reasons(self):
        """counts=[5,3,1,1] with all three reasons -> E= 60% C= 20% P= 20%."""
        result = _format_reject_breakdown(
            [5, 3, 1, 1], reasons_used=frozenset({"energy", "cell", "prior"})
        )
        assert "E= 60%" in result
        assert "C= 20%" in result
        assert "P= 20%" in result

    def test_unknown_reject_flagged(self):
        """If rejects appear outside declared reasons, ???= flag is appended."""
        # 5 rejects in cell bucket (2) but reasons_used only says energy (1)
        result = _format_reject_breakdown(
            [10, 0, 5, 0], reasons_used=frozenset({"energy"})
        )
        # E=  0% from declared, ???=5 from undeclared
        assert "???" in result

    def test_pct_token_width_is_four_chars(self):
        """Every percentage token (label + '=' + digits + '%') must be exactly 5 chars.

        Format: ``{label}={pct:>3.0f}%`` -> e.g. ``"E= 60%"`` (6 chars total for label+token).
        The token itself ``= 60%`` is 5 chars.  We test by splitting on spaces and
        inspecting the numeric portion widths.
        """
        # 53 energy rejects, 47 cell rejects out of 100 total rejects (0 accepted)
        result = _format_reject_breakdown(
            [0, 53, 47, 0], reasons_used=frozenset({"energy", "cell"})
        )
        # Should contain "E= 53%" and "C= 47%"
        assert "E= 53%" in result
        assert "C= 47%" in result

    def test_single_digit_pct_padded(self):
        """Single-digit percent is padded to two leading spaces: ``P=  7%``."""
        # 1 reject in prior out of 14 total
        result = _format_reject_breakdown(
            [1, 7, 6, 1], reasons_used=frozenset({"energy", "cell", "prior"})
        )
        assert "P=  7%" in result


# ---------------------------------------------------------------------------
# ProgressCallback column-alignment tests
# ---------------------------------------------------------------------------

def _capture_progress_lines(info: dict, batched: bool = False) -> list[str]:
    """Run ProgressCallback.on_iteration with a synthetic ns_state and capture the log output.

    Returns the list of lines from the single logger.info call emitted by on_iteration.
    Uses a plain dict for ns_state because _is_batched has a dict branch (.get()).
    """
    import jax.numpy as jnp

    # _is_batched uses .get() when ns_state is not an NSState instance.
    if batched:
        ns_state = {"log_evidence": jnp.array([1.0, 1.1, 1.05])}
    else:
        ns_state = {"log_evidence": jnp.array(1.0)}

    cb = ProgressCallback(info_interval=1)

    # Capture logger output via a handler
    records = []

    class _RecordHandler(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = _RecordHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    target_logger = logging.getLogger("jaxrens.cli.monitor")
    target_logger.addHandler(handler)
    old_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        cb.on_iteration(100, ns_state, info)
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)

    assert records, "ProgressCallback.on_iteration did not emit any log message"
    # The callback emits a single multi-line message; split it.
    return records[0].split("\n")


def _acc_col_offset(row: str) -> int:
    """Return the character index of 'acc=' within a per-move row string."""
    idx = row.find("acc=")
    assert idx != -1, f"'acc=' not found in row: {repr(row)}"
    return idx


class TestProgressCallbackAlignment:
    """Assert that acc= starts at the same column across all per-move rows."""

    def _make_single_run_info(self, n_moves: int = 3, with_rejects: bool = True):
        """Build a minimal info dict for single-run mode."""
        import jax.numpy as jnp

        move_names = ["galilean", "volume", "shear_strain"]
        ss = jnp.array([5.0, 0.011, 0.5])
        acc = jnp.array([0.62, 0.10, 0.50])

        if with_rejects:
            # galilean: energy rejects only; volume: energy+cell; shear: energy only
            rc = jnp.array([
                [620, 380, 0, 0],    # galilean: 38% energy reject
                [100, 90, 810, 0],   # volume: energy + cell rejects
                [500, 500, 0, 0],    # shear: energy rejects
            ])
            move_reject_reasons = (
                frozenset({"energy"}),
                frozenset({"energy", "cell"}),
                frozenset({"energy"}),
            )
        else:
            rc = None
            move_reject_reasons = None

        return {
            "emax": -102.3,
            "step_sizes_per_move": ss,
            "move_names": move_names,
            "n_accepted_per_move": jnp.array([620, 100, 500]),
            "n_proposed_per_move": jnp.array([1000, 1000, 1000]),
            "reject_reason_counts_per_move": rc,
            "move_reject_reasons": move_reject_reasons,
        }

    def _make_batched_info(self, n_runs: int = 3, n_moves: int = 3):
        """Build a minimal info dict for batched (multi-run) mode."""
        import jax.numpy as jnp

        move_names = ["galilean", "volume", "shear_strain"]
        # shape (n_runs, n_moves)
        ss = jnp.ones((n_runs, n_moves)) * jnp.array([5.0, 0.011, 0.5])
        acc = jnp.ones((n_runs, n_moves)) * jnp.array([0.62, 0.10, 0.50])
        rc = jnp.ones((n_runs, n_moves, 4), dtype=jnp.int32) * jnp.array([500, 300, 200, 0])

        return {
            "emax": jnp.array([-102.3, -102.1, -102.5]),
            "step_sizes_per_move": ss,
            "move_names": move_names,
            "n_accepted_per_move": acc * 1000,
            "n_proposed_per_move": jnp.ones((n_runs, n_moves), dtype=jnp.int32) * 1000,
            "reject_reason_counts_per_move": rc,
            "move_reject_reasons": (
                frozenset({"energy"}),
                frozenset({"energy", "cell"}),
                frozenset({"energy"}),
            ),
        }

    def test_single_run_with_rejects_acc_aligned(self):
        """All per-move rows in single-run output have acc= at the same column."""
        info = self._make_single_run_info(with_rejects=True)
        lines = _capture_progress_lines(info, batched=False)
        move_rows = [l for l in lines if "acc=" in l]
        assert len(move_rows) == 3, f"Expected 3 per-move rows, got: {move_rows}"
        offsets = [_acc_col_offset(r) for r in move_rows]
        assert len(set(offsets)) == 1, (
            f"acc= column offset differs across rows: {offsets}\nRows:\n"
            + "\n".join(move_rows)
        )

    def test_single_run_reject_and_no_reject_same_column(self):
        """Rows with and without reject suffix must have acc= at the same column.

        Synthesise: galilean (no reject because 100% accepted), volume (with reject).
        """
        import jax.numpy as jnp

        info = {
            "emax": -50.0,
            "step_sizes_per_move": jnp.array([5.0, 0.011]),
            "move_names": ["galilean", "volume"],
            "n_accepted_per_move": jnp.array([1000, 200]),
            "n_proposed_per_move": jnp.array([1000, 1000]),
            "reject_reason_counts_per_move": jnp.array([
                [1000, 0, 0, 0],   # galilean: all accepted -> no reject suffix
                [200, 500, 300, 0], # volume: energy+cell rejects
            ]),
            "move_reject_reasons": (
                frozenset({"energy"}),
                frozenset({"energy", "cell"}),
            ),
        }
        lines = _capture_progress_lines(info, batched=False)
        move_rows = [l for l in lines if "acc=" in l]
        assert len(move_rows) == 2
        offsets = [_acc_col_offset(r) for r in move_rows]
        assert len(set(offsets)) == 1, (
            f"acc= column not aligned between galilean (no-reject) and volume (with-reject):\n"
            + "\n".join(f"  offset={o}: {repr(r)}" for o, r in zip(offsets, move_rows))
        )
        # Additionally verify that galilean row has no "reject:" and volume row does.
        assert "reject:" not in move_rows[0], "galilean should have no reject suffix"
        assert "reject:" in move_rows[1], "volume should have reject suffix"

    def test_batched_acc_aligned(self):
        """All per-move rows in batched output have acc= at the same column."""
        info = self._make_batched_info()
        lines = _capture_progress_lines(info, batched=True)
        move_rows = [l for l in lines if "acc=" in l]
        assert len(move_rows) == 3
        offsets = [_acc_col_offset(r) for r in move_rows]
        assert len(set(offsets)) == 1, (
            f"acc= column differs in batched rows: {offsets}\nRows:\n"
            + "\n".join(move_rows)
        )

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_monitor(
    n_dead: int = 100,
    n_live: int = 20,
    seed: int = 0,
    label: str = "test",
    with_volumes: bool = False,
    with_trace: bool = False,
) -> Monitor:
    """Build a Monitor from synthetic data."""
    rng = np.random.default_rng(seed)
    dead_energies = np.sort(rng.uniform(0.0, 10.0, n_dead))
    live_energies = rng.uniform(8.0, 12.0, n_live)
    dead_volumes = rng.uniform(5.0, 15.0, n_dead) if with_volumes else None
    live_volumes = rng.uniform(5.0, 15.0, n_live) if with_volumes else None

    energy_trace = rng.uniform(0.0, 10.0, n_dead) if with_trace else None
    iteration_trace = np.arange(n_dead) if with_trace else None

    return Monitor(
        dead_energies=dead_energies,
        dead_volumes=dead_volumes,
        live_energies=live_energies,
        live_volumes=live_volumes,
        log_evidence=42.0,
        iteration=n_dead,
        n_live=n_live,
        n_cull=1,
        label=label,
        energy_trace=energy_trace,
        iteration_trace=iteration_trace,
    )


# ---------------------------------------------------------------------------
# Monitor unit tests
# ---------------------------------------------------------------------------

class TestMonitorConstruction:
    def test_basic_construction(self):
        m = _make_monitor()
        assert m.n_dead == 100
        assert m.n_live == 20
        assert m.label == "test"
        assert m.log_evidence == 42.0

    def test_n_dead_property(self):
        m = _make_monitor(n_dead=50)
        assert m.n_dead == 50
        assert m.dead_energies.shape == (50,)

    def test_is_npt_false_when_no_volumes(self):
        m = _make_monitor(with_volumes=False)
        assert m.is_npt is False

    def test_is_npt_true_when_volumes_present(self):
        m = _make_monitor(with_volumes=True)
        assert m.is_npt is True
        assert m.dead_volumes is not None
        assert m.dead_volumes.shape == (m.n_dead,)

    def test_repr_contains_label_ndead_logz(self):
        m = _make_monitor(label="mylabel")
        r = repr(m)
        assert "mylabel" in r
        assert str(m.n_dead) in r
        assert "log_Z" in r


class TestMonitorObservables:
    """Verify observable methods produce correct shapes and delegate correctly."""

    @pytest.fixture
    def monitor(self):
        return _make_monitor(n_dead=200, n_live=30, seed=7)

    def test_log_z_scalar_input(self, monitor):
        result = monitor.log_Z(1.0)
        assert np.isscalar(result) or np.asarray(result).ndim == 0
        assert np.isfinite(result)

    def test_log_z_array_input_shape(self, monitor):
        T = np.array([0.5, 1.0, 2.0])
        result = monitor.log_Z(T)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_heat_capacity_matches_thermodynamics(self, monitor):
        """Monitor.heat_capacity must match a direct call to thermodynamics.heat_capacity."""
        import jax.numpy as jnp

        T = np.array([0.5, 1.0, 2.0])
        betas = 1.0 / T

        dead_e = jnp.asarray(monitor.dead_energies)
        live_e = jnp.asarray(monitor.live_energies)

        expected = np.array([
            float(thermo_heat_capacity(float(b), dead_e, live_e, n_live=monitor.n_live))
            for b in betas
        ])
        result = monitor.heat_capacity(T)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_heat_capacity_shape(self, monitor):
        T = np.linspace(0.1, 5.0, 10)
        result = monitor.heat_capacity(T)
        assert result.shape == (10,)
        assert np.all(np.isfinite(result))

    def test_expectation_matches_thermodynamics(self, monitor):
        """Monitor.expectation with obs=dead_energies must be consistent."""
        import jax.numpy as jnp

        T = np.array([1.0, 2.0])
        betas = 1.0 / T
        obs = monitor.dead_energies  # per-dead-point observable

        dead_e = jnp.asarray(monitor.dead_energies)
        live_e = jnp.asarray(monitor.live_energies)
        # live obs padded with mean(obs) — same as Monitor.expectation
        live_obs_val = float(np.mean(obs))
        live_obs = jnp.full(live_e.shape[0], live_obs_val)
        obs_full = jnp.concatenate([jnp.asarray(obs), live_obs])

        expected = np.array([
            float(thermo_expectation(obs_full, float(b), dead_e, live_e, n_live=monitor.n_live))
            for b in betas
        ])
        result = monitor.expectation(obs, T)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_expectation_wrong_shape_raises(self, monitor):
        with pytest.raises(ValueError, match="n_dead"):
            monitor.expectation(np.ones(monitor.n_dead + 5), np.array([1.0]))

    def test_expectation_shape(self, monitor):
        obs = np.ones(monitor.n_dead)
        T = np.array([0.5, 1.0, 2.0, 5.0])
        result = monitor.expectation(obs, T)
        assert result.shape == (4,)

    def test_free_energy_shape_and_finite(self, monitor):
        T = np.array([0.5, 1.0, 2.0])
        result = monitor.free_energy(T)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_free_energy_consistent_with_log_z(self, monitor):
        """F = -log_Z / beta."""
        T = np.array([1.0, 2.0])
        betas = 1.0 / T
        log_Z = monitor.partition_function(T)
        F = monitor.free_energy(T)
        expected = -log_Z / betas
        np.testing.assert_allclose(F, expected, rtol=1e-6)

    def test_partition_function_shape(self, monitor):
        T = np.linspace(0.1, 5.0, 8)
        result = monitor.partition_function(T)
        assert result.shape == (8,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Monitor.from_directory
# ---------------------------------------------------------------------------

class TestMonitorFromDirectory:
    """Synthesise a run directory and round-trip through from_directory."""

    def test_from_directory_loads_correctly(self, tmp_path):
        """Construct a checkpoint + energies file, then load with from_directory."""
        from jaxrens.io.checkpoint import save_checkpoint
        from jaxrens.io.energy_log import EnergyLogger

        n_live = 10
        n_dead = 20
        rng = np.random.default_rng(42)

        dead_e = np.sort(rng.uniform(0.0, 5.0, n_dead))
        live_e = rng.uniform(4.0, 6.0, n_live)
        positions = rng.uniform(0.0, 1.0, (n_live, 4, 3))
        types = np.zeros(n_live, dtype=np.int32)

        # Build a minimal ns_state dict matching save_checkpoint expectations.
        import jax.numpy as jnp

        # dead_positions needs to be of size n_dead
        dead_positions = rng.uniform(0.0, 1.0, (n_dead, 4, 3))

        ns_state = {
            "positions": jnp.asarray(positions),
            "types": jnp.asarray(types),
            "energies": jnp.asarray(live_e),
            "cells": None,
            "dead_energies": jnp.asarray(
                np.concatenate([dead_e, np.full(1000, np.inf)])
            ),
            "dead_positions": jnp.asarray(
                np.concatenate([dead_positions, np.zeros((1000, 4, 3))], axis=0)
            ),
            "dead_volumes": None,
            "live_volumes": None,
            "log_evidence": 55.5,
            "iteration": n_dead,
            "emax": float(live_e.max()),
            "n_dead": n_dead,
            "n_walkers": n_live,
        }

        ckpt_path = tmp_path / "ns.final.checkpoint.h5"
        save_checkpoint(ckpt_path, ns_state, symbol_map={0: "Ar"})

        # Write an energies file.
        energy_logger = EnergyLogger(
            tmp_path / "ns.energies",
            n_walkers=n_live,
            n_cull=1,
        )
        energy_logger.write_header()
        for i, e in enumerate(dead_e):
            energy_logger.write_entry(i, float(e))
        energy_logger.close()

        m = Monitor.from_directory(tmp_path, prefix="ns")
        assert m.n_dead == n_dead
        assert m.n_live == n_live
        np.testing.assert_allclose(m.dead_energies, dead_e, rtol=1e-6)
        np.testing.assert_allclose(m.live_energies, live_e, rtol=1e-6)
        assert m.log_evidence == pytest.approx(55.5, rel=1e-4)
        assert m.energy_trace is not None
        assert len(m.energy_trace) == n_dead
        assert m.symbol_map == {0: "Ar"}

    def test_from_directory_prefers_final(self, tmp_path):
        """Prefers .final.checkpoint.h5 over .checkpoint.h5."""
        from jaxrens.io.checkpoint import save_checkpoint
        from jaxrens.io.energy_log import EnergyLogger
        import jax.numpy as jnp

        rng = np.random.default_rng(0)
        n_live, n_dead = 5, 10

        def _write(path, log_evidence_val):
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
                "log_evidence": log_evidence_val,
                "iteration": n_dead,
                "emax": float(live_e.max()),
                "n_dead": n_dead,
                "n_walkers": n_live,
            }
            save_checkpoint(path, ns_state)

        _write(tmp_path / "ns.checkpoint.h5", log_evidence_val=10.0)
        _write(tmp_path / "ns.final.checkpoint.h5", log_evidence_val=99.0)

        energy_logger = EnergyLogger(
            tmp_path / "ns.energies", n_walkers=n_live, n_cull=1,
        )
        energy_logger.write_header()
        for i, e in enumerate(np.sort(rng.uniform(0, 5, n_dead))):
            energy_logger.write_entry(i, float(e))
        energy_logger.close()

        m = Monitor.from_directory(tmp_path, prefix="ns", prefer_final=True)
        assert m.log_evidence == pytest.approx(99.0, rel=1e-4)

    def test_from_directory_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Monitor.from_directory(tmp_path, prefix="ns")


# ---------------------------------------------------------------------------
# Plotting tests
# ---------------------------------------------------------------------------

class TestPlotting:
    @pytest.fixture
    def monitor(self):
        return _make_monitor(n_dead=50, n_live=10, with_trace=True, label="run1")

    def test_plot_energy_trace_returns_axes(self, monitor):
        import matplotlib.pyplot as plt
        ax = plot_energy_trace(monitor)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_plot_energy_trace_axis_labels(self, monitor):
        import matplotlib.pyplot as plt
        ax = plot_energy_trace(monitor)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close("all")

    def test_plot_energy_trace_reuses_ax(self, monitor):
        import matplotlib.pyplot as plt
        _, existing_ax = plt.subplots()
        returned = plot_energy_trace(monitor, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")

    def test_plot_energy_trace_no_trace_raises(self):
        m = _make_monitor(with_trace=False)
        with pytest.raises(ValueError, match="energy_trace"):
            plot_energy_trace(m)

    def test_plot_log_evidence_trace_returns_axes(self, monitor):
        import matplotlib.pyplot as plt
        ax = plot_log_evidence_trace(monitor)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_plot_log_evidence_trace_axis_labels(self, monitor):
        import matplotlib.pyplot as plt
        ax = plot_log_evidence_trace(monitor)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close("all")

    def test_plot_log_evidence_trace_reuses_ax(self, monitor):
        import matplotlib.pyplot as plt
        _, existing_ax = plt.subplots()
        returned = plot_log_evidence_trace(monitor, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")

    def test_plot_heat_capacity_returns_axes(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        ax = plot_heat_capacity(monitor, T)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_plot_heat_capacity_axis_labels(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        ax = plot_heat_capacity(monitor, T)
        assert "T" in ax.get_xlabel()
        assert ax.get_ylabel() != ""
        plt.close("all")

    def test_plot_heat_capacity_reuses_ax(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        _, existing_ax = plt.subplots()
        returned = plot_heat_capacity(monitor, T, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")

    def test_plot_partition_function_returns_axes(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        ax = plot_partition_function(monitor, T)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_plot_partition_function_axis_labels(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        ax = plot_partition_function(monitor, T)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close("all")

    def test_plot_partition_function_reuses_ax(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        _, existing_ax = plt.subplots()
        returned = plot_partition_function(monitor, T, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")

    def test_plot_free_energy_returns_axes(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        ax = plot_free_energy(monitor, T)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_plot_free_energy_axis_labels(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        ax = plot_free_energy(monitor, T)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close("all")

    def test_plot_free_energy_reuses_ax(self, monitor):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        _, existing_ax = plt.subplots()
        returned = plot_free_energy(monitor, T, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")


# ---------------------------------------------------------------------------
# MonitorCollection tests
# ---------------------------------------------------------------------------

class TestMonitorCollection:
    @pytest.fixture
    def three_monitors(self):
        return [
            _make_monitor(n_dead=60, n_live=15, seed=i, label=f"run{i}", with_trace=True)
            for i in range(3)
        ]

    @pytest.fixture
    def collection(self, three_monitors):
        return MonitorCollection(three_monitors)

    def test_len(self, collection):
        assert len(collection) == 3

    def test_iter(self, collection):
        labels = [m.label for m in collection]
        assert labels == ["run0", "run1", "run2"]

    def test_getitem_by_index(self, collection):
        assert collection[0].label == "run0"
        assert collection[2].label == "run2"

    def test_getitem_by_label(self, collection):
        assert collection["run1"].label == "run1"

    def test_getitem_missing_label_raises(self, collection):
        with pytest.raises(KeyError):
            collection["nonexistent"]

    def test_add(self, collection):
        new_m = _make_monitor(label="run99")
        collection.add(new_m)
        assert len(collection) == 4
        assert collection["run99"].label == "run99"

    def test_remove(self, collection):
        collection.remove("run1")
        assert len(collection) == 2
        with pytest.raises(KeyError):
            collection["run1"]

    def test_remove_missing_raises(self, collection):
        with pytest.raises(KeyError):
            collection.remove("ghost")

    def test_summary_structure(self, collection):
        s = collection.summary()
        assert set(s.keys()) == {"labels", "n_dead", "log_evidence", "is_npt"}
        assert len(s["labels"]) == 3
        assert len(s["n_dead"]) == 3
        assert len(s["log_evidence"]) == 3
        assert len(s["is_npt"]) == 3

    def test_summary_values(self, collection):
        s = collection.summary()
        assert s["labels"] == ["run0", "run1", "run2"]
        for npt_val in s["is_npt"]:
            assert npt_val is False

    def test_plot_heat_capacity_three_lines(self, collection):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        ax = collection.plot_heat_capacity(T)
        assert len(ax.get_lines()) == 3
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert "run0" in line_labels
        assert "run1" in line_labels
        assert "run2" in line_labels
        plt.close("all")

    def test_plot_heat_capacity_reuses_ax(self, collection):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 5)
        _, existing_ax = plt.subplots()
        returned = collection.plot_heat_capacity(T, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")

    def test_plot_partition_function_overlay(self, collection):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 4)
        ax = collection.plot_partition_function(T)
        assert len(ax.get_lines()) == 3
        plt.close("all")

    def test_plot_free_energy_overlay(self, collection):
        import matplotlib.pyplot as plt
        T = np.linspace(0.5, 5.0, 4)
        ax = collection.plot_free_energy(T)
        assert len(ax.get_lines()) == 3
        plt.close("all")

    def test_from_directories_round_trip(self, tmp_path):
        """from_directories loads each dir into a Monitor."""
        from jaxrens.io.checkpoint import save_checkpoint
        from jaxrens.io.energy_log import EnergyLogger
        import jax.numpy as jnp

        dirs = []
        for i in range(2):
            d = tmp_path / f"run{i}"
            d.mkdir()
            rng = np.random.default_rng(i)
            n_live, n_dead = 5, 8

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
                "log_evidence": float(i * 10),
                "iteration": n_dead,
                "emax": float(live_e.max()),
                "n_dead": n_dead,
                "n_walkers": n_live,
            }
            save_checkpoint(d / "ns.final.checkpoint.h5", ns_state)

            energy_logger = EnergyLogger(
                d / "ns.energies", n_walkers=n_live, n_cull=1,
            )
            energy_logger.write_header()
            for j, e in enumerate(dead_e):
                energy_logger.write_entry(j, float(e))
            energy_logger.close()

            dirs.append(d)

        coll = MonitorCollection.from_directories(dirs, labels=["a", "b"], prefix="ns")
        assert len(coll) == 2
        assert coll[0].label == "a"
        assert coll[1].label == "b"

    def test_from_directories_label_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="len"):
            MonitorCollection.from_directories(
                [tmp_path / "x", tmp_path / "y"],
                labels=["only_one"],
                prefix="ns",
            )


class TestFromMultiRunDirectoryConfigInference:
    """Auto-detection of prefix + pressure labels from ``config.yaml``."""

    @staticmethod
    def _write_multi_run_artefacts(
        out_dir: Path,
        prefix: str,
        n_runs: int,
        n_live: int = 4,
        n_dead_per_run: int = 6,
    ) -> None:
        """Build a minimal ``(n_runs,)``-shaped checkpoint plus per-replica
        ``.energies`` logs so ``from_multi_run_directory`` can iterate the
        replicas.  No physics — just shapes the loader expects.
        """
        import h5py

        from jaxrens.io.energy_log import EnergyLogger

        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        live_e = rng.uniform(-1.0, 1.0, (n_runs, n_live)).astype(np.float32)
        log_Z = rng.uniform(-1.0, 1.0, (n_runs,)).astype(np.float32)
        iteration = np.full((n_runs,), n_dead_per_run, dtype=np.int32)
        n_dead = np.full((n_runs,), n_dead_per_run, dtype=np.int32)

        n_atoms = 2
        positions = rng.uniform(0.0, 1.0, (n_runs, n_live, n_atoms, 3)).astype(np.float32)
        types = np.zeros((n_runs, n_live, n_atoms), dtype=np.int32)
        ckpt_path = out_dir / f"{prefix}.final.checkpoint.h5"
        with h5py.File(ckpt_path, "w") as f:
            f.create_dataset("positions", data=positions)
            f.create_dataset("types", data=types)
            f.create_dataset("energies", data=live_e)
            f.create_dataset("log_evidence", data=log_Z)
            f.create_dataset("iteration", data=iteration)
            f.create_dataset("n_dead", data=n_dead)
            f.attrs["n_walkers"] = n_live

        for i in range(n_runs):
            elog = EnergyLogger(
                out_dir / f"{prefix}.run{i:02d}.energies",
                n_walkers=n_live, n_cull=1,
            )
            elog.write_header()
            for k in range(n_dead_per_run):
                elog.write_entry(k, energy=float(-k), volume=10.0 + k)
            elog.close()

    def test_prefix_inferred_from_config(self, tmp_path):
        """``output.out_file_prefix`` is picked up when ``prefix`` is omitted."""
        out_dir = tmp_path / "output"
        prefix = "neuralil_si16"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=3)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert len(coll) == 3
        # Default labels (runNN) when ensemble section is absent.
        assert [m.label for m in coll] == ["run00", "run01", "run02"]

    def test_pressure_labels_and_metadata_from_config(self, tmp_path):
        """An NPT pressure list matching n_total drives labels and metadata."""
        out_dir = tmp_path / "output"
        prefix = "sweep"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=3)
        (tmp_path / "config.yaml").write_text(
            "output:\n"
            f"  out_file_prefix: {prefix}\n"
            "ensemble:\n"
            "  type: npt\n"
            "  pressure: [2.0, 4.0, 6.0]\n"
            "  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert [m.label for m in coll] == [
            "P= 2.00 GPa", "P= 4.00 GPa", "P= 6.00 GPa",
        ]
        assert [m.pressure_gpa for m in coll] == [2.0, 4.0, 6.0]

    def test_explicit_args_override_config(self, tmp_path):
        """Explicit ``prefix=`` and ``labels=`` win over config inference."""
        out_dir = tmp_path / "output"
        prefix = "mysweep"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [1.0, 2.0]\n  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(
            out_dir, prefix=prefix, labels=["foo", "bar"],
        )
        assert [m.label for m in coll] == ["foo", "bar"]
        # Pressure metadata still attaches when config matches n_total.
        assert [m.pressure_gpa for m in coll] == [1.0, 2.0]

    def test_falls_back_to_runNN_without_config(self, tmp_path):
        """No config.yaml present → default ``prefix='ns'`` and ``runNN`` labels."""
        out_dir = tmp_path / "output"
        self._write_multi_run_artefacts(out_dir, prefix="ns", n_runs=2)

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert [m.label for m in coll] == ["run00", "run01"]
        # No pressure metadata attached when there's no config to read.
        assert not hasattr(coll[0], "pressure_gpa")

    def test_explicit_config_path_overrides_discovery(self, tmp_path):
        """An explicit ``config=`` argument bypasses auto-discovery."""
        out_dir = tmp_path / "output"
        prefix = "explicit"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        config_path = tmp_path / "elsewhere.yaml"
        config_path.write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [3.0, 5.0]\n  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(
            out_dir, config=config_path,
        )
        assert [m.pressure_gpa for m in coll] == [3.0, 5.0]

    def test_plot_heatmap_smoke(self, tmp_path):
        """End-to-end: from_multi_run_directory + plot_heatmap yields one
        pcolormesh patch with the right cell count and axis labels.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import QuadMesh

        out_dir = tmp_path / "output"
        prefix = "heatmap_smoke"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=3)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [1.0, 4.0, 2.0]\n"
            "  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        T = np.linspace(0.5, 2.0, 5)
        ax = coll.plot_heatmap(T, "heat_capacity")

        # Exactly one QuadMesh laid down by pcolormesh.
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(meshes) == 1
        # Default fmt='PT' → x=P, y=T.
        assert ax.get_xlabel() == "P [GPa]"
        assert ax.get_ylabel() == "T"
        plt.close("all")

    def test_plot_heatmap_fmt_tp_swaps_axes(self, tmp_path):
        import matplotlib.pyplot as plt

        out_dir = tmp_path / "output"
        prefix = "fmt_tp"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [1.0, 2.0]\n"
            "  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        ax = coll.plot_heatmap(np.linspace(0.5, 2.0, 4), "heat_capacity", fmt="TP")
        assert ax.get_xlabel() == "T"
        assert ax.get_ylabel() == "P [GPa]"
        plt.close("all")

    def test_plot_heatmap_falls_back_to_indices(self, tmp_path):
        """Without per-monitor pressure metadata the y-axis is replica index."""
        import matplotlib.pyplot as plt

        out_dir = tmp_path / "output"
        self._write_multi_run_artefacts(out_dir, prefix="ns", n_runs=2)
        # No config.yaml → no pressure_gpa metadata.
        coll = MonitorCollection.from_multi_run_directory(out_dir)
        ax = coll.plot_heatmap(np.linspace(0.5, 2.0, 4), "heat_capacity")
        assert ax.get_xlabel() == "replica index"
        plt.close("all")

    def test_plot_heatmap_callable_observable(self, tmp_path):
        """A custom callable observable feeds the heatmap grid."""
        import matplotlib.pyplot as plt

        out_dir = tmp_path / "output"
        prefix = "callable_obs"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [1.0, 2.0]\n"
            "  pressure_units: gpa\n"
        )
        coll = MonitorCollection.from_multi_run_directory(out_dir)

        def constant_obs(monitor, T):
            return np.full(T.shape, float(monitor.pressure_gpa))

        ax = coll.plot_heatmap(
            np.linspace(0.5, 2.0, 4), constant_obs, cbar_label="P",
        )
        # The pcolormesh should have data spanning the pressure range.
        mesh = ax.collections[0]
        arr = mesh.get_array()
        assert float(arr.min()) == pytest.approx(1.0)
        assert float(arr.max()) == pytest.approx(2.0)
        plt.close("all")

    def test_plot_heatmap_unknown_observable_raises(self, tmp_path):
        out_dir = tmp_path / "output"
        self._write_multi_run_artefacts(out_dir, prefix="ns", n_runs=2)
        coll = MonitorCollection.from_multi_run_directory(out_dir)
        with pytest.raises(ValueError, match="Unknown observable"):
            coll.plot_heatmap(np.linspace(0.5, 2.0, 4), "not_a_real_observable")

    def test_adaptation_trace_loaded_into_collection(self, tmp_path):
        """``from_multi_run_directory`` should pick up ``<prefix>.adaptation.h5``
        and attach it to the collection (not to individual monitors)."""
        import h5py, json

        out_dir = tmp_path / "output"
        prefix = "adap"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)

        # Hand-craft a minimal adaptation.h5 matching the v3 schema.
        ad_path = out_dir / f"{prefix}.adaptation.h5"
        n_entries, n_runs, n_moves = 5, 2, 2
        with h5py.File(ad_path, "w") as f:
            f.attrs["adaptation_log_schema_version"] = 3
            f.attrs["n_moves"] = n_moves
            f.attrs["n_runs"] = n_runs
            f.attrs["move_names"] = json.dumps(["galilean", "volume"])
            f.create_dataset("iterations", data=np.arange(n_entries, dtype=np.int64))
            f.create_dataset(
                "step_sizes",
                data=np.linspace(0.01, 0.1, n_entries * n_runs * n_moves,
                                 dtype=np.float32).reshape(n_entries, n_runs, n_moves),
            )
            f.create_dataset(
                "acceptance_rates",
                data=np.full((n_entries, n_runs, n_moves), 0.4, dtype=np.float32),
            )
            f.create_dataset(
                "n_evaluations",
                data=np.full((n_entries, n_runs, n_moves), 100, dtype=np.int64),
            )
            f.create_dataset(
                "n_grad_evaluations",
                data=np.full((n_entries, n_runs, n_moves), 50, dtype=np.int64),
            )
            stats = f.create_group("adjustment_stats")
            stats.create_dataset(
                "n_rounds",
                data=np.ones((n_entries, n_runs, n_moves), dtype=np.int32),
            )
            stats.create_dataset(
                "converged",
                data=np.ones((n_entries, n_runs, n_moves), dtype=bool),
            )
            stats.create_dataset(
                "cap_hits",
                data=np.zeros((n_entries, n_runs, n_moves), dtype=np.int32),
            )
            stats.create_dataset(
                "floor_hits",
                data=np.zeros((n_entries, n_runs, n_moves), dtype=np.int32),
            )
            stats.create_dataset(
                "bracket_detected",
                data=np.zeros((n_entries, n_runs, n_moves), dtype=bool),
            )
            stats.create_dataset(
                "reject_reason_counts",
                data=np.zeros((n_entries, n_runs, n_moves, 4), dtype=np.int32),
            )
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert coll.adaptation_trace is not None
        assert coll.adaptation_trace.n_runs == 2
        assert coll.adaptation_trace.n_moves == 2
        assert coll.adaptation_trace.step_sizes.shape == (n_entries, 2, 2)

    def test_plot_step_sizes_uses_collection_trace(self, tmp_path):
        """``MonitorCollection.plot_step_sizes`` should plot from the
        cohort-level adaptation trace, not from per-monitor traces."""
        import h5py, json
        import matplotlib.pyplot as plt

        out_dir = tmp_path / "output"
        prefix = "ploth5"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        ad_path = out_dir / f"{prefix}.adaptation.h5"
        with h5py.File(ad_path, "w") as f:
            f.attrs["adaptation_log_schema_version"] = 3
            f.attrs["n_moves"] = 2
            f.attrs["n_runs"] = 2
            f.attrs["move_names"] = json.dumps(["a", "b"])
            f.create_dataset("iterations", data=np.arange(4, dtype=np.int64))
            f.create_dataset(
                "step_sizes",
                data=np.full((4, 2, 2), 0.05, dtype=np.float32),
            )
            f.create_dataset(
                "acceptance_rates",
                data=np.full((4, 2, 2), 0.4, dtype=np.float32),
            )
            f.create_dataset(
                "n_evaluations",
                data=np.full((4, 2, 2), 100, dtype=np.int64),
            )
            f.create_dataset(
                "n_grad_evaluations",
                data=np.full((4, 2, 2), 50, dtype=np.int64),
            )
            stats = f.create_group("adjustment_stats")
            stats.create_dataset("n_rounds", data=np.ones((4, 2, 2), dtype=np.int32))
            stats.create_dataset("converged", data=np.ones((4, 2, 2), dtype=bool))
            stats.create_dataset("cap_hits", data=np.zeros((4, 2, 2), dtype=np.int32))
            stats.create_dataset("floor_hits", data=np.zeros((4, 2, 2), dtype=np.int32))
            stats.create_dataset("bracket_detected", data=np.zeros((4, 2, 2), dtype=bool))
            stats.create_dataset("reject_reason_counts", data=np.zeros((4, 2, 2, 4), dtype=np.int32))
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        # per_run=False → one line per move (mean across replicas)
        ax = coll.plot_step_sizes(per_run=False)
        assert len(ax.get_lines()) == 2  # one per move
        plt.close("all")
        # per_run=True → one line per (move, replica)
        ax = coll.plot_step_sizes(per_run=True)
        assert len(ax.get_lines()) == 4
        plt.close("all")

    def test_plot_heatmap_low_level_shape_validation(self):
        """plot_heatmap rejects mismatched (n_P, n_T) shapes loudly."""
        import matplotlib.pyplot as plt
        from jaxrens.postprocess.plotting import plot_heatmap

        T = np.linspace(0.0, 1.0, 5)
        P = np.linspace(0.0, 1.0, 3)
        Z_bad = np.zeros((4, 5))  # n_P axis disagrees with len(P)=3
        with pytest.raises(ValueError, match=r"Z.shape"):
            plot_heatmap(T, P, Z_bad)
        plt.close("all")

    def test_pressure_list_length_mismatch_falls_back(self, tmp_path):
        """A pressure list whose length disagrees with n_total is ignored."""
        out_dir = tmp_path / "output"
        prefix = "mismatch"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=3)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            # 16-entry pressure list, but only 3 replicas — must not crash.
            "ensemble:\n  type: npt\n  pressure: "
            + "[" + ",".join(str(float(p)) for p in range(1, 17)) + "]\n"
            "  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert [m.label for m in coll] == ["run00", "run01", "run02"]
        assert not hasattr(coll[0], "pressure_gpa")


# ---------------------------------------------------------------------------
# Smoke test: load existing lj8_npt example output (ships in repo)
# ---------------------------------------------------------------------------

_LJ8_NPT_OUTPUT = Path(__file__).parent.parent / "experiments" / "examples" / "lj8_npt" / "output"


class TestSmokeRealRun:
    """Load the pre-computed lj8_npt example artefacts and exercise the full stack.

    The artefacts under ``experiments/examples/lj8_npt/output/`` are
    committed to the repo and produced by running ``jaxrens run -c
    experiments/examples/lj8_npt/config.yaml`` (a sub-second 8-atom LJ NPT
    NS, 200 iterations).
    """

    def test_from_directory_lj8_npt(self):
        m = Monitor.from_directory(_LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke")
        assert m.n_dead > 0
        assert m.n_live > 0
        assert np.isfinite(m.log_evidence)
        assert m.is_npt  # lj8_npt is NPT

    def test_heat_capacity_on_real_run(self, tmp_path):
        import matplotlib.pyplot as plt

        m = Monitor.from_directory(_LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke")
        T = np.array([0.5, 1.0, 2.0])
        Cv = m.heat_capacity(T)
        assert Cv.shape == (3,)
        assert np.all(np.isfinite(Cv))

        ax = plot_heat_capacity(m, T)
        fig = ax.get_figure()
        out_path = tmp_path / "lj8_smoke.png"
        fig.savefig(out_path, dpi=72)
        plt.close("all")
        assert out_path.exists()

    def test_energy_trace_loaded(self):
        m = Monitor.from_directory(_LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke")
        assert m.energy_trace is not None
        assert m.iteration_trace is not None
        assert len(m.energy_trace) == len(m.iteration_trace)


# ---------------------------------------------------------------------------
# Task B: ProgressCallback smoke test with PmapVmapRuns (ndim==2 log_evidence)
# ---------------------------------------------------------------------------


class TestProgressCallbackPmapVmap:
    """ProgressCallback handles PmapVmapRuns (log_evidence shape (G, P))."""

    def test_pmap_vmap_no_exception_and_logz_bracket(self):
        """on_iteration does not raise for (1,2)-shaped log_evidence."""
        import jax.numpy as jnp
        import logging

        from jaxrens.cli.monitor import ProgressCallback
        from jaxrens.sampling.batch_descriptor import PmapVmapRuns

        n_gpu, n_per_gpu = 1, 2
        descriptor = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)

        # Synthetic ns_state dict with (G, P)-shaped log_evidence
        ns_state = {"log_evidence": jnp.array([[5.1, 6.2]])}  # shape (1, 2)

        info = {
            "emax": jnp.array([[-10.0, -9.5]]),   # (G, P) emax
            "_batch": descriptor,
        }

        cb = ProgressCallback(info_interval=1)

        # Capture the log output
        records = []

        class _RecordHandler(logging.Handler):
            def emit(self, record):
                records.append(self.format(record))

        handler = _RecordHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        target_logger = logging.getLogger("jaxrens.cli.monitor")
        target_logger.addHandler(handler)
        old_level = target_logger.level
        target_logger.setLevel(logging.DEBUG)
        try:
            cb.on_iteration(0, ns_state, info)
        finally:
            target_logger.removeHandler(handler)
            target_logger.setLevel(old_level)

        assert records, "ProgressCallback.on_iteration emitted no log output"
        header = records[0].split("\n")[0]
        assert "log_Z=[" in header, (
            f"Expected 'log_Z=[...]' bracket format in header, got: {repr(header)}"
        )
        # Verify min/max are present (floats from (1,2) array)
        assert ".." in header, (
            f"Expected min..max format in log_Z bracket, got: {repr(header)}"
        )


class TestEnergyCheckCallbackPerReplica:
    """Per-replica monotonicity: a global-max reduction would mask the failure."""

    @staticmethod
    def _capture_warnings(cb_action):
        import jax.numpy as jnp  # local; jnp used downstream in test bodies
        records: list[str] = []

        class _Rec(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        h = _Rec()
        target = logging.getLogger("jaxrens.cli.monitor")
        target.addHandler(h)
        old = target.level
        target.setLevel(logging.WARNING)
        try:
            cb_action()
        finally:
            target.removeHandler(h)
            target.setLevel(old)
        return records

    def test_scalar_singlerun_descending_no_warn(self):
        cb = EnergyCheckCallback()
        msgs = self._capture_warnings(lambda: (
            cb.on_iteration(0, None, {"emax": np.array(10.0)}),
            cb.on_iteration(1, None, {"emax": np.array(8.0)}),
            cb.on_iteration(2, None, {"emax": np.array(5.0)}),
        ))
        assert msgs == []

    def test_scalar_singlerun_ascending_warns(self):
        cb = EnergyCheckCallback()
        msgs = self._capture_warnings(lambda: (
            cb.on_iteration(0, None, {"emax": np.array(10.0)}),
            cb.on_iteration(1, None, {"emax": np.array(11.0)}),
        ))
        assert len(msgs) == 1
        assert "non-monotonic" in msgs[0]

    def test_one_replica_ascends_others_descend_warns(self):
        """The bug-of-record case: global max would hide a single regression."""
        cb = EnergyCheckCallback()
        # 16 replicas, all descending — except replica 3 which jumps up.
        prev = np.linspace(50.0, 35.0, 16)            # descending across runs
        curr = np.linspace(48.0, 34.0, 16).copy()     # mostly descending
        curr[3] = prev[3] + 2.5                        # replica 3 ASCENDS
        msgs = self._capture_warnings(lambda: (
            cb.on_iteration(0, None, {"emax": prev}),
            cb.on_iteration(1, None, {"emax": curr}),
        ))
        assert len(msgs) == 1
        assert "non-monotonic on 1 replica" in msgs[0]
        assert "replica[3]" in msgs[0]

    def test_pmap_vmap_shape_warns(self):
        cb = EnergyCheckCallback()
        prev = np.array([[20.0, 19.0], [18.0, 17.0]])
        curr = np.array([[19.0, 19.5], [17.5, 16.5]])  # (0,1) ascends
        msgs = self._capture_warnings(lambda: (
            cb.on_iteration(0, None, {"emax": prev}),
            cb.on_iteration(1, None, {"emax": curr}),
        ))
        assert len(msgs) == 1
        assert "replica[0,1]" in msgs[0]

    def test_iteration_zero_never_warns(self):
        cb = EnergyCheckCallback()
        msgs = self._capture_warnings(lambda: cb.on_iteration(
            0, None, {"emax": np.array([[100.0, 200.0]])}
        ))
        assert msgs == []
