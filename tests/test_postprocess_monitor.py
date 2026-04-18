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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, must be before any other matplotlib import

import numpy as np
import pytest

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
                "n_dead": n_dead,
                "n_walkers": n_live,
            }
            save_checkpoint(path, ns_state)

        _write(tmp_path / "ns.checkpoint.h5", log_evidence_val=10.0)
        _write(tmp_path / "ns.final.checkpoint.h5", log_evidence_val=99.0)

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
                "n_dead": n_dead,
                "n_walkers": n_live,
            }
            save_checkpoint(d / "ns.final.checkpoint.h5", ns_state)
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


# ---------------------------------------------------------------------------
# Smoke test: load existing lj8_npt example output (ships in repo)
# ---------------------------------------------------------------------------

_LJ8_NPT_OUTPUT = Path(__file__).parent.parent / "experiments" / "examples" / "lj8_npt" / "output"


@pytest.mark.skipif(
    not _LJ8_NPT_OUTPUT.exists(),
    reason="lj8_npt example output not present",
)
class TestSmokeRealRun:
    """Load the pre-computed lj8_npt example artefacts and exercise the full stack."""

    def test_from_directory_lj8_npt(self):
        m = Monitor.from_directory(_LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke")
        assert m.n_dead > 0
        assert m.n_live > 0
        assert np.isfinite(m.log_evidence)
        assert m.is_npt  # lj8_npt is NPT

    def test_heat_capacity_on_real_run(self):
        import matplotlib.pyplot as plt

        m = Monitor.from_directory(_LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke")
        T = np.array([0.5, 1.0, 2.0])
        Cv = m.heat_capacity(T)
        assert Cv.shape == (3,)
        assert np.all(np.isfinite(Cv))

        ax = plot_heat_capacity(m, T)
        fig = ax.get_figure()
        fig.savefig("/tmp/lj8_smoke.png", dpi=72)
        plt.close("all")
        assert Path("/tmp/lj8_smoke.png").exists()

    def test_energy_trace_loaded(self):
        m = Monitor.from_directory(_LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke")
        assert m.energy_trace is not None
        assert m.iteration_trace is not None
        assert len(m.energy_trace) == len(m.iteration_trace)
