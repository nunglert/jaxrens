"""Tests for Monitor, MonitorCollection and the postprocess plots.

Most tests build a Monitor from synthetic arrays or a synthetic run
directory (save_checkpoint plus an .energies file). TestSmokeRealRun
loads the lj8_npt output committed under tests/_assets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, must be before any other matplotlib import

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh

from jaxrens.io.adaptation_log import AdaptationLogger
from jaxrens.io.checkpoint import save_checkpoint
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.postprocess import (
    plot_energy_trace,
    plot_free_energy,
    plot_heat_capacity,
    plot_log_evidence_trace,
    plot_partition_function,
)
from jaxrens.postprocess.collection import MonitorCollection
from jaxrens.postprocess.monitor import Monitor
from jaxrens.postprocess.plotting import plot_heatmap
from jaxrens.postprocess.thermodynamics import (
    expectation as thermo_expectation,
)
from jaxrens.postprocess.thermodynamics import (
    heat_capacity as thermo_heat_capacity,
)
from jaxrens.postprocess.thermodynamics import (
    partition_function as thermo_partition_function,
)
from jaxrens.postprocess.thermodynamics import (
    weighted_covariance,
    weighted_variance,
)


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


def _harmonic_monitor(n_dead: int = 3000, n_live: int = 100) -> Monitor:
    """Monitor for E = |x|^2 / 2 in 3D with dead points at their expected
    prior mass, in NS order (descending energy)."""
    x_dead = (n_live / (n_live + 1.0)) ** np.arange(1, n_dead + 1)
    x_live = x_dead[-1] * (np.arange(n_live) + 0.5) / n_live
    return Monitor(
        dead_energies=50.0 * x_dead ** (2.0 / 3.0),
        dead_volumes=None,
        live_energies=50.0 * x_live ** (2.0 / 3.0),
        live_volumes=None,
        log_evidence=0.0,
        iteration=n_dead,
        n_live=n_live,
    )


class TestMonitorEstimators:
    """Check the per-point estimators of Monitor on a harmonic run."""

    T = np.array([0.5, 1.0, 2.0])

    @pytest.fixture
    def m(self):
        return _harmonic_monitor(n_dead=2000, n_live=100)

    def test_weights(self, m):
        assert m.weights(1.0).shape == (m.n_points,)
        w = m.weights(self.T)
        assert w.shape == (3, m.n_points)
        np.testing.assert_allclose(w.sum(axis=1), 1.0)

    def test_live_values_are_used(self, m):
        """A full (n_points,) observable is averaged with its own live
        values, not with the dead-point mean."""
        obs = np.zeros(m.n_points)
        obs[m.n_dead :] = 1.0
        w = m.weights(self.T)
        np.testing.assert_allclose(
            m.expectation(obs, self.T), w[:, m.n_dead :].sum(axis=1)
        )
        # Far below the last dead energy (~1e-4) the live walkers carry
        # most of the weight.
        assert m.expectation(obs, 1e-8) > 0.5

    def test_vector_observable_shapes(self, m):
        vec = np.stack([m.point_energies, m.point_energies**2], axis=1)
        assert m.expectation(vec, self.T).shape == (3, 2)
        assert m.expectation(vec, 1.0).shape == (2,)
        assert m.variance(vec, self.T).shape == (3, 2)

    def test_mean_energy_and_heat_capacity(self, m):
        """<E> = 3 T / 2 and C = 3 / 2 for the 3D harmonic potential."""
        np.testing.assert_allclose(
            m.mean_energy(self.T), 1.5 * self.T, rtol=1e-3
        )
        np.testing.assert_allclose(
            self.T**-2 * m.variance(m.point_energies, self.T),
            m.heat_capacity(self.T),
            rtol=1e-12,
        )

    def test_covariance_is_temperature_derivative(self, m):
        """d<A>/dT = Cov(A, E) / T^2 (k_B = 1), checked by central finite
        difference for A = sin(E)."""
        a = np.sin(m.point_energies)
        T, dT = 1.0, 1e-3
        fd = (m.expectation(a, T + dT) - m.expectation(a, T - dT)) / (2 * dT)
        cov = m.covariance(a, m.point_energies, T) / T**2
        assert float(fd) == pytest.approx(float(cov), rel=1e-5)

    def test_entropy(self, m):
        """S = (<E> - F) / T, and dS/dT = C / T."""
        np.testing.assert_allclose(
            m.entropy(self.T),
            (m.mean_energy(self.T) - m.free_energy(self.T)) / self.T,
            rtol=1e-12,
        )
        T, dT = 1.0, 1e-3
        dS = (m.entropy(T + dT) - m.entropy(T - dT)) / (2 * dT)
        assert float(dS) == pytest.approx(
            float(m.heat_capacity(T)) / T, rel=1e-5
        )

    def test_entropy_k_b(self, m):
        """S scales with k_B at fixed beta."""
        np.testing.assert_allclose(
            m.entropy(self.T / 2.0, k_B=2.0),
            2.0 * m.entropy(self.T),
            rtol=1e-12,
        )

    def test_class_averages(self, m):
        low = m.point_energies < np.median(m.point_energies)
        e = m.point_energies
        p_low = m.class_probability(low, self.T)
        np.testing.assert_allclose(
            p_low + m.class_probability(~low, self.T), 1.0
        )
        np.testing.assert_allclose(
            m.conditional_expectation(e, self.T, mask=low),
            m.expectation(e * low, self.T) / p_low,
            rtol=1e-12,
        )
        with_nan = np.where(low, e, np.nan)
        np.testing.assert_allclose(
            m.conditional_expectation(with_nan, self.T),
            m.conditional_expectation(e, self.T, mask=low),
            rtol=1e-12,
        )

    def test_wrong_mask_length_raises(self, m):
        with pytest.raises(ValueError, match="n_points"):
            m.class_probability(np.ones(m.n_dead + 1, dtype=bool), 1.0)


class TestDeadOnlyObservables:
    """Values given on the dead points only are averaged over the dead
    points alone, with renormalised weights; the live walkers are left
    out, and live_weight tells how much weight that is."""

    T = np.array([0.5, 1.0, 2.0])

    @pytest.fixture
    def m(self):
        return _harmonic_monitor(n_dead=2000, n_live=100)

    @staticmethod
    def _dead_w(m, T):
        w = m.weights(T)[..., : m.n_dead]
        return w / w.sum(axis=-1, keepdims=True)

    def test_expectation_over_dead_points(self, m):
        obs = np.sin(m.dead_energies)
        np.testing.assert_allclose(
            m.expectation(obs, self.T),
            self._dead_w(m, self.T) @ obs,
            rtol=1e-10,
        )

    def test_no_warning(self, m):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.expectation(m.dead_energies, self.T)

    def test_vector_and_variance(self, m):
        vec = np.stack([m.dead_energies, m.dead_energies**2], axis=1)
        w = self._dead_w(m, self.T)
        np.testing.assert_allclose(
            m.expectation(vec, self.T), w @ vec, rtol=1e-10
        )
        np.testing.assert_allclose(
            m.variance(m.dead_energies, self.T),
            [weighted_variance(w[i], m.dead_energies) for i in range(3)],
            rtol=1e-10,
        )

    def test_mixed_covariance_uses_dead_points(self, m):
        """A full and a dead-only observable meet on the dead points."""
        full = np.cos(m.point_energies)
        dead = m.dead_energies
        w = self._dead_w(m, self.T)
        ref = [
            weighted_covariance(w[i], full[: m.n_dead], dead) for i in range(3)
        ]
        np.testing.assert_allclose(
            m.covariance(full, dead, self.T), ref, rtol=1e-10
        )

    def test_dead_only_mask(self, m):
        low = m.dead_energies < np.median(m.dead_energies)
        w = self._dead_w(m, self.T)
        np.testing.assert_allclose(
            m.class_probability(low, self.T), w @ low, rtol=1e-10
        )
        # A full observable with a dead-only mask: both on the dead points.
        np.testing.assert_allclose(
            m.conditional_expectation(m.point_energies, self.T, mask=low),
            (w @ (m.dead_energies * low)) / (w @ low),
            rtol=1e-10,
        )

    def test_finite_where_live_walkers_dominate(self, m):
        """Far below the last dead energy the dead weights are still
        finite and normalised, and the live weight is close to 1."""
        assert m.live_weight(1e-8) > 0.99
        out = m.expectation(m.dead_energies, 1e-8)
        assert np.isfinite(out)
        # The lowest dead points dominate: the result is near the minimum.
        assert out < np.quantile(m.dead_energies, 0.01)

    def test_live_weight(self, m):
        lw = m.live_weight(self.T)
        w = m.weights(self.T)
        assert lw.shape == (3,)
        np.testing.assert_allclose(lw, w[:, m.n_dead :].sum(axis=1))
        assert m.live_weight(1e-8) > m.live_weight(10.0)

    def test_two_dimensional_temperature_raises(self, m):
        with pytest.raises(ValueError, match="scalar or 1-D"):
            m.mean_energy(np.ones((2, 2)))

    def test_point_volumes(self):
        nvt = _make_monitor()
        npt = _make_monitor(with_volumes=True)
        assert nvt.point_volumes is None
        np.testing.assert_array_equal(
            npt.point_volumes,
            np.concatenate([npt.dead_volumes, npt.live_volumes]),
        )


class TestMonitorConstruction:
    """Check the Monitor attributes, is_npt and repr."""

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
    """Check that the Monitor observables have the right shapes and match
    the thermodynamics functions they wrap."""

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
        T = np.array([0.5, 1.0, 2.0])
        betas = 1.0 / T

        dead_e = jnp.asarray(monitor.dead_energies)
        live_e = jnp.asarray(monitor.live_energies)

        expected = np.array(
            [
                float(
                    thermo_heat_capacity(
                        float(b), dead_e, live_e, n_live=monitor.n_live
                    )
                )
                for b in betas
            ]
        )
        result = monitor.heat_capacity(T)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_heat_capacity_shape(self, monitor):
        T = np.linspace(0.1, 5.0, 10)
        result = monitor.heat_capacity(T)
        assert result.shape == (10,)
        assert np.all(np.isfinite(result))

    def test_expectation_matches_thermodynamics(self, monitor):
        """With values on every point, expectation is thermodynamics'
        expectation over dead and live points."""
        T = np.array([1.0, 2.0])
        obs = monitor.point_energies
        expected = [
            float(
                thermo_expectation(
                    obs,
                    1.0 / t,
                    monitor.dead_energies,
                    monitor.live_energies,
                    n_live=monitor.n_live,
                )
            )
            for t in T
        ]
        np.testing.assert_allclose(
            monitor.expectation(obs, T), expected, rtol=1e-12
        )

    def test_expectation_wrong_shape_raises(self, monitor):
        with pytest.raises(ValueError, match="n_dead"):
            monitor.expectation(np.ones(monitor.n_dead + 5), np.array([1.0]))

    def test_expectation_shape(self, monitor):
        obs = np.ones(monitor.n_points)
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

    @pytest.mark.parametrize(
        "method", ["heat_capacity", "free_energy", "partition_function"]
    )
    def test_k_b_scales_temperature(self, monitor, method):
        """beta = 1 / (k_B T), so k_B = 2 at T matches k_B = 1 at 2 T."""
        T = np.array([0.5, 1.0, 2.0])
        fn = getattr(monitor, method)
        np.testing.assert_allclose(fn(T, k_B=2.0), fn(2.0 * T), rtol=1e-12)

    def test_expectation_k_b_scales_temperature(self, monitor):
        obs = monitor.point_energies
        T = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(
            monitor.expectation(obs, T, k_B=2.0),
            monitor.expectation(obs, 2.0 * T),
            rtol=1e-12,
        )

    def test_npt_log_z_uses_stored_enthalpies(self):
        """In NPT the stored energies are already H = U + P V, so log Z
        needs no extra volume term."""
        m = _make_monitor(n_dead=200, n_live=30, seed=7, with_volumes=True)
        T = np.array([0.5, 1.0, 2.0])
        expected = [
            thermo_partition_function(
                1.0 / t, m.dead_energies, m.live_energies, n_live=m.n_live
            )
            for t in T
        ]
        np.testing.assert_allclose(m.log_Z(T), expected, rtol=1e-12)


def _write_run_dir(
    path: Path,
    log_evidence: float = 10.0,
    n_dead: int = 10,
    n_live: int = 5,
    seed: int = 0,
    checkpoint: str = "ns.final.checkpoint.h5",
    symbol_map: dict | None = None,
    write_energies: bool = True,
):
    """Write a checkpoint and ns.energies into path, the files
    Monitor.from_directory reads. Returns (dead_energies, live_energies)."""
    rng = np.random.default_rng(seed)
    dead_e = np.sort(rng.uniform(0.0, 5.0, n_dead))
    live_e = rng.uniform(4.0, 6.0, n_live)
    ns_state = {
        "positions": jnp.asarray(rng.uniform(0.0, 1.0, (n_live, 2, 3))),
        "types": jnp.zeros(2, dtype=jnp.int32),
        "energies": jnp.asarray(live_e),
        "cells": None,
        "log_evidence": log_evidence,
        "iteration": n_dead,
        "emax": float(live_e.max()),
        "n_dead": n_dead,
        "n_walkers": n_live,
    }
    save_checkpoint(path / checkpoint, ns_state, symbol_map=symbol_map)
    if write_energies:
        energy_logger = EnergyLogger(path / "ns.energies", n_walkers=n_live)
        energy_logger.write_header()
        for i, e in enumerate(dead_e):
            energy_logger.write_entry(i, float(e))
        energy_logger.close()
    return dead_e, live_e


class TestMonitorFromDirectory:
    """Check Monitor.from_directory on a synthetic run directory."""

    def test_from_directory_loads_correctly(self, tmp_path):
        dead_e, live_e = _write_run_dir(
            tmp_path,
            log_evidence=55.5,
            n_dead=20,
            n_live=10,
            seed=42,
            symbol_map={0: "Ar"},
        )

        m = Monitor.from_directory(tmp_path, prefix="ns")
        assert m.n_dead == 20
        assert m.n_live == 10
        np.testing.assert_allclose(m.dead_energies, dead_e, rtol=1e-6)
        np.testing.assert_allclose(m.live_energies, live_e, rtol=1e-6)
        assert m.log_evidence == pytest.approx(55.5, rel=1e-4)
        assert m.energy_trace is not None
        assert len(m.energy_trace) == 20
        assert m.symbol_map == {0: "Ar"}
        # Live walkers: positions per walker, types broadcast to every
        # walker, no cells for this non-periodic run.
        assert m.live_positions.shape == (10, 2, 3)
        assert m.live_types.shape == (10, 2)
        assert m.live_cells is None

    def test_unreadable_types_leave_live_types_unset(self, tmp_path):
        """Types that are neither per atom nor per walker and atom do not
        fail the load; only the live types stay unset."""
        _write_run_dir(tmp_path, n_live=5)
        with h5py.File(tmp_path / "ns.final.checkpoint.h5", "a") as f:
            del f["types"]
            f.create_dataset("types", data=np.zeros(7, dtype=np.int32))
        m = Monitor.from_directory(tmp_path, prefix="ns")
        assert m.live_types is None
        assert m.live_positions.shape == (5, 2, 3)
        assert np.isfinite(m.heat_capacity(1.0))

    def test_from_directory_prefers_final(self, tmp_path):
        """.final.checkpoint.h5 wins over .checkpoint.h5."""
        _write_run_dir(
            tmp_path,
            log_evidence=10.0,
            checkpoint="ns.checkpoint.h5",
            write_energies=False,
        )
        _write_run_dir(tmp_path, log_evidence=99.0)

        m = Monitor.from_directory(tmp_path, prefix="ns", prefer_final=True)
        assert m.log_evidence == pytest.approx(99.0, rel=1e-4)

    def test_from_directory_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Monitor.from_directory(tmp_path, prefix="ns")


_PLOTS_OVER_T = {
    "heat_capacity": plot_heat_capacity,
    "partition_function": plot_partition_function,
    "free_energy": plot_free_energy,
}
_PLOTS_OVER_ITER = {
    "energy_trace": plot_energy_trace,
    "log_evidence_trace": plot_log_evidence_trace,
}
_PLOT_NAMES = [*_PLOTS_OVER_ITER, *_PLOTS_OVER_T]


def _draw(name, monitor, ax=None):
    """Draw plot `name`; the thermodynamic ones take a T grid."""
    if name in _PLOTS_OVER_T:
        T = np.linspace(0.5, 5.0, 5)
        return _PLOTS_OVER_T[name](monitor, T, ax=ax)
    return _PLOTS_OVER_ITER[name](monitor, ax=ax)


class TestPlotting:
    """Check the single-monitor plots: labels, ax reuse and the missing
    trace error."""

    @pytest.fixture
    def monitor(self):
        return _make_monitor(
            n_dead=50, n_live=10, with_trace=True, label="run1"
        )

    @pytest.mark.parametrize("name", _PLOT_NAMES)
    def test_axis_labels(self, monitor, name):
        ax = _draw(name, monitor)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        if name in _PLOTS_OVER_T:
            assert "T" in ax.get_xlabel()
        plt.close("all")

    @pytest.mark.parametrize("name", _PLOT_NAMES)
    def test_reuses_ax(self, monitor, name):
        _, existing_ax = plt.subplots()
        assert _draw(name, monitor, ax=existing_ax) is existing_ax
        plt.close("all")

    def test_plot_energy_trace_no_trace_raises(self):
        m = _make_monitor(with_trace=False)
        with pytest.raises(ValueError, match="energy_trace"):
            plot_energy_trace(m)


class TestMonitorCollection:
    """Check MonitorCollection indexing, add/remove, summary, overlay
    plots and from_directories."""

    @pytest.fixture
    def three_monitors(self):
        return [
            _make_monitor(
                n_dead=60, n_live=15, seed=i, label=f"run{i}", with_trace=True
            )
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
        T = np.linspace(0.5, 5.0, 5)
        ax = collection.plot_heat_capacity(T)
        assert len(ax.get_lines()) == 3
        line_labels = [line.get_label() for line in ax.get_lines()]
        assert "run0" in line_labels
        assert "run1" in line_labels
        assert "run2" in line_labels
        plt.close("all")

    def test_plot_heat_capacity_reuses_ax(self, collection):
        T = np.linspace(0.5, 5.0, 5)
        _, existing_ax = plt.subplots()
        returned = collection.plot_heat_capacity(T, ax=existing_ax)
        assert returned is existing_ax
        plt.close("all")

    def test_plot_partition_function_overlay(self, collection):
        T = np.linspace(0.5, 5.0, 4)
        ax = collection.plot_partition_function(T)
        assert len(ax.get_lines()) == 3
        plt.close("all")

    def test_plot_free_energy_overlay(self, collection):
        T = np.linspace(0.5, 5.0, 4)
        ax = collection.plot_free_energy(T)
        assert len(ax.get_lines()) == 3
        plt.close("all")

    def test_from_directories_round_trip(self, tmp_path):
        dirs = []
        for i in range(2):
            d = tmp_path / f"run{i}"
            d.mkdir()
            _write_run_dir(
                d, log_evidence=float(i * 10), n_dead=8, n_live=5, seed=i
            )
            dirs.append(d)

        coll = MonitorCollection.from_directories(
            dirs, labels=["a", "b"], prefix="ns"
        )
        assert len(coll) == 2
        assert coll[0].label == "a"
        assert coll[1].label == "b"
        assert coll[1].log_evidence == pytest.approx(10.0)

    def test_from_directories_label_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="len"):
            MonitorCollection.from_directories(
                [tmp_path / "x", tmp_path / "y"],
                labels=["only_one"],
                prefix="ns",
            )


def _write_adaptation(path, n_entries, n_runs=2, n_moves=2):
    """Write a v3 .adaptation.h5 for n_runs replicas with AdaptationLogger,
    as a multi-run NS writes it."""
    log = AdaptationLogger(
        path, move_names=[f"m{k}" for k in range(n_moves)], n_runs=n_runs
    )
    shape = (n_runs, n_moves)
    stats = {
        "n_rounds": np.ones(shape, dtype=np.int32),
        "converged": np.ones(shape, dtype=bool),
        "cap_hits": np.zeros(shape, dtype=np.int32),
        "floor_hits": np.zeros(shape, dtype=np.int32),
        "bracket_detected": np.zeros(shape, dtype=bool),
        "reject_reason_counts": np.zeros(shape + (4,), dtype=np.int32),
    }
    for i in range(n_entries):
        log.write_entry(
            i,
            np.full(shape, 0.05 + 0.01 * i),
            np.full(shape, 0.4),
            adjustment_stats=stats,
            n_evaluations=np.full(shape, 100),
            n_grad_evaluations=np.full(shape, 50),
        )
    log.close()


class TestFromMultiRunDirectoryConfigInference:
    """Check that from_multi_run_directory takes the prefix and pressure
    labels from config.yaml, and the heatmap and adaptation plots built
    on it."""

    @staticmethod
    def _write_multi_run_artefacts(
        out_dir: Path,
        prefix: str,
        n_runs: int,
        n_live: int = 4,
        n_dead_per_run: int = 6,
        live_volumes: np.ndarray | None = None,
    ) -> None:
        """Write an (n_runs,)-shaped checkpoint and one .energies log per
        replica, with the shapes from_multi_run_directory expects.

        ``live_volumes`` of shape ``(n_runs, n_live)`` makes it an NPT
        checkpoint; NPT runs are the only ones that save it.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        live_e = rng.uniform(-1.0, 1.0, (n_runs, n_live)).astype(np.float32)
        log_Z = rng.uniform(-1.0, 1.0, (n_runs,)).astype(np.float32)
        iteration = np.full((n_runs,), n_dead_per_run, dtype=np.int32)
        n_dead = np.full((n_runs,), n_dead_per_run, dtype=np.int32)

        n_atoms = 2
        positions = rng.uniform(0.0, 1.0, (n_runs, n_live, n_atoms, 3)).astype(
            np.float32
        )
        types = np.zeros((n_runs, n_live, n_atoms), dtype=np.int32)
        ckpt_path = out_dir / f"{prefix}.final.checkpoint.h5"
        with h5py.File(ckpt_path, "w") as f:
            f.create_dataset("positions", data=positions)
            f.create_dataset("types", data=types)
            f.create_dataset("energies", data=live_e)
            f.create_dataset("log_evidence", data=log_Z)
            f.create_dataset("iteration", data=iteration)
            f.create_dataset("n_dead", data=n_dead)
            if live_volumes is not None:
                f.create_dataset("live_volumes", data=live_volumes)
            f.attrs["n_walkers"] = n_live

        for i in range(n_runs):
            elog = EnergyLogger(
                out_dir / f"{prefix}.run{i:02d}.energies",
                n_walkers=n_live,
                n_cull=1,
            )
            elog.write_header()
            for k in range(n_dead_per_run):
                elog.write_entry(k, energy=float(-k), volume=10.0 + k)
            elog.close()

    def test_npt_volumes_are_data_only(self, tmp_path):
        """An NPT checkpoint gives each replica its dead and live volumes
        and is_npt, and leaves log Z and Cv unchanged."""
        live_v = np.arange(8, dtype=np.float64).reshape(2, 4) + 20.0
        self._write_multi_run_artefacts(
            tmp_path / "npt", prefix="ns", n_runs=2, live_volumes=live_v
        )
        self._write_multi_run_artefacts(
            tmp_path / "nvt", prefix="ns", n_runs=2
        )
        npt = MonitorCollection.from_multi_run_directory(
            tmp_path / "npt", prefix="ns"
        )
        nvt = MonitorCollection.from_multi_run_directory(
            tmp_path / "nvt", prefix="ns"
        )

        T = np.array([0.5, 1.0, 2.0])
        for i, (m_npt, m_nvt) in enumerate(zip(npt, nvt, strict=True)):
            assert m_npt.is_npt is True
            assert m_nvt.is_npt is False
            assert m_nvt.dead_volumes is None and m_nvt.live_volumes is None
            np.testing.assert_array_equal(
                m_npt.dead_volumes, 10.0 + np.arange(6)
            )
            np.testing.assert_array_equal(m_npt.live_volumes, live_v[i])
            np.testing.assert_array_equal(
                m_npt.volume_trace, m_npt.dead_volumes
            )
            np.testing.assert_allclose(
                m_npt.partition_function(T), m_nvt.partition_function(T)
            )
            np.testing.assert_allclose(
                m_npt.heat_capacity(T), m_nvt.heat_capacity(T)
            )

    def test_live_configurations_per_replica(self, tmp_path):
        """Each replica gets its own slice of the live positions and
        types."""
        self._write_multi_run_artefacts(tmp_path, prefix="ns", n_runs=3)
        coll = MonitorCollection.from_multi_run_directory(
            tmp_path, prefix="ns"
        )
        with h5py.File(tmp_path / "ns.final.checkpoint.h5", "r") as f:
            positions = f["positions"][()]
        for i, m in enumerate(coll):
            np.testing.assert_array_equal(m.live_positions, positions[i])
            assert m.live_types.shape == (4, 2)
            assert m.live_cells is None

    def test_prefix_inferred_from_config(self, tmp_path):
        """output.out_file_prefix is used when prefix is omitted."""
        out_dir = tmp_path / "output"
        prefix = "neuralil_si16"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=3)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert len(coll) == 3
        # runNN labels when the ensemble section is absent.
        assert [m.label for m in coll] == ["run00", "run01", "run02"]

    def test_pressure_labels_and_metadata_from_config(self, tmp_path):
        """An NPT pressure list of length n_total sets labels and
        pressure_gpa."""
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
            "P= 2.00 GPa",
            "P= 4.00 GPa",
            "P= 6.00 GPa",
        ]
        assert [m.pressure_gpa for m in coll] == [2.0, 4.0, 6.0]

    def test_explicit_args_override_config(self, tmp_path):
        """Explicit prefix and labels win over the config."""
        out_dir = tmp_path / "output"
        prefix = "mysweep"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [1.0, 2.0]\n  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(
            out_dir,
            prefix=prefix,
            labels=["foo", "bar"],
        )
        assert [m.label for m in coll] == ["foo", "bar"]
        # Pressure metadata still attaches when config matches n_total.
        assert [m.pressure_gpa for m in coll] == [1.0, 2.0]

    def test_falls_back_to_runNN_without_config(self, tmp_path):
        """Without config.yaml: prefix "ns" and runNN labels."""
        out_dir = tmp_path / "output"
        self._write_multi_run_artefacts(out_dir, prefix="ns", n_runs=2)

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert [m.label for m in coll] == ["run00", "run01"]
        # No pressure metadata attached when there's no config to read.
        assert not hasattr(coll[0], "pressure_gpa")

    def test_explicit_config_path_overrides_discovery(self, tmp_path):
        """An explicit config= skips the config.yaml lookup."""
        out_dir = tmp_path / "output"
        prefix = "explicit"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        config_path = tmp_path / "elsewhere.yaml"
        config_path.write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [3.0, 5.0]\n  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(
            out_dir,
            config=config_path,
        )
        assert [m.pressure_gpa for m in coll] == [3.0, 5.0]

    def test_plot_heatmap_smoke(self, tmp_path):
        """plot_heatmap draws one pcolormesh with P on x and T on y."""
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

        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(meshes) == 1
        # Default fmt='PT' → x=P, y=T.
        assert ax.get_xlabel() == "P [GPa]"
        assert ax.get_ylabel() == "T"
        plt.close("all")

    def test_plot_heatmap_fmt_tp_swaps_axes(self, tmp_path):
        out_dir = tmp_path / "output"
        prefix = "fmt_tp"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
            "ensemble:\n  type: npt\n  pressure: [1.0, 2.0]\n"
            "  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        ax = coll.plot_heatmap(
            np.linspace(0.5, 2.0, 4), "heat_capacity", fmt="TP"
        )
        assert ax.get_xlabel() == "T"
        assert ax.get_ylabel() == "P [GPa]"
        plt.close("all")

    def test_plot_heatmap_falls_back_to_indices(self, tmp_path):
        """Without pressure metadata the P axis is the replica index."""
        out_dir = tmp_path / "output"
        self._write_multi_run_artefacts(out_dir, prefix="ns", n_runs=2)
        # No config.yaml → no pressure_gpa metadata.
        coll = MonitorCollection.from_multi_run_directory(out_dir)
        ax = coll.plot_heatmap(np.linspace(0.5, 2.0, 4), "heat_capacity")
        assert ax.get_xlabel() == "replica index"
        plt.close("all")

    def test_plot_heatmap_callable_observable(self, tmp_path):
        """A custom callable observable feeds the heatmap grid."""
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
            np.linspace(0.5, 2.0, 4),
            constant_obs,
            cbar_label="P",
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
            coll.plot_heatmap(
                np.linspace(0.5, 2.0, 4), "not_a_real_observable"
            )

    def test_adaptation_trace_loaded_into_collection(self, tmp_path):
        """<prefix>.adaptation.h5 is attached to the collection, not to
        the monitors."""
        out_dir = tmp_path / "output"
        prefix = "adap"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)

        _write_adaptation(out_dir / f"{prefix}.adaptation.h5", n_entries=5)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert coll.adaptation_trace is not None
        assert coll.adaptation_trace.n_runs == 2
        assert coll.adaptation_trace.n_moves == 2
        assert coll.adaptation_trace.step_sizes.shape == (5, 2, 2)

    def test_plot_step_sizes_uses_collection_trace(self, tmp_path):
        """plot_step_sizes reads the collection-level trace."""
        out_dir = tmp_path / "output"
        prefix = "ploth5"
        self._write_multi_run_artefacts(out_dir, prefix=prefix, n_runs=2)
        _write_adaptation(out_dir / f"{prefix}.adaptation.h5", n_entries=4)
        (tmp_path / "config.yaml").write_text(
            f"output:\n  out_file_prefix: {prefix}\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        # per_run=False → one line per move (mean across replicas)
        ax = coll.plot_step_sizes(per_run=False)
        assert len(ax.get_lines()) == 2
        plt.close("all")
        # per_run=True → one line per (move, replica)
        ax = coll.plot_step_sizes(per_run=True)
        assert len(ax.get_lines()) == 4
        plt.close("all")

    def test_plot_heatmap_low_level_shape_validation(self):
        """plot_heatmap rejects Z whose shape is not (n_P, n_T)."""
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
            + "["
            + ",".join(str(float(p)) for p in range(1, 17))
            + "]\n"
            "  pressure_units: gpa\n"
        )

        coll = MonitorCollection.from_multi_run_directory(out_dir)
        assert [m.label for m in coll] == ["run00", "run01", "run02"]
        assert not hasattr(coll[0], "pressure_gpa")


_LJ8_NPT_OUTPUT = (
    Path(__file__).parent.parent / "_assets" / "data" / "lj8_npt" / "output"
)


class TestSmokeRealRun:
    """Check Monitor on the lj8_npt output committed in the repo.

    The files under ``tests/_assets/data/lj8_npt/output/`` come from
    ``jaxrens run -c tests/_assets/data/lj8_npt/config.yaml``, an 8-atom
    LJ NPT run of 200 iterations.
    """

    def test_from_directory_lj8_npt(self):
        m = Monitor.from_directory(
            _LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke"
        )
        assert m.n_dead > 0
        assert m.n_live > 0
        assert np.isfinite(m.log_evidence)
        assert m.is_npt

    def test_heat_capacity_on_real_run(self, tmp_path):
        m = Monitor.from_directory(
            _LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke"
        )
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
        m = Monitor.from_directory(
            _LJ8_NPT_OUTPUT, prefix="lj8_npt", label="lj8_smoke"
        )
        assert m.energy_trace is not None
        assert m.iteration_trace is not None
        assert len(m.energy_trace) == len(m.iteration_trace)
