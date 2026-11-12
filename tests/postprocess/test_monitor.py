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
        """The live walkers get the mean of the dead-point observable."""
        T = np.array([1.0, 2.0])
        betas = 1.0 / T
        obs = monitor.dead_energies

        dead_e = jnp.asarray(monitor.dead_energies)
        live_e = jnp.asarray(monitor.live_energies)
        live_obs_val = float(np.mean(obs))
        live_obs = jnp.full(live_e.shape[0], live_obs_val)
        obs_full = jnp.concatenate([jnp.asarray(obs), live_obs])

        expected = np.array(
            [
                float(
                    thermo_expectation(
                        obs_full,
                        float(b),
                        dead_e,
                        live_e,
                        n_live=monitor.n_live,
                    )
                )
                for b in betas
            ]
        )
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

    @pytest.mark.parametrize(
        "method", ["heat_capacity", "free_energy", "partition_function"]
    )
    def test_k_b_scales_temperature(self, monitor, method):
        """beta = 1 / (k_B T), so k_B = 2 at T matches k_B = 1 at 2 T."""
        T = np.array([0.5, 1.0, 2.0])
        fn = getattr(monitor, method)
        np.testing.assert_allclose(fn(T, k_B=2.0), fn(2.0 * T), rtol=1e-12)

    def test_expectation_k_b_scales_temperature(self, monitor):
        obs = monitor.dead_energies
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
    ) -> None:
        """Write an (n_runs,)-shaped checkpoint and one .energies log per
        replica, with the shapes from_multi_run_directory expects."""
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
