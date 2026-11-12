"""Tests for the multi-replica path of ``resolve`` and its runner.

A list of pressures makes ``resolve`` build one replica per pressure.
``_derive_replica_axes`` splits them over the devices, and two tiny LJ runs
go through ``run_multi_gpu_from_config``. The device count is patched to
one (``n_gpu=1``, ``n_per_gpu>1``) unless a test sets it, so everything runs
on CPU.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import pytest

from jaxrens.cli import resolve as _r
from jaxrens.cli.resolve import ResolvedConfig, _derive_replica_axes, resolve
from jaxrens.cli.run import run_multi_gpu_from_config
from jaxrens.cli.schema import RootSpec
from jaxrens.io.re_stats_log import RELogger
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun


def _lj_multi_run_config(
    pressures,
    *,
    n_live: int = 16,
    max_iterations: int = 5,
    n_mcmc_steps: int = 2,
    working_dir=".",
    inter_re: dict | None = None,
) -> dict:
    cfg = {
        "run": {
            "n_live": n_live,
            "max_iterations": max_iterations,
            "n_mcmc_steps": n_mcmc_steps,
            "n_extra": 0,
            "seed": 0,
        },
        "moves": [
            {
                "move_type": "galilean",
                "step_size": 0.05,
                "n_reflect": 4,
                "weight": 1.0,
            },
        ],
        "backend": {
            "backend_type": "lj",
            "cutoff": 2.5,
            "periodic": True,
        },
        "ensemble": {
            "type": "npt",
            "pressure": list(pressures),
            "pressure_units": "eva3",
        },
        "output": {
            "format": "none",
            "working_dir": str(working_dir),
            "info_interval": 999,
            "traj_interval": 999,
            "snapshot_interval": 999,
            "checkpoint_interval": 999,
        },
        "init": {
            "start_species": "18 8",
            "random_initialise_pos": True,
            "pos_randomization_mode": "grid",
            "grid_distance": 1.0,
            "random_initialise_cell": False,
            "init_distance_criterion": 0.5,
            "random_init_max_n_tries": 10,
            "start_energy_ceiling_per_atom": 1e6,
            "pos_autoscale_cells": False,
        },
        "cell": {
            "max_volume_per_atom": 50.0,
            "min_volume_per_atom": 0.5,
            "min_aspect_ratio": 0.6,
            "flat_V_prior": False,
        },
    }
    if inter_re is not None:
        cfg["inter_re"] = inter_re
    return cfg


class TestDeriveReplicaAxes:
    """Check how the replica count and the device count give
    ``(n_total, n_gpu, n_per_gpu)`` and the per-replica params.
    """

    @pytest.fixture(autouse=True)
    def _single_device(self, monkeypatch):
        """Make the resolver see one device, whatever the host has."""
        monkeypatch.setattr(_r, "_local_device_count", lambda: 1)

    def test_scalar_pressure_is_single_run(self):
        cfg = _lj_multi_run_config([0.01])
        root = RootSpec.model_validate(cfg)
        n_total, n_gpu, n_per_gpu, params = _derive_replica_axes(root)
        assert n_total == 1
        assert n_gpu == 1
        assert n_per_gpu == 1
        assert params == []

    def test_pressure_list_drives_replica_count(self):
        cfg = _lj_multi_run_config([0.01, 0.1, 1.0])
        root = RootSpec.model_validate(cfg)
        n_total, n_gpu, n_per_gpu, params = _derive_replica_axes(root)
        assert (n_total, n_gpu, n_per_gpu) == (3, 1, 3)
        assert [p["pressure"] for p in params] == [0.01, 0.1, 1.0]

    def test_pressure_inter_re_requires_list(self):
        cfg = _lj_multi_run_config(
            [0.01],
            inter_re={
                "flavor": "pressure",
                "re_interval": 1,
                "n_swap_cycles": 1,
            },
        )
        root = RootSpec.model_validate(cfg)
        with pytest.raises(ValueError, match="requires a list-valued"):
            _derive_replica_axes(root)

    def test_replica_list_length_mismatch_raises(self):
        """Two pressures but three xrens composition targets."""
        cfg = _lj_multi_run_config(
            [0.01, 0.1],
            inter_re={
                "flavor": "xrens",
                "composition_targets": [[8], [8], [8]],
            },
        )
        root = RootSpec.model_validate(cfg)
        with pytest.raises(ValueError, match="length mismatch"):
            _derive_replica_axes(root)

    def test_non_divisible_replica_count_raises(self, monkeypatch):
        monkeypatch.setattr(_r, "_local_device_count", lambda: 2)
        cfg = _lj_multi_run_config([0.01, 0.1, 1.0])
        root = RootSpec.model_validate(cfg)
        with pytest.raises(ValueError, match="not divisible"):
            _derive_replica_axes(root)

    def test_three_devices_six_replicas(self, monkeypatch):
        # Six pressures on three GPUs: two replicas per GPU, in list order.
        monkeypatch.setattr(_r, "_local_device_count", lambda: 3)
        cfg = _lj_multi_run_config([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        root = RootSpec.model_validate(cfg)
        n_total, n_gpu, n_per_gpu, params = _derive_replica_axes(root)
        assert (n_total, n_gpu, n_per_gpu) == (6, 3, 2)
        assert len(params) == 6
        assert [round(p["pressure"], 4) for p in params] == [
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
        ]

    def test_more_devices_than_replicas_clamps(self, monkeypatch, caplog):
        monkeypatch.setattr(_r, "_local_device_count", lambda: 4)
        cfg = _lj_multi_run_config([0.01, 0.1])
        root = RootSpec.model_validate(cfg)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            n_total, n_gpu, n_per_gpu, _ = _derive_replica_axes(root)
        assert (n_total, n_gpu, n_per_gpu) == (2, 2, 1)
        assert any("clamping n_gpu" in rec.message for rec in caplog.records)


class TestResolveMultiReplica:
    """Check the batcher, stacked init arrays and inter-RE config that
    ``resolve`` builds for a pressure list.
    """

    @pytest.fixture(autouse=True)
    def _single_device(self, monkeypatch):
        monkeypatch.setattr(_r, "_local_device_count", lambda: 1)

    def test_pressure_list_resolves_to_pmap_vmap_runs(self):
        cfg = _lj_multi_run_config([0.01, 0.1])
        root = RootSpec.model_validate(cfg)
        out = resolve(root)
        assert isinstance(out, ResolvedConfig)
        assert isinstance(out.batcher, PmapVmapRuns)
        assert out.ns.n_gpu * out.ns.n_per_gpu == 2
        assert out.batcher.n_gpu == out.ns.n_gpu
        assert out.batcher.n_per_gpu == out.ns.n_per_gpu

    def test_scalar_pressure_resolves_to_single_run(self):
        cfg = _lj_multi_run_config([0.01])
        root = RootSpec.model_validate(cfg)
        out = resolve(root)
        assert isinstance(out, ResolvedConfig)
        assert isinstance(out.batcher, SingleRun)
        assert len(out.ensemble_params_per_run) == 1

    def test_resolved_init_stacked_shapes(self):
        cfg = _lj_multi_run_config([0.01, 0.1, 1.0])
        root = RootSpec.model_validate(cfg)
        out = resolve(root)
        n_total = out.ns.n_gpu * out.ns.n_per_gpu
        n_live = out.ns.n_live
        pos = out.init.initial_positions
        cells = out.init.initial_cells
        ens = out.init.initial_energies
        assert pos.shape[:2] == (n_total, n_live)
        assert cells.shape[:2] == (n_total, n_live)
        assert ens.shape == (n_total, n_live)
        assert jnp.all(jnp.isfinite(ens))

    def test_per_replica_energies_differ_by_pressure(self):
        # Both replicas start from similar cells, so the larger P V term
        # makes the high-pressure energies larger on average.
        cfg = _lj_multi_run_config([0.01, 10.0])
        root = RootSpec.model_validate(cfg)
        out = resolve(root)
        e_lo = out.init.initial_energies[0]
        e_hi = out.init.initial_energies[1]
        assert float(jnp.mean(e_hi)) > float(jnp.mean(e_lo))

    def test_inter_re_config_propagated(self):
        cfg = _lj_multi_run_config(
            [0.01, 0.1],
            inter_re={
                "flavor": "pressure",
                "re_interval": 2,
                "n_swap_cycles": 3,
            },
        )
        root = RootSpec.model_validate(cfg)
        out = resolve(root)
        assert out.inter_re_config is not None
        assert out.inter_re_config.flavor == "pressure"
        assert out.inter_re_config.re_interval == 2
        assert out.inter_re_config.n_swap_cycles == 3


class TestRunMultiGpuFromConfig:
    """Check two tiny two-replica LJ runs on one device."""

    @pytest.fixture(autouse=True)
    def _single_device(self, monkeypatch):
        monkeypatch.setattr(_r, "_local_device_count", lambda: 1)

    def test_tiny_lj_two_replicas_runs(self, tmp_path):
        cfg = _lj_multi_run_config(
            [0.05, 0.5],
            n_live=12,
            max_iterations=3,
            n_mcmc_steps=2,
            working_dir=tmp_path,
        )
        root = RootSpec.model_validate(cfg)
        resolved = resolve(root)

        result = run_multi_gpu_from_config(resolved)
        assert "log_evidence" in result
        le = jnp.asarray(result["log_evidence"])
        # PmapVmapRuns shape is (G, P) = (1, 2)
        assert le.shape == (1, 2)
        assert jnp.all(jnp.isfinite(le))

        # Per-replica trajectory files and energy logs land under output dir.
        prefix = resolved.output.out_file_prefix
        assert (tmp_path / f"{prefix}.run00.energies").exists()
        assert (tmp_path / f"{prefix}.run01.energies").exists()

    def test_pressure_rens_produces_inter_re_stats(self, tmp_path):
        """With re_interval=1 over 3 iterations, swaps fire at 1 and 2.

        Each fire attempts one swap on the single pair of replicas.
        """
        cfg = _lj_multi_run_config(
            [0.05, 0.5],
            n_live=12,
            max_iterations=3,
            n_mcmc_steps=2,
            working_dir=tmp_path,
            inter_re={
                "flavor": "pressure",
                "re_interval": 1,
                "n_swap_cycles": 1,
            },
        )
        cfg["output"]["save_re_stats"] = True
        root = RootSpec.model_validate(cfg)
        resolved = resolve(root)
        assert resolved.inter_re_config is not None

        result = run_multi_gpu_from_config(resolved)
        le = jnp.asarray(result["log_evidence"])
        assert le.shape == (1, 2)
        assert jnp.all(jnp.isfinite(le))

        trace = RELogger.read(tmp_path / "ns.re_stats.h5")
        assert trace.n_pairs == 1
        assert trace.flavor == "pressure"
        assert list(trace.iterations) == [1, 2]
        assert trace.n_attempted_per_pair.tolist() == [[1], [1]]
