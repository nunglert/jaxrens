"""End-to-end NPT nested sampling of Si with the jax-md Tersoff backend.

Each test builds a config from YAML, resolves it, runs it through
``run_multi_gpu_from_config`` and checks the per-replica energy logs and
trajectories and the final HDF5 checkpoint. The backend computes all
pairs, so no ``max_neighbors`` estimate is involved. Both tests skip when
``jax_md`` is not installed.
"""

from __future__ import annotations

from pathlib import Path

import ase.io
import numpy as np
import pytest
import yaml

from jaxrens.cli.resolve import resolve
from jaxrens.cli.run import run_multi_gpu_from_config
from jaxrens.cli.schema import RootSpec
from jaxrens.io.checkpoint import load_checkpoint
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.sampling.batch_descriptor import PmapVmapRuns

from . import multi_gpu_n_devices

_CONFIG_YAML = """
run:
  n_live: 4
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42

backend:
  type: jaxmd
  potential: tersoff
  tersoff_params: si
  periodic: true

ensemble:
  type: npt
  pressure: [0.0, 1.0]
  pressure_units: gpa

moves:
  - type: galilean
    n_reflect: 2
    step_size: 0.05
    weight: 1.0
  - type: volume
    step_size: 0.05
    weight: 1.0

termination:
  - type: iteration
    max_iterations: 5

adaptation:
  full_auto: true
  adjust_interval: 100
  defaults:
    min_rate: 0.3
    max_rate: 0.5
    adjust_factor: 1.5
    step_size_max: 0.3

init:
  start_species: "14 8"      # 8 Si atoms (Z=14, single-species Tersoff)
  random_initialise_pos: true
  pos_randomization_mode: grid
  grid_distance: 2.0
  start_energy_ceiling_per_atom: 100.0
  random_initialise_cell: false
  initial_walk:
    n_walks: 1
    walklength: 2
    adjust_interval: 1
    emax_offset_per_atom: 1.0

cell:
  max_volume_per_atom: 60.0
  min_volume_per_atom: 12.0
  min_aspect_ratio: 0.5
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: jaxmd_smoke
  info_interval: 1
  traj_interval: 1
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.jaxmd
@pytest.mark.heavy
def test_jaxmd_full_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas, which go through the multi-GPU dispatcher.

    With one GPU this is ``n_gpu=1, n_per_gpu=2``; with two it is
    ``n_gpu=2, n_per_gpu=1``. The checks hold for both layouts.
    """
    pytest.importorskip("jax_md")

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, PmapVmapRuns)

    result = run_multi_gpu_from_config(resolved)
    assert np.all(np.asarray(result["n_dead"]) == 5)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    assert n_total == 2

    for r in range(n_total):
        energies_path = out / f"jaxmd_smoke.run{r:02d}.energies"
        traj_path = out / f"jaxmd_smoke.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    # start_species "14 8" must reach the trajectory as 8 Si atoms through
    # the backend's atomic_numbers table.
    frames = ase.io.read(out / "jaxmd_smoke.run00.traj.extxyz", index=":")
    assert len(frames) > 0
    assert all(f.get_chemical_symbols() == ["Si"] * 8 for f in frames)

    final_ckpt = out / "jaxmd_smoke.final.checkpoint.h5"
    assert final_ckpt.exists()

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (resolved.ns.n_gpu, resolved.ns.n_per_gpu)
    assert np.all(np.isfinite(log_z))
    np.testing.assert_array_equal(result["log_evidence"], log_z)

    # (n_gpu, n_per_gpu, n_live, n_atoms, 3)
    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (
        resolved.ns.n_gpu,
        resolved.ns.n_per_gpu,
        4,
        8,
        3,
    )


@pytest.mark.multi_gpu
@pytest.mark.jaxmd
@pytest.mark.gpu
@pytest.mark.heavy
def test_jaxmd_multi_gpu_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas per GPU with pressure replica exchange.

    This goes through the full ``pmap(vmap(vmap))`` hierarchy on 2 or 4
    devices.
    """
    pytest.importorskip("jax_md")

    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["ensemble"]["pressure"] = [float(i) * 0.25 for i in range(n_total)]
    raw["output"]["out_file_prefix"] = "jaxmd_mgpu"
    raw["inter_re"] = {
        "flavor": "pressure",
        "re_interval": 2,
        "n_swap_cycles": 1,
    }

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, PmapVmapRuns)
    assert resolved.ns.n_gpu == n_gpu
    assert resolved.ns.n_per_gpu == 2

    result = run_multi_gpu_from_config(resolved)
    assert np.all(np.asarray(result["n_dead"]) == 5)

    out = tmp_path / "out"
    for r in range(n_total):
        energies_path = out / f"jaxmd_mgpu.run{r:02d}.energies"
        traj_path = out / f"jaxmd_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    final_ckpt = out / "jaxmd_mgpu.final.checkpoint.h5"
    assert final_ckpt.exists()

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))
    np.testing.assert_array_equal(result["log_evidence"], log_z)

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (n_gpu, 2, 4, 8, 3)
