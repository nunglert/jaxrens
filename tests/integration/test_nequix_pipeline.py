"""End-to-end NPT nested sampling of H with the nequix backend.

The tests run a tiny config (8 atoms, ``n_live=4``, 5 iterations, two
pressures) through ``resolve`` and ``run_multi_gpu_from_config`` with the
``.nqx`` model in ``tests/_assets/models/nequix_small``. This covers the
model load, the ``max_neighbors_list`` bucket ladder, the supercell edge
search and ``value_and_grad`` of nequix in the galilean move. The results
are not checked for physical accuracy. The tests skip when ``nequix`` is
not installed.
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

_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "_assets"
    / "models"
    / "nequix_small"
)
_MODEL_NQX = _FIXTURE / "model.nqx"


_CONFIG_YAML = """
run:
  n_live: 4
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42

backend:
  type: nequix
  checkpoint_path: PLACEHOLDER
  supercell_trafo: [3, 3, 3]
  periodic: true
  max_neighbors_list: [30, 50, 80]
  max_neighbors_offset: 4

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
  start_species: "1 8"      # 8 H atoms; Z=1 is the first slot of the z-table
  random_initialise_pos: true
  pos_randomization_mode: grid
  grid_distance: 1.5
  start_energy_ceiling_per_atom: 100.0
  random_initialise_cell: false
  initial_walk:
    n_walks: 1
    walklength: 2
    adjust_interval: 1
    emax_offset_per_atom: 1.0

cell:
  # The smallest allowed cell edge stays well above
  # 2 * r_cutoff / min(supercell_trafo) = 4 Å, so the 3x3x3 supercell
  # finds every neighbour during the run.
  max_volume_per_atom: 250.0
  min_volume_per_atom: 80.0
  min_aspect_ratio: 0.5
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: nequix_smoke
  info_interval: 1
  traj_interval: 1
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.nequix
@pytest.mark.heavy
def test_nequix_full_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas, which go through the multi-GPU dispatcher.

    On one device the layout is ``(G, P) = (1, 2)``.
    """
    pytest.importorskip("nequix")

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_NQX)
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
        energies_path = out / f"nequix_smoke.run{r:02d}.energies"
        traj_path = out / f"nequix_smoke.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)
        # start_species "1 8" must reach the trajectory as 8 H atoms through
        # the model's z-table.
        frames = ase.io.read(traj_path, index=":")
        assert [f.info["iter"] for f in frames] == [0, 1, 2, 3, 4]
        assert all(f.get_chemical_symbols() == ["H"] * 8 for f in frames)

    final_ckpt = out / "nequix_smoke.final.checkpoint.h5"
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


_NEQUIX_MULTI_GPU_CONFIG_YAML = """
run:
  n_live: 4
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42

backend:
  type: nequix
  checkpoint_path: PLACEHOLDER
  supercell_trafo: [3, 3, 3]
  periodic: true
  max_neighbors_list: [30, 50, 80]
  max_neighbors_offset: 4

ensemble:
  type: npt
  pressure: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
  pressure_units: gpa

# RE every 2 iterations gives 2 swap rounds in 5 iterations, across
# devices.
inter_re:
  flavor: pressure
  re_interval: 2
  n_swap_cycles: 1

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
  start_species: "1 8"
  random_initialise_pos: true
  pos_randomization_mode: grid
  grid_distance: 1.5
  start_energy_ceiling_per_atom: 100.0
  random_initialise_cell: false
  initial_walk:
    n_walks: 1
    walklength: 2
    adjust_interval: 1
    emax_offset_per_atom: 1.0

cell:
  max_volume_per_atom: 250.0
  min_volume_per_atom: 80.0
  min_aspect_ratio: 0.5
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: nequix_mgpu
  info_interval: 1
  traj_interval: 1
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.multi_gpu
@pytest.mark.nequix
@pytest.mark.gpu
@pytest.mark.heavy
def test_nequix_multi_gpu_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas per GPU with pressure replica exchange.

    This goes through the full ``pmap(vmap(vmap))`` hierarchy on 2 or 4
    devices.
    """
    pytest.importorskip("nequix")

    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_NEQUIX_MULTI_GPU_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_NQX)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["ensemble"]["pressure"] = raw["ensemble"]["pressure"][:n_total]

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, PmapVmapRuns)
    assert resolved.ns.n_gpu == n_gpu
    assert resolved.ns.n_per_gpu == 2

    result = run_multi_gpu_from_config(resolved)
    assert np.all(np.asarray(result["n_dead"]) == 5)

    out = tmp_path / "out"
    for r in range(n_total):
        energies_path = out / f"nequix_mgpu.run{r:02d}.energies"
        traj_path = out / f"nequix_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    final_ckpt = out / "nequix_mgpu.final.checkpoint.h5"
    assert final_ckpt.exists()

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))
    np.testing.assert_array_equal(result["log_evidence"], log_z)

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (n_gpu, 2, 4, 8, 3)
