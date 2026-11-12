"""End-to-end NPT nested sampling of Si with the MACE backend.

The tests run a tiny config (8 atoms, ``n_live=4``, 5 iterations, two
pressures) through ``resolve`` and ``run_multi_gpu_from_config`` with the
``mace_mp_small`` bundle in ``tests/_assets/models``. This covers the MACE
backend build, the geometry-only neighbour count (``max_neighbors_for``),
the ``max_neighbors_list`` bucket ladder and ``value_and_grad`` of MACE in
the galilean move. The results are not checked for physical accuracy. The
tests skip when ``mace_jax`` is not installed.
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
    / "mace_mp_small"
)


_CONFIG_YAML = """
run:
  n_live: 4
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42

backend:
  type: mace
  checkpoint_path: PLACEHOLDER
  # The 8-atom Si cell is small next to r_cutoff = 6 Å, so the short axes
  # need image offsets up to ±2 in every direction.
  supercell_trafo: [4, 4, 4]
  periodic: true
  max_neighbors_list: [40, 60, 80]
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

# Iteration count only, so n_dead is fixed for the assertions.
termination:
  - type: iteration
    max_iterations: 5

# adjust_interval > max_iterations, so the step-size bisection never runs;
# with MACE it would only add wall time.
adaptation:
  full_auto: true
  adjust_interval: 100
  defaults:
    min_rate: 0.3
    max_rate: 0.5
    adjust_factor: 1.5
    step_size_max: 0.3

init:
  start_species: "14 8"     # 8 Si atoms (Z=14)
  random_initialise_pos: true
  pos_randomization_mode: grid
  grid_distance: 1.5        # below the Si nearest-neighbour distance, 2.35 Å
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
  out_file_prefix: mace_smoke
  info_interval: 1
  traj_interval: 1
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.mace
@pytest.mark.gpu
@pytest.mark.heavy
def test_mace_full_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas, which go through the multi-GPU dispatcher.

    On one device the layout is ``(G, P) = (1, 2)``.
    """
    pytest.importorskip("mace_jax")

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_FIXTURE)
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
        energies_path = out / f"mace_smoke.run{r:02d}.energies"
        traj_path = out / f"mace_smoke.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)
        # The model's z-table puts Si at index 13; the trajectory must map
        # it back to Si.
        frames = ase.io.read(traj_path, index=":")
        assert [f.info["iter"] for f in frames] == [0, 1, 2, 3, 4]
        assert all(f.get_chemical_symbols() == ["Si"] * 8 for f in frames)

    final_ckpt = out / "mace_smoke.final.checkpoint.h5"
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


_MACE_MULTI_GPU_CONFIG_YAML = """
run:
  n_live: 4
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42

backend:
  type: mace
  checkpoint_path: PLACEHOLDER
  supercell_trafo: [4, 4, 4]
  periodic: true
  max_neighbors_list: [40, 60, 80]
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
  start_species: "14 8"
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
  max_volume_per_atom: 60.0
  min_volume_per_atom: 12.0
  min_aspect_ratio: 0.5
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: mace_mgpu
  info_interval: 1
  traj_interval: 1
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.multi_gpu
@pytest.mark.mace
@pytest.mark.gpu
@pytest.mark.heavy
def test_mace_multi_gpu_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas per GPU with pressure replica exchange.

    This goes through the full ``pmap(vmap(vmap))`` hierarchy on 2 or 4
    devices.
    """
    pytest.importorskip("mace_jax")

    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_MACE_MULTI_GPU_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_FIXTURE)
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
        energies_path = out / f"mace_mgpu.run{r:02d}.energies"
        traj_path = out / f"mace_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    final_ckpt = out / "mace_mgpu.final.checkpoint.h5"
    assert final_ckpt.exists()

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))
    np.testing.assert_array_equal(result["log_evidence"], log_z)

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (n_gpu, 2, 4, 8, 3)
