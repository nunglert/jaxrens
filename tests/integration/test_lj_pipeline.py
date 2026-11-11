"""End-to-end NPT nested sampling of Lennard-Jones argon.

The tests build a config from YAML, resolve it and run it through the CLI
entry points: initial walkers and cells, burn-in, the NS loop with
pressure replica exchange and ``full_auto`` step-size adaptation, and the
streamed outputs (``.energies``, ``.traj.extxyz``, HDF5 checkpoints). The
runs are small (8 atoms, ``n_live=16``, 30 iterations).
"""

from __future__ import annotations

from pathlib import Path

import ase.io
import numpy as np
import pytest
import yaml

from jaxrens.cli.cli import _run_single
from jaxrens.cli.resolve import resolve
from jaxrens.cli.run import run_multi_gpu_from_config
from jaxrens.cli.schema import RootSpec
from jaxrens.io.checkpoint import load_checkpoint
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun

from . import multi_gpu_n_devices

_CONFIG_YAML = """
run:
  n_live: 16
  max_iterations: 30
  n_mcmc_steps: 5
  n_extra: 0
  seed: 42

backend:
  type: lj
  epsilon: 1.0
  sigma: 1.0
  cutoff: 2.5
  periodic: true

ensemble:
  type: npt
  pressure: [0.1, 1.0]
  pressure_units: eva3

# Two replicas with RE every 10 iterations: swaps are attempted twice.
inter_re:
  flavor: pressure
  re_interval: 10
  n_swap_cycles: 1

moves:
  - type: galilean
    n_reflect: 4
    step_size: 0.1
    weight: 2.0
  - type: volume
    step_size: 0.2
    weight: 1.0

# Iteration count only, no prior_mass, so the run length the assertions
# below rely on is fixed.
termination:
  - type: iteration
    max_iterations: 30

# full_auto with adjust_interval=10: bisection runs at iterations 10 and 20.
adaptation:
  full_auto: true
  adjust_interval: 10
  defaults:
    min_rate: 0.3
    max_rate: 0.5
    adjust_factor: 1.5
    step_size_max: 0.5

init:
  start_species: "18 8"     # 8 Argon atoms (Z=18)
  random_initialise_pos: true
  pos_randomization_mode: grid
  grid_distance: 1.0
  random_initialise_cell: true
  start_energy_ceiling_per_atom: 100.0
  initial_walk:
    n_walks: 1
    walklength: 5
    adjust_interval: 1
    emax_offset_per_atom: 1.0

cell:
  # Loose enough that the volume sampler and cell_shape_walk don't produce
  # a cell too small or too skewed for grid positioning.
  max_volume_per_atom: 30.0
  min_volume_per_atom: 1.5
  min_aspect_ratio: 0.6
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: smoke
  info_interval: 5
  traj_interval: 5
  snapshot_interval: 100
  checkpoint_interval: 20
  log_level: info
"""


def test_lj_full_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas with RE and check the result and outputs.

    On one device the layout is ``(G, P) = (1, 2)``.
    """
    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, PmapVmapRuns)

    result = run_multi_gpu_from_config(resolved)

    log_z = np.asarray(result["log_evidence"])
    assert log_z.shape == (resolved.ns.n_gpu, resolved.ns.n_per_gpu)
    assert np.all(np.isfinite(log_z))

    n_dead = np.asarray(result["n_dead"])
    assert np.all(n_dead == 30)

    positions = np.asarray(result["positions"])
    n_atoms = 8
    assert positions.shape == (
        resolved.ns.n_gpu,
        resolved.ns.n_per_gpu,
        16,
        n_atoms,
        3,
    )

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    for r in range(n_total):
        energies_path = out / f"smoke.run{r:02d}.energies"
        traj_path = out / f"smoke.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (30,)
        # traj_interval=5 over 30 iterations, tagged with the 0-based
        # iteration index.
        frames = ase.io.read(traj_path, index=":")
        assert [f.info["iter"] for f in frames] == [0, 5, 10, 15, 20, 25]
        assert all(f.get_chemical_symbols() == ["Ar"] * 8 for f in frames)

    for suffix in (
        "initial.checkpoint.h5",
        "checkpoint.h5",
        "final.checkpoint.h5",
    ):
        assert (out / f"smoke.{suffix}").exists()

    state = load_checkpoint(out / "smoke.final.checkpoint.h5")
    np.testing.assert_array_equal(state["log_evidence"], log_z)
    assert np.asarray(state["n_dead"]).tolist() == [[30, 30]]
    assert np.asarray(state["positions"]).shape == positions.shape
    # The checkpoint does not store dead points; .energies and .traj hold
    # them.
    assert state["dead_energies"] is None


@pytest.mark.parametrize("missing", ["pressures", "inter_re"])
def test_lj_pipeline_smoke_variants(tmp_path: Path, missing: str) -> None:
    """Run the config without RE, once with a scalar pressure, once with two.

    A scalar pressure gives a single run; two pressures without
    ``inter_re`` still go through the multi-replica runner.
    """
    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["output"]["out_file_prefix"] = f"smoke_{missing}"

    raw.pop("inter_re")
    if missing == "pressures":
        raw["ensemble"]["pressure"] = 0.5

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)
    out = tmp_path / "out"

    if missing == "pressures":
        assert isinstance(resolved.batcher, SingleRun)
        _run_single(resolved)
        assert (out / "smoke_pressures.final.checkpoint.h5").exists()
    else:
        assert isinstance(resolved.batcher, PmapVmapRuns)
        assert resolved.inter_re_config is None
        run_multi_gpu_from_config(resolved)
        for r in range(2):
            log = EnergyLogger.read(
                out / f"smoke_inter_re.run{r:02d}.energies"
            )
            assert log.energies.shape == (30,)


_LJ_MULTI_GPU_CONFIG_YAML = """
run:
  n_live: 16
  max_iterations: 30
  n_mcmc_steps: 5
  n_extra: 0
  seed: 42

backend:
  type: lj
  epsilon: 1.0
  sigma: 1.0
  cutoff: 2.5
  periodic: true

ensemble:
  type: npt
  pressure: [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15]
  pressure_units: eva3

inter_re:
  flavor: pressure
  re_interval: 10
  n_swap_cycles: 1

moves:
  - type: galilean
    n_reflect: 4
    step_size: 0.1
    weight: 2.0
  - type: volume
    step_size: 0.2
    weight: 1.0

termination:
  - type: iteration
    max_iterations: 30

adaptation:
  full_auto: true
  adjust_interval: 10
  defaults:
    min_rate: 0.3
    max_rate: 0.5
    adjust_factor: 1.5
    step_size_max: 0.5

init:
  start_species: "18 8"
  random_initialise_pos: true
  pos_randomization_mode: grid
  grid_distance: 1.0
  random_initialise_cell: true
  start_energy_ceiling_per_atom: 100.0
  initial_walk:
    n_walks: 1
    walklength: 5
    adjust_interval: 1
    emax_offset_per_atom: 1.0

cell:
  max_volume_per_atom: 30.0
  min_volume_per_atom: 1.5
  min_aspect_ratio: 0.6
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: smoke_mgpu
  info_interval: 5
  traj_interval: 5
  snapshot_interval: 100
  checkpoint_interval: 20
  log_level: info
"""


@pytest.mark.multi_gpu
@pytest.mark.gpu
@pytest.mark.lj
@pytest.mark.heavy
def test_lj_multi_gpu_pipeline(tmp_path: Path) -> None:
    """Run two pressure replicas per GPU with RE and adaptation.

    Unlike :func:`test_lj_full_pipeline` on one device, the inner vmap has
    width 2 and the RE swaps cross devices through ``pmap``. Needs 2 or 4
    local devices.
    """
    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_LJ_MULTI_GPU_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["ensemble"]["pressure"] = raw["ensemble"]["pressure"][:n_total]

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, PmapVmapRuns)
    assert resolved.ns.n_gpu == n_gpu
    assert resolved.ns.n_per_gpu == 2

    result = run_multi_gpu_from_config(resolved)

    log_z = np.asarray(result["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))

    n_dead = np.asarray(result["n_dead"])
    assert np.all(n_dead == 30)

    out = tmp_path / "out"
    for r in range(n_total):
        energies_path = out / f"smoke_mgpu.run{r:02d}.energies"
        traj_path = out / f"smoke_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (30,)
