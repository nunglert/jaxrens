"""End-to-end integration: Si NPT-NS through the jax-md Tersoff backend.

Mirrors :mod:`tests.integration.test_nequix_pipeline` but with jax-md's
Tersoff '88 Si potential.  Exercises:

* Backend build — ``create_jaxmd(potential='tersoff', ...)``.
* Resolver's pre-NS energy eval (no ``max_neighbors_for`` path — all-pairs).
* Symbol-map plumbing via the backend's ``atomic_numbers`` property.
* Cohort or multi-GPU dispatcher, depending on pressure-list length.
* Streamed I/O — per-replica ``.energies`` / ``.traj.extxyz``, plus
  HDF5 checkpoints.

Skipped when ``jax_md`` is not importable.
Sized for the integration tier — small ``n_live`` and ``max_iterations``,
keeps wall time bounded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

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
  # V/atom large enough to give the all-pairs calculation breathing room.
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
    """Two-pressure jax-md Tersoff NS smoke test.

    Routes through the multi-GPU dispatcher (two pressures → two
    replicas).  On a single-GPU box this becomes ``n_gpu=1, n_per_gpu=2``;
    on a 2-GPU box it becomes ``n_gpu=2, n_per_gpu=1``.  Either is fine —
    the test inspects per-replica outputs and the global checkpoint.
    """
    pytest.importorskip("jax_md")

    from jaxrens.cli.resolve import (
        resolve,
    )
    from jaxrens.sampling.batch_descriptor import PmapVmapRuns
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootSpec

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, PmapVmapRuns), (
        "Two-pressure config should route through the multi-GPU dispatcher."
    )

    run_multi_gpu_from_config(resolved)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    assert n_total == 2, f"expected 2 replicas, got {n_total}"

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"jaxmd_smoke.run{r:02d}.energies"
        traj_path = out / f"jaxmd_smoke.run{r:02d}.traj.extxyz"
        assert energies_path.exists(), (
            f"missing per-replica energy log: {energies_path}"
        )
        assert traj_path.exists(), (
            f"missing per-replica trajectory: {traj_path}"
        )
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,), (
            f"{energies_path.name}: expected 5 data entries, "
            f"got {log.energies.shape}"
        )

    final_ckpt = out / "jaxmd_smoke.final.checkpoint.h5"
    assert final_ckpt.exists(), f"missing final checkpoint: {final_ckpt}"

    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (resolved.ns.n_gpu, resolved.ns.n_per_gpu), (
        f"expected (G, P)=({resolved.ns.n_gpu}, {resolved.ns.n_per_gpu}), "
        f"got {log_z.shape}"
    )
    assert np.all(np.isfinite(log_z)), f"log_evidence not finite: {log_z}"

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (
        resolved.ns.n_gpu, resolved.ns.n_per_gpu, 4, 8, 3,
    )


@pytest.mark.multi_gpu
@pytest.mark.jaxmd
@pytest.mark.gpu
@pytest.mark.heavy
def test_jaxmd_multi_gpu_pipeline(tmp_path: Path) -> None:
    """jax-md Tersoff NPT-NS with pressure replicas distributed 2-per-GPU.

    Exercises the full ``pmap(vmap(vmap))`` hierarchy across multiple
    devices.  Runs on either 2 or 4 local devices.
    """
    pytest.importorskip("jax_md")

    from jaxrens.cli.resolve import (
        resolve,
    )
    from jaxrens.sampling.batch_descriptor import PmapVmapRuns
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootSpec

    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    # Generate enough pressures for n_per_gpu=2 on n_gpu devices.
    raw["ensemble"]["pressure"] = [
        float(i) * 0.25 for i in range(n_total)
    ]
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

    run_multi_gpu_from_config(resolved)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"jaxmd_mgpu.run{r:02d}.energies"
        traj_path = out / f"jaxmd_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    final_ckpt = out / "jaxmd_mgpu.final.checkpoint.h5"
    assert final_ckpt.exists()

    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (n_gpu, 2, 4, 8, 3)
