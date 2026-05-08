"""End-to-end integration: tiny H NS through the nequix backend.

Mirrors :mod:`tests.integration.test_neuralil_pipeline` but for nequix.
The point is to exercise the *nequix wiring* — backend build, ``.nqx``
checkpoint load, the resolver's pre-NS energy eval, the bucketed
``max_neighbors_list`` ladder, supercell edge finding, and a few MCMC
steps with ``value_and_grad(nequix)`` — not to validate any scientific
result.

Skipped when:

* ``nequix`` is not importable.

Sized for the integration tier (seconds to a minute on GPU; longer on
CPU but tractable).  Two-pressure multi-run + an 8-replica multi-GPU
variant, mirroring the LJ / MACE / NeuralIL integration suites.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from . import multi_gpu_n_devices


_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "nequix_small"
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
  full_auto_steps: 100
  defaults:
    min_rate: 0.3
    max_rate: 0.5
    adjust_factor: 1.5
    step_size_max: 0.3

init:
  start_species: "1 8"      # 8 H atoms (Z=1 → first slot of nequix's z-table)
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
  # V/atom large enough that the cubic init cell axis stays well above
  # ``2 * r_cutoff / min(supercell_trafo)`` (= 4 Å here at sc=(3,3,3)),
  # so the supercell expansion captures all true neighbors throughout
  # the run.
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
    """Two-pressure nequix NS run.  Exercises:

    * Resolver — config → ``ResolvedMultiRunConfig`` (two-pressure path).
    * Backend build — ``create_nequix`` loading the bundled ``.nqx`` model.
    * Initial walker sampling + ``_finalise_initial_energies_and_counts``
      (no-``max_neighbors_for`` path for nequix).
    * Bucketed kernel dispatch — ``max_neighbors_list`` ladder.
    * Symbol-map plumbing through the resolver via the backend's
      ``atomic_numbers`` property.
    * ``init_ns_multi_gpu`` building a ``(G, P, ...)`` NSState.
    * Batched burn-in (1 walk, 2 MCMC steps).
    * NS loop with galilean (``value_and_grad(nequix)``) + volume moves
      under pmap-vmap dispatch (single device, ``n_gpu=1, n_per_gpu=2``).
    * Streamed I/O — per-replica ``.energies`` / ``.traj.extxyz``, plus
      global HDF5 checkpoints.
    """
    pytest.importorskip("nequix")

    from jaxrens.cli.resolve import (
        ResolvedMultiRunConfig,
        expand_multi_run_or_cohort,
    )
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootConfig

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_NQX)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootConfig.model_validate(raw)
    resolved = expand_multi_run_or_cohort(root)
    assert isinstance(resolved, ResolvedMultiRunConfig), (
        "Two-pressure config should route through the multi-GPU dispatcher."
    )

    run_multi_gpu_from_config(resolved)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    assert n_total == 2, f"expected 2 replicas, got {n_total}"

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"nequix_smoke.run{r:02d}.energies"
        traj_path = out / f"nequix_smoke.run{r:02d}.traj.extxyz"
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

    final_ckpt = out / "nequix_smoke.final.checkpoint.h5"
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


# ---------------------------------------------------------------------------
# Multi-GPU variant — pressure replicas distributed 2-per-device on
# whatever JAX exposes (either 2 or 4 local devices).
# ---------------------------------------------------------------------------


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

# RE every 2 iters → 2 swap rounds during the run, exercising the
# cross-device gather + accept/reject path.
inter_re:
  flavor: pressure
  every: 2
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
  full_auto_steps: 100
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
    """nequix NPT NS with pressure replicas distributed 2-per-GPU.

    Same code path as :func:`test_nequix_full_pipeline` but with multiple
    real devices and an inner vmap of width 2, exercising the full
    pmap(vmap(vmap)) hierarchy across cross-device pmap communication.
    Runs on either 2 or 4 local devices.
    """
    pytest.importorskip("nequix")

    from jaxrens.cli.resolve import (
        ResolvedMultiRunConfig,
        expand_multi_run_or_cohort,
    )
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootConfig

    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_NEQUIX_MULTI_GPU_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_NQX)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["ensemble"]["pressure"] = raw["ensemble"]["pressure"][:n_total]

    root = RootConfig.model_validate(raw)
    resolved = expand_multi_run_or_cohort(root)
    assert isinstance(resolved, ResolvedMultiRunConfig)
    assert resolved.ns.n_gpu == n_gpu
    assert resolved.ns.n_per_gpu == 2

    run_multi_gpu_from_config(resolved)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"nequix_mgpu.run{r:02d}.energies"
        traj_path = out / f"nequix_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    final_ckpt = out / "nequix_mgpu.final.checkpoint.h5"
    assert final_ckpt.exists()

    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (n_gpu, 2, 4, 8, 3)
