"""End-to-end integration: tiny Si NS through the MACE backend.

Mirrors :mod:`tests.integration.test_lj_pipeline` but for MACE.  The point
is to exercise the *MACE wiring* — backend build, ``max_neighbors_for``
geometry-only neighbor counting, the resolver's pre-NS MACE energy eval,
the bucketed-kernel overflow ladder, and a few MCMC steps with
``value_and_grad(MACE)`` — not to validate any scientific result.

Skipped when:

* ``mace_jax`` is not importable (``pytest.importorskip``).
* The ``tests/_assets/models/mace_mp_small`` model bundle is missing.
* No GPU is available — MACE is slow on CPU; the integration tier runs
  on the GPU CI runner only.

Sized to keep the GPU path under ~2 minutes including JIT compile:
``n_live=4``, ``max_iterations=5``, 8-atom Si, two pressure replicas
(multi-run path → ``run_multi_gpu_from_config`` with single-device pmap
× ``n_per_gpu=2`` vmap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

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
  # Cell is small relative to r_cutoff=6 Å for an 8-atom Si box, so the
  # short axes need ±2 image offsets in every direction.
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

# Only IterationTermination — keeps n_dead deterministic for the assertions.
termination:
  - type: iteration
    max_iterations: 5

# Adaptation must not fire mid-run (we'd hit the bisection on MACE which
# adds wall-clock for no benefit at this scale); set adjust interval > max.
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
  grid_distance: 1.5        # Si NN ≈ 2.35 Å; safe margin below.
  start_energy_ceiling_per_atom: 100.0
  random_initialise_cell: false   # cubic init cell is enough
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
    """Short Si NS run through the MACE backend.  Exercises:

    * Resolver — config → ``ResolvedConfig`` (two-pressure path).
    * Backend build — ``create_mace`` loading the ``mace_mp_small`` bundle.
    * Initial walker sampling + ``_finalise_initial_energies_and_counts``
      (the resolver's pre-NS MACE energy eval; via ``max_neighbors_for``
      this also exercises the geometry-only neighbor-mask path).
    * Bucketed kernel dispatch — ``max_neighbors_list`` ladder.
    * ``init_ns_multi_gpu`` building a ``(G, P, ...)`` NSState.
    * Batched burn-in (1 walk, 2 MCMC steps).
    * NS loop with galilean (``value_and_grad(MACE)``) + volume moves under
      pmap-vmap dispatch (single device, ``n_gpu=1, n_per_gpu=2``).
    * Streamed I/O — per-replica ``.energies`` / ``.traj.extxyz``, plus
      global HDF5 checkpoints.
    """
    pytest.importorskip("mace_jax")

    from jaxrens.cli.resolve import resolve
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootSpec
    from jaxrens.sampling.batch_descriptor import PmapVmapRuns

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_FIXTURE)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)

    # Two-pressure list → multi-run dispatcher.
    assert isinstance(
        resolved.batcher, PmapVmapRuns
    ), "Two-pressure config should route through the multi-GPU dispatcher."

    run_multi_gpu_from_config(resolved)

    # ---- On-disk artefacts ---------------------------------------------------
    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    assert n_total == 2, f"expected 2 replicas, got {n_total}"

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"mace_smoke.run{r:02d}.energies"
        traj_path = out / f"mace_smoke.run{r:02d}.traj.extxyz"
        assert (
            energies_path.exists()
        ), f"missing per-replica energy log: {energies_path}"
        assert (
            traj_path.exists()
        ), f"missing per-replica trajectory: {traj_path}"
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (
            5,
        ), f"{energies_path.name}: expected 5 data entries, got {log.energies.shape}"

    final_ckpt = out / "mace_smoke.final.checkpoint.h5"
    assert final_ckpt.exists(), f"missing final checkpoint: {final_ckpt}"

    # ---- Final checkpoint round-trip ----------------------------------------
    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (resolved.ns.n_gpu, resolved.ns.n_per_gpu), (
        f"expected (G, P)=({resolved.ns.n_gpu}, {resolved.ns.n_per_gpu}), "
        f"got {log_z.shape}"
    )
    assert np.all(np.isfinite(log_z)), f"log_evidence not finite: {log_z}"

    # Live walkers retained their (G, P, n_walkers, n_atoms, 3) layout.
    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (
        resolved.ns.n_gpu,
        resolved.ns.n_per_gpu,
        4,
        8,
        3,
    )


# ---------------------------------------------------------------------------
# Multi-GPU variant — pressure replicas distributed 2-per-device on
# whatever JAX exposes (either 2 or 4 local devices).
# ---------------------------------------------------------------------------


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

# RE every 2 iters → 2 swap rounds during the run, exercising the
# cross-device gather + accept/reject path.
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
    """MACE NPT NS with pressure replicas distributed 2-per-GPU.

    Same code path as :func:`test_mace_full_pipeline` but with multiple
    real devices and an inner vmap of width 2, exercising the full
    pmap(vmap(vmap)) hierarchy.  Mirrors the SrTiO3 burn-in failure
    mode that drove the earlier debugging.  Runs on either 2 or 4 local
    devices.
    """
    pytest.importorskip("mace_jax")

    from jaxrens.cli.resolve import resolve
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootSpec
    from jaxrens.sampling.batch_descriptor import PmapVmapRuns

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

    run_multi_gpu_from_config(resolved)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"mace_mgpu.run{r:02d}.energies"
        traj_path = out / f"mace_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    final_ckpt = out / "mace_mgpu.final.checkpoint.h5"
    assert final_ckpt.exists()

    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (n_gpu, 2, 4, 8, 3)
