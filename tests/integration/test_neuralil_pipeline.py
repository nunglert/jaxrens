"""End-to-end integration: tiny carbon NS through the NeuralIL backend.

Mirrors :mod:`tests.integration.test_lj_pipeline` and :mod:`test_mace_pipeline`
but for NeuralIL.  The point is to exercise the *NeuralIL wiring* — backend
build, model-pickle load, the resolver's pre-NS energy eval, the bucketed
``max_neighbors_list`` ladder, and a few MCMC steps with
``value_and_grad(NeuralIL)`` — not to validate any scientific result.

Skipped when:

* ``neuralil`` is not importable.
* The ``tests/fixtures/neuralil_tiny`` model pickle is missing.

Sized for the integration tier: two-pressure multi-run (``[0.0, 1.0]`` GPa)
routed through ``run_multi_gpu_from_config``; CPU-runnable (pmap-on-one-device
+ vmap), no GPU gate.

Note: ``NeuralILBackendSpec.build_backend()`` always uses
``supercell_trafo=(1, 1, 1)``; we size the cell ≥ ``2 * r_cut`` so the
default-supercell neighbor search is sound.  The model's species table is
``['C', 'H', 'N', 'O']``; with ``NeuralILBackend`` now exposing
``atomic_numbers``, ``start_species: "1 8"`` (8 atoms of Z=1) maps to type
index 1 = 'H' — chemically meaningful, but the test only asserts on shapes,
not chemistry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "neuralil_tiny"
_MODEL_PKL = _FIXTURE / "model.pkl"


_CONFIG_YAML = """
run:
  n_live: 4
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42

backend:
  type: neuralil
  checkpoint_path: PLACEHOLDER
  # Cell is large enough vs. r_cut=3.5 Å that the default
  # supercell_trafo=(1,1,1) used by NeuralILBackendSpec is sufficient.
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

# Only IterationTermination — keeps n_dead deterministic for the assertions.
termination:
  - type: iteration
    max_iterations: 5

# Adaptation must not fire mid-run; set adjust interval > max.
adaptation:
  full_auto: true
  full_auto_steps: 100
  defaults:
    min_rate: 0.3
    max_rate: 0.5
    adjust_factor: 1.5
    step_size_max: 0.3

init:
  start_species: "1 8"      # 8 atoms (Z=1 → 'H' in NeuralIL's table)
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
  # V/atom large enough to easily satisfy 2*r_cut PBC requirement at the
  # default supercell_trafo=(1,1,1) (cubic side > 7 Å for 8 atoms means
  # min_volume_per_atom > 7^3 / 8 ≈ 43).
  max_volume_per_atom: 200.0
  min_volume_per_atom: 50.0
  min_aspect_ratio: 0.5
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: neuralil_smoke
  info_interval: 1
  traj_interval: 1
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.neuralil
@pytest.mark.heavy
def test_neuralil_full_pipeline(tmp_path: Path) -> None:
    """Two-pressure NeuralIL NS run.  Exercises:

    * Resolver — config → ``ResolvedMultiRunConfig`` (two-pressure path).
    * Backend build — ``create_neuralil`` loading the
      ``neuralil_tiny/model.pkl`` bundle.
    * Initial walker sampling + ``_finalise_initial_energies_and_counts``
      (no-``max_neighbors_for`` path for NeuralIL).
    * Bucketed kernel dispatch — ``max_neighbors_list`` ladder.
    * ``init_ns_multi_gpu`` building a ``(G, P, ...)`` NSState.
    * Batched burn-in (1 walk, 2 MCMC steps).
    * NS loop with galilean (``value_and_grad(NeuralIL)``) + volume moves
      under pmap-vmap dispatch (single device, ``n_gpu=1, n_per_gpu=2``).
    * Streamed I/O — per-replica ``.energies`` / ``.traj.extxyz``, plus
      global HDF5 checkpoints.
    """
    pytest.importorskip("neuralil")

    from jaxrens.cli.resolve import (
        ResolvedMultiRunConfig,
        expand_multi_run_or_cohort,
    )
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootConfig

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_PKL)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootConfig.model_validate(raw)
    resolved = expand_multi_run_or_cohort(root)

    # Two-pressure list → multi-run dispatcher.
    assert isinstance(resolved, ResolvedMultiRunConfig), (
        "Two-pressure config should route through the multi-GPU dispatcher."
    )

    run_multi_gpu_from_config(resolved)

    # ---- On-disk artefacts ---------------------------------------------------
    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    assert n_total == 2, f"expected 2 replicas, got {n_total}"

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"neuralil_smoke.run{r:02d}.energies"
        traj_path = out / f"neuralil_smoke.run{r:02d}.traj.extxyz"
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

    final_ckpt = out / "neuralil_smoke.final.checkpoint.h5"
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
        resolved.ns.n_gpu, resolved.ns.n_per_gpu, 4, 8, 3,
    )


# ---------------------------------------------------------------------------
# Multi-GPU variant — 8 pressure replicas distributed 2-per-device on 4 GPUs.
# ---------------------------------------------------------------------------

_REQUIRED_DEVICES = 4


_NEURALIL_MULTI_GPU_CONFIG_YAML = """
run:
  n_live: 4
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42

backend:
  type: neuralil
  checkpoint_path: PLACEHOLDER
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
  max_volume_per_atom: 200.0
  min_volume_per_atom: 50.0
  min_aspect_ratio: 0.5
  flat_V_prior: false

output:
  format: extxyz
  working_dir: PLACEHOLDER
  out_file_prefix: neuralil_mgpu
  info_interval: 1
  traj_interval: 1
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.multi_gpu
@pytest.mark.neuralil
@pytest.mark.gpu
@pytest.mark.heavy
def test_neuralil_multi_gpu_pipeline(tmp_path: Path) -> None:
    """NeuralIL NPT NS with 8 pressure replicas distributed 2-per-GPU on 4 GPUs.

    Same code path as :func:`test_neuralil_full_pipeline` but with four
    real devices and an inner vmap of width 2.  Exercises the full
    pmap(vmap(vmap)) hierarchy with the NeuralIL backend whose dynamics
    model is now built once at construction time (post-cache-refactor)
    and dispatched across devices via the resolver's ``base_backend``.
    """
    pytest.importorskip("neuralil")

    from jaxrens.cli.resolve import (
        ResolvedMultiRunConfig,
        expand_multi_run_or_cohort,
    )
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootConfig

    raw = yaml.safe_load(_NEURALIL_MULTI_GPU_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_PKL)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootConfig.model_validate(raw)
    resolved = expand_multi_run_or_cohort(root)
    assert isinstance(resolved, ResolvedMultiRunConfig)
    assert resolved.ns.n_gpu == _REQUIRED_DEVICES
    assert resolved.ns.n_per_gpu == 2

    run_multi_gpu_from_config(resolved)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu

    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"neuralil_mgpu.run{r:02d}.energies"
        traj_path = out / f"neuralil_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (5,)

    final_ckpt = out / "neuralil_mgpu.final.checkpoint.h5"
    assert final_ckpt.exists()

    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.shape == (_REQUIRED_DEVICES, 2)
    assert np.all(np.isfinite(log_z))

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (_REQUIRED_DEVICES, 2, 4, 8, 3)
