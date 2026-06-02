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

from . import multi_gpu_n_devices


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
  adjust_interval: 100
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

    * Resolver — config → ``ResolvedConfig`` (two-pressure path).
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
        resolve,
    )
    from jaxrens.sampling.batch_descriptor import PmapVmapRuns
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootSpec

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_PKL)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)

    # Two-pressure list → multi-run dispatcher.
    assert isinstance(resolved.batcher, PmapVmapRuns), (
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
# Multi-GPU variant — pressure replicas distributed 2-per-device on
# whatever JAX exposes (either 2 or 4 local devices).
# ---------------------------------------------------------------------------


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
    """NeuralIL NPT NS with pressure replicas distributed 2-per-GPU.

    Same code path as :func:`test_neuralil_full_pipeline` but with
    multiple real devices and an inner vmap of width 2.  Exercises the
    full pmap(vmap(vmap)) hierarchy with the NeuralIL backend whose
    dynamics model is now built once at construction time
    (post-cache-refactor) and dispatched across devices via the
    resolver's ``base_backend``.  Runs on either 2 or 4 local devices.
    """
    pytest.importorskip("neuralil")

    from jaxrens.cli.resolve import (
        resolve,
    )
    from jaxrens.sampling.batch_descriptor import PmapVmapRuns
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootSpec

    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_NEURALIL_MULTI_GPU_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_PKL)
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
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (n_gpu, 2, 4, 8, 3)


# ---------------------------------------------------------------------------
# Sharded single-run variant — ONE NS run whose live population is sharded
# across all local GPUs (ShardedSingleRun / run_sharded_from_config).  This is
# a different code path from the pmap-vmap multi-run tests above: there is a
# single pressure, a single set of output files, and the dead-point /
# snapshot writers run on a per-iteration WalkerState gathered off the mesh.
# ---------------------------------------------------------------------------


_NEURALIL_SHARDED_CONFIG_YAML = """
run:
  n_live: 8                 # divisible by 2 and 4 local GPUs
  max_iterations: 5
  n_mcmc_steps: 2
  n_extra: 0
  seed: 42
  shard_n_gpu: PLACEHOLDER  # set to the local device count at runtime

backend:
  type: neuralil
  checkpoint_path: PLACEHOLDER
  periodic: true
  max_neighbors_list: [30, 50, 80]
  max_neighbors_offset: 4

# Single pressure → still a single NS run; sharding distributes its live
# population across GPUs (a pressure LIST would route to the multi-run path
# and the resolver rejects combining it with shard_n_gpu).
ensemble:
  type: npt
  pressure: 0.0
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
  out_file_prefix: neuralil_sharded
  info_interval: 1
  traj_interval: 1          # write a dead point every iteration
  snapshot_interval: 100
  checkpoint_interval: 100
  log_level: info
"""


@pytest.mark.multi_gpu
@pytest.mark.neuralil
@pytest.mark.gpu
@pytest.mark.heavy
def test_neuralil_sharded_multi_gpu_pipeline(tmp_path: Path) -> None:
    """Single NeuralIL NPT NS run sharded across all local GPUs.

    Exercises the ``ShardedSingleRun`` path end-to-end through
    ``run_sharded_from_config``:

    * Resolver routes ``run.shard_n_gpu > 1`` to ``ShardedSingleRun(n_gpu)``.
    * ``init_ns_sharded`` shards the K=8 population to ``(G, K/G, ...)``.
    * The NS loop runs ``ns_step_sharded`` (galilean ``value_and_grad`` +
      volume moves) with the cross-shard ``lax.psum`` invariants.
    * **The per-iteration dead point is gathered off the mesh and written**
      via the single-run ``TrajectoryCallback`` — this is the path that
      regressed when ``info["dead_walker"]`` kept its leading shard axis, so
      the trajectory-frame assertions below are the regression guard.
    * Streamed I/O is single-run-shaped: one ``.energies`` and one
      ``.traj.extxyz`` (no ``.runNN`` suffix), plus a gathered checkpoint.
    """
    pytest.importorskip("neuralil")

    from jaxrens.cli.resolve import resolve
    from jaxrens.cli.run import run_sharded_from_config
    from jaxrens.cli.schema import RootSpec
    from jaxrens.sampling.batch_descriptor import ShardedSingleRun

    n_gpu = multi_gpu_n_devices()

    raw = yaml.safe_load(_NEURALIL_SHARDED_CONFIG_YAML)
    raw["backend"]["checkpoint_path"] = str(_MODEL_PKL)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["run"]["shard_n_gpu"] = n_gpu

    root = RootSpec.model_validate(raw)
    resolved = resolve(root)

    assert isinstance(resolved.batcher, ShardedSingleRun), (
        "shard_n_gpu > 1 with a single pressure should route through the "
        "sharded single-run dispatcher."
    )
    assert resolved.batcher.n_gpu == n_gpu

    run_sharded_from_config(resolved)

    # ---- On-disk artefacts: single-run-shaped (no per-replica suffix) -------
    out = tmp_path / "out"
    from jaxrens.io.energy_log import EnergyLogger

    energies_path = out / "neuralil_sharded.energies"
    traj_path = out / "neuralil_sharded.traj.extxyz"
    assert energies_path.exists(), f"missing energy log: {energies_path}"
    assert traj_path.exists(), f"missing trajectory: {traj_path}"

    log = EnergyLogger.read(energies_path)
    assert log.energies.shape == (5,), (
        f"expected 5 dead-point energies, got {log.energies.shape}"
    )

    # Dead-point trajectory: one frame per iteration, each the full 8-atom
    # gathered walker.  Before the sharded dead_walker fix this write crashed
    # (the WalkerState reached the writer with a leading shard axis), so these
    # assertions guard that regression.
    from ase.io import read as ase_read

    frames = ase_read(str(traj_path), index=":")
    assert len(frames) == 5, f"expected 5 trajectory frames, got {len(frames)}"
    for fr in frames:
        assert len(fr) == 8, f"expected 8 atoms per frame, got {len(fr)}"

    # ---- Final checkpoint: gathered to single-run (K, ...) shapes -----------
    final_ckpt = out / "neuralil_sharded.final.checkpoint.h5"
    assert final_ckpt.exists(), f"missing final checkpoint: {final_ckpt}"

    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(final_ckpt)
    log_z = np.asarray(state["log_evidence"])
    assert log_z.size == 1, f"expected scalar log_evidence, got shape {log_z.shape}"
    assert np.all(np.isfinite(log_z)), f"log_evidence not finite: {log_z}"

    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape == (8, 8, 3), (
        f"expected gathered (K=8, n_atoms=8, 3), got {saved_positions.shape}"
    )
