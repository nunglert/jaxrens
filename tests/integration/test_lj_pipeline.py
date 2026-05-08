"""End-to-end smoke: LJ multi-pressure NS through every major stage.

Single comprehensive test that exercises:

* **Resolver** — load YAML → ``ResolvedMultiRunConfig``, build LJ backend,
  per-replica EnsembleBackend wrapping, initial walker sampling, cell
  shape walk, initial-energy evaluation.
* **Init** — ``init_ns_multi_gpu`` building a ``(G, P, ...)`` NSState.
* **Burn-in** — ``initial_walk(batched=True)`` with adaptation.
* **NS loop** — ``_run_loop`` driven by termination criteria; the
  ``while True`` form (no static iteration cap).
* **Inter-replica exchange** — pressure-RENS swaps every few iterations.
* **Step-size adaptation** — ``full_auto`` bisection during the loop.
* **Streamed I/O** — ``EnergyLogger`` (``.energies``), trajectory writer
  (``.traj.extxyz``), and HDF5 checkpoint (``.checkpoint.h5`` /
  ``.final.checkpoint.h5``).

Sized to finish in a few seconds on CPU.  Intentionally minimal:
``n_live=16``, ``max_iterations=30``, 8 LJ atoms, 2 replicas — just
enough to fire each stage at least once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

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

# Two replicas + RE every 10 iters → at least 2 swap attempts during the run.
inter_re:
  flavor: pressure
  every: 10
  n_swap_cycles: 1

moves:
  - type: galilean
    n_reflect: 4
    step_size: 0.1
    weight: 2.0
  - type: volume
    step_size: 0.2
    weight: 1.0

# Single termination criterion = the IterationTermination derived from
# run.max_iterations (no prior_mass to keep the run length deterministic
# for the assertions below).
termination:
  - type: iteration
    max_iterations: 30

# full_auto with adjust_interval=10 → bisection fires at iter 10 / 20.
adaptation:
  full_auto: true
  full_auto_steps: 10
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
  # Loose bounds so the volume sampler + cell_shape_walk can't produce a
  # degenerate small/anisotropic cell that breaks grid positioning.
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
    """Run a 30-iter LJ NPT NS with two pressure replicas and verify each
    stage produced its expected on-disk artefact.

    Asserts on:

    * Result dict shape and finiteness.
    * ``.energies``, ``.traj.extxyz``, periodic checkpoint, and final
      checkpoint files exist per replica / globally.
    * The streamed ``.energies`` file matches ``max_iterations`` in length.
    * HDF5 checkpoint loads back without errors and ``log_evidence`` is
      finite with the expected ``(G, P) = (1, 2)`` shape under the
      single-device CI runner.
    """
    from jaxrens.cli.resolve import (
        ResolvedMultiRunConfig,
        expand_multi_run_or_cohort,
    )
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootConfig

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")

    root = RootConfig.model_validate(raw)
    resolved = expand_multi_run_or_cohort(root)
    assert isinstance(resolved, ResolvedMultiRunConfig), (
        "Two-pressure config should route through the multi-GPU dispatcher."
    )

    result = run_multi_gpu_from_config(resolved)

    # ---- Result-dict sanity --------------------------------------------------
    log_z = np.asarray(result["log_evidence"])
    assert log_z.shape == (resolved.ns.n_gpu, resolved.ns.n_per_gpu), (
        f"log_evidence shape {log_z.shape} != "
        f"({resolved.ns.n_gpu}, {resolved.ns.n_per_gpu})"
    )
    assert np.all(np.isfinite(log_z)), f"log_evidence contains non-finite values: {log_z}"

    n_dead = np.asarray(result["n_dead"])
    assert np.all(n_dead == 30), (
        f"Expected n_dead == max_iterations == 30 per replica, got {n_dead}"
    )

    # Live walkers retained their shape after the run.
    positions = np.asarray(result["positions"])
    n_atoms = 8
    assert positions.shape == (
        resolved.ns.n_gpu, resolved.ns.n_per_gpu, 16, n_atoms, 3,
    )

    # ---- On-disk artefacts ---------------------------------------------------
    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    for r in range(n_total):
        energies_path = out / f"smoke.run{r:02d}.energies"
        traj_path = out / f"smoke.run{r:02d}.traj.extxyz"
        assert energies_path.exists(), f"missing per-replica energy log: {energies_path}"
        assert traj_path.exists(), f"missing per-replica trajectory: {traj_path}"
        # EnergyLogger writes a header + one data line per iteration; read via
        # the canonical parser to avoid coupling to format details.
        from jaxrens.io.energy_log import EnergyLogger
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (30,), (
            f"{energies_path.name}: expected 30 data entries (one per iteration), "
            f"got {log.energies.shape}"
        )

    # Initial / periodic / final checkpoints all on disk.
    for suffix in ("initial.checkpoint.h5", "checkpoint.h5", "final.checkpoint.h5"):
        path = out / f"smoke.{suffix}"
        assert path.exists(), f"missing {path}"

    # ---- Final checkpoint round-trip ----------------------------------------
    from jaxrens.io.checkpoint import load_checkpoint

    state = load_checkpoint(out / "smoke.final.checkpoint.h5")
    assert np.all(np.isfinite(np.asarray(state["log_evidence"])))
    # Live walkers are saved in the checkpoint with batched leading axes.
    saved_positions = np.asarray(state["positions"])
    assert saved_positions.shape[-2:] == (n_atoms, 3)
    # Dead arrays were intentionally dropped from HDF5 in the recent refactor;
    # the canonical record lives in .energies / .traj.  Either absent
    # (load_checkpoint returns None) or present-but-empty is acceptable.
    de = state.get("dead_energies")
    if de is not None:
        assert np.asarray(de).size == 0 or np.asarray(de).shape[-1] in (0, 30)


@pytest.mark.parametrize("missing", ["pressures", "inter_re"])
def test_lj_pipeline_smoke_variants(tmp_path: Path, missing: str) -> None:
    """Variant smokes: scalar-pressure single-run path, and multi-pressure
    without inter_re (cohort path).  Both go through different dispatchers
    and cover the non-RENS code paths.

    Skipped when the variant doesn't actually pivot to a different
    dispatcher — e.g. ``missing="inter_re"`` still hits the multi-replica
    multi-GPU path; ``missing="pressures"`` collapses to single-run.
    """
    from jaxrens.cli.resolve import (
        ResolvedMultiRunConfig,
        expand_multi_run_or_cohort,
    )
    from jaxrens.cli.run import run_from_config, run_multi_gpu_from_config
    from jaxrens.cli.schema import RootConfig

    raw = yaml.safe_load(_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["output"]["out_file_prefix"] = f"smoke_{missing}"

    if missing == "pressures":
        raw["ensemble"]["pressure"] = 0.5  # scalar → single-run cohort
        raw.pop("inter_re", None)
    elif missing == "inter_re":
        raw.pop("inter_re", None)  # multi-run path without RENS

    root = RootConfig.model_validate(raw)
    resolved = expand_multi_run_or_cohort(root)

    if isinstance(resolved, ResolvedMultiRunConfig):
        # Without inter_re, the resolver may still build a ResolvedMultiRunConfig
        # if the pressure list is multi-element; exercise that path.
        run_multi_gpu_from_config(resolved)
    else:
        # Single-run (cohort of length 1 for scalar pressure).
        assert len(resolved) >= 1
        from jaxrens.cli.cli import _run_one
        _run_one(resolved[0])

    # Per-config artefact paths use the parametrized prefix.
    out = tmp_path / "out"
    final_ckpts = list(out.glob(f"smoke_{missing}*final.checkpoint.h5"))
    assert final_ckpts, f"no final checkpoint produced for variant {missing!r}"


# ---------------------------------------------------------------------------
# Multi-GPU variant — pressure replicas distributed 2-per-device on
# whatever JAX exposes (either 2 or 4 local devices).
# ---------------------------------------------------------------------------


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
  every: 10
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
  full_auto_steps: 10
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
    """LJ NPT NS with pressure replicas distributed 2-per-GPU.

    Same code path as :func:`test_lj_full_pipeline` but with multiple
    real devices and an inner vmap of width 2, exercising the full
    pmap(vmap(vmap)) hierarchy rather than the degenerate
    ``n_per_gpu=1`` case.  Hits cross-device pmap communication for
    pressure-RENS swaps and ``full_auto`` bisection adaptation.  Runs on
    either 2 or 4 local devices.
    """
    from jaxrens.cli.resolve import (
        ResolvedMultiRunConfig,
        expand_multi_run_or_cohort,
    )
    from jaxrens.cli.run import run_multi_gpu_from_config
    from jaxrens.cli.schema import RootConfig

    n_gpu = multi_gpu_n_devices()
    n_total = n_gpu * 2

    raw = yaml.safe_load(_LJ_MULTI_GPU_CONFIG_YAML)
    raw["output"]["working_dir"] = str(tmp_path / "out")
    raw["ensemble"]["pressure"] = raw["ensemble"]["pressure"][:n_total]

    root = RootConfig.model_validate(raw)
    resolved = expand_multi_run_or_cohort(root)
    assert isinstance(resolved, ResolvedMultiRunConfig)
    assert resolved.ns.n_gpu == n_gpu, (
        f"expected n_gpu={n_gpu}, got {resolved.ns.n_gpu}"
    )
    assert resolved.ns.n_per_gpu == 2, (
        f"expected n_per_gpu=2, got {resolved.ns.n_per_gpu}"
    )

    result = run_multi_gpu_from_config(resolved)

    log_z = np.asarray(result["log_evidence"])
    assert log_z.shape == (n_gpu, 2)
    assert np.all(np.isfinite(log_z))

    n_dead = np.asarray(result["n_dead"])
    assert np.all(n_dead == 30)

    out = tmp_path / "out"
    n_total = resolved.ns.n_gpu * resolved.ns.n_per_gpu
    from jaxrens.io.energy_log import EnergyLogger

    for r in range(n_total):
        energies_path = out / f"smoke_mgpu.run{r:02d}.energies"
        traj_path = out / f"smoke_mgpu.run{r:02d}.traj.extxyz"
        assert energies_path.exists()
        assert traj_path.exists()
        log = EnergyLogger.read(energies_path)
        assert log.energies.shape == (30,)
