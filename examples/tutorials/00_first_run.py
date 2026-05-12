# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jaxrens",
#   "matplotlib",
# ]
# ///

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.16.7
# ---

# %% [markdown]
# # First nested-sampling run: 8-atom Lennard-Jones, NPT
#
# This tutorial runs jaxrens end-to-end on the smallest physically
# meaningful problem: an 8-atom Lennard-Jones cluster in the NPT
# ensemble at reduced pressure P = 0.1 σ⁻³. It shows:
#
# - how a YAML config maps to a runtime dispatch;
# - the single-run code path (`jaxrens run` equivalent, via the
#   library API — nicer than shelling out from a notebook);
# - basic post-processing (log evidence, heat-capacity estimate
#   from the dead-energy ladder).
#
# The run is deliberately tiny (`n_live = 32`, `max_iterations = 500`)
# so it finishes in a few seconds on a CPU. The same config with
# `n_live = 128`, `max_iterations = 2000` would be a realistic demo
# run on a single GPU.

# %% [markdown]
# ## Imports

# %%
import jax
import jax.numpy as jnp
import numpy as np

print("JAX devices:", jax.devices())

# %% [markdown]
# ## The config
#
# We build a `RootSpec` programmatically from a dict. Fields map
# 1-to-1 to YAML keys; see `jaxrens dump-schema` for the full list.

# %%
config_dict = {
    "run": {
        "n_live": 32,
        "max_iterations": 500,
        "n_mcmc_steps": 10,
        "n_extra": 15,
        "seed": 42,
    },
    "backend": {
        "type": "lj",
        "epsilon": 1.0,
        "sigma": 1.0,
        "cutoff": 2.5,
        "periodic": True,
    },
    "ensemble": {
        "type": "npt",
        "pressure": 0.1,
        "pressure_units": "eva3",
    },
    "moves": [
        {"type": "galilean", "n_reflect": 6, "step_size": 0.1, "weight": 4.0},
        {"type": "volume", "step_size": 0.2, "weight": 1.0},
        {"type": "shear", "step_size": 0.08, "weight": 1.0},
        {"type": "stretch", "step_size": 0.08, "weight": 1.0},
    ],
    "termination": [
        {"type": "prior_mass", "threshold": 1.0e-3},
    ],
    "adaptation": {
        "full_auto": True,
        "defaults": {
            "min_rate": 0.3,
            "max_rate": 0.5,
            "adjust_factor": 1.5,
            "step_size_max": 0.2,
        },
        "per_move": {
            "galilean": {"step_size_max": 0.5},
        },
    },
    "init": {
        "start_species": "18 8",
        "random_initialise_pos": True,
        "pos_randomization_mode": "grid",
        "grid_distance": 1.0,
        "random_initialise_cell": True,
        "initial_walk": {
            "n_walks": 3,
            "walklength": 20,
            "adjust_interval": 1,
            "emax_offset_per_atom": 1.0,
        },
    },
    "cell": {
        "max_volume_per_atom": 20.0,
        "min_volume_per_atom": 0.5,
        "min_aspect_ratio": 0.6,
        "flat_V_prior": False,
    },
    "output": {
        "format": "none",
        "working_dir": ".",
        "out_file_prefix": "lj8_npt_tutorial",
        "info_interval": 50,
        # Large intervals effectively disable trajectory / snapshot /
        # checkpoint writes during this tutorial run.  The callbacks
        # are still constructed; they just never fire.
        "traj_interval": 100000,
        "snapshot_interval": 100000,
        "checkpoint_interval": 100000,
    },
}

# %% [markdown]
# ## Schema → resolve → run
#
# Three layers: pydantic validates the YAML shape, the resolver builds
# runtime objects (backend instance, MoveKernels, initial walkers),
# and `run_from_config` executes the NS loop.

# %%
from jaxrens.cli.resolve import resolve
from jaxrens.cli.run import run_from_config
from jaxrens.cli.schema import RootSpec

root = RootSpec.model_validate(config_dict)
resolved = resolve(root)

print(f"n_atoms         = {resolved.init.initial_positions.shape[-2]}")
print(f"n_moves         = {len(resolved.moves)}")
print(f"initial cell    = {np.asarray(resolved.init.initial_cells[0])}")
print(f"move descriptors: {[m.name for m in resolved.move_descriptors]}")

# %% [markdown]
# `run_from_config` takes the resolved dataclasses, wires callbacks
# internally, and returns a result dict with walker / dead-point
# arrays.

# %%
result = run_from_config(
    resolved.ns,
    list(resolved.moves),
    resolved.backend,
    resolved.output,
    initial_positions=resolved.init.initial_positions,
    initial_types=resolved.init.initial_types,
    initial_energies=resolved.init.initial_energies,
    initial_cells=resolved.init.initial_cells,
    move_descriptors=list(resolved.move_descriptors),
    initial_walk_config=resolved.initial_walk_config,
    adaptation_config=resolved.adaptation_cfg,
    termination_criteria=list(resolved.termination),
)

print()
print(f"Completed {int(result['iteration'])} NS iterations")
print(f"n_dead      = {int(result['n_dead'])}")
print(f"log_Z       = {float(result['log_evidence']):.4f}")

# %% [markdown]
# ## Post-processing: heat capacity
#
# The dead-point energies already encode everything we need to
# evaluate thermodynamic expectation values via the standard NS
# weights. `jaxrens.postprocess.thermodynamics` provides the
# analysis kernels.

# %%
from jaxrens.postprocess.thermodynamics import heat_capacity, log_evidence

n_dead = int(result["n_dead"])
dead_energies = jnp.asarray(result["dead_energies"])[:n_dead]
live_energies = jnp.asarray(result["energies"])

log_Z_full = log_evidence(
    dead_energies, live_energies, n_live=resolved.ns.n_live
)
print(f"log_Z (re-computed from dead+live) = {float(log_Z_full):.4f}")

# Heat capacity C(T) on a log-spaced temperature grid in reduced units.
# NS temperature range that's physically meaningful for LJ: roughly
# T* ∈ [0.2, 2.0].  Beta = 1 / (k_B T) with k_B = 1 in reduced units.
T_grid = jnp.logspace(jnp.log10(0.2), jnp.log10(2.0), 32)
beta_grid = 1.0 / T_grid

C = jax.vmap(
    lambda b: heat_capacity(b, dead_energies, live_energies, n_live=resolved.ns.n_live)
)(beta_grid)

# %% [markdown]
# Heat capacity vs temperature — a peak near T* ≈ 0.3–0.5 is the
# expected liquid–solid transition signal for the LJ cluster, but
# with only `n_live = 32` and `max_iter = 500` the statistics are
# quite noisy.  Increase both for a production run.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(T_grid, C, marker="o", markersize=3)
ax.set_xscale("log")
ax.set_xlabel("T* (reduced units)")
ax.set_ylabel("C_V (k_B)")
ax.set_title("Heat capacity from nested sampling, LJ-8 NPT")
ax.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# ## What's next
#
# - Switch to a bigger system: drop `run.n_live` up to 128–512 and
#   increase `run.max_iterations` for a production-quality evidence
#   estimate.
# - Run multiple pressures concurrently with pressure-RENS — see
#   `experiments/lj_rens/` and the (forthcoming) multi-GPU tutorial.
# - Swap the backend to MACE or NeuralIL for a physically realistic
#   system.
# - Use the CLI directly (`jaxrens run -c config.yaml`) once you've
#   moved the config to a YAML file; the command emits the same
#   result dict via registered I/O callbacks, writing trajectories,
#   energy logs, and checkpoints to disk.
