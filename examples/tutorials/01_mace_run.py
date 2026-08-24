# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jaxrens[mace]",
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
# # A MACE nested-sampling run: Si-16, NPT
#
# This tutorial runs jaxrens end-to-end with a **MACE-MP** foundation
# potential on a 16-atom silicon system in the NPT ensemble. It shows:
#
# - how to convert a foundation MACE model to a JAX model (once);
# - how the same YAML `checkpoint_path` accepts whatever the converter
#   emits, via the single {func}`~jaxrens.backends.mace.create_mace`
#   front door;
# - the MACE-specific backend knobs (`supercell_trafo`,
#   `max_neighbors_list`).
#
# Unlike the Lennard-Jones tutorial, this one needs a **GPU** and the
# `mace-jax` stack (`pip install -e ".[mace-convert]"`). The prose here
# is written to render without execution; set
# `JAXRENS_DOCS_EXECUTE=always` on a GPU box to actually run it.
#
# See the {doc}`../user/mace_models` guide for the full story on
# conversion, formats, and troubleshooting.

# %% [markdown]
# ## 0. Get a JAX model (shell step, run once)
#
# The conversion is a command-line step, not Python. On a GPU node:
#
# ```bash
# # certifi CA bundle — only needed when downloading a foundation model
# export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
#
# # download the "small" MACE-MP model and convert it to JAX params
# mace-jax-from-torch --foundation mp --model-name small --output small-jax.npz
# ```
#
# That writes `small-jax.npz` (msgpack params — despite the extension)
# and a sibling `small-jax.json` (config). Point `checkpoint_path` at
# the `.npz`; the loader finds the `.json` next to it.
#
# For a self-contained run without a download, this tutorial instead
# uses the tiny bundled fixture at `tests/_assets/models/mace_mp_small/`
# (a `config.json` + `params.msgpack` bundle directory) so it works
# from a fresh checkout.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

print("JAX devices:", jax.devices())

# Locate the bundled MACE fixture relative to the repo root.  Swap this
# for your own converted model, e.g. Path("small-jax.npz").
REPO_ROOT = Path.cwd()
while (
    not (REPO_ROOT / "pyproject.toml").exists()
    and REPO_ROOT != REPO_ROOT.parent
):
    REPO_ROOT = REPO_ROOT.parent
MODEL_PATH = REPO_ROOT / "tests" / "_assets" / "models" / "mace_mp_small"
print("model path:", MODEL_PATH)

# %% [markdown]
# ## Loading a model directly (the front door)
#
# {func}`~jaxrens.backends.mace.create_mace` is the single entry point
# for *every* on-disk shape a MACE model can take — a bundle directory
# (as here), a `.json`/`.msgpack` pair, an orbax `.ckpt`, the converter's
# `.npz`, or a `.pkl`. You normally don't call it yourself (the config
# path below does), but it's useful to see what a loaded backend exposes.

# %%
from jaxrens.backends.mace import create_mace

backend = create_mace(model_path=str(MODEL_PATH), supercell_trafo=(3, 3, 3))
print("backend         :", type(backend).__name__)
print("r_cutoff        :", backend.r_cutoff)
print("num_species     :", backend.num_species)
print("atomic_numbers  :", backend.atomic_numbers[:8], "...")

# %% [markdown]
# ## The config
#
# Fields map 1-to-1 to YAML keys (`jaxrens dump-schema` lists them all).
# The only MACE-specific part is the `backend` block: `checkpoint_path`
# accepts a bundle dir, a `.json`/`.msgpack` pair, an orbax `.ckpt`, the
# converter's `.npz`, or a `.pkl` — all through one loader.
#
# `supercell_trafo` must satisfy `min(cell_diag · s) ≥ 2 · r_cutoff`;
# MACE-MP has `r_cutoff = 6 Å`, so we tile generously. `start_species`
# is `"14 16"` — 16 Si atoms (Z = 14).

# %%
config_dict = {
    "run": {
        "n_live": 32,
        "max_iterations": 200,
        "n_mcmc_steps": 10,
        "n_extra": 15,
        "seed": 42,
    },
    "backend": {
        "type": "mace",
        "checkpoint_path": str(MODEL_PATH),
        "supercell_trafo": [3, 3, 3],
        "periodic": True,
        "max_neighbors_list": [35, 50, 65, 85, 100],
        "max_neighbors_offset": 4,
    },
    "ensemble": {
        "type": "npt",
        "pressure": 1.0,
        "pressure_units": "gpa",
    },
    "moves": [
        {"type": "galilean", "n_reflect": 4, "step_size": 0.03, "weight": 2.0},
        {"type": "volume", "step_size": 0.05, "weight": 16.0},
        {"type": "shear", "step_size": 0.02, "weight": 8.0},
        {"type": "stretch", "step_size": 0.02, "weight": 8.0},
    ],
    "termination": [
        {"type": "iteration", "max_iterations": 200},
    ],
    "adaptation": {
        "full_auto": True,
        "defaults": {
            "min_rate": 0.3,
            "max_rate": 0.5,
            "adjust_factor": 1.5,
            "step_size_max": 0.2,
        },
    },
    "init": {
        "start_species": "14 16",
        "random_initialise_pos": True,
        "pos_randomization_mode": "grid",
        "grid_distance": 1.5,
        "start_energy_ceiling_per_atom": 100.0,
        "initial_walk": {
            "n_walks": 3,
            "walklength": 20,
            "adjust_interval": 1,
            "emax_offset_per_atom": 1.0,
        },
    },
    "cell": {
        "max_volume_per_atom": 60.0,
        "min_volume_per_atom": 12.0,
        "min_aspect_ratio": 0.5,
        "flat_V_prior": False,
    },
    "output": {
        "format": "none",
        "working_dir": ".",
        "out_file_prefix": "mace_si16_tutorial",
        "info_interval": 25,
        # Large intervals effectively disable disk writes for the demo.
        "traj_interval": 100000,
        "snapshot_interval": 100000,
        "checkpoint_interval": 100000,
    },
}

# %% [markdown]
# ## Schema → resolve → run
#
# pydantic validates the shape, the resolver builds the backend instance
# on `resolved.base_backend` (this is where `create_mace` loads the
# model), and `run_from_config` executes the NS loop. Pass that pre-built
# backend through as `base_backend=` — exactly what `jaxrens run` does —
# so the MACE model is loaded once, not rebuilt from the flat config. The
# first MACE evaluation triggers a JIT compile, so expect a pause before
# iterations start moving.

# %%
from jaxrens.cli.resolve import resolve
from jaxrens.cli.run import run_from_config
from jaxrens.cli.schema import RootSpec

root = RootSpec.model_validate(config_dict)
resolved = resolve(root)

print(f"n_atoms          = {resolved.init.initial_positions.shape[-2]}")
print(f"backend type     = {resolved.backend.backend_type}")
print(f"checkpoint_path  = {resolved.backend.checkpoint_path}")
print(f"move descriptors : {[m.name for m in resolved.move_descriptors]}")

# %%
result = run_from_config(
    resolved.ns,
    list(resolved.moves),
    resolved.backend,
    resolved.output,
    base_backend=resolved.base_backend,
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
# ## What's next
#
# - Swap `checkpoint_path` for your own converted model — a downloaded
#   foundation `.npz`, a bundle directory, a `.ckpt`, whatever the
#   converter or a collaborator handed you. The loader takes them all.
# - Scale up: raise `run.n_live` to 128–512 and `max_iterations` for a
#   production evidence estimate; run multiple pressures with
#   pressure-RENS across GPUs.
# - Move the config to a YAML file and drive it from the CLI:
#   `jaxrens run -c config.yaml`.
