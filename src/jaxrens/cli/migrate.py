"""Migration helper: old-style ns.inp key=value -> new YAML-ready dict.

The entry point is ``migrate_ns_inp(raw)``.  It accepts the flat ``str->str``
dict produced by ``parse_input_file`` and returns a nested dict suitable for
``RootConfig.model_validate(...)``, plus a list of diagnostic log entries.

Usage::

    from jaxrens.cli.parser import parse_input_file
    from jaxrens.cli.migrate import migrate_ns_inp

    raw = parse_input_file("old.inp")
    result = migrate_ns_inp(raw)
    cfg_dict = result["config"]
    for entry in result["logs"]:
        print(entry["level"], entry["message"])
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_to_bool(s: str) -> bool:
    """Parse a jaxnest-style boolean string."""
    if s.lower() in ("t", "true", "1", "yes"):
        return True
    if s.lower() in ("f", "false", "0", "no"):
        return False
    raise ValueError(f"Cannot parse boolean from {s!r}")


def _parse_space_list_int(s: str) -> list[int]:
    return [int(x) for x in s.strip().split()]


def _parse_space_list_float(s: str) -> list[float]:
    return [float(x) for x in s.strip().split()]


def _parse_comma_list_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def _parse_comma_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def _nest(d: dict, *keys: str, value: Any) -> None:
    """Set ``d[k1][k2][...] = value``, creating intermediate dicts."""
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


# ---------------------------------------------------------------------------
# Explicitly dropped keys (from step-1 plan, section 7.8)
# ---------------------------------------------------------------------------

# Maps old key -> human-readable reason for dropping.
_DROPPED: dict[str, str] = {
    # GPU/batching infrastructure — no analogue in jaxrens scheduler
    "platform": "GPU platform selection is not needed; jaxrens uses the JAX default backend",
    "gpu_n_batch": "GPU batching is handled internally by jaxrens",
    "n_model_calls": "internal bookkeeping; not exposed in jaxrens",
    "n_model_calls_expected": "internal bookkeeping; not exposed in jaxrens",
    # Intra/inter swap steps (RENS) — not yet in jaxrens
    "n_swap_steps": "intra-swap (RENS) not yet implemented in jaxrens",
    "n_pressure_swap_steps": "intra-swap (RENS) not yet implemented in jaxrens",
    "n_intra_swap_steps": "intra-swap (RENS) not yet implemented in jaxrens",
    "inter_swap_interval": "inter-GPU replica exchange not yet implemented in jaxrens",
    "n_swap_cycles": "replica exchange not yet implemented in jaxrens",
    "add_swaps_on_top": "replica exchange not yet implemented in jaxrens",
    "add_atoms_swaps_on_top": "replica exchange not yet implemented in jaxrens",
    # Pivot moves — disabled in old parser
    "pivot_interval": "pivot moves are disabled in jaxnest and not implemented in jaxrens",
    # Energy perturbation — disabled in old parser
    "random_energy_perturbation": "disabled in jaxnest; not implemented in jaxrens",
    # Absolute energy ceiling (superseded by per-atom form)
    "start_energy_ceiling": "deprecated; use start_energy_ceiling_per_atom instead",
    # Active-learning commented-out block
    "num_uncertain_configs": "active-learning block (commented out in jaxnest); step 8",
    "compress_every": "active-learning block; step 8",
    "compress_size": "active-learning block; step 8",
    "compress_with_uncert": "active-learning block; step 8",
    "compress_with_self_score": "active-learning block; step 8",
    "compress_with_temperature": "active-learning block; step 8",
    "update_with_uncert": "active-learning block; step 8",
    "update_with_self_score": "active-learning block; step 8",
    "update_with_reference_score": "active-learning block; step 8",
    "update_with_training_score": "active-learning block; step 8",
    "update_with_temperature": "active-learning block; step 8",
    "training_data_processor": "active-learning block; step 8",
    "collect_temperature": "active-learning block; step 8",
    "save_energies_every": "active-learning block; step 8",
    "save_uncertainties_every": "active-learning block; step 8",
    "save_configs_every": "active-learning block; step 8",
    "n_batch": "active-learning block; step 8",
    "sample_size": "active-learning block; step 8",
    "final_compression_size": "active-learning block; step 8",
    "sample_propto_uncty": "active-learning block; step 8",
    "score_func": "active-learning block; step 8",
    "metric": "active-learning block; step 8",
    # Misc fields not yet in jaxrens
    "chemical_numbers": "implicit in backend choice; not a user-facing field in jaxrens",
    "n_sweep_move_atoms": "internal sweep parameter; not exposed in jaxrens",
    "step_rng_seed": "internal RNG seeding; use run.seed instead",
    "profile": "profiling flags are not in jaxrens yet",
    "profile_memory": "profiling flags are not in jaxrens yet",
    "start_cuda_profile_after": "CUDA profiling not in jaxrens",
    "debug": "debug flag is not in jaxrens",
    "reload_from_state": "state reload not yet in jaxrens",
    "supercell_trafo": "supercell transform not yet in jaxrens (available in MACEBackendSpec)",
    "supercell_dim_list": "supercell dim list not yet in jaxrens",
    "ground_state_energy_per_atom": "ground-state energy reference not in jaxrens yet",
    "capacity_factor_offset": "MACE-specific; not yet mapped",
    "capacity_factor_list": "MACE-specific; not yet mapped",
    "reference_edge_count_per_atom": "MACE-specific; not yet mapped",
    "hard_sphere_cutoff": "MACE-specific; not yet mapped",
    "n_runs": "computed automatically from cohort size",
    "n_per_gpu": "computed automatically",
    "lattice_param": "LJ-MSM specific; not in jaxrens",
    "alpha": "LJ-MSM specific; not in jaxrens",
    "r0": "Jagla potential; not in jaxrens",
    "w_r": "Jagla potential; not in jaxrens",
    "w_a": "Jagla potential; not in jaxrens",
    "b": "Jagla potential; not in jaxrens",
    "r_cut": "Jagla potential; not in jaxrens",
    "use_smooth_gradient": "Jagla potential; not in jaxrens",
    "random_initialise_pos": "init randomization; not yet exposed in InitConfig",
}

# ---------------------------------------------------------------------------
# Deferred keys — accepted but not yet consumed at runtime.
# Value goes into the config dict but triggers an INFO log entry.
# ---------------------------------------------------------------------------

# Maps old key -> (target nested path as tuple, coercion callable)
# These keys land in the new schema, but the resolver emits a deferred warning.
_DEFERRED: dict[str, tuple[tuple[str, ...], Any]] = {
    "write_traj_db": (("output", "write_traj_db"), _str_to_bool),
    "write_walkers_db": (("output", "write_walkers_db"), _str_to_bool),
    "wrap_atoms": (("output", "wrap_atoms"), _str_to_bool),
    "snapshot_clean": (("output", "snapshot_clean"), _str_to_bool),
    "save_stepsizes": (("output", "save_stepsizes"), _str_to_bool),
    "snapshot_time": (("output", "snapshot_time"), float),
    # Cell config (deferred: not threaded into move kernels yet)
    "max_volume_per_atom": (("cell", "max_volume_per_atom"), float),
    "min_volume_per_atom": (("cell", "min_volume_per_atom"), float),
    "MC_cell_min_aspect_ratio": (("cell", "min_aspect_ratio"), float),
    "MC_cell_flat_V_prior": (("cell", "flat_V_prior"), _str_to_bool),
    # InitialWalk (deferred: runtime code path doesn't exist yet)
    "initial_walk_N_walks": (("init", "initial_walk", "n_walks"), int),
    "initial_walk_walklength": (("init", "initial_walk", "walklength"), int),
    "initial_walk_adjust_interval": (("init", "initial_walk", "adjust_interval"), int),
    "initial_walk_Emax_offset_per_atom": (("init", "initial_walk", "emax_offset_per_atom"), float),
    "initial_walk_only": (("init", "initial_walk", "only"), str),
    "write_initial_walkers": (("init", "initial_walk", "write_initial_walkers"), _str_to_bool),
}

# ---------------------------------------------------------------------------
# Supported routing table
# ---------------------------------------------------------------------------
# Each entry maps old-key -> handler(value: str, cfg: dict, logs: list) -> None.
# Handlers place the coerced value into *cfg* (mutated in place) and may append
# INFO entries to *logs*.  They may also inspect *cfg* for context (e.g. knowing
# n_walkers when computing n_iter).

Handler = Any  # callable(value: str, cfg: dict, logs: list) -> None


def _h(path: tuple[str, ...], coerce: Any) -> Handler:
    """Factory: simple place-and-coerce handler."""
    def handler(value: str, cfg: dict, logs: list) -> None:
        _nest(cfg, *path, value=coerce(value))
    return handler


# ---------------------------------------------------------------------------
# A — Run identity
# ---------------------------------------------------------------------------

def _handle_n_walkers(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "run", "n_live", value=int(value))


def _handle_n_iter(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "run", "max_iterations", value=int(value))


def _handle_n_iter_times_fraction_killed(value: str, cfg: dict, logs: list) -> None:
    # Deferred resolution: store the fractional form; the caller must have
    # n_walkers and n_cull available to compute the final n_iter.
    # We store it under a private key; the final pass (below) resolves it.
    cfg.setdefault("_fractional_iter", {})["n_iter_times_fraction_killed"] = float(value)
    logs.append({
        "level": "INFO",
        "message": (
            "n_iter_times_fraction_killed stored for resolution after n_walkers/n_cull are known"
        ),
    })


def _handle_n_cull(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "run", "n_cull", value=int(value))


def _handle_convergence_threshold(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "run", "convergence_threshold", value=float(value))


def _handle_seed(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "run", "seed", value=int(value))


def _handle_step_rng_seed_alias(value: str, cfg: dict, logs: list) -> None:
    # Old field 'step_rng_seed' is closer to 'seed' than anything else.
    # Treat as a seed alias but only set if no explicit seed was given.
    if cfg.get("run", {}).get("seed") is None:
        _nest(cfg, "run", "seed", value=int(value))
    else:
        logs.append({
            "level": "INFO",
            "message": f"step_rng_seed={value!r} ignored; run.seed already set",
        })


# ---------------------------------------------------------------------------
# B — Termination
# ---------------------------------------------------------------------------

def _handle_converge_down_to_T(value: str, cfg: dict, logs: list) -> None:
    termination = cfg.setdefault("termination", [])
    termination.append({
        "type": "temperature",
        # n_walkers and n_cull are filled during post-processing
        "n_walkers": cfg.get("run", {}).get("n_live", 500),
        "target_temp": float(value),
        "n_cull": cfg.get("run", {}).get("n_cull", 1),
    })


def _handle_min_Emax(value: str, cfg: dict, logs: list) -> None:
    termination = cfg.setdefault("termination", [])
    termination.append({"type": "energy", "min_energy": float(value)})


# ---------------------------------------------------------------------------
# C — Ensemble / pressure
# ---------------------------------------------------------------------------

def _handle_pressure(value: str, cfg: dict, logs: list) -> None:
    """Route old MC_cell_P (space-sep list, GPa) into ensemble.pressure."""
    parts = [x.strip() for x in value.split() if x.strip()]
    if len(parts) == 1:
        pressures: float | list[float] = float(parts[0])
    else:
        pressures = [float(p) for p in parts]
    cfg.setdefault("ensemble", {})["type"] = "npt"
    cfg["ensemble"]["pressure"] = pressures
    cfg["ensemble"]["pressure_units"] = "gpa"


def _handle_pressure_conversion(value: str, cfg: dict, logs: list) -> None:
    # Only "gpa_to_eva3" and "none" are understood.
    norm = value.strip().lower()
    if norm == "gpa_to_eva3":
        cfg.setdefault("ensemble", {})["pressure_units"] = "gpa"
    elif norm == "none":
        # Already stored in eV/Å³; override the default "gpa" unit tag.
        cfg.setdefault("ensemble", {})["pressure_units"] = "eva3"
    else:
        logs.append({
            "level": "WARNING",
            "message": f"unknown pressure_conversion value {value!r}; assuming gpa",
        })


def _handle_semi_grand_potentials(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "WARNING",
        "message": (
            "semi_grand_potentials: semi-grand ensemble is not yet implemented "
            "in jaxrens; value preserved in _unknown for manual review"
        ),
    })
    cfg.setdefault("_unknown", {})["semi_grand_potentials"] = value


# ---------------------------------------------------------------------------
# D — Move counts (map to moves list)
# ---------------------------------------------------------------------------

def _handle_atom_traj_len(value: str, cfg: dict, logs: list) -> None:
    # Applies as n_reflect for GMC, n_leapfrog for HMC.
    # Store under a scratch key; move-building pass picks it up.
    cfg.setdefault("_move_defaults", {})["n_steps"] = int(value)


def _handle_n_gmc_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "gmc", int(value))


def _handle_n_hmc_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "hmc", int(value))


def _handle_n_random_shift_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "random_walk", int(value))


def _handle_n_single_atom_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "single_atom", int(value))


def _handle_n_single_atom_sweep_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "single_atom_sweep", int(value))


def _handle_n_atom_type_swap_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "single_atom_swap", int(value))


def _handle_n_semi_grand_steps(value: str, cfg: dict, logs: list) -> None:
    count = int(value)
    if count > 0:
        logs.append({
            "level": "WARNING",
            "message": (
                f"n_semi_grand_steps={count}: semi-grand moves are not yet "
                "implemented in jaxrens; value dropped"
            ),
        })


def _handle_n_cell_volume_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "volume", int(value))


def _handle_n_cell_stretch_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "stretch", int(value))


def _handle_n_cell_shear_steps(value: str, cfg: dict, logs: list) -> None:
    _add_or_set_move_count(cfg, "shear", int(value))


def _add_or_set_move_count(cfg: dict, move_type: str, count: int) -> None:
    cfg.setdefault("_move_counts", {})[move_type] = count


# ---------------------------------------------------------------------------
# E — Step sizes (map to adaptation section)
# ---------------------------------------------------------------------------

def _handle_MC_atom_step_size(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("_step_sizes", {})["gmc"] = float(value)
    cfg["_step_sizes"]["random_walk"] = float(value)


def _handle_MC_single_atom_step_size(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("_step_sizes", {})["single_atom"] = float(value)


def _handle_MC_cell_volume_per_atom_step_size(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("_step_sizes", {})["volume"] = float(value)


def _handle_MC_cell_stretch_step_size(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("_step_sizes", {})["stretch"] = float(value)


def _handle_MC_cell_shear_step_size(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("_step_sizes", {})["shear"] = float(value)


def _handle_MD_atom_timestep(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("_step_sizes", {})["hmc"] = float(value)


# ---------------------------------------------------------------------------
# F — Adaptation / step-size handler
# ---------------------------------------------------------------------------

def _handle_full_auto_step_sizes(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "adaptation", "full_auto", value=_str_to_bool(value))


def _handle_full_auto_steps(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "adaptation", "full_auto_steps", value=int(value))


def _handle_GMC_adjust_min_rate(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
        "gmc", {}
    )["min_rate"] = float(value)


def _handle_GMC_adjust_max_rate(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
        "gmc", {}
    )["max_rate"] = float(value)


def _handle_MC_adjust_step_factor(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("adaptation", {}).setdefault("defaults", {})["adjust_factor"] = float(value)


def _handle_cell_adjust_min_rate(value: str, cfg: dict, logs: list) -> None:
    for move in ("volume", "shear", "stretch"):
        cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
            move, {}
        )["min_rate"] = float(value)


def _handle_cell_adjust_max_rate(value: str, cfg: dict, logs: list) -> None:
    for move in ("volume", "shear", "stretch"):
        cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
            move, {}
        )["max_rate"] = float(value)


def _handle_MD_adjust_min_rate(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
        "hmc", {}
    )["min_rate"] = float(value)


def _handle_MD_adjust_max_rate(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
        "hmc", {}
    )["max_rate"] = float(value)


def _handle_MC_atom_step_size_max(value: str, cfg: dict, logs: list) -> None:
    for move in ("gmc", "random_walk"):
        cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
            move, {}
        )["step_size_max"] = float(value)


def _handle_MC_single_atom_step_size_max(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
        "single_atom", {}
    )["step_size_max"] = float(value)


def _handle_MD_atom_timestep_max(value: str, cfg: dict, logs: list) -> None:
    cfg.setdefault("adaptation", {}).setdefault("per_move", {}).setdefault(
        "hmc", {}
    )["step_size_max"] = float(value)


def _handle_adjust_step_interval(value: str, cfg: dict, logs: list) -> None:
    # In old parser: interval in iterations between step-size adjustments.
    # Maps to adaptation.adjust_n_samples (closest analogue).
    _nest(cfg, "adaptation", "adjust_n_samples", value=int(float(value)))


# ---------------------------------------------------------------------------
# G — Calculator / backend
# ---------------------------------------------------------------------------

def _handle_energy_calculator(value: str, cfg: dict, logs: list) -> None:
    norm = value.strip().lower()
    mapping = {
        "lj": "lj",
        "ljmsm": "lj",   # LJ-MSM → closest available; note in log
        "nn": "neuralil",
        "neuralil": "neuralil",
        "mace": "mace",
        "toy": "harmonic",
        "jagla": None,
    }
    new_type = mapping.get(norm)
    if new_type is None:
        logs.append({
            "level": "WARNING",
            "message": (
                f"energy_calculator={value!r} has no direct jaxrens backend; "
                "no backend type set"
            ),
        })
        return
    if norm == "ljmsm":
        logs.append({
            "level": "WARNING",
            "message": "energy_calculator=LJmsm mapped to lj; LJ-MSM parameters (lattice_param, alpha) are dropped",
        })
    _nest(cfg, "backend", "type", value=new_type)


def _handle_n_atoms(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "n_atoms", value=int(value))


def _handle_periodic(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "periodic", value=_str_to_bool(value))


def _handle_lj_r_cut(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "cutoff", value=float(value))


def _handle_lj_sigma(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "sigma", value=float(value))


def _handle_lj_eps(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "epsilon", value=float(value))


def _handle_pickle_file(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "checkpoint_path", value=value)


def _handle_max_neighbors_list(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "max_neighbors_list", value=_parse_space_list_int(value))


def _handle_max_neighbors_offset(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "backend", "max_neighbors_offset", value=int(value))


# ---------------------------------------------------------------------------
# H — Initialization
# ---------------------------------------------------------------------------

def _handle_start_species(value: str, cfg: dict, logs: list) -> None:
    if ":" in value:
        logs.append({
            "level": "WARNING",
            "message": (
                f"start_species={value!r} contains ':' (multi-composition form). "
                "Multi-composition cohort is deferred in step 6/7; the value is "
                "stored as-is and will cause a ValidationError until step 8."
            ),
        })
    _nest(cfg, "init", "start_species", value=value.strip())


def _handle_start_config_file(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "start_config_file", value=value.strip())


def _handle_start_walker_set(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "start_walker_set", value=value.strip())


def _handle_restart_file(value: str, cfg: dict, logs: list) -> None:
    if value.strip():
        _nest(cfg, "init", "restart_file", value=value.strip())


def _handle_start_energy_ceiling_per_atom(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "start_energy_ceiling_per_atom", value=float(value))


def _handle_random_initialise_cell(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "random_initialise_cell", value=_str_to_bool(value))


def _handle_init_distance_criterion(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "init_distance_criterion", value=float(value))


def _handle_random_init_max_n_tries(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "random_init_max_n_tries", value=int(value))


def _handle_pos_autoscale_cells(value: str, cfg: dict, logs: list) -> None:
    # Old: float (-1 means off, >0 means on); new: bool.
    fval = float(value)
    _nest(cfg, "init", "pos_autoscale_cells", value=(fval > 0))


def _handle_pos_randomization_mode(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "pos_randomization_mode", value=value.strip().lower())


def _handle_grid_distance(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "init", "grid_distance", value=float(value))


def _handle_cell_shape_equil_steps(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": (
            f"cell_shape_equil_steps={value!r} has no matching jaxrens field; dropped"
        ),
    })


# ---------------------------------------------------------------------------
# I — Output
# ---------------------------------------------------------------------------

def _handle_config_file_format(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "output", "format", value=value.strip().lower())


def _handle_traj_interval(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "output", "traj_interval", value=int(value))


def _handle_snapshot_interval(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "output", "snapshot_interval", value=int(value))


def _handle_info_interval(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "output", "info_interval", value=int(float(value)))


def _handle_out_file_prefix(value: str, cfg: dict, logs: list) -> None:
    # Old parser appends '.'; strip trailing dots/spaces for clean prefix.
    _nest(cfg, "output", "out_file_prefix", value=value.strip().rstrip("."))


def _handle_working_dir(value: str, cfg: dict, logs: list) -> None:
    _nest(cfg, "output", "working_dir", value=value.strip())


def _handle_snapshot_seq_pairs(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"snapshot_seq_pairs is not in OutputSchema; dropped",
    })


# ---------------------------------------------------------------------------
# J — Atom trajectory / GMC knobs
# ---------------------------------------------------------------------------

def _handle_GMC_no_reverse(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"GMC_no_reverse has no jaxrens field; dropped",
    })


def _handle_GMC_dir_perturb_angle(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"GMC_dir_perturb_angle has no jaxrens field; dropped",
    })


def _handle_GMC_dir_perturb_angle_during(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"GMC_dir_perturb_angle_during has no jaxrens field; dropped",
    })


def _handle_MD_atom_velo_pre_perturb(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"MD_atom_velo_pre_perturb has no jaxrens field; dropped",
    })


def _handle_MD_atom_velo_post_perturb(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"MD_atom_velo_post_perturb has no jaxrens field; dropped",
    })


def _handle_MD_atom_velo_flip_accept(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"MD_atom_velo_flip_accept has no jaxrens field; dropped",
    })


def _handle_atom_velo_rej_free_fully_randomize(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"atom_velo_rej_free_fully_randomize has no jaxrens field; dropped",
    })


def _handle_atom_velo_rej_free_perturb_angle(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"atom_velo_rej_free_perturb_angle has no jaxrens field; dropped",
    })


def _handle_MD_atom_energy_fuzz(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"MD_atom_energy_fuzz has no jaxrens field; dropped",
    })


def _handle_MD_atom_reject_energy_violation(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"MD_atom_reject_energy_violation has no jaxrens field; dropped",
    })


def _handle_KEmax_max_T(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"KEmax_max_T has no jaxrens field; dropped",
    })


# ---------------------------------------------------------------------------
# K — Active learning (non-commented fields that are in jaxnest)
# ---------------------------------------------------------------------------

def _handle_evaluate_uncertainties_traj(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"evaluate_uncertainties_traj: active-learning; step 8; dropped",
    })


def _handle_do_active_learning(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"do_active_learning: active-learning; step 8; dropped",
    })


def _handle_al_interval(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"al_interval: active-learning; step 8; dropped",
    })


def _handle_total_uncertainty_max(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"total_uncertainty_max: active-learning; step 8; dropped",
    })


# ---------------------------------------------------------------------------
# L — Monitoring
# ---------------------------------------------------------------------------

def _handle_monitor_step_interval(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"monitor_step_interval has no jaxrens field; dropped",
    })


def _handle_monitor_step_interval_times_fraction_killed(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"monitor_step_interval_times_fraction_killed has no jaxrens field; dropped",
    })


def _handle_T_estimate_finite_diff_lag(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"T_estimate_finite_diff_lag has no jaxrens field; dropped",
    })


def _handle_adjust_step_interval_times_fraction_killed(value: str, cfg: dict, logs: list) -> None:
    logs.append({
        "level": "INFO",
        "message": f"adjust_step_interval_times_fraction_killed: cannot resolve without n_walkers/n_cull at migration time; dropped",
    })


# ---------------------------------------------------------------------------
# M — Delta random seed (cohort)
# ---------------------------------------------------------------------------

def _handle_delta_random_seed(value: str, cfg: dict, logs: list) -> None:
    seeds = [int(x) for x in value.strip().split()]
    if len(seeds) == 1:
        _nest(cfg, "run", "seed", value=seeds[0])
    elif len(seeds) > 1:
        # Multi-seed cohort: store list; only first is used for now since
        # RunSchema.seed is scalar. Warn about partial support.
        _nest(cfg, "run", "seed", value=seeds[0])
        logs.append({
            "level": "WARNING",
            "message": (
                f"delta_random_seed contains {len(seeds)} values; only the "
                f"first ({seeds[0]}) is placed in run.seed. Remaining seeds "
                "are dropped until cohort expansion supports per-run seeds."
            ),
        })


# ---------------------------------------------------------------------------
# MC_cell_P alias
# ---------------------------------------------------------------------------

# MC_cell_P is the canonical pressure key in the old parser (space-separated GPa).

def _handle_MC_cell_P(value: str, cfg: dict, logs: list) -> None:
    _handle_pressure(value, cfg, logs)


# ---------------------------------------------------------------------------
# Master routing table
# ---------------------------------------------------------------------------

_ROUTING_TABLE: dict[str, Handler] = {
    # -- A: Run identity --
    "n_walkers": _handle_n_walkers,
    "n_iter": _handle_n_iter,
    "n_iter_times_fraction_killed": _handle_n_iter_times_fraction_killed,
    "n_cull": _handle_n_cull,
    "convergence_threshold": _handle_convergence_threshold,
    "seed": _handle_seed,
    # -- B: Termination --
    "converge_down_to_T": _handle_converge_down_to_T,
    "min_Emax": _handle_min_Emax,
    # -- C: Ensemble / pressure --
    "MC_cell_P": _handle_MC_cell_P,
    "pressure": _handle_pressure,  # alias that some old files use
    "pressure_conversion": _handle_pressure_conversion,
    "semi_grand_potentials": _handle_semi_grand_potentials,
    # -- D: Move counts --
    "atom_traj_len": _handle_atom_traj_len,
    "n_gmc_steps": _handle_n_gmc_steps,
    "n_hmc_steps": _handle_n_hmc_steps,
    "n_random_shift_steps": _handle_n_random_shift_steps,
    "n_single_atom_steps": _handle_n_single_atom_steps,
    "n_single_atom_sweep_steps": _handle_n_single_atom_sweep_steps,
    "n_atom_type_swap_steps": _handle_n_atom_type_swap_steps,
    "n_semi_grand_steps": _handle_n_semi_grand_steps,
    "n_cell_volume_steps": _handle_n_cell_volume_steps,
    "n_cell_stretch_steps": _handle_n_cell_stretch_steps,
    "n_cell_shear_steps": _handle_n_cell_shear_steps,
    # -- E: Step sizes --
    "MC_atom_step_size": _handle_MC_atom_step_size,
    "MC_single_atom_step_size": _handle_MC_single_atom_step_size,
    "MC_cell_volume_per_atom_step_size": _handle_MC_cell_volume_per_atom_step_size,
    "MC_cell_stretch_step_size": _handle_MC_cell_stretch_step_size,
    "MC_cell_shear_step_size": _handle_MC_cell_shear_step_size,
    "MD_atom_timestep": _handle_MD_atom_timestep,
    "MC_cell_volume_per_atom_prob": lambda v, c, l: None,  # ignored
    "MC_cell_stretch_prob": lambda v, c, l: None,
    "MC_cell_shear_prob": lambda v, c, l: None,
    "MC_atom_velo_step_size": lambda v, c, l: None,  # no HMC velo step in jaxrens
    # -- F: Adaptation --
    "full_auto_step_sizes": _handle_full_auto_step_sizes,
    "full_auto_steps": _handle_full_auto_steps,
    "GMC_adjust_min_rate": _handle_GMC_adjust_min_rate,
    "GMC_adjust_max_rate": _handle_GMC_adjust_max_rate,
    "MC_adjust_step_factor": _handle_MC_adjust_step_factor,
    "cell_adjust_min_rate": _handle_cell_adjust_min_rate,
    "cell_adjust_max_rate": _handle_cell_adjust_max_rate,
    "MD_adjust_min_rate": _handle_MD_adjust_min_rate,
    "MD_adjust_max_rate": _handle_MD_adjust_max_rate,
    "MD_adjust_step_factor": lambda v, c, l: None,  # no separate HMC factor in jaxrens
    "MC_atom_step_size_max": _handle_MC_atom_step_size_max,
    "MC_single_atom_step_size_max": _handle_MC_single_atom_step_size_max,
    "MD_atom_timestep_max": _handle_MD_atom_timestep_max,
    "MC_cell_volume_per_atom_step_size_max": lambda v, c, l: None,
    "MC_cell_stretch_step_size_max": lambda v, c, l: None,
    "MC_cell_shear_step_size_max": lambda v, c, l: None,
    "adjust_step_interval": _handle_adjust_step_interval,
    "adjust_step_interval_times_fraction_killed": _handle_adjust_step_interval_times_fraction_killed,
    "throw_small_error": lambda v, c, l: None,
    "adjust_smart": lambda v, c, l: None,
    # -- G: Calculator / backend --
    "energy_calculator": _handle_energy_calculator,
    "n_atoms": _handle_n_atoms,
    "periodic": _handle_periodic,
    "lj_r_cut": _handle_lj_r_cut,
    "lj_sigma": _handle_lj_sigma,
    "lj_eps": _handle_lj_eps,
    "pickle_file": _handle_pickle_file,
    "max_neighbors_list": _handle_max_neighbors_list,
    "max_neighbors_offset": _handle_max_neighbors_offset,
    # -- H: Initialization --
    "start_species": _handle_start_species,
    "start_config_file": _handle_start_config_file,
    "start_walker_set": _handle_start_walker_set,
    "restart_file": _handle_restart_file,
    "start_energy_ceiling_per_atom": _handle_start_energy_ceiling_per_atom,
    "random_initialise_cell": _handle_random_initialise_cell,
    "init_distance_criterion": _handle_init_distance_criterion,
    "random_init_max_n_tries": _handle_random_init_max_n_tries,
    "pos_autoscale_cells": _handle_pos_autoscale_cells,
    "pos_randomization_mode": _handle_pos_randomization_mode,
    "grid_distance": _handle_grid_distance,
    "cell_shape_equil_steps": _handle_cell_shape_equil_steps,
    # -- I: Output --
    "config_file_format": _handle_config_file_format,
    "traj_interval": _handle_traj_interval,
    "snapshot_interval": _handle_snapshot_interval,
    "info_interval": _handle_info_interval,
    "out_file_prefix": _handle_out_file_prefix,
    "working_dir": _handle_working_dir,
    "snapshot_seq_pairs": _handle_snapshot_seq_pairs,
    # -- J: Atom trajectory / GMC knobs --
    "GMC_no_reverse": _handle_GMC_no_reverse,
    "GMC_dir_perturb_angle": _handle_GMC_dir_perturb_angle,
    "GMC_dir_perturb_angle_during": _handle_GMC_dir_perturb_angle_during,
    "MD_atom_velo_pre_perturb": _handle_MD_atom_velo_pre_perturb,
    "MD_atom_velo_post_perturb": _handle_MD_atom_velo_post_perturb,
    "MD_atom_velo_flip_accept": _handle_MD_atom_velo_flip_accept,
    "atom_velo_rej_free_fully_randomize": _handle_atom_velo_rej_free_fully_randomize,
    "atom_velo_rej_free_perturb_angle": _handle_atom_velo_rej_free_perturb_angle,
    "MD_atom_energy_fuzz": _handle_MD_atom_energy_fuzz,
    "MD_atom_reject_energy_violation": _handle_MD_atom_reject_energy_violation,
    "KEmax_max_T": _handle_KEmax_max_T,
    # -- K: Active learning --
    "evaluate_uncertainties_traj": _handle_evaluate_uncertainties_traj,
    "do_active_learning": _handle_do_active_learning,
    "al_interval": _handle_al_interval,
    "total_uncertainty_max": _handle_total_uncertainty_max,
    # -- L: Monitoring --
    "monitor_step_interval": _handle_monitor_step_interval,
    "monitor_step_interval_times_fraction_killed": _handle_monitor_step_interval_times_fraction_killed,
    "T_estimate_finite_diff_lag": _handle_T_estimate_finite_diff_lag,
    # -- M: Delta random seed --
    "delta_random_seed": _handle_delta_random_seed,
}


# ---------------------------------------------------------------------------
# Post-processing: resolve scratch keys into final config shape
# ---------------------------------------------------------------------------

def _build_moves_from_scratch(cfg: dict, logs: list) -> None:
    """Construct the ``moves`` list from ``_move_counts`` and ``_step_sizes``."""
    counts: dict[str, int] = cfg.pop("_move_counts", {})
    sizes: dict[str, float] = cfg.pop("_step_sizes", {})
    move_defaults: dict[str, Any] = cfg.pop("_move_defaults", {})
    n_steps_default = move_defaults.get("n_steps", 8)

    if not counts:
        return

    moves = []
    # Cell moves need n_atoms in their spec.
    n_atoms = cfg.get("backend", {}).get("n_atoms", 13)

    for move_type, count in counts.items():
        if count <= 0:
            continue
        spec: dict[str, Any] = {"type": move_type, "weight": float(count)}
        if move_type in sizes:
            spec["step_size"] = sizes[move_type]
        if move_type in ("gmc", "galilean"):
            spec["n_reflect"] = n_steps_default
        elif move_type == "hmc":
            spec["n_leapfrog"] = n_steps_default
        elif move_type in ("volume", "shear", "stretch"):
            spec["n_atoms"] = n_atoms
        elif move_type == "single_atom_sweep":
            spec["n_atoms"] = n_atoms
        moves.append(spec)

    if moves:
        cfg.setdefault("moves", moves)
    # If moves already present (from a direct field not via counts), don't overwrite.


def _resolve_fractional_iter(cfg: dict, logs: list) -> None:
    """Resolve n_iter_times_fraction_killed using n_walkers and n_cull."""
    scratch = cfg.pop("_fractional_iter", {})
    if not scratch:
        return
    frac = scratch["n_iter_times_fraction_killed"]
    n_live = cfg.get("run", {}).get("n_live", 500)
    n_cull = cfg.get("run", {}).get("n_cull", 1)
    n_iter = int(round(frac / (float(n_cull) / float(n_live))))
    _nest(cfg, "run", "max_iterations", value=n_iter)
    logs.append({
        "level": "INFO",
        "message": (
            f"n_iter_times_fraction_killed={frac} resolved to "
            f"max_iterations={n_iter} with n_live={n_live}, n_cull={n_cull}"
        ),
    })


def _strip_private_keys(cfg: dict) -> None:
    """Remove internal scratch keys (prefixed ``_``), except ``_unknown``.

    ``_unknown`` is kept so callers can render its contents as YAML comments
    rather than live schema keys.
    """
    for k in list(cfg):
        if k.startswith("_") and k != "_unknown":
            del cfg[k]


def _ensure_required_sections(cfg: dict) -> None:
    """Guarantee sections that RootConfig requires are present with defaults."""
    # RootConfig.output has no default_factory — must always be present.
    cfg.setdefault("output", {})
    # RootConfig.run and RootConfig.backend must be present.
    cfg.setdefault("run", {})
    cfg.setdefault("backend", {})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def migrate_ns_inp(raw: dict[str, str]) -> dict[str, Any]:
    """Migrate a flat key->str dict from an old ns.inp file to a RootConfig dict.

    The returned dict has two keys:
      - ``"config"``  — nested dict ready for ``RootConfig.model_validate``
      - ``"logs"``    — list of dicts with ``"level"`` and ``"message"`` keys

    Severity levels used in logs:
      ``"INFO"``    — informational; migration chose a sensible default.
      ``"WARNING"`` — data was dropped or needs manual review.

    Unknown keys (not in the routing table, drop list, or deferred map) are
    placed under ``"_unknown"`` in cfg (printed as YAML comments by the CLI)
    and emit a WARNING.

    Args:
        raw: Flat string->string dict as returned by ``parse_input_file``.

    Returns:
        Dict with ``"config"`` and ``"logs"`` keys.
    """
    cfg: dict[str, Any] = {}
    logs: list[dict[str, str]] = []

    # Take a mutable copy so callers' raw dict is unmodified.
    remaining = dict(raw)

    # -- Deferred keys: place value and emit INFO --
    for old_key, (path, coerce) in _DEFERRED.items():
        if old_key in remaining:
            value = remaining.pop(old_key)
            try:
                _nest(cfg, *path, value=coerce(value))
                logs.append({
                    "level": "INFO",
                    "message": (
                        f"deferred: {old_key}={value!r} -> "
                        f"{'.'.join(path)}; accepted by schema but not yet "
                        "consumed by runtime"
                    ),
                })
            except (ValueError, TypeError) as exc:
                logs.append({
                    "level": "WARNING",
                    "message": f"deferred key {old_key}={value!r} failed coercion: {exc}; dropped",
                })

    # -- Dropped keys: emit WARNING, do not place in config --
    for old_key, reason in _DROPPED.items():
        if old_key in remaining:
            value = remaining.pop(old_key)
            logs.append({
                "level": "WARNING",
                "message": f"dropped: {old_key}={value!r} — {reason}",
            })

    # -- Routing table: dispatch each known key --
    for old_key, handler in _ROUTING_TABLE.items():
        if old_key in remaining:
            value = remaining.pop(old_key)
            try:
                handler(value, cfg, logs)
            except (ValueError, TypeError) as exc:
                logs.append({
                    "level": "WARNING",
                    "message": (
                        f"handler for {old_key}={value!r} raised {type(exc).__name__}: "
                        f"{exc}; key dropped"
                    ),
                })

    # -- Unknown keys --
    for old_key, value in remaining.items():
        logs.append({
            "level": "WARNING",
            "message": (
                f"unknown old-parser key: {old_key!r}; value {value!r} preserved "
                "under _unknown for manual review"
            ),
        })
        cfg.setdefault("_unknown", {})[old_key] = value

    # -- Post-processing --
    _resolve_fractional_iter(cfg, logs)
    _build_moves_from_scratch(cfg, logs)
    _ensure_required_sections(cfg)
    _strip_private_keys(cfg)

    return {"config": cfg, "logs": logs}
