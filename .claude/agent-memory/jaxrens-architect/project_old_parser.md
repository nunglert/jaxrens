---
name: old jaxnest input_parser scope
description: What the old jaxnest/input_parser.py covers and its structure
type: project
---

`/home/nunglert/code/jaxnest_refactor/jaxns-devAS/src/jaxnest/input_parser.py` (1342 lines) is the old parser. It groups parameters into 9 `SimpleNamespace` blocks:

1. **algorithm_args**: n_cull, n_walkers, n_iter, n_iter_times_fraction_killed, adjust_step_interval(_times_fraction_killed), pressure (list), semi_grand_potentials (list), delta_random_seed (list), pivot_interval, step_rng_seed
2. **computation_args**: platform (cpu/gpu/multi-gpu), gpu_n_batch, n_calc_batch (derived), n_gpu_parallel, working_dir, profile, profile_memory, start_cuda_profile_after, debug, reload_from_state, bit_precision (in arg_parser only)
3. **termination_args**: min_Emax, converge_down_to_T
4. **cellstep_args**: max/min volume_per_atom, MC_cell_min_aspect_ratio, MC_cell_flat_V_prior
5. **run_monitor_args**: info_interval, monitor_step_interval(_times_fraction_killed), T_estimate_finite_diff_lag
6. **initialization_args**: start_species, n_species, start_config_file, start_walker_set, restart_file, energy ceilings, random_initialise_pos/cell, pos_autoscale_cells, pos_randomization_mode, grid_distance, cell_shape_equil_steps, initial_walk_*, random_init_max_n_tries, init_distance_criterion, write_initial_walkers
7. **calculator_args**: energy_calculator (LJ/NN/MACE/Jagla/Toy/LJMSM), supercell_trafo, pickle_file, LJ params (r_cut/sigma/eps), Jagla params (r0/w_r/w_a/b/r_cut), MACE params (reference_edge_count_per_atom, capacity_factor_*, hard_sphere_cutoff), NN params (max_neighbors_list/offset), ground_state_energy_per_atom, supercell_dim_list
8. **structure_output_args**: config_file_format, traj_interval, snapshot_interval/time/seq_pairs/clean, out_file_prefix, write_traj_db, write_walkers_db, wrap_atoms, save_stepsizes
9. **atomstep_args**: GMC_no_reverse, GMC_dir_perturb_angle[_during], MD_atom_velo_pre/post_perturb, MD_atom_velo_flip_accept, atom_velo_rej_free_*, MD_atom_energy_fuzz, MD_atom_reject_energy_violation, KEmax_max_T, atom_traj_len
10. **randomwalk_args**: n_gmc/hmc/cell_volume/cell_stretch/cell_shear/random_shift/single_atom/single_atom_sweep/intra_swap/atom_type_swap/semi_grand_steps, inter_swap_interval, n_model_calls(_expected), n_swap_cycles, n_sweep_move_atoms, add_swaps_on_top, add_atoms_swaps_on_top, random_energy_perturbation, re_type (derived)
11. **stepsize_args**: MC_atom/single_atom/atom_velo_step_size, MD_atom_timestep, MC_cell_volume_per_atom_step_size/prob, MC_cell_stretch_step_size/prob, MC_cell_shear_step_size/prob
12. **stepsize_handler_args**: full_auto_step_sizes, full_auto_steps, {cell,GMC,MD}_adjust_{min,max}_rate, MC/MD_adjust_step_factor, *_step_size_max, throw_small_error, adjust_smart
13. **active_learning_args**: evaluate_uncertainties_traj, do_active_learning, al_interval, total_uncertainty_max (+ many commented-out)

**Cross-group interactions** (things the old parser derives at parse time, not just stores):
- `n_iter` derivable from `n_iter_times_fraction_killed * n_walkers / n_cull`
- `adjust_step_interval` and `monitor_step_interval` similarly
- Batch-axis alignment over `pressure / semi_grand_potentials / delta_random_seed / species_list` (seed generation via os.urandom when missing)
- `n_model_calls_expected` summed from all n_*_steps * atom_traj_len then divided by n_calc_batch/n_per_gpu
- `re_type` derived from which batch axis is nontrivial ("pressure"/"chemical"/"composition")
- Pressure unit conversion (GPa->eV/A^3)
- Various cross-validations (start_species XOR start_config_file; n_atom_type_swap_steps requires multi-species; active_learning requires NN calculator; cell moves required when random_initialise_cell; toy disallows cell moves)

**Dead / commented-out params to drop**: the big commented block in `active_learning_args` (num_uncertain_configs, compress_*, update_with_*, sample_size, etc.), `pivot_interval` (explicitly disabled), `random_energy_perturbation` (explicitly disabled), `n_swap_steps`/`n_pressure_swap_steps` backward-compat aliases (kept for `n_intra_swap_steps`), `semi_grand_potentials` block form `{Z:mu,...}` (code has a parser but input layer uses list form instead).

**How to apply:** When designing the new config, use these 13 groups as the parameter taxonomy. Derived-at-parse-time quantities should be expressed as methods/properties on the schema rather than computed in the parser; batch-axis logic belongs in a separate "expand" pass after validation, not inside the schema itself.
