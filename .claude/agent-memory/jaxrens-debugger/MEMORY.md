# jaxrens-debugger memory index

Each entry points to a pathology memory file. Read all of these at the start of a diagnosis session. Add a new file when you identify a pathology not already catalogued. Update an existing file when you learn a new root cause, fix, or symptom for an already-catalogued one.

## Pathology files

- [Cell moves 0% acceptance](pathology_cell_moves_zero_acc.md) — volume/shear/stretch get stuck at zero, step sizes collapse
- [TF32 matmul floor on cell moves](pathology_tf32_cell_move_floor.md) — GPU default-precision matmul introduces ~4e-3 Å noise on `positions @ I`, floors ss at 1e-20
- [Step sizes explode](pathology_step_size_explosion.md) — ss reaches 1e7+, all moves reject
- [Termination at n_live+2](pathology_early_termination.md) — prior-mass fires immediately after n_live iterations
- [Log columns mislabeled](pathology_log_columns_mislabeled.md) — format-string / arg-order mismatches
- [Slow Monitor observables](pathology_slow_monitor_observables.md) — plot.py takes minutes per 200 T points
- [NeuralILwithMorse NaN grad on padded atoms](pathology_neuralil_morse_padded_inf_nan.md) — center_at_atoms emits inf radii for type<0 pairs; MorseModel's exp(-a*(inf-b)) yields 0*inf=NaN in the gradient even though the forward is fine

## Diagnostic templates

- [Kernel isolation script template](template_kernel_diagnostic.md) — how to build /tmp/diag_*.py for any move kernel

## Lessons

- [Config vs log mtime skew](lesson_config_log_skew.md) — user may have edited config after the bad run
- [initial_step_size plumbing uses first move](lesson_initial_step_size_plumbing.md) — per-move YAML step_sizes 2..N ignored at init

## Conventions

- `pathology_<snake_case_symptom>.md` for failure-mode knowledge
- `template_<name>.md` for reusable diagnostic scaffolds
- `lesson_<snake_case>.md` for single-sentence takeaways that don't fit a full pathology
