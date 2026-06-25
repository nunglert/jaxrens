# TODO

## Ensemble specifications in RunSpec
The RunSpec currently takes only a pressure, this needs to be more general to also account for chemical potentials, etc. 

target_acceptance

- The migrator has a bug: it emits n_walkers/n_cull into temperature-termination specs that the current schema forbids, so migrate-ns-inp --validate fails on
  any temperature-terminated config. Easy fix in migrate.py:_handle_converge_down_to_T (stop adding those keys). Want me to patch it?

- A separate SingleRun bug: my first smoke test used a single pressure and crashed in the adaptation logger (Expected shape (1, 5), got (1, 1) at
  monitor.py:516 → adaptation_log.py:300). The real config uses 24 pressures (VmapRuns path), which works fine — but the single-replica baseline-row path looks
  broken. Note this is unrelated to your working-tree changes in nested_sampling.py. Let me know if you'd like me to dig into it.

- ImportErrors for backend dependencies should suggest "pip install .[neuralil]" etc. 

- [done] for restarts, do we also save stepsizes? Now persisted in the
  checkpoint (`step_sizes` dataset) and restored into the live population by
  `init_ns`. No `SamplerState` needed — `step_sizes` is the only sampler state
  that persists between iterations (adaptation is stateless per call). Legacy
  checkpoints (no field) fall back to the configured initial step.

- what happens if outfile_prefix is not set?

- check which XLA flags from the batch script we actually need, e.g. should we force no autotuning?

- trial_batch_size in auto stepsize handler? Did we resolve that problem?

- re_stats plot in CLI is outdated
