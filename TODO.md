# TODO

## Ensemble specifications in RunSpec

- target_acceptance

- The migrator has a bug: it emits n_walkers/n_cull into temperature-termination specs that the current schema forbids, so migrate-ns-inp --validate fails on
  any temperature-terminated config. Easy fix in migrate.py:_handle_converge_down_to_T (stop adding those keys). Want me to patch it?

- A separate SingleRun bug: my first smoke test used a single pressure and crashed in the adaptation logger (Expected shape (1, 5), got (1, 1) at
  monitor.py:516 → adaptation_log.py:300). The real config uses 24 pressures (VmapRuns path), which works fine — but the single-replica baseline-row path looks
  broken. Note this is unrelated to your working-tree changes in nested_sampling.py. Let me know if you'd like me to dig into it.

- check which XLA flags from the batch script we actually need, e.g. should we force no autotuning?

- trial_batch_size in auto stepsize handler? Did we resolve that problem?

- re_stats plot in CLI is outdated

- Do we carefully obey a walker state contract? Check this. We shouldnt switch arbitrarily between dict and WalkerState

- debug.log is not really used, mostly redundant

- better OOM errors. E.g. in the burn-in phase "Decrease walker_batch_size", in ns_loop "Decrease n_extra", ...

- jaxtyping in replica exchange manager

- ship a jax-mace foundation model? or maybe even the utils to pull and convert? So it becomes more plug and play 
