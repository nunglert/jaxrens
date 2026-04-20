---
name: Task C — dual_averaging_update removal and run_ns_parallel migration
description: step_size.py deleted, run_ns/run_ns_parallel both use bisection only, adapt_warmup removed, 2026-04-18
type: project
---

Task C landed 2026-04-18.

**Why:** Consolidate adaptation to a single path (bisection via `adjust_step_size`) and remove the Nesterov dual-averaging fallback that was only active for `run_ns(use_full_auto=False)`.

**What was deleted:**
- `sampling/adaptation/step_size.py` — `AdaptationState`, `init_adaptation`, `dual_averaging_update`, `get_step_size`
- `adapt_warmup` parameter from `run_ns` and `run_ns_parallel`
- `adapt_state` / `init_adaptation` init and `dual_averaging_update` update calls in `run_ns`
- Python-side `_last_acc_rates_per_move` etc. caches in `run_ns` (replaced by loop-scoped vars)
- Entire per-run adaptation state (`adapt_states`) in `run_ns_parallel`

**What changed:**
- `run_ns(use_full_auto=False)` keeps step sizes at `initial_step_size` forever — behaviour change
- `run_ns_parallel` migrated to vmapped `adjust_step_size`:
  - Per-move `jax.jit(jax.vmap(_per_run))` closures, closed over static config
  - PRNG: `adapt_keys` advanced via `jax.vmap(jax.random.split)` each adjust call
  - Step sizes: `(n_runs, n_moves)` written back as `broadcast_to((n_runs, n_walkers, n_moves))`
- `run_ns_parallel` gained same params as `run_ns`: `per_move_fns`, `move_descriptors`, `adjust_interval`, `adjust_n_samples`, `adjust_max_rounds`, `adjust_factor`
- `cli/run.py` call site: `adapt_warmup` kwarg removed
- `test_adaptation.py` fully rewritten (7 bisection-based tests)
- `test_nested_sampling.py` gains `TestParityNRunsOne` and `TestParallelVmappedAdjustJIT`

**Restart format:** checkpoint.py never serialized `AdaptationState` — no breaking restart change.

**How to apply:** When looking at adaptation code, there is no dual-averaging path. All adaptation is bisection. `run_ns(use_full_auto=False)` = static step sizes.
