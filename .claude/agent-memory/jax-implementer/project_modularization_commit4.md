---
name: Modularization commit 4
description: _run_loop in run_loop.py, thin wrappers for run_ns/run_ns_parallel, is_batched property, _is_batched fallback, 2026-04-18
type: project
---

`sampling/run_loop.py` introduced with `_run_loop(*)` and four helpers moved from `nested_sampling.py`:
- `_bump_cumulative_counters`
- `_inject_cumulative_into_info`
- `_pack_adjustment_info`
- `_dispatch_callbacks`

`run_ns` and `run_ns_parallel` are now thin wrappers (~50-55 lines each): validate → init → descriptor + AdaptationManager → `_run_loop` → package result.

**Why:** nested_sampling.py had ~1090 lines with duplicated outer-loop logic; commit 4 reduces it to ~720 lines.

**is_batched property added to BatchDescriptor hierarchy:**
- `SingleRun.is_batched = False`
- `VmapRuns.is_batched = True`
- `PmapVmapRuns.is_batched = True`

`_is_batched(ns_state, info=None)` in `cli/monitor.py` updated to prefer `info["_batch"].is_batched` when info dict is passed; falls back to ndim-sniff for backward compat.

`_run_loop` attaches `info["_batch"] = descriptor` before each callback dispatch.

**Known deviation:** `_pack_adjustment_info` is only called for `SingleRun` in `_run_loop` (not VmapRuns), because VmapRuns per_move_outputs have `(n_runs, n_moves, ...)` shape incompatible with the scalar-per-move `[int(v) for v in ...]` pattern. VmapRuns never called it before commit 4 either.

**emax extraction:** `is_vmap` → `jnp.max(pop.energy, axis=1)`; SingleRun → `jnp.max(pop.energy)`.

**circular import:** `run_loop.py` defers `from jaxrens.sampling.nested_sampling import ns_step` inside `_run_loop()` function body to break the circular import.

**Tests:** 11 new tests in `test_run_loop_equivalence.py`. Golden determinism, parallel parity (tolerance 5.0 log-units), overflow retry behavior (Python `continue` = advance to next i, NOT retry same i). 248 scoped tests pass. LJ-8 NPT bit-identical to commit 3.
