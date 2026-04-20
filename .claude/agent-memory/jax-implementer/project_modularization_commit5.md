---
name: Modularization commit 5
description: Parallel restart support — restart_states param in init_ns_parallel/run_ns_parallel, 13 new tests, 2026-04-18
type: project
---

`init_ns_parallel` gains `restart_states: list[RestartBundle] | None` parameter.
When provided, each `restart_states[i]` (or `None` for a fresh run) is passed as
`restart_state=` to the per-run `init_ns` call.  Length validation raises `ValueError`
when `len(restart_states) != n_runs`.

`run_ns_parallel` gains the same `restart_states` parameter and forwards it to
`init_ns_parallel`.  Default is `None` (full backward compat).

Mixed restart: `[bundle, None, bundle, ...]` is valid — restarts some runs from
checkpoint, fresh-starts others.

**Why:** single-run restart already worked via `run_ns(restart_state=bundle)`.
Parallel runs had no equivalent.  Commit 5 closes that gap.

**How to apply:** Both functions are at their prior call sites; the new parameter
is keyword-only at the tail of the signature so all existing call sites are
unaffected.

13 new tests in `tests/test_parallel_restart.py`:
- `TestInitNsParallelRestart` (7): seeds n_dead, iteration, log_evidence; mixed
  restart; wrong-length ValueError; dead-arrays padding; no-restart fresh start.
- `TestRunNsParallelRestart` (6): n_dead increments, finite log_evidence, output
  shapes, parity with single-run restart, fresh-start default, wrong-length propagation.

**Design note:** restart logic is entirely at the `init_ns`/`init_ns_parallel`
level; `_run_loop` receives a fully-initialized `NSState` and is unaware of
restart.  This is the minimal-change design consistent with single-run restart.

**Open follow-ups:**
- VmapRuns-aware `_pack_adjustment_info` (currently skipped for parallel runs).
- `BatchDescriptor.init_state()` method if a future commit routes the full init
  path through the descriptor.
