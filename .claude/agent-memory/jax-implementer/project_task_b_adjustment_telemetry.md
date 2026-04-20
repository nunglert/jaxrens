---
name: Task B adjustment telemetry
description: Per-adjust-call diagnostics added to .adaptation.h5 (n_rounds, converged, cap_hits, floor_hits, bracket_detected), schema v2, implemented 2026-04-18
type: project
---

`adjust_step_size` now returns 8 values: `(new_ss, final_rate, final_counts, n_rounds, converged, cap_hits, floor_hits, bracket_detected)`. `_process_rate_jax` returns 6 values: `(new_ss, converged, cap_hit, floor_hit, too_high, too_low)`. All new accumulators (`cap_hits`, `floor_hits`, `saw_too_high`, `saw_too_low`) are carried through the `lax.while_loop` — no Python control flow, fully vmap-safe.

**Why:** Task B of three-part refactor. `.adaptation.h5` previously recorded only final step sizes/rates; the new `adjustment_stats/` HDF5 group explains why the adapter landed where it did.

**How to apply:** Task C (vmap `adjust_step_size` in `run_ns_parallel`) can vmap the full 8-tuple return. All existing callers that unpacked 3 values must now unpack 8 (no 3-value callers remain after this task).

Schema: `adaptation_log_schema_version=2` attr on v2 files. `AdaptationLog.adjustment_stats` is `None` for v1 files. `AdaptationCallback.on_iteration` auto-collects `adjustment_*` info keys and `reject_counts_per_move` → maps to `reject_reason_counts` in HDF5.
