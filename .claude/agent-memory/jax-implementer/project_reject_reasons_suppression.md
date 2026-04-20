---
name: Per-move reject_reasons column suppression
description: MoveKernel.reject_reasons field and monitor column filtering, implemented 2026-04-19
type: project
---

Added `reject_reasons: frozenset[str]` to `MoveKernel` (default `frozenset({"energy"})`). Schema specs set overrides via `_reject_reasons()` in `cli/schema/moves.py`: `VolumeMoveSpec` → `{"energy","cell","prior"}`, `ShearMoveSpec`/`StretchMoveSpec` → `{"energy","cell"}`, all others keep default.

`run_ns` emits `info["move_reject_reasons"]` (tuple of frozensets) on adapt-fired iterations alongside `move_names`. `_format_reject_breakdown` gained `reasons_used: frozenset | None` parameter; when provided, filters to declared columns only; appends `???=N` if undeclared reasons appear. Monitor `on_iteration` passes `move_reject_reasons[k]` into `_format_reject_breakdown` per row (both batched and single-run paths).

**Why:** Galilean/random_walk/HMC never emit cell or prior rejection — printing `C=0% P=0%` was pure noise. Python-side display metadata change only; JAX-traced code (`reject_reason` scalar int32, `reject_reason_counts_per_move` shape `(n_moves, 4)`) unchanged.

**How to apply:** When adding a new move spec in `cli/schema/moves.py`, always override `_reject_reasons()` if the move kernel can emit cell (bucket 2) or prior (bucket 3) rejections. Read the kernel source to confirm. Default stays energy-only.
