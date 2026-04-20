---
name: Modularization commit 3 — AdaptationManager
description: AdaptationManager in sampling/adaptation/manager.py encapsulates per-move JIT bisection; both run_ns call sites refactored 2026-04-18
type: project
---

AdaptationManager landed in `sampling/adaptation/manager.py`. Owns per-move JIT'd `adjust_step_size` callables, built once at `__init__`. Dispatches single/vmap via `BatchDescriptor`.

**Why:** Removes ~40-line duplication of per-move JIT construction + bisection loop between `run_ns` and `run_ns_parallel`. Both now delegate to `adapt_mgr.fires(i)` / `adapt_mgr.apply(...)`.

**API deviation from spec:** `apply` returns 3-tuple `(new_step_sizes, per_move_outputs, new_rng_key)` rather than 2-tuple. The RNG key is consumed across n_moves splits internally and must be returned to caller. Both `run_ns` and `run_ns_parallel` updated accordingly.

**How to apply:** When extending adaptation logic, modify `AdaptationManager.apply` and its `_build_jit_fns`. Keep `_pack_adjustment_info` in `nested_sampling.py` as-is — it accepts lists extracted from the `per_move_outputs` dict.
