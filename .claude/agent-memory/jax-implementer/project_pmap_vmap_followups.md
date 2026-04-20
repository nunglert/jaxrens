---
name: PmapVmapRuns follow-ups: callbacks + ndim fixes
description: callbacks wired into run_ns_multi_gpu; monitor.py fixed for ndim>=2 log_evidence; checkpoint/postprocess flagged as follow-ups
type: project
---

Landed 2026-04-18.

**Task A:** `run_ns_multi_gpu` gains `callbacks: list[Any] | None = None` parameter; passed to `_run_loop` (was hardcoded `[]`). `on_finish` dispatch added after final evidence update. Three tests in `test_pmap_vmap.py` (`TestRunNsMultiGpuCallbacks`): count, `_batch` presence, `is_batched==True`.

**Task B ndim fixes in `cli/monitor.py`:**
- `ProgressCallback.on_iteration` batched per-move branch: `ss.ndim >= 2` + `reshape(-1, n_moves)` to flatten `(G,P,n_moves)`.
- `ProgressCallback.on_finish`: handles `ndim > 0` for batched `iteration` / `log_evidence` with `jnp.min/max`.
- `AdaptationCallback.on_iteration`: `elif ss_np.ndim >= 3` branch flattens `(G,P,n_moves)` → `(G*P,n_moves)`.

**Flagged as follow-up (not fixed):**
- `io/checkpoint.py:60` — `float(log_evidence)` assumes scalar; multi-GPU checkpointing out of scope.
- `postprocess/thermodynamics.py` — 1-D `dead_energies` assumption; multi-GPU post-processing out of scope.

**Monitor smoke test** added to `tests/test_postprocess_monitor.py::TestProgressCallbackPmapVmap`.

**Why:** `_run_loop` already dispatches callbacks and attaches `info["_batch"] = descriptor`; the fix was purely wiring the `callbacks` param through `run_ns_multi_gpu`. The ndim fixes ensure existing callbacks (ProgressCallback, AdaptationCallback) don't crash when `log_evidence` / `step_sizes` have shape `(G,P,...)`.
