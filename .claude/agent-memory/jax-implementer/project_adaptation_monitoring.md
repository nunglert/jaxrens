---
name: Adaptation monitoring implementation
description: HDF5 adaptation trace + ProgressCallback multi-line log wired 2026-04-17
type: project
---

Adaptation monitoring was wired in on 2026-04-17. Key design decisions:

- `AdaptationLogger` (src/jaxrens/io/adaptation_log.py): buffered HDF5 writer, flushes every 1000 entries; empty logger creates no file.
- `AdaptationCallback` (src/jaxrens/cli/monitor.py): wraps logger, writes on iterations where `info["step_sizes_per_move"]` is present.
- Canonical periodic summary: `ProgressCallback.on_iteration` in `cli/monitor.py`. The duplicate `logger.info` in `nested_sampling.py` ~line 543 was removed; per-move adjust log downgraded to DEBUG.
- `info["step_sizes_per_move"]` and `info["acceptance_rates_per_move"]` are populated in `run_ns` only on iterations where full_auto adjustment fires (guarded by `i % adjust_interval == 0`).
- `Monitor.adaptation_trace` field added; `from_directory` loads `<prefix>.adaptation.h5` if present.
- `run_from_config` in `cli/run.py` creates `AdaptationLogger` and `AdaptationCallback` when `adaptation_config.full_auto` is True.

**Why:** User wants concise log + detailed HDF5 trace for postprocess plotting of step-size convergence.

**How to apply:** When extending the adaptation system, `info["step_sizes_per_move"]` is the signal that both `ProgressCallback` and `AdaptationCallback` listen to.
