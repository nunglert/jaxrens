# Pathology: log output columns mislabeled

## Symptom

Values in the log don't make physical sense — `acc=117.69` (an acceptance rate > 1), or `log_Z=0.0000` while the run is clearly progressing. Usually caught by the user noticing a value that can't physically exist.

## Root cause pattern

Format-string and arg-tuple misalignment in `logger.info(...)` calls. Python's `%` formatting is positional; mismatched arg order produces plausible-looking but wrong labels.

## Known occurrences

### 2026-04-18: nested_sampling.py:544

Format: `"iter=%d  Emax=%.6g  log_Z=%.4f  acc=%.2f  ss=%s"`
Wrong args (in order): iter, emax, **acceptance_rate**, **log_evidence**, ss.
Symptom: log_Z column showed 0.00-0.9 range (acc values); acc column showed 50-100+ range (log_Z values).

### 2026-04-18: cli/monitor.py ProgressCallback.on_iteration

Same bug in the other periodic log. Same swap.

## Detection heuristic

Before any deeper diagnosis on "run looks wrong", check log column sanity:
- `log_Z` should start negative or near 0, grow monotonically, end tens-to-hundreds positive for typical systems.
- `acc` should be in [0, 1]. Any value ≥ 1 or < 0 is mislabeled.
- `Emax` should decrease monotonically (modulo tiny overflows). If it rises, either the log is mislabeled or there's a real sampling bug.

## Fix recipe

If mislabeled, inspect the exact `logger.info(format, *args)` call, match format specifiers to args positionally. Never assume the code is correct — both bugs above were shipped.

## Adjacent clean-up opportunity

Two separate `logger.info` sites (`run_ns` + `ProgressCallback`) had the same bug. Having two periodic-summary sources is itself a smell. After 2026-04-18 reorg, ProgressCallback is canonical; the duplicate in `nested_sampling.py` was removed. If a future edit re-introduces a duplicate, flag it.
