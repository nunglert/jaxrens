# Hysteresis-gated bucket shrinking

## Context

The current overflow-retry mechanism in `src/jaxrens/sampling/run_loop.py` only ever **enlarges** the `max_neighbors` bucket (`_pick_next_bucket` at `:262-288` requires `b > current`). Once escalated, a run stays at the larger bucket even if the system later relaxes into a regime where a smaller bucket would suffice — wasting memory and dispatch overhead on every kernel call for the rest of the run.

Since JAX caches compiled functions per input signature, returning to a previously-seen bucket reuses the cached compilation. The only real risk is **thrashing** at a ladder boundary, which hysteresis (gap + dwell time) resolves.

## Design

Add a downward retry path next to the existing upward one, gated by two parameters:

- **Gap** (`shrink_margin`): only shrink when `observed_max + offset + margin <= next_smaller_bucket`. Default `margin = max_neighbors_offset` (so the shrink threshold mirrors the existing growth headroom).
- **Dwell** (`shrink_dwell`): require K consecutive iterations satisfying the gap condition before shrinking. Default `0` (= shrinking disabled), opt-in for backward compatibility.

The dwell counter lives in `_run_loop` Python state alongside `i` — not in `NSState`. Resets to 0 on any iteration that violates the gap. Resets to 0 after each shrink (so successive shrinks need K more iterations each).

Shrink one ladder step at a time, never bypassing intermediate entries.

## Files to touch

### `src/jaxrens/sampling/run_loop.py`

1. **Add `_pick_prev_bucket(true_max, current, ladder, offset, margin)` →  `int | None`** next to `_pick_next_bucket` (`:262-288`). Returns the largest ladder entry strictly less than `current` that still satisfies `entry >= true_max + offset + margin`. Returns `None` when no smaller entry qualifies. No exception — shrinking is best-effort.

2. **Add two parameters to `_run_loop`** (`:291`): `max_neighbors_shrink_dwell: int = 0`, `max_neighbors_shrink_margin: int | None = None` (default = `max_neighbors_offset` when None).

3. **Initialize `low_count = 0` before the `while True:` loop**.

4. **Insert a shrink-check block immediately after `ns_state = new_ns_state` (`:458`)**, before the inter-RE phase:
   - If `shrink_dwell == 0`: skip entirely (zero-cost when disabled).
   - Read `true_max = int(ns_state.population.max_neighbor_count.max())`.
   - Read `current = int(ns_state.population.max_neighbors)`.
   - Call `_pick_prev_bucket(...)`. If it returns `None`, reset `low_count = 0`.
   - Else: `low_count += 1`. If `low_count >= shrink_dwell`, log an info-level "shrinking bucket X -> Y" message, set `ns_state.population.max_neighbors = new_smaller`, reset `low_count = 0`.

5. **Reset `low_count = 0` inside the overflow-retry block** (after the bucket has just been grown — `:443-456`). Prevents a previously-accumulated low streak from triggering a spurious shrink on the next iteration.

### `src/jaxrens/state/config.py`

Add to `BackendConfig` (`:25-42`):

```python
max_neighbors_shrink_dwell: int = 0       # 0 disables shrink
max_neighbors_shrink_margin: int | None = None  # None → uses offset
```

### `src/jaxrens/cli/schema/backend.py`

Mirror the same two pydantic fields with docstrings.

### `src/jaxrens/cli/resolve.py` and `src/jaxrens/cli/run.py`

Thread the two fields through every `_run_loop` invocation. Grep-confirmed call sites:

- `cli/run.py:457-458, 524-525, 665-666, 853-854, 979-980, 1139-1140`
- `cli/resolve.py:1061-1062, 1367-1368`

Each currently passes `max_neighbors_list` and `max_neighbors_offset` — extend each kwarg block with the two new fields. Mechanical; ~16 lines total.

### `src/jaxrens/cli/migrate.py`

Add handlers for the two new keys mirroring `_handle_max_neighbors_list` / `_handle_max_neighbors_offset` (`:534-539`, `:896-897`). Only needed if backward-compat migration from the legacy `ns.inp` format must support them — likely not strictly required for new options (existing configs have `0` shrink_dwell and work unchanged).

## Tests

New file `tests/test_bucket_shrink.py`:

1. **`_pick_prev_bucket` unit tests** (parallel to existing `_pick_next_bucket` tests):
   - Returns next-smaller entry when target leaves enough headroom.
   - Returns `None` when target requires current or larger.
   - Returns `None` when current is already the smallest entry.
   - Honors the margin (would-shrink without margin, won't shrink with margin).

2. **Loop-level integration**:
   - Use a mock `ns_step` that returns a population with `max_neighbor_count` controllable per call.
   - Verify shrink fires exactly when low streak hits `shrink_dwell`.
   - Verify single spike in observed max resets the streak.
   - Verify `shrink_dwell = 0` keeps the bucket pinned (no shrink ever).
   - Verify an overflow during the wait resets the streak and re-grows the bucket cleanly.

3. **End-to-end smoke**:
   - Tiny LJ run, manually preload the bucket to a high entry (e.g. 50), enable `shrink_dwell = 5`, run 20 iterations on a sparse system whose true max is ~10, assert bucket ends at the smallest ladder entry that satisfies `true_max + offset + margin`.

## Verification

```bash
cd /home/nico.unglert/code/jaxrens
/home/nico.unglert/miniconda3/envs/jaxrens/bin/python -m pytest tests/test_bucket_shrink.py -v 2>&1 | tee /tmp/pytest_bucket_shrink.log
```

If unit tests pass, run the existing overflow regression suite to confirm no upward-path regression:

```bash
/home/nico.unglert/miniconda3/envs/jaxrens/bin/python -m pytest tests/ -k "overflow or max_neighbor" -v 2>&1 | tee /tmp/pytest_overflow_regression.log
```

## Effort estimate

- Core logic in `run_loop.py`: ~30 lines.
- Config + schema + resolve/run plumbing: ~30 lines mechanical.
- Tests: ~150 lines.
- One sitting (~2-3 hours) including running the suite.

## Notes / non-goals

- **Not changing** `NSState` — dwell counter is Python-side.
- **Not changing** the JIT compilation strategy — relies on JAX's existing per-signature cache.
- **No checkpoint persistence** of `low_count` — restart resets to 0. Conservative, acceptable.
- **No symmetric multi-step shrink** — one ladder step per dwell window. Avoids over-correction.
- **Default off** (`shrink_dwell = 0`) — opt-in only. Existing configs and benchmarks unchanged.
