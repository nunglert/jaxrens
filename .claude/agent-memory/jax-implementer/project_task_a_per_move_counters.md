---
name: Task A per-move chain-level counters
description: Chain-level per-move acceptance/reject counters landed in ns_step info (2026-04-18)
type: project
---

Task A of the three-part refactor landed 2026-04-18.

**What changed:**
- `MoveInfo` in `base.py` gained `move_idx: jnp.ndarray = jnp.int32(0)` (5th field, default 0).
- MWG `step_fn` in `mwg.py` injects `move_idx` via `info._replace(move_idx=...)` after `lax.switch`.
- `ns_step` scan body in `nested_sampling.py` accumulates three per-move arrays per MCMC step; sums are aggregated across the walker vmap axis.
- `ns_step` `info` dict now always contains: `n_accepted_per_move (n_moves,) int32`, `n_proposed_per_move (n_moves,) int32`, `reject_reason_counts_per_move (n_moves, 4) int32`.
- `ProgressCallback` in `monitor.py` computes `acc = n_accepted / max(n_proposed, 1)` from new keys; falls back to trial-phase `acceptance_rates_per_move` if absent.

**Bucket convention (load-bearing):**
`reject_reason_counts_per_move[:, 0]` = accepted count; `[:, 1]` = energy reject; `[:, 2]` = cell reject; `[:, 3]` = prior reject. Rows sum to `n_proposed_per_move`. This means `rr[:, 0] == n_accepted_per_move` is an invariant enforced by tests.

**Old caches kept:** `_last_acc_rates_per_move` / `_last_reject_counts_per_move` in `run_ns` are redundant but left for task C cleanup.

**Why:** Trial-phase `acc` column in monitor was from `adjust_step_size` (50 walkers × 1 move, only on adjust intervals) — a different metric from chain-level.

**How to apply:** When looking at monitor log, `acc=X` is now chain-level. If debugging why a move has 0% acceptance, check `reject_reason_counts_per_move` from `ns_step` info for the per-reason breakdown every iteration, not just at adjust intervals.
