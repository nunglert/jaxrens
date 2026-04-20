---
name: initial_step_size plumbing uses first move's step_size for all moves
description: run.py passes initial_step_size=first_mc.step_size to run_ns, which jnp.fulls it across all n_moves. Per-move YAML step_size values (other than the first) are ignored at init.
type: project
---

`cli/run.py:370` calls `run_ns(..., initial_step_size=first_mc.step_size, ...)`.
Inside `run_ns` (`nested_sampling.py:446`):

    step_sizes=jnp.full(n_moves, initial_step_size)

So all moves start at `first_mc.step_size` (whatever the FIRST move in YAML has).
The per-move YAML `step_size` fields for moves 2..N are silently discarded at
init. The adapter then takes over, so in practice this is benign — adaptation
overwrites within a few rounds — but it causes "initial step size reported in
log is 0.1 for volume even though YAML says 0.3" confusion.

**Why:** flagged during LJ-64 NPT debug round 2 (2026-04-18): h5 trace showed
volume moves shrinking from 0.1, not from 0.3 as the config specified.
**How to apply:** when a user asks "why did volume start at 0.1 when I set 0.3"
— this is why. Not a bug; minor plumbing loss.
