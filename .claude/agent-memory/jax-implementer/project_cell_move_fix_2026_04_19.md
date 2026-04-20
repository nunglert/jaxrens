---
name: Cell move 100% rejection root cause and fixes
description: Four fixes applied 2026-04-19 to resolve 100% energy rejection in LJ-64 NPT cell moves
type: project
---

Root cause of cell-move 100% rejection (LJ-64 NPT, `lj8_npt` experiment) found and fixed 2026-04-19.
Four distinct bugs were present; all required to get non-zero acceptance.

**Fix 1 (TF32 precision) — was pre-applied:**
`volume.py`, `stretch.py`, `cell.py` use `jnp.einsum(..., precision=jax.lax.Precision.HIGHEST)` for
`positions @ T` matmuls. On GPU, default TF32 introduces ~3.7e-3 noise even for identity T, spiking
LJ energy at dense packing.

**Fix 2 (reject_reason gating) — `nested_sampling.py` scan body:**
Non-cell moves leave `MoveInfo.reject_reason=0` (default = accepted bucket). When rejected, they were
counted in bucket 0 (accepted), inflating reported acceptance. Fix: `scatter_col = jnp.where(accepted,
0, jnp.maximum(reject_reason, 1))`. Preserves `rr[:, 0] == n_accepted_per_move`.

**Fix 3 (step_sizes injection in adaptation) — `stepsize_handler.py`:**
`body_fn` injected trial ss into `state.step_size` (scalar) only, but MWG wrapper reads
`state.step_sizes[move_idx]`. Bisection was measuring rate at original population step_sizes always,
driving ss to the 1e-20 floor. Fix: also set `step_sizes = jnp.broadcast_to(ss, (n_samples, n_moves))`.

**Fix 4 (initial energy scale mismatch — primary root cause) — `resolve.py`:**
`_resolve_init` received the bare LJ backend (`energy_backend = root.backend.build_backend()`) for
computing initial walker energies. The NS loop uses `EnsembleBackend` (+P*V for NPT). All walkers had
bare LJ energies stored; cell moves compute NPT energies; systematic `new_energy > emax` caused 100%
rejection. Confirmed by constant offset = P*V_current = 127.3755 across all walkers in checkpoint.
Fix: wrap in `resolve.py` before `_resolve_init` call when pressure is set:
  `energy_backend = EnsembleBackend(energy_backend, pressure=float(pressure))`

**Why:** Ensemble energy scale must match between init and NS loop. Any move computing ensemble energy
will always reject when emax is on the bare potential scale.

**How to apply:** When debugging NPT cell-move rejection, check whether stored energies match
recomputed energies: constant offset = P*V is the signature of this bug. Also look for this bug in
any other ensemble correction scenarios (muPT, etc.).

**Result:** LJ-8 NPT run at 3000 iters with acc: volume=0.61-0.67, shear=0.30-0.62, stretch=0.42-0.65.
