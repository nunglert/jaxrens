# Pathology: step sizes explode to 1e6+

## Symptom

Log reports `ss=1.278e+07` (or similar absurd value) for one or more moves. All move proposals reject because every move sends walkers into chaotic configurations.

## Canonical diagnosis

1. **Is adaptation in the `full_auto` branch?** `use_full_auto` in `run_ns` requires:
   - `adaptation_config.full_auto: true` (user config)
   - `adaptation_config.full_auto_steps > 0` (default 50 since 2026-04-18; was 0 before)
   - `per_move_fns is not None` (set when `move_descriptors` passed to `run_ns`)

   If any condition fails, runs through `dual_averaging_update` fallback — **which has no clamp on log_step_size**. With `gamma=0.05`, `log_step_size = mu - sqrt(m)/gamma * h_avg` can grow unboundedly.

2. **Is `desc.step_size_max` being respected?** Grep for `jnp.clip(step_size * scale, 1e-20, max_step_size)` in `_process_rate_jax` — if hit, step size caps. Check the value passed: `desc.step_size_max` should equal the user's `adaptation.defaults.step_size_max` (or `adaptation.per_move[name].step_size_max`). The resolver injects this via `dataclasses.replace(descriptor, step_size_max=policy.step_size_max)`; if that seam is broken, defaults (10.0) apply.

## Known root causes

### 2026-04-18: full_auto_steps defaulting to 0 disabled full_auto

Fixed by changing default to 50 in `adaptation.py`. If you see this again after the fix, the user likely set `full_auto_steps: 0` explicitly.

### 2026-04-18: resolver not injecting adaptation_policy into MoveKernel

Fixed by `dataclasses.replace(descriptor, min_rate=..., max_rate=..., step_size_max=...)` in `_resolve_config`. If `desc.step_size_max` ever reads 10.0 (the MoveKernel default) when the user explicitly set something else, the injection is broken again.

## Dual-averaging path (fallback) remains unclamped

When user intentionally disables full_auto (rare), the fallback dual_averaging path can still explode. If you confirm this, suggest either:
- Enabling full_auto, OR
- Adding a defensive clamp in `dual_averaging_update`: `log_step_size = jnp.clip(log_step_size, -40, 10)` (cap at exp(10) ≈ 2e4).

Don't apply the clamp unless asked — it's a defensive fix, not a root-cause fix.

## Arithmetic fingerprints (for future forensics)

When confirming that a cap was exceeded, compute backwards:
- `step_size_max=5.0` starting from ss=0.1 with adjust_factor=1.5, always-scale-up:
  reaches 5.0 after 10 scale-ups (0.1 * 1.5^10 ≈ 5.77, clip → 5.0).
- `step_size_max=10.0`: reaches 10.0 after 11 scale-ups (0.1 * 1.5^11 ≈ 8.65 → next
  round 8.65 * 1.5 ≈ 13, clip → 10.0).
- At max_rounds=15 and rate=0 (always-scale-down), cell ss shrinks by 1.5^15 ≈ 437.9:
  e.g. start 0.1 → 2.28e-4, start 0.2 → 4.57e-4.

If the h5 trace shows ss=EXACTLY some round number like 5.0, 10.0, 0.5 — that's
the cap, and it's trivial to read off which value was in effect.

## Config vs run-time mismatch

When diagnosing from a log file: check config mtime vs log mtime. If config is
newer, the user may have edited the config AFTER the bad run. In that case the
current config values DO NOT describe what was actually run. Look at the h5
trace's numerics to infer the cap that was actually in effect.
