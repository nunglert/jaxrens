# Pathology: NS terminates at iteration ≈ n_live + small constant

## Symptom

Run finishes immediately after `n_live` iterations — e.g. `iter=258` with `n_live=256`, or `iter=514` with `n_live=512`. Pattern `iteration ≈ n_live + 2` is the giveaway.

## Root cause (2026-04-18)

`PriorMassTermination.check` in `sampling/termination.py` had two bugs:

1. Missing the `log_L_max_remaining` factor. Canonical Skilling check is:
   ```
   log(X_i * L_max_remaining) < log(Z * threshold)
   log(X_i) + (-emax) < log(Z) + log(threshold)
   ```
   The original code compared `log(X_i)` (prior mass alone) to `log(Z) - threshold`, ignoring the likelihood envelope.

2. `threshold` treated as log-space instead of linear. `log(Z) - self.threshold` with `threshold=0.001` meant "remaining X < Z / exp(0.001)" which fires trivially as soon as `log(Z) > 0`. Should be `log(Z) + log(threshold)` → "remaining contribution < threshold × Z" in linear space.

Fixed. If you see this pattern recur, the fix was reverted or a different termination criterion has the same bug.

## Verification procedure

1. Add a one-liner to a scratch script:
   ```python
   from jaxrens.sampling.termination import PriorMassTermination
   p = PriorMassTermination(n_live=256, threshold=0.001)
   p.update_evidence(50.0)  # realistic log_Z
   # should NOT fire at iter ~ n_live with typical emax ~ -80
   print(p.check(iteration=258, emax=-80.0))
   ```
   If `True`, termination logic is broken again.

2. Confirm user's `termination` block isn't setting a different criterion that mimics this. E.g., `type: iteration, max_iterations: 258` would also produce the same pattern but is clearly user-specified.

## Adjacent issue: `run.max_iterations` as silent hard cap

Not this pathology, but often confused with it. `run_ns`'s Python for-loop is `for i in range(max_iterations)`. If user sets `max_iterations=1000` in `run` but wants 100k iterations, the loop exits silently at 1000 with message "NS complete: 1000 dead points" (no "Terminated by criterion" line). If the termination pattern is EXACTLY `iteration == run.max_iterations`, it's the loop cap, not a criterion bug.
