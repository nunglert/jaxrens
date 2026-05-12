---
name: NeuralILwithMorse NaN gradient on padded atoms
description: Step-1 loss is NaN in NeuralILwithMorse training because center_at_atoms emits inf for padded radii, MorseModel's exp(-a*(inf-b)) produces 0*inf=NaN in the gradient even though the forward value is correctly zeroed.
type: project
---

## Symptom

- `NeuralILwithMorse` training script produces finite loss at step 0 (e.g. 4.6
  on first batch), then NaN from step 1 onward.
- Forward pass at step-0 params is finite. Gradient at step-0 is NaN ONLY on
  `params.morse.embed.embedding` and `params.morse.switch_param`. The rest of
  the tree (core, embed, denormalizer) is clean at step 0.
- After one optimizer step those NaN morse params poison the next forward
  pass; step-1 grad has NaN on every leaf.
- Reproduces on ANY batch — does not require the user's "real atom at (0,0,0)
  coincident with padded atom" suspicion. (That red-herring hypothesis is
  separately sound but is not what's firing.)

## Root cause

`bessel_descriptors.center_at_atoms` (and `center_at_point` / `center_at_points`)
sets the radius to `jnp.inf` whenever either side of the pair is a padded
atom (type < 0):

```python
radius = jnp.where(types < 0, jnp.inf, _sqrt(jnp.sum(delta**2, axis=-1)))
```

`MorseModel.calc_atomic_energies` (neuralil/model.py:153) attempts to make
those padded pairs harmless by shifting them out of cutoff:

```python
radii = radii + 2.0 * mask * self.r_cut
```

But `inf + 2*4.5 = inf`. Downstream:

```python
exponential = jnp.exp(-a * (r_morse - b))   # exp(-inf) = 0 forward
contributions = d * exponential * (exponential - 2.0)  # 0
contributions *= jnp.logical_not(jnp.isclose(0.0, radii))  # 0
cutoffs = smooth_cutoff(radii, r_switch, r_cut)  # 0 at inf
result = (cutoffs * contributions).sum(...)  # 0 forward — correct
```

Forward is OK. But the **gradient** w.r.t. `a` is
`d/da exp(-a(inf-b)) = -(inf-b) * exp(...) = -inf * 0 = NaN`.
The cutoff multiplier outside doesn't save you — `0 * NaN = NaN`. Same path
through `r_switch`, hence `switch_param`'s grad is NaN.

## Canonical diagnosis

1. Run forward + `value_and_grad` on a tiny 1-2-frame batch with
   `NeuralILwithMorse`.
2. Use `jax.tree_util.tree_leaves_with_path(grad)` + per-leaf
   `jnp.isnan(g).sum()`. If only `morse.embed.embedding` and
   `morse.switch_param` are NaN at step 0, this pathology is firing.
3. Confirm with bisection: replace `NeuralILwithMorse` with plain `NeuralIL`.
   Plain version stays clean -> Morse is the culprit.
4. Sanity check minimal repro:
   ```python
   morse = MorseModel(n_types=1, r_cut=4.5)
   # 'all-real' radii -> finite grad
   # radii with one jnp.inf entry -> NaN on embed.embedding
   # radii with 100.0 (out of cutoff but finite) -> finite grad
   ```

## Fix recipes (training-script-level, do NOT edit neuralil source)

Preferred order:

1. **Pad atom positions to a "safe" finite-distance location.**
   - Shift padded coordinates well outside the cutoff but within the cell,
     e.g. place them on a diagonal offset that PBC-wraps to >2*r_cut from any
     real atom — actually simpler: leave positions at (0,0,0) but **change
     center_at_atoms behaviour**: not possible if read-only on neuralil.
   - Alternative: pad positions to a fixed location that, after PBC mapping,
     yields radii inside the cutoff but with type<0. That doesn't help —
     `center_at_atoms` still emits `inf` because of the type<0 check.
   - **THE ROOT-CAUSE FIX REQUIRES EDITING `center_at_atoms` (or
     `MorseModel.calc_atomic_energies`).** No training-script monkey-patch
     can avoid the inf without changing the descriptor or Morse logic.

2. **`optax.zero_nans()` in the chain BEFORE clip_by_global_norm.**
   Pragmatic: masks the bad Morse gradient (which is genuinely zero in
   information content — the forward value didn't depend on those params via
   the padded pairs). This is the cheapest fix; arguably correct because the
   NaN grad is a `0 * inf` artifact, not a real signal.
   ```python
   optimizer = optax.chain(
       optax.zero_nans(),
       optax.clip_by_global_norm(10.0),
       create_one_cycle_minimizer(...),
   )
   ```
   Caveat: this also masks any future genuine NaN gradient. Pair with a
   periodic "did params explode?" check.

3. **Switch to plain `NeuralIL`** (drop the Morse short-range term). Verified
   clean in repro. Loses the physically-motivated repulsive prior.

4. **Monkey-patch `MorseModel.calc_atomic_energies`** in the training script
   to replace inf radii with `2*r_cut` before computing `r_morse`:
   ```python
   from neuralil import model as _m
   _orig = _m.MorseModel.calc_atomic_energies
   def _safe(self, radii, probe_types, source_types):
       radii = jnp.where(jnp.isinf(radii), 2.0 * self.r_cut, radii)
       return _orig(self, radii, probe_types, source_types)
   _m.MorseModel.calc_atomic_energies = _safe
   ```
   This is editing module-level code at import time — borderline "editing
   production" but technically training-script-local. The most surgical
   correctness-preserving option.

## Detection heuristic

If `NeuralILwithMorse` training shows:
- finite loss at exactly step 0
- NaN from step 1 onward
- grad-tree inspection shows step-0 NaNs **only** on `morse.embed.embedding`
  and `morse.switch_param`

You are seeing this bug. Bisect with `NeuralIL` (no Morse) — if that's clean,
diagnosis is locked.

## What is NOT the cause (ruled out in this debug)

- The 40 frames with a real Si at exactly `(0,0,0)` are a red herring. The
  bug fires on any batch.
- The `_sqrt(0)` JVP at coincident atoms is safe (custom JVP returns 0 at
  x=0).
- The `nan_to_num` in `EllChannel._function` is not implicated — the failing
  grad is on Morse params, not on the descriptor params.
- The user's `delta_sq + 1e-12` epsilon in the forces loss is genuinely
  helpful for the `sqrt(0)` case but does not address this Morse bug.
