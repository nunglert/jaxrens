---
name: TF32 matmul floor on cell-move step sizes
description: On CUDA GPU with default precision (TF32), `positions @ T` introduces ~4e-3 Å noise even for T=I, making cell moves at small ss pure noise, 100% E-reject, ss collapses to 1e-20 floor.
type: project
---

# Pathology: TF32 matmul floor on cell moves

## Symptom (2026-04-18)

- Cell moves (volume/shear/stretch) show `acc=0.00` with `reject: E=100% C=0% P=0%`
  throughout an NS run.
- Step sizes adapt down from ~0.1 to 1e-5 within ~100 iterations, then continue
  collapsing to 1e-20 (the `_process_rate_jax` floor) and stay there.
- Galilean move is unaffected — its chain-level acc reads 97-100% at capped ss=0.5.
- Emax descends normally (galilean is doing useful work), but the cell moves
  contribute nothing — NPT exploration is broken.
- Run is on NVIDIA GPU (`jax.default_backend() == "gpu"`).

## Root cause

JAX on CUDA uses `jax.lax.Precision.DEFAULT` for matmuls, which maps to **TF32**
— 10-bit mantissa for multiplication. The three cell-move kernels use matrix
multiplies to apply the cell transform to positions:

- `volume.py:55`  → `new_positions = state.positions @ transform`  (transform = I*scale)
- `stretch.py:54` → `new_positions = state.positions @ transform`  (transform = diag(exp(±rv)))
- `shear.py:82`   → `new_positions = transform_positions(state.positions, state.cell, new_cell)`
                     which internally does `positions @ solve(cell, new_cell)`

At ss=1e-20, `transform` is numerically the identity matrix, but `positions @ I`
under TF32 produces position perturbations of up to **~4e-3 Å on a cell of size
~11 Å** (tested: n_atoms=64 LJ, max_|Δpos|=3.696e-3 while `jnp.all(T == I)`
returns True).

The TF32-induced position shift is INDEPENDENT of `step_size`. When the intended
move signal `ss * proposal` falls below the TF32 noise floor (~4e-3 Å), the move
degenerates into pure TF32 noise applied to the configuration. On a compressed
LJ solid (V/atom ~ 2-4, common late in NS), that noise produces random energy
changes of order +50..+4000 units. Once galilean has pushed walker E up against
Emax (which it does reliably with acc=1.0), any positive TF32-noise dE → energy
reject. Empirical: 100% cell-move E-reject.

Adapter's response: bisect ss downward → useless because noise is ss-independent.
Floors at 1e-20.

## Detection heuristic

If you see ALL of these together:
1. Cell-move `acc=0.00` with `reject: E=100% C=0% P=0%` for many iterations
2. Cell step sizes < 1e-4 (well below any physical scale)
3. Galilean acc high (>0.9) — proves walker population is healthy
4. Backend is on GPU (`jax.default_backend() == "gpu"`)
5. Config uses `float32` positions (the jaxrens default)

→ Immediately suspect TF32 matmul. Confirm with:
```python
positions = <realistic walker positions, n_atoms ~ 64, cell ~ 10 Å>
I = jnp.eye(3, dtype=positions.dtype)
diff = jnp.max(jnp.abs(positions @ I - positions))
print(diff)   # if > 1e-5 on GPU default precision → TF32 is active
```
If it prints ~4e-3 the bug is present; if 0.0 the system is not TF32-affected.

This is NOT a CPU-reproducible bug — on CPU, default matmul precision is already
highest-equivalent. The same NS run on CPU would NOT show this pathology.

## Canonical diagnosis

1. Read the log — confirm `reject: E=100% C=0% P=0%` and ss collapsing to 1e-20.
2. Run `jax.default_backend()`; if "gpu", proceed.
3. Isolated reproducer (<30 lines): build grid positions in a realistic cell,
   compute `positions @ jnp.eye(3)`, compare. If nonzero → TF32 confirmed.
4. If TF32 is confirmed, the code paths to inspect are:
   - `volume.py` line 55: `state.positions @ transform`
   - `stretch.py` line 54: `state.positions @ transform`
   - `shear.py` line 82: via `transform_positions` in `utils/cell.py`
   - `utils/cell.py:84`: `positions @ T`

## Fix recipe (DO NOT apply from debugger; escalate to implementer)

**Option A (simple, coarse)**: At jaxrens entry point:
```python
jax.config.update("jax_default_matmul_precision", "highest")
```
Forces bit-exact matmul globally. Cost: LJ/MACE backend matmuls become
~2-3x slower on GPU. For pure-LJ small-N runs this is acceptable.

**Option B (scoped, recommended)**: Use `precision="highest"` only where it
matters. JAX exposes this via `jnp.matmul(..., precision=...)` and
`jnp.einsum(..., precision=...)`. The `@` operator does NOT take a precision
kwarg. Rewrite each `positions @ T` as:
```python
new_positions = jnp.einsum("ij,jk->ik", state.positions, transform,
                           precision=jax.lax.Precision.HIGHEST)
```

Touch points:
- `jaxrens/src/jaxrens/sampling/moves/volume.py` line 55
- `jaxrens/src/jaxrens/sampling/moves/stretch.py` line 54
- `jaxrens/src/jaxrens/utils/cell.py` line 84 (inside `transform_positions`)
- (shear inherits from transform_positions; no direct change needed)

**Option C (not recommended)**: Branch on "transform is near identity" and skip.
Doesn't address the intermediate-ss regime and adds JIT branching.

## Why this wasn't caught

- Unit tests likely run on CPU (where default ≡ highest). The TF32 path is only
  active on GPU.
- Shape/identity tests pass because `jnp.all(T == I)` is True; the matmul just
  happens to re-quantize the float32 result.
- JIT caching won't reveal it — the bug is in the numerics, not the trace.

## Related

Prior round 2 finding in this pathology file blamed galilean ss_max=5 as the
cause (chain-pushes-walker-into-pathological-config). That's a REAL pathology
but distinct from this one — galilean-driven pathological configs would show
up as C-rejections or mixed E/C rejections, not 100% E. 100% E with ss<1e-3 is
the TF32 signature.
