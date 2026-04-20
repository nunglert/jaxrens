# Pathology: cell moves at 0% acceptance

## Symptom

Volume, shear, and/or stretch moves show `acc=0.00` in the periodic log. Step sizes shrink rapidly through successive adaptation rounds (1e-3 → 1e-7 → 1e-12 within a few iterations). Adaptation never recovers.

## Canonical diagnosis order

1. **Check the reject breakdown.** If present in the log (`reject: E=X% C=Y% P=Z%`), this tells you immediately:
   - E=100% → walker population already in a bad state (energies at ceiling)
   - C-dominated → cell constraints firing (V at V_max, aspect < min_aspect)
   - P-dominated → V^N prior rejecting volume decreases (usually not the bottleneck at small ss)

2. **If no reject breakdown in log, verify the kernel in isolation.** Template at `template_kernel_diagnostic.md`. On a clean walker (grid positions + sample_initial_volume cell + emax=E_init+100), all cell kernels should show 100% acceptance at ss=1e-5. If they don't, the kernel itself is broken.

3. **If kernel is fine but real run fails, walker population is in a bad state.** Usually caused by:
   - Excessive galilean step_size_max (adapts to ≥1 for 64-atom LJ, drives walkers into high-E regions)
   - Initial positions produce high-E configurations (grid too coarse, or uniform random overlap)
   - Initial V is at V_max via V^N prior sampling

## Known root causes (chronological)

### 2026-04-18: galilean step_size_max=5 driving walkers into unescapable regions

Config: LJ-64, n_live=512, n_mcmc_steps=20, galilean+volume+shear+stretch, `adaptation.defaults.step_size_max=5.0`.

Trace: galilean acc=0.90 at ss=5 in trial (adjust_step_size's 1-move test), but ns_step does 20 chain moves per walker. Each accepted move drifts walker positions by up to 5 Å; after 20 steps walker is in a chaotic/overlapping configuration. Cell-move trials on those walkers show 100% energy-rejects — new cell perturbation raises energy further past emax. Step sizes collapse because adaptation can't find a working value.

Fix: set per-move `step_size_max` — galilean ≤ 0.5, cell moves ≤ 0.2:
```yaml
adaptation:
  defaults: {step_size_max: 0.2}
  per_move:
    galilean: {step_size_max: 0.5}
```

### 2026-04-18: TF32 matmul floor on cell moves (GPU only)

When cell moves (`positions @ transform`) run on CUDA under JAX's default matmul
precision, TF32 (10-bit mantissa) introduces position perturbations of order
`|positions| * 2^-11` ≈ 4e-3 Å even for `T = jnp.eye(3)`. The noise is
step-size-INDEPENDENT: below ss ~ 1e-3 the intended motion is overwhelmed by
TF32 noise. On compressed LJ walkers (V/atom < 4) this produces dE of +50..+4000,
100% E-reject even at ss=1e-20. Adapter bisects ss down forever → floors at 1e-20.

Signature: `acc=0.00` with `reject: E=100% C=0% P=0%`, ss < 1e-4, galilean
unaffected, GPU backend. Full diagnosis and fix options in
`pathology_tf32_cell_move_floor.md`.

### 2026-04-18: Initial V at V_max blocks volume-increase moves

Config: `max_volume_per_atom: 4.0`, `min_volume_per_atom: 0.5`, `flat_V_prior: false`.

`sample_initial_volume` with V^N prior peaks at V = n_atoms * V_max * (1 - 1/(N+1)) ≈ 0.985 * V_max. Any volume-increase proposal fails `check_cell_shape` → C=50% rejects.

Fix: loosen `max_volume_per_atom` (20-100 for LJ-NS is typical) AND use `flat_V_prior: true` if the system's equilibrium volume is far below the max.

## Trial-vs-chain acceptance gap (general)

`adjust_step_size` returns the 1-move acceptance rate from vmap'd trials. `ns_step` runs `n_mcmc_steps=20` consecutive moves per chain. The chain-level rate is NEVER better than 1-step and can be dramatically worse when step sizes are large. Rule of thumb: if 1-step acceptance > 0.7, 20-step chain has <20% chance all moves accept. Most NS runs want per-step rate in [0.3, 0.5] for healthy chain decorrelation.

Set `max_rate: 0.5` in AdaptationPolicy to push adaptation toward lower rates. The default `max_rate: 0.65` is too permissive.

## Signs to watch for in future diagnoses

- Step size drops to float32-subnormal range (~1e-38) → adaptation is dividing by adjust_factor indefinitely without convergence
- All three cell moves (volume + shear + stretch) fail together → population issue, not move-specific
- Only volume fails but shear/stretch work → V^N prior or V_max boundary issue
- Only shear/stretch fail but volume works → aspect-ratio constraint
