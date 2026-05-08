# Moves and the Metropolis-within-Gibbs scheduler

jaxrens decouples *what* configurations to propose from *how* they
combine in a single NS iteration. Individual MCMC kernels — random
walk, galilean, HMC, volume, shear, stretch, single-atom swap,
alchemical — each implement the `MoveKernel` protocol. They're
composed at run time into a single step function by
{func}`~jaxrens.sampling.mwg.build_mwg`, which implements
Metropolis-within-Gibbs (MWG): at each MCMC step, one kernel is
sampled and invoked.

## The MWG step

Given a list of $M$ kernels with weights $\{w_k\}_{k=1}^{M}$, each
MCMC step picks kernel $k$ with probability

$$
p_k = \frac{w_k}{\sum_{j=1}^{M} w_j},
$$

runs one step of that kernel, and records which one ran in
`MoveInfo.move_idx`. Over `n_mcmc_steps` calls per walker, each
kernel fires roughly $p_k \cdot n_\mathrm{mcmc}$ times. YAML:

```yaml
moves:
  - type: galilean
    step_size: 0.1
    weight: 4.0      # 4/7 of steps
  - type: volume
    step_size: 0.3
    weight: 1.0      # 1/7
  - type: shear
    step_size: 0.1
    weight: 1.0
  - type: stretch
    step_size: 0.1
    weight: 1.0
```

```{mermaid}
flowchart LR
    W["walker state x_t"] --> S["sample move idx<br/>p_k = w_k / Σ w_j"]
    S --> R{"which kernel?"}
    R -->|galilean| G["galilean.step"]
    R -->|volume| V["volume.step"]
    R -->|shear| SH["shear.step"]
    R -->|stretch| ST["stretch.step"]
    G --> O["x_{t+1},<br/>MoveInfo(move_idx, accepted, ...)"]
    V --> O
    SH --> O
    ST --> O
```

## Acceptance at the NS constraint

NS enforces a hard likelihood threshold instead of a
Metropolis-style temperature criterion. A proposed move is
accepted iff

$$
H(y) < E_\mathrm{max},
$$

where $y$ is the proposed configuration and $E_\mathrm{max}$ is
the current likelihood constraint. Equivalently,

$$
\alpha(x \to y) = \mathbf{1}\bigl[H(y) < E_\mathrm{max}\bigr].
$$

Each kernel must satisfy detailed balance *on the constrained
prior* — propose symmetrically with respect to $\pi$, and the hard
acceptance automatically gives a valid transition. Most kernels
meet this by construction (random walk, reflection-based galilean,
HMC with momentum resampling); the volume move uses Jacobian
factors to compensate for its non-symmetric proposal.

## Per-move acceptance and adaptation

`ns_step` accumulates per-move acceptance / proposal counts and
reject-reason counts across the scan:

- `n_accepted_per_move[k]`, `n_proposed_per_move[k]`
- `reject_reason_counts_per_move[k, 0..3]` — `[accepted, energy,
  cell, prior]`

The `AdaptationManager` reads these counters and runs a bisection
loop over each kernel's `step_size` so its long-run acceptance
stays in `[min_rate, max_rate]` — typically `[0.3, 0.5]`. Step
sizes are clamped to `[floor=1e-20, step_size_max]`.

```{image} /_static/figures/mwg_acceptance.png
:alt: per-move acceptance under adaptation
:align: center
```

The green band shows the acceptance window; rates outside it
trigger a bisection round.

### How `AdaptationManager.apply` dispatches the bisection

Same construction-time-cache pattern as the
{class}`~jaxrens.sampling.inter_re_manager.InterREManager`: the
manager is built once before the loop and stores one
`jax.jit`-compiled `adjust_step_size` callable per move type;
`_run_loop` only calls `fires(i)` and (on a fire) `apply(...)`.
Inside `apply`, the work iterates over moves in Python and
dispatches each to a JIT'd `adjust_step_size` whose `lax.while_loop`
body runs the bisection.

```{mermaid}
flowchart TB
    LOOP["NS outer loop<br/>iteration i"]
    LOOP --> FIRES{"manager.fires(i)<br/>i &gt; 0  ∧  i mod adjust_interval == 0"}
    FIRES -- no --> CONT["continue NS step"]
    FIRES -- yes --> APPLY["manager.apply(pop, emax, key, step_sizes)"]

    APPLY --> LOOPK{"for each move k<br/>in move_descriptors"}
    LOOPK --> DESC{"BatchDescriptor"}
    DESC -- SingleRun --> SR["adjust_step_size(...)<br/>scalar ss"]
    DESC -- VmapRuns --> VR["jax.vmap(adjust_step_size)<br/>(R, ...)"]
    DESC -- PmapVmapRuns --> PR["jax.pmap(jax.vmap(adjust_step_size))<br/>(G, P, ...)"]

    subgraph jit_box["adjust_step_size — JIT + lax.while_loop"]
        direction TB
        SAMPLE["sample n_samples walkers<br/>jax.random.choice"]
        TRIAL["run trial moves<br/>(vmap or chunked lax.map)"]
        RATE["rate = mean(accepted)<br/>+ reject-reason counts"]
        PROC{"_process_rate_jax<br/>rate ∈ [min_rate, max_rate]?"}
        UP["new_ss = ss × adjust_factor<br/>(rate too low — too small steps)<br/>clamp to step_size_max"]
        DN["new_ss = ss / adjust_factor<br/>(rate too high — too big steps)<br/>clamp to floor 1e-20"]
        CONV(["converged ✓<br/>or bracket detected"])
        NEXT{"round &lt; max_rounds<br/>and not converged?"}

        SAMPLE --> TRIAL --> RATE --> PROC
        PROC -- yes --> CONV
        PROC -- "no, too low" --> UP --> NEXT
        PROC -- "no, too high" --> DN --> NEXT
        NEXT -- yes --> SAMPLE
        NEXT -- no --> CONV
    end

    SR --> SAMPLE
    VR --> SAMPLE
    PR --> SAMPLE

    CONV --> WRITE["update step_sizes[k] ← new_ss<br/>+ collect 9 diagnostics<br/>(rate, n_rounds, cap_hits, …)"]
    WRITE --> NEXTK{"more moves?"}
    NEXTK -- yes --> LOOPK
    NEXTK -- no --> RET["return (new_step_sizes,<br/>per_move_outputs, key)"]

    classDef decision fill:#fff7e0,stroke:#a07000,color:#222
    classDef path fill:#eef5ff,stroke:#1565c0,color:#222
    classDef jitBox fill:#fff7e0,stroke:#a07000,color:#5a3a00
    classDef io fill:#f5f5f5,stroke:#444,color:#222

    class FIRES,LOOPK,DESC,PROC,NEXT,NEXTK decision
    class SR,VR,PR,SAMPLE,TRIAL,RATE,UP,DN path
    class jit_box jitBox
    class LOOP,APPLY,CONT,WRITE,RET,CONV io
```

A few invariants:

- **Cached compilation.** `_build_jit_fns` runs once in
  `__init__` and stores one `jax.jit`-compiled `adjust_step_size`
  per move type. Each subsequent `apply` hits the cache; no
  per-iteration retracing.
- **Move-by-move Python loop, JIT'd inner.** `apply` iterates over
  moves in plain Python so `step_sizes[k]` can be updated
  one entry at a time and per-move diagnostics can be logged at
  `DEBUG` level. The hot work — the bisection — happens inside the
  JIT'd callable.
- **`lax.while_loop` bisection.** Each round samples
  `adjust_n_samples` walkers, runs trial moves through the same
  move kernel that NS uses (so acceptance is measured against the
  same `Emax`), and updates `ss` by `adjust_factor` based on
  whether the rate is below `min_rate` or above `max_rate`.
  Convergence flags are set once the rate enters
  `[min_rate, max_rate]` *or* a proper bracket forms (one round
  too-low, one too-high) — the latter prevents oscillation when
  the optimum lies between the discrete `factor`-spaced rungs.
- **Optional chunked-vmap.** `trial_batch_size` toggles
  `jax.vmap` vs `jax.lax.map` for the trial-move evaluation; the
  latter caps peak memory at `trial_batch_size × per-trial tape`,
  matters for HMC at large `n_samples`.

## Adding a new kernel

A move kernel is anything that:

1. Declares what extra per-walker state it needs
   (`extra_state_fields`).
2. Provides a `build_kernel(backend, **kernel_kwargs)` factory that
   returns a pure function
   `step(rng_key, state, likelihood_constraint) → (new_state, MoveInfo)`.
3. Optionally implements `init()` for any move-specific state
   setup.

A minimal example:

```python
from jaxrens.sampling.move_kernel import MoveKernel

def build_my_move(backend):
    def step(rng_key, state, emax):
        # propose state', compute new energy, accept iff H < emax
        ...
        return new_state, MoveInfo(...)
    return step

descriptor = MoveKernel(
    name="my_move",
    build_kernel=build_my_move,
    step_size=0.1,
    weight=1.0,
)
```

Then plug it into `build_mwg(backend, [descriptor, ...])` like any
built-in kernel.

## Where this lives in the code

| Concern | File |
|---|---|
| Kernel protocol / descriptor | {class}`jaxrens.sampling.move_kernel.MoveKernel` |
| MWG composer | {func}`jaxrens.sampling.mwg.build_mwg` |
| Individual kernels | `sampling/moves/{random_walk,galilean,hmc,single_atom,alchemical,volume,shear,stretch,replica_exchange}.py` |
| Info payload | `jaxrens.base.MoveInfo` |
| Adaptation | `sampling/adaptation/{stepsize_handler,manager}.py` |
