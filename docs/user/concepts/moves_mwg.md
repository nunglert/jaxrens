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
