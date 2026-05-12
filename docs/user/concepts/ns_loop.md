# The nested-sampling loop

Nested sampling (Skilling, 2006) converts a multi-dimensional integral
over configuration space into a one-dimensional integral over *prior
mass*. jaxrens implements the standard live-walker scheme with one
wrinkle: the loop is split into a **Python outer loop** around a
**JIT-compiled inner scan**, and the boundary between them is
load-bearing.

## What NS computes

For a prior $\pi(r)$ and likelihood $L(r) = e^{-H(r)}$,
the evidence is

$$
Z = \int L(r)\,\pi(r)\,\mathrm{d}r
  = \int_0^1 L(X)\,\mathrm{d}X,
$$

where the transformation

$$
X(L^\star) = \int_{L(r) > L^\star} \pi(r)\,\mathrm{d}r
$$

collapses every equi-likelihood shell into a scalar *prior mass*
$X \in [0, 1]$. Sort the $X$'s by decreasing likelihood and the
integral becomes 1-D.

At iteration $i$, NS maintains $N$ live walkers all drawn from the
prior restricted to $L > L_i^\star$. The worst walker is retired —
it becomes a *dead point* with likelihood $L_i^\star$ — and replaced
by a new one drawn from the same constrained prior. In expectation,

$$
X_i \approx \exp(-i/N),
$$

and the trapezoid-rule approximation gives the evidence estimate

$$
Z \approx \sum_i w_i L_i^\star,
\qquad
w_i = \tfrac{1}{2}\bigl(X_{i-1} - X_{i+1}\bigr).
$$

In jaxrens, the constrained prior draw is produced by running MCMC
(one of the {doc}`move kernels <moves_mwg>`) starting from a
randomly chosen live walker, using the current likelihood threshold
as a hard constraint.

```{image} /_static/figures/ns_prior_mass.png
:alt: log_X and Emax vs iteration
:align: center
```

## Why two loops

Every NS iteration needs to do three things:

1. **Pick the worst live walker** (Python-side argmax; cheap).
2. **Run `n_mcmc_steps` of [MWG sampling](moves_mwg.md)** to produce a new walker
   under the updated likelihood constraint (dense JAX compute).
3. **Adapt step sizes** every `adjust_interval` iterations, retry
   on MLIP neighbor-buffer overflow, check termination, and dispatch
   callbacks (Python-side control flow).

Step 2 is shape-stable and fully differentiable — a perfect target
for `jax.jit` over a `lax.scan`. Steps 1 and 3 are not: step-size
adaptation calls a `while_loop` whose bounds depend on trial
acceptance rates; overflow retry uses runtime-evaluated neighbor
counts to pick a new bucket; callbacks write to disk. Forcing them
into JIT would either fail to compile (dynamic shapes) or require
one `jit` retrace per adaptation event.

jaxrens puts step 2 inside a JIT'd `ns_step` and leaves steps 1 and
3 in Python:

```{mermaid}
%%{init: {"layout": "elk"}}%%
flowchart TB
 subgraph inner["Inner loop — JIT"]
        S["ns_step<br>n_mcmc_steps × MWG<br>over all walkers"]
  end
 subgraph adapt["Stepsize adjustion — JIT"]
        AM["AdaptationManager.apply"]
  end
 subgraph jit_region["Core"]
    direction TB
        inner
        adapt
  end
  START["Start (i = 1)"]
  DOT@{ shape: f-circ }
  STOP["Done"]
 subgraph outer["Outer loop (Python)"]
    direction TB
        jit_region
        ITER(["Iteration i"])
        O{"Neighbor-buffer<br>overflow?"}
        ESC["escalate max_neighbors,<br>re-run inner loop"]
        C["Callback dispatch<br>(monitor / trajectory / checkpoint)"]
        T{"Termination<br>satisfied?"}
        EVERY{"adjust_interval<br>satisfied?"}
  end
    START --> DOT
    ITER --> EVERY
    EVERY -- yes --> AM
    EVERY -- no --> S
    AM --> S
    S --> O
    O -- yes --> ESC
    O -- no --> C
    C --> T
    T -- yes --> STOP
    ESC --> EVERY
    T -- no — i ← i+1 --> DOT
    DOT --> ITER

     inner:::jitBox
     adapt:::jitBox
     jit_region:::regionBox
     O:::decision
     ESC:::escBox
     T:::decision
    classDef jitBox fill:#fff7e0,stroke:#a07000,color:#222
    classDef regionBox fill:#fff3c4,stroke:#a07000,color:#5a3a00,stroke-dasharray:6 4
    classDef outerBox fill:#f5f5f5,stroke:#888,color:#222
    classDef decision fill:#eef5ff,stroke:#1565c0,color:#222
    classDef escBox fill:#ffe0e0,stroke:#c62828,color:#5a0000
    linkStyle 8 stroke:#888,stroke-dasharray:3 3,fill:none
    linkStyle 9 stroke:#000000,fill:none
    linkStyle 10 stroke:#000000,fill:none
```

The dashed **Core** region groups the two JIT-compiled pieces of
work as amber subgraphs around the actual function calls:
`ns_step` (the inner `lax.scan` over all walkers) and
`AdaptationManager.apply` (one `jax.jit`-compiled
`adjust_step_size` call per move kernel, dispatched through
{class}`~jaxrens.sampling.adaptation.manager.AdaptationManager`).
The blue diamond above Core — `adjust_interval satisfied?` — is
the modulus check that gates the adaptation; on a non-firing
iteration the path skips straight into `ns_step`, otherwise it
routes through `AdaptationManager.apply` first. The red
**escalate max_neighbors** node lives outside Core on the
*overflow* back-edge — when `ns_step` reports a neighbor-list
overflow, the outer loop bumps `max_neighbors` to the next bucket
and re-enters the same iteration's `adjust_interval` check (so
the run stays cached on the new `(max_neighbors, kernel)` pair —
each escalation triggers exactly one recompile and is then cached
for the rest of the run).

`Start (i = 1)` and `Done` sit outside the outer-loop box as the
overall entry / exit; the small black dot is a routing junction
that gives the cycle a clean rejoin point — both the initial
entry and the dashed grey "no — i ← i+1" loop-back from
`Termination` flow into it before continuing as `Iteration i`.
The dot is purely a layout aid (its `f-circ` shape is just so
ELK has a non-trivial node to anchor the back-edge on); without
it the cycle-breaker would route the long return arrow across
the diagram and disturb the forward flow. See {doc}`moves_mwg`
for what happens inside `AdaptationManager.apply`.

The handoff payload is the `info` dict emitted by `ns_step`:
`emax`, per-move acceptance / proposal counters, reject-reason
counts, neighbor-count max, overflow flag, and cumulative
evaluation counters.

## What happens inside the inner scan

`ns_step` takes a `(K, …)`-shaped population, picks the worst-energy
walker to cull, samples `n_extra` additional walkers for reseeding
MCMC chains, and runs `n_mcmc_steps` of the MWG step function on
each. The `lax.scan` carries the per-step walker state and
accumulates per-move acceptance counts.

Concretely, at iteration $i$ the worst walker's energy $E_i$
becomes the new *energy maximum* $E_\mathrm{max}$, and all MCMC
steps in the next iteration run under the constraint
$H(r) < E_\mathrm{max}$. Over many iterations, $E_\mathrm{max}$
decreases monotonically — nesting the likelihood shells — and the
walker distribution concentrates toward the ground state of $H$.

## Extending the scheme

- **n_cull > 1**: retire the `n_cull` worst walkers per iteration.
  Changes the contraction rate to
  $X_i \approx ((N - n_\mathrm{cull})/(N+1-n_\mathrm{cull}))^i$.
  See {attr}`jaxrens.state.config.NSConfig.n_cull`.
- **n_extra**: reseed MCMC chains starting from additional random
  live walkers per iteration. Reduces chain correlation at the cost
  of more MCMC work per iteration.
- **Inter-replica exchange**: in multi-run dispatch, after each
  `ns_step` the {doc}`replicas` swap configurations between adjacent
  replicas. Independent from the Python-vs-JIT split above.

## Where this lives in the code

| Concern | File |
|---|---|
| Outer Python loop | {func}`jaxrens.sampling.nested_sampling.run_ns`, {func}`~jaxrens.sampling.nested_sampling.run_ns_parallel`, {func}`~jaxrens.sampling.nested_sampling.run_ns_multi_gpu` |
| Shared loop body | `sampling/run_loop.py::_run_loop` |
| Inner JIT'd step | {func}`jaxrens.sampling.nested_sampling.ns_step` |
| Step-size adaptation | `sampling/adaptation/manager.py::AdaptationManager` |
| Termination | `sampling/termination.py` |
