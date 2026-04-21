# The nested-sampling loop

Nested sampling (Skilling, 2006) converts a multi-dimensional integral
over configuration space into a one-dimensional integral over *prior
mass*. jaxrens implements the standard live-walker scheme with one
wrinkle: the loop is split into a **Python outer loop** around a
**JIT-compiled inner scan**, and the boundary between them is
load-bearing.

## What NS computes

For a prior $\pi(\theta)$ and likelihood $L(\theta) = e^{-H(\theta)}$,
the evidence is

$$
Z = \int L(\theta)\,\pi(\theta)\,\mathrm{d}\theta
  = \int_0^1 L(X)\,\mathrm{d}X,
$$

where the transformation

$$
X(L^\star) = \int_{L(\theta) > L^\star} \pi(\theta)\,\mathrm{d}\theta
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
2. **Run `n_mcmc_steps` of MWG sampling** to produce a new walker
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
flowchart TB
    subgraph outer["Outer loop (Python)"]
        direction TB
        T["Termination check"]
        A["Step-size adaptation<br/>(every adjust_interval)"]
        O["Overflow retry<br/>(max_neighbors escalation)"]
        C["Callback dispatch<br/>(monitor / trajectory / checkpoint)"]
    end
    subgraph inner["Inner loop (JIT + lax.scan)"]
        S["ns_step:<br/>n_mcmc_steps × MWG over all walkers"]
    end
    T -->|"not terminated"| S
    S -->|"info (emax, acc, counters)"| A
    A --> O
    O --> C
    C --> T
```

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
$H(\theta) < E_\mathrm{max}$. Over many iterations, $E_\mathrm{max}$
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
