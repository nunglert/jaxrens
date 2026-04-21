# Core concepts

Six ideas hold up the rest of the library. Each maps to a small set
of files and a configuration section; once you know these you can
read everything else. Each has a dedicated subpage with the
relevant mathematics, diagrams, and pointers into the code:

- {doc}`concepts/schema_resolve` — how a YAML file becomes a
  running NS job (schema → resolve → run).
- {doc}`concepts/ns_loop` — the two-loop NS structure, prior-mass
  contraction, evidence estimator.
- {doc}`concepts/pytree_state` — how walkers live as JAX pytrees
  and how batch axes flow through `jit`/`vmap`/`pmap`.
- {doc}`concepts/moves_mwg` — individual move kernels and how MWG
  composes them into a single step function.
- {doc}`concepts/backends` — the `EnergyBackend` protocol and how
  `EnsembleBackend` adds NPT/μVT corrections per call.
- {doc}`concepts/replicas` — how `n_total`, `n_gpu`, `n_per_gpu`
  are derived and how inter-replica exchange (RENS) swaps work.

```{toctree}
:hidden:
:maxdepth: 1

concepts/schema_resolve
concepts/ns_loop
concepts/pytree_state
concepts/moves_mwg
concepts/backends
concepts/replicas
```

The summaries below are the 30-second version of each; follow the
links for detail.

## 1. The two-loop NS structure

Nested sampling in jaxrens runs as an **outer Python loop** around
an **inner `lax.scan`**. The boundary is load-bearing:

- **Outer loop** — step-size adaptation, move-kernel dispatch,
  overflow retry for MLIP neighbor-buffer resizing, termination
  checks, callback dispatch, I/O. Written in plain Python because
  it needs dynamic shapes and Python-side control flow.
- **Inner scan** — `n_mcmc_steps` of MWG sampling per dead-point
  replacement. Fully JIT-compiled; shape-stable.

Code: {mod}`jaxrens.sampling.nested_sampling`, especially
`run_ns`, `run_ns_parallel`, `run_ns_multi_gpu`.

## 2. Walkers are JAX pytrees

The per-walker state is a {class}`~jaxrens.state.mc_state.MCState`
dataclass — positions, cell, species, step-size, energy,
ensemble-params, plus any move-specific extra fields (e.g. a
momentum for HMC, a velocity direction for galilean). It's
registered with JAX's pytree machinery so `jit`, `vmap`, and `pmap`
just work on it.

NS-level state ({class}`~jaxrens.state.ns.NSState`) wraps the
batched walker population plus dead-point history, iteration
counter, log-evidence, and cumulative counters.

Static fields (compile-time constants like `n_atoms`, `max_neighbors`)
are tagged with `static_field()` so changing them triggers
recompilation, while dynamic fields are leaves.

## 3. Moves compose through MWG

Individual MCMC kernels live under
{mod}`jaxrens.sampling.moves` — `random_walk`, `galilean`, `hmc`,
`single_atom`, `alchemical`, `volume`, `shear`, `stretch`,
`replica_exchange`. Each implements the `MoveKernel` protocol
(init / build_kernel / as_top_level_api).

`build_mwg(backend, descriptors)` in
{mod}`jaxrens.sampling.mwg` composes any list of them into a
single Metropolis-within-Gibbs step function, with per-move weights
controlling dispatch probabilities.

YAML: the `moves:` section is an ordered list; each entry's `type`
selects the kernel and `weight` sets its MWG probability.

## 4. Backends and ensembles

An energy model is anything implementing the
{class}`~jaxrens.backends.base.EnergyBackend` protocol:
`(positions, species, cell, max_neighbors, ensemble_params) →
(energy, neighbor_count, overflow)`.

Built-ins:

- **`lj`** — Lennard-Jones with periodic cutoff.
- **`mace`** — MACE-JAX wrapper with supercell neighbor expansion.
- **`neuralil`** — NeuralIL with bucketed kernel compilation.
- **`harmonic`**, **`double_well`**, **`gaussian_mixture`** — toy
  potentials for testing.

Ensemble corrections are applied by a thin wrapper,
{class}`~jaxrens.backends.ensemble.EnsembleBackend`:

- **NVT** — no correction (just use the base backend).
- **NPT** — `H = U + P·V`.
- **μPT / semi-grand** — `H = U + P·V − μ·N`.

`ensemble_params` is passed per call, so different replicas can run
at different pressures / chemical potentials without rebuilding the
backend.

## 5. Replica axes: n_total, n_gpu, n_per_gpu

Multi-run dispatch activates when the YAML config implies more than
one replica via a **replica-differentiating list**:

- `ensemble.pressure: [1, 2, 4, 8]` → 4 replicas at those pressures.
- `inter_re.composition_targets: [[8, 0], [4, 4]]` → 2 replicas (XRENS).
- `inter_re.chemical_potentials: [...]` → N replicas (semi-grand).

The resolver reads `len(jax.local_devices())` at run time and splits
the replicas evenly:

- `n_total = len(list)` — total replica count.
- `n_gpu = len(jax.local_devices())` — detected.
- `n_per_gpu = n_total / n_gpu` — derived (must divide evenly).

The NS state then has shape `(n_gpu, n_per_gpu, n_walkers, ...)` on
every dynamic field, and execution is `pmap(vmap(vmap(ns_step)))`.

See the {doc}`../tutorials/index` for a concrete multi-GPU
example and the config reference (follow-up PR) for the exact
divisibility rules.

## What's where — quick map

| I want to... | Look in |
|---|---|
| Read the YAML schema (fields, types, validation) | {mod}`jaxrens.cli.schema` |
| Understand how a YAML becomes a runtime object | {mod}`jaxrens.cli.resolve` |
| Swap in a new energy model | {mod}`jaxrens.backends.base` (protocol) |
| Add a new MCMC move | {mod}`jaxrens.sampling.move_kernel`, {mod}`jaxrens.sampling.mwg` |
| Debug a run that looks wrong | monitor log `*.log` + `*.adaptation.h5` |
| Post-process evidence / heat capacity | {mod}`jaxrens.postprocess.thermodynamics` |
