# Core concepts

Seven ideas hold up the rest of the library. Each maps to a small set
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
  `EnsembleBackend` adds NPT / semi-grand μPT corrections per call.
- {doc}`concepts/replicas` — how `n_total`, `n_gpu`, `n_per_gpu`
  are derived and how inter-replica exchange (RENS) swaps work.
- {doc}`concepts/restart` — fresh-vs-restart lifecycle, output-dir gate,
  `--resume` auto-discovery, and the strict compatibility validator.

```{toctree}
:hidden:
:maxdepth: 1

concepts/schema_resolve
concepts/ns_loop
concepts/pytree_state
concepts/moves_mwg
concepts/backends
concepts/replicas
concepts/restart
```

The summaries below are the 30-second version of each; follow the
links for detail.

## 1. YAML becomes a run in three layers

A config file reaches the sampler through three layers with strictly
separated jobs: the **schema** (pydantic models in
{mod}`jaxrens.cli.schema`) validates shape, types and ranges without
importing JAX; the **resolver** ({mod}`jaxrens.cli.resolve`) turns the
validated spec into runtime objects — backend instance, initial walker
positions, move descriptors, per-replica ensemble params — without
importing pydantic; and the **core** consumes frozen dataclasses and
JAX arrays, knowing nothing about YAML.

That separation is why `jaxrens validate --parse-only` is cheap (schema
only) while `jaxrens validate` does the full resolve including backend
construction. Every field you can write is documented at
{doc}`/reference/config`.

## 2. The two-loop NS structure

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

## 3. Walkers are JAX pytrees

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

## 4. Moves compose through MWG

Individual MCMC kernels live under
{mod}`jaxrens.sampling.moves` — `random_walk`, `galilean`, `hmc`,
`single_atom`, `alchemical`, `swap`, `volume`, `shear`, `stretch`,
`replica_exchange`. Each exposes a `build_kernel(energy_fn, params,
**kernel_kwargs)` factory returning a `step_fn(rng_key, state,
likelihood_constraint) -> (state, MoveInfo)`.

A kernel is handed to the sampler wrapped in a
{class}`~jaxrens.sampling.move_kernel.MoveKernel` descriptor — a
dataclass carrying the kernel's `name`, its `build_kernel`, the
`kernel_kwargs` to bake in, its `weight` and `step_size`, any
`extra_state_fields` it needs on `MCState`, and which state aspects it
`mutates` (so the constraint machinery knows which moves to gate).

`build_mwg(backend, descriptors)` in
{mod}`jaxrens.sampling.mwg` composes any list of them into a
single Metropolis-within-Gibbs step function, with per-move weights
controlling dispatch probabilities.

YAML: the `moves:` section is an ordered list; each entry's `type`
selects the kernel and `weight` sets its MWG probability.

## 5. Backends and ensembles

An energy model is anything implementing the
{class}`~jaxrens.backends.base.EnergyBackend` protocol:
`(positions, species, cell, max_neighbors, ensemble_params) →`
{class}`~jaxrens.backends.base.BackendResult`, a `NamedTuple` whose
`energy` field is the only universally-meaningful one —
`max_neighbor_count` and `overflow` drive the neighbor-bucket manager,
and `forces` is filled only on the `energy_and_forces` path.

Built-ins:

- **`lj`** — Lennard-Jones with periodic cutoff.
- **`mace`** — MACE-JAX wrapper with supercell neighbor expansion.
- **`neuralil`** — NeuralIL with bucketed kernel compilation.
- **`nequix`** — Nequix, from a local checkpoint or a bundled model.
- **`jaxmd`** — jax-md analytic potentials (Tersoff, EAM), all-pairs.
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

## 6. Replica axes: n_total, n_gpu, n_per_gpu

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
example, and {doc}`/reference/config` for the exact divisibility
rules and every field that can drive a replica axis.

## 7. Restart, resume, and the output-dir gate

A run is either **fresh** or a **restart**, and the two are kept apart
deliberately. A fresh run refuses to start if `working_dir` already
holds artifacts under the same `out_file_prefix`, so a second run can
never silently append to or overwrite the first — pass `--force` to
clear them intentionally.

A restart either names its checkpoint explicitly
(`init.restart_file`) or lets `--resume` auto-discover the newest one
in `working_dir`. Either way a compatibility validator checks the
config against what the checkpoint was written with and refuses
mismatches that would corrupt the estimate, rather than continuing
from an inconsistent state. Burn-in is skipped on restart.

## What's where — quick map

| I want to... | Look in |
|---|---|
| Read the YAML schema (fields, types, validation) | {mod}`jaxrens.cli.schema` |
| Understand how a YAML becomes a runtime object | {mod}`jaxrens.cli.resolve` |
| Swap in a new energy model | {mod}`jaxrens.backends.base` (protocol) |
| Add a new MCMC move | {mod}`jaxrens.sampling.move_kernel`, {mod}`jaxrens.sampling.mwg` |
| Debug a run that looks wrong | monitor log `*.log` + `*.adaptation.h5` |
| Post-process evidence / heat capacity | {mod}`jaxrens.postprocess.thermodynamics` |
| Look up what a symbol (`K`, `P`, `μ`, `E_max`, `c_i`, …) means | {doc}`/reference/notation` |
