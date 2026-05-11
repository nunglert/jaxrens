# Schema → resolve → run

Every `jaxrens run -c config.yaml` invocation flows through three
layers. Each layer has one job, and the boundaries are enforced by
the type system:

- **Schema** — pydantic models in `jaxrens.cli.schema.*`. Validates
  the YAML shape and ranges. Does not import JAX.
- **Resolver** — pure Python in `jaxrens.cli.resolve`. Builds the
  runtime objects (backend instance, walker positions, move
  descriptors, per-replica ensemble params). Imports JAX, does not
  import pydantic.
- **Core** — the library itself (`jaxrens.sampling`, `jaxrens.state`,
  `jaxrens.backends`, …). Consumes frozen dataclasses (`NSConfig`,
  `MoveConfig`) and JAX arrays. Imports JAX, does not import
  pydantic or yaml.

```{mermaid}
flowchart TB
    subgraph L1["Layer 1 — wire format (YAML → pydantic)"]
        direction LR
        Y["config.yaml"] --> RC["RootConfig<br/>(pydantic)"]
    end
    subgraph L2["Layer 2 — resolver (pydantic → runtime objects)"]
        direction TB
        RES["expand_multi_run_or_cohort(root)"]
        RES --> RO1["ResolvedConfig<br/>single-run"]
        RES --> RO2["ResolvedMultiRunConfig<br/>multi-replica"]
    end
    subgraph L3["Layer 3 — core (dataclass → NS state pytrees)"]
        direction LR
        RFC["run_from_config"]:::layer3
        RMG["run_multi_gpu_from_config"]:::layer3
        RN["run_ns"]:::layer3
        RMGC["run_ns_multi_gpu"]:::layer3
    end
    RC --> RES
    RO1 --> RFC --> RN
    RO2 --> RMG --> RMGC
    classDef layer3 fill:#e7f1fb,stroke:#666
```

## What each layer owns

| Concern | Layer | Where |
|---|---|---|
| YAML key names, types, defaults | **Schema** | `cli/schema/{run,moves,backend,ensemble,inter_re,init,cell,output,adaptation,termination}.py` |
| Cross-field validators ("XRENS requires composition_targets") | **Schema** | `@model_validator` methods |
| Unit conversion (`gpa` → `eva3`) | **Schema** | `NPTEnsembleSpec.to_ensemble_params` |
| Deprecated-alias migration | **Schema** | `cli/migrate.py` |
| "Build a MACE backend instance" | **Resolver** | `MACEBackendSpec.build_backend()` |
| "Compute initial walker energies at the right ensemble scale" | **Resolver** | `_finalise_initial_energies_and_counts` + `EnsembleBackend` wrap |
| Device-topology derivation from `jax.local_devices()` | **Resolver** | `_derive_replica_axes` |
| Species mapping to backend z-table | **Resolver** | `_resolve_init_species` |
| Pick `n_live`, `n_mcmc_steps` for the scan | **Core** | `run_ns` signature |
| `jax.pmap(jax.vmap(...))` dispatch | **Core** | `sampling/batch_descriptor.py` |

Two asymmetries worth knowing:

- **Schema doesn't import JAX.** You can run `jaxrens validate -c
  config.yaml` on a CPU-only machine with no GPU initialization.
  Validation is cheap.
- **Core doesn't import pydantic.** `run_ns(positions, types, …)`
  takes arrays and a `NSConfig` dataclass. Tests in
  `tests/test_nested_sampling.py` never touch the schema layer, so
  the core is testable without a YAML round-trip.

## The dispatch branch

The CLI's `_cmd_run` delegates to `expand_multi_run_or_cohort`,
which produces **one of two output types**:

```{mermaid}
flowchart LR
    Y["config.yaml"] --> RC["RootConfig"]
    RC --> DR{"_derive_replica_axes:<br/>n_total > 1 ?"}
    DR -->|"no"| EC["expand_cohort(root)<br/>→ list[ResolvedConfig]"]
    DR -->|"yes"| MR["_resolve_multi_run(root)<br/>→ ResolvedMultiRunConfig"]
    EC -->|"len == 1"| RFC["run_from_config<br/>→ run_ns"]
    EC -->|"len > 1"| LOOP["sequential cohort loop<br/>one run_from_config per element"]
    MR --> RMG["run_multi_gpu_from_config<br/>→ run_ns_multi_gpu"]
```

`n_total` comes from whichever YAML list implies it:
`ensemble.pressure` (NPT), `inter_re.composition_targets` (XRENS),
or `inter_re.chemical_potentials` (semi-grand). See
{doc}`replicas` for the full derivation.

When `n_total == 1`, `expand_cohort` returns a single-element list —
the CLI runs it once. A cohort sweep (multiple sequential runs, one
per pressure) is available but rarely used now that multi-run
dispatch is automatic: just make the pressure list long enough and
replicas run *concurrently* instead of serially.

## Inside the resolver

Once the dispatcher has picked a branch, both paths do the same three
jobs — derive the runtime topology, lay out per-walker initial state,
and price the initial energies on the right ensemble scale — they
just differ in *where the parallelism lives*. The single-run / cohort
path builds one replica at a time, then runs a single
`_finalise_initial_energies_and_counts` call to price its initial
energies; the multi-run path builds N replicas in a tight Python loop
(cheap, no JIT compiles), stacks them, and runs the same helper once
on the stacked `(G, P, K, …)` arrays under `pmap(vmap(...))`. The
helper is shared verbatim — only the `batcher` argument differs
(`SingleRun()` vs `PmapVmapRuns(G, P)`). Mode helpers themselves
return structural init only (positions / cells / types / restart
state); finalize is always the caller's responsibility.

```{mermaid}
%%{init: {"layout": "elk"}}%%
flowchart TB
    RS["RootSpec<br>(validated pydantic)"]
    EMC{"replica-axis<br>list present?"}

    subgraph cohort["Cohort path — expand_cohort"]
        direction TB
        SIZE["_cohort_size(root)<br>→ n"]
        COHFOR(["for i in range(n)"])
        RONE["_resolve_one(root, i)"]
    end

    subgraph multi["Multi-run path — _resolve_multi_run"]
        direction TB
        DRA["_derive_replica_axes<br>→ (n_gpu, n_per_gpu,<br>params_per_run)"]
        BATCH["batcher = PmapVmapRuns(G, P)"]
        MFOR(["for r in range(n_total)"])
        STK["jnp.stack along replica axis<br>(n_total, K, …) → (G, P, K, …)"]
    end

    subgraph one["_resolve_one"]
        direction TB
        EP["ensemble.to_ensemble_params(i)<br>→ pressure"]
        BB1["build_backend(spec)<br>→ base_backend"]
        EBW["wrap EnsembleBackend<br>(if pressure)"]
    end

    subgraph init["_resolve_init — structural init"]
        direction TB
        MODE{"select mode<br>(A / B / C / D)"}
        A["A: start_species<br>sample cell + positions"]
        B["B: start_config_file<br>load + replicate"]
        C["C: start_walker_set<br>load verbatim"]
        D["D: restart_file<br>load checkpoint"]
        VAL["_validate_cells"]
    end

    subgraph fin["_finalise_initial_energies_and_counts"]
        direction TB
        HAS{"backend has<br>max_neighbors_for?"}
        CNT["counts = batcher.wrap_for_batch(<br>vmap(max_neighbors_for))<br>(positions, cells)"]
        BUCK["init_bucket =<br>_choose_starting_bucket(<br>counts, ladder, offset)"]
        EN["energies = batcher.wrap_for_batch(<br>vmap(backend(…, init_bucket)))<br>(positions, cells)"]
        EN0["energies = batcher.wrap_for_batch(<br>vmap(backend(…, 0)))<br>(positions, cells)"]
    end

    OUT1["ResolvedConfig<br>(or list thereof)"]
    OUT2["ResolvedMultiRunConfig<br>(carries batcher field)"]

    RS --> EMC
    EMC -- no --> SIZE
    EMC -- yes --> DRA
    SIZE --> COHFOR --> RONE
    DRA --> BATCH --> MFOR
    RONE -.calls.-> EP
    EP --> BB1 --> EBW --> MODE
    MFOR -.->|"structural init only"| MODE
    MODE -- A --> A --> VAL
    MODE -- B --> B --> VAL
    MODE -- C --> C --> VAL
    MODE -- D --> D --> VAL
    VAL -->|"single-run (SingleRun,<br>ladder + offset)"| HAS
    VAL -->|"multi-run: stack first"| STK
    STK -->|"one call (PmapVmapRuns)"| HAS
    HAS -- yes --> CNT --> BUCK --> EN
    HAS -- no --> EN0
    EN --> OUT1
    EN --> OUT2
    EN0 --> OUT1
    EN0 --> OUT2

    cohort:::pyBox
    multi:::pyBox
    one:::pyBox
    init:::pyBox
    fin:::jitBox
    EMC:::decision
    MODE:::decision
    HAS:::decision
    classDef pyBox fill:#f5f5f5,stroke:#888,color:#222
    classDef jitBox fill:#fff7e0,stroke:#a07000,color:#222
    classDef decision fill:#eef5ff,stroke:#1565c0,color:#222
```

The amber box (`_finalise_initial_energies_and_counts`) is the only
JIT-compiled work in the resolver — everything else is plain Python
shuffling pydantic specs into dataclasses. That single helper is
*also* where multi-GPU dispatch enters the resolver, via the
`batcher` argument.

### Topology derivation

`_derive_replica_axes` reads three replica-axis hints from the
`RootSpec` — `ensemble.pressure` length, `inter_re.composition_targets`,
`inter_re.chemical_potentials` — checks they agree, looks up
`jax.local_devices()` to pick `n_gpu`, and demands
`n_total % n_gpu == 0`. The output is the `(n_gpu, n_per_gpu)` shape
that the rest of the runtime (`pmap(vmap(...))`, `AdaptationManager`,
`InterREManager`) inherits. Cohort path skips this entirely — its
topology is `SingleRun()`.

### The four init modes

`_resolve_init` is a router. The init spec must set exactly one of:

| Mode | YAML key | Helper | What it does |
|---|---|---|---|
| A | `start_species` | `_resolve_init_species` | Parse species string ("Si16"), sample cell + positions from priors |
| B | `start_config_file` | `_resolve_init_config_file` | Load a single founder structure, replicate to `n_live` walkers (with optional cell-shape walk + position re-sampling) |
| C | `start_walker_set` | `_resolve_init_walker_set` | Load a pre-computed `(n_live, N, 3)` walker file verbatim |
| D | `restart_file` | `_resolve_init_restart` | Resume from an NS checkpoint; carries `RestartBundle` |

All four converge on `_validate_cells` (mins/maxes on volume per atom
and aspect ratio — fail fast on a bad input rather than later in the
NS loop with a misleading "walker produced invalid cell" message)
and then return structural init only: positions / cells / types /
symbol map / (Mode D) restart bundle, with
`initial_energies = initial_max_neighbor_counts = None`. Pricing the
energies is the caller's job — `_resolve_one` does it on the
single-replica arrays it just built, and `_resolve_multi_run` does it
once on the stacked `(G, P, K, …)` pytree after the per-replica loop
finishes. Mode helpers therefore see exactly one concern (place the
walkers) and never have to know whether they were called from the
single-run or multi-run path.

### The consolidated finalize

`_finalise_initial_energies_and_counts` is the resolver's compile
hot-spot, and the *only* place initial energies are priced.  It is
called exactly once per resolution path — once from `_resolve_one`
for the single-run / cohort path, once from `_resolve_multi_run`
after stacking N replicas.  It earns three optimisations:

1. **Bucket-aware compile.** For backends with bucketed kernel
   dispatch (MACE, NeuralIL — see {doc}`backends`), the helper first
   computes per-walker neighbor counts via `backend.max_neighbors_for`
   (geometry-only, no energy compile), then picks
   `init_bucket = _choose_starting_bucket(counts, ladder, offset)` —
   the same bucket the NS step's bucket-escalation logic would land
   on. Both resolution paths now pass `ladder = backend.max_neighbors_list`
   and `offset = backend.max_neighbors_offset` (single-run previously
   used the legacy `int(max(counts))` fallback; the paths agreed only
   on the multi-run side). The initial-energy compile therefore lands
   in the same JIT cache slot the burn-in and the NS step will hit,
   instead of wasting a compile at `max_neighbors=0`. Backends without
   `max_neighbors_for` (LJ, toy, jax-md) pass `max_neighbors=0`
   straight through; the bucket arg is inert for them.
2. **`batcher.wrap_for_batch(per_replica_fn)` dispatch.** The same
   helper body works for all topologies because the `batcher` argument
   chooses `jax.jit` / `jax.jit(vmap)` / `pmap(vmap)` underneath.
   Cohort path passes `SingleRun()`; the multi-run path passes the
   same `PmapVmapRuns(G, P)` instance it stores on
   `ResolvedMultiRunConfig.batcher` — so the resolver, burn-in, and
   NS step all dispatch through one shared batcher instance.
3. **Per-replica pressure under one wrapper.** The multi-run path
   wraps the base backend in a *single* `EnsembleBackend(pressure=0.0)`
   and threads each replica's pressure through the per-call
   `ensemble_params={"pressure": p}` kwarg, vmapped over the replica
   axis. Avoiding N separate `EnsembleBackend(base, pressure=p)`
   objects keeps the pmap call signature uniform — pressure is just
   another array on the vmap axis.

The end result is that for an 8-replica NeuralIL run on a 4-GPU
node, the resolver compiles the energy function *once* at shape
`(4, 2, n_live, n_atoms, 3)` at the same `max_neighbors` bucket
burn-in will use — instead of once per replica at the wrong bucket.
That's where the resolver's ~10× speedup on heavy-backend configs
comes from.

## The two resolved types

`ResolvedConfig` and `ResolvedMultiRunConfig` are sibling frozen
dataclasses, not a subclass. The asymmetry is real and intentional:

| Field | `ResolvedConfig` | `ResolvedMultiRunConfig` |
|---|---|---|
| `energy_backend` | wrapped `EnsembleBackend(pressure=P)` | unwrapped base backend |
| ensemble params | `ensemble_params: dict` (one dict) | `ensemble_params_per_run: tuple[dict, ...]` |
| `init.initial_positions` shape | `(n_live, A, 3)` | `(n_total, n_live, A, 3)` |
| `init.initial_energies` shape | `(n_live,)` | `(n_total, n_live)` |

Why they differ: in single-run mode the resolver pre-wraps the
backend at a specific pressure, so the initial walker energies
already include the `+P·V` term. In multi-run mode we wrap once at
`pressure=0.0` and the per-call `ensemble_params` dict overrides
that default, so one backend instance can serve eight replicas at
eight different pressures without rebuilding. That decision is a
runtime-object concern — it lives in the resolver, not the schema.

## Concrete trace

Take the MACE pressure-RENS config at
`experiments/mace_srtio3/config.yaml`:

```yaml
ensemble:
  type: npt
  pressure: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  pressure_units: gpa

inter_re:
  flavor: pressure
  every: 5
```

Step by step:

1. **Load & validate**: `cli/cli.py::_load_and_validate` reads the
   YAML and calls `RootConfig.model_validate`. pydantic runs all
   validators; errors surface here with line-accurate messages.
2. **Unit conversion**: `NPTEnsembleSpec._pressure_list()` returns
   `[1.0, …, 8.0]`. When called via `to_ensemble_params(i)` with
   `pressure_units="gpa"`, each scalar is multiplied by
   `_GPA_TO_EVA3` (schema-layer concern).
3. **Dispatcher**: `_cmd_run` calls
   `expand_multi_run_or_cohort(root)`. It sees the 8-element list
   and dispatches to `_resolve_multi_run`.
4. **Topology**: `_derive_replica_axes` computes
   `n_total = 8`, reads `len(jax.local_devices()) = 4` on a
   `--gres=gpu:4` SLURM allocation, returns
   `(n_gpu=4, n_per_gpu=2)`. Divisibility is checked here.
5. **Per-replica structural init**: for each of 8 replicas,
   `_resolve_init` builds positions / cells / types only. A
   per-replica `EnsembleBackend(base, pressure=P_r)` is constructed
   for the rejection-mode ceiling check inside
   `_sample_per_walker_positions`, but no initial energies are
   computed yet. Per-replica arrays are stacked along axis 0 into
   `(n_total, K, …)`.
6. **Consolidated finalize**: `_resolve_multi_run` reshapes to
   `(G, P, K, …)`, wraps the base backend in *one*
   `EnsembleBackend(pressure=0.0)`, and calls
   `_finalise_initial_energies_and_counts` with
   `batcher=PmapVmapRuns(G, P)`, the per-replica pressures fed
   through `ensemble_params={"pressure": P_r}` on the vmap axis, and
   `ladder` / `offset` from the backend config. One compile across
   all 4 GPUs at the bucket the NS step will use.
7. **Runtime handoff**: `ResolvedMultiRunConfig` is returned to
   `_cmd_run`, which calls
   `cli/run.py::run_multi_gpu_from_config`. From here on, no
   schema, no resolver — just JAX arrays and dataclasses flowing
   through `run_ns_multi_gpu` with its `pmap(vmap(ns_step))`
   dispatch.

## Why this structure

Three properties the three-layer split buys you:

1. **Early, single-point validation.** Everything you can check
   without running JAX is checked in `jaxrens validate`. You don't
   lose 10 minutes of GPU time to a typo like `n_liv: 500`.
2. **Independent evolution of the YAML surface and the core.** Add
   a new YAML shortcut in the schema without touching the core;
   rename a `NSConfig` field in the core without breaking YAML
   compatibility (the resolver adapts).
3. **Programmatic entry point.** Nothing forces you through the
   YAML layer. Build a `ResolvedMultiRunConfig` by hand in Python
   and call `run_multi_gpu_from_config` directly — exactly what
   `tests/test_cli_multi_run.py` does.

## Where this lives in the code

| Layer | File(s) |
|---|---|
| Schema | `src/jaxrens/cli/schema/` (10 spec modules) + `cli/schema/__init__.py` re-export |
| Resolver | `src/jaxrens/cli/resolve.py` (~1000 LoC, one concern per function) |
| Single-run entry point | `src/jaxrens/cli/run.py::run_from_config` |
| Multi-run entry point | `src/jaxrens/cli/run.py::run_multi_gpu_from_config` |
| CLI dispatch | `src/jaxrens/cli/cli.py::_cmd_run` |
| Library core | `src/jaxrens/{sampling,state,backends,init,io,postprocess}/` |
