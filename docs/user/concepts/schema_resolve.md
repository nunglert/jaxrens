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
        Y["config.yaml"] --> RC["RootSpec<br/>(pydantic)"]
    end
    subgraph L2["Layer 2 — resolver (pydantic → runtime objects)"]
        direction TB
        RES["resolve(root)"]
        RES --> RO["ResolvedConfig<br/>(carries a batcher)"]
    end
    subgraph L3["Layer 3 — core (dataclass → NS state pytrees)"]
        direction LR
        RFC["run_from_config"]:::layer3
        RMG["run_multi_gpu_from_config"]:::layer3
        RN["run_ns"]:::layer3
        RMGC["run_ns_multi_gpu"]:::layer3
    end
    RC --> RES
    RO -->|"batcher = SingleRun()"| RFC --> RN
    RO -->|"batcher = PmapVmapRuns(G, P)"| RMG --> RMGC
    classDef layer3 fill:#e7f1fb,stroke:#666
```

## What each layer owns

| Concern | Layer | Where |
|---|---|---|
| YAML key names, types, defaults | **Schema** | `cli/schema/{run,moves,backend,ensemble,inter_re,init,cell,output,adaptation,termination}.py` |
| Cross-field validators ("XRENS requires composition_targets") | **Schema** | `@model_validator` methods |
| Unit conversion (`gpa` → `eva3`) | **Schema** | `NPTEnsembleSpec.to_ensemble_params` |
| Interval-unit scaling (`per_walker` → absolute iters) | **CLI / Resolver** | `_apply_interval_units` (run from cli before the resolver, second pass inside `resolve` is idempotent) |
| Deprecated-alias migration | **Schema** | `cli/migrate.py` |
| "Build a MACE backend instance" | **Resolver** | `MACEBackendSpec.build_backend()` |
| "Compute initial walker energies at the right ensemble scale" | **Resolver** | `_finalise_initial_energies_and_counts` + `EnsembleBackend` wrap |
| Device-topology derivation from `jax.local_devices()` | **Resolver** | `_derive_replica_axes` |
| Species mapping to backend z-table | **Resolver** | `_resolve_init_species` |
| Pick `n_live`, `n_mcmc_steps` for the scan | **Core** | `run_ns` signature |
| `jax.pmap(jax.vmap(...))` dispatch | **Core** | `sampling/batch_descriptor.py` |

Two asymmetries worth knowing:

- **Schema doesn't import JAX.** `jaxrens validate --parse-only -c
  config.yaml` runs pydantic-only and is cheap; the full `validate`
  (without `--parse-only`) runs the resolver and may build a backend.
- **Core doesn't import pydantic.** `run_ns(positions, types, …)`
  takes arrays and a `NSConfig` dataclass. Tests in
  `tests/test_nested_sampling.py` never touch the schema layer, so
  the core is testable without a YAML round-trip.

## The unified entry point

The resolver exposes **one** public function:

```python
def resolve(root: RootSpec) -> ResolvedConfig: ...
```

Whether the YAML describes a single NS run or 16 replicas across 4
GPUs, the return type is the same — a single `ResolvedConfig` that
carries a `batcher` field describing the topology. The CLI inspects
that field to pick the right runtime dispatcher:

```{mermaid}
flowchart LR
    Y["config.yaml"] --> RC["RootSpec"]
    RC --> RES["resolve(root)"]
    RES --> RC2["ResolvedConfig"]
    RC2 -->|"batcher = SingleRun()<br/>n_total = 1"| RFC["run_from_config<br/>→ run_ns"]
    RC2 -->|"batcher = PmapVmapRuns(G, P)<br/>n_total = G·P"| RMG["run_multi_gpu_from_config<br/>→ run_ns_multi_gpu"]
```

`n_total` comes from whichever YAML list implies it:
`ensemble.pressure` (NPT), `inter_re.composition_targets` (XRENS),
or `inter_re.chemical_potentials` (semi-grand). See
{doc}`replicas` for the full derivation. When `n_total == 1` the
batcher is `SingleRun()`; otherwise it is `PmapVmapRuns(G, P)` with
`G = max(1, len(jax.local_devices()))` clamped to `n_total` and
`P = n_total // G`.

## Inside the resolver

`resolve` is a thin dispatcher: it derives the topology and delegates
to one of two private branches (`_resolve_single_replica` or
`_resolve_multi_replica`). Both branches do the same three jobs —
derive the runtime topology, lay out per-walker initial state, and
price the initial energies on the right ensemble scale — they just
differ in *where the parallelism lives*.

Both paths walk the same per-iteration setup: look up the iteration's
pressure, build the base backend, wrap it in
`EnsembleBackend(base, pressure=P)` (the rejection-mode ceiling check
inside `_sample_per_walker_positions` needs ensemble-corrected
energies), then dispatch the right `_resolve_init` mode helper.  Mode
helpers return structural init only (positions / cells / types /
restart state); finalize is always the caller's responsibility.

The paths then diverge:

- **Single-replica.** After `_validate_cells`, `_resolve_single_replica`
  calls `_finalise_initial_energies_and_counts` immediately on its
  single-replica arrays with `batcher=SingleRun()`.
- **Multi-replica.** The structural-init step runs once per replica in
  a tight Python loop (cheap, no JIT compiles), stacks the resulting
  `(K, …)` arrays into `(n_total, K, …)`, then reshapes to
  `(G, P, K, …)`.  Before finalize the resolver does a *second*
  ensemble wrap — one `EnsembleBackend(base, pressure=0.0)` shared by
  all replicas; the per-replica pressure flows in through the
  `ensemble_params={"pressure": p}` kwarg on the vmap axis.  Then the
  same `_finalise_initial_energies_and_counts` helper is called once
  with `batcher=PmapVmapRuns(G, P)`.

The helper body is identical between paths — only the `batcher` and
the per-replica `pressures` argument differ.

### Structural init — `_resolve_init`

Before showing the full `resolve` flow, here is `_resolve_init`
(resolve.py:604) in isolation. Both branches call it the same way
— the only difference is *how many times* it runs (once for
single-replica, n_total times in the multi-replica loop, each with
its own seed and per-replica `EnsembleBackend` wrapper). The four
modes are mutually exclusive (the init spec must set exactly one
of the four discriminator fields) and all converge on
`_validate_cells`; none of them prices energies — that is the
caller's job after `_resolve_init` returns.

```{mermaid}
%%{init: {"layout": "elk"}}%%
flowchart LR
    subgraph INIT["_resolve_init"]
        direction LR
        IN["init_spec, n_live, seed,<br>energy_backend, cell_cfg"]
        MODE{"init spec field set"}
        MA["A: start_species<br>_resolve_init_species<br>(sample cell + positions from priors)"]
        MB["B: start_config_file<br>_resolve_init_config_file<br>(load founder, replicate to n_live)"]
        MC["C: start_walker_set<br>_resolve_init_walker_set<br>((n_live, N, 3) verbatim)"]
        MD["D: restart_file<br>_resolve_init_restart<br>(checkpoint → RestartBundle)"]
        VAL["_validate_cells<br>(volume/atom + aspect-ratio bounds)"]
        OUT["ResolvedInit(<br>positions, cells, types,<br>symbol_map, restart_state,<br>energies=None, counts=None)"]

        IN --> MODE
        MODE -- A --> MA --> VAL
        MODE -- B --> MB --> VAL
        MODE -- C --> MC --> VAL
        MODE -- D --> MD --> VAL
        VAL --> OUT
    end

    classDef pyBox fill:#f5f5f5,stroke:#888,color:#222
    classDef decision fill:#eef5ff,stroke:#1565c0,color:#222
    classDef initFrame fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#004d40
    IN:::pyBox
    OUT:::pyBox
    MA:::pyBox
    MB:::pyBox
    MC:::pyBox
    MD:::pyBox
    VAL:::pyBox
    MODE:::decision
    class INIT initFrame
```

The `energy_backend` argument is the only place the caller's
ensemble wrap leaks in — rejection-mode initialization uses it for
the ceiling check inside `_sample_per_walker_positions`. Grid-mode
ignores it. The four mode helpers return *structural init only*
(`energies=None`, `counts=None`); pricing them is the next
subroutine, `_finalise_initial_energies_and_counts`.

### Initial-energy pricing — `_finalise_initial_energies_and_counts`

The second subroutine called from both branches (resolve.py:333).
It is the resolver's only JIT-compiled work and the only place
initial energies are priced. The `batcher` argument is the seam
through which multi-GPU dispatch enters the resolver:
`SingleRun()` → plain `jax.jit`; `PmapVmapRuns(G, P)` →
`jax.pmap(jax.vmap(...))`. Both branches pass the backend's
`ladder` / `offset` so the chosen `init_bucket` matches what
burn-in and the NS step will compile against (same JIT cache slot).

```{mermaid}
%%{init: {"layout": "elk"}}%%
flowchart LR
    subgraph FIN["_finalise_initial_energies_and_counts"]
        direction LR
        FN["backend, positions, types, cells,<br>batcher, ladder, offset,<br>pressures=None"]
        HAS{"backend has<br>max_neighbors_for?"}
        CNT["counts = batcher.wrap_for_batch(<br>vmap(max_neighbors_for))(positions, cells)"]
        BUCK["init_bucket =<br>_choose_starting_bucket(counts, ladder, offset)"]
        EN["energies = batcher.wrap_for_batch(<br>vmap(backend(…, init_bucket)))<br>(positions, cells [, ensemble_params])"]
        EN0["energies = batcher.wrap_for_batch(<br>vmap(backend(…, 0)))<br>(positions, cells [, ensemble_params])"]
        OUTF["(energies, counts)"]

        FN --> HAS
        HAS -- yes --> CNT --> BUCK --> EN --> OUTF
        HAS -- no --> EN0 --> OUTF
    end

    classDef pyBox fill:#f5f5f5,stroke:#888,color:#222
    classDef decision fill:#eef5ff,stroke:#1565c0,color:#222
    classDef finFrame fill:#fff7e0,stroke:#a07000,stroke-width:2px,color:#5a4000
    FN:::pyBox
    OUTF:::pyBox
    CNT:::pyBox
    BUCK:::pyBox
    EN:::pyBox
    EN0:::pyBox
    HAS:::decision
    class FIN finFrame
```

`pressures` is only passed in by the multi-replica branch, where
it is a `(G, P)` array — each replica's NPT scalar — fed into the
`ensemble_params` kwarg on the vmap axis. The single-replica
branch's per-call pressure is already baked into its
`init_backend` wrapper (one `EnsembleBackend(base, P)`), so the
kwarg is `None`.

### The full `resolve` flow

In the diagram below, the **teal** boxes (`_resolve_init(...)`)
and the **amber** boxes (`_finalise_initial_energies_and_counts`)
are placeholders for the two mini-diagrams above. The matching
colors are deliberate — the call site in the main flow links
visually to the subroutine that drew that color.

```{mermaid}
%%{init: {"layout": "elk"}}%%
flowchart TB
    %% --- resolve(root) prelude (resolve.py:1432) ---
    RS["RootSpec<br>(validated pydantic)"]
    SCALE["_apply_interval_units(root)<br>(idempotent — CLI already scaled)"]
    DRA["_derive_replica_axes(root)<br>→ (n_total, n_gpu, n_per_gpu, params_per_run)"]
    GUARD{"init.restart_file set<br>AND n_total &gt; 1?"}
    GE["raise ValueError"]
    BR{"n_total == 1?"}

    RS --> SCALE --> DRA --> GUARD
    GUARD -- yes --> GE
    GUARD -- no --> BR

    %% --- single-replica branch (resolve.py:918) ---
    subgraph S["_resolve_single_replica"]
        direction TB
        S1["base_backend = root.backend.build_backend()"]
        S2["init_backend =<br>EnsembleBackend(base, P) if P else base<br>(P from ensemble_params)"]
        subgraph S3box["_resolve_init"]
            S3in["root.init, n_live,<br>seed=root.run.seed,<br>init_backend, cell_cfg"]
        end
        subgraph S4box["_finalise_initial_energies_and_counts"]
            S4in["init_backend, positions, types, cells,<br>batcher=SingleRun(),<br>ladder, offset"]
        end
        S5["return ResolvedConfig(<br>batcher=SingleRun(),<br>ensemble_params_per_run=(eparams,))"]
        S1 --> S2 --> S3in --> S4in --> S5
    end

    %% --- multi-replica branch (resolve.py:1181) ---
    subgraph M["_resolve_multi_replica"]
        direction TB
        M1["batcher = PmapVmapRuns(n_gpu, n_per_gpu)"]
        M2["base_backend = root.backend.build_backend()<br>(once, outside the loop)"]
        M3(["for r in range(n_total)"])
        M3a["P_r = params_per_run[r].get('pressure')<br>per_run_backend =<br>EnsembleBackend(base, P_r) if P_r else base"]
        subgraph M3box["_resolve_init"]
            M3bin["root.init, n_live,<br>seed=root.run.seed + r,<br>per_run_backend, cell_cfg"]
        end
        M4["jnp.stack per-replica init along axis 0<br>→ (n_total, K, …)"]
        M5["reshape → (G, P, K, …)"]
        M6["finalize_backend =<br>EnsembleBackend(base, P=0.0) if any_pressure else base"]
        subgraph M7box["_finalise_initial_energies_and_counts"]
            M7in["finalize_backend,<br>(G,P,K,N,3) positions, types, (G,P,K,3,3) cells,<br>batcher=PmapVmapRuns,<br>ladder, offset, pressures=(G,P)"]
        end
        M8["reshape (G,P,K,…) back to (n_total, K, …)<br>(downstream dispatcher input)"]
        M9["return ResolvedConfig(<br>batcher=PmapVmapRuns(G,P),<br>ensemble_params_per_run=tuple(params_per_run))"]
        M1 --> M2 --> M3
        M3 --> M3a --> M3bin
        M3bin -.->|"replica r built;<br>loop continues"| M3
        M3 -->|"all n_total replicas built"| M4 --> M5 --> M6 --> M7in --> M8 --> M9
    end

    BR -- yes --> S1
    BR -- no --> M1

    classDef pyBox fill:#f5f5f5,stroke:#888,color:#222
    classDef initFrame fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#004d40
    classDef finFrame fill:#fff7e0,stroke:#a07000,stroke-width:2px,color:#5a4000
    classDef decision fill:#eef5ff,stroke:#1565c0,color:#222
    S1:::pyBox
    S2:::pyBox
    S3in:::pyBox
    S4in:::pyBox
    S5:::pyBox
    M1:::pyBox
    M2:::pyBox
    M3a:::pyBox
    M3bin:::pyBox
    M4:::pyBox
    M5:::pyBox
    M6:::pyBox
    M7in:::pyBox
    M8:::pyBox
    M9:::pyBox
    GUARD:::decision
    BR:::decision
    class S3box initFrame
    class M3box initFrame
    class S4box finFrame
    class M7box finFrame
```

Reading the diagram:

- The prelude (`_apply_interval_units` → `_derive_replica_axes` →
  restart guard → topology branch) is the first ~30 lines of
  `resolve()`. After that the two branches are entirely separate.
- Inner step boxes are gray — same as every other resolver step.
  The two subroutines that have their own mini-diagrams above are
  shown as **colored wrappers** around the gray call-args box:
  teal frame for `_resolve_init`, amber frame for
  `_finalise_initial_energies_and_counts`. The wrapper's name
  matches the mini-diagram heading so the reader can drill in.
- `_resolve_single_replica` is linear: build backend → wrap → init
  (teal wrapper) → finalize (amber wrapper) → return.
- `_resolve_multi_replica` has the per-replica Python loop on the
  inside (`build_backend` runs *once* outside it; `EnsembleBackend`
  wrap and `_resolve_init` (teal wrapper) run *n_total* times);
  then the stack-reshape-finalize (amber wrapper)-reshape chain
  produces the final `(n_total, K, …)` arrays.
- The amber `_finalise_…` wrapper is the only JIT-compiled work in
  the resolver. Everything else is plain Python shuffling pydantic
  specs into dataclasses.

### Topology derivation

`_derive_replica_axes` reads three replica-axis hints from the
`RootSpec` — `ensemble.pressure` length, `inter_re.composition_targets`,
`inter_re.chemical_potentials` — checks they agree, looks up
`jax.local_devices()` to pick `n_gpu`, and demands
`n_total % n_gpu == 0`. The output is the `(n_gpu, n_per_gpu)` shape
that the rest of the runtime (`pmap(vmap(...))`, `AdaptationManager`,
`InterREManager`) inherits. Single-replica path skips most of this —
its batcher is `SingleRun()` and there is no replica axis.

### Interval-unit scaling

Every "do this every N iterations" knob in the config — how often to
write a frame, flush the log, check termination, attempt a replica
swap, re-tune the step size — is counted in NS iterations by default.
But a single NS iteration replaces just `n_cull` of the `n_live`
walkers, so the *natural* cadence for most of these is one **walker
sweep** (`n_live` iterations — roughly "every walker touched once"),
not one iteration. The catch is that a sweep is `n_live` iterations,
so an interval tuned for `n_live=500` is off by 4× at `n_live=2000`.

`interval_units` removes that coupling. It is a single top-level key
with two modes:

- **`absolute`** (default) — interval fields are raw NS iteration
  counts. What you write is what the runtime sees.
- **`per_walker`** — interval fields are walker-sweeps. The resolver
  multiplies each by `run.n_live`, so the *same* YAML keeps the same
  physical cadence no matter what `n_live` you run at.

```yaml
run:
  n_live: 1000
interval_units: per_walker   # everything below is in sweeps now
output:
  info_interval: 1           # → log every 1000 iterations
  traj_interval: 10          # → dump a frame every 10 sweeps = 10 000 iters
  snapshot_interval: 0.001   # → 0.001 × 1000 = every 1 iteration
```

The resolver applies this in `_apply_interval_units(root)`, which
scales these fields (and leaves an unset `None`, e.g. an omitted
`run.max_iterations`, untouched):

| Section | Fields |
|---|---|
| `output` | `info`, `traj`, `snapshot`, `checkpoint`, `flush`, `temperature_lag`, `temperature`, `acc_rates`, `max_neighbors`, `collision_check` `_interval` |
| `run` | `max_iterations` |
| `termination` | `max_iterations` of any `iteration` criterion |
| `inter_re` | `re_interval` |
| `adaptation` | `adjust_interval` |

Two practical notes:

- **Rounding.** Each scaled value is rounded to the nearest int and
  clamped to `>= 1`, so a fractional sweep like `snapshot_interval:
  0.001` resolves to a real iteration count and never collapses to 0.
  This is why the interval fields accept `int | float` in the schema.
- **Restart.** Because the unit only changes *how the YAML is read*,
  a restart config may switch `interval_units` freely — the restart
  validator scales both sides to absolute iterations before comparing
  (see {doc}`restart`).

Implementation detail: the CLI calls `_apply_interval_units` early —
in `_cmd_run`, right after `_load_and_validate` — and then flips
`root.interval_units` to `"absolute"`. The resolver's own call at the
top of `resolve()` therefore becomes a no-op second pass (factor=1).
Direct callers of `resolve` (tests, scripts) that bypass the CLI still
get correct scaling automatically.

### The four init modes (reference)

`_resolve_init` is the router shown in the mini-diagram above. The
init spec must set exactly one of:

| Mode | YAML key | Helper | What it does |
|---|---|---|---|
| A | `start_species` | `_resolve_init_species` | Parse species string ("Si16"), sample cell + positions from priors |
| B | `start_config_file` | `_resolve_init_config_file` | Load a single founder structure, replicate to `n_live` walkers (with optional cell-shape walk + position re-sampling) |
| C | `start_walker_set` | `_resolve_init_walker_set` | Load a pre-computed `(n_live, N, 3)` walker file verbatim |
| D | `restart_file` | `_resolve_init_restart` | Resume from an NS checkpoint; carries `RestartBundle` |

`_validate_cells` enforces volume-per-atom and aspect-ratio bounds —
fail fast on a bad input rather than later in the NS loop with a
misleading "walker produced invalid cell" message. Pricing the
energies is the caller's job, not the mode helper's:
`_resolve_single_replica` does it on the single-replica arrays it
just built; `_resolve_multi_replica` does it once on the stacked
`(G, P, K, …)` pytree after the per-replica loop finishes. Mode
helpers therefore see exactly one concern (place the walkers) and
never have to know which path called them.

Mode D (restart) is incompatible with multi-replica topologies — the
resolver raises a `ValueError` when `restart_file` is set alongside a
replica-axis list. Restart is single-NS only by design; the
checkpoint carries one trajectory, not N.

### Why the consolidated finalize matters

The mini-diagram earlier shows *what* `_finalise_initial_energies_and_counts`
does. This section is the *why* — the three optimisations the
helper earns by being the single shared entry point for both
branches:

1. **Bucket-aware compile.** For backends with bucketed kernel
   dispatch (MACE, NeuralIL — see {doc}`backends`), the helper first
   computes per-walker neighbor counts via `backend.max_neighbors_for`
   (geometry-only, no energy compile), then picks
   `init_bucket = _choose_starting_bucket(counts, ladder, offset)` —
   the same bucket the NS step's bucket-escalation logic would land
   on. Both resolution paths now pass `ladder = backend.max_neighbors_list`
   and `offset = backend.max_neighbors_offset`. The initial-energy
   compile therefore lands in the same JIT cache slot the burn-in and
   the NS step will hit, instead of wasting a compile at
   `max_neighbors=0`. Backends without `max_neighbors_for` (LJ, toy,
   jax-md) pass `max_neighbors=0` straight through; the bucket arg is
   inert for them.
2. **`batcher.wrap_for_batch(per_replica_fn)` dispatch.** The same
   helper body works for all topologies because the `batcher` argument
   chooses `jax.jit` / `jax.jit(vmap)` / `pmap(vmap)` underneath.
   Single-replica path passes `SingleRun()`; the multi-replica path
   passes the same `PmapVmapRuns(G, P)` instance it stores on
   `ResolvedConfig.batcher` — so the resolver, burn-in, and NS step
   all dispatch through one shared batcher instance.
3. **Per-replica pressure under one wrapper.** The multi-replica path
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

## The unified ResolvedConfig

`ResolvedConfig` is a single frozen dataclass for both single- and
multi-replica runs. The shape depends on the batcher:

| Field | `batcher = SingleRun()` | `batcher = PmapVmapRuns(G, P)` |
|---|---|---|
| `base_backend` | unwrapped backend (e.g. raw MACE / LJ instance) | unwrapped backend |
| `ensemble_params_per_run` | length-1 tuple | length-`n_total` tuple, flat order `g * P + p` |
| `init.initial_positions` shape | `(n_live, A, 3)` | `(n_total, n_live, A, 3)` |
| `init.initial_energies` shape | `(n_live,)` | `(n_total, n_live)` |
| `batcher` | `SingleRun()` | `PmapVmapRuns(n_gpu, n_per_gpu)` |

`base_backend` is always the unwrapped backend. The runtime wraps it
into an `EnsembleBackend` if needed; the resolver applies the same
wrapping locally only to compute initial energies on the right scale,
then discards the wrapper. This means a multi-replica run can serve
8 different pressures from one `EnsembleBackend(pressure=0.0)`
instance, with each replica's pressure flowing in through
`ensemble_params` at call time — no per-replica backend rebuild.

The CLI dispatches on `isinstance(resolved.batcher, SingleRun)`:
single → `run_from_config`, multi → `run_multi_gpu_from_config`.
That's the only place the two paths diverge after the resolver.

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
  re_interval: 5
```

Step by step:

1. **Load & validate**: `cli/cli.py::_load_and_validate` reads the
   YAML and calls `RootSpec.model_validate`. pydantic runs all
   validators; errors surface here with line-accurate messages (and
   "did you mean" suggestions for typos).
2. **Interval-unit scaling (CLI)**: `_cmd_run` calls
   `_apply_interval_units(root)` and flips `interval_units` to
   `"absolute"`, so the logged "Parsed configuration" block already
   shows resolved iteration counts.
3. **Unit conversion**: `NPTEnsembleSpec._pressure_list()` returns
   `[1.0, …, 8.0]`. When called via `to_ensemble_params(i)` with
   `pressure_units="gpa"`, each scalar is multiplied by
   `_GPA_TO_EVA3` (schema-layer concern).
4. **Resolve**: `_cmd_run` calls `resolve(root)`. The resolver
   computes `n_total = 8` from the pressure list, reads
   `len(jax.local_devices()) = 4` on a `--gres=gpu:4` SLURM
   allocation, returns `(n_gpu=4, n_per_gpu=2)`, and dispatches to
   `_resolve_multi_replica`. Divisibility is checked here.
5. **Per-replica structural init**: for each of 8 replicas,
   `_resolve_init` builds positions / cells / types only. A
   per-replica `EnsembleBackend(base, pressure=P_r)` is constructed
   for the rejection-mode ceiling check inside
   `_sample_per_walker_positions`, but no initial energies are
   computed yet. Per-replica arrays are stacked along axis 0 into
   `(n_total, K, …)`.
6. **Consolidated finalize**: `_resolve_multi_replica` reshapes to
   `(G, P, K, …)`, wraps the base backend in *one*
   `EnsembleBackend(pressure=0.0)`, and calls
   `_finalise_initial_energies_and_counts` with
   `batcher=PmapVmapRuns(G, P)`, the per-replica pressures fed
   through `ensemble_params={"pressure": P_r}` on the vmap axis, and
   `ladder` / `offset` from the backend config. One compile across
   all 4 GPUs at the bucket the NS step will use.
7. **Runtime handoff**: `ResolvedConfig` is returned to `_cmd_run`,
   which inspects `resolved.batcher` and calls
   `cli/run.py::run_multi_gpu_from_config` (the `PmapVmapRuns`
   branch). From here on, no schema, no resolver — just JAX arrays
   and dataclasses flowing through `run_ns_multi_gpu` with its
   `pmap(vmap(ns_step))` dispatch.

For a scalar-pressure version of the same config, steps 4–7 collapse
to: `n_total = 1`, batcher is `SingleRun()`, resolver calls
`_resolve_single_replica`, CLI calls `run_from_config`. The
`ResolvedConfig` type is identical; only the array shapes and the
batcher differ.

## Why this structure

Three properties the three-layer split buys you:

1. **Early, single-point validation.** Everything you can check
   without running JAX is checked in `jaxrens validate --parse-only`.
   You don't lose 10 minutes of GPU time to a typo like `n_liv: 500` —
   the schema's `extra="forbid"` rejects it instantly, and the CLI
   formats the error with a "did you mean" suggestion.
2. **Independent evolution of the YAML surface and the core.** Add
   a new YAML shortcut in the schema without touching the core;
   rename a `NSConfig` field in the core without breaking YAML
   compatibility (the resolver adapts).
3. **Programmatic entry point.** Nothing forces you through the
   YAML layer. Build a `RootSpec` by hand in Python, call `resolve`,
   then `run_from_config` / `run_multi_gpu_from_config` directly —
   exactly what `tests/test_cli_multi_run.py` does.

## Where this lives in the code

| Layer | File(s) |
|---|---|
| Schema | `src/jaxrens/cli/schema/` (10 spec modules) + `cli/schema/__init__.py` re-export |
| Resolver | `src/jaxrens/cli/resolve.py` (~1500 LoC, one concern per function) |
| Single-replica runtime entry | `src/jaxrens/cli/run.py::run_from_config` |
| Multi-replica runtime entry | `src/jaxrens/cli/run.py::run_multi_gpu_from_config` |
| CLI dispatch | `src/jaxrens/cli/cli.py::_cmd_run` |
| Library core | `src/jaxrens/{sampling,state,backends,init,io,postprocess}/` |
