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
| "Compute initial walker energies at the right ensemble scale" | **Resolver** | `_resolve_init` + `EnsembleBackend` wrap |
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
5. **Per-replica init**: for each of 8 replicas, `_resolve_init`
   builds positions / cells / energies using a dedicated
   `EnsembleBackend(base, pressure=P_r)`, so initial energies
   already include the correct $+P·V$. Per-replica arrays are
   stacked along axis 0.
6. **Runtime handoff**: `ResolvedMultiRunConfig` is returned to
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
