# CLI reference

`jaxrens` is installed as a console script (`jaxrens`) and as
`python -m jaxrens.cli.cli`. Four subcommands are available:
`run`, `validate`, `dump-schema`, and `plot`.

Before the per-command details: every CLI invocation flows through
the **schema → resolve → run** pipeline — pydantic validates the
YAML, the resolver builds runtime objects (backend instance, walker
positions, replica topology from `jax.local_devices()`), then the
library core runs NS. That pipeline is documented in
{doc}`../user/concepts/schema_resolve` with diagrams and a
concrete trace; this page focuses on the subcommand surface itself.

## `jaxrens validate`

Schema-check a YAML config and print a one-screen summary.
Resolves the full config (backend construction, replica
derivation, initial walker positions) without running NS — a
useful sanity check before submitting a long job.

```bash
jaxrens validate -c config.yaml
jaxrens validate -c config.yaml --set run.n_live=64
jaxrens validate -c config.yaml --parse-only
```

Pass `--parse-only` to stop after pydantic schema validation, skipping
the resolver entirely (no structure-file read, no backend build, no
walker placement). It is fast — for catching typos and wrong field
names without paying for heavy-backend initialization.

Exit code is zero on success. For a multi-run config it prints the
derived topology:

```
OK — multi-run dispatch
  topology: n_gpu=4 × n_per_gpu=2 = 8 replica(s)
  run:     n_live=64, max_iterations=1000
  moves:   4 move(s) [galilean, volume, shear, stretch]
  backend: mace, n_atoms=40
  output:  format=extxyz, prefix=mace_srtio3
```

## `jaxrens run`

Run NS with the given config. Dispatches to single-run
(`run_ns`) or multi-GPU (`run_ns_multi_gpu`) automatically based on
the replica axis implied by the config.

```bash
jaxrens run -c config.yaml
jaxrens run -c config.yaml --set run.max_iterations=5000 \
                          --set run.seed=7
```

All fields accept `--set key.path=value` overrides; nested paths use
dots (`run.n_live`, `ensemble.pressure`, `inter_re.re_interval`). Lists
and nested structures are YAML-parsed, so `--set
'ensemble.pressure=[0.5, 1.0, 2.0]'` works.

Output artifacts land under `output.working_dir` (default `./`)
with the prefix from `output.out_file_prefix`.

Additional flags guard the run and control restart behaviour:

| Flag | Effect |
|---|---|
| `--n-gpus N` | Assert JAX sees exactly `N` local GPU devices; exit non-zero on mismatch. Typically passed from SLURM as `--n-gpus $SLURM_GPUS_ON_NODE` so jobs fail fast when the scheduler silently downgrades the allocation. |
| `--force` | Delete pre-existing artifacts in `working_dir` matching `out_file_prefix` (`.energies`, `.traj.*`, `.adaptation.h5`, `.checkpoint.h5`, …) before starting. Without it, the run aborts if any such file exists, to prevent silent overwrite/append corruption. |
| `--resume` | Resume by auto-discovering a checkpoint in `working_dir` (prefers `<prefix>.final.checkpoint.h5`, falls back to `<prefix>.checkpoint.h5`; mtime tie-break). Skips the output-dir gate and switches loggers to append mode. Mutually exclusive with `--force` and with `init.restart_file` in the YAML. |

```bash
jaxrens run -c config.yaml --n-gpus $SLURM_GPUS_ON_NODE
jaxrens run -c config.yaml --resume
jaxrens run -c config.yaml --force
```

## `jaxrens dump-schema`

Emit the JSON schema for `RootConfig`. Useful for editor
autocomplete (point your YAML LSP at the generated schema).

```bash
jaxrens dump-schema --format json > jaxrens.schema.json
```

## `jaxrens plot`

Render a quick-look PNG from a single run artefact. The kind of plot is
auto-detected from the filename suffix, so you just point it at a file:

```bash
jaxrens plot output/run.adaptation.h5
jaxrens plot output/run.energies -o energies.png
```

By default the PNG is written next to the input as
`<stem>.<kind>.png`; override with `-o/--output`. Four artefact kinds
are recognised:

| Suffix | Plot |
|---|---|
| `.adaptation.h5` | 2-panel step-size + acceptance-rate trace (mean ± std across replicas) |
| `.re_stats.h5` | swap acceptance per adjacent replica pair vs iteration |
| `.max_neighbors.h5` | 2-panel: per-walker neighbor-count percentiles (top) + distribution heatmap (bottom), with bucket overlay |
| `.energies` | dead-point energy trail (and volume, if present) |

This is the "quick look at one file" utility. For full multi-run
cohort analysis, use `MonitorCollection.from_multi_run_directory` and
the methods on the collection (see {doc}`index`).

## The YAML surface

Every YAML config has ten top-level sections. Not all are required
— most have sensible defaults.

| Section | Purpose | Cardinality |
|---|---|---|
| `run` | NS parameters (n_live, max_iter, seed, …) | required |
| `moves` | ordered list of MCMC kernels | required |
| `backend` | energy model (lj/mace/neuralil/harmonic/…) | required |
| `output` | file format, intervals, working dir | required |
| `ensemble` | NVT / NPT + optional pressure list | optional |
| `inter_re` | replica-exchange flavor (pressure/xrens/semi_grand) | optional |
| `adaptation` | step-size bisection settings | optional |
| `termination` | list of stopping criteria | optional |
| `init` | starting walkers (species / config file / restart) | optional |
| `cell` | cell-shape / volume constraints | optional |

For the full schema — every field of every section, with its type,
default, and constraints — see the {doc}`config`. For a
machine-readable version (e.g. editor autocomplete), use the JSON dump
(`jaxrens dump-schema`).
