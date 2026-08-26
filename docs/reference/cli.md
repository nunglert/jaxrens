# CLI reference

`jaxrens` is installed as a console script (`jaxrens`) and as
`python -m jaxrens.cli.cli`. Five subcommands are available:
`run`, `validate`, `dump-schema`, `plot`, and `analyze`.

Before the per-command details: every CLI invocation flows through
the **schema → resolve → run** pipeline — pydantic validates the
YAML, the resolver builds runtime objects (backend instance, walker
positions, replica topology from `jax.local_devices()`), then the
library core runs NS. That pipeline is documented in
{doc}`../user/concepts/schema_resolve` with diagrams and a
concrete trace; this page focuses on the subcommand surface itself.

## `jaxrens validate`

Check a YAML config and print a one-screen summary, without running NS.
Validation comes in three tiers, because the checks differ enormously in
cost — the expensive ones build an energy model and compile a kernel that
validation never calls.

```bash
jaxrens validate -c config.yaml --parse-only   # schema only
jaxrens validate -c config.yaml                # + resolver plan (default)
jaxrens validate -c config.yaml --full         # + startup rehearsal
```

| Tier | Checks | Cost |
|---|---|---|
| `--parse-only` | pydantic only: field names, types, ranges, cross-field validators | milliseconds |
| *(default)* | the above, plus the **resolver plan** — replica topology and divisibility, interval-unit scaling, `shard_n_gpu` compatibility, cell-prior geometry bounds, and that every path the config names exists and is readable | well under a second |
| `--full` | the above, plus **startup rehearsal** — builds the backend, places the walker population, evaluates its initial energies | seconds (toy backends) to minutes (MLIPs) |

The default tier deliberately stops short of loading anything. It ends
with a `skipped` line naming what it did not check, so a pass is never
mistaken for a full rehearsal:

```
✓ OK — configuration plan valid
  topology  SingleRun (1 replica, 1 GPU)
  run       n_live=128, max_iterations=2000
  moves     4 move(s) [gmc, volume, shear, stretch]
  backend   lj, n_atoms=64
  output    format=extxyz, prefix=lj8_npt
  skipped   backend build, walker placement, initial energies — rerun with --full to check those
```

`n_atoms` is derived from `init.start_species` (or a single read of
`init.start_config_file`), not from placed walkers. A config initialised
from a walker set or a restart file carries its atom count inside the
data, so the plan tier omits the field rather than guessing.

Use `--full` before submitting a long job: it is the tier that proves the
checkpoint actually loads and that a valid initial configuration exists.

```bash
jaxrens validate -c config.yaml --set run.n_live=64
```

Exit code is zero on success. For a multi-run config it prints the
derived topology:

```
OK — multi-run dispatch
  topology: n_gpu=4 × n_per_gpu=2 = 8 replica(s)
  run:     n_live=64, max_iterations=1000
  moves:   4 move(s) [gmc, volume, shear, stretch]
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

Emit the JSON schema for {class}`~jaxrens.cli.schema.root.RootSpec`,
the top-level config model. Useful for editor
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

## `jaxrens analyze`

Turn a run's dead-point energy ladder into a thermodynamic observable —
heat capacity, log partition function, or free energy — over a
temperature sweep. Unlike `plot`, which dispatches on a single
self-contained artefact, `analyze` dispatches on a **checkpoint** file
and pulls the sibling `.energies` log from the same directory (via
`Monitor.from_directory`), since the live-walker energies and the
dead-point history live in different files.

```bash
jaxrens analyze output/run.final.checkpoint.h5 --t-min 0.05 --t-max 3.0
```

The primary output is data, not a picture — `T` against the observable,
ready for a notebook or another run's comparison without a detour
through a PNG:

```text
Wrote output/run.heat_capacity.csv
```

```text
               T,              Cv
            0.05,      0.44239143
     0.064824121,      0.72091224
...
```

| Flag | Effect |
|---|---|
| `CHECKPOINT` (positional) | `<prefix>.checkpoint.h5` or `<prefix>.final.checkpoint.h5`. |
| `--observable` | `heat_capacity` (default), `partition_function`, or `free_energy`. |
| `--t-min` / `--t-max` | Temperature sweep bounds, in the run's energy units divided by `--k-b`. **Required, no default** — the right scale depends on the backend's energy units, so the CLI asks rather than guesses. |
| `--n-t` | Number of temperature points (default 200). |
| `--k-b` | Boltzmann constant in energy-units-per-T, e.g. `8.617e-5` (eV/K) for a MACE run reporting eV so `T` reads in Kelvin. Default `1.0` (reduced units). |
| `--format` | `csv` (default) — fixed-width, right-aligned columns, meant to be opened and actually read, not just parsed; scalar observables only. `json` — self-describing (`observable`, `column`, `prefix`, `k_b`, `T`, `<column>` keys), and nests whatever shape the observable has, so it also covers a future non-scalar observable CSV's fixed columns can't. |
| `-o`/`--output` | Data-file path. Default: sibling `<prefix>.<observable>.{csv,json}`. |
| `--plot` | Also render a PNG of the same data, via the same `plot_*` functions `jaxrens plot` uses elsewhere. |
| `--plot-output` | PNG path when `--plot` is set. Default: sibling `<prefix>.<observable>.png`. |

See {doc}`../tutorials/02_lj_cluster` for a worked example, including the
`--format json` output shape.

## The YAML surface

Every YAML config has eleven top-level sections plus one scalar key.
Not all are required — most have sensible defaults.

| Section | Purpose | Cardinality |
|---|---|---|
| `run` | NS parameters (n_live, max_iter, seed, …) | required |
| `moves` | ordered list of MCMC kernels | required |
| `backend` | energy model (lj/mace/neuralil/nequix/jaxmd/toy) | required |
| `output` | file format, intervals, working dir | required |
| `ensemble` | NVT / NPT / semi-grand + optional pressure list | optional |
| `inter_re` | replica-exchange flavor (pressure/xrens/semi_grand) | optional |
| `adaptation` | step-size adaptation policy + per-move overrides | optional |
| `termination` | list of stopping criteria | optional |
| `init` | starting walkers (species / config file / restart) | optional |
| `cell` | cell-shape / volume constraints | optional |
| `constraints` | hard configuration constraints (e.g. minimum distance) | optional |
| `interval_units` | scalar: `absolute` (default) or `per_walker` — rescales every iteration-counted field at once | optional |

For the full schema — every field of every section, with its type,
default, and constraints — see the {doc}`config`. For a
machine-readable version (e.g. editor autocomplete), use the JSON dump
(`jaxrens dump-schema`).
