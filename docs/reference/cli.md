# CLI reference

`jaxrens` is installed as a console script (`jaxrens`) and as
`python -m jaxrens.cli.cli`. Four subcommands are available.

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
```

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

## `jaxrens dump-schema`

Emit the JSON schema for `RootConfig`. Useful for editor
autocomplete (point your YAML LSP at the generated schema).

```bash
jaxrens dump-schema --format json > jaxrens.schema.json
```

## `jaxrens migrate-ns-inp`

Convert a legacy pymatnest / jaxnest `ns.inp` key=value file into a
jaxrens YAML config.

```bash
jaxrens migrate-ns-inp -i old_run/ns.inp -o new_run/config.yaml
jaxrens migrate-ns-inp -i ns.inp -o - --validate
```

Passing `--validate` round-trips the generated YAML through the
schema to confirm the migration produced a usable config.
Unknown/unsupported keys emit warnings but do not fail; check the
output log to see what was dropped.

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
