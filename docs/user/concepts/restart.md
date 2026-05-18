# Restart, resume, and the output-dir lifecycle

A jaxrens run writes a fixed set of artifacts into one
`output.working_dir`: `*.energies`, `*.traj.*`, `*.adaptation.h5`,
`*.checkpoint.h5`, and friends. Two distinct concerns live on top of
that directory:

1. **Avoiding accidental clobber** — re-running the same config
   against the same output dir should never silently destroy or corrupt
   the prior run's data.
2. **Resuming a prior run** — from a crash, from a clean completion,
   or to retune the MCMC, sometimes a new invocation legitimately needs
   to continue an earlier output dir.

These are opposite ends of the same lifecycle, so jaxrens treats them
with a single decision: **is this a restart, or is it fresh?**

```
restart_intent  =  (init.restart_file is set in YAML)
                OR (--resume passed on the CLI)
```

The predicate is computed once in `_cmd_run` before any I/O, and it
drives the rest of the lifecycle: whether the output-dir gate fires,
which mode the writers open in, and whether the compatibility validator
runs.

## Three start modes

| Mode | Trigger | Gate | Writer mode | Walker state from |
|---|---|---|---|---|
| Fresh | nothing set | enforce clean dir | `"w"` (truncate) | `init.start_*` |
| Fresh + force | `--force` | wipe artifacts | `"w"` | `init.start_*` |
| Explicit restart | `init.restart_file: <path>` in YAML | skip | `"a"` (append) | the checkpoint at `<path>` |
| Auto restart | `--resume` | skip | `"a"` | auto-discovered checkpoint in `working_dir` |

`--force` is fresh-only. The combinations `--force + --resume`,
`--force + init.restart_file`, and `--resume + init.restart_file` are
mutually exclusive and rejected at argparse with `SystemExit(2)`.

## The output-dir gate

For fresh runs only. Scans `working_dir` for any of these patterns:

```
{prefix}.energies                   {prefix}.run*.energies
{prefix}.traj.*                     {prefix}.run*.traj.*
{prefix}.adaptation.h5              {prefix}.re_stats.h5
{prefix}.max_neighbors.h5           {prefix}.acc_rates.h5
{prefix}.checkpoint.h5              {prefix}.initial.checkpoint.h5
{prefix}.final.checkpoint.h5        {prefix}.config.snapshot.yaml
```

If any match exists, the run aborts with a message listing the first
ten offending files. Pass `--force` to delete them and proceed.

Code: {mod}`jaxrens.cli.output_gate`, `enforce_clean_output_dir`.

The glob set is the catalogue of every file a run writes into
`working_dir`. Log files (`{prefix}.log`, `{prefix}.debug.log`) live in
the *parent* directory and are out of scope. Per-walker snapshot files
(`{prefix}.traj.snap.NNNN.extxyz`) are included via the `.traj.*` glob
— without that, `--force` would leave stale snapshots from a longer
prior run silently mixing iteration ranges into a re-run.

## Auto-discovery (`--resume`)

When `--resume` is passed, jaxrens scans `working_dir` for a checkpoint
matching the configured prefix. The rule:

1. Look for `{prefix}.final.checkpoint.h5` and `{prefix}.checkpoint.h5`.
2. If neither exists → abort with a listing of expected names.
3. If exactly one exists → use it.
4. If both exist → use whichever has the higher mtime.
5. On exact mtime tie → `.final.checkpoint.h5` wins (a clean termination
   beats the last rolling write that produced it).

The chosen path is logged so the audit trail records which file drove
the restart:

```
[--resume] selected ns.final.checkpoint.h5 (mtime=…); also found
           ns.checkpoint.h5 (mtime=…)
```

`{prefix}.initial.checkpoint.h5` is **not** a discovery candidate —
it's written once at run start, before any iterations have run, and
restarting from it would just redo the work.

Code: `discover_checkpoint` in {mod}`jaxrens.cli.output_gate`.

## The config snapshot

Every fresh run writes `{prefix}.config.snapshot.yaml` into
`working_dir` immediately after the gate clears. The payload is exactly
`root.model_dump(mode="json")` — the same validated config that goes
into the run log, but in a structured file the restart validator can
read back.

When a later invocation restarts (either via `--resume` or via
`init.restart_file: <path>`), the snapshot located beside the
checkpoint is the source of truth for "what physics did the original
run use?". Without it, the validator can only do the shape checks the
resolver does anyway.

```
output/
├── ns.config.snapshot.yaml      ← written on fresh start
├── ns.checkpoint.h5             ← rolling, updated mid-run
├── ns.final.checkpoint.h5       ← written on clean termination
├── ns.energies
└── ns.traj.extxyz
```

If you restart from a checkpoint produced by a pre-2026-05-18 jaxrens
(no snapshot beside it), the validator logs a warning and proceeds with
shape checks only.

## The compatibility validator (strict)

When `restart_intent` is true, after the resolver has built the
runtime config but before any callback wiring,
`validate_restart_compatibility` diffs the snapshot against the current
run's config. Each field falls into one of three categories.

### Hard refuse — immutable across restart

A mismatch on any of these is a `SystemExit(2)` with a per-field diff:

| Field | Why it must match |
|---|---|
| `run.n_live` | walker-population shape; NS estimator depends on constant `n_live` |
| `run.n_gpu`, `run.n_per_gpu` | `pmap × vmap` topology; checkpoint pytree shapes won't match otherwise |
| `backend` subtree | energy function identity — including model file paths for ML backends |
| `ensemble` subtree | pressure, temperature, volume limits — physical meaning of the trace |
| `inter_re` subtree | replica-exchange flavor + per-replica conditions (pressures, μ) |
| `cell` subtree | PBC flags, ndim — periodic-volume vs free-space |

The refusal message looks like:

```
jaxrens run: restart from output/ns.final.checkpoint.h5 is incompatible
with the current config on immutable fields:
  run.n_live   snapshot=64    current=128
  ensemble     snapshot={'type': 'npt', 'pressure': 0.001}  current={'type': 'npt', 'pressure': 0.005}
  backend      snapshot={'type': 'lj', ...}  current={'type': 'neuralil', ...}
  snapshot: output/ns.config.snapshot.yaml
Use a different working_dir for this configuration, or revert the
conflicting fields.
Aborting.
```

No override flag. If you genuinely want to start from these positions
under different physics, point at a new output dir — the checkpoint can
be read from anywhere, the gate only protects the *write* side.

### Soft warn — mutable but worth flagging

A change here logs a warning and proceeds:

- `moves` — added/removed kernels, retuned step sizes/weights, different
  target acceptance.
- `run.n_mcmc_steps`, `run.max_iterations`
- `termination`, `adaptation` subtrees
- `output.traj_interval`, `output.snapshot_interval`,
  `output.checkpoint_interval`, `output.info_interval`, `output.format`
- `interval_units`

### Unchanged-seed warning

`run.seed` is normally mutable, but a restart that keeps the seed
identical produces a continuation using the same PRNG stream from this
point — statistically useless as a second ensemble member. The
validator emits a single warning when this happens:

```
[restart] run.seed (1) is unchanged from the original run.
Bump the seed if you intend the restart as an independent ensemble member.
```

Code: {mod}`jaxrens.cli.restart_validate`.

## Writer behavior across restart

All seven I/O writers honor a `mode: Literal["w", "a"]` constructor
parameter set once per run by the dispatcher.

| Writer | `mode="w"` (fresh) | `mode="a"` (restart) |
|---|---|---|
| `EnergyLogger` (`.energies`) | truncate, write header | open in append mode, header skipped if file non-empty |
| `ExtxyzTrajectoryWriter` (`.traj.extxyz`) | overwrite on first frame, then append | append from the first frame |
| `H5TrajectoryWriter` (`.traj.h5`) | `h5py.File(mode="w")` | `h5py.File(mode="a")` |
| `AdaptationLogger` (`.adaptation.h5`) | first flush truncates, subsequent flushes append | every flush appends |
| `RELogger` (`.re_stats.h5`) | same | same |
| `MaxNeighborsLogger` (`.max_neighbors.h5`) | same | same |
| `AccRatesLogger` (`.acc_rates.h5`) | same | same |

On a restart, each file ends up containing the prior run's
0..N<sub>checkpoint</sub> rows followed by the new segment's
N<sub>checkpoint</sub>..N<sub>final</sub> rows — one continuous trace
per file, indistinguishable from a single long run aside from the
iteration index at the boundary.

## What you cannot do across restart

The strict policy refuses these by design:

- **Resize the walker population** — `n_live`, `n_gpu`, `n_per_gpu`
  immutable. The NS evidence estimator assumes constant `n_live` across
  iterations; changing it silently invalidates `log_Z`.
- **Swap the energy function** — `backend` immutable. The energy scale
  shifts, the contour history becomes meaningless, the appended trace
  is garbage.
- **Change physical conditions** — `pressure`, `chemical_potentials`,
  `temperature` immutable. A trace spanning two conditions cannot be
  post-processed as one run; per-segment analysis is what you actually
  want, and that means separate output dirs.
- **Change the cell topology** — PBC flags / ndim immutable. NPT volume
  moves are meaningless on a non-PBC checkpoint and vice versa.

For all of these the workaround is the same: **set
`output.working_dir` to a new path and load the checkpoint via
`init.restart_file`**. The new dir is empty so the gate passes, the
new run produces fresh files, and the original artifacts are
preserved.

## Common scenarios

### Crash recovery

```yaml
# config.yaml — unchanged from the original run
init:
  start_species: "18 64"
output:
  working_dir: ./output
  out_file_prefix: ns
```

```bash
jaxrens run -c config.yaml --resume
```

The checkpoint is auto-discovered. Validator passes (config identical),
emits one warning about the unchanged seed. Writers append.

### Extending a completed run

The previous run terminated cleanly on `iteration: 10000`. You want
20000.

```yaml
run:
  seed: 7              # bump from the original to suppress the seed warning
termination:
  - type: iteration
    max_iterations: 20000
```

```bash
jaxrens run -c config.yaml --resume
```

Picks `ns.final.checkpoint.h5`. Validator warns
`termination.max_iterations changed`, proceeds. The next 10000
iterations append into the same files.

### Retuning MCMC mid-run

Acceptance was bad; tighten step sizes, drop a problematic kernel.

```yaml
moves:
  - type: random_walk
    step_size: 0.05
    weight: 1.0
  # galilean removed
```

```bash
jaxrens run -c config.yaml --resume
```

Validator warns `moves changed`, proceeds. New segment uses the
retuned MCMC; the adaptation log gains a new row when adaptation fires.

### Branching to a new output dir

Take a checkpoint, fork three independent continuations.

```yaml
# branch_seed_42.yaml
run:
  seed: 42
output:
  working_dir: ./output_branch_42
  out_file_prefix: ns
init:
  restart_file: ./output/ns.final.checkpoint.h5
```

```bash
jaxrens run -c branch_seed_42.yaml
```

The new working_dir is empty so the gate passes; `init.restart_file`
triggers the restart branch (validator runs on the snapshot beside the
original checkpoint). Writers truncate `output_branch_42/*`; the
checkpoint is read from the *original* dir, not touched.

### Fresh re-run, accepting that the previous output is gone

```bash
jaxrens run -c config.yaml --force
```

The gate finds the old artifacts and deletes them; writers truncate;
the run starts from `init.start_*` as if the previous dir never
existed.

## Code map

| Concern | File | Symbol |
|---|---|---|
| Output-dir gate, artifact globs | {mod}`jaxrens.cli.output_gate` | `enforce_clean_output_dir` |
| `--resume` auto-discovery | {mod}`jaxrens.cli.output_gate` | `discover_checkpoint` |
| Config snapshot read/write | {mod}`jaxrens.cli.output_gate` | `write_config_snapshot`, `read_config_snapshot`, `snapshot_path_for_checkpoint` |
| Strict compatibility validator | {mod}`jaxrens.cli.restart_validate` | `validate_restart_compatibility` |
| Mode plumbing into the run | {mod}`jaxrens.cli.cli` (`_cmd_run`) → {mod}`jaxrens.cli.run` (`run_*_from_config`) | `writer_mode` parameter |
| Walker-state loader for Mode D | {mod}`jaxrens.init.restart` | `load_restart` |
| Resolver branch for restart | {mod}`jaxrens.cli.resolve` | `_resolve_init_restart` |
