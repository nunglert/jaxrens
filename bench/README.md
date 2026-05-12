# `ns_step` performance benchmark

Append-only wall-time log for the inner NS step, one row per backend per
invocation.

This is **not** a regression test: there are no thresholds and no
assertions.  It exists so that after a major refactor you can re-run the
suite and skim `view.py` to see whether the headline timings shifted.

Three entry-points:
- **Local** — `python bench/run.py`, results land in `bench/results.jsonl`
  (gitignored), use `view.py` for the trend.  "One GPU process at a time"
  on the local cluster — never run two of these in parallel.
- **SLURM** — `sbatch bench/submit.slurm` requests a single GPU and runs
  the same orchestrator from the repo root.
- **CI on PRs** — `.github/workflows/bench.yml` runs the bench on PR head
  and on the PR's base SHA on a single ephemeral g5.xlarge runner and
  posts a delta table as a PR comment via `bench/compare.py`.  No
  pass/fail; informational only.

## Run

```bash
# All backends with a config in bench/configs/, one subprocess each:
python bench/run.py

# Just one backend:
python bench/run.py --backend lj

# Longer / shorter steady-state window (default 50 steps):
python bench/run.py --n-timed-steps 100
```

Backends with missing optional deps (`mace_jax`, `neuralil`, `nequix`,
`jax_md`) or missing model fixtures are skipped silently — `run.py`
prints `[skipped]` and continues.

## View

```bash
python bench/view.py                       # whole log, table
python bench/view.py --backend jaxmd       # one backend
python bench/view.py --last 20             # last 20 rows after filtering
python bench/view.py --plot                # matplotlib trend (warm.mean)
python bench/view.py --plot --metric cold  # plot cold compile time instead
```

The table columns: `timestamp | sha[*] | backend | n_live | t_cold[s] |
t_warm.mean[ms] | t_warm.p95[ms] | hardware`.  A `*` next to the SHA
means there were uncommitted changes when the row was recorded.

## Files

| Path | Purpose |
|------|---------|
| `_one.py` | Single-backend runner: load → resolve → init → cold call → warm loop → append row.  Exits 77 if the backend's optional dep / fixture is missing. |
| `run.py`  | Orchestrator: spawns one `_one.py` subprocess per backend, sequentially.  Subprocess isolation keeps each backend's cold-compile time uncontaminated by previous backends' JAX startup amortisation. |
| `view.py` | Reads `results.jsonl`, prints table or plots trend. |
| `compare.py` | Diffs two `results.jsonl` files (typically PR head vs base SHA) and prints a markdown table for the GHA PR-comment workflow.  Stdlib-only. |
| `submit.slurm` | SLURM batch wrapper for the local cluster.  Self-locates to the repo root and forwards CLI args to `run.py`. |
| `configs/` | One YAML per backend at bench scale (n_live=200 for cheap, n_live=32 for heavy NN backends).  These are *not* the production `experiments/*/config.yaml` files — bench scales and bench-only output settings (no I/O). |
| `results.jsonl` | Gitignored; one JSON object per line. |
| `results.jsonl.example` | Committed; one hand-crafted row that documents the schema. |

## Schema (one row, one JSON object)

```json
{
  "timestamp_utc": "2026-05-12T13:42:01Z",
  "git_sha": "abc1234",
  "git_dirty": false,
  "backend": "jaxmd",
  "config_file": "bench/configs/jaxmd.yaml",
  "scale": {
    "n_live": 200, "n_atoms": 16,
    "n_mcmc_steps": 20, "n_extra": 49,
    "n_moves": 4, "starting_bucket": 30,
    "pressure_eva3": 0.00624
  },
  "n_warmup_steps": 1,
  "n_timed_steps": 50,
  "t_setup_s": 4.21,
  "t_cold_compile_s": 8.74,
  "t_warm_per_step_s": {
    "mean": 0.0312, "std": 0.0008,
    "p50": 0.0310, "p95": 0.0331, "min": 0.0301
  },
  "env": {
    "jax_version": "0.4.35",
    "jaxlib_version": "0.4.35",
    "jaxrens_version": "0.1.0",
    "n_devices": 4,
    "device_kind": "NVIDIA A100-SXM4-40GB",
    "platform": "gpu",
    "hostname": "node12.cluster"
  }
}
```

The two headline metrics are:

- **`t_cold_compile_s`** — wall time of the *first* jitted `ns_step`
  call.  Dominated by JIT compile.  Goes up if a refactor broke a JIT
  seam or added a retrace.
- **`t_warm_per_step_s.mean`** — mean wall time of the next
  `n_timed_steps` calls.  Steady-state.  Goes up if a refactor added
  ops to the scan body or changed the algorithmic cost.

`t_setup_s` (resolver + init + finalize) is recorded for context but is
not the main signal — it lumps disk I/O, structural init, and the
finalize compile that lives in a different JIT slot.

## Cross-machine comparison

The bench is per-machine.  `results.jsonl` is gitignored; do not commit.
Each row carries an `env` block (`device_kind`, `n_devices`,
`jax_version`, `hostname`) so when sharing numbers by hand, filter on
matching hardware.  A run on a different GPU model is generally not
directly comparable.

## What this does *not* cover

- Per-move kernel microbenchmarks (no `bench/moves/`).
- Multi-GPU / multi-replica timings (only `SingleRun` is exercised).
- Pass/fail regression assertions.
- JIT-trace-count or jaxpr-op-count tracking (deterministic alternative
  to wall time; not bundled — open as a follow-up if the wall-time noise
  ever becomes a problem in practice).
