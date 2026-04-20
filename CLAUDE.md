# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session continuity

Read `jaxrens/WORKLOG.md` at the start of every session — it is the short record of what landed recently and what's probably next.

Update it **mid-session, not at the end** — sessions often close without warning, so end-of-session updates get lost. Whenever a meaningful chunk of work lands (feature complete, bug fixed, design decision made, refactor finished), append a bullet to today's `## YYYY-MM-DD` heading *before* reporting back to the user. Create the heading if today has no entry yet. Keep a `**Next:**` bullet at the bottom of today's block; overwrite it as priorities shift so the most recent one wins.

## Workspace layout

This directory is a research workspace, not a single package:

- `jaxrens/` — **the active codebase**. JAX-based nested sampling for atomistic systems. All new feature work happens here.
- `jaxns-devAS/` — the legacy `jaxnest` codebase being refactored. Read-only reference for behavior parity; do not edit unless explicitly asked.
- `reference_codes/` — nine external scientific JAX repos (BlackJAX, JAX-MD, JAXNS, MACE-JAX, mlff, mlip, NetKet, NeuralIL, so3lr) cloned for architectural inspiration.
- `experiments/` — scratch scripts, fixtures, design notes.
- `fwf_proposal.pdf` — the NS-SMC grant proposal motivating upcoming features.

## Running things

The project uses a conda env. Always invoke the env's python explicitly:

```bash
# Interpreter:
/home/nunglert/miniconda3/envs/jaxrens/bin/python

# Full test suite (from jaxrens/):
cd jaxrens && /home/nunglert/miniconda3/envs/jaxrens/bin/python -m pytest tests/ -v

# Single test file / single test:
/home/nunglert/miniconda3/envs/jaxrens/bin/python -m pytest tests/test_cell_moves.py -v
/home/nunglert/miniconda3/envs/jaxrens/bin/python -m pytest tests/test_cell_moves.py::test_volume_move_jit -v

# Heavy tests are excluded by default. To include them:
/home/nunglert/miniconda3/envs/jaxrens/bin/python -m pytest tests/ -m heavy
/home/nunglert/miniconda3/envs/jaxrens/bin/python -m pytest tests/ -m ""    # all markers
```

Run tests scoped to the edit, not the full suite, unless asked otherwise.

CLI entry point (installed as `jaxrens` console script; also `python -m jaxrens.cli.cli`):

```bash
jaxrens validate -c config.yaml          # schema-check a YAML config
jaxrens run      -c config.yaml          # run NS with the config
jaxrens dump-schema                      # JSON schema for editor autocomplete
jaxrens run -c config.yaml --set run.n_live=64 --set moves[0].step_size=0.05
```

Example configs are under `jaxrens/experiments/examples/` (e.g. `lj8_npt/`).

## Core architecture (jaxrens)

**Two-loop nested sampling** (`src/jaxrens/sampling/nested_sampling.py`):
- **Outer Python loop** — step-size adaptation, multi-kernel dispatch/retry on neighbor-count violations, termination checks, callbacks, logging.
- **Inner `lax.scan`** inside `ns_step` — `n_mcmc_steps` of MCMC per dead-point replacement, fully JIT-compiled.

The boundary between the two loops is load-bearing: anything requiring Python control flow (kernel re-selection, dynamic shape changes, I/O) belongs outside `ns_step`; anything shape-stable belongs inside.

**Three-level parallelism** — `pmap(vmap(vmap(...)))` with data shaped `(G, P, K, ...)`:
- `G` = `n_gpu_parallel` (pmap axis across GPUs)
- `P` = `n_runs_per_gpu` (outer vmap, independent NS runs)
- `K` = `n_walkers` (inner vmap, walkers within a run)

This shape convention flows through `WalkerState`, `NSState`, moves, energies, step sizes, and key management. **Do not break it.** See `sampling/batch_wrapper.py` for the central wrapper construction.

**State as JAX pytrees** — `state/walker.py` (`WalkerState`) and `state/ns.py` (`NSState`) are dataclasses registered with JAX's pytree machinery. Fields tagged `static_field()` live in aux data (changing them triggers recompilation); other fields are pytree leaves. Functional update via `.set(field=value)`.

**Move kernels** (`sampling/moves/*.py`) follow the `init/step` `MoveKernel` protocol (`sampling/move_kernel.py`). Each move exposes `init()`, `build_kernel()`, `as_top_level_api()`. Current kernels: `random_walk`, `galilean`, `hmc`, `single_atom`, `alchemical`, `volume`, `stretch`, `shear`, `replica_exchange`. They compose into a Metropolis-within-Gibbs scheduler via `build_mwg` (`sampling/mwg.py`).

**Backends** (`backends/`) implement the `EnergyBackend` protocol (`backends/base.py`). `load_backend()` dispatches by config. `EnsembleBackend` wraps any backend to add a PV term for NPT.

**NeuralIL uses bucketed kernel compilation, not neighbor lists.** `backends/kernel_dispatch.py` pre-compiles kernels for a list of `max_neighbors` values and dispatches at runtime; on a neighbor-count violation the outer loop escalates to a larger bucket. This intentionally replaces JAX-MD's allocate/update/overflow pattern — do not introduce explicit neighbor-list state.

**Configuration** is pydantic-based (`state/config.py`: `NSConfig`, `MoveConfig`, `BackendConfig`, `OutputConfig`). YAML files are the source of truth; `cli/parser.py` loads and validates them; `cli/resolve.py` converts configs into runtime objects.

**`init/` vs `sampling/`** — `init/` handles first-time walker initialization (positions, cells, burn-in, rejection, restart). `sampling/` runs the NS loop on already-initialized walkers.

## Conventions & gotchas

- **One GPU process at a time.** Never launch parallel GPU-using jobs (two background `run_in_background` JAX runs, `pytest-xdist` on JAX tests, etc.) — they OOM the device. Never fall back to `JAX_PLATFORMS=cpu` or `CUDA_VISIBLE_DEVICES=""` as a workaround; serialize and wait for GPU instead. Brief subagents explicitly when they might otherwise parallelize (tests + example run + reproducer).
- **Run expensive tests once, log to file, read the file.** Never re-run a pytest suite or example run just to see a different part of the output. Use `... 2>&1 | tee /tmp/pytest_<topic>.log` (or the runtime-provided background log file) and `head`/`tail`/`grep` against the file. GPU tests are slow; re-running doubles the cost with zero information gained.
- **No multi-line `python -c "..."` with `#` comments.** The pattern triggers a permission safety check that blocks regardless of allowlist. For any diagnostic longer than a one-liner, write it to `/tmp/<topic>.py` and run `python /tmp/<topic>.py` — matches the existing `/tmp/` scratch convention. If inline is truly unavoidable, use `python - <<'EOF' ... EOF` heredoc instead.
- **JIT testing is mandatory**: any code path intended to be JIT-compilable must have a test that exercises it under `jax.jit`, not just eager.
- **Scratch / diagnostic scripts go in `/tmp/`**, not in the repo.
- **Do not use `split_float` / float32x2 tricks.** Energy precision is handled by comparing `E / n_atoms` with random tie-breaking (see `_find_worst_walker` in `sampling/nested_sampling.py`).
- **pmap, not `shard_map`** — explicit user preference. Don't migrate.
- **Test markers**: `heavy` (slow), `gpu`, `multi_gpu`. Default `pytest` run excludes `heavy`.

## Subagents — use them

Three specialized subagents exist for this codebase. Prefer them over working directly on the matching task type:

- **`jaxrens-architect`** — for architectural / design questions, "does this fit the design?" reviews, cross-referencing against the reference codes in `reference_codes/`, and planning new features so they land cohesively. Use before writing code for anything non-trivial.
- **`jax-implementer`** — for translating an already-agreed design into working JAX code with JIT-tested unit tests. Invoke once the architect's advice is concrete.
- **`jaxrens-debugger`** — for pathology diagnosis: 0% cell-move acceptance, step-size collapse/explosion, premature termination, log_Z drift, NaN/Inf energies, JIT-retrace suspicions, cohort/vmap inconsistencies. It writes isolated reproducers in `/tmp/` and returns an analysis + minimal fix — it does not edit production code unless asked.

Typical flow for a new feature: architect → implementer → (if it misbehaves) debugger.
