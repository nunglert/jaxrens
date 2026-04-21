# TODO

Deferred work items. Entries should be self-contained enough that a future session can pick them up without re-planning.

---

## Intra-RE (within-chain replica exchange)

**Status:** deferred 2026-04-20. Design decision recorded.

### Goal

Replica exchange swaps *inside* the MCMC chain (i.e., between walkers within a single `run_one_chain` call). This is distinct from inter-RE (commit 2–5), which fires once per NS iteration after `ns_step` returns.

### Design alternatives

**Option A — `scan(vmap(...))`**: Restructure `run_one_chain` so the scan body operates on a *pool* of walkers and proposes inter-replica swaps at each step. Requires lifting `run_one_chain` to operate on `(n_pool, ...)` rather than a single walker; `vmap` handles parallel walks, `scan` steps through time. Compatible with the current `ns_step` shape. Principal cost: `run_one_chain` signature change propagates through `ns_step`, JIT, and all tests.

**Option B — post-scan pool phase**: Keep `run_one_chain` as-is (single-walker scan). After the vmap-over-walkers scan in `ns_step`, introduce a second pool-phase scan that proposes swaps among the walked walkers. Requires adding a pool-swap step between the existing scan and the scatter-back at line `ns_step:7`. Less intrusive than Option A; no scan signature change. Downside: swaps happen only after the full MCMC chain for each walker, not interleaved.

**Option C — deferred outer-loop mini-chains**: Expose an optional `n_intra_re_cycles` parameter in `_run_loop`. After each `ns_step`, run `n_intra_re_cycles` additional single-step MCMC walks with RE accepted in between. Closest to inter-RE mentally; requires `step_fn` to accept multiple walkers and a swap rule.

### Recommended starting point

Option B (post-scan pool phase) is the lowest-risk first implementation: no `run_one_chain` signature change, no new scan axes, and the pool-phase can reuse the existing `replica_exchange_step` logic with minor shape adaptation.

### Touch points

- `sampling/nested_sampling.py` — `ns_step`: add pool-swap phase after `jax.vmap(run_one_chain)`.
- `sampling/moves/replica_exchange.py` — existing `replica_exchange_step`/`SwapKernel` reuse with walker-level (not run-level) indexing.
- New parameter `intra_re_kernel: SwapKernel | None = None` in `ns_step`.
- Tests: `tests/test_ns_step.py` + new `tests/test_intra_re.py`.

---

## Optional MCMC-trajectory debug capture

**Status:** deferred 2026-04-19. Design complete; implementation paused.

### Goal

Opt-in, config-gated capture of the full MCMC chain (per walker, per step within `n_mcmc_steps`) produced inside `ns_step`, written to the HDF5 trajectory file. Intended for debugging runs where the chain-level acceptance numbers alone aren't enough (e.g. diagnosing why a specific walker gets stuck). Off by default; zero performance impact when disabled.

### Design

**Off by default; overwrite-on-write; warn prominently when enabled.**

- A second JITted variant of `run_one_chain` emits per-step state arrays through the scan output tuple. The standard path is unchanged, so disabled users pay nothing (no extra compile, no warning, no disk usage).
- Walker selection is Python-side post-scan: the full `(n_walk, n_mcmc_steps, ...)` trace comes back from `ns_step`, gets sliced to `walker_indices` (or kept whole when `None`), then handed to the writer.
- **Only the latest recorded iteration lives on disk** — each write overwrites the previous snapshot. Total file size is bounded by one iteration's worth of trace, regardless of run length.
- Only `H5TrajectoryWriter` supports this; `ExtxyzTrajectoryWriter` falls through to a no-op with a one-time warning (chain-per-walker-per-step doesn't map cleanly onto the extxyz frame model).
- **On first activation**, emit a prominent `logger.warning`: "MCMC trajectory tracing is ENABLED. This triggers an extra JIT compile and extra device→host transfers per recorded iteration; expect measurable slowdown. Use only for debugging." One-time, at startup.

### Config schema

New frozen dataclass in `jaxrens/src/jaxrens/state/config.py`:
```python
@dataclass(frozen=True)
class MCMCTrajectoryConfig:
    interval: int = 1                                 # overwrite snapshot every N NS iterations
    walker_indices: tuple[int, ...] | None = None     # None = all walkers (1 + n_extra)
    fields: frozenset[str] = frozenset({
        "positions", "energy", "cell",
        "move_idx", "accepted", "reject_reason",
    })
```
Attach as `OutputConfig.mcmc_trajectory: MCMCTrajectoryConfig | None = None`. `None` → disabled, zero overhead. Any non-None value → traced variant compiled + warning emitted.

### Capture path

- `jaxrens/src/jaxrens/sampling/nested_sampling.py` — add `run_one_chain_with_trace` whose `scan_body` extends the scan output with a `trace` dict containing the requested fields at each step. `lax.scan` stacks them naturally.
- JIT gating: `mcmc_trajectory is None` → use existing `run_one_chain`; otherwise compile the traced variant. Always-capture-then-slice is simpler than scan-time walker filtering, and the transient HBM cost (~15 KB × n_walk per NS iter) is negligible. The real cost is device→host transfer of the full trace on every recorded iteration — hence the debug-only warning.
- Writer method `write_mcmc_chain(iteration: int, walker_idx: int, chain: dict[str, np.ndarray]) -> None`:
  - `H5TrajectoryWriter`: write to a **fixed** group `/mcmc_chain/walker_{walker_idx:04d}/` with one dataset per field shape `(n_mcmc_steps, ...)`. Overwrite existing datasets (delete-then-create, or `f[path][...] = data` with shape check). Stamp group-level attr `iteration` with the NS iteration number. Also stamp a file-level attr `mcmc_chain_iteration`. No iteration key in the path.
  - `ExtxyzTrajectoryWriter`: no-op + `logger.warning` once ("mcmc_trajectory requires format=h5").
  - `NullTrajectoryWriter`: no-op.
- New `MCMCTrajectoryCallback` in `cli/monitor.py`:
  - At `__init__`, emit the "ENABLED / debug only" warning exactly once.
  - `on_iteration` fires when `iteration % config.interval == 0`, reads `info["mcmc_trace"]`, converts to numpy, slices walker axis to `config.walker_indices` (or keeps all if None), dispatches per-walker `write_mcmc_chain`. Each call overwrites the previous snapshot.

### Touch points

- `jaxrens/src/jaxrens/state/config.py` — `MCMCTrajectoryConfig` + `OutputConfig.mcmc_trajectory` field.
- `jaxrens/src/jaxrens/sampling/nested_sampling.py` — add `run_one_chain_with_trace`; gate at jit-compile site.
- `jaxrens/src/jaxrens/io/trajectory.py` — add `write_mcmc_chain` to all three writer classes.
- `jaxrens/src/jaxrens/base.py` — add `write_mcmc_chain` to the `TrajectoryWriter` Protocol.
- `jaxrens/src/jaxrens/cli/monitor.py` — new `MCMCTrajectoryCallback`.
- `jaxrens/src/jaxrens/cli/parser.py` + relevant `schema/` spec file — parse the new config section.
- `jaxrens/src/jaxrens/cli/run.py` — register `MCMCTrajectoryCallback` when `OutputConfig.mcmc_trajectory is not None`.

### Tests

- New `tests/test_mcmc_trajectory.py`:
  - Enable with `interval=1, walker_indices=(0, 3)`, `fields={"positions", "energy"}`. Run 5 NS iters. Assert HDF5 has `/mcmc_chain/walker_0000/positions` shape `(n_mcmc_steps, n_atoms, 3)` and walker 3; walker indices 1, 2, 4+ absent. Assert file attr `mcmc_chain_iteration == 4`.
  - Overwrite semantics: capture known-value at iter 2 and iter 4; assert only iter-4 data present (iter-2 overwritten, not appended).
  - `walker_indices=None` → all `n_walk` walker groups present.
  - Field filter: `fields={"positions"}` → only `positions` dataset per walker group.
  - Extxyz warning path: run with extxyz writer + config enabled, assert no error and one warning logged.
  - Warning emission: exactly once at startup when enabled; zero times when disabled.
- Zero-overhead test: time `ns_step` compile + run with capture disabled vs baseline. Disabled variant must match baseline within noise.

### Verification

After implementation:
- Re-run lj8_npt with `mcmc_trajectory.interval=1, walker_indices=[0, 1, 2]` added to `config.yaml`. Confirm "ENABLED / debug only" warning appears once in log. `h5ls -r output/lj8_npt.traj.h5` → `/mcmc_chain/walker_0000/positions` etc. (no iteration key in path), each shape `(n_mcmc_steps, n_atoms, 3)`. `h5dump -A output/lj8_npt.traj.h5 | grep mcmc_chain_iteration` = final NS iteration. Disk delta bounded by one iteration of trace.
- Re-run with `mcmc_trajectory` absent from config: no warning, runtime + file sizes match baseline (no regression).

## Initial energy max_neighbors safe

During run initialization, we compute the energies via 

```python
    if initial_energies is None:
        logger.debug(
            "initial_energies missing from ResolvedInit; computing in run_from_config. "
            "Prefer populating ResolvedInit.initial_energies via the resolver."
        )
        def eval_one(pos):
            e, _, _ = backend(pos, initial_types,
                             initial_cells[0] if initial_cells is not None else jnp.zeros((3, 3)),
                             0)
            return e
        initial_energies = jax.vmap(eval_one)(initial_positions)
```

As far as I understand, this requires the max_neighbors being properlly set. And at that stage, the backend hasnt seen the dataset yet. So I think we would have to "calibrate" the mach_neigbhors first, before we evaluate.

## Energy degeneracies

In practice of NS simulations at 32bit float precision, we often face the problem of degenerate walkers. If the highest energy walker happens to be degenerate with another walker, it needs a statistically sound way of deciding which one to cull. I think uniform sampling is required here, but at some point we need to carefully think about this.