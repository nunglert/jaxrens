---
name: Checkpoint and postprocess batch-shape support
description: checkpoint.py and thermodynamics.py extended for (G, P, ...) shaped state from run_ns_multi_gpu, 2026-04-18
type: project
---

Removed scalar assumptions from `io/checkpoint.py` and extended `postprocess/thermodynamics.py` to handle arbitrary leading batch dims.

**Why:** `run_ns_multi_gpu` returns `(G, P)`-shaped `log_evidence`, `n_dead`, `iteration`, and `(G, P, max_dead)` dead arrays. The prior code called `float(log_evidence)` / `int(n_dead)` which crashes on batched arrays.

**`io/checkpoint.py` changes:**
- `_store_field(f, key, value)` helper: stores ndim==0 as attr, ndim>=1 as dataset.
- `save_checkpoint`: scalar single-run checkpoints compact-slice `dead_energies[:n_dead]`; batched checkpoints save full padded arrays. `log_evidence` goes through `_store_field` (dataset when batched).
- `load_checkpoint`: `_read_field` reads from dataset or attr transparently. Dead-array padding only applied when `n_dead` is scalar. Returns `log_evidence` with stored shape — caller inspects `.shape` to infer batch variant.
- Mixed restart across (G, P) shards deferred: no unified multi-GPU restart from a single checkpoint. Per-shard RestartBundle approach works.

**`postprocess/thermodynamics.py` changes:**
- Strategy: reshape-then-vmap. Leading dims merged to n_flat, `jax.vmap` over 1-D kernels, reshape back.
- New private `_*_1d` functions hold unchanged per-run logic.
- Public API: 1-D input still returns scalar (fast path, no vmap overhead).
- All six functions extended: `log_evidence`, `partition_function`, `heat_capacity`, `expectation`, `free_energy`, `calc_log_weights` (unchanged, inherently 1-D).

**How to apply:** When adding new postprocess functions, follow the reshape-then-vmap pattern. When adding checkpoint fields that can be batched, use `_store_field` not `f.attrs[key] = float(...)`.
