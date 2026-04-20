---
name: Modularization commit 6 — PmapVmapRuns
description: PmapVmapRuns fully implemented: wrap_step/split_keys/reduce, AdaptationManager pmap branch, init_ns_multi_gpu/run_ns_multi_gpu, run_loop extended
type: project
---

PmapVmapRuns descriptor fully implemented (was a stub with NotImplementedError). Shape convention: (G, P, K, ...) where G=n_gpu (pmap axis), P=n_per_gpu (vmap axis), K=n_walkers.

**Why:** Enable multi-GPU NS runs using pmap(vmap(ns_step)) dispatch. n_gpu=1 supported and tested; multi-GPU generalizes.

**How to apply:** When working on multi-GPU NS features, check:
- `batch_descriptor.py`: PmapVmapRuns.wrap_step uses pmap(vmap), NOT jit(pmap) — pmap is self-JIT-compiling
- `adaptation/manager.py`: PmapVmapRuns branch uses `[:, :, move_idx]` indexing (axis 2 for n_moves), `_split_pmap_vmap_keys` splits (G,P) keys
- `run_loop.py`: `is_pmap_vmap` flag alongside `is_vmap`; emax extracted via `axis=2`; cumulative counters use `descriptor.shape_prefix + (n_moves,)`
- `nested_sampling.py`: `init_ns_multi_gpu` reshapes from (G*P,...) to (G,P,...); `run_ns_multi_gpu` is the entry point (~80 lines)
- Tests in `tests/test_pmap_vmap.py` (32 tests); `test_batch_descriptor.py` updated from NotImplementedError assertions to real-behavior tests

Known limitations:
- pmap/restart mixing: untested, deferred
- Overflow retry: stop-the-world across all devices (TODO in run_loop.py)
- `_pack_adjustment_info` not called for PmapVmapRuns (shapes incompatible)
- `callbacks=` not wired into run_ns_multi_gpu
