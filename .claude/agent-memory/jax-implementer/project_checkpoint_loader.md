---
name: Unified checkpoint loader (load_restart shape dispatch)
description: load_restart now dispatches on log_evidence.ndim; infer_restart_shape helper added; roundtrip tests for all three entry points
type: project
---

Shape-aware `load_restart` implemented in `init/restart.py` (2026-04-18).

**Return types by ndim:**
- `ndim == 0` → `(WalkerSet, RestartBundle)` — backward-compatible scalar path
- `ndim == 1` → `list[RestartBundle]` — for `run_ns_parallel`
- `ndim == 2` → `list[list[RestartBundle]]` — for `run_ns_multi_gpu`
- `ndim >= 3` → `ValueError` mentioning `ndim`

**Key helper:** `_build_bundle_from_ckpt(ckpt, idx)` — slices each field by index
tuple and trims dead arrays to `n_dead[idx]` entries. Schema-drift safe: the
`dataclasses.fields(RestartBundle)` test in `test_init_restart.py::TestSchemaDrift`
catches any new field added to `RestartBundle` that isn't covered.

**`infer_restart_shape(bundle)`** added — returns `"single"` / `"parallel"` / `"multi_gpu"`.

**Gotcha**: `max_iterations` in the resuming call must be >= `n_dead` stored in the
bundle. `init_ns` does `jnp.full(max_dead=max_iterations, ...)` then `.at[:n_dead].set(...)`,
so if `n_dead > max_iterations` you get a JAX broadcast error.

**Why:** `_build_bundle_from_ckpt` must be updated whenever a new field is added
to `RestartBundle`. The schema drift tests enforce this.

**How to apply:** When implementing new checkpoint fields, always check that
`_build_bundle_from_ckpt` in `init/restart.py` covers the new field for both
the scalar (`idx=None`) and batched (`idx=(...)`) paths.
