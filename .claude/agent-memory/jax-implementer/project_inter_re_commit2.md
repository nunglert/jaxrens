---
name: Inter-RE commit 2 — InterREManager + _run_loop wiring
description: InterREManager class, _run_loop wiring, InterREConfig dataclass, CLI schema, monitor output, LJ-8 demo; pressure extraction shape bug fix
type: project
---

Commit 2 of the inter-RE plan landed 2026-04-20.

**Why:** Enable multi-run nested sampling with pressure-RENS replica exchange swaps between runs at different pressures.

**Key files:**
- `sampling/inter_re_manager.py` (new) — `InterREManager` class
- `sampling/run_loop.py` — `inter_re_mgr` param, scalar `inter_re_key`, post-step swap phase
- `sampling/nested_sampling.py` — `run_ns_parallel`/`run_ns_multi_gpu` gain `inter_re_config`, `backend`, `callbacks` params
- `state/config.py` — `InterREConfig` frozen dataclass; `NSConfig.inter_re` field
- `cli/schema/inter_re.py` (new) — `InterREConfigSpec` pydantic model
- `cli/monitor.py` — inter-RE stats row in `ProgressCallback.on_iteration`
- `experiments/examples/lj8_npt/run_inter_re.py` (new) — standalone LJ-8 NPT demo

**Shape bug:** `population.ensemble_params["pressure"]` after `init_ns_parallel` is `(n_runs, n_walkers)` (NOT `(n_runs,)`) because MCState vmapping stacks each walker's scalar pressure into `(n_walkers,)` before stacking runs. Fix in `_extract_swap_inputs`: `arr[:, 0]` slice for ndim==2.

**Galilean move in demo scripts:** Always pass `extra_state_fields={"direction": (jnp.ndarray, lambda pos, types: jnp.zeros_like(pos))}` to `MoveKernel` for galilean. Also pass `move_descriptors=descriptors` to `run_ns_parallel` so `n_moves` is inferred correctly — default `n_moves=1` causes shape mismatch with `(n_moves,)` evals from MWG.

**`run_ns_parallel` now has `callbacks` param** (was hardcoded `[]`). Enables `ProgressCallback` to show inter-RE stats line: `inter_re  n_pairs=N  acc=X.XX  evals=N`.

**Tests:** 21 in `test_inter_re_manager.py`, 12 in `test_inter_re_integration.py`. 46 scoped tests pass.

**Demo result:** `|log_Z[0] - log_Z[1]| = 2.41` for P=[0.01, 0.1] eV/Å³, confirming different pressures produce different thermodynamics.

**How to apply:** Next is commit 3 — XRENSSwap concrete SwapKernel with energy-evaluation path for composition swaps.
