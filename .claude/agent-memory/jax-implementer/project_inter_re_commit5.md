---
name: Inter-RE commit 5 — SemiGrandSwap
description: SemiGrandSwap, semi_grand_replica_exchange_step, CLI schema semi_grand, flavor dispatch, 26 new tests, 2026-04-20
type: project
---

SemiGrandSwap landed 2026-04-20. Inter-RE plan complete (all 5 commits done).

**Why:** Chemical-potential RE flavor for μVT/μPT ensembles. Pure arithmetic on stored E, types, μ — zero backend calls.

**Sign convention:** state.energy = raw potential U (not grand-canonical Ω). Grand-canonical energy under swapped μ: `Ω_A = U_A - μ_B · N_A` where `N_A = bincount(types_A, length=n_species)`. Accept iff `Ω_A < Emax_A AND Ω_B < Emax_B`. Matches legacy `jaxns-devAS/src/jaxnest/replica_exchange.py::create_perform_semi_grand_swap` lines 195-249.

**How to apply:** Use `InterREConfig(flavor="semi_grand", chemical_potentials=((mu0_s0, mu0_s1), (mu1_s0, mu1_s1)))` and pass to `run_ns_parallel`/`run_ns_multi_gpu` with `backend=...`. `chemical_potentials` injected as float32 JAX array into `ensemble_params_per_run`; InterREManager extracts via `ep["chemical_potentials"]` (shape `(n_runs, n_walkers, n_species)` after vmap stacking → sliced to `[:, 0, :]`).

**Key files changed:**
- `sampling/moves/replica_exchange.py` — SemiGrandSwap class + semi_grand_replica_exchange_step
- `state/config.py` — chemical_potentials field type updated to tuple[tuple[float,...],...]
- `cli/schema/inter_re.py` — semi_grand added to _IMPLEMENTED_FLAVORS, chemical_potentials validated
- `sampling/inter_re_manager.py` — _is_semi_grand flag, new pmap/vmap dispatch branches, _extract_swap_inputs returns 8-tuple
- `sampling/nested_sampling.py` — semi_grand elif branch in run_ns_parallel and run_ns_multi_gpu
- `tests/test_semigrand.py` — 26 new tests
- `tests/test_inter_re_integration.py` — test_semi_grand_raises_at_run_time renamed, expects ValueError not NotImplementedError
- `tests/test_xrens.py` — test_semi_grand_raises renamed for same reason
- `jaxrens/WORKLOG.md` — commit 5 bullet appended under 2026-04-20
- `jaxrens/TODO.md` — intra-RE entry added with scan(vmap) vs post-scan design alternatives

**Test run:** 180 passed in 190s (test_semigrand, test_xrens, test_morph, test_inter_re_manager, test_inter_re_integration, test_replica_exchange, test_nested_sampling).

**Final feature surface:** pressure, xrens, semi_grand all work end-to-end with VmapRuns and PmapVmapRuns(n_gpu=1).

**Open items:** (a) intra-RE (scan(vmap) vs post-scan pool phase, captured in TODO.md), (b) cross-device ppermute swaps for n_gpu>1, (c) adaptive swap frequency.
