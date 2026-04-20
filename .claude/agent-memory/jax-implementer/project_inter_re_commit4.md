---
name: Inter-RE commit 4 (XRENSSwap)
description: XRENSSwap implementation, xrens_replica_exchange_step, CLI schema validation, end-to-end test, 2026-04-20
type: project
---

XRENSSwap (composition-morphing replica exchange) was already implemented in prior work; commit 4 validated the full pipeline end-to-end.

**Key design points:**
- `XRENSSwap(n_species)` subclasses `SwapKernel`. `propose()` calls `morph_types_to_composition` in both directions then re-evaluates energies via backend. Returns `(proposed, 2, 0)`. `accept()` delegates to `PressureRENSSwap.accept`.
- `xrens_replica_exchange_step` is the standalone entry-point; tracks `n_energy_evals` in `swap_info`.
- `InterREManager` XRENS path dispatches to `xrens_replica_exchange_step`; `_extract_swap_inputs` handles `target_composition` from `ensemble_params` for ndim 2 (VmapRuns) and 3 (PmapVmapRuns).
- `run_ns_parallel` / `run_ns_multi_gpu` inject `target_composition` into `ensemble_params_per_run` and re-run `init_ns_parallel` when `flavor="xrens"`.

**CLI schema (`cli/schema/inter_re.py`):**
- `InterREConfigSpec` validates `flavor="xrens"` requires `composition_targets`.
- Checks row-length consistency (n_species) and row-sum consistency (n_atoms).
- `semi_grand` raises `NotImplementedError`.

**Backend for E2E tests:** `SpeciesHarmonicBackend` (inline in `test_xrens.py`) — E = Σ w[types[i]] · 0.5 · ||posᵢ||². No periodic/cutoff needed; energy explicitly depends on composition.

**Observed acceptance:** identical weights (w₀=w₁=1.0) → 100% acceptance (morph doesn't change energy). Different weights with energy-reeval → composition-dependent.

**Why:** `target_composition` must be injected into `ensemble_params_per_run` before `init_ns_parallel` so the vmap-stacked `ns_state.population.ensemble_params["target_composition"]` has shape `(n_runs, n_walkers, n_species)` — `_extract_swap_inputs` then takes `[:, 0, :]` to get per-run targets.

**How to apply:** When adding new XRENS features, trace the `target_composition` shape through: injection → vmap stacking → `_extract_swap_inputs` → `xrens_replica_exchange_step`.

**138 scoped tests pass** (test_xrens + test_morph + test_inter_re_manager + test_replica_exchange + test_nested_sampling) in 143s as of 2026-04-20.
