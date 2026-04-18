---
name: Walker initialization package (jaxrens.init)
description: Status of the 7-step walker initialization build-out; what exists in each step
type: project
---

Steps 1–5 of 7 complete as of 2026-04-17.

Step 1 (`cells.py`, `tests/test_init_cells.py`): 15 tests, all pass.

Step 2 (`positions.py`, `rejection.py`, resolver rewrite):
- `positions.py`: `uniform_positions_in_cell` (frac @ cell, fully JIT/vmap-safe) and `grid_positions_in_cell` (numpy-side grid counts, jax.random.choice for selection; NOT JIT-safe when cell is a traced argument — use from Python loops only; vmap over key works).
- `rejection.py`: `rejection_sample_positions` Python while-loop; JIT-compiled `_inner_check` (energy + minimum-image pair-distance) built once before loop; reject_code ∈ {0=ok, 1=energy_over_ceiling, 2=nan_energy, 3=atoms_too_close}; raises `RuntimeError` with dict counters on exhaustion.
- `_resolve_init` in `cli/resolve.py` fully rewritten: volume via `sample_initial_volume`, cells via `cell_shape_walk` (vmap over walkers), positions via rejection or grid (Python loop over walkers), energies via backend vmap. `energy_backend` and `cell_cfg` now passed into `_resolve_init`.
- `run_from_config` energy fallback kept as `logging.debug` for backward compatibility.
- `__init__.py` re-exports all 5 public functions.
- Tests: `test_init_positions.py` (16 tests), `test_init_rejection.py` (10 tests), `TestInitConfigResolver` in `test_schema.py` (8 tests). Total 289/289 green.

Step 3 (`structure.py`, Mode B in `cli/resolve.py`):
- `src/jaxrens/init/structure.py`: `load_structure(path)` returns `(positions, types, cell, symbol_map)` — reads via `ase.io.read(str(path), index=":")` (must use `index=":"` to correctly detect multi-frame files; `index=0` always returns a single `Atoms` even for multi-frame), validates non-zero cell vectors, builds `symbol_map` in first-appearance order.
- `ResolvedInit` extended with `symbol_map: dict[int,str] | None = None` field. Mode A populates from `ase.data.chemical_symbols`; Mode B from file.
- `_resolve_init` split into `_resolve_init_species` (Mode A) + `_resolve_init_config_file` (Mode B), with shared helpers `_build_cells` and `_validate_cells`. `_sample_per_walker_positions` extracted for DRY re-use.
- Mode B `random_initialise_pos=False` emits `logging.warning` with "burn-in" text pointing to `InitialWalkConfig.n_walks`.
- `tests/test_init_structure.py`: 19 tests. Key: multi-frame detection only works with `index=":"`.
- `TestInitConfigResolver` in `test_schema.py` extended with 10 new Mode B tests (symbol_map, identical positions, burn-in warning, cell divergence, JIT end-to-end). Lives in the SECOND `TestInitConfigResolver` class (Python shadowing: duplicate class name, pytest only picks the last one).
- Total suite: 317/317 green.

Key gotcha found: `grid_positions_in_cell` calls `float(np.linalg.norm(np.array(cell[...])))` which uses numpy to extract concrete Python floats. Under JAX JIT, even captured jnp arrays in closures become abstract tracers — `cell[0]` is traced even when cell is a module-level constant captured in a closure. The function is intentionally not JIT-safe over the cell argument; the JIT test uses `jax.random.choice` as a proxy to verify the JAX-random-selection part compiles.

Step 4 (`walker_set.py`, Mode C in `cli/resolve.py`):
- `src/jaxrens/init/walker_set.py`: `load_walker_set(path, n_live_expected) -> WalkerSet` frozen dataclass. Dispatches by extension: `.extxyz`/`.xyz` → ASE multi-frame, `.h5`/`.hdf5` → h5py direct (not delegated to `io/checkpoint.load_checkpoint` because that loads NS-state fields Mode C doesn't need). Validates: frame count, atom count consistency, composition consistency, non-zero cells. HDF5 synthesizes integer-coded symbol_map if attribute missing.
- `_build_symbol_map_from_symbols(symbols) -> (symbol_map, type_indices)` extracted from `load_structure` into `structure.py` (identical logic, exact duplication justified by step 4). `load_structure` now calls this helper.
- `_resolve_init_walker_set` in `cli/resolve.py`: warns if `random_initialise_pos/cell=True` and ignores them; calls `load_walker_set`; validates cells via `_validate_cells(cells, n_atoms, cell_cfg)`; vmaps energy_backend over all walkers to recompute energies.
- `_resolve_init` dispatcher: `start_walker_set` branch now calls `_resolve_init_walker_set` instead of raising `NotImplementedError`.
- `init/__init__.py` exports `WalkerSet` and `load_walker_set`.
- Tests: `tests/test_init_walker_set.py` (38 tests), `TestInitConfigResolverModeC` in `test_schema.py` (14 new tests including end-to-end JIT). Total suite: 369/369 green.
- The old `test_start_walker_set_raises_not_implemented` test was replaced with `test_start_walker_set_nonexistent_raises_file_not_found`.

Key gotcha found (step 3): `test_schema.py` has two classes with the same name `TestInitConfigResolver` (lines ~1665 and ~2286). Python class redefinition means pytest only collects the last one. New tests MUST go in the second class (or a new, differently-named class).

Step 5 (`restart.py`, Mode D in `cli/resolve.py`):
- `src/jaxrens/init/restart.py`: `load_restart(path) -> (WalkerSet, RestartBundle)`. Pre-validates HDF5 file has `energies`, `dead_energies`, `dead_positions` datasets and `log_evidence`, `iteration`, `n_dead` attributes before calling `load_checkpoint` (bare walker-set files have positions/types/cells but not these fields; must check upfront because `load_checkpoint` crashes on missing `energies`). Dead arrays returned as compact slices (n_dead,), not padded — padding done in `init_ns`.
- `RestartBundle` frozen dataclass: `dead_energies`, `dead_positions`, `dead_volumes|None`, `log_evidence: float`, `iteration: int`, `n_dead: int`.
- `ResolvedInit` extended with `restart_state: RestartBundle | None = None`.
- `_resolve_init_restart` in `cli/resolve.py`: warns if `random_initialise_pos/cell=True`, calls `load_restart`, validates cells, vmaps energy_backend to recompute energies from current backend.
- Cohort-size guard added to `expand_cohort`: raises `ValueError` with cohort size if `restart_file is not None and n > 1`.
- `init_ns` extended with `restart_state=None` kwarg: when provided, pads compact RestartBundle arrays into pre-allocated max_dead arrays and seeds `log_evidence`, `iteration`, `n_dead`. `run_ns` forwards `restart_state` to `init_ns`.
- `run_from_config` extended with `restart_state=None` kwarg, forwarded to `run_ns`.
- Tests: `tests/test_init_restart.py` (24 tests). `TestInitConfigResolverModeD` added to `test_schema.py` (16 tests).
- Old test `test_restart_file_raises_not_implemented` replaced with `test_restart_file_nonexistent_raises_file_not_found`.

Key gotcha (step 5): bare walker-set HDF5 has no `energies` dataset, so `load_checkpoint` crashes with `KeyError` before any validation code runs. Must pre-check required fields by inspecting the HDF5 directly before delegating to `load_checkpoint`.

Step 6 (`burn_in.py` rewrite, schema fields, cli wiring):
- `src/jaxrens/init/burn_in.py` extended with `batched`, `walker_batch_size`, `run_batch_size` params.
- `_one_walk` private fn handles a single-run NSState: vmap or lax.map over walkers.
- `_apply_adaptation` handles single-run and batched cases; batched path vmaps `adjust_step_size` over run axis.
- Top-level `initial_walk` validates divisibility of chunk sizes, computes emax (scalar or (n_runs,)), builds JIT-compiled inner loop, Python outer loop with optional adaptation.
- `jax.lax.map` accepts `batch_size` kwarg in this environment (confirmed with `inspect.signature`).
- `InitialWalkConfig` in `cli/schema/init.py` extended with `walker_batch_size` and `run_batch_size` fields (both `int | None = Field(default=None, ...)`).
- `run_from_config` in `cli/run.py` passes `batched=False`, `walker_batch_size`, `run_batch_size` to `initial_walk`.
- Tests: `test_init_burn_in.py` extended from 14 to 24 tests (added walker-chunking tests 8–10 and batched tests 11–15). `TestInitConfigBurnIn` in `test_schema.py` extended with 4 new tests (schema field acceptance, runtime ValueError for bad walker_batch_size, successful run with walker_batch_size). Total 436/436 green.

Key gotcha (step 6): `lax.map` with `batch_size` vmaps items in chunks; the lambda must unpack a tuple of (population[i], chain_keys[i]) not the full batch. For batched runs, `_batched_one_walk` must split `key` into `n_runs` run-keys and vmap `one_run(run_key, run_state, run_emax)`.

**Why:** Architect spec step 6 of 7-step walker init plan.
**How to apply:** Step 7 adds CellConfig unification on top of this foundation.
