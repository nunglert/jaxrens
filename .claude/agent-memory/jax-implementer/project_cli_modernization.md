---
name: CLI modernization plan
description: 8-step plan to modernize jaxrens CLI input/config layer; steps 1–7 complete as of 2026-04-17
type: project
---

Steps 1-3 of the CLI modernization are complete (2026-04-17).

**Step 1** introduced:
- `src/jaxrens/cli/schema/` — pydantic v2 BaseModels mirroring the four library dataclasses
- `src/jaxrens/cli/resolve.py` — pure `resolve(RootConfig) -> ResolvedConfig` seam
- `src/jaxrens/cli/cli.py` — argparse entry point with `run`, `validate`, `dump-schema` subcommands
- `[project.scripts]` entry in `pyproject.toml`
- `tests/test_schema.py` (24 tests) and `tests/data/cli/minimal.yaml`

**Step 2** replaced the flat `MoveSchema` with a discriminated union of per-move-type specs:
- `BaseMoveSpec` + 12 concrete `*MoveSpec` classes in `schema/moves.py`
- `MoveSpec = Annotated[Union[...], Field(discriminator="type")]` exported from `schema/__init__.py`
- `BaseMoveSpec.to_move_config()` and `.to_descriptor()` replace `_MOVE_REGISTRY`, `_build_kernel_kwargs`, `_extra_state_fields` from `cli/run.py`
- `ResolvedConfig` gains `move_descriptors: tuple[MoveDescriptor, ...]`
- Legacy `move_type` key in YAML dicts is rewritten to `type` in `RootConfig._normalize_moves` for backward compat
- `MoveSchema` retained as backward-compat shim; `move_type` property on `BaseMoveSpec` keeps old assertions working
- `_move_config_to_descriptor()` in `run.py` bridges old `MoveConfig`-based callers (simple move types only)
- `tests/data/cli/mixed_moves.yaml` fixture added; `tests/test_schema.py` extended to 58 tests (34 new)
- JIT test: `TestJitEndToEnd::test_spec_descriptor_mwg_ns_step_jit` — uses `static_argnames=("step_fn", "n_mcmc_steps")`

**Step 3** replaced the flat `BackendSchema` with a discriminated union of per-backend-type specs:
- `BaseBackendSpec` + 6 concrete `*BackendSpec` classes in `schema/backend.py`: `HarmonicBackendSpec`, `DoubleWellBackendSpec`, `GaussianMixtureBackendSpec`, `LJBackendSpec`, `NeuralILBackendSpec`, `MACEBackendSpec`
- `BackendSpec = Annotated[Union[...], Field(discriminator="type")]` exported from `schema/__init__.py`
- `BaseBackendSpec.to_backend_config()` and `.build_backend()` are the seam — each spec owns its constructor kwargs
- `ResolvedConfig` gains `energy_backend: EnergyBackend` (pre-built backend object); `backend: BackendConfig` retained
- `resolve()` calls `root.backend.to_backend_config()` and `root.backend.build_backend()` — no more `load_backend()` in the resolve path
- Legacy `backend_type` key in YAML dicts is rewritten to `type` in `RootConfig._normalize_backend` for backward compat
- `BackendSchema` retained as alias for `BackendSpec` for backward compat
- `cli/run.py` `run_from_config()` still uses `load_backend()` via `BackendConfig` — unchanged because test_cli.py passes `BackendConfig` directly
- `tests/data/cli/lj_backend.yaml` fixture added
- `tests/test_schema.py` extended from 68 to 102 tests (34 new, covering discriminated union, extra-field rejection, `to_backend_config()`, `build_backend()`, `energy_backend` in `ResolvedConfig`, JIT end-to-end)
- NeuralIL/MACE `build_backend()` are not exercised in tests — schema-level instantiation only; heavy backends gated by test markers

**Step 4** added TerminationSpec and AdaptationConfig:
- `schema/termination.py`: `BaseTerminationSpec` + 4 concrete specs (`IterationTerminationSpec`, `PriorMassTerminationSpec`, `TemperatureTerminationSpec`, `EnergyTerminationSpec`). Each has `to_criterion()` returning the library runtime object. `TerminationSpec` is the discriminated union.
- `schema/adaptation.py`: `AdaptationPolicy` (all-None overlay), `AdaptationConfig` (defaults + per_move dict + full_auto flags). `resolve_for(name)` returns `ResolvedAdaptationPolicy` with no None fields, using library fallbacks (min_rate=0.25, max_rate=0.65, adjust_factor=1.5, step_size_max=10.0) from `MoveDescriptor` defaults.
- `RootConfig` gains `termination: list[TerminationSpec] | None` (None = legacy behavior) and `adaptation: AdaptationConfig` (default-constructed = no change).
- `ResolvedConfig` gains `termination: tuple[TerminationCriterion, ...]` (never empty — None resolves to legacy pair) and `adaptation_policies: tuple[ResolvedAdaptationPolicy, ...]` (one per move, same order as `moves`).
- `cli/run.py::run_from_config` gains `termination_criteria: list | None` param and passes it to `run_ns`.
- `run_ns` already accepted `termination_criteria` and used `check_any` — no sampling-side changes needed.
- Fallback values for `AdaptationPolicy` match `MoveDescriptor` defaults (hardcoded in `adaptation.py`).
- Tests: 131 total (29 new). JIT test: `TestTerminationEndToEndJit::test_iteration_termination_stops_early_under_jit` — calls `run_ns` (which JITs `ns_step` internally) with `IterationTermination(5)` and asserts `n_dead <= 6`.
- Fixtures: `tests/data/cli/termination_iteration.yaml`, `tests/data/cli/adaptation_overlay.yaml`.

**Step 5** added EnsembleSpec and cohort expansion (2026-04-17):
- `schema/ensemble.py`: `BaseEnsembleSpec` + `NVTEnsembleSpec` + `NPTEnsembleSpec`. `EnsembleSpec` is the discriminated union. `NPTEnsembleSpec.pressure` accepts scalar or list[float]; `cohort_size()` returns list length. `to_ensemble_params(cohort_index)` returns `{"pressure": float}` in eV/Å³, converting from GPa via `_GPA_TO_EVA3 = 0.006241509` (= 1e9 * 0.6241509e-11). muVT/semi-grand deferred — no chemical-potential machinery in the runtime.
- `RootConfig` gains `ensemble: EnsembleSpec = NVTEnsembleSpec()` default. `@model_validator(mode="after")` synthesizes `NPTEnsembleSpec` from legacy `run.pressure` field (backward compat); raises `ValidationError` if both `run.pressure` and explicit `ensemble:` are provided.
- `resolve.py`: `ResolvedConfig` gains `cohort_index: int = 0` and `ensemble_params: dict = {}`. `_resolve_one(root, cohort_index)` is the single-cohort resolver. `expand_cohort(root) -> list[ResolvedConfig]` is the new public API — returns one element per cohort index. `resolve(root)` is a thin wrapper that asserts cohort_size==1 and returns `expand_cohort(root)[0]`.
- `cli.py::_cmd_run` uses `expand_cohort`; loops sequentially for n>1, printing progress per run. `_cmd_validate` reports `"OK — cohort size: N"`.
- Seed handling: cohort element i gets `seed + i` (deterministic from base seed).
- Pre-existing bug noted: `monitor.py::_ns_state_to_checkpoint_dict` calls `float(jnp_array)` on NPT ensemble_params pressure, which fails if ndim>0. Not step-5 scope. End-to-end NPT tests use `run_ns` directly (bypass monitor) to avoid this.
- `run_ns_parallel` exists in `nested_sampling.py` (takes `ensemble_params_per_run: list[dict]`). Parallel cohort dispatch is step-7 material.
- Tests: 163 total (32 new). Fixtures: `tests/data/cli/npt_scalar.yaml`, `tests/data/cli/npt_sweep.yaml`.

**Step 6** added InitConfig, CellConfig, and extended OutputConfig (2026-04-17):
- `schema/init.py`: `InitialWalkConfig` (n_walks=0 disabled default; deferred) + `InitConfig`. Mutually-exclusive source-of-atoms enforced by `@model_validator`. `parsed_species()` parses single-composition species strings (e.g. `"14 14 8"` -> `{8:1, 14:2}`). Multi-composition (`:` separator) raises `ValueError` — deferred to future cohort expansion. `start_config_file`, `start_walker_set`, `restart_file` accepted but resolve to `NotImplementedError` (no structure reader / walker loader in runtime).
- `schema/cell.py`: `CellConfig` with `max_volume_per_atom`, `min_volume_per_atom`, `min_aspect_ratio`, `flat_V_prior`. Accepted but NOT threaded into move kernels (option 2 chosen — less than 20 lines of change to kernels would have been needed but unification of per-move copies into a shared config is a larger design task). Resolver emits `logging.warning` when non-default values are set.
- `schema/output.py`: Added deferred fields: `snapshot_time`, `snapshot_clean`, `wrap_atoms`, `save_stepsizes`, `write_traj_db`, `write_walkers_db`. All accepted, validated. Non-default values emit `logging.warning` via `_warn_unused_output_fields()` in `resolve.py`. Deferred fields NOT propagated into `OutputConfig` library dataclass (which is unchanged).
- `schema/root.py`: `RootConfig` gains `init: InitConfig = InitConfig(start_species="1")` and `cell: CellConfig = CellConfig()` defaults.
- `resolve.py`: `ResolvedInit` dataclass holds `initial_positions`, `initial_types`, `initial_cells`, `initial_energies`. `_resolve_init()` dispatches on source field. `_warn_unused_output_fields()` and CellConfig non-default warning added. `ResolvedConfig` gains `init: ResolvedInit` and `cell: CellConfig`.
- `cli.py::_run_one`: Replaced step-1 placeholder (ad hoc random positions) with `resolved.init.initial_positions/initial_types/initial_cells/initial_energies`.
- `state/config.py::OutputConfig`: Unchanged — deferred output fields are CLI-schema-only.
- Cell move kernels (`volume.py`, `shear.py`, `stretch.py`): Unchanged — they already accept per-spec kwargs; unified CellConfig threading deferred.
- Tests: 204 total (41 new). Fixture: `tests/data/cli/full_config.yaml` exercises all 8 sections. JIT test: `TestFullConfigFixture::test_full_config_init_positions_jit_compatible` — `ResolvedInit.initial_positions` from `start_species` feeds into `ns_step` under `jax.jit`.

**Step 7** added `migrate-ns-inp` subcommand and retired the legacy `cli/parser.py` dataclass-construction path (2026-04-17):
- `cli/migrate.py`: Pure `migrate_ns_inp(raw: dict[str, str]) -> dict` with a routing table (`_ROUTING_TABLE`), explicit drop list (`_DROPPED`), and deferred map (`_DEFERRED`). Three buckets: routed (placed in config), deferred (placed with INFO log), dropped (WARNING log, not placed), unknown (_unknown bucket + WARNING).
- Pressure routing: `MC_cell_P` (space-separated GPa) -> `ensemble.npt` with `pressure_units: "gpa"`. Single value → scalar pressure; multiple → list for cohort expansion.
- `n_iter_times_fraction_killed` resolved post-routing using resolved `n_walkers`/`n_cull` values.
- `_build_moves_from_scratch()`: builds `moves` list from `_move_counts` and `_step_sizes` scratch keys.
- `cli/parser.py`: Narrowed to `parse_input_file` only — `raw_to_configs`, `load_config` removed.
- `cli/run.py`: `run_from_file` removed (depended on deleted `load_config`).
- `jaxrens/__init__.py`: `run_from_file` removed from exports.
- `cli/cli.py`: `migrate-ns-inp` subcommand added; `__main__` guard added for `python -m jaxrens.cli.cli` invocation.
- `tests/test_migrate.py`: 47 new tests covering round-trip, pressure conversion, list cohort, drop list, unknown keys, deferred fields, --validate flag, end-to-end CLI.
- `tests/test_cli.py`: Removed `test_raw_to_configs` and `test_load_config_roundtrip`; kept raw-read test; added `test_parser_public_api` and `test_migrate_importable`.
- Tests: 251 total (47 new in test_migrate.py).
- Key implementation note: `_unknown` bucket is kept in the returned `cfg` dict (not stripped by `_strip_private_keys`); the CLI pops it and renders as YAML comments to avoid `RootConfig.extra="forbid"` rejection.

**Why:** Discriminator-is-the-registry pattern, consistent with step 2's `MoveSpec` approach. Each spec owns construction, no dispatch table needed.

**How to apply:** The existing `cli/parser.py` and `state/config.py` are untouched — they are the stable library core. The pydantic schemas are CLI-only validation wrappers; `resolve()` is the seam between them. Steps 6+ (remove RunSchema.pressure, InitConfig, CellConfig, OutputConfig extension) land in `resolve.py` or the schema layer.
