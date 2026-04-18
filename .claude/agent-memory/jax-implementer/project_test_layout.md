---
name: Test file layout after schema reorganization
description: Where each category of test lives after the 2026-04-17 test_schema.py split
type: project
---

After the 2026-04-17 reorganization:

- `tests/test_schema.py` — pure Pydantic schema validation (115 tests, ~90s)
- `tests/test_resolve.py` — resolver-layer tests: TestResolve, TestToDescriptor, TestToMoveConfig, TestResolvedDescriptors, TestToBackendConfig, TestBuildBackend, TestResolveEnergyBackend, TestToCriterion, TestAdaptationResolve, TestEnsembleResolver, TestCohortExpansionResolver, TestCellResolve, TestExtendedOutputResolve, TestFullConfigResolver, TestInitConfigResolverPartA (renamed from first TestInitConfigResolver), TestInitConfigResolverPartB (renamed from second TestInitConfigResolver, minus 2 E2E tests)
- `tests/test_mwg.py` — single canonical JIT plumbing test (spec→descriptor→build_mwg→ns_step under jit)
- `tests/test_init_walker_set.py` — Mode C resolver tests appended (TestInitConfigResolverModeC)
- `tests/test_init_restart.py` — Mode D resolver tests appended (TestInitConfigResolverModeD)
- `tests/test_termination.py` — TestTerminationEndToEndJit appended
- `tests/test_backends.py` — TestHarmonicBackendCallable appended
- `tests/test_init_positions.py` — TestStartSpeciesE2ERunNS appended
- `tests/test_init_structure.py` — TestModeBEndToEndJit appended

**Why:** test_schema.py was 3352 lines with 241 tests taking ~3min; reorganized to separate schema validation (fast) from resolver logic (medium) and E2E tests (slow, living with their components).

**How to apply:** When adding new tests, place schema-validation tests in test_schema.py, resolver tests in test_resolve.py, and component-specific E2E tests in the relevant component test file.

Two classes were renamed to avoid collision:
- TestInitConfigResolver (line 1705) → TestInitConfigResolverPartA in test_resolve.py
- TestInitConfigResolver (line 2080) → TestInitConfigResolverPartB in test_resolve.py

Deleted: TestInitConfigBurnIn (8 tests, all covered by test_init_burn_in.py), TestJitEndToEnd, TestJitEndToEndBackendSpec, test_full_config_init_positions_jit_compatible (replaced by test_mwg.py canonical test).

Merged: test_npt_three_pressures_three_configs + test_npt_three_pressures_correct_values → test_npt_three_pressures_and_values (in both test_schema.py and test_resolve.py).

Total count: 241 (original test_schema.py) → 492 passing across all affected files (including pre-existing tests in target files).
