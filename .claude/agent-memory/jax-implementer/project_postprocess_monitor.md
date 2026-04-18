---
name: Postprocess monitor layer
description: Monitor/MonitorCollection/plotting added 2026-04-17; key API notes for future extension
type: project
---

`Monitor`, `MonitorCollection`, and plotting helpers added 2026-04-17.

- `src/jaxrens/postprocess/monitor.py` — Monitor class (~230 lines)
- `src/jaxrens/postprocess/plotting.py` — plot_* helpers (~155 lines)
- `src/jaxrens/postprocess/collection.py` — MonitorCollection (~170 lines)
- `tests/test_postprocess_monitor.py` — 53 tests covering all three modules

**Why:** User requested postprocess analysis + plotting layer on top of existing thermodynamics.py.

**How to apply:** When adding new thermodynamic observables, add a thin wrapper method on Monitor that calls the thermodynamics function. Every observable method loops over T values calling the scalar thermodynamics function — no vectorised vmap here since Monitor is numpy-level only.

Key design decisions:
- `expectation()` on Monitor accepts only shape `(n_dead,)` observable and pads live points with `mean(obs)`. This is a deliberate simplification; full (n_dead+n_live,) interface lives in thermodynamics.py.
- `load_checkpoint` does not expose symbol_map; Monitor.from_directory re-opens the HDF5 to read `f.attrs["symbol_map"]` directly via h5py.
- The pre-computed lj8_npt example artefacts in `experiments/examples/lj8_npt/output/` are used as the smoke test fixture — no real GPU run needed in CI.
- matplotlib is imported lazily inside each plotting function (import inside function body) so headless/no-matplotlib imports still work.
