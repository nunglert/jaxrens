---
name: Walker initialization package (jaxrens.init)
description: Status of the 7-step walker initialization build-out; what exists in each step
type: project
---

Step 1 of 7 complete as of 2026-04-17.

`src/jaxrens/init/` package created with:
- `cells.py`: `sample_initial_volume` (V^N and flat-V priors) and `cell_shape_walk` (fori_loop, shear+stretch, aspect-ratio gating).
- `__init__.py`: re-exports both public functions.
- `tests/test_init_cells.py`: 15 tests, all pass, JIT + vmap covered.

Key reuse decisions made in step 1:
- `_build_shear_cell` imported directly from `sampling/moves/shear.py` (pure function, reusable).
- Stretch logic was inlined as `_propose_stretch` (the kernel in `stretch.py` is bound to MCState/backend and cannot be reused cleanly; extraction is under 30 lines).
- `get_volume` and `min_aspect_ratio` from `utils/cell.py` used throughout.
- Volume-rescaling after each fori_loop body step to suppress float32 drift.

**Why:** Architect spec step 1 of 7-step walker init plan.
**How to apply:** Steps 2–7 build positions, rejection sampling, structure loading, walker-set files, restart, and burn-in on top of this foundation. `cli/resolve.py::_resolve_init` is still a placeholder — wired in step 2.
