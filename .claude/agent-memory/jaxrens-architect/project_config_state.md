---
name: jaxrens current config surface
description: What exists today for configuration in jaxrens — dataclasses, parser, CLI
type: project
---

As of 2026-04-17, jaxrens configuration is:
- `src/jaxrens/state/config.py` defines four frozen `@dataclass` configs: `NSConfig`, `MoveConfig`, `BackendConfig`, `OutputConfig`. Flat — no nesting.
- `src/jaxrens/cli/parser.py` reads an `ns.inp`-style `key=value` file (regex, no sections) into a flat `dict[str,str]`, then `raw_to_configs()` hardcodes the mapping to the four dataclasses with tolerant aliasing (`n_walkers|n_live`, `atom_traj_len|n_steps|n_mcmc_steps`, `backend|backend_type`, etc.).
- No argparse CLI; no YAML support; single `MoveConfig` only — the parser cannot express a list of moves even though `setup_mwg` and the MWG machinery accept `list[MoveConfig]`.
- `src/jaxrens/cli/run.py` wires configs to backend loader + MWG + `run_ns`. It is also where `_MOVE_REGISTRY` maps move-type strings to `build_kernel` callables and where per-move `kernel_kwargs` / `extra_state_fields` get materialized.

**Why:** The refactor's first priority was the sampling core (EnergyBackend, MWG, vmap, ensemble). The config layer was deliberately kept minimal and aliased to accept old ns.inp files for transitional convenience.

**How to apply:** When proposing a config redesign, note that (a) the four existing dataclasses are imported widely, (b) the `list[MoveConfig]` gap in the parser is a real feature gap not a design choice, (c) the `_MOVE_REGISTRY` in `cli/run.py` is the de-facto move-type enum.
