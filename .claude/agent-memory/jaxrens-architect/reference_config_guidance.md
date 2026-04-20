---
name: two-layer config guidance from research
description: Research notes recommend library-layer dataclass configs + app-layer YAML parsed into them
type: reference
---

`workspace/research/arch-comparison-matrix.md` section 5 ("Configuration & I/O") explicitly recommends for jaxrens:

> Two-layer configuration:
> 1. Library layer: Function/constructor arguments with dataclass configs (BlackJAX + JAXNS pattern)
> 2. Application layer: YAML/TOML config files parsed into dataclasses (simpler than gin)

Other reference observations:
- **MLIP** uses pydantic for config (`MaceConfig`, `NequipConfig`, `VisnetConfig`, `OptimizerConfig`, `SimulationConfig`) — production-grade but pulls in pydantic dependency.
- **MACE-JAX** uses gin (`.gin` + CLI overrides) — flexible but adds dependency and non-Pythonic syntax.
- **mlff** uses argparse + dataclass bundling (Coach dataclass) — basic but workable.
- **BlackJAX / JAXNS / JAX-MD** use plain function args with dataclass defaults — no external config system.

The research matrix favors the BlackJAX/JAXNS minimalism for the core library APIs and a thin YAML/TOML layer only at the application entrypoint. This aligns with jaxrens's existing layering (`state/config.py` is library-layer dataclasses; `cli/parser.py` is the app-layer file reader).

**How to apply:** Preserve the `state/config.py` dataclasses as the library's truth; YAML and argparse belong strictly in `cli/`. Don't bake pydantic into the library layer unless the user explicitly wants runtime validation across all API boundaries (not just at the CLI).
