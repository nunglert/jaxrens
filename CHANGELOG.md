# Changelog

All notable changes to `jaxrens` are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/). Versions follow
semantic versioning; while in `0.x`, **minor** releases may include breaking
changes and **patch** releases are bug fixes only.

The version is derived from git tags by `setuptools-scm` (tag `v0.1.0` →
version `0.1.0`). To cut a release: add a dated section below, merge to `main`,
then `git tag -a vX.Y.Z`.

## [0.3.0] — 2026-08-24

Multi-component sampling gets two purpose-built moves, the codebase grows a way
to mark code that has never been validated in a production run, and the
top-level module layer is dissolved. Note the **breaking changes** below: two
move types and four import paths are gone.

### Added
- **Species-swap move (`type: species_swap`).** Exchanges the identities of two
  atoms of *different* species at fixed composition and geometry — the
  productive degree of freedom for alloys, where a displacement move has to
  tunnel through a barrier to achieve what one swap does directly. The pair is
  drawn unlike *by construction* (species pairs weighted by `n_s · n_t`), so no
  backend evaluation is wasted on a same-species draw and the acceptance rate
  is not capped by the composition. `species: [Ge, Si]` restricts the exchange
  to named elements.
- **Species-scoped GMC moves.** A `gmc` move can be restricted to one
  sublattice, with the reflection normal projected onto the moving subspace.
  Scoped moves are auto-named so they stay distinguishable in the monitor
  columns, the adaptation diagnostics, and `adaptation.resolve_for` overrides.
- **Unvalidated-feature markers (`jaxrens.unvalidated`).** A decorator and
  registry for code that ships but has never been exercised in a production
  simulation — orthogonal to test coverage, which is why it is not a
  `# pragma: no cover`. Each marker records *what specifically* is not trusted
  and what run would clear it, warns once per feature on first call, and is
  enumerable via `REGISTRY`. `JAXRENS_UNVALIDATED=ignore|warn|error` selects the
  policy; `error` gives a production campaign a hard guarantee that nothing
  unvalidated was touched. Currently applied to the HMC, sweep, morph, swap and
  XRENS move builders and the nequix backend factory.
- **`--require-backends` for the test suite.** Optional backends normally
  degrade to skips when absent, and skips do not change the exit code — so a
  soft install failure could report green. CI now names the backends it
  installed (`--require-backends=all`, or `JAXRENS_REQUIRE_BACKENDS`) and any
  that are not importable abort the session before a test runs.
- **MACE model conversion** documentation, a `mace-convert` extra, and a
  dedicated tutorial (`examples/tutorials/01_mace_run.py`), plus a
  troubleshooting page in the user guide.

### Changed
- **Optional-backend import guards now catch `OSError`, not just
  `ImportError`.** A backend can install cleanly and still fail at import when a
  native library is missing (mace-jax dlopens `libcue_ops.so`, which needs
  `libcuda.so.1`). The narrow guard let that escape and take down every importer
  of the module — including the default test suite, which deselects the backend
  but still collects it. A broken native install now degrades to
  `is_available() == False`.
- **Walker-state contract unified at the IO boundary.** `WalkerState.from_record`
  plus format helpers replace the dual-branch handling in `io/formats.py`;
  writers and the monitor now pass typed `WalkerState` throughout. On-disk keys
  are unchanged. Replica-exchange swap dicts remain a deliberately distinct,
  documented protocol.
- **jaxtyping annotations** extended across `sampling/`, `state/`, and the
  replica-exchange code.
- **XLA GPU autotuning pinned to level 0 by default** (`JAXRENS_XLA_AUTOTUNE`
  to override), avoiding effectively unbounded JIT compile times.
- **Module-level JAX constants replaced with numpy** in `init/rejection.py` and
  `init/cells.py`. A module-level `jnp.int32(0)` executes a JAX op at import,
  forcing a backend up — including a CUDA probe that errors on GPU-less nodes —
  and defeating `JAXRENS_SKIP_RUNTIME_CHECKS=1`.
- **Error messages expanded** across 15 modules: 22 terse messages now say what
  the value is for and how to fix it.
- **Test suite restructured** to mirror the package: one directory per
  subpackage, `log_io/` → `io/` (with a `tests/__init__.py` so it no longer
  shadows the stdlib `io`), and `data/` + `fixtures/` consolidated into
  `_assets/`. Backend suites that previously ran under any selection now carry
  a module-level `pytestmark`.
- **README** rebuilt around the project logo, with a runnable quick start.

### Removed
- **BREAKING — `alchemical_shift` move type.** It proposed a *rigid translation
  of the whole cell*, which leaves the energy of any translation-invariant
  potential exactly unchanged. Every production backend is translation
  invariant, so the move accepted ~100% of proposals, changed nothing but the
  origin, burned one full energy evaluation per proposal, and fed a meaningless
  100% acceptance rate into step-size adaptation. Its tests only passed because
  they used an absolute-position harmonic toy potential.
- **BREAKING — `single_atom_swap` move type.** Superseded by `species_swap`. It
  drew two atom indices uniformly and force-rejected same-species draws *after*
  the energy call, wasting `Σ_s x_s²` of all evaluations — 50% for an equimolar
  binary, ~97% for a single solute in 63 host atoms. Migrate to
  `type: species_swap`; note the reported acceptance rate will rise, because the
  old move counted a like-pair draw as a rejection.
- **BREAKING — top-level `jaxrens.base` and `jaxrens.types`.** `MoveInfo` moved
  to `jaxrens.sampling.base`, the `TrajectoryWriter` protocol to
  `jaxrens.io.trajectory`, and the `NSCallback` protocol to
  `jaxrens.sampling.run_loop` (both now `@runtime_checkable` and used as real
  annotations). The unused `StepFn` protocol, the stale `EnergyFn` protocol
  (superseded by `backends.base.EnergyBackend`), and all of `jaxrens.types` are
  deleted.
- **BREAKING — `postprocess.steinhardt` and `postprocess.uncertainty`.**
  Steinhardt order parameters and post-run trajectory uncertainty annotation
  have moved out of jaxrens.

### Fixed
- **`run_ns` single-run path**, and `_check_initial_constraints` against the
  changed types contract.
- **Positional `BackendResult` access** in the CLI resolver, the last
  tuple-style access left from before the typed-result refactor.

## [0.2.1] — 2026-07-07

Bug-fix release for the restart and sharded / multi-replica start paths. No
new features or config changes.

### Fixed
- **Restart now takes precedence over other init modes.** An auto-discovered
  checkpoint (or an explicit `init.restart_file`) could be shadowed when a
  `start_walker_set` / `start_species` / `start_config_file` was also present:
  the resolver checked `start_walker_set` before `restart_file`, and checkpoint
  auto-discovery injected `restart_file` without clearing the competing init
  fields. A resumed run could silently re-initialise from scratch instead of
  from the checkpoint. Restart is now resolved first, and auto-discovery clears
  the other init modes.
- **Sharded runs no longer re-run initial burn-in on restart.**
  `run_sharded_from_config` executed the burn-in walk unconditionally, throwing
  away the checkpointed walkers' equilibration. Burn-in is now skipped when a
  restart state is present, and the restart state is threaded through to the
  sharded run.
- **Correct initial energies and constraint checks on the batched paths.**
  `initial_types` is now carried as a per-walker `(n_live, n_atoms)` array
  throughout the resolver; the initial-energy finalize seam and the
  initial-constraint check map it over the walker (and replica) axes alongside
  positions/cells instead of closing over a single shared `(n_atoms,)` vector.
  This fixes `vmap`/`pmap` rank errors in the multi-replica and sharded
  initial-energy compute and lets composition vary per replica (semi-grand μPT
  / alchemical).

## [0.2.0] — 2026-06-25

Adds composable constraints and a semi-grand μPT ensemble, fixes a periodic
neighbor-graph correctness bug, and retires the `ns.inp` migration path. Note
the **breaking changes** below (`run.pressure` and `migrate-ns-inp` removed).

### Added
- **Semi-grand μPT ensemble.** New `ensemble: {type: semi_grand}` spec taking a
  per-species `chemical_potentials` vector (plus optional `pressure`); the
  backend applies `H = U + P·V − μ·N`. Works for a single run or fanned out
  across replicas, with chemical potentials threaded through the generic
  `ensemble_params` dict end-to-end.
- **Composable configuration-constraints framework** (`constraints/`):
  minimum-interatomic-distance and cell-geometry constraints, gated centrally
  in the MWG move kernel via a flexible `mutates`/`depends_on` contract.
- **Step sizes persisted across restarts.** Checkpoints now store adapted
  per-move step sizes; a resumed run continues from the converged step instead
  of re-running burn-in adaptation from the configured initial step.

### Changed
- **Ensemble parameters are now fully generic.** All ensemble corrections flow
  through a single per-run `ensemble_params` dict from resolver to runtime;
  `NSConfig` no longer carries ensemble-specific fields. Initial-energy
  computation is unified to use the same dict, so resolved energies match the
  NS loop by construction.
- **Trajectory atoms are wrapped into the cell by default** (`output.wrap_atoms`
  now defaults to `True`); both the extxyz and h5 writers honor it. Atoms drift
  arbitrarily far from the cell over a run, so unwrapped output was rarely
  useful. Set `wrap_atoms: false` to keep absolute Cartesians.
- **Backend dependency `ImportError`s** now name the install extra
  (`pip install '.[mace]'`, etc.) instead of only echoing the import failure.
- `mace-jax` dependency rewired to the maintained fork.
- Internal refactors: unified pytree-registration helpers, an h5-logger
  lifecycle base class, deduplicated move/cell/softcore helpers, and a
  consolidated initial-energy path.

### Removed
- **`run.pressure`** (breaking). Use `ensemble: {type: npt, pressure: …}`.
- **`ns.inp → YAML` migration** (breaking): the `migrate-ns-inp` CLI
  subcommand and the `cli/migrate.py` + `cli/parser.py` modules.

### Fixed
- **Periodic neighbor-graph correctness in the GNN backends** (MACE/nequix):
  atoms are now wrapped into the unit cell before supercell neighbor search.
  Previously, atoms that had drifted out of the cell could silently lose edges
  and yield a wrong, origin-dependent energy. No-op for non-periodic systems.
- **`format: h5` trajectories** no longer crash at writer construction
  (`H5TrajectoryWriter` now accepts the `wrap` argument the run path passes).

## [0.1.0] — 2026-06-22

First deliberately versioned release. Prior development was never tagged; this
draws the line and switches the package version over to git tags.

### Added
- JAX-based nested sampling for atomistic systems.
- YAML config pipeline (`cli/`): pydantic `schema/` → `resolve.py` runtime
  dataclasses → `run.py` callback/writer wiring, plus `migrate.py` to port
  legacy `ns.inp` configs into the new schema.
- Pluggable potential backends behind a typed `BackendResult` boundary
  (mace-jax, NeuralIL, nequix, jax-md), selected via optional-dependency
  extras.
- Pluggable `TrajectoryWriter` implementations (`extxyz` / `h5` / `none`) in
  `io/trajectory.py`.
- Post-hoc uncertainty evaluation.

### Changed
- The package version is now derived from git tags (`setuptools-scm`) instead
  of a hardcoded literal; `jaxrens.__version__` reads installed package
  metadata via `importlib.metadata`.
