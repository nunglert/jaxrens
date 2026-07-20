# Changelog

All notable changes to `jaxrens` are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/). Versions follow
semantic versioning; while in `0.x`, **minor** releases may include breaking
changes and **patch** releases are bug fixes only.

The version is derived from git tags by `setuptools-scm` (tag `v0.1.0` →
version `0.1.0`). To cut a release: add a dated section below, merge to `main`,
then `git tag -a vX.Y.Z`.

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
