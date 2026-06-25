# Changelog

All notable changes to `jaxrens` are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/). Versions follow
semantic versioning; while in `0.x`, **minor** releases may include breaking
changes and **patch** releases are bug fixes only.

The version is derived from git tags by `setuptools-scm` (tag `v0.1.0` →
version `0.1.0`). To cut a release: add a dated section below, merge to `main`,
then `git tag -a vX.Y.Z`.

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
