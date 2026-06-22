# Changelog

All notable changes to `jaxrens` are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/). Versions follow
semantic versioning; while in `0.x`, **minor** releases may include breaking
changes and **patch** releases are bug fixes only.

The version is derived from git tags by `setuptools-scm` (tag `v0.1.0` →
version `0.1.0`). To cut a release: add a dated section below, merge to `main`,
then `git tag -a vX.Y.Z`.

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
