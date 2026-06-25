# jaxrens

[![codecov](https://codecov.io/github/nunglert/jaxrens/graph/badge.svg?token=KY8R8JZ9FC)](https://codecov.io/github/nunglert/jaxrens)
[![CI](https://github.com/nunglert/jaxrens/actions/workflows/ci.yml/badge.svg)](https://github.com/nunglert/jaxrens/actions/workflows/ci.yml)

JAX-based nested sampling for atomistic systems — multi-GPU parallel replicas,
pressure / composition / semi-grand inter-replica exchange (RENS / XRENS /
semi-grand), and a pluggable backend interface.

## What it does

Two-loop nested sampling with a JIT-compiled inner `lax.scan` and a Python
outer loop for adaptation, kernel dispatch, callbacks, and termination.
Three-level parallelism (`pmap(vmap(vmap(...)))`) with data shaped
`(G, P, K, ...)`:

- `G` = `n_gpu_parallel` (pmap axis across GPUs)
- `P` = `n_runs_per_gpu` (independent NS runs)
- `K` = `n_walkers` (walkers within a run)

Pluggable energy backends: Lennard-Jones (incl. per-species LJ tables),
[MACE-JAX](https://github.com/nunglert/mace-jax), [NeuralIL](https://github.com/nunglert/neuralil-jaxrens),
[nequix](https://github.com/atomicarchitects/nequix), and toy potentials.
NPT and semi-grand μPT supported via an `EnsembleBackend` wrapper that adds
the `P·V` and `−μ·N` terms.

## Install

```bash
pip install -e .                  # core
pip install -e .[dev,docs]        # dev tooling + Sphinx docs
pip install -e .[neuralil]        # optional: NeuralIL backend
pip install -e .[mace]            # optional: MACE-JAX backend
pip install -e .[nequix]          # optional: nequix backend
```

Requires Python ≥ 3.11 and a CUDA 12 capable GPU for production runs.

## CLI

`jaxrens` is installed as a console script (also reachable via
`python -m jaxrens.cli.cli`):

```bash
jaxrens validate -c config.yaml          # schema-check a YAML config
jaxrens run      -c config.yaml          # run NS with the config
jaxrens dump-schema                      # JSON schema for editor autocomplete
jaxrens run -c config.yaml --set run.n_live=64 --set moves[0].step_size=0.05
```

Example configs live under `experiments/examples/` (e.g. `lj8_npt/`).

## Tests

```bash
pytest tests/                                  # default suite
pytest tests/ -m heavy                         # include slow tests
pytest tests/integration -m multi_gpu          # multi-GPU parity (needs ≥2 GPUs)
pytest tests/ -m mace                          # MACE backend (needs mace-jax + GPU)
pytest tests/ -m neuralil                      # NeuralIL backend
pytest tests/ -m nequix                        # nequix backend
```

Markers: `heavy`, `gpu`, `multi_gpu`, `mace`, `neuralil`, `nequix`. The default
run excludes `heavy`, `multi_gpu`, and the per-backend opt-ins.

## CI

The `CI` workflow (`.github/workflows/ci.yml`) routes by ref on
[RunsOn](https://runs-on.com/) GPU runners:

- non-`main` push → **cheap** (g4dn.xlarge): smoke + integration only
- PR / push to `main` / manual dispatch → **full** (g5.xlarge, full pytest with
  coverage upload to Codecov) **+ multi_gpu** (g4dn.12xlarge, 4× T4) in parallel

## Docs

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

The User Guide, API reference, and developer notes are under `docs/`.
