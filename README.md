<p align="center">
  <img src="docs/_static/jaxrens_logo.svg" alt="jaxrens" width="540">
</p>

<p align="center">
  <b>Nested sampling for atomistic systems, in JAX.</b><br>
  Multi-GPU parallel replicas · pressure, composition and semi-grand replica exchange · pluggable ML potentials
</p>

<p align="center">
  <a href="https://github.com/nunglert/jaxrens/actions/workflows/ci.yml"><img src="https://github.com/nunglert/jaxrens/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/github/nunglert/jaxrens"><img src="https://codecov.io/github/nunglert/jaxrens/graph/badge.svg?token=KY8R8JZ9FC" alt="codecov"></a>
  <a href="https://nunglert.github.io/jaxrens/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="Docs"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
</p>

---

## Quick start

```bash
pip install -e .
jaxrens run -c examples/lennard_jones/single_run/config.yaml
```

That runs an 8-atom Lennard-Jones solid at constant pressure — a few minutes on
one GPU — writing a trajectory, energy log, and checkpoints to `./output/`.

A config is one YAML file. The pieces you will touch most:

```yaml
run:
  n_live: 128            # walkers in the live set
  n_mcmc_steps: 20       # decorrelation steps per NS iteration
  max_iterations: 2000

backend:
  type: lj               # or: mace, neuralil, nequix, jaxmd
  epsilon: 1.0
  sigma: 1.0
  cutoff: 2.5

ensemble:
  type: npt              # nvt | npt | semi_grand
  pressure: 0.1

moves:                   # mixed Metropolis-within-Gibbs move set
  - {type: galilean, n_reflect: 10, step_size: 0.1, weight: 4.0}
  - {type: volume,   step_size: 0.3, weight: 1.0}

termination:
  - {type: prior_mass, threshold: 1.e-5}
```

Validate before you burn GPU hours, and override single keys from the command
line:

```bash
jaxrens validate -c config.yaml
jaxrens run -c config.yaml --set run.n_live=256 --set moves[0].step_size=0.05
jaxrens dump-schema > schema.json      # editor autocomplete for the YAML
```

More: [`examples/lennard_jones/`](examples/lennard_jones/) (single run and
replica exchange, each with a plotting script) and
[`examples/tutorials/`](examples/tutorials/).

## How it works

Two-loop nested sampling: a JIT-compiled inner `lax.scan` does the MCMC walk,
while a Python outer loop handles step-size adaptation, kernel dispatch,
callbacks, and termination.

Three levels of parallelism via `pmap(vmap(vmap(...)))`, with population data
shaped `(G, P, K, ...)`:

| axis | meaning | set by |
| --- | --- | --- |
| `G` | pmap axis across GPUs | `run.shard_n_gpu` |
| `P` | independent NS runs (replicas) | a *list* of pressures / μ-vectors under `ensemble` |
| `K` | walkers in the live set | `run.n_live` |

Independent runs can exchange configurations mid-flight: RENS (pressure),
XRENS (composition-morphing), and semi-grand swaps are all built in.

## Backends

| backend | extra | notes |
| --- | --- | --- |
| Lennard-Jones | — | built in, incl. per-species ε/σ tables |
| toy potentials | — | harmonic, double-well, Gaussian mixture (known-answer tests) |
| [MACE-JAX](https://github.com/nunglert/mace-jax) | `[mace]` | converted from a Torch MACE model |
| [NeuralIL](https://github.com/nunglert/neuralil-jaxrens) | `[neuralil]` | |
| [nequix](https://github.com/atomicarchitects/nequix) | `[nequix]` | |
| [jax-md](https://github.com/jax-md/jax-md) | `[jaxmd]` | Tersoff / EAM |

NPT and semi-grand μPT come from an `EnsembleBackend` wrapper that adds the
`P·V` and `−μ·N` terms, so every backend gets them for free.

## Install

```bash
pip install -e .                    # core
pip install -e ".[all]"             # every backend + MACE conversion tooling
pip install -e ".[dev,docs]"        # test + docs tooling
pip install -e ".[mace]"            # or pick backends individually:
pip install -e ".[neuralil]"        #   neuralil, nequix, jaxmd
```

Requires Python ≥ 3.11. Production runs need a CUDA 12 capable GPU; the CPU
path works but is untested and warns on startup.

## Tests

```bash
pytest tests/                              # default suite
pytest tests/ -n 4                         # one process per GPU (see CLAUDE.md)
pytest tests/ -m heavy                     # include slow tests
pytest tests/ -m multi_gpu                 # multi-GPU parity (needs ≥2 GPUs)
pytest tests/ -m mace                      # a single backend suite
```

Markers: `heavy`, `gpu`, `multi_gpu`, `lj`, `mace`, `neuralil`, `nequix`,
`jaxmd`. The default run excludes `heavy`, `multi_gpu`, and every per-backend
opt-in, so a machine without the optional extras still gets a clean suite.

Optional backends normally degrade to skips when absent. Because skips do not
change the exit code, a broken install could otherwise pass as green — so CI
names what it installed and turns a missing backend into a hard failure:

```bash
pytest tests/ --require-backends=mace,neuralil      # or =all
```

Layout mirrors the package: one test directory per subpackage, plus
`_assets/` (committed models and reference data), `integration/`, and `smoke/`.

## CI

GitHub Actions on [RunsOn](https://runs-on.com/) ephemeral GPU runners, routed
by ref:

- **push to a non-`main` branch** → *cheap* (g4dn.xlarge): smoke + integration
  only, the fast narrow-down signal.
- **PR to `main` / push to `main` / manual dispatch** → *full* (one
  g4dn.12xlarge, 4× T4): the whole suite under `-n 4` with coverage to Codecov,
  then the `multi_gpu` tests — both sequentially on the same instance, so a run
  costs one GPU node rather than two.

## Docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

User guide, API reference, tutorials, and developer notes are under `docs/`,
published at [nunglert.github.io/jaxrens](https://nunglert.github.io/jaxrens/).

## License

MIT. See the `license` field in `pyproject.toml`.
