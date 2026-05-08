# Installation

## Base install

jaxrens targets Python 3.11+ and requires JAX with CUDA 12 for GPU
execution. On CPU it runs fine but large atomistic runs will be slow.

```bash
conda create -n jaxrens python=3.11 -y
conda activate jaxrens
cd /path/to/jaxrens
pip install -e ".[dev]"
```

This pulls `jax[cuda12]`, `jaxlib`, numpy, h5py, ASE, pydantic,
pyyaml, and the dev tooling (`pytest`, `pytest-xdist`).

## Optional backends

### MACE-JAX

```bash
pip install -e ".[dev,mace]"
```

The upstream `ACEsuit/mace-jax` package at the time of writing has
two packaging bugs:

- `mace_jax/adapters/` and `mace_jax/adapters/e3nn/` lack
  `__init__.py`, so setuptools' `packages = find:` silently drops
  the subtree from the wheel.
- The `[torch]` extra references a non-existent PyPI package
  (`cuequivariance-ops-torch` — the right name is
  `cuequivariance-ops-torch-cu12`).

jaxrens' `pyproject.toml` pins a patched fork
(`nunglert/mace-jax@fix_install`) that resolves both. If upstream
lands the fixes, the pin can drop back to `ACEsuit/mace-jax@main`.

### NeuralIL

```bash
pip install -e ".[dev,neuralil]"
```

Pulls a jaxrens-compatible fork (`nunglert/neuralil-jaxrens@jaxrens`).

### Nequix

```bash
pip install -e ".[dev,nequix]"
```

Pulls upstream `atomicarchitects/nequix@main`. Optional backend
that exposes the same `EnergyBackend` protocol as MACE/NeuralIL;
nequix tests sit behind the `nequix` pytest marker (off by
default, opt-in via `pytest -m nequix`).

### Docs

```bash
pip install -e ".[docs]"
```

Adds Sphinx, Furo, MyST, nbsphinx, jupytext, sphinx-design,
sphinx-copybutton, and autodoc-pydantic. See
{doc}`../tutorials/index` for the executable tutorials and the
docs `README` for build instructions.

## GPU sanity check

```bash
python -c "import jax; print(jax.devices())"
```

Should list one or more `CudaDevice(...)` entries. If it prints
`CpuDevice(...)` only, JAX fell back to CPU — check
`nvidia-smi`, `$LD_LIBRARY_PATH`, and that `jaxlib` installed
the CUDA build (`pip show jaxlib` should show a `+cuda12...` tag
on the version).

## Running the test suite

```bash
cd /path/to/jaxrens
pytest tests/ -v
```

Default run excludes the `heavy` marker. To include:

```bash
pytest tests/ -v -m heavy
pytest tests/ -v -m ""    # all markers, including heavy / gpu
```

Markers:

- `heavy` — slow tests (long example runs).
- `gpu` — requires CUDA.
- `multi_gpu` — requires 2+ CUDA devices.

## Scratch / temp paths on SLURM

Some GPU nodes deny writes to `/tmp`, which breaks XLA's PTX
compile cache. SLURM submit scripts in `experiments/` redirect
temp I/O into the job directory via `TMPDIR`:

```bash
export TMPDIR="${PWD}/tmp/${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export JAX_COMPILATION_CACHE_DIR="${PWD}/.jax_cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"
trap 'rm -rf "$TMPDIR"' EXIT
```

Copy that block into any new submit script.
