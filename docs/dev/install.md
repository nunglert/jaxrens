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
pyyaml, and the dev tooling (`pytest`, `pytest-xdist`, `pre-commit`, ...).
`pip install -e` is additive — installing another extra afterwards
doesn't remove what `[dev]` already put in the environment, so the
snippets below only add the extra they need. Drop `dev` from any of
them if you just want to run jaxrens rather than test or contribute
to it.

## Optional backends

### MACE-JAX

The `[mace]` extra installs the **runtime** MACE backend — everything
needed to *load and evaluate* a converted JAX model:

```bash
pip install -e ".[mace]"
```

To also **convert** a Torch MACE checkpoint (or download + convert a
foundation model) into a JAX model, add the `[mace-convert]` extra,
which pulls the Torch-side `mace-torch` package the converter imports:

```bash
pip install -e ".[mace-convert]"
```

`[all]` installs every optional backend plus the conversion tooling in
one shot. See the {doc}`../user/mace_models` guide for the full
download → convert → run workflow.

The upstream `ACEsuit/mace-jax` package at the time of writing has
two packaging bugs:

- `mace_jax/adapters/` and `mace_jax/adapters/e3nn/` lack
  `__init__.py`, so setuptools' `packages = find:` silently drops
  the subtree from the wheel.
- The `[torch]` extra references a non-existent PyPI package
  (`cuequivariance-ops-torch` — the right name is
  `cuequivariance-ops-torch-cu12`).

jaxrens' `pyproject.toml` pins a patched fork
(`nunglert/mace-jax@fixes`) that resolves both. If upstream lands the
fixes, the pin can drop back to `ACEsuit/mace-jax@main`.

### NeuralIL

```bash
pip install -e ".[neuralil]"
```

Pulls a jaxrens-compatible fork (`nunglert/neuralil-jaxrens@jaxrens`).

### Nequix

```bash
pip install -e ".[nequix]"
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
compile cache. SLURM submit scripts can redirect
temp I/O into the job directory via `TMPDIR`:

```bash
export TMPDIR="${PWD}/tmp/${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export JAX_COMPILATION_CACHE_DIR="${PWD}/.jax_cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"
trap 'rm -rf "$TMPDIR"' EXIT
```

Copy that block into any new submit script before launching JAXRENS.
