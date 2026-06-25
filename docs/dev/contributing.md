# Contributing

Thanks for considering a contribution! This page sketches how we expect
changes to land: how the project is laid out, the conventions we follow, and
what a reviewable pull request looks like.

## Development setup

Install the package editable with the development extras and enable the
pre-commit hooks:

```bash
git clone git@github.com:nunglert/jaxrens.git
cd jaxrens
pip install -e ".[dev]"      # pytest + xdist/cov/timeout + beartype + pre-commit
pre-commit install           # run black/isort/eof checks on every commit
```

Add the relevant backend extra(s) only if you touch code that imports them
(see {doc}`install`). For docs work, add `docs` and use the live preview:

```bash
pip install -e ".[docs]"
bash docs/autobuild.sh       # sphinx-autobuild docs docs/_build/html -j auto
```

## Where code belongs

The library lives under `src/jaxrens/`; please keep contributions on the right
side of these boundaries (see {doc}`../reference/index` for the module map):

`cli/`
: The YAML config pipeline. Data flows one way:
  `schema/` (pydantic specs) → `resolve.py` (specs → runtime dataclasses in
  `state/config.py`) → `run.py` (wires callbacks/writers). Keep validation in
  the schema layer and runtime wiring in `run.py`.

`io/trajectory.py`
: Pluggable `TrajectoryWriter` implementations (`extxyz` / `h5` / `none`).
  New output formats go here behind the same protocol.

Potential backends
: Each backend returns a typed `BackendResult` and is gated behind an optional
  dependency extra (`mace`, `neuralil`, `nequix`, `jaxmd`). A module that needs
  `mace_jax` / `neuralil` / `nequix` / `jax-md` at import time belongs behind
  its extra, never in the always-imported core.

`import jaxrens` deliberately does **not** import JAX — it exposes only
`__version__`. The float32 precision config runs before any JAX op via
`jaxrens._jax_init`, imported at the top of every JAX-using subpackage. Don't
add a top-level JAX import that would break that contract.

## Code style

Formatting is enforced by pre-commit, so you rarely need to think about it —
but for reference:

- **black** and **isort** (`--profile black`), **line length 79**.
- `end-of-file-fixer` and a large-file guard (`--maxkb=40000`) also run.
- Run everything manually with `pre-commit run --all-files`.

A few project-specific conventions the formatters can't catch:

- **Runtime type checks are on in tests.** The default pytest run enables
  jaxtyping + beartype over `jaxrens.state` and `jaxrens.sampling`, so array
  shape/dtype annotations are checked at call time. Annotate array arguments
  with jaxtyping (e.g. `Float[Array, "n_live n_atoms 3"]`) and keep them
  honest — a wrong annotation becomes a test failure.
- **`BatchDescriptor` instances are named `batcher`** (variables, attributes,
  and info-dict keys), for consistency across the codebase.
- **Respect the dependency floors** (`jax`, `numpy`, `ase`, `pydantic`, … in
  the core deps; the git-pinned forks in the backend extras). If you touch a
  backend, match the version its extra pins.

### Docstrings

Docstrings are Google-style and rendered by Sphinx + napoleon (plus
autodoc-pydantic for the schema models). To keep the docs build clean:

- Put a **blank line before** `Args:` / `Returns:` / `Example:` sections.
- Don't hand-write `Attributes:` sections that just relist members — autodoc
  already documents them. Document a dataclass field with an inline `#:`
  comment instead.
- Use raw strings (`r"""`) for docstrings containing backslashes (e.g. LaTeX
  or regex examples).

Check the docs build before submitting doc changes:

```bash
sphinx-build -b html --keep-going docs docs/_build/html
```

## Testing

Tests live in `tests/` and mirror the package layout. Backend-heavy tests are
marker-gated; the markers are registered in `pyproject.toml`:

*(unmarked)*
: Pure-logic unit tests (CLI / schema / IO / resolver). No GPU or backend
  needed. Always run.

`gpu` / `multi_gpu`
: Need one / two-or-more CUDA devices. `multi_gpu` is opt-in via
  `-m multi_gpu`.

`mace` / `neuralil` / `nequix` / `jaxmd`
: Exercise a specific potential backend; each is **opt-in only** and needs the
  matching extra installed.

`heavy`
: Slow tests (long example runs).

The default run (`pytest`) deselects `heavy`, `multi_gpu`, and all backend
markers, and turns on the jaxtyping/beartype runtime checks. Useful
invocations:

```bash
# CPU-only, fast & deterministic — good for CLI/schema/IO/resolver work
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" pytest tests/cli -q

# Multi-GPU dev box: one process per device (check `nvidia-smi` count first)
pytest -n 2

# Opt into a backend or the multi-GPU suite
pytest -m mace        # needs the `mace` extra + a GPU
pytest -m multi_gpu   # needs >=2 local devices
```

What we expect of a contribution:

- **New behavior comes with a test**, placed in the lightest tier that can
  exercise it — prefer an unmarked CPU unit test over a backend/GPU one when
  the logic can be isolated.
- **Tests must self-skip, never hard-fail, when a capability is missing** (no
  GPU, no backend package). Follow the existing marker conventions.

## Submitting changes

- Branch off `main`; keep the change focused.
- Make sure `pre-commit run --all-files` and `pytest` are green locally, and
  that the Sphinx build is warning-free if you touched docs or docstrings.
- Open a pull request against `main` with a short description of *what* and
  *why*. CI (GitHub Actions) runs pytest in a Docker CI image — smoke /
  integration on cheap-GPU branches, the full suite plus coverage on PRs and
  `main`, and an opt-in multi-GPU job — and builds the docs.

## Versioning and releases

The package version is **derived from git tags** by `setuptools-scm` — there is
no version string to edit. A clean checkout on tag `v0.1.0` reports `0.1.0`;
commits past a tag report a dev version like `0.1.1.dev3+g<sha>`. Where the git
history isn't available (the `.git`-less Docker CI build, a shallow clone, a
tarball) the build falls back to `0.0.0` rather than erroring. At runtime,
`jaxrens.__version__` reads installed package metadata via `importlib.metadata`.

Semantic versioning applies, but **while in 0.x**:

- **minor** (`0.1 → 0.2`) — new features *or* breaking changes;
- **patch** (`0.1.0 → 0.1.1`) — bug fixes only.

(Once the public API and config schema settle, cut `1.0.0` and switch to full
SemVer, where breaking changes require a major bump.)

To cut a release:

1. Add a dated section to `CHANGELOG.md` (note anything under **Breaking** so
   downstream consumers know what to fix).
2. Merge to `main`.
3. Tag the release commit on a clean tree and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a v0.2.0 -m "jaxrens 0.2.0"
   git push origin v0.2.0
   ```

Downstream projects should pin jaxrens to a tag
(`jaxrens @ git+https://github.com/nunglert/jaxrens@v0.2.0`) rather than a
branch, so upgrades are deliberate.
