# Using MACE models

This guide takes you from *"I have (or want) a MACE potential"* to a running
nested-sampling job. It covers the three things that trip people up: getting a
JAX-loadable model out of a Torch MACE checkpoint, the small zoo of on-disk
formats, and the YAML block that wires the model into a run.

For the *why* behind the neighbor-buffer and species-mapping machinery this
backend uses, see {doc}`concepts/backends`. For install specifics (extras, the
fork pin, the global float32 pin) see {doc}`../dev/install`.

## The big picture

jaxrens' MACE backend only ever **consumes a JAX model** at runtime — it
imports `mace_jax` + `flax` and never Torch. The conversion from a Torch MACE
checkpoint to JAX parameters happens once, out-of-band, via the
`mace-jax-from-torch` console script that ships with `mace-jax`. After that,
you point a YAML `checkpoint_path` at the converted files and never think about
Torch again.

```{mermaid}
flowchart LR
    T["Torch MACE<br/>(.model / foundation download)"] -->|mace-jax-from-torch| J["JAX params<br/>(.npz) + config (.json)"]
    J -->|create_mace| B["MACEBackend"]
    B --> NS["nested sampling run"]
```

## Step 1 — get a JAX model

You have two starting points.

### A) Convert your own Torch checkpoint

```bash
mace-jax-from-torch --torch-model path/to/your_model.model --output my_model-jax.npz
```

### B) Download + convert a foundation model

Omit `--torch-model` and name a foundation family + variant; the Torch
checkpoint is downloaded, then converted:

```bash
mace-jax-from-torch --foundation mp --model-name small --output small-jax.npz
```

Valid `--foundation` families are `mp` / `off` / `anicc` / `omol`. For
`--foundation mp` the `--model-name` variants are:

```
small  medium  large
small-0b  medium-0b
small-0b2  medium-0b2  large-0b2
medium-0b3
medium-mpa-0        # the default when --model-name is omitted
small-omat-0  medium-omat-0
mace-matpes-pbe-0  mace-matpes-r2scan-0
mh-0  mh-1
```

Regenerate this list any time with:

```bash
python -c "from mace_jax.tools.foundation_models import get_mace_mp_names; print([n for n in get_mace_mp_names() if n])"
```

### Prerequisites for conversion

Conversion is the one step with Torch-side requirements:

- **The converter dependency.** `mace-jax-from-torch` imports `mace` (the
  `mace-torch` package), which the runtime `[mace]` extra does *not* pull.
  Install the convert extra: `pip install -e ".[mace-convert]"` (or `".[all]"`).
- **A GPU node.** `mace-jax` imports `cuequivariance`, which dlopens
  `libcuda` at import time — so the converter won't even start on a login node
  without an NVIDIA driver. Run it inside a GPU allocation
  (`srun --gres=gpu:1 …`). This is an import-time constraint, not because
  conversion does heavy GPU math.
- **CA certificates**, only when *downloading* a foundation model. Some hosts
  ship no default CA path and the download fails with
  `CERTIFICATE_VERIFY_FAILED`. Point at certifi's bundle:

  ```bash
  export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
  ```

A complete, non-destructive download-convert-load cycle is pinned by
`tests/backends/test_mace.py::TestConversionRoundtrip` — read it for the exact
invocation.

## Step 2 — understand what you got (the format zoo)

The converter writes **two files**: params and a sibling config.

```
small-jax.npz     # the parameters — msgpack bytes, NOT a numpy .npz (misnomer)
small-jax.json    # the model config
```

```{warning}
That `.npz` is a msgpack blob, not a numpy archive — don't try to `np.load` it.
It also only resolves together with its **same-stem** `.json` sibling: keep the
pair side by side, and if you rename one, rename both.
```

There are a handful of other shapes a MACE model can arrive in — a bundle
directory, a loose `config.json` / `params.msgpack` pair, an orbax `.ckpt`, or
a `.pkl` carrying a pre-split `graphdef` + `state`. You don't need to memorize
them, because a single loader handles all of them (next step).

## Step 3 — point a run at it

{func}`~jaxrens.backends.mace.create_mace` is the **single front door**: give
it a path and it dispatches by shape. Every format from Step 2 works, so the
YAML `checkpoint_path` is uniform:

| What you have on disk | `checkpoint_path` value |
|---|---|
| Converter output | `.../small-jax.npz` (the `.json` sits beside it) |
| Bundle directory | `.../my_bundle` (contains `config.json` + `params.msgpack`) |
| Loose config/params pair | either `.../model.json` or `.../model.msgpack` |
| Orbax checkpoint | `.../model.ckpt` |
| Pre-split pickle | `.../model_bundle.pkl` |

A minimal MACE backend block in a run config:

```yaml
backend:
  type: mace
  checkpoint_path: /path/to/small-jax.npz   # any shape from the table above
  supercell_trafo: [3, 3, 3]                 # neighbor-finding supercell
  periodic: true
  max_neighbors_list: [35, 50, 65, 85, 100]  # allowed edge-buffer buckets
  max_neighbors_offset: 4                     # headroom on the initial bucket
```

Then run it the usual way:

```bash
jaxrens run -c config.yaml
```

### The knobs that matter

- **`supercell_trafo: [sa, sb, sc]`** — how many times the cell is tiled for
  neighbor finding. It must be large enough that
  `min(cell_diagonal · s) ≥ 2 · r_cutoff`; too small and pairs across the
  periodic boundary are missed. MACE-MP models have `r_cutoff = 6 Å`, so a
  ~4 Å cell needs `s ≥ 4` per short axis.
- **`max_neighbors_list` / `max_neighbors_offset`** — the edge-buffer bucket
  allowlist and its safety margin. These are shared by every MLIP backend and
  explained in full (with the overflow-escalation diagram) in
  {doc}`concepts/backends`.

### Species are mapped to the model's z-table

You always write **atomic numbers** in YAML (`init.start_species: "14 16"` for
16 Si atoms). For MACE, the resolver maps each Z to its slot in the *model's*
z-table rather than reindexing to `[0, n_unique)` — the backend exposes an
`atomic_numbers` attribute and jaxrens honors it. You don't have to do anything;
it's automatic. See the "Species indexing" section of {doc}`concepts/backends`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'mace'` running the converter | The `[mace]` runtime extra doesn't include Torch-side `mace-torch`. Install `".[mace-convert]"`. |
| `libcuda.so.1: cannot open shared object file` | You're on a login node. `cuequivariance` needs a driver at import; run inside a GPU allocation. |
| `CERTIFICATE_VERIFY_FAILED` when downloading a foundation model | No default CA path. `export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')`. |
| `Unable to locate JAX model configuration at …json` | The `.npz`/`.msgpack` params file lost its same-stem `.json` sibling. Keep them together. |
| Energies look wrong / neighbor overflow every step | `supercell_trafo` too small for `2·r_cutoff`, or `max_neighbors_list` buckets too small. Raise both. |

## See also

- {doc}`concepts/backends` — the `EnergyBackend` protocol, neighbor buckets,
  ensembles, species indexing.
- {doc}`../dev/install` — the `[mace]` / `[mace-convert]` / `[all]` extras and
  the mace-jax fork pin.
- {doc}`../reference/config` — the full YAML schema, including every backend
  field.
- The `01_mace_run` tutorial — a runnable end-to-end MACE NS walkthrough.
