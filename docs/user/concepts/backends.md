# Backends and ensembles

An *energy backend* is anything that maps atomic positions + species
+ cell to an energy. A thin wrapper,
{class}`~jaxrens.backends.ensemble.EnsembleBackend`, converts that
bare potential into the *effective energy* appropriate for the
thermodynamic ensemble. Moves see only the wrapped backend; the
rest of the library doesn't care which ensemble is active.

## The `EnergyBackend` protocol

```{mermaid}
flowchart LR
    M["Move kernel"] --> B["EnsembleBackend(base)"]
    B --> Base["base.EnergyBackend<br/>(lj / mace / neuralil / …)"]
    Base --> U["U  (bare potential)"]
    B --> H["H = U + PV − μ·N<br/>(ensemble-corrected)"]
    H --> M
```

Every backend — LJ, MACE-JAX, NeuralIL, and the toy potentials —
exposes the same call signature:

```python
energy, neighbor_count, overflow = backend(
    positions,        # (A, 3)
    species,          # (A,)   z-table indices
    cell,             # (3, 3) lattice vectors as rows
    max_neighbors,    # int    MLIP-side buffer size
    ensemble_params,  # dict   {"pressure": ..., "mu": ...} or None
)
```

Returning a `neighbor_count` and `overflow` flag alongside the
energy is how the NS outer loop handles MLIP neighbor-buffer
overflow: when a move pushes atoms closer together than
`max_neighbors` can hold, the backend signals the overflow, the
loop retries at a bigger bucket, and JAX reuses the JIT'd kernel
compiled for that bucket size.

Built-in backends:

| `type:` | Class | Notes |
|---|---|---|
| `lj` | {class}`jaxrens.backends.lj.LJBackend` | periodic or finite, static cutoff |
| `mace` | {class}`jaxrens.backends.mace.MACEBackend` | mace-jax, supercell neighbor expansion, pins float32 globally (see {doc}`../../dev/install`) |
| `neuralil` | `jaxrens.backends.neuralil.NeuralILBackend` | bucketed `max_neighbors_list` pre-compilation |
| `harmonic`, `double_well`, `gaussian_mixture` | `jaxrens.backends.toy` | analytical test potentials |

## Ensembles as additive corrections

The `EnsembleBackend` wraps a base backend and adds the appropriate
thermodynamic term **per call**, reading the ensemble parameters
from the `ensemble_params` dict. This means one wrapper instance
serves multiple replicas at different pressures / chemical
potentials — no rebuild needed.

For a configuration $\theta = (\vec r, \mathbf{L}, \vec\tau)$ with
volume $V = |\det(\mathbf{L})|$ and species counts $N_i$,

$$
H_\mathrm{NVT}(\theta) = U(\theta),
$$
$$
H_\mathrm{NPT}(\theta) = U(\theta) + P\,V,
$$
$$
H_{\mu V T}(\theta) = U(\theta) + P\,V - \sum_i \mu_i N_i.
$$

The NS loop always works with $H$, not $U$. Same sampler, same
move kernels, same acceptance criterion; the ensemble only changes
what gets compared to $E_\mathrm{max}$.

```{image} /_static/figures/ensemble_tilt.png
:alt: H(V) curves for NVT, NPT, muVT
:align: center
```

Physically: the $+PV$ term pushes walkers toward smaller volumes at
high pressure (compressing the system), and the $-\mu N$ term
favours adding atoms when $\mu > 0$ (grand-canonical chemistry).

## Species indexing

Two conventions coexist in jaxrens:

- **Zero-based unique-Z mapping** — for LJ and toy backends, the
  resolver takes the unique atomic numbers from `start_species` and
  reindexes them to `[0, n_unique)`.
- **Backend-native z-table** — for MACE (and any backend exposing
  an `atomic_numbers` attribute), the resolver maps each Z to its
  position in the model's z-table. For the `mace_mp_small` fixture
  that's the standard periodic-table order, so Sr → 37, Ti → 21,
  O → 7.

The choice happens automatically in
`cli.resolve._resolve_init_species`; user-facing YAML always uses
atomic numbers. Symbol tables (`symbol_map`) track the inverse.

## Where this lives in the code

| Concern | File |
|---|---|
| Backend protocol | {class}`jaxrens.backends.base.EnergyBackend` |
| Ensemble wrapper | {class}`jaxrens.backends.ensemble.EnsembleBackend` |
| Backend loader (by type string) | {func}`jaxrens.backends.loader.load_backend` |
| MACE backend construction | {func}`jaxrens.backends.mace.create_mace` |
| Species mapping fix | `cli/resolve.py::_resolve_init_species` |
