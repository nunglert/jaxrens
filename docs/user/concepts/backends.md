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

## The neighbor problem

Every modern MLIP — MACE, NeuralIL, NEQUIX, … — computes the
energy as a sum over local atomic contributions determined by the environment around the atom. 
Irrespective of the underlying architecture, this involves the determination of neighbours within some cutoff radius
$r_\mathrm{cut}$. 
In Behler-Parrinello-like ML potetials, like NeuralIL, the atomic environment within the cutoff is used to directly compute a descriptor. 
For graph-neural-networks (GNNs), like MACE or NEQUIX, a graph consisting of nodes connected by edges is constructed instead.

In both cases, the number of neighbors determined is intrinsically a dynamic quantity.
That clashes with JAX's static-shape rule: `jax.jit` needs to
know the size of every array at trace time. The neighbour count
is *not* known at trace time — it depends on the live walker
position

Hence, in practice one needs to find a solution that lets one treat both cases with static array shapes. The unifying quantity is `max_neighbors`, which sets the upper bound for the allowed number of neighbors and fixes corresponding arrays to that size. In the case of GNNs, this corresponds to a maximum number of edges of $N_\mathrm{atoms} \times \texttt{max\_neighbors}$.

```{mermaid}
flowchart LR
    INIT["init walkers<br/>compute true max_n at start"]
    INIT --> BUCKET["pick smallest bucket b ∈ max_neighbors_list<br/>with b ≥ true_max + max_neighbors_offset"]
    BUCKET --> CALL["backend(positions, species, cell,<br/>max_neighbors=b, ensemble_params)"]
    CALL --> RES{"overflow?"}
    RES -- "no" --> NEXT["return (energy,<br/>actual_max_n,<br/>overflow=False)"]
    RES -- "yes" --> ESC["escalate: b ← next bucket in list,<br/>re-run inner loop with new b"]
    ESC --> CALL

    classDef init fill:#fff7e0,stroke:#a07000,color:#222
    classDef decision fill:#eef5ff,stroke:#1565c0,color:#222
    classDef esc fill:#ffe0e0,stroke:#c62828,color:#5a0000
    class INIT,BUCKET init
    class RES decision
    class ESC esc

    linkStyle 5 stroke:#c62828,color:#c62828,stroke-width:2px
```

`max_neighbors` is a *static* argument to the JIT'd backend
call — every distinct value triggers one fresh compile and is
then cached forever. Backends signal overflow by returning
`overflow=True` together with the *true* per-atom neighbour count
(`actual_max_n`), so the outer loop can size the next bucket
correctly. The shared helper
{func}`~jaxrens.backends._graph_neighbors._compute_true_max_neighbors`
runs the same neighbour-mask logic without allocating the edge
buffer, so init-time bucket sizing is cheap.

The user surface is two YAML knobs on each MLIP backend
({class}`~jaxrens.backends.mace.MACEBackend`,
{mod}`jaxrens.backends.neuralil`,
{mod}`jaxrens.backends.nequix`):

- `max_neighbors_list: [50, 75, 100, 150]` — the buckets to
  pre-compile. The resolver picks the smallest entry $\geq$
  observed max + offset for the initial run; on overflow the
  outer loop walks to the next entry.
- `max_neighbors_offset: 5` — safety margin added to the observed
  max when picking the initial bucket. Keeps you off the
  knife-edge where one MCMC step pushes a single atom over the
  buffer.

This is why `ns_step` returns a `neighbor_count` and an
`overflow` flag in its `info` dict: the outer loop reads them,
and on overflow it bumps `max_neighbors` and re-enters the same
iteration. The red `escalate max_neighbors` node in the NS-loop
figure ({doc}`ns_loop`) is exactly this back-edge.

## Ensembles as additive corrections

The `EnsembleBackend` wraps a base backend and adds the appropriate
thermodynamic term **per call**, reading the ensemble parameters
from the `ensemble_params` dict. This means one wrapper instance
serves multiple replicas at different pressures / chemical
potentials — no rebuild needed.

For a configuration $r = (\mathbf q, \mathbf{L}, \sigma)$ with
volume $V = |\det(\mathbf{L})|$ and species counts $N_i$,

$$
H_\mathrm{NVT}(r) = U(r),
$$
$$
H_\mathrm{NPT}(r) = U(r) + P\,V,
$$
$$
H_{\mu P T}(r) = U(r) + P\,V - \sum_i \mu_i N_i.
$$

The NS loop always works with $H$, not $U$. Same sampler, same
move kernels, same acceptance criterion; the ensemble only changes
what gets compared to $E_\mathrm{max}$.

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
