# Backends and ensembles

An *energy backend* is anything that maps atomic positions + species
+ cell to an energy. The quantity the sampler actually works with is a
{class}`~jaxrens.backends.hamiltonian.Hamiltonian`: one bare backend
(the *model*) plus a flat list of additive *energy terms* — the
ensemble correction $PV - \boldsymbol\mu\cdot\mathbf N$, an optional
soft-core repulsion, …. Moves see only the Hamiltonian; the rest of
the library doesn't care which ensemble or which corrections are
active.

## The `EnergyBackend` protocol

```{mermaid}
flowchart LR
    M["Move kernel"] --> B["Hamiltonian(model, terms)"]
    B --> Base["model: EnergyBackend<br/>(lj / mace / neuralil / …)"]
    B --> SC["SoftCoreTerm<br/>(optional)"]
    B --> ET["EnsembleTerm<br/>(NPT / μPT)"]
    Base --> U["U  (bare potential)"]
    SC --> EC["E_core"]
    ET --> PV["PV − μ·N"]
    U --> H["H = U + E_core + PV − μ·N"]
    EC --> H
    PV --> H
```

Every backend — LJ, MACE-JAX, NeuralIL, Nequix, jax-md and the toy
potentials — exposes the same call signature:

```python
result = backend(
    positions,        # (A, 3)
    species,          # (A,)   z-table indices
    cell,             # (3, 3) lattice vectors as rows
    max_neighbors,    # int    MLIP-side buffer size
    ensemble_params,  # dict   {"pressure": ..., "chemical_potentials": ...} or None
)
result.energy              # scalar per walker — the only universal field
result.max_neighbor_count  # largest neighbor count seen this call
result.overflow            # did that exceed the max_neighbors buffer?
```

The return value is a
{class}`~jaxrens.backends.base.BackendResult` — a `NamedTuple`, so it
threads through `jit` / `vmap` / `scan` as a pytree unchanged.  Only
`energy` is universally meaningful; the rest are backend-specific and
carry sentinel defaults (`0` / `False` / `None`) where a backend cannot
produce them.  A backend must return the **same field set on every
call**: `None`-versus-array is part of the pytree treedef, and a
structure that changes between calls would break `lax.scan` /
`lax.cond`.

Carrying `max_neighbor_count` and `overflow` alongside the energy is
how the NS outer loop handles MLIP neighbor-buffer overflow: when a
move pushes atoms closer together than `max_neighbors` can hold, the
backend signals the overflow, the loop retries at a bigger bucket, and
JAX reuses the JIT'd kernel compiled for that bucket size.  Backends
without a neighbor list (LJ, the toy potentials, all-pairs jax-md)
leave both at their sentinels.

`forces` is populated only on the force path, not on the energy-only
`__call__` above. Callers go through
{func}`~jaxrens.backends.base.eval_energy_and_forces`, which uses a
backend's own `energy_and_forces` method when it defines one (e.g.
NeuralIL's native force evaluation) and otherwise falls back to
reverse-mode autodiff of the energy. `energy_members` /
`forces_members` carry per-committee-member predictions for
ensemble (committee) models; see
{func}`~jaxrens.backends.base.committee_uncertainty`.

Built-in backends:

| `type:` | Class | Notes |
|---|---|---|
| `lj` | {class}`jaxrens.backends.lj.LJBackend` | periodic or finite, static cutoff |
| `mace` | {class}`jaxrens.backends.mace.MACEBackend` | mace-jax, supercell neighbor expansion, pins float32 globally (see {doc}`../../dev/install`) |
| `neuralil` | `jaxrens.backends.neuralil.NeuralILBackend` | bucketed `max_neighbors_list` JIT cache (lazy per-bucket compile) |
| `nequix` | `jaxrens.backends.nequix.NequixBackend` | local `.nqx` checkpoint or a bundled model name |
| `jaxmd` | `jaxrens.backends.jaxmd.JaxMDBackend` | Tersoff / EAM, all-pairs — ignores `max_neighbors_list` |
| `harmonic`, `double_well`, `gaussian_mixture` | `jaxrens.backends.toy` | analytical test potentials |

## The neighbor problem

Every modern MLIP — MACE, NeuralIL, NEQUIX, … — computes the
energy as a sum of local atomic contributions, each determined
by the environment around its atom. Whatever the architecture,
the first step is to find every neighbour within a cutoff
radius $r_\mathrm{cut}$. Behler–Parrinello-style potentials like
NeuralIL feed those neighbours into per-atom descriptors
directly; graph neural networks such as MACE or NEQUIX instead
build an explicit graph whose nodes are atoms and whose edges
connect neighbour pairs, then pass messages along it.

Either way, the number of neighbours is intrinsically dynamic —
it depends on the live walker's geometry, which changes every
MCMC step. That clashes with JAX's static-shape rule: `jax.jit`
needs the size of every array at trace time, so we can't let the
neighbour count vary freely.

The unifying static quantity is `max_neighbors`, an upper bound
on the allowed neighbour count per atom; the corresponding
arrays are pre-allocated at that size. For GNNs this caps the
edge buffer at
$N_\mathrm{atoms} \times \texttt{max\_neighbors}$.

For jaxrens we went with the following strategy: every backend
call takes a set of configurations and one fixed bucket size
$b = \texttt{max\_neighbors}$, builds its neighbour data into
arrays pre-allocated at that size, and at the same time computes
the *true* per-atom neighbour count from the actual geometry. If
that count fits — `true_max ≤ b` — the energy is returned
together with the observed count and `overflow=False`. If it
doesn't, the backend reports `overflow=True` together with the
observed `actual_max_n`; the outer loop picks the smallest entry
of `max_neighbors_list` that clears
$\texttt{actual\_max\_n} + \texttt{max\_neighbors\_offset}$, and
the same configuration is re-evaluated against that larger
bucket. The `max_neighbors_offset` knob is the headroom that
prevents the very next MCMC step from tripping the same overflow
again after a trivial cell fluctuation.



```{mermaid}
flowchart LR
    CFG["configurations<br/>(positions, species, cell)<br/>+ ensemble_params<br/>+ current bucket b"]
    CFG --> CALL["backend(...,<br/>max_neighbors=b)"]
    CALL --> RES{"true_max &gt; b ?"}
    RES -- "no" --> OK["return (energy,<br/>actual_max_n,<br/>overflow=False)"]
    RES -- "yes" --> ESC["escalate:<br/>b ← smallest b' ∈ max_neighbors_list<br/>with b' ≥ true_max + max_neighbors_offset,<br/>re-run inner loop"]
    ESC --> CALL

    classDef input fill:#fff7e0,stroke:#a07000,color:#222
    classDef decision fill:#eef5ff,stroke:#1565c0,color:#222
    classDef esc fill:#ffe0e0,stroke:#c62828,color:#5a0000
    class CFG input
    class RES decision
    class ESC esc

    linkStyle 3,4 stroke:#c62828,color:#c62828,stroke-width:2px
```

Because `max_neighbors` is a *static* JIT argument,
`max_neighbors_list` is an **allowlist** of permitted bucket
sizes rather than a set of kernels built in advance: each entry
is JIT-traced lazily on its first call and then cached for the
rest of the run. The initial `b` is chosen once before the loop
starts — the resolver calls
{func}`~jaxrens.backends._graph_neighbors._compute_true_max_neighbors`
on the starting walker geometry (no edge buffer allocated yet)
and picks the smallest entry of `max_neighbors_list` that clears
`true_max + max_neighbors_offset`. That entry becomes the first
kernel to compile.

The user surface is two YAML knobs on each MLIP backend
({class}`~jaxrens.backends.mace.MACEBackend`,
{mod}`jaxrens.backends.neuralil`,
{mod}`jaxrens.backends.nequix`):

- `max_neighbors_list: [50, 75, 100, 150]` — the bucket sizes
  the run is allowed to use. Each entry is JIT-compiled the first
  time it's called (not at startup); the resolver picks the
  smallest entry $\geq$ observed max + offset for the initial
  run, and on overflow the outer loop walks to the next entry,
  triggering one fresh compile that's then cached for the rest
  of the run.
- `max_neighbors_offset: 5` — safety margin added to the observed
  max when picking the initial bucket. Keeps you off the
  knife-edge where one MCMC step pushes a single atom over the
  buffer.

This is why `ns_step` returns a `neighbor_count` and an
`overflow` flag in its `info` dict: the outer loop reads them,
and on overflow it bumps `max_neighbors` and re-enters the same
iteration. The red `escalate max_neighbors` node in the NS-loop
figure ({doc}`ns_loop`) is exactly this back-edge.

## The Hamiltonian: model + additive terms

Corrections on top of the bare potential are not wrappers around the
backend; they are {class}`~jaxrens.backends.hamiltonian.EnergyTerm`
objects composed *next to* the model in a flat
{class}`~jaxrens.backends.hamiltonian.Hamiltonian`:

```python
from jaxrens.backends.ensemble import EnsembleTerm
from jaxrens.backends.hamiltonian import Hamiltonian
from jaxrens.backends.softcore import SoftCoreTerm

H = Hamiltonian(model, [SoftCoreTerm(), EnsembleTerm(pressure=0.0)])
H(positions, species, cell, max_neighbors, ensemble_params).energy
# = U + E_core + P·V − μ·N
```

A `Hamiltonian` satisfies the same `EnergyBackend` call protocol, so
move kernels take it unchanged. The resolver builds it from the YAML
(`backend:` → model, `backend.softcore_repulsion:` → `SoftCoreTerm`,
`ensemble:` → `EnsembleTerm`); you rarely construct one by hand.

**Energies and forces compose per layer.** Every layer contributes
its *own* energy and its *own* forces, and the Hamiltonian sums them.
Nothing is differentiated through a layer it does not own:

| Layer | Energy | Forces |
|---|---|---|
| model | `model(...)` | native `energy_and_forces` if the model defines one, else autodiff of the model alone |
| `SoftCoreTerm` | repulsive Morse $E_\mathrm{core}$ | autodiff of $E_\mathrm{core}$ alone |
| `EnsembleTerm` | $PV - \boldsymbol\mu\cdot\mathbf N$ | analytically zero (no position dependence) |

A new correction is a subclass of `EnergyTerm` implementing
`energy(positions, species, cell, ensemble_params)`; its forces
default to autodiff of that method, and it can override
`energy_and_forces` to supply custom ones.

**Model metadata is reached explicitly.** The Hamiltonian does *not*
forward attribute access to its model. Model-level facts —
`atomic_numbers`, `max_neighbors_for`, `is_ensemble`, … — are read
from `H.model` or via
{func}`~jaxrens.backends.hamiltonian.unwrap_model`, which also accepts a
bare backend.

`Hamiltonian(Hamiltonian(model, [a]), [b])` flattens to
`Hamiltonian(model, [a, b])`, and `H.with_terms(...)` returns a new
Hamiltonian with terms appended. `EnsembleBackend(base, ...)` and
`SoftCoreBackend(base, ...)` remain as convenience functions that
return such a flat Hamiltonian.

### Ensemble corrections

The `EnsembleTerm` adds the thermodynamic term **per call**, reading
the ensemble parameters from the `ensemble_params` dict (keys
`pressure`, `chemical_potentials`) and falling back to its own
defaults. This means one term instance serves multiple replicas at
different pressures / chemical potentials — no rebuild needed.

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
what gets compared to $E_\mathrm{max}$. With soft-core repulsion
enabled, $U$ above is read as $U + E_\mathrm{core}$.

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
| Hamiltonian (model + terms) | {class}`jaxrens.backends.hamiltonian.Hamiltonian` |
| Ensemble term ($PV - \mu N$) | {class}`jaxrens.backends.ensemble.EnsembleTerm` |
| Soft-core term | {class}`jaxrens.backends.softcore.SoftCoreTerm` |
| Force dispatch (native vs. autodiff) | {func}`jaxrens.backends.base.eval_energy_and_forces` |
| Backend loader (by type string) | {func}`jaxrens.backends.loader.load_backend` |
| MACE backend construction | {func}`jaxrens.backends.mace.create_mace` |
| Species mapping fix | `cli/resolve.py::_resolve_init_species` |
