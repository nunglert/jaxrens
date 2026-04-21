# Walkers are JAX pytrees

A walker's state — positions, cell, species, step sizes, energy,
ensemble parameters, and any move-specific extras — is a
{class}`jaxrens.state.mc_state.MCState` dataclass registered with
`jax.tree_util`. Every leaf is a JAX array; every leading axis is a
batch axis. `jit`, `vmap`, and `pmap` operate on the whole pytree
transparently, so going from one walker → many walkers → many
independent NS runs → many GPUs requires **no code changes in the
inner kernels** — only extra leading axes.

## The tree

```{mermaid}
flowchart TD
    NS["NSState"] --> P["population: MCState (batched)"]
    NS --> DE["dead_energies: (max_dead,)"]
    NS --> DP["dead_positions: (max_dead, A, 3)"]
    NS --> DV["dead_volumes: (max_dead,)"]
    NS --> IT["iteration: int32"]
    NS --> ND["n_dead: int32"]
    NS --> LZ["log_evidence: float"]
    P --> POS["positions: (K, A, 3)"]
    P --> CELL["cell: (K, 3, 3)"]
    P --> TYP["types: (K, A)"]
    P --> STEP["step_sizes: (K, n_moves)"]
    P --> EN["energy: (K,)"]
    P --> EP["ensemble_params: dict  (pressure/mu/...)"]
    P --> EX["extra fields<br/>e.g. direction (galilean), momenta (HMC)"]
    P -. static .-> MN["max_neighbors: int"]
    P -. static .-> NA["n_atoms: int"]
```

`max_neighbors`, `n_atoms` and other fields marked `static_field()`
live in the pytree's *aux data*, not the leaves. Changing them
triggers a JIT recompilation; changing a leaf does not.

## Dynamic class construction

`MCState` isn't a fixed dataclass — it's built on demand by
{func}`~jaxrens.state.mc_state.make_mc_state_class` based on the
move kernels in play. The galilean move needs a `direction` field;
HMC needs momentum; alchemical-morph moves need per-walker species
counts. `build_mwg` collects each kernel's declared `extra_state_fields`
and generates an `MCState` subclass with exactly those fields + the
core. All subclasses share the same batch-axis convention, so the
NS loop is dtype-generic.

Two consequences:

- You can add a new move with new per-walker state without touching
  `MCState` directly — just declare `extra_state_fields` in your
  `MoveKernel`. See {doc}`moves_mwg`.
- Tests that mock `MCState` need to match the extra-field set the
  real kernels would request; `build_mwg(backend, descriptors)`
  returns a ready-made `init_fn` that does this for you.

## Batch-axis shapes across the three descriptors

Every array in the NS state has the same batch prefix — determined
by which {class}`~jaxrens.sampling.batch_descriptor.BatchDescriptor`
is dispatching the loop. The descriptor is attached to `info["_batch"]`
so callbacks can branch on it.

```{image} /_static/figures/pytree_shapes.png
:alt: shape table for SingleRun / VmapRuns / PmapVmapRuns
:align: center
```

- **`SingleRun`** — one process, one GPU, one NS instance.
  `K = n_walkers`, `A = n_atoms`.
- **`VmapRuns`** — one GPU, `R = n_runs` parallel NS instances.
  Used on CPU nodes or when `len(jax.local_devices()) == 1`.
- **`PmapVmapRuns`** — `G × P` replicas distributed across `G` GPUs
  with `P = n_per_gpu` replicas on each. Execution is
  `pmap(vmap(...))`.

The loop body is identical in all three. Only the descriptor's
`wrap_step`, `split_keys`, and `reduce_for_termination` methods
differ. See `sampling/batch_descriptor.py`.

## Why pytrees (and not, say, flat arrays)

1. **Moves only see a single walker.** A move kernel like
   `galilean` is written as a pure function of *one* `MCState`; it
   never sees batch axes. `vmap` lifts it into a batched kernel
   automatically.
2. **Adding a state field is local.** A new move declares a new
   extra field. The rest of the library (I/O, callbacks, NS loop)
   never mentions it by name.
3. **Static vs dynamic is an explicit decoration.** If a field
   mutates during a run (step_size adapts; max_neighbors grows on
   overflow), it's a leaf. If it's a compile-time constant (n_atoms),
   it's in aux data.

The pytree identity

$$
\texttt{x} = \text{unflatten}\bigl(\text{flatten}(\texttt{x})\bigr)
$$

is what makes all three descriptor modes equivalent at the kernel
level — the leaves change shape, the tree structure and aux data do
not.

## Where this lives in the code

| Concern | File |
|---|---|
| Dynamic `MCState` factory | {func}`jaxrens.state.mc_state.make_mc_state_class` |
| Static field decorator | `state/mc_state.py::static_field` |
| Top-level NS state | {class}`jaxrens.state.ns.NSState` |
| Batch-shape dispatch | `sampling/batch_descriptor.py` |
| Pytree helpers (pack/unpack) | `utils/pytree.py` |
