# Walkers are JAX pytrees

A walker's state — positions, cell, species, step sizes, energy,
ensemble parameters, and any move-specific extras — is a
{class}`jaxrens.state.mc_state.MCState` dataclass registered with
`jax.tree_util`. Every leaf is a JAX array; every leading axis is a
batch axis. `jit`, `vmap`, and `shard_map` operate on the whole pytree
transparently, so going from one walker → many walkers → many
independent NS runs → many GPUs requires **no code changes in the
inner kernels** — only extra leading axes.

## The tree

`NSState` is the top-level pytree the NS loop carries. Its single
non-trivial child is `population`, a batched
{class}`~jaxrens.state.mc_state.MCState` whose leaves carry the
`(K, …)` per-walker arrays. Solid arrows are pytree leaves; dashed
arrows mark `static_field()` entries that live in *aux data* and
trigger a JIT recompile when they change.

```{mermaid}
flowchart LR
    NS["NSState"]
    NS --> P["population<br/>(MCState, batched)"]
    NS --> LZ["log_evidence:<br/>scalar"]
    NS --> IT["iteration:<br/>int32"]
    NS --> KEY["rng_key:<br/>PRNGKey"]
    NS -. static .-> NW["n_walkers: int"]
    NS -. static .-> NA0["n_atoms: int"]

    subgraph MC_leaves["MCState leaves (per-walker, batch K)"]
        direction LR
        subgraph mc_config["configuration"]
            direction TB
            POS["positions: (K, A, 3)"]
            CELL["cell: (K, 3, 3)"]
            TYP["types: (K, A)"]
            EN["energy: (K,)"]
        end
        subgraph mc_mcmc["MCMC stats"]
            direction TB
            SS["step_sizes: (K, n_moves)"]
            ACC["n_accepted: (K, n_moves)"]
            PROP["n_proposed: (K, n_moves)"]
        end
        subgraph mc_misc["overflow / ensemble / extras"]
            direction TB
            MNC["max_neighbor_count: (K,)"]
            OV["overflow: (K,) bool"]
            EP["ensemble_params: dict<br/>(pressure / μ / target_comp / …)"]
            EX["extras<br/>e.g. direction (galilean),<br/>momenta (HMC)"]
        end
    end

    subgraph MC_aux["MCState aux (static_field)"]
        direction TB
        MN["max_neighbors: int"]
        NA1["n_atoms: int"]
    end

    P --> MC_leaves
    P -. static .-> MC_aux

    classDef root fill:#fff7e0,stroke:#a07000,color:#222
    classDef leafBox fill:#eef5ff,stroke:#1565c0,color:#222
    classDef auxBox fill:#f5f5f5,stroke:#888,color:#222
    classDef colBox fill:#fafdff,stroke:#9bb8d6,color:#222
    class NS,P root
    class MC_leaves leafBox
    class MC_aux auxBox
    class mc_config,mc_mcmc,mc_misc colBox

    linkStyle 4,5,7 stroke:#888,color:#555
```

`NSState` sits on the left; its level-2 children stack vertically
in the next column; the batched `MCState` leaves form the right
panel split into three logical sub-columns — *configuration*
(positions / cell / types / energy), *MCMC stats* (step sizes,
accept / propose counters), and *overflow / ensemble / extras*
(neighbour-count + ensemble parameters + per-kernel extras). The
two dashed edges at the `NSState` and `population` level point to
aux blocks — `static_field()` entries that live outside the
pytree leaves and gate recompilation.

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

## Batch-axis shapes across the descriptors

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
  `jax.jit(shard_map(vmap(...)))` over a `"gpu"`-named device mesh.
- **`ShardedSingleRun`** — *one* NS run whose `K` walkers are split
  into `G` chunks of `K / G`, one per GPU (`run.shard_n_gpu > 1`).
  Population leaves are `(G, K / G, A, 3)`; the leading axis is a
  *sharding* axis, not a replica axis, so `n_runs == 1`. The step is
  `ns_step_sharded`, which uses `all_gather` / `psum` collectives over
  a `"shard"`-named mesh to act on the global population. This is for
  memory scaling when a heavy ML backend with a large `n_live` doesn't
  fit on one device. (The figure above shows only the three replica
  descriptors.)

The loop body is identical in all four. Only the descriptor's
`wrap_step`, `split_keys`, and `reduce_for_termination` methods
differ. See `sampling/batch_descriptor.py`.

`BatchDescriptor` is a type alias
(`SingleRun | VmapRuns | PmapVmapRuns | ShardedSingleRun`), not a
base class. `isinstance(x, BatchDescriptor)` works, but you cannot
subclass it. The three replica descriptors share their shape helpers
through a private `_UniformBatcher` base, and `ShardedSingleRun`
stands alone because `n_runs == prod(shape_prefix)` doesn't hold for
it.

(multi-device-shard-map)=
### Multi-device execution: `shard_map`

Both multi-device descriptors run on
[`jax.shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
rather than `jax.pmap`. Two pieces in
{mod}`jaxrens.sampling.mesh` make this work:

- **`build_mesh(axis_name, n_devices)`** builds the 1-D
  `jax.sharding.Mesh` over the first `n_devices` local devices. Each
  batcher builds its mesh once and caches it as `batcher.mesh`. Every
  consumer (`wrap_step`, `wrap_for_batch`, the `InterREManager` swap
  step) reuses that one object instead of building its own.
- **`pmap_like(fn, axis_name, mesh)`** wraps `fn` in `shard_map` with
  `in_specs = out_specs = PartitionSpec(axis_name)` on every leaf. It
  also squeezes the size-1 per-device axis off the inputs and adds it
  back to the outputs, so `fn` is written exactly as it would be for
  `jax.pmap`. The per-device bodies (`ns_step_sharded`, the swap
  kernels, `adjust_step_size`) are unchanged by the migration.
  Collectives inside them (`all_gather`, `axis_index`, `psum`) behave
  exactly as they did under `pmap`.

The practical differences from `pmap`:

- **Composes with `jax.jit`.** Every wrapped step is returned as
  `jax.jit(shard_map(...))`, and it can be nested inside other jitted
  code, which `pmap` doesn't support cleanly.
- **Per-argument partition specs.** Where one uniform spec isn't
  enough, the code calls `shard_map` directly.
  `ShardedSingleRun.wrap_for_batch(..., replicated=...)` marks
  arguments such as the coherent adaptation key as replicated
  (`PartitionSpec()`), so `shard_map` hands the same value to every
  device without a pre-broadcast. The `InterREManager` swap step
  returns its statistics dict with `out_specs=PartitionSpec()`, so it
  comes back replicated instead of stacked `G` times.
- **`check_vma` is off.** `shard_map`'s varying-manual-axis type
  checker has no `pmap` equivalent. JAXRENS runs with
  `check_vma=False` (pmap's "trust the caller" model), because the
  checker also traces into backend energy functions it doesn't
  control. For example, NeuralIL's cutoff `lax.cond` has branches that
  type as varying vs. unvarying. `wrap_for_batch` still takes a
  `check_vma` flag for callers whose body is known to type-check.

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
