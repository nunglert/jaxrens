# Replica topology and inter-RE

When the YAML config carries a *replica-differentiating list* —
multiple pressures, multiple composition targets, or multiple
chemical-potential vectors — jaxrens runs several NS instances
concurrently and optionally lets them swap configurations. The
resolver figures out the device topology from
`jax.local_devices()`; the YAML never mentions GPUs.

## Where n_total, n_gpu, n_per_gpu come from

The replica axis is determined by whichever YAML list implies it:

| Source | Condition | Replica count |
|---|---|---|
| `ensemble.pressure: [P_1, …, P_n]` | NPT with list | $n_\text{total} = n$ |
| `inter_re.composition_targets: [[..], …]` (XRENS) | list length | $n_\text{total} = $ list length |
| `inter_re.chemical_potentials: [[..], …]` (semi-grand) | list length | $n_\text{total} = $ list length |

Given $n_\text{total}$, the resolver computes

$$
n_\text{gpu} = \operatorname{len}\bigl(\texttt{jax.local\_devices()}\bigr),
\qquad
n_\text{per\_gpu} = \frac{n_\text{total}}{n_\text{gpu}}.
$$

The divisibility constraint $n_\text{total} \bmod n_\text{gpu} = 0$
is enforced at resolve time with a clear error. If
$n_\text{gpu} > n_\text{total}$, the extra devices sit idle and
the resolver clamps $n_\text{gpu} \leftarrow n_\text{total}$.

Running the same YAML under different SLURM allocations produces
different valid topologies automatically — the config is
device-count-independent.

::::{tab-set}

:::{tab-item} 4 GPUs
:sync: gpu-topology

```{mermaid}
flowchart LR
    YAML["ensemble.pressure:<br/>[1, 2, 3, 4, 5, 6, 7, 8] GPa"]
    RES["resolver:<br/>n_total = 8<br/>n_gpu = 4<br/>n_per_gpu = 2"]
    YAML --> RES
    RES --> R0

    subgraph G0["GPU 0"]
        direction LR
        R0(["R0<br/>P=1"]) <-. even .-> R1(["R1<br/>P=2"])
    end
    subgraph G1["GPU 1"]
        direction LR
        R2(["R2<br/>P=3"]) <-. even .-> R3(["R3<br/>P=4"])
    end
    subgraph G2["GPU 2"]
        direction LR
        R4(["R4<br/>P=5"]) <-. even .-> R5(["R5<br/>P=6"])
    end
    subgraph G3["GPU 3"]
        direction LR
        R6(["R6<br/>P=7"]) <-. even .-> R7(["R7<br/>P=8"])
    end

    R1 <-. odd .-> R2
    R3 <-. odd .-> R4
    R5 <-. odd .-> R6

    linkStyle 0 stroke:#555,color:#222
    linkStyle 1 stroke:#555,color:#222
    linkStyle 2,3,4,5 stroke:#1565c0,color:#1565c0,stroke-width:2px
    linkStyle 6,7,8 stroke:#c62828,color:#c62828,stroke-width:2px

    classDef gpuBox fill:#f5f5f5,stroke:#888,color:#222
    classDef rep fill:#fff,stroke:#444,color:#222
    class G0,G1,G2,G3 gpuBox
    class R0,R1,R2,R3,R4,R5,R6,R7 rep
```

:::

:::{tab-item} 2 GPUs
:sync: gpu-topology

```{mermaid}
flowchart LR
    YAML["ensemble.pressure:<br/>[1, 2, 3, 4] GPa"]
    RES["resolver:<br/>n_total = 4<br/>n_gpu = 2<br/>n_per_gpu = 2"]
    YAML --> RES
    RES --> R0

    subgraph G0["GPU 0"]
        direction LR
        R0(["R0<br/>P=1"]) <-. even .-> R1(["R1<br/>P=2"])
    end
    subgraph G1["GPU 1"]
        direction LR
        R2(["R2<br/>P=3"]) <-. even .-> R3(["R3<br/>P=4"])
    end

    R1 <-. odd .-> R2

    linkStyle 0 stroke:#555,color:#222
    linkStyle 1 stroke:#555,color:#222
    linkStyle 2,3 stroke:#1565c0,color:#1565c0,stroke-width:2px
    linkStyle 4 stroke:#c62828,color:#c62828,stroke-width:2px

    classDef gpuBox fill:#f5f5f5,stroke:#888,color:#222
    classDef rep fill:#fff,stroke:#444,color:#222
    class G0,G1 gpuBox
    class R0,R1,R2,R3 rep
```

:::

::::

The diagram reads left-to-right: YAML → resolver → device
topology. Each GPU subgraph holds its `n_per_gpu = 2` replicas
side by side; **blue** edges are *even* swaps (always within a
GPU when `n_per_gpu = 2`) and **red** edges are *odd* swaps
(crossing GPU boundaries). Together they cover every adjacent
pair on the pressure ladder. The same YAML produces both
topologies — only `len(jax.local_devices())` differs.

```{tip}
All diagrams support **pan / zoom**: scroll-wheel zooms,
click-drag pans. The toolbar in the upper-LEFT (`+`, `−`, `↺`)
gives discrete zoom-in / zoom-out / reset; the `⛶` button in the
upper-RIGHT opens the diagram in a fullscreen modal.
```

### Even/odd swap schedule

In each swap pass, replica pairs are chosen in two phases:

- **Even pairs**: $(0,1), (2,3), (4,5), \ldots$
- **Odd pairs**: $(1,2), (3,4), (5,6), \ldots$

Running even then odd covers every adjacent pair. With
`n_swap_cycles > 1`, the whole (even, odd) cycle repeats that many
times.

## Inter-replica exchange (RENS)

Every `inter_re.every` iterations, the manager runs `n_swap_cycles`
even/odd swap passes between adjacent replicas in the pressure (or
composition / μ) ladder. Three flavors exist:

### How the InterREManager dispatches swaps

The manager is constructed once before the NS loop starts; its
JIT-compiled swap step is cached and reused. On each NS iteration
`_run_loop` calls `fires(i)`, and on a fire calls
`apply(state, key)`, which dispatches on the
{class}`~jaxrens.sampling.batch_descriptor.BatchDescriptor` and on
the swap-kernel flavor:

```{mermaid}
flowchart TB
    LOOP["NS outer loop<br/>iteration i"]
    LOOP --> FIRES{"manager.fires(i)<br/>i &gt; 0  ∧  i mod every == 0"}
    FIRES -- no --> CONT["continue NS step"]
    FIRES -- yes --> APPLY["manager.apply(state, key)"]

    APPLY --> EXTR["_extract_swap_inputs(state):<br/>positions, types, energies, cells, emax,<br/>+ pressures / composition_targets / μ<br/>(from pop.ensemble_params)"]
    EXTR --> DESC{"BatchDescriptor"}

    DESC -- SingleRun --> NOOP["no-op<br/>(empty stats)"]
    DESC -- VmapRuns --> VMAP["_jit_vmap_swap<br/>(R, K, …) directly"]
    DESC -- PmapVmapRuns --> AGG["lax.all_gather axis 'gpu'<br/>(P, K, …) → (G, P, K, …)<br/>on every device"]
    AGG --> FLAT["reshape (G·P, K, …)<br/>same RNG ⇒ identical decisions"]
    FLAT --> SWAP["_jit_pmap_swap"]

    VMAP --> KFLAV{"swap kernel"}
    SWAP --> KFLAV

    KFLAV -- "PressureRENS" --> PR["replica_exchange_step<br/>swap thermo context only<br/>(0 backend calls)"]
    KFLAV -- "XRENSSwap" --> XR["xrens_replica_exchange_step<br/>swap species labels<br/>(re-evaluates U)"]
    KFLAV -- "SemiGrandSwap" --> SG["semi_grand_replica_exchange_step<br/>swap N at fixed μ<br/>(0 backend calls)"]

    PR --> CYC["loop n_swap_cycles ×<br/>(even pass, odd pass)"]
    XR --> CYC
    SG --> CYC

    CYC --> SHARDQ{"PmapVmapRuns?"}
    SHARDQ -- yes --> SHARD["slice this device's shard<br/>via lax.axis_index('gpu')"]
    SHARDQ -- no --> WRITE
    SHARD --> WRITE["population.set(positions=…, types=…,<br/>energy=…, cell=…)"]
    NOOP --> RET
    WRITE --> RET["return (NSState, swap_stats, key)"]

    classDef decision fill:#fff7e0,stroke:#a07000,color:#222
    classDef path fill:#eef5ff,stroke:#1565c0,color:#222
    classDef kernel fill:#fde8e8,stroke:#c62828,color:#222
    classDef io fill:#f5f5f5,stroke:#444,color:#222

    class FIRES,DESC,KFLAV,SHARDQ decision
    class VMAP,AGG,FLAT,SWAP,SHARD,CYC path
    class PR,XR,SG kernel
    class LOOP,APPLY,EXTR,WRITE,RET,CONT,NOOP io
```

A few invariants worth pinning down:

- **Cached compilation.** `_build_jit_fns` runs once in
  `__init__` and stores `_jit_vmap_swap` / `_jit_pmap_swap`.
  Subsequent `apply` calls hit the cache — no re-tracing per
  iteration.
- **Identical swap decisions across devices.** For
  `PmapVmapRuns`, the same scalar `swap_key` is broadcast to all
  devices before `pmap`. After the `all_gather` every device sees
  the same `(G, P, K, …)` tensor and runs the same code, so all
  shards agree on which pairs to swap; the per-device `axis_index`
  slice at the end re-establishes the original sharding.
- **`n_gpu = 1` is free.** With one device the `all_gather` is a
  no-op and `pmap` collapses to `vmap`. The same code path runs
  unconditionally so multi-GPU correctness is exercised by the
  single-GPU test suite.
- **Kernel choice is set at construction.** `_is_xrens` /
  `_is_semi_grand` are checked once and select which
  `replica_exchange_step` variant to JIT; they are not re-checked
  per call.


### Pressure-RENS

The two replicas keep their configurations; only the *thermodynamic
context* is swapped. Configuration $r_i$ from replica $i$ is
evaluated under replica $j$'s ensemble:

$$
\alpha_{ij} = \mathbf{1}\bigl[H_j(r_i) < E_{\max, j}\bigr] \cdot
              \mathbf{1}\bigl[H_i(r_j) < E_{\max, i}\bigr],
$$

where $H_j$ includes replica $j$'s pressure. Zero backend calls —
the energies $U(r)$ are already stored; only the $+PV$ term
recomputes.

### XRENS (composition morphing)

Each replica $i$ holds a target composition vector
$\mathbf c_i \in \mathbb Z_{\ge 0}^{S}$ with $\sum_s c_{i,s} = N$.
Let $\mathcal{M}_{\mathbf c}(\sigma, \xi)$ denote the random
*morph operator*: it permutes the species labels of $\sigma$ to a
new label vector with composition exactly $\mathbf c$, with
randomness $\xi$ choosing which atoms swap species
(see {func}`jaxrens.sampling.morph.morph_types_to_composition`).

A swap of replicas $(i, j)$ proposes the morphed exchange

$$
\tilde r_i = \bigl(\mathbf q_j,\; \mathcal{M}_{\mathbf c_i}(\sigma_j, \xi_i),\; h_j\bigr),
\qquad
\tilde r_j = \bigl(\mathbf q_i,\; \mathcal{M}_{\mathbf c_j}(\sigma_i, \xi_j),\; h_i\bigr),
$$

so the configuration arriving at replica $i$ has run-$i$'s target
composition but is built from run-$j$'s positions and cell. The
morphed potential energies $\tilde U_i = U(\tilde r_i)$ and
$\tilde U_j = U(\tilde r_j)$ are recomputed by the backend
(2 evaluations per attempted pair). Acceptance follows the same
cross-bound enthalpy form as Pressure-RENS, applied to the new
energies:

$$
\alpha_{ij} = \mathbf 1\!\bigl[\tilde U_i + P_j\, V(h_j) < E_{\max,j}\bigr]\,
              \mathbf 1\!\bigl[\tilde U_j + P_i\, V(h_i) < E_{\max,i}\bigr].
$$

In NVT the $PV$ term drops and the criterion reduces to
$\tilde U_i < E_{\max,j} \,\wedge\, \tilde U_j < E_{\max,i}$.

### Semi-grand swap

Each replica $i$ carries a per-species chemical-potential vector
$\boldsymbol\mu_i \in \mathbb R^{S}$. Walkers, cells, and species
stay put — only the $\boldsymbol\mu$ assignment is exchanged. Let
$\mathbf N(\sigma)_s = \#\{a : \sigma_a = s\}$ be the species-count
vector of walker $\sigma$. The grand-canonical energy under the
*new* chemical-potential vector after the swap is

$$
\Omega_i^{\text{new}} = U_i \;-\; \boldsymbol\mu_j \cdot \mathbf N(\sigma_i),
\qquad
\Omega_j^{\text{new}} = U_j \;-\; \boldsymbol\mu_i \cdot \mathbf N(\sigma_j),
$$

and because nothing moves, each replica simply checks its own NS
threshold:

$$
\alpha_{ij} = \mathbf 1\!\bigl[\Omega_i^{\text{new}} < E_{\max,i}\bigr]\,
              \mathbf 1\!\bigl[\Omega_j^{\text{new}} < E_{\max,j}\bigr].
$$

Zero backend calls — the $U_i$'s are already stored, so the
update is a pair of $S$-dimensional dot products.


## Where this lives in the code

| Concern | File |
|---|---|
| Topology derivation | `cli/resolve.py::_derive_replica_axes` |
| Multi-run dispatch | {func}`jaxrens.sampling.nested_sampling.run_ns_multi_gpu` |
| Swap manager | `sampling/inter_re_manager.py::InterREManager` |
| Swap kernels (3 flavors) | `sampling/moves/replica_exchange.py` |
| Even/odd pair helper | `sampling/moves/replica_exchange.py::get_swap_pairs` |
