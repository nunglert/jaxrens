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

```{mermaid}
flowchart TB
    subgraph YAML
        P["ensemble.pressure: [1, 2, 3, 4, 5, 6, 7, 8] GPa"]
    end
    P --> RES["resolver:<br/>n_total = 8<br/>n_gpu = len(jax.local_devices()) = 4<br/>n_per_gpu = 2"]
    RES --> G0["GPU 0: replicas (0,1)<br/>P = 1, 2"]
    RES --> G1["GPU 1: replicas (2,3)<br/>P = 3, 4"]
    RES --> G2["GPU 2: replicas (4,5)<br/>P = 5, 6"]
    RES --> G3["GPU 3: replicas (6,7)<br/>P = 7, 8"]
    G0 -. "even swap (0,1)" .-> G0
    G1 -. "even swap (2,3)" .-> G1
    G2 -. "even swap (4,5)" .-> G2
    G3 -. "even swap (6,7)" .-> G3
    G0 -. "odd swap (1,2)" .-> G1
    G1 -. "odd swap (3,4)" .-> G2
    G2 -. "odd swap (5,6)" .-> G3
```

## Inter-replica exchange (RENS)

Every `inter_re.every` iterations, the manager runs `n_swap_cycles`
even/odd swap passes between adjacent replicas in the pressure (or
composition / μ) ladder. Three flavors exist:

### Pressure-RENS

The two replicas keep their configurations; only the *thermodynamic
context* is swapped. Configuration $x_i$ from replica $i$ is
evaluated under replica $j$'s ensemble:

$$
\alpha_{ij} = \mathbf{1}\bigl[H_j(x_i) < E_{\max, j}\bigr] \cdot
              \mathbf{1}\bigl[H_i(x_j) < E_{\max, i}\bigr],
$$

where $H_j$ includes replica $j$'s pressure. Zero backend calls —
the energies $U(x)$ are already stored; only the $+PV$ term
recomputes.

### XRENS (composition morphing)

Two replicas with different target compositions swap particle
labels: walker $x_i$'s species vector is reshuffled to match
replica $j$'s composition target, and the new energy is evaluated
against replica $j$'s constraint.

### Semi-grand swap

Two replicas at different chemical-potential vectors swap. The
acceptance depends on the change in $-\sum_i \mu_i N_i$.

```{image} /_static/figures/rens_acceptance.png
:alt: synthetic RENS acceptance vs log pressure ratio
:align: center
```

Acceptance falls off with adjacent-pair pressure spacing: a
geometric ladder (ratio $\sqrt 2$) typically gives better mixing
than a linear one (ratio $+1$ GPa per step) at the top of the
range.

## Even/odd swap schedule

In each swap pass, replica pairs are chosen in two phases:

- **Even pairs**: $(0,1), (2,3), (4,5), \ldots$
- **Odd pairs**: $(1,2), (3,4), (5,6), \ldots$

Running even then odd covers every adjacent pair. With
`n_swap_cycles > 1`, the whole (even, odd) cycle repeats that many
times.

## Pressure-RENS vs cohort sweeps

jaxrens also supports a *sequential cohort sweep* — `expand_cohort`
produces one `ResolvedConfig` per pressure and the CLI runs them
serially. That path is unchanged for configs with no
replica-differentiating list. The moment the config implies more
than one replica (list pressure / composition / μ), dispatch flips
to multi-run and you get inter-RE for free.

## Where this lives in the code

| Concern | File |
|---|---|
| Topology derivation | `cli/resolve.py::_derive_replica_axes` |
| Multi-run dispatch | {func}`jaxrens.sampling.nested_sampling.run_ns_multi_gpu` |
| Swap manager | `sampling/inter_re_manager.py::InterREManager` |
| Swap kernels (3 flavors) | `sampling/moves/replica_exchange.py` |
| Even/odd pair helper | `sampling/moves/replica_exchange.py::get_swap_pairs` |
