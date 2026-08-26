# 2. RENS toy — replica exchange

The cheapest possible `jaxrens` run, and the only one small enough that you
can see what replica exchange is actually doing. Two particles in a periodic
one-dimensional box, three pressures, swaps between them — a few seconds on a
CPU, no model file, no GPU.

This is the toy model from the paper the method comes from: N. Unglert, L. B.
Pártay and G. K. H. Madsen, *Replica Exchange Nested Sampling*,
J. Chem. Theory Comput. **21**, 7304 (2025).

## The model

A pair interaction built from two pieces — a repulsive core and an attractive
well at separation `mu`:

$$
E_\mathrm{rep}(d) = \varepsilon_\mathrm{rep}\,e^{-h_\mathrm{rep} d^2},
\qquad
E_\mathrm{attr}(d) = -\varepsilon_\mathrm{attr}\,
   e^{-\frac{1}{2}\left(\frac{d-\mu}{\sigma}\right)^2},
\qquad
E_\mathrm{toy} = E_\mathrm{rep} + E_\mathrm{attr}
$$

summed over both particles and their periodic images inside a cutoff. A
configuration is fully described by two numbers: the box length $a$ and the
particle separation $d$. Pressure acts on $a$ the way it acts on volume in
three dimensions, giving the enthalpy that constant-pressure NS samples:

$$
H(x_1, x_2; a) = U(x_1, x_2; a) + P \cdot a
$$

Two degrees of freedom, and yet the enthalpy surface is genuinely multi-modal:
box lengths commensurate with `mu` are favoured, so there are competing basins
at $a \approx \mu, 2\mu, 3\mu, \dots$ **That** is the point. Independent NS
runs at different pressures fall into different basins and stay there; replica
exchange is what lets them escape.

```{image} ../_static/figures/tutorials/rens_toy_surface.png
:alt: irreducible wedge of the toy-model enthalpy surface at three pressures
:width: 100%
```

Only the wedge $d \leq a/2$ is shown. Under periodicity a separation $d$ and
$a - d$ describe the same configuration, so everything above the diagonal is a
mirror image; masking it is what makes the structure readable.

Read the three panels left to right and the tutorial's result is already
visible. At $P = 0.5$ the deepest basin sits at $a \approx 2, d \approx 1$ —
both particles in the attractive well with the box commensurate. As pressure
rises the $P a$ term tilts the surface toward small $a$, and by $P = 1.5$ the
compressed basin near $a \approx 1$ has taken over. Three replicas, two
competing minima, and a swap is the only cheap way across.

## The config

`examples/tutorials/01_rens_toy/config.yaml`:

```{literalinclude} ../../examples/tutorials/01_rens_toy/config.yaml
:language: yaml
```

Three things here have no analogue in the other tutorials.

**A list-valued pressure creates the replicas.** `pressure: [0.5, 1.0, 1.5]`
is the entire declaration — three independent NS runs, one per pressure. The
resolver derives the device topology from it, which `validate` reports back:

```text
topology  n_gpu=1 × n_per_gpu=3 = 3 replica(s)
```

**`inter_re` turns three independent runs into RENS.** Without it you get
three unconnected runs; with it, adjacent replicas offer each other walkers
every `re_interval` iterations. Omitting the key is exactly how you produce
the "independent NS" baseline to compare against.

**`lattice_1d` and `distance_1d` replace the usual moves.** The 1-D system
lives on the x-axis with a cell of `diag(a, 1, 1)`, so `det(cell) = a` and the
standard NPT term `P·V` *is* `P·a`. The general `volume` move scales the cell
isotropically and would destroy that structure, so the toy model gets its own
pair of moves — one changing the box length, one the separation, in the
paper's 1:1 ratio. `cell.min_aspect_ratio: 0.0` is required for the same
reason: a `diag(a, 1, 1)` cell has aspect ratio `1/a`, so the usual check is
meaningless here.

## Run it

```bash
cd examples/tutorials/01_rens_toy
jaxrens validate -c config.yaml --full
jaxrens run -c config.yaml
tail -f toy.log      # from a second shell; `run` prints nothing
```

The monitor reports **ranges across the replicas** rather than single numbers,
and adds a swap-statistics line:

```text
Starting multi-GPU NS: n_gpu=1, n_per_gpu=3 (n_total=3), n_walkers=64, n_mcmc=20
iter=0    Emax=[5.876..14.94]   log_Z=[-19.1008..-10.0348]  dt=3.2s
iter=250  Emax=[0.2319..1.344]  log_Z=[-4.6910..-2.6638]    dt=1.8s
  inter_re          n_pairs=  2  acc=1.00  per_pair=1.00±0.00
iter=500  Emax=[-0.8391..0.5813]  log_Z=[-4.0914..-2.3761]  dt=0.8s
  inter_re          n_pairs=  2  acc=0.00  per_pair=0.00±0.00
...
NS terminated at iter 1113: Prior mass negligible compared to evidence
NS finished: 1114 iterations, log_Z=[-4.0776..-2.3658], elapsed=8s
```

Watch the `inter_re` line. Early on, acceptance is **1.00**: the replicas are
still exploring overlapping regions of enthalpy, so almost any swap is legal.
By iteration 500 it has fallen to **0.00** — the replicas have separated into
their own basins and a walker from one no longer satisfies the other's
enthalpy threshold. That decay is not a failure; it is the run telling you
where the pressures stop overlapping. If acceptance collapses *immediately*,
the pressures are spaced too far apart to exchange anything, and the ladder
needs more rungs.

## What came out

Each replica writes its own trajectory and energy ladder, tagged `run00`,
`run01`, `run02` in pressure order:

```bash
ls output/
```

```text
toy.run00.energies   toy.run00.traj.extxyz   toy.re_stats.h5
toy.run01.energies   toy.run01.traj.extxyz   toy.acc_rates.h5
toy.run02.energies   toy.run02.traj.extxyz   toy.adaptation.h5
```

The final configurations show the transition the model was built to have:

| Pressure | final `a` | final `d` |
|---|---|---|
| 0.5 | 1.98 | 0.99 |
| 1.0 | 1.95 | 0.97 |
| 1.5 | 0.67 | 0.34 |

At the two lower pressures the system settles at `a ≈ 2μ` with the particles
sitting exactly in the attractive well, `d ≈ μ = 1`. At `P = 1.5` the `P·a`
term wins, and the system is crushed into a completely different basin at a
third of the box length. Three replicas, two phases.

## Plots

```bash
jaxrens plot output/toy.re_stats.h5      # swap acceptance per adjacent pair
jaxrens plot output/toy.run00.energies   # dead-point enthalpy ladder
jaxrens plot output/toy.adaptation.h5    # step sizes and acceptance per move
```

```{image} ../_static/figures/tutorials/toy.re_stats.png
:alt: per-pair RE swap acceptance, stacked
:width: 100%
```

The `re_stats` plot is the one specific to RENS: swap acceptance per adjacent
replica pair, each pair in its own lane so a dying rung cannot hide underneath
a healthy one. It is how you tune a pressure ladder — a pair whose acceptance
falls to zero early is a broken rung, and the fix is another replica between
them. Here `0↔1` decays gradually while `1↔2` holds at 100% throughout, which
says the 1.0/1.5 pair still overlaps long after 0.5/1.0 has separated.

```{image} ../_static/figures/tutorials/toy.energies.png
:alt: dead-point enthalpy and box-length trails for all three replicas
:width: 100%
```

Pointing `jaxrens plot` at any one replica's `.energies` file picks up all of
them, so the comparison is the default rather than something you assemble by
hand. The lower panel is the interesting one: all three replicas contract from
`a ≈ 10`, but `run02` at `P = 1.5` peels away below `a ≈ 1` while the other
two settle at `a ≈ 2`. That is the phase separation, visible directly.

## Things worth trying

Because the whole run takes seconds, this is a good place to build intuition:

```bash
# The baseline: three independent runs, no exchange at all.
jaxrens run -c config.yaml --set inter_re=null

# A denser pressure ladder -- watch swap acceptance stay alive longer.
jaxrens run -c config.yaml --set 'ensemble.pressure=[0.5,0.75,1.0,1.25,1.5]'

# Move the attractive well; the commensurate basins move with it.
jaxrens run -c config.yaml --set backend.mu=1.5
```

- {doc}`02_lj_cluster` — the same workflow on a real interatomic potential.
- {doc}`/user/concepts/replicas` — how the topology is derived, and what the
  swap actually does.
- {doc}`/reference/config` — the `inter_re:` and `ensemble:` surfaces in full.
