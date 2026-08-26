# 3. LJ cluster — heat capacity

The same 8 atoms as {doc}`03_lj_npt`, with the cell dropped. No periodic
images, no pressure, no `volume`/`shear`/`stretch` moves — just a cluster in
vacuum and the classic nested-sampling question: what does its heat capacity
look like? Finishes in well under a minute, even on a CPU.

## The config

`examples/tutorials/02_lj_cluster/config.yaml`:

```{literalinclude} ../../examples/tutorials/02_lj_cluster/config.yaml
:language: yaml
```

Two things changed relative to {doc}`03_lj_npt`, and everything else follows
from them:

`backend.periodic: false` means there are no periodic images and no
`supercell_trafo` to tune — the `cell:` block still exists, but only to bound
where the initial 8 atoms are drawn from. It is never sampled.

`ensemble.type: nvt` drops the `P·V` term, which is what makes `volume`,
`shear` and `stretch` pointless here: with no cell to act on, a single
`gmc` move is the whole sampler.

## 1. Run it

```bash
cd examples/tutorials/02_lj_cluster
jaxrens validate -c config.yaml --full
jaxrens run -c config.yaml
tail -f lj8cluster.log     # from a second shell; `run` prints nothing
```

```text
Starting NS run: 128 walkers, 8 atoms, max_iter=4000, n_mcmc=10
iter=0     Emax=7.66511   log_Z=-12.5171  dt=4.2s  nE=6.42e+03  nG=6.42e+03
iter=200   Emax=-2.02892  log_Z=0.8851    dt=0.5s  nE=2.14e+05  nG=2.14e+05
iter=200   T_est=9904.120 K
iter=600   Emax=-3.75971  log_Z=1.6527    dt=0.3s  nE=6.10e+05  nG=6.10e+05
iter=1000  Emax=-5.45988  log_Z=1.7730    dt=0.3s  nE=1.00e+06  nG=1.00e+06
iter=1600  Emax=-7.62947  log_Z=1.8001    dt=0.4s  nE=1.60e+06  nG=1.60e+06
NS terminated at iter 1663: Prior mass negligible compared to evidence
NS complete: 1664 dead points, log_Z=1.8022
NS finished: 1664 iterations, log_Z=1.8022, elapsed=7s
```

Two differences from the NPT run worth noticing. First, this one terminates
itself at iteration 1663 instead of running to `max_iterations=4000` — the
`prior_mass` criterion fires once the remaining prior volume is too small to
move `log_Z` further, and a cluster's configuration space is small enough
that this happens quickly. Second, `elapsed=7s`: dropping the cell doesn't
just remove parameters, it removes an entire class of moves (and their JIT
compilation), which is most of why this finishes faster than the NPT run
despite needing more than three times as many iterations.

## 2. What you get

```bash
ls output/
```

| File | What it holds |
|---|---|
| `lj8cluster.energies` | the dead-point energy ladder — the input to every thermodynamic estimator |
| `lj8cluster.traj.extxyz` | the culled walker at each iteration, ASE-readable |
| `lj8cluster.adaptation.h5` | step-size and acceptance traces (one move here: `gmc`) |
| `lj8cluster.final.checkpoint.h5` | the restart point `jaxrens analyze` reads below |
| `lj8cluster.config.snapshot.yaml` | the fully-defaulted config this run actually used |

## 3. Plot it

```bash
jaxrens plot output/lj8cluster.energies
jaxrens plot output/lj8cluster.adaptation.h5
```

```{image} ../_static/figures/tutorials/lj8cluster.energies.png
:alt: dead-point energy trail for the non-periodic Lennard-Jones cluster
:width: 100%
```

The bottom panel is flat: `jaxrens plot` always draws a volume trail when one
is present in the log, and here it is — just constant, because the cell is
never sampled. That flat line *is* the NVT/non-periodic story: compare it to
the same panel in {doc}`03_lj_npt`.

```{image} ../_static/figures/tutorials/lj8cluster.adaptation.png
:alt: step-size and acceptance-rate trace for the single gmc move
:width: 100%
```

One move, one trace: `gmc`'s acceptance rate settles in the 0.3–0.6 band the
`adaptation.defaults` block asks for, with the step size shrinking in steps
as the accessible region contracts.

## 4. Heat capacity: `jaxrens analyze`

Energies and acceptance rates are sampler diagnostics — they tell you the run
behaved. The heat capacity is the physics: `jaxrens analyze` turns the same
dead-point ladder into a thermodynamic observable by reweighting it at a
sweep of temperatures. The primary output is data, not a picture — `T`
against the observable, ready for a notebook or a paper plot without a
detour through a PNG. `--format csv` (the default) writes fixed-width,
right-aligned columns — a CSV meant to actually be read, not just parsed:

```bash
jaxrens analyze output/lj8cluster.final.checkpoint.h5 \
    --t-min 0.05 --t-max 3.0
```

```text
Wrote output/lj8cluster.heat_capacity.csv
```

```text
               T,              Cv
            0.05,      0.44239143
     0.064824121,      0.72091224
     0.079648241,      0.88485042
...
```

`--format json` is the other option — self-describing, and it nests however
many dimensions an observable actually has rather than forcing one row per
temperature, which matters the moment an observable stops being a single
scalar per `T` (e.g. a per-species heat capacity):

```bash
jaxrens analyze output/lj8cluster.final.checkpoint.h5 \
    --t-min 0.05 --t-max 3.0 --format json
```

```text
{
  "observable": "heat_capacity",
  "column": "Cv",
  "prefix": "lj8cluster",
  "k_b": 1.0,
  "T": [
    0.05,
    0.06482412060301508,
    0.07964824120603016,
    ...
  ],
  "Cv": [
    0.4423914311076833,
    0.7209122395837029,
    0.8848504158531625,
    ...
  ]
}
```

Add `--plot` to also render a PNG from the same data, the same way `jaxrens
plot` renders one from an artefact file (works with either `--format`):

```bash
jaxrens analyze output/lj8cluster.final.checkpoint.h5 \
    --t-min 0.05 --t-max 3.0 --plot
```

```text
Wrote output/lj8cluster.heat_capacity.csv
Wrote output/lj8cluster.heat_capacity.png
```

```{image} ../_static/figures/tutorials/lj8cluster.heat_capacity.png
:alt: heat capacity Cv vs temperature for the 8-atom LJ cluster, showing a sharp peak near T=0.5
:width: 70%
:align: center
```

Unlike `jaxrens plot`, `analyze` dispatches on a **checkpoint** file rather
than a single self-contained artefact: `n_live` and the live-walker energies
live in `lj8cluster.final.checkpoint.h5`, but the dead-point energies are in
the sibling `lj8cluster.energies` — `analyze` loads both from the same
directory (via `Monitor.from_directory` under the hood), which is why it
needs no second file argument. There is no default `--t-min`/`--t-max`: the
right temperature range depends on the backend's energy units, so the CLI
asks rather than guesses.

The peak near `T ≈ 0.5` is a structural transition — the cluster has more
than one geometrically distinct low-energy arrangement, and around that
temperature the population starts spreading across them instead of sitting
in the single deepest one. `--observable` also accepts `partition_function`
(`log Z(T)`) and `free_energy` (`F(T)`), the other two quantities a Monitor
computes from the same dead-point ladder.

## Next

- {doc}`03_lj_npt` — the same 8 atoms, with the cell back: periodic images,
  pressure, and the moves that act on a cell.
- {doc}`/reference/config` — every key you can put in that YAML.
- {doc}`/user/concepts/ns_loop` — what the two loops are actually doing.
