# Periodic Lennard-Jones: 8 atoms, NPT

A first run on a real interatomic potential, start to finish, from the
terminal. If you have not seen a `jaxrens` config before, start with
{doc}`01_rens_toy`. It is deliberately tiny — 32 live walkers, 500 iterations — so it
finishes in well under a minute on one GPU, while still exercising mixed-move
MWG, burn-in, step-size adaptation and cell constraints.

Everything here is one YAML file and three commands.

## The config

`examples/tutorials/02_lj_npt/config.yaml`:

```{literalinclude} ../../examples/tutorials/02_lj_npt/config.yaml
:language: yaml
```

Two things worth pausing on:

`ensemble.type: npt` is what makes the cell moves meaningful. Under NVT there
is no `P·V` term to balance a volume change, and the cell would simply drift
to whichever bound the prior allows.

`backend.supercell_trafo` is doing real work. With only 8 atoms, the smallest
cell the prior permits is under 1 Å across — far thinner than the 5 Å that a
2.5 σ cutoff needs under the minimum-image convention. Tiling the periodic
images fixes it. Drop it to `[1, 1, 1]` and the next step will tell you.

## 1. Check it before you run it

```bash
cd examples/tutorials/02_lj_npt
jaxrens validate -c config.yaml
```

```text
✓ OK — configuration plan valid
  topology  SingleRun (1 replica, 1 GPU)
  run       n_live=32, max_iterations=500
  moves     4 move(s) [gmc, volume, shear, stretch]
  backend   lj, n_atoms=8
  output    format=extxyz, prefix=lj8
  skipped   backend build, walker placement, initial energies — rerun with --full to check those
```

That is the default tier: topology, divisibility, path existence and geometry
bounds, in about half a second. Before a long job, spend the extra seconds on
`--full`, which additionally builds the backend, places the walkers and
evaluates their initial energies — the tier that proves your model file
actually loads.

This is also where the cutoff problem surfaces. With `supercell_trafo` set to
`[1, 1, 1]`, `validate` warns:

```text
LJ cutoff vs cell-prior bounds: smallest legal cell has worst-case
perpendicular distance 0.9524 A ... below the required 2 * cutoff = 5.0000 A.
LJ energies will undercount neighbours on the tight end of the cell-prior
range.
```

## 2. Run it

```bash
jaxrens run -c config.yaml
```

:::{note}
`jaxrens run` writes no progress to the terminal — it returns when the run
finishes. Progress goes to a log file, so watch it from a second shell:

```bash
tail -f lj8.log
```

The log lands next to `output.working_dir`, not inside it: with
`working_dir: ./output` you get `./lj8.log` beside the `output/` directory
rather than in it. Both are covered by the repository's `.gitignore`.
:::

The interesting lines are the per-iteration monitor rows, one every
`output.info_interval`:

```text
Starting NS run: 32 walkers, 8 atoms, max_iter=500, n_mcmc=10
iter=0    Emax=21.2346  log_Z=-24.7003  dt=29.7s  nE=7.88e+03  nG=5.47e+03
iter=50   Emax=12.6356  log_Z=-14.0262  dt=8.9s   nE=4.20e+04  nG=3.38e+04
iter=100  Emax=11.2156  log_Z=-13.2186  dt=0.4s   nE=7.60e+04  nG=6.21e+04
iter=100  T_est=26350.617 K
iter=150  Emax=10.017   log_Z=-12.8879  dt=0.4s   nE=1.10e+05  nG=9.08e+04
iter=200  Emax=8.74885  log_Z=-12.6956  dt=0.4s   nE=1.44e+05  nG=1.20e+05
iter=200  T_est=9299.724 K
```

Read it as: `Emax` is the nested-sampling energy threshold, descending
monotonically by construction — that is the run making progress. `log_Z`
climbs toward the converged log-evidence. `T_est` is the finite-difference
temperature estimate, which needs two samples before it can report. `nE` and
`nG` count energy and gradient evaluations.

The `dt` column is worth a second look: the first interval costs ~30 s and the
next ~9 s, then it drops to 0.4 s. That is JIT compilation, once per distinct
array shape, not the sampler being slow. A production run amortises it away
entirely.

## 3. What you get

```bash
ls output/
```

| File | What it holds |
|---|---|
| `lj8.energies` | the dead-point energy ladder — the input to every thermodynamic estimator |
| `lj8.traj.extxyz` | the culled walker at each iteration, ASE-readable |
| `lj8.adaptation.h5` | per-move step-size and acceptance traces |
| `lj8.checkpoint.h5`, `lj8.final.checkpoint.h5` | restart points |
| `lj8.config.snapshot.yaml` | the fully-defaulted config this run actually used |

That last one matters more than it looks: it records every value the run used,
including the defaults you never wrote down, which is what makes a result
reproducible months later.

## 4. Plot it

No Python required — `jaxrens plot` picks the plot from the file suffix:

```bash
jaxrens plot output/lj8.energies
jaxrens plot output/lj8.adaptation.h5
```

```text
Wrote output/lj8.energies.png
Wrote output/lj8.adaptation.png
```

```{image} ../_static/figures/tutorials/lj8.energies.png
:alt: dead-point energy and volume trails for the Lennard-Jones run
:width: 100%
```

The energies plot is the dead-point trail, with the cell volume alongside it
because this run is NPT. For a multi-replica run the same command overlays
every replica on both panels.

```{image} ../_static/figures/tutorials/lj8.adaptation.png
:alt: per-move step-size and acceptance-rate traces
:width: 100%
```

The adaptation plot carries one trace per move, which is what makes a mixed
move set debuggable: `gmc`, `volume`, `shear` and `stretch` adapt
independently, and a single misbehaving one shows up immediately. Use `-o` to
choose the output path.

If the acceptance trace sits pinned at 0 or 1 for a move, that move is
misconfigured — see {doc}`../user/troubleshooting`.

- {doc}`03_mace_run` — the same shape with a machine-learned potential.
- {doc}`/reference/config` — every key you can put in that YAML.
- {doc}`/user/concepts/ns_loop` — what the two loops are actually doing.
