# Introduction

## What jaxrens is

**jaxrens** is a JAX implementation of nested sampling (NS) aimed at
atomistic systems. It was built to make two things routine that
traditional NS codes make awkward:

- **Replica-parallel NS on modern GPUs.** The inner MCMC loop JITs to
  a single `lax.scan`; multiple walkers within a run vmap; multiple
  independent runs (different pressures / compositions / seeds) map
  across devices via `pmap(vmap(...))`.
- **Inter-replica exchange.** Pressure-RENS, composition-morphing
  XRENS, and chemical-potential "semi-grand" swaps are first-class
  features — not post-hoc bolt-ons. You get them by setting
  `inter_re.flavor` in the YAML.

## When to use it

jaxrens is a good fit when you:

- Want NS evidence / phase-transition signals for an atomistic system
  with a differentiable, JAX-native potential or an MLIP you can
  wrap in the `EnergyBackend` protocol.
- Want to run multiple pressures or compositions *concurrently* and
  exchange configurations between them.
- Already live inside the JAX ecosystem (models trained with
  MACE-JAX, NeuralIL, or your own Flax/JAX code).

It's probably **not** the right choice if you're starting from a
pre-trained torch / lammps MLIP and don't want to wrap it in JAX, or
if you need an NS algorithm variant that jaxrens doesn't yet
implement (dynamic NS, posterior re-weighting, etc.).

## The 30-second tour

A minimal run, end to end:

```bash
conda activate jaxrens
cd experiments/lj_rens
sbatch submit.slurm           # or: jaxrens run -c config.yaml
```

`config.yaml` declares: n_live, pressure list (one entry per replica),
moves, backend (`lj`/`mace`/`neuralil`/`harmonic`/...), and
`inter_re.flavor`. The CLI dispatches to single-run or multi-GPU NS
automatically based on how many replicas the config implies.

Output artifacts land in `./output/`:

- `*.log` — Python logger output (live per-iteration monitor).
- `*.run{NN}.traj.extxyz` — per-replica culled-walker trajectories.
- `*.run{NN}.energies` — per-replica Emax log.
- `*.checkpoint.h5` — batched HDF5 restart point.

See the {doc}`../tutorials/index` for runnable walk-throughs and
{doc}`overview` for the conceptual map.

## Quickstart install

```bash
conda create -n jaxrens python=3.11 -y
conda activate jaxrens
cd /path/to/jaxrens
pip install -e ".[dev]"

# For the MACE backend:
pip install -e ".[dev,mace]"

# For the NeuralIL backend:
pip install -e ".[dev,neuralil]"
```

See {doc}`../dev/install` for GPU / CUDA notes and the MACE fork
pin explanation.
