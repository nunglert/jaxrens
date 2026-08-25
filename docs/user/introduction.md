# Introduction

## What JAXRENS is

**JAXRENS** is a JAX implementation of nested sampling (NS) aimed at
atomistic systems. Its development was driven by the need for the
following features, which JAXRENS supports:

- **parallel NS on modern GPUs.** Multiple
  independent runs (different pressures / compositions / seeds) can be run on a single gpu via `vmap(...)` or across multiple devices via `pmap(vmap(...))`.
- **Replica-exchange nested sampling.** Pressure-RENS, composition-morphing
  XRENS, and chemical-potential RENS are naturally integrated.
- **State-of-the-art ML force fields.** Interfaces for NeuralIL, MACE and
  Nequix are implemented, alongside jax-md's analytic Tersoff / EAM
  potentials. Pull requests for other maintained models are welcome.

## When to use it

JAXRENS is a good fit when you:

- Are interested in the unbiased thermodynamic sampling of small (bulk) systems, directly yielding the global partition function.
- Already live inside the JAX ecosystem (models trained with
  MACE-JAX, NeuralIL, or your own Flax/JAX code). 

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
pin explanation. The full set of extras is `mace`, `neuralil`,
`nequix` and `jaxmd`.

## Core modules

`sampling/` (NS loop + move kernels + adaptation) and `cli/` (schema
+ resolver + run entry points) carry most of the complexity;
`backends/` hosts the energy-model adapters (LJ, MACE, NeuralIL,
Nequix, jax-md, toy potentials); `state/` carries the pytree
dataclasses everything else consumes; `constraints/` holds the hard
configuration constraints; `init/`, `io/` and `postprocess/` handle
walker setup, output writers, and thermodynamic post-processing
respectively.

{doc}`/reference/index` opens with an interactive treemap of the
whole package, sized by lines of code, if you want to see the shape
of it before reading any of it.

See {doc}`overview` for a conceptual map of how the subpackages
talk to each other, {doc}`concepts/schema_resolve` for the
CLI's three-layer data flow, and {doc}`/reference/notation` for
the symbols and shape conventions used throughout the rest of
the documentation.

## License

JAXRENS is released under an MIT license.

## Citation

JAXRENS implements the replica-exchange nested sampling method introduced
in:

> N. Unglert, L. B. Pártay and G. K. H. Madsen,
> *Replica Exchange Nested Sampling*,
> J. Chem. Theory Comput. **21**, 7304–7319 (2025).
> [10.1021/acs.jctc.5c00588](https://doi.org/10.1021/acs.jctc.5c00588)

If you use it for active-learning phase-diagram work, please also cite:

> N. Unglert, M. Ketter and G. K. H. Madsen,
> *Active learning potentials for first-principles phase diagrams using
> replica-exchange nested sampling*,
> npj Comput. Mater. **12**, 107 (2026).
> [10.1038/s41524-026-01989-z](https://doi.org/10.1038/s41524-026-01989-z)

BibTeX entries for both are in `citations.bib` at the repository root.
