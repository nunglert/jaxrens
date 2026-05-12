# Introduction

## What jaxrens is

**jaxrens** is a JAX implementation of nested sampling (NS) aimed at
atomistic systems. It's development was driven by the need for the following features, which JAXRENS supports:

- **parallel NS on modern GPUs.** Multiple
  independent runs (different pressures / compositions / seeds) can be run on a single gpu via `vmap(...)` or across multiple devices via `pmap(vmap(...))`.
- **Replica-exchange nested sampling.** Pressure-RENS, composition-morphing
  XRENS, and chemical-potential RENS are naturally integrated.
- **State-of-the-art ML force fields.** Interfaces for NeuralIL, MACE and nequix are implemented. Pull requests for other maintained models are welcome.

## When to use it

jaxrens is a good fit when you:

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
pin explanation.

## Core modules

Subpackages are sized by lines of code — a crude proxy for "where
the complexity lives". `sampling/` (NS loop + move kernels +
adaptation) and `cli/` (schema + resolver + run entry points) are
the heavy hitters; `backends/` hosts the energy-model adapters
(LJ, MACE, NeuralIL, toy potentials); `state/` carries the pytree
dataclasses everything else consumes; `init/`, `io/`,
`postprocess/` handle walker setup, output writers, and
thermodynamic post-processing respectively.

:::{only} html
:::{raw} html
<iframe
    src="../_static/figures/pkg_treemap.html"
    width="100%"
    height="620"
    style="border: 1px solid #e1e1e1; border-radius: 4px; background: white;"
    loading="lazy"
    title="jaxrens package treemap (interactive)"
></iframe>
:::
:::

:::{only} latex
```{image} /_static/figures/pkg_treemap.svg
:alt: treemap of the jaxrens package — subpackages and modules sized by LoC
:align: center
:width: 100%
```
:::

See {doc}`overview` for a conceptual map of how the subpackages
talk to each other, {doc}`concepts/schema_resolve` for the
CLI's three-layer data flow, and {doc}`/reference/notation` for
the symbols and shape conventions used throughout the rest of
the documentation.

## License

JAXRENS is released under an MIT license.

## Citation

If you use JAXRENS please cite our publication.

