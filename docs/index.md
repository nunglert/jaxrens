---
myst:
  html_meta:
    "description": "JAX-based nested sampling for atomistic systems"
---

# jaxrens

JAX-based nested sampling for atomistic systems — multi-GPU parallel
replicas, pressure-RENS inter-replica exchange, and a pluggable
backend interface (Lennard-Jones, MACE-JAX, NeuralIL, toy potentials).

::::{grid} 2
:gutter: 3

:::{grid-item-card} User Guide
:link: user/introduction
:link-type: doc

Start here. What jaxrens is, when to use it, and how the pieces fit
together.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Runnable end-to-end walk-throughs — Lennard-Jones NS, multi-GPU
pressure-RENS, MACE.
:::

:::{grid-item-card} API Reference
:link: reference/index
:link-type: doc

Every public symbol, every CLI subcommand, every YAML schema section.
:::

:::{grid-item-card} Developer Guide
:link: dev/install
:link-type: doc

Setting up the environment, adding backends, adding moves,
contributing.
:::
::::

```{toctree}
:maxdepth: 1
:hidden:

user/introduction
user/overview
tutorials/index
reference/index
reference/cli
dev/install
```
