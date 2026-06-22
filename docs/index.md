---
myst:
  html_meta:
    "description": "JAX-based nested sampling for atomistic systems"
---

```{toctree}
:caption: User Guide
:hidden:

user/introduction
user/overview
<!-- tutorials/index -->
```

```{toctree}
:caption: Reference
:hidden:

reference/index
reference/cli
reference/config
reference/notation
```

```{toctree}
:caption: Developer Guide
:hidden:

dev/install
dev/contributing
```

# jaxrens documentation

**Date**: {sub-ref}`today`

**Useful links**:
[Source Repository](https://github.com/nunglert/jaxrens) |
[Issues & Ideas](https://github.com/nunglert/jaxrens/issues)

JAX-based nested sampling for atomistic systems — multi-GPU parallel
replicas, pressure-RENS inter-replica exchange, and a pluggable
backend interface (Lennard-Jones, MACE-JAX, NeuralIL, toy potentials).

::::{grid} 3 1 1 1
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: user/introduction
:link-type: doc
:class-header: bg-light
**User Guide** 🚀
^^^
Start here. What jaxrens is, when to use it, and how the pieces
fit together — plus runnable end-to-end tutorials (Lennard-Jones
NS, multi-GPU pressure-RENS, MACE).
:::

:::{grid-item-card}
:link: reference/index
:link-type: doc
:class-header: bg-light
**API reference** 📖
^^^
Every public symbol, every CLI subcommand, every YAML schema
section.  Plus a notation page that pins down what `K`, `P`,
`μ`, `E_max`, `c_i`, … mean across the docs.
:::

:::{grid-item-card}
:link: dev/install
:link-type: doc
:class-header: bg-light
**Developer guide** 👩‍💻
^^^
Setting up the environment, GPU / CUDA notes, adding backends or
move kernels, contributing.
:::
::::
