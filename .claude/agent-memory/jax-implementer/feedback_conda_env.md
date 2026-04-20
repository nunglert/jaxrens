---
name: Conda env for tests
description: Python interpreter to use when running tests in jaxrens project
type: feedback
---

Always use `/home/nunglert/miniconda3/envs/jaxrens/bin/python` when running pytest or any Python commands for this project.

**Why:** Project uses a conda environment named `jaxrens`; system Python lacks the required JAX/Flax/h5py dependencies.

**How to apply:** Prefix every `python -m pytest` or `python` call with this path.
