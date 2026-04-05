"""Shared type aliases for jaxrens."""

from typing import Any

import jax
import jax.numpy as jnp

# Array types
Positions = jnp.ndarray  # (n_atoms, 3)
Types = jnp.ndarray  # (n_atoms,) integer atom type codes
Box = jnp.ndarray  # (3, 3) cell matrix or None for non-periodic
PRNGKey = jax.Array  # JAX PRNG key

# Opaque pytree: Flax params, plain arrays, or any JAX-compatible tree
Params = Any
