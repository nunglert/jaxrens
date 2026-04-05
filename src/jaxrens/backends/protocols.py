"""Energy backend protocols and types.

Formalizes the EnergyFn protocol and related types used by all backends.
"""

from typing import Any, Protocol

import jax.numpy as jnp

from jaxrens.types import Box, Params, Positions, Types


class EnergyFn(Protocol):
    """Unified energy backend interface.

    All backends implement this signature. Params is an opaque pytree
    that flows through the NS loop without inspection.

    NeuralIL-style backends compute neighbors internally during descriptor
    calculation -- no neighbor_list argument. The max_neighbors parameter
    is a compile-time constant baked into the descriptor generator, handled
    by the CompiledKernelSet dispatch layer (see kernel_dispatch.py).

    Implementations must be:
    - Pure functions (no side effects)
    - Compatible with jax.jit, jax.vmap, jax.pmap
    - Differentiable wrt positions (for gradient-based moves)
    """

    def __call__(
        self,
        params: Params,
        positions: Positions,
        types: Types,
        box: Box | None = None,
        **unused_kwargs: Any,
    ) -> jnp.ndarray:
        """Compute total potential energy.

        Args:
            params: Opaque pytree of backend-specific parameters.
            positions: Atomic positions, shape (n_atoms, 3).
            types: Integer atom type codes, shape (n_atoms,).
            box: Unit cell matrix (3, 3) or None for non-periodic.
            **unused_kwargs: Forward-compatibility for future args
                (e.g., temperature for NS-SMC).

        Returns:
            Scalar potential energy.
        """
        ...


class BackendFactory(Protocol):
    """Factory protocol for creating energy backends."""

    def __call__(self, **kwargs: Any) -> tuple[EnergyFn, Params]:
        """Create an energy function and its initial parameters.

        Returns:
            (energy_fn, params) tuple.
        """
        ...
