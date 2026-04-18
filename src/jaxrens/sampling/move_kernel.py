"""MoveKernel: declarative description of an MC move type.

Used by the MWG factory (build_mwg) to assemble move kernels and
dispatch weights without the user touching build_kernel directly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class MoveKernel:
    """Describes one move type for the MWG sampler.

    Attributes:
        name: Human-readable label (e.g. "random_walk", "volume").
        build_kernel: Reference to the module's build_kernel function.
            Signature: build_kernel(energy_fn, params, **kernel_kwargs)
            -> step_fn(rng_key, state, likelihood_constraint) -> (state, MoveInfo)
        kernel_kwargs: Extra keyword arguments forwarded to build_kernel
            (e.g. n_reflect for Galilean, n_atoms for volume).
        weight: Relative probability of selecting this move type.
            Weights are normalized to probabilities by the MWG factory.
        step_size: Per-move step size. Injected into MCState.step_size
            before calling the move's step_fn.
        extra_state_fields: Move-specific fields to add to MCState.
            Keys are field names, values are (type, initializer_fn) tuples.
            The initializer is called as initializer(positions, types) and
            must return the initial value for that field.
            Example for Galilean:
                {"direction": (jnp.ndarray, lambda pos, types: jnp.zeros_like(pos))}
            The MWG factory unions extra_state_fields from all descriptors
            to build the MCState class dynamically.
    """

    name: str
    build_kernel: Callable
    kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    step_size: float = 0.1
    step_size_max: float = 10.0
    min_rate: float = 0.25
    max_rate: float = 0.65
    extra_state_fields: dict[str, tuple[type, Callable]] = field(default_factory=dict)
