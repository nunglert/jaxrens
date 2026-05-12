"""Shared protocols and interfaces for jaxrens.

Every NS move, energy backend, trajectory writer, and callback
implements one of these protocols.
"""

from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp

from jaxrens.types import Box, Params, Positions, Types


# ---------------------------------------------------------------------------
# Move kernel protocol
# ---------------------------------------------------------------------------


class MoveInfo(NamedTuple):
    """Metadata returned by every move step.

    ``reject_reason`` is an int32 scalar:
        0 = accepted
        1 = rejected by likelihood constraint (energy >= emax)
        2 = rejected by cell-geometry constraint (max/min vol, aspect)
        3 = rejected by move-specific prior (e.g. volume V^N factor)

    Moves that only reject by the likelihood constraint (random_walk,
    galilean, etc.) may leave ``reject_reason`` at 0 (default) — callers
    that want per-reason stats should rely on ``accepted`` for those moves.

    ``move_idx`` is the integer index of the move type that was executed,
    as assigned by the MWG wrapper (0-based, matching move_descriptors order).
    Individual move kernels leave this at 0 (default); the MWG step_fn
    overwrites it with the actual chosen index before returning.
    """

    accepted: jnp.ndarray  # bool scalar
    log_likelihood: jnp.ndarray  # float scalar
    n_evaluations: int
    reject_reason: jnp.ndarray = jnp.int32(0)
    move_idx: jnp.ndarray = jnp.int32(0)
    n_grad_evaluations: int = jnp.int32(0)


class StepFn(Protocol):
    def __call__(
        self,
        rng_key: jax.Array,
        state: Any,
        likelihood_constraint: float,
    ) -> tuple[Any, MoveInfo]:
        """Propose a new state satisfying the likelihood constraint.

        Operates on a SINGLE walker. Batched via vmap/pmap by the
        MWG wrapper layer.

        Returns (new_state, info).
        """
        ...


# ---------------------------------------------------------------------------
# Energy backend protocol
# ---------------------------------------------------------------------------


class EnergyFn(Protocol):
    """Unified energy backend interface.

    All backends implement this signature. Params is an opaque pytree
    that flows through the NS loop without inspection.

    NeuralIL-style backends compute neighbors internally during descriptor
    calculation -- no neighbor_list argument. The max_neighbors parameter
    is a compile-time constant; JAX's compilation cache handles retrace
    when the value changes.
    """

    def __call__(
        self,
        params: Params,
        positions: Positions,
        types: Types,
        cell: Box | None = None,
        **unused_kwargs: Any,
    ) -> float: ...


# ---------------------------------------------------------------------------
# Trajectory writer protocol
# ---------------------------------------------------------------------------


class TrajectoryWriter(Protocol):
    """Protocol for trajectory output backends."""

    def write_dead_point(
        self, iteration: int, walker: Any, energy: float
    ) -> None: ...

    def write_walker_snapshot(self, iteration: int, walkers: Any) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# NS callback protocol
# ---------------------------------------------------------------------------


class NSCallback(Protocol):
    """Protocol for NS loop callbacks.

    Called at the outer loop boundary (Python level, outside JIT/pmap).
    """

    def on_start(self, ns_state: Any, start_info: dict | None = None) -> None: ...

    def on_iteration(
        self, iteration: int, ns_state: Any, info: dict
    ) -> None: ...

    def on_dead_point(
        self, iteration: int, dead_walker: Any, energy: float
    ) -> None: ...

    def on_finish(self, ns_state: Any) -> None: ...
