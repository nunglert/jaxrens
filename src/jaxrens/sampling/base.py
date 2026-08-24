"""The value every move step returns.

Lives next to the move kernels that produce it and the MWG wrapper that
consumes it, mirroring ``jaxrens.backends.base`` for the energy contract.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op


class MoveInfo(NamedTuple):
    """Metadata returned by every move step.

    ``reject_reason`` is an int32 scalar:
        0 = accepted
        1 = rejected by likelihood constraint (energy >= emax)
        2 = rejected by cell-geometry constraint (max/min vol, aspect)
        3 = rejected by move-specific prior (e.g. volume V^N factor)
        4 = rejected by a configuration constraint (e.g. minimum distance);
            see jaxrens.constraints. Set by the MWG constraint gate, not by
            the move kernels themselves.

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
    # NumPy (not jnp) scalars: a ``jnp.int32(0)`` default would execute a JAX
    # op at class-definition (import) time, forcing JAX to bring up a backend
    # — including a CUDA probe that errors on GPU-less nodes — even for
    # import-only consumers like ``jaxrens dump-schema``.  ``np.int32(0)``
    # carries the same int32 dtype without touching a device.
    reject_reason: jnp.ndarray = np.int32(0)
    move_idx: jnp.ndarray = np.int32(0)
    n_grad_evaluations: int = np.int32(0)
