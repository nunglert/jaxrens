"""Prior-mass bookkeeping for nested sampling with ``n_cull >= 1``.

Single source of truth for the prior-mass ladder ``X_n`` and the dead-point
weights ``a_n``, shared by the in-JIT evidence accumulator
(:mod:`jaxrens.sampling.nested_sampling`), the termination criteria
(:mod:`jaxrens.sampling.termination`) and post-processing
(:mod:`jaxrens.postprocess.thermodynamics`).

Convention follows pymatnest's ``ns_analyse.calc_log_a`` (SENS paper, PRX 4,
031034 (2014), Eq. 3).  With ``N`` live walkers and ``P = n_cull`` walkers
removed per iteration, dead point ``n`` (0-indexed over the global, per-cull
sequence; within an iteration the culled walkers are ordered by descending
energy) carries

.. math::

    \\log X_n = \\sum_{i=0}^{n} \\left[\\log(N - i\\%P) - \\log(N + 1 - i\\%P)\\right]

    \\log a_n = \\log X_n - \\log(N + 1 - (n+1)\\%P)

so one full iteration contracts the prior mass by ``(N + 1 - P) / (N + 1)``.
After ``D`` dead points each of the ``N`` live walkers carries
``X_D / N`` (``X_D`` with ``D + 1`` terms, matching the one-step shift of
``a_n = X_n - X_{n+1}``), so dead + live weights sum to the constant ``X_0``.

NumPy only — imported by the JAX-free post-processing path.
"""

from __future__ import annotations

import numpy as np


def _shrinkage_terms(n_terms: int, n_live: int, n_cull: int) -> np.ndarray:
    """Per-dead-point ``log((N - i%P) / (N + 1 - i%P))`` for ``i < n_terms``."""
    j = np.mod(np.arange(n_terms, dtype=np.int64), n_cull).astype(np.float64)
    n = float(n_live)
    return np.log(n - j) - np.log(n + 1.0 - j)


def check_n_cull(n_live: int, n_cull: int) -> None:
    """Raise if ``(n_live, n_cull)`` do not define a valid prior-mass ladder."""
    if int(n_cull) < 1:
        raise ValueError(f"n_cull must be >= 1, got {n_cull}")
    if int(n_cull) >= int(n_live):
        raise ValueError(
            f"n_cull={n_cull} must be smaller than n_live={n_live}"
        )


def log_alpha(n_live: int, n_cull: int = 1) -> float:
    """Log prior-mass contraction per iteration, ``log((N+1-P)/(N+1))``."""
    check_n_cull(n_live, n_cull)
    return float(np.log(n_live + 1.0 - n_cull) - np.log(n_live + 1.0))


def log_prior_mass(n_dead, n_live: int, n_cull: int = 1) -> np.ndarray:
    """``log X_n`` for dead points ``n = 0 .. n_dead-1``; shape ``(n_dead,)``."""
    check_n_cull(n_live, n_cull)
    return np.cumsum(_shrinkage_terms(int(n_dead), n_live, n_cull))


def log_dead_weights(n_dead, n_live: int, n_cull: int = 1) -> np.ndarray:
    """``log a_n`` for dead points ``n = 0 .. n_dead-1``; shape ``(n_dead,)``.

    Identical to pymatnest's ``calc_log_a(n_dead, n_live, n_cull)``.
    """
    n_dead = int(n_dead)
    log_x = log_prior_mass(n_dead, n_live, n_cull)
    j_next = np.mod(np.arange(1, n_dead + 1, dtype=np.int64), n_cull)
    return log_x - np.log(float(n_live) + 1.0 - j_next.astype(np.float64))


def log_live_weight(n_dead, n_live: int, n_cull: int = 1) -> np.ndarray:
    """Log weight carried by *each* live walker after ``n_dead`` dead points.

    ``log(X_D / N)`` where ``X_D`` sums the first ``D + 1`` shrinkage terms.
    Returns a 0-d float64 array.
    """
    check_n_cull(n_live, n_cull)
    log_x = np.sum(_shrinkage_terms(int(n_dead) + 1, n_live, n_cull))
    return np.asarray(log_x - np.log(float(n_live)), dtype=np.float64)


def iteration_offsets(
    n_live: int, n_cull: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Within-iteration offsets for closed-form weights at any iteration.

    Dead point ``k`` (``0 <= k < n_cull``, descending energy) of iteration
    ``i`` is global dead point ``n = i * n_cull + k``, and

    * ``log X_n = i * log_alpha + x_offset[k]``
    * ``log a_n = i * log_alpha + w_offset[k]``

    This lets the JIT step and the termination criteria evaluate weights at
    iteration ``i`` in O(n_cull) without materialising the whole ladder.

    Returns:
        ``(x_offset, w_offset)``, both float64 arrays of shape ``(n_cull,)``.
    """
    x_offset = log_prior_mass(n_cull, n_live, n_cull)
    w_offset = log_dead_weights(n_cull, n_live, n_cull)
    return x_offset, w_offset
