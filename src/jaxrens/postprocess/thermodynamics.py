"""Thermodynamic post-processing of nested sampling results.

NumPy-based for unconditional float64 precision.  Sampling is pinned to
float32 by the root ``__init__`` (see the comment there); post-processing
runs in a separate process, so the precision regimes don't interfere.

Energy convention: ``E = -log(L)``, so higher energy = lower likelihood.

Batch-shape support
-------------------
All public functions accept ``dead_energies`` with arbitrary leading batch
dimensions ``(*batch, max_dead)``.  Output shape mirrors the input batch
prefix (output of the trailing axis-reduction).  Examples:

* ``(max_dead,)``          → scalar output
* ``(n_runs, max_dead)``   → ``(n_runs,)``
* ``(G, P, max_dead)``     → ``(G, P)``

JAX-array inputs are auto-converted to NumPy via ``np.asarray``; outputs are
always NumPy float64 (or 0-d / array thereof).  JIT-compilation of these
functions is **not** supported; they run on the host in fp64.  Cv etc.
on a (n_dead,) array of a few thousand entries is far below the cost of
a typical sampling iteration, so JITability bought nothing in practice
while masking precision loss.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_f64(x) -> np.ndarray:
    """Coerce array-like (including JAX arrays) to ``np.ndarray[float64]``."""
    return np.asarray(x, dtype=np.float64)


def _log_t(n_live, n_cull: int = 1) -> np.ndarray:
    """Per-iteration log shrinkage factor, ``log((N+1-c-1)/(N+1-c))``.

    Computed as ``log(N - c) - log(N + 1 - c)`` for numerical stability.
    """
    nl = _as_f64(n_live)
    nc = _as_f64(n_cull)
    return np.log(nl - nc) - np.log(nl + 1.0 - nc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calc_log_weights(
    n_dead,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Compute log prior-mass weights for dead points.

    For dead point ``i`` (0-indexed)::

        log_t   = log((N - n_cull) / (N + 1 - n_cull))
        log_X_i = i * log_t
        log_w_i = log_X_i + log(1 - exp(log_t))

    Args:
        n_dead: Number of dead points collected (scalar).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration (default 1).

    Returns:
        ``np.ndarray[float64]`` of shape ``(n_dead,)``.
    """
    n_dead = int(np.asarray(n_dead))
    log_t = _log_t(n_live, n_cull)
    log_shell = np.log1p(-np.exp(log_t))
    indices = np.arange(n_dead, dtype=np.float64)
    return indices * log_t + log_shell


def calc_log_weights_live(
    n_dead,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Compute log weight for each remaining live walker.

    After ``n_dead`` iterations the remaining prior mass is
    ``X_final = exp(n_dead * log_t)``; each live walker carries
    ``log_w_live = log(X_final / n_live)``.

    Args:
        n_dead: Number of dead points collected (scalar).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration.

    Returns:
        0-d ``np.ndarray[float64]`` (the scalar log weight per live walker).
    """
    n_dead_f = _as_f64(n_dead)
    n_live_f = _as_f64(n_live)
    log_t = _log_t(n_live, n_cull)
    return n_dead_f * log_t - np.log(n_live_f)


def log_evidence(
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Total log evidence ``log Z`` from dead and live points.

    Supports arbitrary leading batch dimensions on ``dead_energies`` and
    ``live_energies``.  Output batch shape mirrors the input (minus the
    trailing ``max_dead`` / ``n_live`` axes).

    Args:
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)`` (broadcast over batch).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration.

    Returns:
        ``log Z`` with shape ``batch`` — scalar for 1-D input.
    """
    dead_E = _as_f64(dead_energies)
    live_E = _as_f64(live_energies)
    n_dead = dead_E.shape[-1]

    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)      # (n_dead,)
    log_w_live = calc_log_weights_live(n_dead, n_live, n_cull)  # scalar

    # Broadcasting handles arbitrary batch prefix; axis=-1 reduces the trailing
    # dead/live walker dimension.
    log_Z_dead = logsumexp(log_w_dead + (-dead_E), axis=-1)
    log_Z_live = logsumexp(log_w_live + (-live_E), axis=-1)
    return np.logaddexp(log_Z_dead, log_Z_live)


def partition_function(
    beta,
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
    dead_volumes=None,
    live_volumes=None,
) -> np.ndarray:
    """Log partition function ``log Z(beta)`` at inverse temperature ``beta``.

    ``Z(beta) = Σ_i w_i exp(-beta · E_i)`` over the merged dead+live ensemble.

    Args:
        beta: Inverse temperature ``1/(k_B T)``.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.
        dead_volumes: Optional PV contributions for dead points, shape must
            broadcast with ``dead_energies``.
        live_volumes: Optional PV contributions for live points, shape must
            broadcast with ``live_energies``.

    Returns:
        ``log Z(beta)`` with shape ``batch`` — scalar for 1-D input.
    """
    beta = _as_f64(beta)
    dead_E = _as_f64(dead_energies)
    live_E = _as_f64(live_energies)
    if dead_volumes is not None:
        dead_E = dead_E + _as_f64(dead_volumes)
    if live_volumes is not None:
        live_E = live_E + _as_f64(live_volumes)

    n_dead = dead_E.shape[-1]
    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    log_w_live = calc_log_weights_live(n_dead, n_live, n_cull)

    log_terms_dead = log_w_dead - beta * dead_E   # (*batch, n_dead)
    log_terms_live = log_w_live - beta * live_E   # (*batch, n_live)

    # Concatenate along the trailing axis (works for any batch prefix).
    log_terms_live_b = np.broadcast_to(
        log_terms_live, log_terms_dead.shape[:-1] + (log_terms_live.shape[-1],),
    )
    all_log_terms = np.concatenate([log_terms_dead, log_terms_live_b], axis=-1)
    return logsumexp(all_log_terms, axis=-1)


def heat_capacity(
    beta,
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Heat capacity ``C_v(beta) = beta² · Var(E)`` at inverse temperature beta.

    Uses the **deviation form** ``Σ w_i (E_i - <E>)²`` instead of the algebraic
    identity ``<E²> - <E>²``: the latter exhibits catastrophic cancellation at
    low T (distribution collapses onto a single walker, ``<E²>`` and ``<E>²``
    both approach ``E_min²``, and their difference is dominated by float noise).
    Even in fp64 the deviation form is the safer default; in fp32 the algebraic
    form gives visible negative-Cv spikes.

    Args:
        beta: Inverse temperature.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        ``C_v`` with shape ``batch`` — scalar for 1-D input.
    """
    beta = _as_f64(beta)
    dead_E = _as_f64(dead_energies)
    live_E = _as_f64(live_energies)
    n_dead = dead_E.shape[-1]

    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    log_w_live_scalar = calc_log_weights_live(n_dead, n_live, n_cull)
    log_w_live = np.full(live_E.shape[-1], log_w_live_scalar, dtype=np.float64)

    # Broadcast to a common batch prefix so concatenation along the trailing
    # axis works for arbitrary input ndims.
    live_E_b = np.broadcast_to(
        live_E, dead_E.shape[:-1] + (live_E.shape[-1],),
    )
    all_E = np.concatenate([dead_E, live_E_b], axis=-1)
    all_log_w = np.concatenate([log_w_dead, log_w_live])  # (n_dead + n_live,)

    log_unnorm = all_log_w - beta * all_E                # (*batch, n_total)
    log_Z = logsumexp(log_unnorm, axis=-1, keepdims=True)
    w = np.exp(log_unnorm - log_Z)                       # normalised weights

    mean_E = np.sum(w * all_E, axis=-1, keepdims=True)
    var_E = np.sum(w * (all_E - mean_E) ** 2, axis=-1)
    return beta**2 * var_E


def expectation(
    observable_values,
    beta,
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Thermal expectation ``<O>(beta)`` for an observable on the dead+live set.

    Args:
        observable_values: Observable values, shape
            ``(*batch, n_dead + n_live)`` or ``(n_dead + n_live,)``
            (broadcast over batch).
        beta: Inverse temperature.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        ``<O>`` with shape ``batch`` — scalar for 1-D input.
    """
    beta = _as_f64(beta)
    dead_E = _as_f64(dead_energies)
    live_E = _as_f64(live_energies)
    obs = _as_f64(observable_values)
    n_dead = dead_E.shape[-1]

    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    log_w_live_scalar = calc_log_weights_live(n_dead, n_live, n_cull)
    log_w_live = np.full(live_E.shape[-1], log_w_live_scalar, dtype=np.float64)

    live_E_b = np.broadcast_to(
        live_E, dead_E.shape[:-1] + (live_E.shape[-1],),
    )
    all_E = np.concatenate([dead_E, live_E_b], axis=-1)
    all_log_w = np.concatenate([log_w_dead, log_w_live])

    # Broadcast obs to the batch prefix when given as 1-D.
    obs_b = np.broadcast_to(obs, all_E.shape)

    log_unnorm = all_log_w - beta * all_E
    log_Z = logsumexp(log_unnorm, axis=-1, keepdims=True)
    w = np.exp(log_unnorm - log_Z)
    return np.sum(w * obs_b, axis=-1)


def free_energy(beta, log_Z_beta) -> np.ndarray:
    """Helmholtz free energy ``F = -log Z(beta) / beta``.

    Args:
        beta: Inverse temperature (scalar or broadcastable).
        log_Z_beta: Log partition function at ``beta``, any shape.

    Returns:
        ``F`` with the same shape as ``log_Z_beta``.
    """
    beta = _as_f64(beta)
    log_Z_beta = _as_f64(log_Z_beta)
    return -log_Z_beta / beta
