"""Thermodynamic post-processing of nested sampling results.

All functions are pure JAX and jit-compatible. Energy convention:
E = -log(L), so higher energy = lower likelihood.

Batch-shape support
-------------------
All public functions accept ``dead_energies`` with arbitrary leading batch
dimensions ``(*batch, max_dead)``.  The ``max_dead`` axis is always last.

Examples:

* ``(max_dead,)``          — single run (``SingleRun``)
* ``(n_runs, max_dead)``   — parallel runs (``VmapRuns``)
* ``(G, P, max_dead)``     — multi-GPU runs (``PmapVmapRuns``)

The strategy is **reshape-then-compute**:

1. Merge all leading dims into a single ``n_flat`` axis.
2. Run the existing 1-D per-run kernel via ``jax.vmap``.
3. Reshape outputs back to the original batch prefix.

This avoids doubling the JIT surface and gives predictable output shapes:
output batch dims mirror input batch dims (minus ``max_dead``).

Existing callers passing 1-D ``dead_energies`` receive scalar outputs, as
before.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Internal 1-D per-run kernels (unchanged from the original implementation)
# ---------------------------------------------------------------------------


def _calc_log_weights_1d(
    n_dead: int | jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Per-run log weights — operates on scalar n_dead / n_live."""
    n_live = jnp.asarray(n_live, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
    n_cull = jnp.asarray(n_cull, dtype=n_live.dtype)

    log_t = jnp.log(n_live - n_cull) - jnp.log(n_live + 1.0 - n_cull)
    log_shell = jnp.log1p(-jnp.exp(log_t))

    indices = jnp.arange(n_dead, dtype=n_live.dtype)
    log_X = indices * log_t
    log_w = log_X + log_shell
    return log_w


def _calc_log_weights_live_1d(
    n_dead: int | jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Per-run live log weight — scalar output."""
    n_live = jnp.asarray(n_live, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
    n_cull = jnp.asarray(n_cull, dtype=n_live.dtype)
    n_dead = jnp.asarray(n_dead, dtype=n_live.dtype)

    log_t = jnp.log(n_live - n_cull) - jnp.log(n_live + 1.0 - n_cull)
    log_X_final = n_dead * log_t
    log_w_live = log_X_final - jnp.log(n_live)
    return log_w_live


def _log_evidence_1d(
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Log evidence for a single 1-D dead_energies array."""
    n_dead = dead_energies.shape[0]
    log_w_dead = _calc_log_weights_1d(n_dead, n_live, n_cull)
    log_Z_dead = logsumexp(log_w_dead + (-dead_energies))

    log_w_live = _calc_log_weights_live_1d(n_dead, n_live, n_cull)
    log_Z_live = logsumexp(log_w_live + (-live_energies))

    return jnp.logaddexp(log_Z_dead, log_Z_live)


def _partition_function_1d(
    beta: jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
    dead_volumes: jnp.ndarray | None = None,
    live_volumes: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Partition function for a single 1-D dead_energies array."""
    n_dead = dead_energies.shape[0]
    log_w_dead = _calc_log_weights_1d(n_dead, n_live, n_cull)
    dead_E = dead_energies if dead_volumes is None else dead_energies + dead_volumes
    log_terms_dead = log_w_dead - beta * dead_E

    log_w_live = _calc_log_weights_live_1d(n_dead, n_live, n_cull)
    live_E = live_energies if live_volumes is None else live_energies + live_volumes
    log_terms_live = log_w_live - beta * live_E

    return logsumexp(jnp.concatenate([log_terms_dead, log_terms_live]))


def _heat_capacity_1d(
    beta: jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Heat capacity for a single 1-D dead_energies array.

    Computes ``β² · Var(E)`` where ``Var(E) = Σ w_i (E_i - <E>)²``.  The
    deviation form is used in place of the algebraic ``<E²> - <E>²``
    identity to avoid catastrophic cancellation at low T (where the
    distribution collapses onto a single walker, so both moments are
    close to ``E_min²`` and their difference is dominated by float noise
    — visible as wild spikes / negative Cv in fp32).
    """
    n_dead = dead_energies.shape[0]
    log_w_dead = _calc_log_weights_1d(n_dead, n_live, n_cull)
    log_w_live = _calc_log_weights_live_1d(n_dead, n_live, n_cull)
    log_w_live_arr = jnp.full(live_energies.shape[0], log_w_live)

    all_log_w = jnp.concatenate([log_w_dead, log_w_live_arr])
    all_E = jnp.concatenate([dead_energies, live_energies])

    log_unnorm = all_log_w - beta * all_E
    log_Z = logsumexp(log_unnorm)
    w = jnp.exp(log_unnorm - log_Z)

    mean_E = jnp.sum(w * all_E)
    var_E = jnp.sum(w * (all_E - mean_E) ** 2)
    return beta**2 * var_E


def _expectation_1d(
    observable_values: jnp.ndarray,
    beta: jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Thermal expectation for a single 1-D dead_energies array."""
    n_dead = dead_energies.shape[0]
    log_w_dead = _calc_log_weights_1d(n_dead, n_live, n_cull)
    log_w_live = _calc_log_weights_live_1d(n_dead, n_live, n_cull)
    log_w_live_arr = jnp.full(live_energies.shape[0], log_w_live)

    all_log_w = jnp.concatenate([log_w_dead, log_w_live_arr])
    all_E = jnp.concatenate([dead_energies, live_energies])

    log_unnorm = all_log_w - beta * all_E
    log_Z = logsumexp(log_unnorm)
    w = jnp.exp(log_unnorm - log_Z)
    return jnp.sum(w * observable_values)


# ---------------------------------------------------------------------------
# Batch-shape dispatch helpers
# ---------------------------------------------------------------------------


def _batch_apply(fn_1d, *replica_arrays: jnp.ndarray) -> jnp.ndarray:
    """Apply a 1-D per-run function across any leading batch dims.

    The first positional array's leading shape (minus its last axis) defines
    the batch prefix.  Subsequent arrays either share that prefix or are
    1-D (broadcast over the batch).

    Args:
        fn_1d: Function whose positional arguments correspond 1-to-1 with
            ``replica_arrays`` after flattening to single-replica shape.
            Returns a scalar (or any pytree of scalars) per replica.
        *replica_arrays: One or more per-replica arrays.  Each has either
            a ``(*batch, last)`` shape (gets flattened to ``(n_flat, last)``)
            or a ``(last,)`` shape (gets broadcast to ``(n_flat, last)``).

    Returns:
        Output with shape ``batch``.  Scalar (0-D) for 1-D input.
    """
    if len(replica_arrays) == 0:
        raise ValueError("_batch_apply requires at least one replica array")

    primary = jnp.asarray(replica_arrays[0])
    batch_shape = primary.shape[:-1]

    if len(batch_shape) == 0:
        # 1-D primary: skip vmap, call directly (callers expect a scalar).
        return fn_1d(*(jnp.asarray(a) for a in replica_arrays))

    n_flat = int(np.prod(batch_shape))
    flat = []
    for a in replica_arrays:
        a = jnp.asarray(a)
        if a.ndim == 1:
            flat.append(jnp.broadcast_to(a[None], (n_flat, a.shape[0])))
        else:
            flat.append(a.reshape((n_flat,) + a.shape[len(batch_shape):]))

    out_flat = jax.vmap(fn_1d)(*flat)
    return out_flat.reshape(batch_shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calc_log_weights(
    n_dead: int | jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute log prior-mass weights for dead points.

    For dead point i (0-indexed):
        log_t = log((N - n_cull) / (N + 1 - n_cull))
        log_X_i = i * log_t
        log_w_i = log_X_i + log(1 - exp(log_t))

    This function operates on **scalar** ``n_dead`` / ``n_live`` and returns
    a 1-D weight vector of length ``n_dead``.  It is not batch-aware by
    itself; batch handling lives in the per-run callers.

    Args:
        n_dead: Number of dead points collected (scalar).
        n_live: Number of live walkers (scalar).
        n_cull: Number of walkers culled per iteration (default 1).

    Returns:
        Array of log weights, shape ``(n_dead,)``.
    """
    return _calc_log_weights_1d(n_dead, n_live, n_cull)


def calc_log_weights_live(
    n_dead: int | jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute log weight for each remaining live walker.

    After n_dead iterations the remaining prior mass is
    X_final = exp(n_dead * log_t). Each live walker gets
    log_w_live = log(X_final / n_live).

    Args:
        n_dead: Number of dead points collected (scalar).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration.

    Returns:
        Scalar log weight for each live walker.
    """
    return _calc_log_weights_live_1d(n_dead, n_live, n_cull)


def log_evidence(
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute total log evidence Z from dead and live points.

    Supports arbitrary leading batch dimensions on ``dead_energies`` and
    ``live_energies``.  The batch shape of the output mirrors the input
    (minus the trailing ``max_dead`` / ``n_live`` axes).

    Args:
        dead_energies: Energies of dead points, shape ``(*batch, n_dead)``.
        live_energies: Energies of remaining live walkers,
            shape ``(*batch, n_live)`` or ``(n_live,)`` (broadcast).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration.

    Returns:
        Log Z with shape ``batch`` — scalar for 1-D input.
    """
    return _batch_apply(
        lambda de, le: _log_evidence_1d(de, le, n_live, n_cull),
        dead_energies, live_energies,
    )


def partition_function(
    beta: float | jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
    dead_volumes: jnp.ndarray | None = None,
    live_volumes: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute log Z(beta) at inverse temperature beta.

    Z(beta) = sum_i w_i * exp(-beta * E_i)

    Supports arbitrary leading batch dimensions on ``dead_energies`` /
    ``live_energies``.  Output shape mirrors the batch prefix.

    Args:
        beta: Inverse temperature 1/(k_B T).
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.
        dead_volumes: Optional PV contributions for dead points;
            shape must match ``dead_energies``.
        live_volumes: Optional PV contributions for live points;
            shape must match ``live_energies``.

    Returns:
        Log Z(beta) with shape ``batch`` — scalar for 1-D input.
    """
    beta = jnp.asarray(beta)
    # The four-way None dispatch keeps optional volumes out of the vmap
    # signature; ``_batch_apply`` then handles flatten/vmap/reshape uniformly.
    if dead_volumes is not None and live_volumes is not None:
        return _batch_apply(
            lambda de, le, dv, lv: _partition_function_1d(
                beta, de, le, n_live, n_cull, dv, lv,
            ),
            dead_energies, live_energies, dead_volumes, live_volumes,
        )
    if dead_volumes is not None:
        return _batch_apply(
            lambda de, le, dv: _partition_function_1d(
                beta, de, le, n_live, n_cull, dv, None,
            ),
            dead_energies, live_energies, dead_volumes,
        )
    if live_volumes is not None:
        return _batch_apply(
            lambda de, le, lv: _partition_function_1d(
                beta, de, le, n_live, n_cull, None, lv,
            ),
            dead_energies, live_energies, live_volumes,
        )
    return _batch_apply(
        lambda de, le: _partition_function_1d(
            beta, de, le, n_live, n_cull, None, None,
        ),
        dead_energies, live_energies,
    )


def heat_capacity(
    beta: float | jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute heat capacity C_v at inverse temperature beta.

    C_v = beta^2 * (<E^2> - <E>^2)

    Supports arbitrary leading batch dimensions on ``dead_energies`` /
    ``live_energies``.  Output shape mirrors the batch prefix.

    Args:
        beta: Inverse temperature.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        C_v with shape ``batch`` — scalar for 1-D input.
    """
    beta = jnp.asarray(beta)
    return _batch_apply(
        lambda de, le: _heat_capacity_1d(beta, de, le, n_live, n_cull),
        dead_energies, live_energies,
    )


def expectation(
    observable_values: jnp.ndarray,
    beta: float | jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute thermal expectation <O>(beta) for an observable.

    Supports arbitrary leading batch dimensions.  ``observable_values``
    must have shape ``(*batch, n_dead + n_live)`` or ``(n_dead + n_live,)``
    (broadcast over batch).

    Args:
        observable_values: Observable values, shape
            ``(*batch, n_dead + n_live)`` or ``(n_dead + n_live,)``.
        beta: Inverse temperature.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        <O> with shape ``batch`` — scalar for 1-D input.
    """
    beta = jnp.asarray(beta)
    # ``_batch_apply`` keys the batch shape off its first replica array;
    # passing ``dead_energies`` first matches the public batch convention
    # (output shape mirrors ``dead_energies.shape[:-1]``).
    return _batch_apply(
        lambda de, le, obs: _expectation_1d(obs, beta, de, le, n_live, n_cull),
        dead_energies, live_energies, observable_values,
    )


def free_energy(
    beta: float | jnp.ndarray,
    log_Z_beta: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Helmholtz free energy F = -log Z(beta) / beta.

    Accepts any shape for ``log_Z_beta``; the batch shape is preserved.

    Args:
        beta: Inverse temperature (scalar or broadcastable).
        log_Z_beta: Log partition function at beta, shape ``batch``.

    Returns:
        F with the same shape as ``log_Z_beta``.
    """
    beta = jnp.asarray(beta)
    return -log_Z_beta / beta
