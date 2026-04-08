"""Thermodynamic post-processing of nested sampling results.

All functions are pure JAX and jit-compatible. Energy convention:
E = -log(L), so higher energy = lower likelihood.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def calc_log_weights(n_dead: int | jnp.ndarray, n_live: int | jnp.ndarray, n_cull: int = 1) -> jnp.ndarray:
    """Compute log prior-mass weights for dead points.

    For dead point i (0-indexed):
        log_t = log((N - n_cull) / (N + 1 - n_cull))
        log_X_i = i * log_t
        log_w_i = log_X_i + log(1 - exp(log_t))

    Args:
        n_dead: Number of dead points collected.
        n_live: Number of live walkers.
        n_cull: Number of walkers culled per iteration (default 1).

    Returns:
        Array of log weights, shape (n_dead,).
    """
    n_live = jnp.asarray(n_live, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
    n_cull = jnp.asarray(n_cull, dtype=n_live.dtype)

    log_t = jnp.log(n_live - n_cull) - jnp.log(n_live + 1.0 - n_cull)
    # log(1 - exp(log_t)) = log(1 - t) where t = (N-n_cull)/(N+1-n_cull)
    # = log(1/(N+1-n_cull)) = -log(N+1-n_cull)
    log_shell = jnp.log1p(-jnp.exp(log_t))

    indices = jnp.arange(n_dead, dtype=n_live.dtype)
    log_X = indices * log_t
    log_w = log_X + log_shell
    return log_w


def calc_log_weights_live(n_dead: int | jnp.ndarray, n_live: int | jnp.ndarray, n_cull: int = 1) -> jnp.ndarray:
    """Compute log weight for each remaining live walker.

    After n_dead iterations the remaining prior mass is
    X_final = exp(n_dead * log_t). Each live walker gets
    log_w_live = log(X_final / n_live).

    Args:
        n_dead: Number of dead points collected.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        Scalar log weight for each live walker.
    """
    n_live = jnp.asarray(n_live, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
    n_cull = jnp.asarray(n_cull, dtype=n_live.dtype)
    n_dead = jnp.asarray(n_dead, dtype=n_live.dtype)

    log_t = jnp.log(n_live - n_cull) - jnp.log(n_live + 1.0 - n_cull)
    log_X_final = n_dead * log_t
    log_w_live = log_X_final - jnp.log(n_live)
    return log_w_live


def log_evidence(
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute total log evidence Z from dead and live points.

    Args:
        dead_energies: Energies of dead points, shape (n_dead,).
        live_energies: Energies of remaining live walkers, shape (n_live,).
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        Scalar log Z.
    """
    n_dead = dead_energies.shape[0]

    # Dead contribution
    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    log_L_dead = -dead_energies
    log_Z_dead = logsumexp(log_w_dead + log_L_dead)

    # Live contribution
    log_w_live = calc_log_weights_live(n_dead, n_live, n_cull)
    log_L_live = -live_energies
    log_Z_live = logsumexp(log_w_live + log_L_live)

    return jnp.logaddexp(log_Z_dead, log_Z_live)


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

    Args:
        beta: Inverse temperature 1/(k_B T).
        dead_energies: Dead point energies, shape (n_dead,).
        live_energies: Live walker energies, shape (n_live,).
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.
        dead_volumes: Optional PV contributions for dead points.
        live_volumes: Optional PV contributions for live points.

    Returns:
        Scalar log Z(beta).
    """
    beta = jnp.asarray(beta)
    n_dead = dead_energies.shape[0]

    # Dead contributions
    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    dead_E = dead_energies
    if dead_volumes is not None:
        dead_E = dead_E + dead_volumes
    log_terms_dead = log_w_dead - beta * dead_E

    # Live contributions
    log_w_live = calc_log_weights_live(n_dead, n_live, n_cull)
    live_E = live_energies
    if live_volumes is not None:
        live_E = live_E + live_volumes
    log_terms_live = log_w_live - beta * live_E

    all_log_terms = jnp.concatenate([log_terms_dead, log_terms_live])
    return logsumexp(all_log_terms)


def heat_capacity(
    beta: float | jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute heat capacity C_v at inverse temperature beta.

    C_v = beta^2 * (<E^2> - <E>^2)

    Args:
        beta: Inverse temperature.
        dead_energies: Dead point energies, shape (n_dead,).
        live_energies: Live walker energies, shape (n_live,).
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        Scalar C_v.
    """
    beta = jnp.asarray(beta)
    n_dead = dead_energies.shape[0]

    # Compute normalized weights
    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    log_w_live = calc_log_weights_live(n_dead, n_live, n_cull)
    log_w_live_arr = jnp.full(live_energies.shape[0], log_w_live)

    all_log_w = jnp.concatenate([log_w_dead, log_w_live_arr])
    all_E = jnp.concatenate([dead_energies, live_energies])

    # log unnormalized weight: log_w - beta * E
    log_unnorm = all_log_w - beta * all_E
    log_Z = logsumexp(log_unnorm)

    # Normalized weights
    log_norm_w = log_unnorm - log_Z
    w = jnp.exp(log_norm_w)

    mean_E = jnp.sum(w * all_E)
    mean_E2 = jnp.sum(w * all_E**2)

    return beta**2 * (mean_E2 - mean_E**2)


def expectation(
    observable_values: jnp.ndarray,
    beta: float | jnp.ndarray,
    dead_energies: jnp.ndarray,
    live_energies: jnp.ndarray,
    n_live: int | jnp.ndarray,
    n_cull: int = 1,
) -> jnp.ndarray:
    """Compute thermal expectation <O>(beta) for an observable.

    Args:
        observable_values: Observable values, shape (n_dead + n_live,).
        beta: Inverse temperature.
        dead_energies: Dead point energies, shape (n_dead,).
        live_energies: Live walker energies, shape (n_live,).
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        Scalar <O>.
    """
    beta = jnp.asarray(beta)
    n_dead = dead_energies.shape[0]

    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    log_w_live = calc_log_weights_live(n_dead, n_live, n_cull)
    log_w_live_arr = jnp.full(live_energies.shape[0], log_w_live)

    all_log_w = jnp.concatenate([log_w_dead, log_w_live_arr])
    all_E = jnp.concatenate([dead_energies, live_energies])

    log_unnorm = all_log_w - beta * all_E
    log_Z = logsumexp(log_unnorm)
    log_norm_w = log_unnorm - log_Z
    w = jnp.exp(log_norm_w)

    return jnp.sum(w * observable_values)


def free_energy(
    beta: float | jnp.ndarray,
    log_Z_beta: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Helmholtz free energy F = -log Z(beta) / beta.

    Args:
        beta: Inverse temperature.
        log_Z_beta: Log partition function at beta.

    Returns:
        Scalar F.
    """
    beta = jnp.asarray(beta)
    return -log_Z_beta / beta
