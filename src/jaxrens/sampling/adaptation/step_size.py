"""Step size adaptation using dual averaging.

Wraps any move kernel with automatic step size tuning.
Following BlackJAX's window_adaptation pattern:
- Warmup phase: adapt step_size toward target acceptance rate
- Production phase: use the tuned step_size as a fixed constant

The adapted kernel has the same interface as the base kernel,
so it composes transparently with vmap/pmap/lax.scan.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class AdaptationState(NamedTuple):
    """State for dual-averaging step size adaptation."""

    log_step_size: jnp.ndarray  # log of current step size
    log_step_size_avg: jnp.ndarray  # averaged log step size (Robbins-Monro)
    step: jnp.ndarray  # adaptation step counter
    mu: jnp.ndarray  # log(10 * initial_step_size) — shrinkage target
    h_avg: jnp.ndarray  # running average of acceptance statistic


def init_adaptation(
    initial_step_size: float = 0.1,
    target_acceptance: float = 0.5,
) -> AdaptationState:
    """Initialize adaptation state.

    Args:
        initial_step_size: Starting step size.
        target_acceptance: Target acceptance rate (typically 0.2-0.5 for NS).

    Returns:
        Initial AdaptationState.
    """
    return AdaptationState(
        log_step_size=jnp.log(jnp.array(initial_step_size)),
        log_step_size_avg=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
        mu=jnp.log(10.0 * jnp.array(initial_step_size)),
        h_avg=jnp.array(0.0),
    )


def dual_averaging_update(
    adapt_state: AdaptationState,
    accepted: jnp.ndarray,
    target_acceptance: float = 0.5,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> AdaptationState:
    """One step of Nesterov dual averaging for step size adaptation.

    Based on Algorithm 5 from Hoffman & Gelman (2014) "The No-U-Turn Sampler".

    Args:
        adapt_state: Current adaptation state.
        accepted: Whether the last proposal was accepted (bool or 0/1).
        target_acceptance: Target acceptance rate.
        gamma: Controls amount of shrinkage toward mu.
        t0: Stabilization offset for early iterations.
        kappa: Power for step size schedule (0.5 < kappa <= 1).

    Returns:
        Updated AdaptationState.
    """
    m = adapt_state.step + 1
    w = 1.0 / (m + t0)

    # Update running average of acceptance statistic
    h_avg = (1.0 - w) * adapt_state.h_avg + w * (target_acceptance - accepted)

    # Update log step size
    log_step_size = adapt_state.mu - jnp.sqrt(m) / gamma * h_avg

    # Update averaged log step size (Robbins-Monro)
    mk = m ** (-kappa)
    log_step_size_avg = mk * log_step_size + (1.0 - mk) * adapt_state.log_step_size_avg

    return AdaptationState(
        log_step_size=log_step_size,
        log_step_size_avg=log_step_size_avg,
        step=m,
        mu=adapt_state.mu,
        h_avg=h_avg,
    )


def get_step_size(adapt_state: AdaptationState, use_averaged: bool = False) -> jnp.ndarray:
    """Extract current step size from adaptation state.

    Args:
        adapt_state: Current adaptation state.
        use_averaged: If True, return the Robbins-Monro averaged step size
            (more stable, use for production). If False, return the latest
            step size (use during warmup).

    Returns:
        Step size as a scalar.
    """
    if use_averaged:
        return jnp.exp(adapt_state.log_step_size_avg)
    return jnp.exp(adapt_state.log_step_size)
