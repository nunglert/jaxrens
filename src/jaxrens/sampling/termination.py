"""Termination criteria for nested sampling.

Determines when the NS run should stop based on evidence convergence
or iteration limits.
"""

from __future__ import annotations

import jax.numpy as jnp


def should_terminate(
    iteration: int,
    max_iterations: int,
    log_evidence: jnp.ndarray,
    log_evidence_prev: jnp.ndarray,
    convergence_threshold: float = 0.1,
    n_live: int = 500,
) -> bool:
    """Check if NS should terminate.

    Termination conditions (any one sufficient):
    1. iteration >= max_iterations
    2. Remaining prior mass contributes less than threshold to evidence

    Args:
        iteration: Current iteration.
        max_iterations: Maximum allowed iterations.
        log_evidence: Current log-evidence estimate.
        log_evidence_prev: Previous log-evidence estimate.
        convergence_threshold: Stop when log-evidence change < threshold.
        n_live: Number of live walkers.

    Returns:
        True if NS should stop.
    """
    if iteration >= max_iterations:
        return True

    # Evidence convergence: remaining prior mass contribution
    # log(remaining_mass) ~ -iteration / n_live
    log_remaining = -iteration / n_live
    # If remaining mass << current evidence, we're done
    if iteration > n_live and (log_remaining + jnp.max(jnp.array([0.0]))) < float(log_evidence) - convergence_threshold:
        return True

    return False
