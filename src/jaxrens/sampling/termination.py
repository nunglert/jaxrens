"""Termination criteria for nested sampling.

Pluggable termination system: each criterion implements the
TerminationCriterion protocol. The outer NS loop calls check()
after each iteration and terminates when any criterion is satisfied.

Available criteria:
- IterationTermination: stop after max_iterations
- PriorMassTermination: stop when remaining prior mass is negligible
- TempTermination: stop when Z(T_target) has converged (Baldock et al. 2017)
- EnergyTermination: stop when E_max < target energy
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Boltzmann constant in eV/K
KB_EV = 8.6173324e-5


class TerminationCriterion(Protocol):
    """Protocol for termination criteria.

    Each criterion tracks state across iterations and reports
    whether the NS run should stop.
    """

    def check(self, iteration: int, emax: float) -> bool:
        """Update internal state and return True if termination condition met.

        Args:
            iteration: Current NS iteration (0-indexed).
            emax: Energy (or enthalpy) of the dead point at this iteration.

        Returns:
            True if criterion is satisfied and NS should stop.
        """
        ...

    def message(self) -> str:
        """Human-readable description of why termination triggered."""
        ...


class IterationTermination:
    """Terminate after a fixed number of iterations."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def check(self, iteration: int, emax: float) -> bool:
        return iteration >= self.max_iterations - 1

    def message(self) -> str:
        return f"Reached max iterations ({self.max_iterations})"


class PriorMassTermination:
    """Terminate when remaining prior mass contributes negligibly to evidence.

    At iteration i with n_live walkers, remaining log prior mass is
    approximately -i / n_live. Stop when this is much smaller than
    the current log evidence.
    """

    def __init__(self, n_live: int, threshold: float = 0.1):
        self.n_live = n_live
        self.threshold = threshold
        self._log_evidence = -jnp.inf

    def update_evidence(self, log_evidence: float) -> None:
        """Update the current evidence estimate (called by outer loop)."""
        self._log_evidence = log_evidence

    def check(self, iteration: int, emax: float) -> bool:
        if iteration <= self.n_live:
            return False
        log_remaining = -iteration / self.n_live
        converged = log_remaining < float(self._log_evidence) - self.threshold
        if converged:
            logger.debug(
                "PriorMassTermination: log_remaining=%.4f < log_Z=%.4f - %.2f",
                log_remaining, float(self._log_evidence), self.threshold,
            )
        return converged

    def message(self) -> str:
        return "Prior mass negligible compared to evidence"


class TempTermination:
    """Temperature-based termination following Baldock et al. (2017).

    Monitors the partition function Z(T) at a target temperature.
    Terminates when new contributions to Z(T) are more than 10 orders
    of magnitude below the maximum contribution seen so far, meaning
    Z(T_target) has converged.

    The partition function contribution at iteration i is:
        log_Z_term_i = log_w_i - beta * E_max_i

    where log_w_i is the log prior-mass weight and beta = 1/(kB * T).
    """

    def __init__(
        self,
        n_walkers: int,
        target_temp: float,
        n_cull: int = 1,
        threshold: float = 10.0,
    ):
        """
        Args:
            n_walkers: Number of live walkers.
            target_temp: Target temperature in Kelvin.
            n_cull: Number of walkers culled per iteration.
            threshold: Number of orders of magnitude below max term
                for convergence (default 10.0).
        """
        self.n_walkers = n_walkers
        self.n_cull = n_cull
        self.beta = 1.0 / (KB_EV * target_temp)
        self.target_temp = target_temp
        self.threshold = threshold

        # Precompute shrinkage factor
        n = n_walkers
        k = n_cull
        # log_t = log((N - k) / (N + 1 - k))
        self._log_t = math.log(n - k) - math.log(n + 1 - k)
        # log_shell = log(1 - t) = log(1/(N+1-k)) = -log(N+1-k)
        self._log_shell = -math.log(n + 1 - k)

        # Running state
        self._log_Z_term_max = -math.inf
        self._log_Z_term_last = -math.inf
        self._fulfilled = False

    def check(self, iteration: int, emax: float) -> bool:
        # log weight at this iteration: log_X_i + log_shell
        # where log_X_i = i * log_t
        log_w = iteration * self._log_t + self._log_shell

        # Partition function contribution: w_i * exp(-beta * E_i)
        # In log space: log_w - beta * E
        log_Z_term = log_w - self.beta * float(emax)

        self._log_Z_term_last = log_Z_term
        if log_Z_term > self._log_Z_term_max:
            self._log_Z_term_max = log_Z_term

        # Converged when last term is threshold orders of magnitude below max
        if self._log_Z_term_max > -math.inf:
            self._fulfilled = log_Z_term < (self._log_Z_term_max - self.threshold)

        if self._fulfilled:
            logger.debug(
                "TempTermination: T=%.1fK, log_Z_term=%.4f, max=%.4f, gap=%.1f > %.1f",
                self.target_temp, log_Z_term, self._log_Z_term_max,
                self._log_Z_term_max - log_Z_term, self.threshold,
            )

        return self._fulfilled

    @property
    def log_Z_term_max(self) -> float:
        return self._log_Z_term_max

    @property
    def log_Z_term_last(self) -> float:
        return self._log_Z_term_last

    def message(self) -> str:
        diff = self._log_Z_term_max - self._log_Z_term_last
        return (
            f"Z(T={self.target_temp:.1f}K) converged: "
            f"last term {diff:.1f} orders below max"
        )


class EnergyTermination:
    """Terminate when the dead-point energy falls below a target.

    Useful when only a specific energy range is of interest.
    """

    def __init__(self, min_energy: float):
        """
        Args:
            min_energy: Terminate when emax < min_energy.
        """
        self.min_energy = min_energy
        self._fulfilled = False

    def check(self, iteration: int, emax: float) -> bool:
        self._fulfilled = float(emax) < self.min_energy
        return self._fulfilled

    def message(self) -> str:
        return f"E_max below target energy ({self.min_energy})"


def check_any(
    criteria: list[TerminationCriterion],
    iteration: int,
    emax: float,
) -> tuple[bool, str]:
    """Check all termination criteria and return on first satisfied.

    Args:
        criteria: List of termination criteria.
        iteration: Current NS iteration.
        emax: Energy/enthalpy of the dead point.

    Returns:
        (should_terminate, message) tuple.
    """
    for criterion in criteria:
        if criterion.check(iteration, emax):
            return True, criterion.message()
    return False, ""
