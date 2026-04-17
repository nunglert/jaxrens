"""Toy energy backends for testing and benchmarking.

These are the known-answer problems used for evidence accuracy tests.
Each backend satisfies the EnergyBackend protocol.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


class HarmonicBackend:
    """Harmonic potential: E = 0.5 * k * sum(positions^2).

    Known log-evidence for N atoms in 3D with prior volume V:
        log Z = -N * 3/2 * log(k / (2*pi)) + log(V)
    """

    def __init__(self, k: float = 1.0):
        self.k = k
        self.r_cutoff = 0.0

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        energy = 0.5 * self.k * jnp.sum(positions**2)
        return energy, 0, False


class DoubleWellBackend:
    """Double-well potential along x-axis.

    E = sum_i a * (x_i^2 - b)^2 + 0.5 * (y_i^2 + z_i^2)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = a
        self.b = b
        self.r_cutoff = 0.0

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        x = positions[:, 0]
        yz = positions[:, 1:]
        energy = jnp.sum(self.a * (x**2 - self.b) ** 2) + 0.5 * jnp.sum(yz**2)
        return energy, 0, False


class GaussianMixtureBackend:
    """Gaussian mixture energy (negative log-likelihood).

    E = -log(sum_k w_k * exp(-0.5 * ||pos - mu_k||^2 / sigma^2))
    """

    def __init__(
        self,
        centers: list[list[float]] | None = None,
        sigma: float = 0.5,
    ):
        if centers is None:
            centers = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        self.centers = jnp.array(centers)
        self.sigma = sigma
        self.log_weights = jnp.zeros(len(centers))
        self.r_cutoff = 0.0

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        com = jnp.mean(positions, axis=0)
        diff = com[None, :] - self.centers
        dist_sq = jnp.sum(diff**2, axis=-1)
        log_components = (
            self.log_weights
            - 0.5 * dist_sq / self.sigma**2
            - 1.5 * jnp.log(2.0 * jnp.pi * self.sigma**2)
        )
        energy = -_logsumexp(log_components)
        return energy, 0, False


def _logsumexp(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable logsumexp."""
    x_max = jnp.max(x)
    return x_max + jnp.log(jnp.sum(jnp.exp(x - x_max)))


# ---------------------------------------------------------------------------
# Factory functions (backward compatibility with loader.py)
# ---------------------------------------------------------------------------


def create_harmonic(k: float = 1.0) -> HarmonicBackend:
    """Create a harmonic potential backend."""
    return HarmonicBackend(k=k)


def create_double_well(a: float = 1.0, b: float = 1.0) -> DoubleWellBackend:
    """Create a double-well potential backend."""
    return DoubleWellBackend(a=a, b=b)


def create_gaussian_mixture(
    centers: list[list[float]] | None = None,
    sigma: float = 0.5,
) -> GaussianMixtureBackend:
    """Create a Gaussian mixture potential backend."""
    return GaussianMixtureBackend(centers=centers, sigma=sigma)
