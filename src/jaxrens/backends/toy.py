"""Toy energy backends for testing and benchmarking.

These are the known-answer problems used for evidence accuracy tests.
Each function conforms to the EnergyFn protocol.
"""

from typing import Any

import jax.numpy as jnp

from jaxrens.types import Box, Params, Positions, Types


# ---------------------------------------------------------------------------
# Harmonic potential: E = 0.5 * k * sum(positions^2)
# ---------------------------------------------------------------------------


def create_harmonic(k: float = 1.0) -> tuple:
    """Create a harmonic potential.

    Known log-evidence for N atoms in 3D with prior volume V:
        log Z = -N * 3/2 * log(k / (2*pi)) + log(V)

    Args:
        k: Spring constant.

    Returns:
        (energy_fn, params) tuple.
    """
    params = {"k": jnp.array(k)}

    def energy_fn(
        params: Params,
        positions: Positions,
        types: Types,
        box: Box | None = None,
        **unused_kwargs: Any,
    ) -> jnp.ndarray:
        return 0.5 * params["k"] * jnp.sum(positions**2)

    return energy_fn, params


# ---------------------------------------------------------------------------
# Double-well potential: E = a*(x^2 - b)^2 summed over atoms
# ---------------------------------------------------------------------------


def create_double_well(a: float = 1.0, b: float = 1.0) -> tuple:
    """Create a double-well potential along x-axis.

    E = sum_i a * (x_i^2 - b)^2 + 0.5 * (y_i^2 + z_i^2)

    Useful for testing multi-modal sampling.

    Args:
        a: Barrier height parameter.
        b: Well separation parameter.

    Returns:
        (energy_fn, params) tuple.
    """
    params = {"a": jnp.array(a), "b": jnp.array(b)}

    def energy_fn(
        params: Params,
        positions: Positions,
        types: Types,
        box: Box | None = None,
        **unused_kwargs: Any,
    ) -> jnp.ndarray:
        x = positions[:, 0]
        yz = positions[:, 1:]
        return jnp.sum(params["a"] * (x**2 - params["b"]) ** 2) + 0.5 * jnp.sum(
            yz**2
        )

    return energy_fn, params


# ---------------------------------------------------------------------------
# Gaussian mixture: E = -log(sum_k w_k * N(positions; mu_k, sigma_k))
# ---------------------------------------------------------------------------


def create_gaussian_mixture(
    centers: list[list[float]] | None = None,
    sigma: float = 0.5,
) -> tuple:
    """Create a Gaussian mixture energy (negative log-likelihood).

    E = -log(sum_k w_k * exp(-0.5 * ||pos - mu_k||^2 / sigma^2))

    For a single atom in 3D, log-evidence = log(sum_k w_k) (when prior
    covers all modes). Useful for testing multi-modal evidence calculation.

    Args:
        centers: List of 3D center coordinates. Default: two centers at
            [-1,0,0] and [1,0,0].
        sigma: Width of each Gaussian component.

    Returns:
        (energy_fn, params) tuple.
    """
    if centers is None:
        centers = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    params = {
        "centers": jnp.array(centers),
        "sigma": jnp.array(sigma),
        "log_weights": jnp.zeros(len(centers)),  # equal weights
    }

    def energy_fn(
        params: Params,
        positions: Positions,
        types: Types,
        box: Box | None = None,
        **unused_kwargs: Any,
    ) -> jnp.ndarray:
        # positions: (n_atoms, 3), centers: (n_centers, 3)
        # For simplicity, use center-of-mass of all atoms
        com = jnp.mean(positions, axis=0)  # (3,)
        diff = com[None, :] - params["centers"]  # (n_centers, 3)
        dist_sq = jnp.sum(diff**2, axis=-1)  # (n_centers,)
        log_components = (
            params["log_weights"]
            - 0.5 * dist_sq / params["sigma"] ** 2
            - 1.5 * jnp.log(2.0 * jnp.pi * params["sigma"] ** 2)
        )
        # E = -log p(x) = -logsumexp(log_components)
        return -jax_logsumexp(log_components)

    return energy_fn, params


def jax_logsumexp(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable logsumexp."""
    x_max = jnp.max(x)
    return x_max + jnp.log(jnp.sum(jnp.exp(x - x_max)))
