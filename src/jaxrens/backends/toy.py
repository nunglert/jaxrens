"""Toy energy backends for testing and benchmarking.

These are the known-answer problems used for evidence accuracy tests.
Each backend satisfies the EnergyBackend protocol.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jaxrens.backends.base import BackendResult


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
    ) -> BackendResult:
        energy = 0.5 * self.k * jnp.sum(positions**2)
        return BackendResult(energy=energy)


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
    ) -> BackendResult:
        x = positions[:, 0]
        yz = positions[:, 1:]
        energy = jnp.sum(self.a * (x**2 - self.b) ** 2) + 0.5 * jnp.sum(
            yz**2
        )
        return BackendResult(energy=energy)


class RENSToyBackend:
    """Two particles in a periodic 1-D box — the toy model from the RENS paper.

    Reproduces the model of Unglert, Pártay and Madsen, *Replica Exchange
    Nested Sampling*, J. Chem. Theory Comput. **21**, 7304 (2025), eqs 14-17.
    A pair interaction built from a repulsive core and an attractive well::

        E_rep(d)  =  eps_rep * exp(-h_rep * d**2)
        E_attr(d) = -eps_attr * exp(-0.5 * ((d - mu) / sigma)**2)
        E_toy(d)  =  E_rep(d) + E_attr(d)

    summed over pairs and their periodic images within a cutoff ``r_cut``::

        U(x_1, x_2; a) = 1/2 * sum_{i != j} sum_images E_toy(|x_i - x_j + n a|)

    Despite its simplicity the resulting enthalpy surface is genuinely
    multi-modal in ``(a, d)``, which is what makes it a useful demonstration
    of replica exchange: independent NS runs at different pressures get stuck
    in different basins, and the swaps are what rescue them.

    **Embedding.** jaxrens carries 3-D positions and a 3x3 cell, so the
    1-D system lives on the x-axis with a cell of ``diag(a, 1, 1)``.  Then
    ``V = det(cell) = a`` exactly, and the NPT enthalpy the ensemble wrapper
    already computes, ``H = U + P * V``, *is* eq 18 of the paper with no
    special-casing.  The ``y`` and ``z`` coordinates do not enter the energy;
    drive the system with the ``lattice_1d`` and ``distance_1d`` moves, which
    keep the cell diagonal and the particles on the axis.

    **Image sum.** ``n_images`` periodic images are summed on each side.  The
    count is static (it fixes the compiled shape) and the ``r_cut`` mask does
    the physics, so it only needs to be large enough that
    ``n_images * a_min >= r_cut`` for the smallest box the prior allows.

    Self-images (``i == j``, ``n != 0``) are **excluded**, following the
    ``i != j`` restriction in eq 17.
    """

    def __init__(
        self,
        eps_rep: float = 1.0,
        h_rep: float = 3.0,
        eps_attr: float = 1.0,
        mu: float = 1.0,
        sigma: float = 0.2,
        r_cut: float = 3.0,
        n_images: int = 8,
    ):
        self.eps_rep = eps_rep
        self.h_rep = h_rep
        self.eps_attr = eps_attr
        self.mu = mu
        self.sigma = sigma
        self.r_cut = r_cut
        self.n_images = int(n_images)
        self.r_cutoff = r_cut

    def pair_energy(self, d: jnp.ndarray) -> jnp.ndarray:
        """``E_toy`` for a separation ``d`` (eq 16). Public: the docs plot it."""
        rep = self.eps_rep * jnp.exp(-self.h_rep * d**2)
        attr = -self.eps_attr * jnp.exp(
            -0.5 * ((d - self.mu) / self.sigma) ** 2
        )
        return rep + attr

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        x = positions[:, 0]
        a = cell[0, 0]

        # All ordered pairs, then mask the diagonal: i != j, counted twice and
        # halved, exactly as eq 17 is written.
        dx = x[:, None] - x[None, :]
        off_diagonal = 1.0 - jnp.eye(x.shape[0])

        shifts = jnp.arange(-self.n_images, self.n_images + 1) * a
        d = jnp.abs(dx[:, :, None] + shifts[None, None, :])

        within = d <= self.r_cut
        pair = jnp.where(within, self.pair_energy(d), 0.0)
        energy = 0.5 * jnp.sum(pair * off_diagonal[:, :, None])

        return BackendResult(energy=energy)


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
    ) -> BackendResult:
        com = jnp.mean(positions, axis=0)
        diff = com[None, :] - self.centers
        dist_sq = jnp.sum(diff**2, axis=-1)
        log_components = (
            self.log_weights
            - 0.5 * dist_sq / self.sigma**2
            - 1.5 * jnp.log(2.0 * jnp.pi * self.sigma**2)
        )
        energy = -_logsumexp(log_components)
        return BackendResult(energy=energy)


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


def create_rens_toy(
    eps_rep: float = 1.0,
    h_rep: float = 3.0,
    eps_attr: float = 1.0,
    mu: float = 1.0,
    sigma: float = 0.2,
    r_cut: float = 3.0,
    n_images: int = 8,
) -> RENSToyBackend:
    """Create the RENS-paper 1-D toy backend."""
    return RENSToyBackend(
        eps_rep=eps_rep,
        h_rep=h_rep,
        eps_attr=eps_attr,
        mu=mu,
        sigma=sigma,
        r_cut=r_cut,
        n_images=n_images,
    )


def create_gaussian_mixture(
    centers: list[list[float]] | None = None,
    sigma: float = 0.5,
) -> GaussianMixtureBackend:
    """Create a Gaussian mixture potential backend."""
    return GaussianMixtureBackend(centers=centers, sigma=sigma)
