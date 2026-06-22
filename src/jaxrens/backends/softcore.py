"""SoftCoreBackend — wraps any EnergyBackend with a fixed repulsive Morse term.

MLIPs (MACE, NeuralIL, Nequix, …) generally have no defined behaviour for
close-contact configurations far outside their training distribution, where
the learned potential can become attractive or NaN. During nested sampling
at high pressure or with aggressive cell moves walkers can drift into such
configurations and irreversibly collapse atoms onto each other.

This wrapper adds a parameter-free repulsive Morse term

    phi(r) = d0 * exp(-2 * a0 * (r - b0))

multiplied by a smooth cutoff function that is exactly 1 below ``r_switch``
and exactly 0 above ``r_cut`` (default 1.25 Å). The contribution is summed
over all atom pairs using minimum-image distances when a periodic cell is
present (so close contacts across a boundary are counted), and raw
Cartesian distances for non-periodic systems. MIC is single-image — no
supercell expansion — which is exact as long as ``r_cut`` stays below half
the shortest cell vector.

Usage:
    base = MACEBackend(...)
    backend = SoftCoreBackend(base)                          # defaults
    H = EnsembleBackend(backend, pressure=P)                  # NPT stack

The wrapper satisfies the EnergyBackend protocol and forwards unknown
attributes to ``base`` via ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jaxrens.backends.base import BackendResult

DEFAULT_SOFTCORE_KWARGS: dict[str, float] = {
    "a0": 1.0,
    "b0": 3.0,
    "d0": 1.0,
    "r_core_cut": 1.25,
    "r_core_switch": 0.75,
}


def _aux_f(t: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(t > 0.0, jnp.exp(-1.0 / jnp.where(t > 0.0, t, 1.0)), 0.0)


def _aux_g(t: jnp.ndarray) -> jnp.ndarray:
    f_t = _aux_f(t)
    return f_t / (f_t + _aux_f(1.0 - t))


def _smooth_cutoff(
    r: jnp.ndarray, r_switch: float, r_cut: float,
) -> jnp.ndarray:
    """Smooth-bump partition-of-unity cutoff.

    Equals 1 for ``r <= r_switch``, 0 for ``r >= r_cut``, C^infty in between.
    Ported verbatim from ``neuralil.model.smooth_cutoff`` so the soft-core
    wrapper has no NeuralIL dependency.
    """
    r2 = r * r
    rs2 = r_switch * r_switch
    rc2 = r_cut * r_cut
    return 1.0 - _aux_g((r2 - rs2) / (rc2 - rs2))


def _softcore_energy(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    a0: float,
    b0: float,
    d0: float,
    r_core_switch: float,
    r_core_cut: float,
) -> jnp.ndarray:
    """Total soft-core Morse repulsion for one configuration.

    Sums ``phi(r_ij) * smooth_cutoff(r_ij)`` over all distinct pairs ``i < j``.
    Distances use the minimum-image convention when ``cell`` is a real
    (non-degenerate) lattice, so close contacts across a periodic boundary
    are counted; for a non-periodic system (``cell`` all zeros / singular)
    raw Cartesian distances are used. MIC is single-image ("111", O(N^2),
    no supercell expansion) — exact as long as ``r_core_cut`` is below half
    the shortest cell vector, which holds for the short soft-core range at
    any non-pathological cell.

    The diagonal is masked by pushing self-distances past the cutoff.

    Args:
        positions: ``(N, 3)`` atomic positions.
        cell: ``(3, 3)`` lattice matrix (rows are lattice vectors); zeros
            for a non-periodic system.
        a0, b0, d0: Fixed Morse parameters (steepness, equilibrium offset,
            prefactor).
        r_core_switch: Radius at which the cutoff starts ramping from 1.
        r_core_cut: Radius beyond which the contribution is exactly 0.

    Returns:
        Scalar total soft-core repulsion energy.
    """
    raw_delta = positions[:, None, :] - positions[None, :, :]

    # Minimum-image displacement when a real cell is present; raw
    # displacement otherwise. ``safe_cell`` keeps the inverse finite on the
    # non-periodic branch so the discarded MIC values never poison gradients.
    periodic = jnp.abs(jnp.linalg.det(cell)) > 1e-10
    safe_cell = jnp.where(periodic, cell, jnp.eye(3, dtype=cell.dtype))
    frac = positions @ jnp.linalg.inv(safe_cell)
    df = frac[:, None, :] - frac[None, :, :]
    df = df - jnp.round(df)
    mic_delta = df @ safe_cell

    delta = jnp.where(periodic, mic_delta, raw_delta)
    r = jnp.linalg.norm(delta, axis=-1)

    # Self-pairs and any spurious near-zeros: push past the cutoff so they
    # contribute exactly zero through the smooth_cutoff factor.
    diag_mask = r < 1e-10
    r_safe = jnp.where(diag_mask, 2.0 * r_core_cut, r)

    phi = d0 * jnp.exp(-2.0 * a0 * (r_safe - b0))
    cutoffs = _smooth_cutoff(r_safe, r_core_switch, r_core_cut)
    contributions = phi * cutoffs

    # 0.5 to undo the i<->j double-count of the full (N, N) sum.
    return 0.5 * jnp.sum(contributions)


class SoftCoreBackend:
    """Wraps any EnergyBackend with a fixed repulsive Morse soft core.

    Satisfies the EnergyBackend protocol. The soft-core term depends only
    on geometry (positions, cell), not on species or trainable parameters,
    so any backend (LJ, MACE, Nequix, NeuralIL, jaxmd, toy) can be wrapped.

    For NPT runs, stack with :class:`EnsembleBackend`:

        EnsembleBackend(SoftCoreBackend(base_backend), pressure=p)

    SoftCore sits closest to the bare backend (adds to ``U`` first); the
    ensemble correction (``+ P*V``) is then added on top.
    """

    def __init__(
        self,
        base: Any,
        a0: float = 1.0,
        b0: float = 3.0,
        d0: float = 1.0,
        r_core_cut: float = 1.25,
        r_core_switch: float = 0.75,
    ):
        self.base = base
        self.r_cutoff = base.r_cutoff
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.d0 = float(d0)
        self.r_core_cut = float(r_core_cut)
        self.r_core_switch = float(r_core_switch)

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Return the wrapped backend's result with ``E_core`` added to energy."""
        res = self.base(
            positions, species, cell, max_neighbors,
            ensemble_params=ensemble_params,
        )
        E_core = _softcore_energy(
            positions, cell,
            self.a0, self.b0, self.d0,
            self.r_core_switch, self.r_core_cut,
        )
        return res._replace(energy=res.energy + E_core)

    def __getattr__(self, name: str) -> Any:
        # Called only when normal lookup fails, so this does not shadow
        # the wrapper's own attributes. Lets resolver code read e.g.
        # ``wrapped.atomic_numbers`` transparently.
        return getattr(self.__dict__["base"], name)
