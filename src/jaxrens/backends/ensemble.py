"""EnsembleBackend — wraps any EnergyBackend with ensemble corrections.

Adds thermodynamic potential terms (PV, μN) to the raw backend energy.
The wrapped backend satisfies the same EnergyBackend protocol.

Usage:
    base = HarmonicBackend(k=1.0)
    backend = EnsembleBackend(base, pressure=0.01)
    H, count, overflow = backend(positions, species, cell, max_neighbors)
    # H = U + P*V

For per-run vmap with different pressures, pass ensemble_params:
    backend(pos, species, cell, mn, ensemble_params={"pressure": 0.02})
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jaxrens.utils.cell import get_volume


class EnsembleBackend:
    """Wraps any EnergyBackend with ensemble corrections (PV, μN).

    Satisfies EnergyBackend protocol. Can be used anywhere a plain
    backend is expected.

    Ensemble corrections:
        NVT:  H = U                    (no correction, don't wrap)
        NPT:  H = U + P*V
        μPT:  H = U + P*V - μ·N

    For per-run vmap: pass ensemble_params kwarg to override the
    closured defaults. This allows different runs to have different
    pressures/chemical potentials.
    """

    def __init__(
        self,
        base: Any,
        pressure: float = 0.0,
        chemical_potentials: jnp.ndarray | None = None,
    ):
        self.base = base
        self.r_cutoff = base.r_cutoff
        self.pressure = pressure
        self.chemical_potentials = chemical_potentials

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        """Compute ensemble-corrected energy.

        Calls the base backend for the raw potential U, then adds
        ensemble terms (PV, μN).
        """
        U, count, overflow = self.base(
            positions, species, cell, max_neighbors,
        )

        # Use per-run params if provided, else closured defaults
        pressure = self.pressure
        mu = self.chemical_potentials
        if ensemble_params is not None:
            pressure = ensemble_params.get("pressure", pressure)
            mu = ensemble_params.get("mu", mu)

        H = U + pressure * get_volume(cell)

        if mu is not None:
            n_species = mu.shape[0]
            if n_species > 0:
                counts = jnp.zeros(n_species, dtype=species.dtype)
                counts = counts.at[species].add(1)
                H = H - jnp.dot(mu, counts.astype(jnp.float32))

        return H, count, overflow

    def __getattr__(self, name: str) -> Any:
        # Called only when normal lookup fails, so this does not shadow
        # ``base`` / ``pressure`` / ``chemical_potentials`` / ``r_cutoff``.
        # Lets resolver code read e.g. ``wrapped.atomic_numbers`` through
        # the wrapper without special-casing.
        return getattr(self.__dict__["base"], name)


def make_ensemble_params(
    pressure: float = 0.0,
    chemical_potentials: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Create ensemble_params dict for MCState.

    Returns a dict suitable for storing on MCState.ensemble_params
    and passing to EnsembleBackend via the ensemble_params kwarg.
    """
    params: dict[str, jnp.ndarray] = {"pressure": jnp.asarray(pressure)}
    if chemical_potentials is not None:
        params["mu"] = jnp.asarray(chemical_potentials)
    return params
