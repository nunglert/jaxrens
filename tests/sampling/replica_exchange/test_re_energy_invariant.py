"""Stored energies stay consistent with each run's ensemble after swaps.

After any number of swap cycles, every stored ``energy[r, k]`` must equal a
fresh backend evaluation of that walker under run r's ensemble params.  This
is the invariant the old semi-grand swap broke (energies written with
the partner's μ while the run kept its own), checked here for every kernel
against a real ``Hamiltonian(model, [EnsembleTerm()])``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.ensemble import EnsembleTerm
from jaxrens.backends.hamiltonian import Hamiltonian
from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.moves.replica_exchange import (
    EnsembleSwap,
    SwapWalker,
    XRENSSwap,
    replica_exchange_step,
)

N_RUNS, N_WALKERS, N_ATOMS, N_SPECIES = 4, 6, 6, 2


def _evaluate(backend, walkers: SwapWalker, run_params):
    """Backend energy of every walker under its own run's params."""

    def per_run(pos, typ, cell, params):
        return jax.vmap(
            lambda p, t, c: backend(p, t, c, 0, ensemble_params=params).energy
        )(pos, typ, cell)

    return jax.vmap(per_run)(
        walkers.positions, walkers.types, walkers.cell, run_params
    )


def _population(seed):
    k_pos, k_typ, k_cell = jax.random.split(jax.random.key(seed), 3)
    positions = jax.random.normal(k_pos, (N_RUNS, N_WALKERS, N_ATOMS, 3))
    types = jax.random.randint(
        k_typ, (N_RUNS, N_WALKERS, N_ATOMS), 0, N_SPECIES
    )
    sides = jax.random.uniform(
        k_cell, (N_RUNS, N_WALKERS), minval=2.0, maxval=3.0
    )
    cells = sides[..., None, None] * jnp.eye(3)
    return positions, types, cells


@pytest.mark.parametrize(
    "flavor", ["pressure", "semi_grand", "semi_grand_mupt", "xrens"]
)
def test_stored_energy_matches_backend_after_swaps(flavor):
    backend = Hamiltonian(create_harmonic(k=1.0), [EnsembleTerm()])
    positions, types, cells = _population(seed=0)

    run_params = {}
    kernel = EnsembleSwap()
    if flavor in ("pressure", "semi_grand_mupt", "xrens"):
        run_params["pressure"] = jnp.array([0.01, 0.05, 0.2, 0.5])
    if flavor in ("semi_grand", "semi_grand_mupt"):
        run_params["chemical_potentials"] = jnp.array(
            [[0.0, 0.0], [0.3, -0.2], [-0.5, 0.4], [1.0, 0.5]]
        )
        kernel = EnsembleSwap(required_params=("chemical_potentials",))
    if flavor == "xrens":
        run_params["target_composition"] = jnp.array(
            [[6, 0], [4, 2], [3, 3], [1, 5]], dtype=jnp.int32
        )
        kernel = XRENSSwap(n_species=N_SPECIES)
        # Start every run at its own target composition.
        n_ones = run_params["target_composition"][:, 1]
        per_run = jnp.arange(N_ATOMS)[None, :] >= N_ATOMS - n_ones[:, None]
        types = jnp.broadcast_to(
            per_run.astype(jnp.int32)[:, None, :], types.shape
        )

    walkers = SwapWalker(
        positions, types, jnp.zeros((N_RUNS, N_WALKERS)), cells
    )
    backend_params = {
        k: v for k, v in run_params.items() if k != "target_composition"
    }
    walkers = walkers._replace(
        energy=_evaluate(backend, walkers, backend_params)
    )

    new, info = replica_exchange_step(
        jax.random.key(1),
        walkers,
        jnp.full((N_RUNS,), 1e6),
        run_params,
        kernel,
        backend=backend,
        n_swap_cycles=4,
    )

    assert int(info["n_accepted"]) == int(info["n_attempted"]) > 0
    assert not jnp.array_equal(new.positions, walkers.positions)
    expected = _evaluate(backend, new, backend_params)
    assert jnp.allclose(new.energy, expected, rtol=1e-5, atol=1e-4)
