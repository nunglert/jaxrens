"""Standalone demo: 2-run pressure-RENS with LJ-8 NPT and inter-RE.

Two parallel NS runs at P=0.01 and P=0.1 eV/Å³. Pressure-RENS swaps
fire every iteration via InterREConfig(flavor="pressure", every=1).

Usage (from the repo root):
    cd jaxrens && python experiments/examples/lj8_npt/run_inter_re.py 2>&1 | tee /tmp/run_inter_re.log
"""

from __future__ import annotations

import logging
import sys

import jax
import jax.numpy as jnp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

from jaxrens.backends.loader import load_backend
from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.cli.monitor import ProgressCallback
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves.galilean import build_kernel as galilean_kernel
from jaxrens.sampling.moves.volume import build_kernel as volume_kernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns_parallel
from jaxrens.sampling.termination import IterationTermination
from jaxrens.state.config import InterREConfig

logger = logging.getLogger("run_inter_re")


def main():
    n_atoms = 8
    n_runs = 2
    n_walkers = 32
    pressures = [0.01, 0.1]   # eV/Å³
    max_iterations = 100

    base_backend = load_backend("lj", cutoff=2.5)
    ensemble_backends = [EnsembleBackend(base_backend, pressure=p) for p in pressures]
    ensemble_params_per_run = [{"pressure": p} for p in pressures]

    # Build MWG with galilean + volume moves.
    descriptors = [
        MoveKernel(
            "galilean", galilean_kernel,
            kernel_kwargs={"n_reflect": 5},
            extra_state_fields={"direction": (jnp.ndarray, lambda pos, types: jnp.zeros_like(pos))},
            step_size=0.1, step_size_max=0.5,
            min_rate=0.3, max_rate=0.6,
            weight=4.0,
        ),
        MoveKernel(
            "volume", volume_kernel,
            kernel_kwargs={"n_atoms": n_atoms},
            step_size=0.05, step_size_max=0.2,
            min_rate=0.3, max_rate=0.6,
            weight=1.0,
        ),
    ]
    init_fn, step_fn, _ = build_mwg(base_backend, descriptors)

    # Initialize positions: simple cubic packing for 8 atoms.
    key = jax.random.key(42)
    rng_keys = jax.random.split(key, n_runs)
    pos_key, _ = jax.random.split(jax.random.key(99))

    # Cubic cell: ~2.2 sigma side for N=8 at LJ equilibrium density.
    box_side = float((n_atoms * 1.1) ** (1 / 3))
    cell = box_side * jnp.eye(3)
    spacing = box_side / 2.0

    n_side = 2  # 2^3 = 8 atoms
    grid = jnp.array([
        [i * spacing, j * spacing, k * spacing]
        for i in range(n_side) for j in range(n_side) for k in range(n_side)
    ], dtype=jnp.float32)  # (8, 3)

    # Add per-walker noise so walkers differ.
    noise = jax.random.normal(pos_key, (n_runs, n_walkers, n_atoms, 3)) * 0.08
    positions = grid[None, None, :, :] + noise
    positions = positions % box_side   # periodic wrap
    cells = jnp.broadcast_to(cell[None, None, :, :], (n_runs, n_walkers, 3, 3))

    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    # Evaluate initial energies using the per-run ensemble backend.
    energies_list = []
    for r in range(n_runs):
        eb = ensemble_backends[r]
        run_energies = jax.vmap(
            lambda pos, cell: eb(pos, types, cell, 0)[0]
        )(positions[r], cells[r])
        energies_list.append(run_energies)
    energies = jnp.stack(energies_list)

    inter_re_cfg = InterREConfig(flavor="pressure", every=1, n_swap_cycles=1)

    logger.info(
        "Starting 2-run pressure-RENS LJ-8 NPT: n_atoms=%d, n_walkers=%d, "
        "pressures=%s eV/Å³, max_iter=%d, inter_re.every=1",
        n_atoms, n_walkers, pressures, max_iterations,
    )
    logger.info("box_side=%.3f sigma", box_side)

    result = run_ns_parallel(
        positions=positions,
        types=types,
        energies=energies,
        cells=cells,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        n_walkers=n_walkers,
        max_iterations=max_iterations,
        n_mcmc_steps=10,
        termination_criteria=[IterationTermination(max_iterations)],
        ensemble_params_per_run=ensemble_params_per_run,
        move_descriptors=descriptors,
        inter_re_config=inter_re_cfg,
        callbacks=[ProgressCallback(info_interval=20)],
    )

    logger.info("Run complete:")
    for r in range(n_runs):
        logger.info(
            "  Run %d (P=%.3f eV/Å³): n_dead=%d  log_Z=%.4f",
            r, pressures[r], int(result["n_dead"][r]), float(result["log_evidence"][r]),
        )

    dz = abs(float(result["log_evidence"][0]) - float(result["log_evidence"][1]))
    logger.info("  |log_Z[0] - log_Z[1]| = %.4f (expected > 0 for different pressures)", dz)
    return result


if __name__ == "__main__":
    main()
