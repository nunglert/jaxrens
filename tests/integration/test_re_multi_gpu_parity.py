"""Check that inter-replica exchange gives the same result on any GPU layout.

On the ``PmapVmapRuns`` path, ``InterREManager`` all-gathers every
device's replicas, swaps on the full population with the same RNG on every
device, and re-shards with ``axis_index``. With one device the all-gather
does nothing, so only a real multi-device run can catch a wrong gather
axis, swap decisions that differ between devices, or a wrong re-shard
slice.

Each RE flavor (``pressure``, ``semi_grand``, ``xrens``) runs twice from
the same per-replica seeds and initial walkers: once as
``(n_gpu=1, n_per_gpu=R)`` and once as ``(n_gpu=N, n_per_gpu=R/N)``. The
per-replica ``log_Z`` must agree within a small float32 tolerance, and the
per-iteration swap attempt and accept counts must match exactly. The
backend is a two-species LJ (per-species ``epsilon`` and ``sigma``), so
XRENS sees energies that depend on composition. The test is marked
``multi_gpu`` but not ``heavy``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.lj import create_lj
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns_multi_gpu
from jaxrens.sampling.termination import IterationTermination
from jaxrens.state.config import InterREConfig

from . import multi_gpu_n_devices

_BASE_SEED = 20260508
_N_TOTAL = 4  # number of replicas (must divide both 1 and N_GPU choices)
_N_WALKERS = 8
_N_ATOMS = 4
_CELL_SIDE = 4.0
_MAX_ITERS = 8
_N_MCMC = 3
_INITIAL_STEP_SIZE = 0.15

# Two species, so XRENS swaps change the energy.
_EPS_TABLE = (1.0, 0.8)
_SIG_TABLE = (1.0, 1.1)
_CUTOFF = 2.5

# Per-replica composition targets (must each sum to _N_ATOMS).
_COMP_TARGETS = ((4, 0), (3, 1), (2, 2), (1, 3))


def _initial_types_for_composition(comp: tuple[int, ...]) -> jnp.ndarray:
    parts = [jnp.full(c, s, dtype=jnp.int32) for s, c in enumerate(comp)]
    return jnp.concatenate(parts)


def _build_problem(flavor: str):
    """Build the per-replica inputs for ``run_ns_multi_gpu``.

    All randomness is seeded, so two calls with the same flavor return
    identical inputs.
    """
    backend = create_lj(epsilon=_EPS_TABLE, sigma=_SIG_TABLE, cutoff=_CUTOFF)

    descriptors = [
        MoveKernel(
            "rw",
            random_walk.build_kernel,
            step_size=_INITIAL_STEP_SIZE,
            step_size_max=1.0,
        )
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    base_key = jax.random.key(_BASE_SEED)
    pos_key, types_key, run_keys_key = jax.random.split(base_key, 3)

    # Flat (n_total,) keys, so replica i gets the same key for any (G, P).
    rng_keys_flat = jax.random.split(run_keys_key, _N_TOTAL)

    # Keep atoms 0.5 away from the cell faces.
    positions = jax.random.uniform(
        pos_key,
        (_N_TOTAL, _N_WALKERS, _N_ATOMS, 3),
        minval=0.5,
        maxval=_CELL_SIDE - 0.5,
    )

    # Pressure RE needs the cell volume.
    cells = jnp.broadcast_to(
        _CELL_SIDE * jnp.eye(3),
        (_N_TOTAL, _N_WALKERS, 3, 3),
    )

    # Each replica starts at its XRENS composition target; the other flavors
    # only carry the types along.
    types_per_replica = jnp.stack(
        [_initial_types_for_composition(c) for c in _COMP_TARGETS]
    )  # (n_total, n_atoms)

    def _energy_one(pos_aw3, types_a):
        return backend(pos_aw3, types_a, _CELL_SIDE * jnp.eye(3), 0)[0]

    energies = jax.vmap(
        lambda pos_kaw3, types_a: jax.vmap(lambda p: _energy_one(p, types_a))(
            pos_kaw3
        )
    )(positions, types_per_replica)

    if flavor == "pressure":
        inter_re_cfg = InterREConfig(
            flavor="pressure",
            re_interval=1,
            n_swap_cycles=1,
        )
        ensemble_params_per_run = [
            {"pressure": 0.0},
            {"pressure": 0.05},
            {"pressure": 0.1},
            {"pressure": 0.15},
        ]
    elif flavor == "semi_grand":
        inter_re_cfg = InterREConfig(
            flavor="semi_grand",
            re_interval=1,
            n_swap_cycles=1,
            chemical_potentials=(
                (0.0, 0.0),
                (0.5, 0.0),
                (0.0, 0.5),
                (0.3, 0.3),
            ),
        )
        ensemble_params_per_run = None  # injected by run_ns_multi_gpu
    elif flavor == "xrens":
        inter_re_cfg = InterREConfig(
            flavor="xrens",
            re_interval=1,
            n_swap_cycles=1,
            composition_targets=_COMP_TARGETS,
        )
        ensemble_params_per_run = None  # injected by run_ns_multi_gpu
    else:
        raise ValueError(f"unknown flavor {flavor!r}")

    return {
        "positions": positions,
        "types": types_per_replica,
        "energies": energies,
        "cells": cells,
        "rng_keys": rng_keys_flat,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "backend": backend,
        "inter_re_cfg": inter_re_cfg,
        "ensemble_params_per_run": ensemble_params_per_run,
        "move_descriptors": descriptors,
    }


class _SwapStatsCollector:
    """Records per-iteration ``inter_re_stats`` for offline comparison."""

    def __init__(self):
        self.attempted: list[int] = []
        self.accepted: list[int] = []

    def on_iteration(self, iteration, ns_state, info):
        s = info.get("inter_re_stats")
        if s is None:
            return
        self.attempted.append(int(s["n_swap_pairs_attempted"]))
        self.accepted.append(int(s["n_swap_pairs_accepted"]))


def _run(flavor: str, n_gpu: int) -> dict[str, Any]:
    """Drive ``run_ns_multi_gpu`` once for a given device topology."""
    assert _N_TOTAL % n_gpu == 0
    n_per_gpu = _N_TOTAL // n_gpu

    prob = _build_problem(flavor)
    collector = _SwapStatsCollector()

    result = run_ns_multi_gpu(
        positions=prob["positions"],
        types=prob["types"],
        energies=prob["energies"],
        cells=prob["cells"],
        init_fn=prob["init_fn"],
        step_fn=prob["step_fn"],
        rng_keys=prob["rng_keys"],
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
        n_walkers=_N_WALKERS,
        max_iterations=_MAX_ITERS,
        n_mcmc_steps=_N_MCMC,
        initial_step_size=_INITIAL_STEP_SIZE,
        termination_criteria=[IterationTermination(_MAX_ITERS)],
        ensemble_params_per_run=prob["ensemble_params_per_run"],
        move_descriptors=prob["move_descriptors"],
        inter_re_config=prob["inter_re_cfg"],
        backend=prob["backend"],
        callbacks=[collector],
    )

    log_z = np.asarray(result["log_evidence"]).reshape(-1)  # (G*P,) → (R,)
    n_dead = np.asarray(result["n_dead"]).reshape(-1)
    return {
        "log_evidence": log_z,
        "n_dead": n_dead,
        "swap_attempted": tuple(collector.attempted),
        "swap_accepted": tuple(collector.accepted),
        "result": result,
    }


# log_Z comes from each replica's own live set with no cross-device
# reduction, so the two layouts should agree almost bit for bit. The
# tolerance absorbs float32 ordering differences in the pmap collectives.
_LOG_Z_ATOL = 1e-3
_LOG_Z_RTOL = 1e-3


@pytest.mark.multi_gpu
@pytest.mark.parametrize("flavor", ["pressure", "semi_grand", "xrens"])
def test_re_pmap_topology_parity(flavor: str) -> None:
    """Check that one device and N devices give the same log_Z and swaps."""
    n_devices = multi_gpu_n_devices()
    if _N_TOTAL % n_devices != 0:
        pytest.fail(
            f"test sized for n_total={_N_TOTAL} replicas; n_devices={n_devices} "
            f"does not divide it.  Adjust _N_TOTAL or _COMP_TARGETS."
        )

    out_a = _run(flavor, n_gpu=1)
    out_b = _run(flavor, n_gpu=n_devices)

    # Per-replica log-evidence must agree.
    np.testing.assert_allclose(
        out_a["log_evidence"],
        out_b["log_evidence"],
        atol=_LOG_Z_ATOL,
        rtol=_LOG_Z_RTOL,
    )
    assert (out_a["n_dead"] == out_b["n_dead"]).all()

    # Swap counts are integers. Every device decides on the same gathered
    # population with the same RNG, so they must match exactly; a mismatch
    # means divergent swap RNG or a float-order difference that flipped a
    # borderline accept.
    assert out_a["swap_attempted"] == out_b["swap_attempted"]
    assert out_a["swap_accepted"] == out_b["swap_accepted"]

    # Without any swap attempt the parity checks above would be vacuous.
    assert sum(out_a["swap_attempted"]) > 0

    # The final live walkers must also land on the same replica. XRENS and
    # semi-grand swaps change the types, which log_Z alone may not reveal.
    res_a, res_b = out_a["result"], out_b["result"]
    shape = (_N_TOTAL, _N_WALKERS, _N_ATOMS)
    np.testing.assert_array_equal(
        np.asarray(res_a["types"]).reshape(shape),
        np.asarray(res_b["types"]).reshape(shape),
    )
    np.testing.assert_allclose(
        np.asarray(res_a["energies"]).reshape(shape[:2]),
        np.asarray(res_b["energies"]).reshape(shape[:2]),
        atol=_LOG_Z_ATOL,
    )
