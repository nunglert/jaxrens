"""Multi-GPU parity test for inter-replica-exchange (RE) flavors.

Why this test exists
--------------------
The pmap RE path in ``InterREManager`` (``src/jaxrens/sampling/inter_re_manager.py``)
all-gathers each device's shard, swaps on the full population using the same
RNG on every device, then re-shards via ``axis_index``.  On ``n_gpu=1`` the
all_gather is structurally a no-op, so single-device tests cannot probe:

* axis-correctness of the all_gather → reshape → step → reshape → slice
  round-trip for ``n_gpu>1``;
* the "every device produces identical swap decisions" invariant that
  guarantees consistent re-sharding;
* whether per-replica ``log_Z`` and per-iteration swap-accept counts depend
  on the device topology.

This file runs each RE flavor (``pressure``, ``semi_grand``, ``xrens``) twice
with the *same per-replica seed and same initial walkers* under two device
layouts:

* topology A — ``(n_gpu=1, n_per_gpu=R)``  (single-device pmap, degenerate);
* topology B — ``(n_gpu=N, n_per_gpu=R/N)`` (real cross-device pmap).

It then asserts:

* per-replica ``log_Z`` agrees within a tight tolerance (float32 reductions
  may differ in the last bit between devices);
* per-iteration swap-attempt and swap-accept counts match exactly (these are
  integer outputs and should be bit-identical given the same RNG and the same
  all-gathered population).

Marked ``@pytest.mark.multi_gpu`` only (no ``heavy``) so it runs as a fast
check on a 2- or 4-GPU host.  Uses the cheap multi-species LJ backend
(``ε`` and ``σ`` per species, Lorentz-Berthelot mixing) so XRENS sees real
composition-dependent energies.
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


# ---------------------------------------------------------------------------
# Problem setup helpers
# ---------------------------------------------------------------------------


_BASE_SEED = 20260508
_N_TOTAL = 4          # number of replicas (must divide both 1 and N_GPU choices)
_N_WALKERS = 8
_N_ATOMS = 4
_CELL_SIDE = 4.0
_MAX_ITERS = 8
_N_MCMC = 3
_INITIAL_STEP_SIZE = 0.15

# Two species so XRENS / per-species LJ have real work to do.
_EPS_TABLE = (1.0, 0.8)
_SIG_TABLE = (1.0, 1.1)
_CUTOFF = 2.5

# Per-replica composition targets (must each sum to _N_ATOMS).
_COMP_TARGETS = ((4, 0), (3, 1), (2, 2), (1, 3))


def _initial_types_for_composition(comp: tuple[int, ...]) -> jnp.ndarray:
    parts = [jnp.full(c, s, dtype=jnp.int32) for s, c in enumerate(comp)]
    return jnp.concatenate(parts)


def _build_problem(flavor: str):
    """Build the shared per-replica problem that's identical across topologies.

    Returns a dict carrying everything needed to feed both ``run_ns_multi_gpu``
    invocations.  All randomness is seeded so two calls with the same flavor
    produce identical inputs.
    """
    backend = create_lj(epsilon=_EPS_TABLE, sigma=_SIG_TABLE, cutoff=_CUTOFF)

    descriptors = [
        MoveKernel(
            "rw", random_walk.build_kernel,
            step_size=_INITIAL_STEP_SIZE, step_size_max=1.0,
        )
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    base_key = jax.random.key(_BASE_SEED)
    pos_key, types_key, run_keys_key = jax.random.split(base_key, 3)

    # Per-replica RNG keys — flat (n_total,) so that reshape((G, P)) is
    # layout-invariant: replica i gets the same key regardless of (G, P).
    rng_keys_flat = jax.random.split(run_keys_key, _N_TOTAL)

    # Initial positions: random in a cube smaller than the cell to avoid
    # immediate hard-overlap; same per-replica positions in both topologies.
    positions = jax.random.uniform(
        pos_key,
        (_N_TOTAL, _N_WALKERS, _N_ATOMS, 3),
        minval=0.5, maxval=_CELL_SIDE - 0.5,
    )

    # Cells: identical cubic cell on every replica/walker so volume is
    # well-defined (pressure-RENS uses _get_volume(cell)).
    cells = jnp.broadcast_to(
        _CELL_SIDE * jnp.eye(3),
        (_N_TOTAL, _N_WALKERS, 3, 3),
    )

    # Per-replica types that match each replica's composition target.  Even
    # for non-XRENS flavors this is harmless (the type vector is just along
    # for the ride).  Same dtype as the existing tests use.
    types_per_replica = jnp.stack(
        [_initial_types_for_composition(c) for c in _COMP_TARGETS]
    )  # (n_total, n_atoms)

    # Initial energies: evaluate the LJ backend on each (replica, walker) pair
    # using that replica's types.  Vmap nesting matches positions shape.
    def _energy_one(pos_aw3, types_a):
        return backend(pos_aw3, types_a, _CELL_SIDE * jnp.eye(3), 0)[0]

    energies = jax.vmap(
        lambda pos_kaw3, types_a: jax.vmap(lambda p: _energy_one(p, types_a))(
            pos_kaw3
        )
    )(positions, types_per_replica)

    if flavor == "pressure":
        inter_re_cfg = InterREConfig(
            flavor="pressure", every=1, n_swap_cycles=1,
        )
        ensemble_params_per_run = [
            {"pressure": 0.0},
            {"pressure": 0.05},
            {"pressure": 0.1},
            {"pressure": 0.15},
        ]
    elif flavor == "semi_grand":
        inter_re_cfg = InterREConfig(
            flavor="semi_grand", every=1, n_swap_cycles=1,
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
            flavor="xrens", every=1, n_swap_cycles=1,
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
    assert _N_TOTAL % n_gpu == 0, (
        f"_N_TOTAL={_N_TOTAL} must be divisible by n_gpu={n_gpu}"
    )
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


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


# Tight float32 tolerance: log_Z is dominated by per-replica live-set
# evolution (no cross-device reductions), so we expect near-bit-exact
# agreement.  Loosened slightly to absorb the small float-order difference
# in pmap collectives versus single-device pmap.
_LOG_Z_ATOL = 1e-3
_LOG_Z_RTOL = 1e-3


@pytest.mark.multi_gpu
@pytest.mark.parametrize("flavor", ["pressure", "semi_grand", "xrens"])
def test_re_pmap_topology_parity(flavor: str) -> None:
    """``(n_gpu=1, n_per_gpu=R)`` and ``(n_gpu=N, n_per_gpu=R/N)`` agree.

    The all-gather/replicate/swap path must produce identical per-replica
    log-evidence and identical per-iteration swap-accept counts regardless
    of how the R replicas are distributed across devices.  Mismatches here
    indicate a real cross-device correctness bug — wrong ``axis=`` on the
    all_gather, divergent swap RNG between devices, or a re-shard slicing
    error.
    """
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
        out_a["log_evidence"], out_b["log_evidence"],
        atol=_LOG_Z_ATOL, rtol=_LOG_Z_RTOL,
        err_msg=(
            f"[{flavor}] log_Z disagrees between (n_gpu=1, n_per_gpu={_N_TOTAL}) "
            f"and (n_gpu={n_devices}, n_per_gpu={_N_TOTAL // n_devices}):\n"
            f"  topology A: {out_a['log_evidence']}\n"
            f"  topology B: {out_b['log_evidence']}"
        ),
    )

    # NS termination counter (n_dead) must match exactly.
    assert (out_a["n_dead"] == out_b["n_dead"]).all(), (
        f"[{flavor}] n_dead diverged: A={out_a['n_dead']}, B={out_b['n_dead']}"
    )

    # Per-iteration swap stats are integers — every device produces the same
    # decisions on the same all-gathered population, so this should match
    # exactly.  A mismatch here is the smoking gun for divergent swap RNG
    # or a float-order difference flipping a borderline accept/reject.
    assert out_a["swap_attempted"] == out_b["swap_attempted"], (
        f"[{flavor}] per-iter n_swap_pairs_attempted diverged: "
        f"A={out_a['swap_attempted']}, B={out_b['swap_attempted']}"
    )
    assert out_a["swap_accepted"] == out_b["swap_accepted"], (
        f"[{flavor}] per-iter n_swap_pairs_accepted diverged: "
        f"A={out_a['swap_accepted']}, B={out_b['swap_accepted']}"
    )

    # Sanity: the swap pass actually fired and proposed at least one pair.
    # If everything in this test is a no-op the parity assertion above is
    # vacuous.
    assert sum(out_a["swap_attempted"]) > 0, (
        f"[{flavor}] no swap pairs attempted across {_MAX_ITERS} iters — "
        f"the swap pass never fired and the parity assertion is vacuous."
    )
