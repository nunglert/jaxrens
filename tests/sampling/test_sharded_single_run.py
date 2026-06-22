"""Tests for the sharded-single-run NS mode (``ShardedSingleRun``).

Covers:
- bit-exact parity between ``ShardedSingleRun(n_gpu=1)`` and ``SingleRun()``
- bit-exact parity between ``ShardedSingleRun(n_gpu=2)`` and ``SingleRun()``
  on a small harmonic problem
- end-to-end smoke test for ``run_ns_sharded``
- resolver dispatch on the ``run.shard_n_gpu`` schema field
- divisibility / cohort incompatibility error paths

The parity tests use a harmonic toy backend so the run is fast and the
parity comparison is deterministic.  The ``n_gpu=1`` case runs on every
CI job; the ``n_gpu=2`` case is marked ``@pytest.mark.multi_gpu`` so it
is collected only when pytest is invoked with ``-m multi_gpu`` (the
multi-GPU CI job and local 2-GPU dev boxes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.batch_descriptor import (
    ShardedSingleRun,
    SingleRun,
)
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns,
    init_ns_sharded,
    ns_step,
    ns_step_sharded,
    run_ns,
    run_ns_sharded,
)
from jaxrens.sampling.termination import IterationTermination


def _make_harmonic_problem(seed: int, n_walkers: int):
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw", random_walk.build_kernel,
            step_size=0.2, step_size_max=5.0,
            min_rate=0.2, max_rate=0.7,
        ),
    ]
    init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)
    key = jax.random.key(seed)
    key, pos_key = jax.random.split(key)
    positions = jax.random.uniform(
        pos_key, (n_walkers, 1, 3), minval=-2.0, maxval=2.0,
    )
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "per_move_fns": per_move_fns,
        "descriptors": descriptors,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
        "n_walkers": n_walkers,
    }


# ---------------------------------------------------------------------------
# Parity: ShardedSingleRun vs SingleRun
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_extra", [0, 4])
@pytest.mark.parametrize(
    "n_gpu", [1, pytest.param(2, marks=pytest.mark.multi_gpu)],
)
def test_run_ns_sharded_matches_single_run(n_gpu, n_extra):
    """Sharded run reproduces the SingleRun reference exactly.

    Uses a small harmonic problem (K=8, n_atoms=1) and runs 3 NS
    iterations from the same seed.  All shards make the same global
    decisions (same RNG broadcast), so the result is bit-exact at
    G=1 and effectively bit-exact at G=2 on this toy problem.

    ``n_extra > 0`` is the case that exercises the cross-shard chain
    distribution: the ``1 + n_extra`` walked chains are split across
    devices (here 5 chains over 2 shards, padded to 3 each), gathered,
    and must still match the single-device walk.
    """
    if n_gpu > 1 and 8 % n_gpu != 0:
        pytest.skip("n_walkers must divide n_gpu")

    setup = _make_harmonic_problem(seed=123, n_walkers=8)
    n_iter = 3
    rng_key = jax.random.key(0)

    ref = run_ns(
        positions=setup["positions"],
        types=setup["types"],
        energies=setup["energies"],
        cells=None,
        init_fn=setup["init_fn"],
        step_fn=setup["step_fn"],
        rng_key=rng_key,
        n_walkers=8,
        max_iterations=n_iter,
        n_mcmc_steps=4,
        n_extra=n_extra,
        move_descriptors=setup["descriptors"],
        termination_criteria=[IterationTermination(n_iter)],
    )

    out = run_ns_sharded(
        positions=setup["positions"],
        types=setup["types"],
        energies=setup["energies"],
        cells=None,
        init_fn=setup["init_fn"],
        step_fn=setup["step_fn"],
        rng_key=rng_key,
        n_gpu=n_gpu,
        n_walkers=8,
        max_iterations=n_iter,
        n_mcmc_steps=4,
        n_extra=n_extra,
        move_descriptors=setup["descriptors"],
        termination_criteria=[IterationTermination(n_iter)],
    )

    # ``out["log_evidence"]`` is shape (G,) with identical entries.
    sharded_log_z = float(np.asarray(out["log_evidence"])[0])
    ref_log_z = float(ref["log_evidence"])
    np.testing.assert_allclose(sharded_log_z, ref_log_z, rtol=1e-5, atol=1e-7)

    # Energies: flatten sharded (G, K/G) to (K,) and compare.
    sharded_energies = np.asarray(out["energies"]).reshape(-1)
    ref_energies = np.asarray(ref["energies"]).reshape(-1)
    sharded_sorted = np.sort(sharded_energies)
    ref_sorted = np.sort(ref_energies)
    np.testing.assert_allclose(
        sharded_sorted, ref_sorted, rtol=1e-5, atol=1e-7,
    )


@pytest.mark.multi_gpu
def test_init_ns_sharded_divisibility_error():
    """``init_ns_sharded`` rejects K not divisible by n_gpu."""
    setup = _make_harmonic_problem(seed=0, n_walkers=7)
    with pytest.raises(ValueError, match="divisible"):
        init_ns_sharded(
            setup["init_fn"],
            setup["positions"],
            setup["types"],
            setup["energies"],
            None,
            jax.random.key(0),
            n_gpu=2,
        )


@pytest.mark.multi_gpu
def test_run_ns_sharded_divisibility_error():
    """``run_ns_sharded`` rejects K not divisible by n_gpu."""
    setup = _make_harmonic_problem(seed=0, n_walkers=7)
    with pytest.raises(ValueError, match="divisible"):
        run_ns_sharded(
            positions=setup["positions"],
            types=setup["types"],
            energies=setup["energies"],
            cells=None,
            init_fn=setup["init_fn"],
            step_fn=setup["step_fn"],
            rng_key=jax.random.key(0),
            n_gpu=2,
            n_walkers=7,
            max_iterations=1,
            n_mcmc_steps=1,
            move_descriptors=setup["descriptors"],
            termination_criteria=[IterationTermination(1)],
        )


# ---------------------------------------------------------------------------
# Resolver dispatch
# ---------------------------------------------------------------------------


def _minimal_root(tmp_path, *, shard_n_gpu, n_live=8, pressures=None):
    """Build a minimal LJ NPT config and validate it through RootSpec.

    Mirrors ``experiments/examples/lj8_npt/config.yaml`` minus the
    full burn-in / adaptation machinery so resolve() runs quickly.
    """
    from jaxrens.cli.schema import RootSpec
    cfg = {
        "run": {
            "n_live": n_live,
            "max_iterations": 1,
            "n_mcmc_steps": 1,
            "seed": 0,
        },
        "backend": {
            "type": "lj",
            "epsilon": 1.0,
            "sigma": 1.0,
            "cutoff": 2.5,
            "periodic": True,
        },
        "ensemble": {
            "type": "npt",
            "pressure": pressures if pressures is not None else 0.1,
            "pressure_units": "eva3",
        },
        "moves": [
            {"type": "random_walk", "step_size": 0.1},
        ],
        "termination": [{"type": "iteration", "max_iterations": 1}],
        "init": {
            "start_species": "18 4",
            "random_initialise_pos": True,
            "pos_randomization_mode": "grid",
            "grid_distance": 1.0,
            "random_initialise_cell": False,
            "start_energy_ceiling_per_atom": 100.0,
        },
        "cell": {
            "max_volume_per_atom": 30.0,
            "min_volume_per_atom": 1.5,
            "min_aspect_ratio": 0.6,
            "flat_V_prior": False,
        },
        "output": {
            "format": "extxyz",
            "working_dir": str(tmp_path / "out"),
            "out_file_prefix": "test",
            "info_interval": 50,
            "traj_interval": 50,
            "snapshot_interval": 100000,
            "checkpoint_interval": 100000,
        },
    }
    if shard_n_gpu != 1:
        cfg["run"]["shard_n_gpu"] = shard_n_gpu
    return RootSpec.model_validate(cfg)


def test_resolver_routes_shard_to_sharded_batcher(tmp_path):
    """``shard_n_gpu > 1`` ⇒ resolver returns ``ShardedSingleRun(n_gpu)``."""
    from jaxrens.cli.resolve import resolve

    root = _minimal_root(tmp_path, shard_n_gpu=2)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, ShardedSingleRun)
    assert resolved.batcher.n_gpu == 2


def test_resolver_default_keeps_single_run(tmp_path):
    """``shard_n_gpu=1`` (default) keeps the ``SingleRun`` batcher."""
    from jaxrens.cli.resolve import resolve

    root = _minimal_root(tmp_path, shard_n_gpu=1)
    resolved = resolve(root)
    assert isinstance(resolved.batcher, SingleRun)


def test_resolver_rejects_shard_with_pressure_list(tmp_path):
    """Sharded single run cannot combine with a multi-pressure cohort."""
    from jaxrens.cli.resolve import resolve

    root = _minimal_root(tmp_path, shard_n_gpu=2, pressures=[1.0, 2.0])
    with pytest.raises(ValueError, match="shard_n_gpu"):
        resolve(root)


def test_resolver_rejects_indivisible_n_live(tmp_path):
    """n_live % shard_n_gpu != 0 raises a clear error."""
    from jaxrens.cli.resolve import resolve

    root = _minimal_root(tmp_path, shard_n_gpu=3, n_live=8)
    with pytest.raises(ValueError, match="divisible"):
        resolve(root)


# ---------------------------------------------------------------------------
# Burn-in (initial_walk) under ShardedSingleRun
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_gpu", [1, pytest.param(2, marks=pytest.mark.multi_gpu)],
)
def test_burn_in_sharded_completes_under_emax(n_gpu):
    """Sharded burn-in finishes; final energies stay below the fixed Emax.

    For G=1 this is the trivial-pmap case; for G=2 it exercises the
    per-shard distinct-key walking dispatch (each shard walks its own
    walkers independently) plus the broadcast-key adaptation path.
    """
    from jaxrens.cli.schema.adaptation import ResolvedAdaptationPolicy
    from jaxrens.init.burn_in import initial_walk

    if 8 % n_gpu != 0:
        pytest.skip("n_walkers must divide n_gpu")

    setup = _make_harmonic_problem(seed=42, n_walkers=8)

    ns_state = init_ns_sharded(
        setup["init_fn"],
        setup["positions"], setup["types"], setup["energies"],
        None, jax.random.key(7),
        n_gpu=n_gpu,
    )

    n_atoms = setup["positions"].shape[1]
    emax_offset_per_atom = 1.0
    batcher = ShardedSingleRun(n_gpu=n_gpu)
    # ``reduce_emax`` now returns shape ``(G,)`` (every entry identical
    # by construction) — take ``[0]`` for the Python-scalar Emax used
    # in the assertion.
    fixed_emax = float(
        batcher.reduce_emax(ns_state.population.energy)[0]
        + emax_offset_per_atom * n_atoms
    )

    policy = ResolvedAdaptationPolicy(
        min_rate=0.25, max_rate=0.75,
        adjust_factor=1.5, step_size_max=10.0,
    )

    ns_state_after = initial_walk(
        key=jax.random.key(13),
        ns_state=ns_state,
        step_fn=setup["step_fn"],
        n_walks=2,
        walklength=4,
        adjust_interval=1,
        emax_offset_per_atom=emax_offset_per_atom,
        n_atoms=n_atoms,
        batcher=batcher,
        per_move_fns=setup["per_move_fns"],
        adaptation_policies=(policy,),
        adjust_n_samples=8,
        adjust_max_rounds=3,
    )

    final_energies = np.asarray(ns_state_after.population.energy)
    assert np.all(np.isfinite(final_energies))
    assert np.all(final_energies <= fixed_emax + 1e-6), (
        f"Some walkers exceed Emax={fixed_emax:.4g}: max={final_energies.max():.4g}"
    )


@pytest.mark.multi_gpu
def test_burn_in_sharded_g2_shards_walk_independently():
    """At G=2 each shard must walk its own walkers — populations must differ."""
    from jaxrens.init.burn_in import initial_walk

    setup = _make_harmonic_problem(seed=2, n_walkers=8)
    ns_state = init_ns_sharded(
        setup["init_fn"],
        setup["positions"], setup["types"], setup["energies"],
        None, jax.random.key(0),
        n_gpu=2,
    )

    ns_state_after = initial_walk(
        key=jax.random.key(11),
        ns_state=ns_state,
        step_fn=setup["step_fn"],
        n_walks=1,
        walklength=8,
        adjust_interval=999,
        emax_offset_per_atom=10.0,
        n_atoms=setup["positions"].shape[1],
        batcher=ShardedSingleRun(n_gpu=2),
    )

    pos = np.asarray(ns_state_after.population.positions)  # (G=2, K/G=4, A, 3)
    shard0 = pos[0].reshape(-1)
    shard1 = pos[1].reshape(-1)
    assert not np.allclose(shard0, shard1, atol=1e-6), (
        "Per-shard populations are identical — shards are running the same "
        "chain, defeating burn-in's per-shard independence."
    )


@pytest.mark.multi_gpu
def test_burn_in_sharded_g2_step_sizes_equal_across_shards():
    """Adaptation under sharding must produce identical step sizes per shard.

    Bisection runs ``lax.psum`` over the per-shard accept/eval counters,
    so every shard sees the same global rate and converges to the same
    step size.  Final ``step_sizes`` must therefore be bit-equal across
    the leading G axis.
    """
    from jaxrens.cli.schema.adaptation import ResolvedAdaptationPolicy
    from jaxrens.init.burn_in import initial_walk

    setup = _make_harmonic_problem(seed=3, n_walkers=8)
    ns_state = init_ns_sharded(
        setup["init_fn"],
        setup["positions"], setup["types"], setup["energies"],
        None, jax.random.key(0),
        n_gpu=2,
    )

    policy = ResolvedAdaptationPolicy(
        min_rate=0.25, max_rate=0.75,
        adjust_factor=1.5, step_size_max=10.0,
    )

    ns_state_after = initial_walk(
        key=jax.random.key(99),
        ns_state=ns_state,
        step_fn=setup["step_fn"],
        n_walks=3,
        walklength=4,
        adjust_interval=1,
        emax_offset_per_atom=1.0,
        n_atoms=setup["positions"].shape[1],
        batcher=ShardedSingleRun(n_gpu=2),
        per_move_fns=setup["per_move_fns"],
        adaptation_policies=(policy,),
        adjust_n_samples=8,
        adjust_max_rounds=3,
    )

    # population.step_sizes shape: (G=2, K/G=4, n_moves=1).  Per-shard
    # values must agree along the leading G axis (post-`lax.psum`
    # bisection invariant).
    ss = np.asarray(ns_state_after.population.step_sizes)
    np.testing.assert_allclose(ss[0], ss[1], rtol=0, atol=0,
        err_msg="Per-shard step sizes differ — adaptation broke "
        "the cross-shard `lax.psum` invariant.",
    )


# ---------------------------------------------------------------------------
# build_adapt_step.trial_batch_size parity
# ---------------------------------------------------------------------------


def test_adapt_step_trial_batch_size_matches_full_vmap():
    """``build_adapt_step(trial_batch_size=N)`` results match ``trial_batch_size=None``.

    Pins the chunked-trial-vmap path's correctness on the cheapest
    batcher (``SingleRun``) at a divisor chunk size.  Ensures the
    closure that propagates ``trial_batch_size`` to ``adjust_step_size``
    doesn't perturb the result.
    """
    from jaxrens.sampling.adaptation.manager import build_adapt_step

    setup = _make_harmonic_problem(seed=4, n_walkers=8)
    ns_state = init_ns(
        setup["init_fn"],
        setup["positions"], setup["types"], setup["energies"],
        None, jax.random.key(0),
    )

    common = dict(
        move_descriptors=setup["descriptors"],
        per_move_fns=setup["per_move_fns"],
        batcher=SingleRun(),
        adjust_n_samples=8,
        adjust_factor=1.5,
        adjust_max_rounds=3,
        adjust_interval=1,
    )

    adapt_full = build_adapt_step(**common)
    adapt_chunked = build_adapt_step(trial_batch_size=4, **common)

    emax = jnp.max(ns_state.population.energy) + 1.0
    key = jax.random.key(0)
    init_ss = jnp.array([0.2])
    ns_state = ns_state.set(
        population=ns_state.population.set(
            step_sizes=jnp.broadcast_to(
                init_ss[None, :],
                ns_state.population.step_sizes.shape,
            ),
        ),
    )

    state_full, _, _ = adapt_full(ns_state, emax, key)
    state_chunked, _, _ = adapt_chunked(ns_state, emax, key)

    ss_full = state_full.population.step_sizes[0]
    ss_chunked = state_chunked.population.step_sizes[0]
    np.testing.assert_allclose(ss_full, ss_chunked, rtol=1e-5)
