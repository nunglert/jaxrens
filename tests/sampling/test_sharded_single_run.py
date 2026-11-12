"""Tests for the sharded single-run NS mode (ShardedSingleRun).

The ``n_gpu=2`` cases are marked ``multi_gpu`` and run only with
``-m multi_gpu`` on a machine with two GPUs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.adaptation import ResolvedAdaptationPolicy
from jaxrens.init.burn_in import initial_walk
from jaxrens.sampling.batch_descriptor import ShardedSingleRun
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    init_ns_sharded,
    run_ns,
    run_ns_sharded,
)
from jaxrens.sampling.termination import IterationTermination


def _make_harmonic_problem(seed: int, n_walkers: int):
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            "rw",
            random_walk.build_kernel,
            step_size=0.2,
            step_size_max=5.0,
            min_rate=0.2,
            max_rate=0.7,
        ),
    ]
    init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)
    key = jax.random.key(seed)
    key, pos_key = jax.random.split(key)
    positions = jax.random.uniform(
        pos_key,
        (n_walkers, 1, 3),
        minval=-2.0,
        maxval=2.0,
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


@pytest.mark.parametrize("n_extra", [0, 4])
@pytest.mark.parametrize(
    "n_gpu",
    [1, pytest.param(2, marks=pytest.mark.multi_gpu)],
)
def test_run_ns_sharded_matches_single_run(n_gpu, n_extra):
    """run_ns_sharded matches run_ns on 8 walkers and 3 iterations.

    All shards make the same global decisions from the same broadcast key.
    With ``n_extra > 0`` the ``1 + n_extra`` walked chains are split across
    devices (5 chains over 2 shards, padded to 3 each) and gathered, and
    the result must still match the single-device walk.
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

    # log_evidence has shape (G,) with identical entries.
    sharded_log_z = float(np.asarray(out["log_evidence"])[0])
    ref_log_z = float(ref["log_evidence"])
    np.testing.assert_allclose(sharded_log_z, ref_log_z, rtol=1e-5, atol=1e-7)

    # Flatten the sharded (G, K/G) energies to (K,) and compare sorted.
    sharded_energies = np.asarray(out["energies"]).reshape(-1)
    ref_energies = np.asarray(ref["energies"]).reshape(-1)
    sharded_sorted = np.sort(sharded_energies)
    ref_sorted = np.sort(ref_energies)
    np.testing.assert_allclose(
        sharded_sorted,
        ref_sorted,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.multi_gpu
def test_init_ns_sharded_divisibility_error():
    """init_ns_sharded rejects K not divisible by n_gpu."""
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
    """run_ns_sharded rejects K not divisible by n_gpu."""
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


@pytest.mark.parametrize(
    "n_gpu",
    [1, pytest.param(2, marks=pytest.mark.multi_gpu)],
)
def test_burn_in_sharded_completes_under_emax(n_gpu):
    """Sharded burn-in finishes with all energies below the fixed Emax.

    At G=2 each shard walks its own walkers with its own key, while the
    adaptation uses a key broadcast to all shards.
    """
    if 8 % n_gpu != 0:
        pytest.skip("n_walkers must divide n_gpu")

    setup = _make_harmonic_problem(seed=42, n_walkers=8)

    ns_state = init_ns_sharded(
        setup["init_fn"],
        setup["positions"],
        setup["types"],
        setup["energies"],
        None,
        jax.random.key(7),
        n_gpu=n_gpu,
    )

    n_atoms = setup["positions"].shape[1]
    emax_offset_per_atom = 1.0
    batcher = ShardedSingleRun(n_gpu=n_gpu)
    # reduce_emax returns shape (G,) with identical entries; take the first.
    fixed_emax = float(
        batcher.reduce_emax(ns_state.population.energy)[0]
        + emax_offset_per_atom * n_atoms
    )

    policy = ResolvedAdaptationPolicy(
        min_rate=0.25,
        max_rate=0.75,
        adjust_factor=1.5,
        step_size_max=10.0,
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
    assert np.all(final_energies <= fixed_emax + 1e-6)


@pytest.mark.multi_gpu
def test_burn_in_sharded_g2_shards_walk_independently():
    """At G=2 both shards move their walkers, and not in the same way."""
    setup = _make_harmonic_problem(seed=2, n_walkers=8)
    ns_state = init_ns_sharded(
        setup["init_fn"],
        setup["positions"],
        setup["types"],
        setup["energies"],
        None,
        jax.random.key(0),
        n_gpu=2,
    )

    pos_before = np.asarray(ns_state.population.positions)

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

    pos_after = np.asarray(ns_state_after.population.positions)
    for shard in range(2):
        assert not np.allclose(pos_after[shard], pos_before[shard], atol=1e-6)
    assert not np.allclose(pos_after[0], pos_after[1], atol=1e-6)


@pytest.mark.multi_gpu
def test_burn_in_sharded_g2_step_sizes_equal_across_shards():
    """Adaptation gives bit-equal step sizes on both shards.

    The bisection sums the accept and evaluation counters over shards with
    ``lax.psum``, so every shard sees the same global rate.
    """
    setup = _make_harmonic_problem(seed=3, n_walkers=8)
    ns_state = init_ns_sharded(
        setup["init_fn"],
        setup["positions"],
        setup["types"],
        setup["energies"],
        None,
        jax.random.key(0),
        n_gpu=2,
    )

    policy = ResolvedAdaptationPolicy(
        min_rate=0.25,
        max_rate=0.75,
        adjust_factor=1.5,
        step_size_max=10.0,
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

    ss = np.asarray(ns_state_after.population.step_sizes)
    np.testing.assert_allclose(ss[0], ss[1], rtol=0, atol=0)
