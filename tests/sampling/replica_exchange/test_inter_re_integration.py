"""Integration tests for inter-replica exchange inside the NS drivers.

The tests run short NS with inter_re_config through run_ns_parallel
(VmapRuns) and run_ns_multi_gpu (PmapVmapRuns, one GPU), check the
flavor errors raised at run time, and check that RECallback writes the
per-pair swap counts to a .re_stats.h5 file.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.monitor import RECallback
from jaxrens.io.re_stats_log import RELogger
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns_multi_gpu, run_ns_parallel
from jaxrens.sampling.termination import IterationTermination
from jaxrens.state.config import InterREConfig

from .test_xrens import SpeciesHarmonicBackend


def _make_parallel_problem(
    n_runs: int,
    n_walkers: int,
    seed: int = 42,
):
    """Build one-atom harmonic walkers for n_runs runs and an MwG kernel."""
    backend = create_harmonic(k=1.0)

    descriptors = [
        MoveKernel(
            "rw", random_walk.build_kernel, step_size=0.3, step_size_max=2.0
        )
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    key = jax.random.key(seed)
    keys = jax.random.split(key, n_runs + 2)
    rng_keys = keys[:n_runs]
    pos_key = keys[-1]

    positions = jax.random.uniform(
        pos_key, (n_runs, n_walkers, 1, 3), minval=-1.5, maxval=1.5
    )
    types = jnp.zeros((1,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)

    return {
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": None,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "rng_keys": rng_keys,
        "backend": backend,
    }


class TestPressureRENSEndToEnd:
    """Check a 5-iteration pressure RE run with 2 runs and re_interval=1.

    The walkers have no cell, so P*V is zero and the swaps compare bare
    energies. The run is shared by the tests in the class.
    """

    _result = None
    _re_stats_per_iter = None

    @classmethod
    def _run(cls):
        if cls._result is not None:
            return

        # Seed 123 gives at least one accepted swap in 5 iterations.
        prob = _make_parallel_problem(
            n_runs=2,
            n_walkers=15,
            seed=123,
        )

        inter_re_cfg = InterREConfig(
            flavor="pressure", re_interval=1, n_swap_cycles=1
        )
        re_stats_log = []

        class _Collector:
            def on_iteration(self, iteration, ns_state, info):
                s = info.get("inter_re_stats")
                if s is not None:
                    re_stats_log.append((iteration, dict(s)))

        termination = [IterationTermination(5)]

        ensemble_params_per_run = [
            {"pressure": 0.01},
            {"pressure": 0.1},
        ]

        cls._result = run_ns_parallel(
            positions=prob["positions"],
            types=prob["types"],
            energies=prob["energies"],
            cells=prob["cells"],
            init_fn=prob["init_fn"],
            step_fn=prob["step_fn"],
            rng_keys=prob["rng_keys"],
            n_walkers=15,
            max_iterations=5,
            n_mcmc_steps=5,
            termination_criteria=termination,
            ensemble_params_per_run=ensemble_params_per_run,
            inter_re_config=inter_re_cfg,
            callbacks=[_Collector()],
        )
        cls._re_stats_per_iter = re_stats_log

    def test_re_stats_present_every_iter(self):
        """The manager fires on iterations 1-4 and accepts at least one
        swap."""
        self._run()
        iterations = [it for it, _ in self._re_stats_per_iter]
        assert iterations == [1, 2, 3, 4]
        n_accepted = sum(
            s["n_swap_pairs_accepted"] for _, s in self._re_stats_per_iter
        )
        assert n_accepted >= 1

    def test_result_shapes(self):
        self._run()
        result = self._result
        assert result["positions"].shape[0] == 2
        assert result["log_evidence"].shape == (2,)
        assert result["n_dead"].shape == (2,)
        assert result["energies"].shape == (2, 15)

    def test_log_evidence_finite(self):
        self._run()
        assert jnp.all(jnp.isfinite(self._result["log_evidence"]))


def test_single_run_via_parallel_n_runs_1():
    """run_ns_parallel with one run and inter_re_config has no pair to
    swap and still returns a valid result."""
    backend = create_harmonic(k=1.0)
    descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    n_walkers = 10
    key = jax.random.key(0)
    pos_key, rng_key = jax.random.split(key)

    positions = jax.random.uniform(
        pos_key, (1, n_walkers, 1, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)
    rng_keys = jax.random.split(rng_key, 1)

    inter_re_cfg = InterREConfig(flavor="pressure", re_interval=1)
    attempted = []

    class _Collector:
        def on_iteration(self, iteration, ns_state, info):
            s = info.get("inter_re_stats")
            if s is not None:
                attempted.append(s["n_swap_pairs_attempted"])

    result = run_ns_parallel(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        max_iterations=5,
        n_mcmc_steps=3,
        termination_criteria=[IterationTermination(5)],
        inter_re_config=inter_re_cfg,
        callbacks=[_Collector()],
    )

    assert result["log_evidence"].shape == (1,)
    assert jnp.all(jnp.isfinite(result["log_evidence"]))
    assert attempted == [0, 0, 0, 0]


def test_pmap_vmap_pressure_smoke():
    """Pressure RE runs through run_ns_multi_gpu with one GPU and two
    runs per GPU."""
    backend = create_harmonic(k=1.0)
    descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    n_gpu, n_per_gpu, n_walkers = 1, 2, 8
    key = jax.random.key(5)
    keys = jax.random.split(key, n_gpu * n_per_gpu + 1)
    rng_keys = keys[:-1]
    pos_key = keys[-1]

    positions = jax.random.uniform(
        pos_key,
        (n_gpu * n_per_gpu, n_walkers, 1, 3),
        minval=-2.0,
        maxval=2.0,
    )
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)

    inter_re_cfg = InterREConfig(flavor="pressure", re_interval=1)

    result = run_ns_multi_gpu(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        n_gpu=n_gpu,
        n_per_gpu=n_per_gpu,
        max_iterations=5,
        n_mcmc_steps=3,
        termination_criteria=[IterationTermination(5)],
        inter_re_config=inter_re_cfg,
    )

    assert result["log_evidence"].shape == (n_gpu, n_per_gpu)
    assert jnp.all(jnp.isfinite(result["log_evidence"]))


def test_pmap_vmap_xrens_run():
    """XRENS runs through run_ns_multi_gpu: the manager fires on every
    iteration after the first and each fire costs 2 energy evaluations."""
    backend = SpeciesHarmonicBackend([1.0, 1.5])
    descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.3)]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    keys = jax.random.split(jax.random.key(42), 3)
    positions = jax.random.uniform(
        keys[2], (2, 10, 8, 3), minval=-0.5, maxval=0.5
    )
    types_0 = jnp.zeros(8, dtype=jnp.int32)
    types = jnp.stack([types_0, types_0.at[4:].set(1)])
    energies = jnp.stack(
        [
            jax.vmap(lambda p: backend(p, types[r], jnp.zeros((3, 3))).energy)(
                positions[r]
            )
            for r in range(2)
        ]
    )

    re_stats_log = []

    class _Collector:
        def on_iteration(self, iteration, ns_state, info):
            s = info.get("inter_re_stats")
            if s is not None:
                re_stats_log.append(dict(s))

    run_ns_multi_gpu(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=keys[:2],
        n_gpu=1,
        n_per_gpu=2,
        n_walkers=10,
        max_iterations=6,
        n_mcmc_steps=3,
        termination_criteria=[IterationTermination(6)],
        inter_re_config=InterREConfig(
            flavor="xrens",
            re_interval=1,
            composition_targets=((8, 0), (4, 4)),
        ),
        backend=backend,
        callbacks=[_Collector()],
    )

    assert len(re_stats_log) == 5
    assert all(s["n_energy_evals"] == 2 for s in re_stats_log)


def _run_without_per_run_params(flavor, driver):
    """Start a 2-run NS whose inter_re_config lacks the flavor's per-run
    parameters."""
    backend = create_harmonic(k=1.0)
    descriptors = [MoveKernel("rw", random_walk.build_kernel, step_size=0.2)]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    n_runs, n_walkers = 2, 8
    key = jax.random.key(0)
    rng_keys = jax.random.split(key, n_runs)
    pos_key, _ = jax.random.split(key)
    positions = jax.random.uniform(pos_key, (n_runs, n_walkers, 1, 3))
    types = jnp.zeros((1,), dtype=jnp.int32)
    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)

    kwargs = dict(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        max_iterations=3,
        n_mcmc_steps=2,
        termination_criteria=[IterationTermination(3)],
        inter_re_config=InterREConfig(flavor=flavor),
    )
    if driver == "multi_gpu":
        run_ns_multi_gpu(n_gpu=1, n_per_gpu=n_runs, **kwargs)
    else:
        run_ns_parallel(**kwargs)


@pytest.mark.parametrize("driver", ["parallel", "multi_gpu"])
def test_xrens_raises_at_run_time_without_composition_targets(driver):
    with pytest.raises(ValueError, match="requires composition_targets"):
        _run_without_per_run_params("xrens", driver)


@pytest.mark.parametrize("driver", ["parallel", "multi_gpu"])
def test_semi_grand_raises_at_run_time_without_chemical_potentials(driver):
    with pytest.raises(ValueError, match="requires chemical_potentials"):
        _run_without_per_run_params("semi_grand", driver)


class TestRECallbackEndToEnd:
    """Check that RECallback, in the run_ns_multi_gpu callback list,
    writes one row of per-pair counts per manager fire to
    <prefix>.re_stats.h5, and no file when the manager never fires."""

    def test_writes_file_with_per_pair_counts(self, tmp_path):
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw", random_walk.build_kernel, step_size=0.2)
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_gpu, n_per_gpu, n_walkers = 1, 3, 8
        n_total = n_gpu * n_per_gpu
        key = jax.random.key(11)
        rng_keys = jax.random.split(key, n_total)
        pos_key, _ = jax.random.split(key)
        positions = jax.random.uniform(
            pos_key,
            (n_total, n_walkers, 1, 3),
            minval=-1.5,
            maxval=1.5,
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)

        inter_re_cfg = InterREConfig(flavor="pressure", re_interval=1)

        path = tmp_path / "test.re_stats.h5"
        re_logger = RELogger(path=path, n_pairs=n_total - 1, flavor="pressure")
        callbacks = [RECallback(re_logger)]

        run_ns_multi_gpu(
            positions=positions,
            types=types,
            energies=energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
            n_walkers=n_walkers,
            max_iterations=5,
            n_mcmc_steps=3,
            termination_criteria=[IterationTermination(5)],
            inter_re_config=inter_re_cfg,
            callbacks=callbacks,
        )

        assert path.exists()
        log = RELogger.read(path)
        assert log.n_pairs == n_total - 1
        assert log.flavor == "pressure"
        # With 3 runs each pair is attempted once per fire.
        assert np.array_equal(log.iterations, [1, 2, 3, 4])
        assert np.all(log.n_attempted_per_pair == 1)
        assert np.all(log.n_accepted_per_pair <= log.n_attempted_per_pair)

    def test_no_file_when_re_interval_skips_all_iterations(self, tmp_path):
        """re_interval=999 with 3 iterations never fires the manager."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw", random_walk.build_kernel, step_size=0.2)
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_gpu, n_per_gpu, n_walkers = 1, 2, 8
        n_total = n_gpu * n_per_gpu
        key = jax.random.key(11)
        rng_keys = jax.random.split(key, n_total)
        pos_key, _ = jax.random.split(key)
        positions = jax.random.uniform(
            pos_key,
            (n_total, n_walkers, 1, 3),
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)

        inter_re_cfg = InterREConfig(flavor="pressure", re_interval=999)

        path = tmp_path / "test.re_stats.h5"
        re_logger = RELogger(path=path, n_pairs=n_total - 1, flavor="pressure")
        callbacks = [RECallback(re_logger)]

        run_ns_multi_gpu(
            positions=positions,
            types=types,
            energies=energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
            n_walkers=n_walkers,
            max_iterations=3,
            n_mcmc_steps=2,
            termination_criteria=[IterationTermination(3)],
            inter_re_config=inter_re_cfg,
            callbacks=callbacks,
        )

        assert not path.exists()

    def test_re_interval_2_writes_subsampled_iterations(self, tmp_path):
        """With re_interval=2 and 5 iterations the manager fires at
        iterations 2 and 4 (iteration > 0 and iteration % 2 == 0)."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw", random_walk.build_kernel, step_size=0.2)
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_gpu, n_per_gpu, n_walkers = 1, 2, 8
        n_total = n_gpu * n_per_gpu
        key = jax.random.key(11)
        rng_keys = jax.random.split(key, n_total)
        pos_key, _ = jax.random.split(key)
        positions = jax.random.uniform(
            pos_key,
            (n_total, n_walkers, 1, 3),
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)

        inter_re_cfg = InterREConfig(flavor="pressure", re_interval=2)

        path = tmp_path / "test.re_stats.h5"
        re_logger = RELogger(path=path, n_pairs=n_total - 1, flavor="pressure")
        callbacks = [RECallback(re_logger)]

        run_ns_multi_gpu(
            positions=positions,
            types=types,
            energies=energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
            n_walkers=n_walkers,
            max_iterations=5,
            n_mcmc_steps=2,
            termination_criteria=[IterationTermination(5)],
            inter_re_config=inter_re_cfg,
            callbacks=callbacks,
        )

        log = RELogger.read(path)
        assert np.array_equal(log.iterations, [2, 4])
