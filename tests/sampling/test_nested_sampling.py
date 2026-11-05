"""Tests for full nested sampling runs: run_ns and run_ns_parallel."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.monitor import AccRatesCallback, AdaptationCallback
from jaxrens.io.acc_rates_log import AccRatesLogger
from jaxrens.io.adaptation_log import AdaptationLogger
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns, run_ns_parallel


@pytest.fixture
def harmonic_setup():
    """Single-run harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("random_walk", random_walk.build_kernel),
        ],
    )

    n_walkers = 50
    n_atoms = 1
    key = jax.random.key(42)

    key, init_key = jax.random.split(key)
    positions = jax.random.uniform(
        init_key, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "key": key,
        "n_walkers": n_walkers,
    }


@pytest.fixture
def parallel_setup():
    """2-run parallel harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("random_walk", random_walk.build_kernel),
        ],
    )

    n_runs = 2
    n_walkers = 20
    n_atoms = 1

    keys = jax.random.split(jax.random.key(0), n_runs)
    positions = jax.vmap(
        lambda k: jax.random.uniform(
            k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
        )
    )(keys)

    types = jnp.zeros((n_atoms,), dtype=jnp.int32)

    energies = jax.vmap(
        lambda pos: jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(pos)
    )(positions)

    rng_keys = jax.random.split(jax.random.key(42), n_runs)

    return {
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "rng_keys": rng_keys,
        "n_runs": n_runs,
        "n_walkers": n_walkers,
    }


class TestRunNS:
    """Check run_ns on a harmonic well: completion, evidence and callbacks."""

    def test_runs_to_completion(self, harmonic_setup):
        s = harmonic_setup
        result = run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=100,
            n_mcmc_steps=5,
            initial_step_size=0.3,
        )
        assert result["iteration"] > 0
        assert result["n_dead"] > 0
        assert jnp.isfinite(result["log_evidence"])

    @pytest.mark.heavy
    def test_harmonic_evidence_accuracy(self):
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel("random_walk", random_walk.build_kernel),
            ],
        )

        n_walkers = 30
        L = 5.0
        key = jax.random.key(123)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, 1, 3), minval=-L, maxval=L
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        result = run_ns(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=500,
            n_mcmc_steps=5,
            initial_step_size=0.5,
            target_acceptance=0.4,
            convergence_threshold=0.5,
        )

        log_evidence = float(result["log_evidence"])

        log_prior_volume = 3.0 * jnp.log(2.0 * L)
        log_Z_analytical = 1.5 * jnp.log(2.0 * jnp.pi) - log_prior_volume

        assert abs(log_evidence - float(log_Z_analytical)) < 1.5

    def test_callbacks_invoked(self, harmonic_setup):
        s = harmonic_setup

        iterations_seen = []

        class TestCallback:
            def on_iteration(self, iteration, ns_state, info):
                iterations_seen.append(iteration)

            def on_finish(self, ns_state):
                iterations_seen.append("finish")

        result = run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=20,
            n_mcmc_steps=5,
            callbacks=[TestCallback()],
        )

        assert len(iterations_seen) > 1
        assert iterations_seen[-1] == "finish"
        assert 0 in iterations_seen

    def test_logs_total_mcmc_steps_per_iteration(self, harmonic_setup, caplog):
        """The logged total is (1 + n_extra) * n_mcmc_steps, as in ns_step."""
        s = harmonic_setup
        with caplog.at_level(
            logging.INFO, logger="jaxrens.sampling.nested_sampling"
        ):
            run_ns(
                s["positions"],
                s["types"],
                s["energies"],
                cells=None,
                init_fn=s["init_fn"],
                step_fn=s["step_fn"],
                rng_key=s["key"],
                max_iterations=1,
                n_mcmc_steps=4,
                n_extra=2,
            )
        assert any(
            "total_mcmc_steps_per_iter=12" in r.message for r in caplog.records
        )


class TestRunNsParallel:
    """Check run_ns_parallel shapes and agreement with run_ns."""

    def test_basic_completion(self, parallel_setup):
        s = parallel_setup
        result = run_ns_parallel(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=s["rng_keys"],
            max_iterations=50,
            n_mcmc_steps=5,
            initial_step_size=0.3,
        )

        assert result["n_runs"] == 2
        assert result["log_evidence"].shape == (2,)
        assert result["iteration"].shape == (2,)
        assert jnp.all(jnp.isfinite(result["log_evidence"]))
        assert jnp.all(result["n_dead"] > 0)

    def test_parallel_matches_sequential(self, parallel_setup):
        s = parallel_setup
        n_iter = 50
        n_mcmc = 5

        result_par = run_ns_parallel(
            s["positions"],
            s["types"],
            s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=s["rng_keys"],
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        result_seq_0 = run_ns(
            s["positions"][0],
            s["types"],
            s["energies"][0],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["rng_keys"][0],
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        log_Z_par = float(result_par["log_evidence"][0])
        log_Z_seq = float(result_seq_0["log_evidence"])
        # The two paths split keys differently, so the evidence agrees only
        # up to RNG noise, within about one log unit.
        assert abs(log_Z_par - log_Z_seq) < 1.0


class TestAdjustmentInfoKeys:
    """Check the adjustment_* info keys emitted by run_ns."""

    def test_adjustment_keys_present_on_adjust_iterations(self):
        """All adjustment_* keys appear together, only on adjust iterations."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "random_walk",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_walkers = 20
        key = jax.random.key(42)
        key, init_key = jax.random.split(key)
        positions = 0.5 * jax.random.normal(init_key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        _ADJ_KEYS = (
            "adjustment_n_rounds",
            "adjustment_converged",
            "adjustment_cap_hits",
            "adjustment_floor_hits",
            "adjustment_bracket_detected",
        )
        adjust_interval = 10
        adjust_iterations = []
        other_iterations = []
        partial_iterations = []

        class _RecordCallback:
            def on_iteration(self, iteration, ns_state, info):
                present = [k in info for k in _ADJ_KEYS]
                if all(present):
                    adjust_iterations.append(iteration)
                elif any(present):
                    partial_iterations.append(iteration)
                else:
                    other_iterations.append(iteration)

            def on_finish(self, ns_state):
                pass

        run_ns(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=35,
            n_mcmc_steps=3,
            per_move_fns=per_move_fns,
            move_descriptors=descriptors,
            adjust_interval=adjust_interval,
            adjust_n_samples=10,
            callbacks=[_RecordCallback()],
        )

        assert len(adjust_iterations) > 0
        assert all(it % adjust_interval == 0 for it in adjust_iterations)
        assert len(other_iterations) > 0
        assert partial_iterations == []

    def test_adjustment_arrays_have_correct_shapes(self):
        """Adjustment info arrays have shape (n_moves,) and correct dtypes."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "rw0",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
            MoveKernel(
                "rw1",
                random_walk.build_kernel,
                step_size=0.2,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_walkers = 20
        key = jax.random.key(7)
        key, init_key = jax.random.split(key)
        positions = 0.5 * jax.random.normal(init_key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        captured_infos = []

        class _CaptureCallback:
            def on_iteration(self, iteration, ns_state, info):
                if "adjustment_n_rounds" in info:
                    captured_infos.append(dict(info))

            def on_finish(self, ns_state):
                pass

        run_ns(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=25,
            n_mcmc_steps=3,
            per_move_fns=per_move_fns,
            move_descriptors=descriptors,
            adjust_interval=10,
            adjust_n_samples=10,
            callbacks=[_CaptureCallback()],
        )

        assert len(captured_infos) > 0
        info = captured_infos[0]

        n_moves = len(descriptors)
        assert jnp.asarray(info["adjustment_n_rounds"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_converged"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_cap_hits"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_floor_hits"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_bracket_detected"]).shape == (
            n_moves,
        )


class TestAdjustmentInfoKeysBatched:
    """Check the adjustment info of batched runs under run_ns_parallel.

    ``info`` carries per-replica arrays of shape ``(n_runs, n_moves[, 4])``,
    and AdaptationCallback with AdaptationLogger writes an ``.adaptation.h5``
    whose ``step_sizes`` have shape ``(n_entries, n_runs, n_moves)``.
    """

    def test_batched_info_has_per_replica_shapes(self):
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "rw0",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
            MoveKernel(
                "rw1",
                random_walk.build_kernel,
                step_size=0.2,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_runs = 3
        n_walkers = 20
        n_atoms = 1
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)

        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(
                k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
            )
        )(keys)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)
        rng_keys = jax.random.split(jax.random.key(42), n_runs)

        captured: list[dict] = []

        class _Capture:
            def on_iteration(self, iteration, ns_state, info):
                if "adjustment_n_rounds" in info:
                    captured.append(
                        {
                            k: info[k]
                            for k in (
                                "step_sizes_per_move",
                                "acceptance_rates_per_move",
                                "reject_counts_per_move",
                                "adjustment_n_rounds",
                                "adjustment_converged",
                                "adjustment_cap_hits",
                                "adjustment_floor_hits",
                                "adjustment_bracket_detected",
                                "trial_n_evaluations_per_move",
                                "trial_n_grad_evaluations_per_move",
                            )
                        }
                    )

            def on_finish(self, ns_state):
                pass

        run_ns_parallel(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            max_iterations=25,
            n_mcmc_steps=3,
            per_move_fns=per_move_fns,
            move_descriptors=descriptors,
            adjust_interval=10,
            adjust_n_samples=10,
            callbacks=[_Capture()],
        )

        assert len(captured) >= 1
        info = captured[0]
        n_moves = len(descriptors)

        assert jnp.asarray(info["step_sizes_per_move"]).shape == (
            n_runs,
            n_moves,
        )
        assert jnp.asarray(info["acceptance_rates_per_move"]).shape == (
            n_runs,
            n_moves,
        )
        assert jnp.asarray(info["reject_counts_per_move"]).shape == (
            n_runs,
            n_moves,
            4,
        )
        for k in (
            "adjustment_n_rounds",
            "adjustment_converged",
            "adjustment_cap_hits",
            "adjustment_floor_hits",
            "adjustment_bracket_detected",
            "trial_n_evaluations_per_move",
            "trial_n_grad_evaluations_per_move",
        ):
            assert jnp.asarray(info[k]).shape == (n_runs, n_moves), k

    def test_batched_adaptation_logger_round_trip(self, tmp_path):
        """The adaptation log has shape (n_entries, n_runs, n_moves)."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "rw0",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
            MoveKernel(
                "rw1",
                random_walk.build_kernel,
                step_size=0.2,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_runs = 2
        n_walkers = 20
        n_atoms = 1
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(
                k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0
            )
        )(keys)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)
        rng_keys = jax.random.split(jax.random.key(7), n_runs)

        adapt_log_path = tmp_path / "test.adaptation.h5"
        adaptation_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=[d.name for d in descriptors],
            n_runs=n_runs,
        )
        adaptation_cb = AdaptationCallback(adaptation_logger)

        run_ns_parallel(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            max_iterations=30,
            n_mcmc_steps=3,
            per_move_fns=per_move_fns,
            move_descriptors=descriptors,
            adjust_interval=10,
            adjust_n_samples=10,
            callbacks=[adaptation_cb],
        )
        adaptation_logger.close()

        assert adapt_log_path.exists()

        log = AdaptationLogger.read(adapt_log_path)
        n_moves = len(descriptors)
        assert log.n_runs == n_runs
        assert log.n_moves == n_moves
        assert log.iterations.ndim == 1
        n_entries = log.iterations.shape[0]
        # iter-0 baseline row + at least one adjust event.
        assert n_entries >= 2
        assert log.step_sizes.shape == (n_entries, n_runs, n_moves)
        assert log.acceptance_rates.shape == (n_entries, n_runs, n_moves)
        assert log.adjustment_stats is not None
        assert log.adjustment_stats["n_rounds"].shape == (
            n_entries,
            n_runs,
            n_moves,
        )
        assert log.adjustment_stats["reject_reason_counts"].shape == (
            n_entries,
            n_runs,
            n_moves,
            4,
        )
        assert log.n_evaluations is not None
        assert log.n_evaluations.shape == (n_entries, n_runs, n_moves)
        # Row 0 is the iter-0 baseline (n_rounds=0, converged=True).  The
        # adjuster fires before the first ns step, so row 1 is also at
        # iter 0 and differs from the baseline by n_rounds >= 1.
        assert int(log.iterations[0]) == 0
        assert np.all(log.adjustment_stats["n_rounds"][0] == 0)
        assert np.all(log.adjustment_stats["converged"][0])
        assert np.all(log.iterations % 10 == 0)
        assert np.all(log.adjustment_stats["n_rounds"][1:] >= 1)


class TestAdaptationIter0BaselineRow:
    """Check the iter-0 row of the adaptation log without full-auto.

    The row is written even when no bisection ever fires, so the file
    documents the constant step size.  Its adjustment fields say that no
    bisection ran (n_rounds=0, converged=True).
    """

    def test_single_run_no_adjustment_writes_iter0_only(self, tmp_path):
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "rw",
                random_walk.build_kernel,
                step_size=0.25,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_walkers = 12
        types = jnp.zeros((1,), dtype=jnp.int32)
        key = jax.random.key(11)
        key, init_key = jax.random.split(key)
        positions = jax.random.uniform(
            init_key, (n_walkers, 1, 3), minval=-2, maxval=2
        )
        energies = jax.vmap(
            lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        adapt_log_path = tmp_path / "no_full_auto.adaptation.h5"
        adapt_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=["rw"],
            n_runs=1,
        )

        run_ns(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=15,
            n_mcmc_steps=3,
            initial_step_size=0.25,
            # Without per_move_fns there is no adaptation, but the callback
            # still writes the iter-0 baseline row.
            callbacks=[AdaptationCallback(adapt_logger)],
        )
        adapt_logger.close()

        assert adapt_log_path.exists()
        log = AdaptationLogger.read(adapt_log_path)
        assert log.iterations.shape == (1,)
        assert int(log.iterations[0]) == 0
        np.testing.assert_allclose(
            log.step_sizes[0, 0],
            np.array([0.25], dtype=np.float32),
            rtol=1e-5,
        )
        assert log.adjustment_stats is not None
        assert int(log.adjustment_stats["n_rounds"][0, 0, 0]) == 0
        assert bool(log.adjustment_stats["converged"][0, 0, 0]) is True

    def test_multi_run_baseline_per_replica_shape(self, tmp_path):
        """The iter-0 row has the (n_runs, n_moves) shape."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "rw0",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
            MoveKernel(
                "rw1",
                random_walk.build_kernel,
                step_size=0.3,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_runs, n_walkers = 2, 12
        types = jnp.zeros((1,), dtype=jnp.int32)
        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(
                k, (n_walkers, 1, 3), minval=-2, maxval=2
            )
        )(keys)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)
        rng_keys = jax.random.split(jax.random.key(33), n_runs)

        adapt_log_path = tmp_path / "multi.adaptation.h5"
        adapt_logger = AdaptationLogger(
            path=adapt_log_path,
            move_names=["rw0", "rw1"],
            n_runs=n_runs,
        )

        run_ns_parallel(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            max_iterations=10,
            n_mcmc_steps=2,
            initial_step_size=0.4,
            # Without per_move_fns adaptation stays off; only row 0 is written.
            move_descriptors=descriptors,
            callbacks=[AdaptationCallback(adapt_logger)],
        )
        adapt_logger.close()

        log = AdaptationLogger.read(adapt_log_path)
        assert log.n_runs == n_runs
        assert log.iterations.shape == (1,)
        assert log.step_sizes.shape == (1, n_runs, 2)
        np.testing.assert_allclose(
            log.step_sizes[0],
            np.full((n_runs, 2), 0.4, dtype=np.float32),
            rtol=1e-5,
        )


class TestAccRatesCallbackEndToEnd:
    """Check that AccRatesCallback logs every iteration without full-auto."""

    def test_per_iter_logging_no_full_auto(self, tmp_path):
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "rw",
                random_walk.build_kernel,
                step_size=0.25,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_runs, n_walkers = 2, 12
        types = jnp.zeros((1,), dtype=jnp.int32)
        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(
                k, (n_walkers, 1, 3), minval=-2, maxval=2
            )
        )(keys)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)
        rng_keys = jax.random.split(jax.random.key(99), n_runs)

        acc_path = tmp_path / "no_full_auto.acc_rates.h5"
        acc_logger = AccRatesLogger(
            path=acc_path,
            move_names=["rw"],
            n_runs=n_runs,
        )

        max_iter = 12
        run_ns_parallel(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            max_iterations=max_iter,
            n_mcmc_steps=3,
            callbacks=[AccRatesCallback(acc_logger, interval=1)],
        )
        acc_logger.close()

        assert acc_path.exists()
        log = AccRatesLogger.read(acc_path)
        assert log.n_runs == n_runs
        assert log.n_moves == 1
        # n_entries can vary by one with the termination timing.
        n_entries = log.iterations.shape[0]
        assert n_entries >= max_iter - 2
        assert log.n_accepted.shape == (n_entries, n_runs, 1)
        assert log.n_proposed.shape == (n_entries, n_runs, 1)
        assert np.all(log.n_proposed > 0)
        assert np.all(log.n_accepted <= log.n_proposed)


class TestParityNRunsOne:
    """Check run_ns_parallel with a single run against run_ns."""

    def test_n_runs_1_matches_run_ns_no_adaptation(self):
        """Without adaptation both paths give a finite, similar evidence."""
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel("random_walk", random_walk.build_kernel),
            ],
        )

        n_walkers = 20
        n_atoms = 1
        key = jax.random.key(77)
        key, init_key = jax.random.split(key)
        positions = jax.random.uniform(
            init_key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
        )
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        n_iter = 60
        n_mcmc = 5

        result_seq = run_ns(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        rng_keys = jax.random.split(key, 1)
        result_par = run_ns_parallel(
            positions[None],
            types,
            energies[None],
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        log_Z_seq = float(result_seq["log_evidence"])
        log_Z_par = float(result_par["log_evidence"][0])

        assert jnp.isfinite(jnp.array(log_Z_seq))
        assert jnp.isfinite(jnp.array(log_Z_par))
        # The vmapped path draws different random numbers; the measured gap
        # for this seed is 0.2.
        assert abs(log_Z_par - log_Z_seq) < 1.0

    def test_n_runs_1_with_full_auto_bisection(self):
        """run_ns_parallel(n_runs=1) with full-auto bisection completes."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "random_walk",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_walkers = 20
        n_atoms = 1
        key = jax.random.key(99)
        key, init_key = jax.random.split(key)
        positions = jax.random.uniform(
            init_key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
        )
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        rng_keys = jax.random.split(key, 1)
        result = run_ns_parallel(
            positions[None],
            types,
            energies[None],
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            max_iterations=50,
            n_mcmc_steps=5,
            initial_step_size=0.3,
            per_move_fns=per_move_fns,
            move_descriptors=descriptors,
            adjust_interval=15,
            adjust_n_samples=10,
        )
        assert result["n_runs"] == 1
        assert jnp.isfinite(result["log_evidence"][0])
        assert int(result["n_dead"][0]) > 0


class TestParallelFullAuto:
    """Check run_ns_parallel output shapes with full-auto adaptation."""

    def test_run_ns_parallel_full_auto_correct_shapes(self):
        """The outputs have a leading n_runs axis."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "random_walk",
                random_walk.build_kernel,
                step_size=0.1,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_runs = 2
        n_walkers = 15
        n_atoms = 1
        keys = jax.random.split(jax.random.key(11), n_runs + 1)
        positions = jax.vmap(
            lambda k: jax.random.uniform(
                k, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
            )
        )(keys[:n_runs])
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)

        result = run_ns_parallel(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=keys[:n_runs],
            max_iterations=40,
            n_mcmc_steps=5,
            initial_step_size=0.3,
            per_move_fns=per_move_fns,
            move_descriptors=descriptors,
            adjust_interval=10,
            adjust_n_samples=10,
        )

        assert result["n_runs"] == n_runs
        assert result["log_evidence"].shape == (n_runs,)
        assert jnp.all(jnp.isfinite(result["log_evidence"]))
        assert jnp.all(result["n_dead"] > 0)


class TestMoveRejectReasons:
    """Check the move_reject_reasons entry of the info dict."""

    @pytest.mark.parametrize("n_moves", [2, 3])
    def test_one_reject_reason_set_per_move(self, n_moves):
        """Adjust iterations carry one reject-reason set per move."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                f"rw{i}",
                random_walk.build_kernel,
                step_size=0.1 * (i + 1),
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            )
            for i in range(n_moves)
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_walkers = 20
        key = jax.random.key(11)
        key, init_key = jax.random.split(key)
        positions = 0.5 * jax.random.normal(init_key, (n_walkers, 1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        captured = []

        class _Capture:
            def on_iteration(self, iteration, ns_state, info):
                if "move_reject_reasons" in info:
                    captured.append(dict(info))

            def on_finish(self, ns_state):
                pass

        run_ns(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=30,
            n_mcmc_steps=3,
            per_move_fns=per_move_fns,
            move_descriptors=descriptors,
            adjust_interval=10,
            adjust_n_samples=10,
            callbacks=[_Capture()],
        )

        assert len(captured) > 0
        info = captured[0]
        mrr = info["move_reject_reasons"]
        assert len(mrr) == len(info["move_names"]) == n_moves
        # random_walk only rejects on energy.
        assert all(reasons == frozenset({"energy"}) for reasons in mrr)
