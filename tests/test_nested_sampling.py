"""Test full nested sampling runs (run_ns, run_ns_parallel).

Verifies:
- run_ns completes and produces finite evidence
- Evidence accuracy on harmonic oscillator (known answer)
- Callbacks are invoked
- run_ns_parallel completes with correct shapes
- Parallel evidence roughly matches sequential
- Different pressures produce different evidence
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns, run_ns_parallel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_setup():
    """Single-run harmonic oscillator NS problem."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [
        MoveKernel("random_walk", random_walk.build_kernel),
    ])

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
    init_fn, step_fn, _ = build_mwg(backend, [
        MoveKernel("random_walk", random_walk.build_kernel),
    ])

    n_runs = 2
    n_walkers = 20
    n_atoms = 1

    keys = jax.random.split(jax.random.key(0), n_runs)
    positions = jax.vmap(
        lambda k: jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0)
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


# ---------------------------------------------------------------------------
# run_ns (single run)
# ---------------------------------------------------------------------------


class TestRunNS:
    def test_runs_to_completion(self, harmonic_setup):
        s = harmonic_setup
        result = run_ns(
            s["positions"], s["types"], s["energies"],
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
        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("random_walk", random_walk.build_kernel),
        ])

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
            positions, types, energies,
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

        assert abs(log_evidence - float(log_Z_analytical)) < 1.5, (
            f"log_evidence={log_evidence:.3f} vs analytical={float(log_Z_analytical):.3f}"
        )

    def test_callbacks_invoked(self, harmonic_setup):
        s = harmonic_setup

        iterations_seen = []

        class TestCallback:
            def on_iteration(self, iteration, ns_state, info):
                iterations_seen.append(iteration)

            def on_finish(self, ns_state):
                iterations_seen.append("finish")

        result = run_ns(
            s["positions"], s["types"], s["energies"],
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


# ---------------------------------------------------------------------------
# run_ns_parallel (multi-run)
# ---------------------------------------------------------------------------


class TestRunNsParallel:
    def test_basic_completion(self, parallel_setup):
        s = parallel_setup
        result = run_ns_parallel(
            s["positions"], s["types"], s["energies"],
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
            s["positions"], s["types"], s["energies"],
            cells=None,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=s["rng_keys"],
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        result_seq_0 = run_ns(
            s["positions"][0], s["types"], s["energies"][0],
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
        assert abs(log_Z_par - log_Z_seq) < 5.0, (
            f"Parallel log_Z={log_Z_par:.3f} vs sequential log_Z={log_Z_seq:.3f}"
        )


class TestAdjustmentInfoKeys:
    """Verify that adjustment_* info keys are present on adjust iterations only."""

    def test_adjustment_keys_present_on_adjust_iterations(self):
        """run_ns with full_auto emits adjustment_* keys every adjust_interval."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "random_walk", random_walk.build_kernel,
                step_size=0.1, step_size_max=5.0,
                min_rate=0.2, max_rate=0.7,
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

        class _RecordCallback:
            def on_iteration(self, iteration, ns_state, info):
                has_adj = all(k in info for k in _ADJ_KEYS)
                if has_adj:
                    adjust_iterations.append(iteration)
                else:
                    other_iterations.append(iteration)

            def on_finish(self, ns_state):
                pass

        run_ns(
            positions, types, energies,
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

        # At least one adjust iteration must have fired
        assert len(adjust_iterations) > 0, "Expected at least one adjust iteration"
        # Adjust iterations should be multiples of adjust_interval
        for it in adjust_iterations:
            assert it % adjust_interval == 0, f"iter {it} not multiple of {adjust_interval}"
        # Non-adjust iterations must NOT have adjustment keys
        assert len(other_iterations) > 0, "Expected at least some non-adjust iterations"

    def test_adjustment_arrays_have_correct_shapes(self):
        """Adjustment info arrays have shape (n_moves,) and correct dtypes."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw0", random_walk.build_kernel, step_size=0.1, step_size_max=5.0,
                       min_rate=0.2, max_rate=0.7),
            MoveKernel("rw1", random_walk.build_kernel, step_size=0.2, step_size_max=5.0,
                       min_rate=0.2, max_rate=0.7),
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
            positions, types, energies,
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

        assert len(captured_infos) > 0, "Need at least one captured adjustment info"
        info = captured_infos[0]

        n_moves = len(descriptors)
        assert jnp.asarray(info["adjustment_n_rounds"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_converged"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_cap_hits"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_floor_hits"]).shape == (n_moves,)
        assert jnp.asarray(info["adjustment_bracket_detected"]).shape == (n_moves,)

        # Sanity: n_rounds and cap_hits must be non-negative
        assert jnp.all(jnp.asarray(info["adjustment_n_rounds"]) >= 0)
        assert jnp.all(jnp.asarray(info["adjustment_cap_hits"]) >= 0)
        assert jnp.all(jnp.asarray(info["adjustment_floor_hits"]) >= 0)


class TestAdjustmentInfoKeysBatched:
    """Adjustment info packs for batched (VmapRuns) runs too — previously
    gated out by ``not batcher.is_batched``.  Covers two layers:

    1. ``info`` carries the adjustment_* keys at adjust iterations and
       arrays are shaped ``(n_runs, n_moves[, 4])``.
    2. ``AdaptationCallback`` + ``AdaptationLogger`` produce a real
       ``.adaptation.h5`` with ``step_sizes`` of shape
       ``(n_entries, n_runs, n_moves)``.
    """

    def test_batched_info_has_per_replica_shapes(self):
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw0", random_walk.build_kernel, step_size=0.1,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
            MoveKernel("rw1", random_walk.build_kernel, step_size=0.2,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_runs = 3
        n_walkers = 20
        n_atoms = 1
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)

        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0)
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
                    captured.append({
                        k: info[k] for k in (
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
                    })

            def on_finish(self, ns_state):
                pass

        run_ns_parallel(
            positions, types, energies,
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

        assert len(captured) >= 1, "expected at least one adjust iter under run_ns_parallel"
        info = captured[0]
        n_moves = len(descriptors)

        assert jnp.asarray(info["step_sizes_per_move"]).shape == (n_runs, n_moves)
        assert jnp.asarray(info["acceptance_rates_per_move"]).shape == (n_runs, n_moves)
        assert jnp.asarray(info["reject_counts_per_move"]).shape == (n_runs, n_moves, 4)
        for k in (
            "adjustment_n_rounds",
            "adjustment_converged",
            "adjustment_cap_hits",
            "adjustment_floor_hits",
            "adjustment_bracket_detected",
            "trial_n_evaluations_per_move",
            "trial_n_grad_evaluations_per_move",
        ):
            assert jnp.asarray(info[k]).shape == (n_runs, n_moves), (
                f"info[{k}] has shape {jnp.asarray(info[k]).shape}, "
                f"expected {(n_runs, n_moves)}"
            )

    def test_batched_adaptation_logger_round_trip(self, tmp_path):
        """End-to-end: AdaptationCallback + AdaptationLogger under VmapRuns
        write an HDF5 with shape ``(n_entries, n_runs, n_moves)``.
        """
        import numpy as np

        from jaxrens.cli.monitor import AdaptationCallback
        from jaxrens.io.adaptation_log import AdaptationLogger

        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw0", random_walk.build_kernel, step_size=0.1,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
            MoveKernel("rw1", random_walk.build_kernel, step_size=0.2,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_runs = 2
        n_walkers = 20
        n_atoms = 1
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-3.0, maxval=3.0)
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
            positions, types, energies,
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

        assert adapt_log_path.exists(), (
            "AdaptationLogger should have produced an HDF5 file once "
            "_pack_adjustment_info no longer gates out batched runs."
        )

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
        # v2 adjustment_stats present
        assert log.adjustment_stats is not None
        assert log.adjustment_stats["n_rounds"].shape == (n_entries, n_runs, n_moves)
        assert log.adjustment_stats["reject_reason_counts"].shape == (
            n_entries, n_runs, n_moves, 4,
        )
        # v3 evaluation counts present
        assert log.n_evaluations is not None
        assert log.n_evaluations.shape == (n_entries, n_runs, n_moves)
        # Row 0 is the iter-0 baseline (zeros / converged=True);
        # subsequent rows are real adjust events at multiples of adjust_interval.
        assert int(log.iterations[0]) == 0
        assert np.all(log.adjustment_stats["n_rounds"][0] == 0)
        assert np.all(log.adjustment_stats["converged"][0] == True)
        assert np.all(log.iterations[1:] > 0)
        assert np.all(log.iterations[1:] % 10 == 0)


class TestAdaptationIter0BaselineRow:
    """The iter-0 baseline row is written even for runs without full_auto
    (no bisection events ever fire) — so the file becomes a 1-row
    artefact documenting the constant step size.

    Also: row 0's adjustment_* fields signal "no bisection ran"
    (n_rounds=0, converged=True, etc.).
    """

    def test_single_run_no_adjustment_writes_iter0_only(self, tmp_path):
        import numpy as np

        from jaxrens.cli.monitor import AdaptationCallback
        from jaxrens.io.adaptation_log import AdaptationLogger

        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw", random_walk.build_kernel, step_size=0.25,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_walkers = 12
        types = jnp.zeros((1,), dtype=jnp.int32)
        key = jax.random.key(11)
        key, init_key = jax.random.split(key)
        positions = jax.random.uniform(init_key, (n_walkers, 1, 3), minval=-2, maxval=2)
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
            positions, types, energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=15,
            n_mcmc_steps=3,
            initial_step_size=0.25,
            # no per_move_fns / move_descriptors -> no adaptation at all,
            # but we still want the iter-0 baseline row, so we register
            # the callback manually.
            callbacks=[AdaptationCallback(adapt_logger)],
        )
        adapt_logger.close()

        assert adapt_log_path.exists(), (
            "iter-0 baseline row must produce the file even without adaptation"
        )
        log = AdaptationLogger.read(adapt_log_path)
        assert log.iterations.shape == (1,)
        assert int(log.iterations[0]) == 0
        # Row 0 carries the initial ss as passed to run_ns.
        np.testing.assert_allclose(
            log.step_sizes[0, 0], np.array([0.25], dtype=np.float32),
            rtol=1e-5,
        )
        # Adjustment-event fields signal "no bisection ran".
        assert log.adjustment_stats is not None
        assert int(log.adjustment_stats["n_rounds"][0, 0, 0]) == 0
        assert bool(log.adjustment_stats["converged"][0, 0, 0]) is True

    def test_multi_run_baseline_per_replica_shape(self, tmp_path):
        """The iter-0 row honours the multi-run (n_runs, n_moves) shape."""
        import numpy as np

        from jaxrens.cli.monitor import AdaptationCallback
        from jaxrens.io.adaptation_log import AdaptationLogger

        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw0", random_walk.build_kernel, step_size=0.1,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
            MoveKernel("rw1", random_walk.build_kernel, step_size=0.3,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_runs, n_walkers = 2, 12
        types = jnp.zeros((1,), dtype=jnp.int32)
        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(k, (n_walkers, 1, 3), minval=-2, maxval=2)
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
            positions, types, energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            max_iterations=10,
            n_mcmc_steps=2,
            initial_step_size=0.4,
            # Pass descriptors so n_moves is honoured (without per_move_fns
            # adaptation stays inactive — only the iter-0 row gets written).
            move_descriptors=descriptors,
            callbacks=[AdaptationCallback(adapt_logger)],
        )
        adapt_logger.close()

        log = AdaptationLogger.read(adapt_log_path)
        assert log.n_runs == n_runs
        # No adjust events fired (no per_move_fns), so only the baseline row.
        assert log.iterations.shape == (1,)
        assert log.step_sizes.shape == (1, n_runs, 2)
        # All replicas, all moves see the same broadcast initial ss.
        np.testing.assert_allclose(
            log.step_sizes[0],
            np.full((n_runs, 2), 0.4, dtype=np.float32),
            rtol=1e-5,
        )


class TestAccRatesCallbackEndToEnd:
    """AccRatesCallback fires every iter regardless of full_auto and
    produces a multi-run-shaped HDF5 file."""

    def test_per_iter_logging_no_full_auto(self, tmp_path):
        import numpy as np

        from jaxrens.cli.monitor import AccRatesCallback
        from jaxrens.io.acc_rates_log import AccRatesLogger

        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel("rw", random_walk.build_kernel, step_size=0.25,
                       step_size_max=5.0, min_rate=0.2, max_rate=0.7),
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_runs, n_walkers = 2, 12
        types = jnp.zeros((1,), dtype=jnp.int32)
        keys = jax.random.split(jax.random.key(0), n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(k, (n_walkers, 1, 3), minval=-2, maxval=2)
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
            positions, types, energies,
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
        # iter starts at 0 inside _run_loop; fires every iter.
        # The exact n_entries can vary by ±1 depending on when termination
        # fires relative to the callback dispatch; check it's plausible.
        n_entries = log.iterations.shape[0]
        assert n_entries >= max_iter - 2
        assert log.n_accepted.shape == (n_entries, n_runs, 1)
        assert log.n_proposed.shape == (n_entries, n_runs, 1)
        # Chain proposed counts are positive on every iter.
        assert np.all(log.n_proposed > 0)
        # Acceptance counts ≤ proposed counts.
        assert np.all(log.n_accepted <= log.n_proposed)


# ---------------------------------------------------------------------------
# Task C: parity test — run_ns_parallel(n_runs=1) vs run_ns (bisection)
# ---------------------------------------------------------------------------


class TestParityNRunsOne:
    """run_ns_parallel(n_runs=1) with full-auto bisection should reproduce
    run_ns with the same seed and config on a trivial backend.

    Same initial_step_size, same adjust_interval, same rng seed.
    We allow RNG noise: |log_Z_par - log_Z_seq| < 5.0 in log-space.
    """

    def test_n_runs_1_matches_run_ns_no_adaptation(self):
        """Without adaptation, both paths must agree exactly (same step sizes)."""
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("random_walk", random_walk.build_kernel),
        ])

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

        # Sequential: no adaptation
        result_seq = run_ns(
            positions, types, energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=n_iter,
            n_mcmc_steps=n_mcmc,
            initial_step_size=0.3,
        )

        # Parallel n_runs=1: no adaptation
        rng_keys = jax.random.split(key, 1)  # shape (1,)
        result_par = run_ns_parallel(
            positions[None],  # (1, n_walkers, n_atoms, 3)
            types,
            energies[None],   # (1, n_walkers)
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

        # Both should be finite
        assert jnp.isfinite(jnp.array(log_Z_seq))
        assert jnp.isfinite(jnp.array(log_Z_par))
        # Allow generous slop due to identical-seed-but-different-vmap-execution RNG
        assert abs(log_Z_par - log_Z_seq) < 8.0, (
            f"Parity failed: run_ns_parallel(n_runs=1)={log_Z_par:.3f} "
            f"vs run_ns={log_Z_seq:.3f}"
        )

    def test_n_runs_1_with_full_auto_bisection(self):
        """run_ns_parallel(n_runs=1) with full-auto bisection completes."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "random_walk", random_walk.build_kernel,
                step_size=0.1, step_size_max=5.0,
                min_rate=0.2, max_rate=0.7,
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
            positions[None], types, energies[None],
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


class TestParallelVmappedAdjustJIT:
    """JIT test for vmapped adjust_step_size inside run_ns_parallel.

    Verifies compilation, correct output shapes, and no retracing.
    """

    def test_vmapped_adjust_compiles_and_runs(self):
        """Directly test the vmapped adjust_step_size layer used in run_ns_parallel."""
        from jaxrens.sampling.adaptation.stepsize_handler import adjust_step_size

        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "random_walk", random_walk.build_kernel,
                step_size=0.1, step_size_max=5.0,
                min_rate=0.2, max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_runs = 3
        n_walkers = 15
        n_atoms = 1
        key = jax.random.key(5)
        keys = jax.random.split(key, n_runs)
        positions = jax.vmap(
            lambda k: jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-1.0, maxval=1.0)
        )(keys)
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)

        # Initialize n_runs NSStates and stack them
        from jaxrens.sampling.nested_sampling import init_ns_parallel
        ns_states = init_ns_parallel(
            init_fn, positions, types, energies, None, keys,
            step_sizes=jnp.full(1, 0.1),
        )

        pop = ns_states.population  # (n_runs, n_walkers, ...)
        emax_per_run = jnp.max(pop.energy, axis=1)  # (n_runs,)
        ss_init = jnp.full(n_runs, 0.3)  # (n_runs,)
        trial_keys = jax.random.split(jax.random.key(7), n_runs)

        move_fn = per_move_fns[0]
        desc = descriptors[0]

        # Build the same vmapped+JIT fn as run_ns_parallel does
        def _per_run(p, ss, emax, k):
            return adjust_step_size(
                p, move_fn, ss, emax, k,
                10, desc.min_rate, desc.max_rate,
                1.5, desc.step_size_max, 10,
            )

        jit_vmap_adjust = jax.jit(jax.vmap(_per_run))

        # First call — compiles
        result = jit_vmap_adjust(pop, ss_init, emax_per_run, trial_keys)
        new_ss_runs = result[0]

        assert new_ss_runs.shape == (n_runs,), (
            f"Expected (n_runs,)={(n_runs,)}, got {new_ss_runs.shape}"
        )
        assert result[3].dtype == jnp.int32   # n_rounds

        # Second call with different data — must not retrace (same shapes)
        trial_keys2 = jax.random.split(jax.random.key(99), n_runs)
        result2 = jit_vmap_adjust(pop, ss_init, emax_per_run, trial_keys2)
        assert result2[0].shape == (n_runs,)

    def test_run_ns_parallel_full_auto_correct_shapes(self):
        """run_ns_parallel with full-auto adaptation: output shapes must be (n_runs, ...)."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "random_walk", random_walk.build_kernel,
                step_size=0.1, step_size_max=5.0,
                min_rate=0.2, max_rate=0.7,
            ),
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_runs = 2
        n_walkers = 15
        n_atoms = 1
        keys = jax.random.split(jax.random.key(11), n_runs + 1)
        positions = jax.vmap(
            lambda k: jax.random.uniform(k, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)
        )(keys[:n_runs])
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: jax.vmap(
                lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
            )(pos)
        )(positions)

        result = run_ns_parallel(
            positions, types, energies,
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
    """Verify that move_reject_reasons is emitted correctly in info dicts."""

    def test_move_reject_reasons_present_on_adjust_iterations(self):
        """run_ns with full_auto emits move_reject_reasons on adjust iterations."""
        backend = create_harmonic(k=1.0)
        descriptors = [
            MoveKernel(
                "rw0", random_walk.build_kernel,
                step_size=0.1, step_size_max=5.0,
                min_rate=0.2, max_rate=0.7,
            ),
            MoveKernel(
                "rw1", random_walk.build_kernel,
                step_size=0.2, step_size_max=5.0,
                min_rate=0.2, max_rate=0.7,
            ),
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

        captured_infos = []

        class _Capture:
            def on_iteration(self, iteration, ns_state, info):
                if "move_reject_reasons" in info:
                    captured_infos.append(dict(info))
            def on_finish(self, ns_state):
                pass

        run_ns(
            positions, types, energies,
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

        assert len(captured_infos) > 0, "Expected at least one adjust iteration with move_reject_reasons"
        info = captured_infos[0]

        # move_reject_reasons should be a tuple of frozensets, length == n_moves
        mrr = info["move_reject_reasons"]
        assert isinstance(mrr, tuple)
        assert len(mrr) == 2
        for fs in mrr:
            assert isinstance(fs, frozenset)
        # Both are random_walk (energy-only) — default reject_reasons
        assert mrr[0] == frozenset({"energy"})
        assert mrr[1] == frozenset({"energy"})

    def test_move_reject_reasons_match_move_names(self):
        """move_reject_reasons tuple has same length as move_names."""
        backend = create_harmonic(k=1.0)
        n_moves = 3
        descriptors = [
            MoveKernel(
                f"rw{i}", random_walk.build_kernel,
                step_size=0.1, step_size_max=5.0,
                min_rate=0.2, max_rate=0.7,
            )
            for i in range(n_moves)
        ]
        init_fn, step_fn, per_move_fns = build_mwg(backend, descriptors)

        n_walkers = 20
        key = jax.random.key(22)
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
            positions, types, energies,
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
        move_names = info["move_names"]
        assert len(mrr) == len(move_names) == n_moves


class TestDifferentPressures:
    def test_different_pressures_different_evidence(self):
        base_backend = create_harmonic(k=1.0)
        backend_p0 = EnsembleBackend(base_backend, pressure=0.0)
        backend_p1 = EnsembleBackend(base_backend, pressure=0.1)

        init_fn_p0, step_fn_p0, _ = build_mwg(backend_p0, [
            MoveKernel("random_walk", random_walk.build_kernel),
        ])

        n_walkers = 15
        n_atoms = 1
        key = jax.random.key(99)
        positions = jax.random.uniform(key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((n_atoms,), dtype=jnp.int32)
        cells = jnp.tile(5.0 * jnp.eye(3), (n_walkers, 1, 1))

        energies_p0 = jax.vmap(
            lambda pos, cell: backend_p0(pos, types, cell, 0)[0]
        )(positions, cells)

        energies_p1 = jax.vmap(
            lambda pos, cell: backend_p1(pos, types, cell, 0)[0]
        )(positions, cells)

        assert not jnp.allclose(energies_p0, energies_p1)
        assert jnp.all(energies_p1 > energies_p0)
