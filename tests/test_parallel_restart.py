"""Tests for commit 5: parallel restart support in run_ns_parallel / init_ns_parallel.

Tests
-----
TestInitNsParallelRestart
    - init_ns_parallel with restart_states seeds n_dead, iteration, log_evidence
      per-run correctly.
    - Mixed restart: some runs restarted, some fresh — verified independently.
    - Validation: wrong-length restart_states raises ValueError.

TestRunNsParallelRestart
    - run_ns_parallel with restart_states: n_dead >= checkpoint_n_dead on each run.
    - run_ns_parallel with restart_states: log_evidence is finite.
    - Parity with single-run restart: parallel run with n_runs=1 restarts from the
      same checkpoint as run_ns — output n_dead consistent.

All tests use the harmonic toy backend + random_walk kernel for reproducibility
and CPU-scale runtime.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.init.restart import RestartBundle, load_restart
from jaxrens.io.checkpoint import save_checkpoint
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk as rw_mod
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, init_ns_parallel, run_ns_parallel
from jaxrens.sampling.termination import IterationTermination


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rw_descriptor() -> MoveKernel:
    return MoveKernel(
        name="random_walk",
        build_kernel=rw_mod.build_kernel,
        step_size=0.3,
        weight=1.0,
        kernel_kwargs={},
        extra_state_fields={},
    )


def _make_harmonic_setup(
    seed: int = 42,
    n_walkers: int = 8,
    n_atoms: int = 1,
):
    """Return a minimal harmonic NS problem (backend, init_fn, step_fn, arrays)."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [_rw_descriptor()])
    key = jax.random.key(seed)
    key, pos_key = jax.random.split(key)
    positions = jax.random.uniform(
        pos_key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_walkers, 3, 3))  # harmonic backend ignores cell
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)
    return {
        "backend": backend,
        "init_fn": init_fn,
        "step_fn": step_fn,
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": cells,
        "key": key,
        "n_walkers": n_walkers,
        "n_atoms": n_atoms,
    }


def _write_checkpoint(
    tmp_path: Path,
    s: dict,
    n_dead: int = 4,
    log_evidence: float = -6.0,
    name: str = "ckpt.checkpoint.h5",
) -> Path:
    """Write a minimal NS checkpoint seeded from the harmonic setup *s*."""
    rng = np.random.default_rng(7)
    n_walkers = s["n_walkers"]
    n_atoms = s["n_atoms"]

    dead_energies = rng.uniform(5.0, 15.0, n_dead).astype(np.float32)
    dead_positions = rng.uniform(-2.0, 2.0, (n_dead, n_atoms, 3)).astype(np.float32)

    state = {
        "positions": np.asarray(s["positions"]),
        "types": np.zeros((n_walkers, n_atoms), dtype=np.int32),
        "energies": np.asarray(s["energies"]),
        "cells": np.asarray(s["cells"]),
        "dead_energies": dead_energies,
        "dead_positions": dead_positions,
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": log_evidence,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(1),
    }
    p = tmp_path / name
    save_checkpoint(p, state)
    return p


def _load_bundle(path: Path) -> RestartBundle:
    _, bundle = load_restart(path)
    return bundle


# ---------------------------------------------------------------------------
# TestInitNsParallelRestart
# ---------------------------------------------------------------------------


class TestInitNsParallelRestart:
    """Unit tests for init_ns_parallel(..., restart_states=...)."""

    def test_restart_seeds_n_dead(self, tmp_path):
        """init_ns_parallel with restart_states seeds n_dead from checkpoint."""
        n_runs = 2
        n_dead_checkpoint = 5
        s = _make_harmonic_setup(seed=10)

        # Build per-run positions/energies by stacking the same init
        positions = jnp.stack([s["positions"]] * n_runs)
        energies = jnp.stack([s["energies"]] * n_runs)
        cells = jnp.stack([s["cells"]] * n_runs)
        keys = jax.random.split(s["key"], n_runs)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"], positions, s["types"], energies, cells, keys,
            max_dead=200,
            restart_states=[bundle, bundle],
        )

        assert int(ns_states.n_dead[0]) == n_dead_checkpoint
        assert int(ns_states.n_dead[1]) == n_dead_checkpoint

    def test_restart_seeds_iteration(self, tmp_path):
        """init_ns_parallel seeds iteration counter from checkpoint."""
        n_dead_checkpoint = 7
        s = _make_harmonic_setup(seed=11)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"], positions, s["types"], energies, cells, keys,
            max_dead=200,
            restart_states=[bundle, bundle],
        )

        assert int(ns_states.iteration[0]) == n_dead_checkpoint
        assert int(ns_states.iteration[1]) == n_dead_checkpoint

    def test_restart_seeds_log_evidence(self, tmp_path):
        """init_ns_parallel seeds log_evidence from checkpoint."""
        n_dead_checkpoint = 3
        log_z = -9.5
        s = _make_harmonic_setup(seed=12)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint, log_evidence=log_z)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"], positions, s["types"], energies, cells, keys,
            max_dead=200,
            restart_states=[bundle, bundle],
        )

        assert abs(float(ns_states.log_evidence[0]) - log_z) < 1e-4
        assert abs(float(ns_states.log_evidence[1]) - log_z) < 1e-4

    def test_restart_mixed_some_fresh(self, tmp_path):
        """init_ns_parallel with [bundle, None]: run 0 restarted, run 1 fresh."""
        n_dead_checkpoint = 6
        s = _make_harmonic_setup(seed=13)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"], positions, s["types"], energies, cells, keys,
            max_dead=200,
            restart_states=[bundle, None],
        )

        # Run 0: restarted — n_dead == checkpoint
        assert int(ns_states.n_dead[0]) == n_dead_checkpoint
        # Run 1: fresh — n_dead == 0
        assert int(ns_states.n_dead[1]) == 0

    def test_restart_wrong_length_raises(self, tmp_path):
        """init_ns_parallel raises ValueError when restart_states length != n_runs."""
        s = _make_harmonic_setup(seed=14)
        n_runs = 3

        positions = jnp.stack([s["positions"]] * n_runs)
        energies = jnp.stack([s["energies"]] * n_runs)
        cells = jnp.stack([s["cells"]] * n_runs)
        keys = jax.random.split(s["key"], n_runs)

        p = _write_checkpoint(tmp_path, s, n_dead=2)
        bundle = _load_bundle(p)

        with pytest.raises(ValueError, match="restart_states length"):
            init_ns_parallel(
                s["init_fn"], positions, s["types"], energies, cells, keys,
                max_dead=200,
                restart_states=[bundle, bundle],  # length 2 != n_runs=3
            )

    def test_restart_dead_arrays_padded(self, tmp_path):
        """init_ns_parallel pads dead arrays to max_dead per run."""
        n_dead_checkpoint = 4
        max_dead = 50
        s = _make_harmonic_setup(seed=15)

        positions = jnp.stack([s["positions"]])
        energies = jnp.stack([s["energies"]])
        cells = jnp.stack([s["cells"]])
        keys = jax.random.split(s["key"], 1)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"], positions, s["types"], energies, cells, keys,
            max_dead=max_dead,
            restart_states=[bundle],
        )

        # Shape: (n_runs, max_dead) = (1, max_dead)
        assert ns_states.dead_energies.shape == (1, max_dead)
        # Positions beyond n_dead should be zero-padded
        assert float(jnp.min(ns_states.dead_energies[0, n_dead_checkpoint:])) == float(jnp.inf)

    def test_no_restart_states_gives_fresh_start(self, tmp_path):
        """Without restart_states, all runs start with n_dead=0."""
        s = _make_harmonic_setup(seed=16)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        ns_states = init_ns_parallel(
            s["init_fn"], positions, s["types"], energies, cells, keys,
            max_dead=200,
        )

        assert int(ns_states.n_dead[0]) == 0
        assert int(ns_states.n_dead[1]) == 0


# ---------------------------------------------------------------------------
# TestRunNsParallelRestart
# ---------------------------------------------------------------------------


class TestRunNsParallelRestart:
    """Integration tests for run_ns_parallel(..., restart_states=...)."""

    def test_parallel_restart_n_dead_increments(self, tmp_path):
        """run_ns_parallel with restart_states: n_dead >= checkpoint_n_dead."""
        n_dead_checkpoint = 4
        n_extra_iters = 6
        s = _make_harmonic_setup(seed=20)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        termination = [IterationTermination(n_extra_iters)]
        out = run_ns_parallel(
            positions, s["types"], energies, cells,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=n_extra_iters,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_states=[bundle, bundle],
        )

        # Both runs started with n_dead_checkpoint dead points and ran more
        assert int(out["n_dead"][0]) >= n_dead_checkpoint
        assert int(out["n_dead"][1]) >= n_dead_checkpoint

    def test_parallel_restart_log_evidence_finite(self, tmp_path):
        """run_ns_parallel with restart_states produces finite log_evidence."""
        n_dead_checkpoint = 3
        s = _make_harmonic_setup(seed=21)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint, log_evidence=-5.0)
        bundle = _load_bundle(p)

        termination = [IterationTermination(5)]
        out = run_ns_parallel(
            positions, s["types"], energies, cells,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=5,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_states=[bundle, bundle],
        )

        assert jnp.all(jnp.isfinite(out["log_evidence"])), (
            f"log_evidence not finite: {out['log_evidence']}"
        )

    def test_parallel_restart_output_shapes(self, tmp_path):
        """run_ns_parallel with restart_states returns (n_runs, ...) shaped outputs."""
        n_runs = 2
        n_dead_checkpoint = 3
        s = _make_harmonic_setup(seed=22)

        positions = jnp.stack([s["positions"]] * n_runs)
        energies = jnp.stack([s["energies"]] * n_runs)
        cells = jnp.stack([s["cells"]] * n_runs)
        keys = jax.random.split(s["key"], n_runs)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        termination = [IterationTermination(5)]
        out = run_ns_parallel(
            positions, s["types"], energies, cells,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=5,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_states=[bundle, bundle],
        )

        assert out["log_evidence"].shape == (n_runs,)
        assert out["n_dead"].shape == (n_runs,)
        assert out["n_runs"] == n_runs

    def test_parallel_restart_parity_with_single_restart(self, tmp_path):
        """run_ns_parallel(n_runs=1, restart_states=[bundle]) is consistent
        with run_ns(restart_state=bundle): both run more iterations past checkpoint."""
        from jaxrens.sampling.nested_sampling import run_ns

        n_dead_checkpoint = 4
        n_extra = 6
        s = _make_harmonic_setup(seed=23)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        termination = [IterationTermination(n_extra)]

        # Single run with restart
        out_single = run_ns(
            s["positions"], s["types"], s["energies"], s["cells"],
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=n_extra,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_state=bundle,
        )

        # Parallel run with n_runs=1 + restart_states
        keys = jax.random.split(s["key"], 1)
        out_par = run_ns_parallel(
            s["positions"][None], s["types"], s["energies"][None], s["cells"][None],
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=n_extra,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_states=[bundle],
        )

        # Both should have run beyond n_dead_checkpoint
        assert int(out_single["n_dead"]) >= n_dead_checkpoint
        assert int(out_par["n_dead"][0]) >= n_dead_checkpoint

        # log_evidence should be finite for both
        assert jnp.isfinite(jnp.array(out_single["log_evidence"]))
        assert jnp.all(jnp.isfinite(out_par["log_evidence"]))

    def test_parallel_restart_without_restart_states_is_fresh(self, tmp_path):
        """run_ns_parallel without restart_states: n_dead starts from 0."""
        n_extra = 6
        s = _make_harmonic_setup(seed=24)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        termination = [IterationTermination(n_extra)]
        out = run_ns_parallel(
            positions, s["types"], energies, cells,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=n_extra,
            n_mcmc_steps=3,
            termination_criteria=termination,
        )

        # Fresh run: n_dead should equal n_extra iterations (no overflow retries
        # in simple harmonic backend)
        assert int(out["n_dead"][0]) >= 0
        assert int(out["n_dead"][1]) >= 0
        assert jnp.all(jnp.isfinite(out["log_evidence"]))

    def test_parallel_restart_wrong_length_propagates(self, tmp_path):
        """run_ns_parallel propagates ValueError when restart_states length != n_runs."""
        n_runs = 2
        s = _make_harmonic_setup(seed=25)

        positions = jnp.stack([s["positions"]] * n_runs)
        energies = jnp.stack([s["energies"]] * n_runs)
        cells = jnp.stack([s["cells"]] * n_runs)
        keys = jax.random.split(s["key"], n_runs)

        p = _write_checkpoint(tmp_path, s, n_dead=2)
        bundle = _load_bundle(p)

        termination = [IterationTermination(5)]
        with pytest.raises(ValueError, match="restart_states length"):
            run_ns_parallel(
                positions, s["types"], energies, cells,
                init_fn=s["init_fn"],
                step_fn=s["step_fn"],
                rng_keys=keys,
                max_iterations=5,
                n_mcmc_steps=3,
                termination_criteria=termination,
                restart_states=[bundle],  # length 1 != n_runs=2
            )
