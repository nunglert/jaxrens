"""Tests for restart support in init_ns_parallel and run_ns_parallel."""

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
from jaxrens.sampling.nested_sampling import (
    init_ns_parallel,
    run_ns,
    run_ns_parallel,
)
from jaxrens.sampling.termination import IterationTermination


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
    """Return a small harmonic NS problem (functions and arrays)."""
    backend = create_harmonic(k=1.0)
    init_fn, step_fn, _ = build_mwg(backend, [_rw_descriptor()])
    key = jax.random.key(seed)
    key, pos_key = jax.random.split(key)
    positions = jax.random.uniform(
        pos_key, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_walkers, 3, 3))
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
    step_sizes: np.ndarray | None = None,
) -> Path:
    """Write a small NS checkpoint seeded from the harmonic setup ``s``.

    ``step_sizes`` of shape ``(K, n_moves)`` is written when given.  ``None``
    emulates a legacy checkpoint without step sizes.
    """
    rng = np.random.default_rng(7)
    n_walkers = s["n_walkers"]
    n_atoms = s["n_atoms"]

    dead_energies = rng.uniform(5.0, 15.0, n_dead).astype(np.float32)
    dead_positions = rng.uniform(-2.0, 2.0, (n_dead, n_atoms, 3)).astype(
        np.float32
    )

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
        "emax": float(np.asarray(s["energies"]).max()),
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(1),
    }
    if step_sizes is not None:
        state["step_sizes"] = np.asarray(step_sizes)
    p = tmp_path / name
    save_checkpoint(p, state)
    return p


def _load_bundle(path: Path) -> RestartBundle:
    _, bundle = load_restart(path)
    return bundle


class TestInitNsParallelRestart:
    """Check how init_ns_parallel seeds each run from restart_states.

    The iteration counter and log evidence come from the checkpoint, a
    ``None`` entry starts a run fresh, and a wrong length raises ValueError.
    """

    def test_restart_seeds_iteration_and_log_evidence(self, tmp_path):
        """Each run takes its iteration counter and log evidence."""
        n_dead_checkpoint = 7
        log_z = -9.5
        s = _make_harmonic_setup(seed=11)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(
            tmp_path, s, n_dead=n_dead_checkpoint, log_evidence=log_z
        )
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"],
            positions,
            s["types"],
            energies,
            cells,
            keys,
            restart_states=[bundle, bundle],
        )

        np.testing.assert_array_equal(
            np.asarray(ns_states.iteration), [n_dead_checkpoint] * 2
        )
        np.testing.assert_allclose(
            np.asarray(ns_states.log_evidence), [log_z] * 2, atol=1e-4
        )

    def test_restart_mixed_some_fresh(self, tmp_path):
        """With [bundle, None] run 0 restarts and run 1 starts fresh."""
        n_dead_checkpoint = 6
        s = _make_harmonic_setup(seed=13)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"],
            positions,
            s["types"],
            energies,
            cells,
            keys,
            restart_states=[bundle, None],
        )

        assert int(ns_states.iteration[0]) == n_dead_checkpoint
        assert int(ns_states.iteration[1]) == 0

    def test_restart_wrong_length_raises(self, tmp_path):
        """A restart_states list of the wrong length raises ValueError."""
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
                s["init_fn"],
                positions,
                s["types"],
                energies,
                cells,
                keys,
                restart_states=[bundle, bundle],
            )

    def test_no_restart_states_gives_fresh_start(self, tmp_path):
        """Without restart_states every run starts at iteration 0."""
        s = _make_harmonic_setup(seed=16)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        ns_states = init_ns_parallel(
            s["init_fn"],
            positions,
            s["types"],
            energies,
            cells,
            keys,
        )

        assert int(ns_states.iteration[0]) == 0
        assert int(ns_states.iteration[1]) == 0


class TestStepSizeRestore:
    """Check that adapted step sizes survive a checkpoint and a restart."""

    def test_bundle_carries_step_sizes(self, tmp_path):
        """load_restart fills RestartBundle.step_sizes when they are saved."""
        s = _make_harmonic_setup(seed=30)
        n_walkers = s["n_walkers"]
        # One move; 0.0123 differs from every default step size.
        saved_ss = np.full((n_walkers, 1), 0.0123, dtype=np.float32)

        p = _write_checkpoint(tmp_path, s, step_sizes=saved_ss)
        bundle = _load_bundle(p)

        assert bundle.step_sizes is not None
        assert bundle.step_sizes.shape == (n_walkers, 1)
        np.testing.assert_allclose(
            np.asarray(bundle.step_sizes), saved_ss, atol=1e-6
        )

    def test_legacy_checkpoint_step_sizes_none(self, tmp_path):
        """A checkpoint without step_sizes gives bundle.step_sizes None."""
        s = _make_harmonic_setup(seed=31)
        p = _write_checkpoint(tmp_path, s)
        bundle = _load_bundle(p)
        assert bundle.step_sizes is None

    def test_restored_step_sizes_land_in_population(self, tmp_path):
        """Restored step sizes win over the initial step_sizes argument."""
        s = _make_harmonic_setup(seed=32)
        n_walkers = s["n_walkers"]
        saved_ss = np.full((n_walkers, 1), 0.0123, dtype=np.float32)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s, step_sizes=saved_ss)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"],
            positions,
            s["types"],
            energies,
            cells,
            keys,
            step_sizes=jnp.full(1, 0.5),
            restart_states=[bundle, bundle],
        )

        pop_ss = np.asarray(ns_states.population.step_sizes)
        assert pop_ss.shape == (2, n_walkers, 1)
        np.testing.assert_allclose(pop_ss[0], saved_ss, atol=1e-6)
        np.testing.assert_allclose(pop_ss[1], saved_ss, atol=1e-6)

    def test_each_run_gets_its_own_step_sizes(self, tmp_path):
        """Two checkpoints with different step sizes land in separate runs."""
        s = _make_harmonic_setup(seed=34)
        n_walkers = s["n_walkers"]
        ss_run0 = np.full((n_walkers, 1), 0.0123, dtype=np.float32)
        ss_run1 = np.full((n_walkers, 1), 0.0456, dtype=np.float32)

        bundles = [
            _load_bundle(
                _write_checkpoint(
                    tmp_path, s, name=f"run{k}.checkpoint.h5", step_sizes=ss
                )
            )
            for k, ss in enumerate((ss_run0, ss_run1))
        ]

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        ns_states = init_ns_parallel(
            s["init_fn"],
            positions,
            s["types"],
            energies,
            cells,
            keys,
            restart_states=bundles,
        )

        pop_ss = np.asarray(ns_states.population.step_sizes)
        np.testing.assert_allclose(pop_ss[0], ss_run0, atol=1e-6)
        np.testing.assert_allclose(pop_ss[1], ss_run1, atol=1e-6)

    def test_legacy_restart_keeps_initial_step_sizes(self, tmp_path):
        """Without saved step_sizes the population keeps the initial ones."""
        s = _make_harmonic_setup(seed=33)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        p = _write_checkpoint(tmp_path, s)
        bundle = _load_bundle(p)

        ns_states = init_ns_parallel(
            s["init_fn"],
            positions,
            s["types"],
            energies,
            cells,
            keys,
            step_sizes=jnp.full(1, 0.5),
            restart_states=[bundle, bundle],
        )

        pop_ss = np.asarray(ns_states.population.step_sizes)
        np.testing.assert_allclose(pop_ss, 0.5, atol=1e-6)


class TestRunNsParallelRestart:
    """Check run_ns_parallel with restart_states."""

    def test_parallel_restart_continues_from_checkpoint(self, tmp_path):
        """Each run adds n_extra iterations to the checkpoint's dead points."""
        n_runs = 2
        n_dead_checkpoint = 4
        n_extra = 6
        s = _make_harmonic_setup(seed=20)

        positions = jnp.stack([s["positions"]] * n_runs)
        energies = jnp.stack([s["energies"]] * n_runs)
        cells = jnp.stack([s["cells"]] * n_runs)
        keys = jax.random.split(s["key"], n_runs)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        out = run_ns_parallel(
            positions,
            s["types"],
            energies,
            cells,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=n_extra,
            n_mcmc_steps=3,
            termination_criteria=[IterationTermination(n_extra)],
            restart_states=[bundle] * n_runs,
        )

        assert out["n_runs"] == n_runs
        assert out["n_dead"].shape == (n_runs,)
        assert out["log_evidence"].shape == (n_runs,)
        np.testing.assert_array_equal(
            np.asarray(out["n_dead"]), [n_dead_checkpoint + n_extra] * n_runs
        )
        assert jnp.all(jnp.isfinite(out["log_evidence"]))

    def test_parallel_restart_parity_with_single_restart(self, tmp_path):
        """A single parallel run matches run_ns after a restart."""
        n_dead_checkpoint = 4
        n_extra = 6
        s = _make_harmonic_setup(seed=23)

        p = _write_checkpoint(tmp_path, s, n_dead=n_dead_checkpoint)
        bundle = _load_bundle(p)

        termination = [IterationTermination(n_extra)]

        out_single = run_ns(
            s["positions"],
            s["types"],
            s["energies"],
            s["cells"],
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_key=s["key"],
            max_iterations=n_extra,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_state=bundle,
        )

        keys = jax.random.split(s["key"], 1)
        out_par = run_ns_parallel(
            s["positions"][None],
            s["types"],
            s["energies"][None],
            s["cells"][None],
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=n_extra,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_states=[bundle],
        )

        assert int(out_single["n_dead"]) == n_dead_checkpoint + n_extra
        assert int(out_par["n_dead"][0]) == int(out_single["n_dead"])
        np.testing.assert_allclose(
            float(out_par["log_evidence"][0]),
            float(out_single["log_evidence"]),
            rtol=1e-4,
        )

    def test_parallel_restart_without_restart_states_is_fresh(self, tmp_path):
        """Without restart_states the runs start fresh."""
        n_extra = 6
        s = _make_harmonic_setup(seed=24)

        positions = jnp.stack([s["positions"]] * 2)
        energies = jnp.stack([s["energies"]] * 2)
        cells = jnp.stack([s["cells"]] * 2)
        keys = jax.random.split(s["key"], 2)

        termination = [IterationTermination(n_extra)]
        out = run_ns_parallel(
            positions,
            s["types"],
            energies,
            cells,
            init_fn=s["init_fn"],
            step_fn=s["step_fn"],
            rng_keys=keys,
            max_iterations=n_extra,
            n_mcmc_steps=3,
            termination_criteria=termination,
        )

        np.testing.assert_array_equal(np.asarray(out["n_dead"]), [n_extra] * 2)
        np.testing.assert_array_equal(
            np.asarray(out["iteration"]), [n_extra] * 2
        )
        assert jnp.all(jnp.isfinite(out["log_evidence"]))

    def test_parallel_restart_wrong_length_propagates(self, tmp_path):
        """run_ns_parallel raises ValueError for a wrong-length list."""
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
                positions,
                s["types"],
                energies,
                cells,
                init_fn=s["init_fn"],
                step_fn=s["step_fn"],
                rng_keys=keys,
                max_iterations=5,
                n_mcmc_steps=3,
                termination_criteria=termination,
                restart_states=[bundle],
            )
