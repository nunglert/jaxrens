"""Tests for ``jaxrens.cli.run``.

They cover ``setup_mwg``, ``run_from_config`` on the harmonic toy backend
and ``configure_file_logging``.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxrens.cli.run as run_mod
from jaxrens.backends.softcore import SoftCoreBackend, SoftCoreTerm
from jaxrens.backends.toy import HarmonicBackend, create_harmonic
from jaxrens.cli.run import (
    _move_config_to_descriptor,
    configure_file_logging,
    run_from_config,
    setup_mwg,
)
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk as _rw
from jaxrens.state.config import (
    BackendConfig,
    MoveConfig,
    NSConfig,
    OutputConfig,
)


class TestRun:
    """Check ``setup_mwg`` and ``run_from_config``.

    After the basic run, each test turns on one option of
    ``run_from_config`` (soft core, pressure, output format, loggers) on
    the harmonic toy backend, which keeps every run under a second.
    """

    def test_setup_mwg_random_walk(self):
        backend = create_harmonic()
        move_config = MoveConfig(move_type="random_walk")
        init_fn, step_fn, _ = setup_mwg(move_config, backend)
        assert callable(init_fn)
        assert callable(step_fn)

    @pytest.mark.parametrize(
        "move_type, name, kernel_kwargs",
        [
            ("random_walk", "random_walk", {}),
            ("galilean", "gmc", {"n_reflect": 7}),
            ("gmc", "gmc", {"n_reflect": 7}),
            ("hmc", "hmc", {"n_leapfrog": 7}),
            ("single_atom", "single_atom", {}),
        ],
    )
    def test_move_config_to_descriptor(self, move_type, name, kernel_kwargs):
        """``n_steps`` maps to ``n_reflect`` (GMC) or ``n_leapfrog`` (HMC)."""
        desc = _move_config_to_descriptor(
            MoveConfig(move_type=move_type, n_steps=7, step_size=0.2)
        )
        assert desc.name == name
        assert desc.kernel_kwargs == kernel_kwargs
        assert desc.step_size == pytest.approx(0.2)

    def test_setup_mwg_unknown_raises(self):
        backend = create_harmonic()
        move_config = MoveConfig(move_type="nonexistent")
        with pytest.raises(ValueError, match="Unknown move type"):
            setup_mwg(move_config, backend)

    def test_run_from_config_toy(self, tmp_path):
        """End-to-end: run NS on harmonic with config objects."""
        ns_config = NSConfig(
            n_live=20, max_iterations=50, n_mcmc_steps=5, seed=42
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="none", working_dir=tmp_path, info_interval=999
        )

        key = jax.random.key(0)
        positions = jax.random.uniform(
            key, (20, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)

        result = run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
        )

        assert result["iteration"] > 0
        assert jnp.isfinite(result["log_evidence"])

    def test_run_from_config_with_softcore_wrap(self, tmp_path):
        """A run with ``softcore_repulsion`` in the BackendConfig finishes."""
        ns_config = NSConfig(
            n_live=12, max_iterations=10, n_mcmc_steps=2, seed=0
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(
            backend_type="harmonic",
            softcore_repulsion={
                "a0": 3.0,
                "b0": 1.0,
                "d0": 100.0,
                "r_core_switch": 0.75,
                "r_core_cut": 1.25,
            },
        )
        output_config = OutputConfig(
            format="none",
            working_dir=tmp_path,
            info_interval=999,
            temperature_lag_interval=None,
        )
        key = jax.random.key(1)
        positions = jax.random.uniform(
            key, (12, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        result = run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
        )
        assert jnp.isfinite(result["log_evidence"])

    def test_run_from_config_does_not_double_softcore(
        self, tmp_path, monkeypatch
    ):
        """A resolver-built ``base_backend`` already carries the soft-core
        term, so ``run_from_config`` must not add it a second time.
        """
        captured = {}

        class _Stop(Exception):
            pass

        def _capture(move_config, backend):
            captured["backend"] = backend
            raise _Stop

        monkeypatch.setattr(run_mod, "setup_mwg", _capture)

        softcore = {"a0": 1.0, "b0": 3.0, "d0": 1.0}
        with pytest.raises(_Stop):
            run_from_config(
                NSConfig(n_live=4, max_iterations=1, n_mcmc_steps=1, seed=0),
                MoveConfig(move_type="random_walk", step_size=0.3),
                BackendConfig(
                    backend_type="harmonic", softcore_repulsion=softcore
                ),
                OutputConfig(format="none", working_dir=tmp_path),
                initial_positions=jnp.zeros((4, 2, 3)),
                initial_types=jnp.zeros((2,), dtype=jnp.int32),
                initial_energies=jnp.zeros(4),
                base_backend=SoftCoreBackend(HarmonicBackend(), **softcore),
            )

        terms = captured["backend"].terms
        assert sum(isinstance(t, SoftCoreTerm) for t in terms) == 1

    def test_run_from_config_with_pressure(self, tmp_path):
        """In an NPT run every final live energy is U + P V."""
        ns_config = NSConfig(
            n_live=12,
            max_iterations=10,
            n_mcmc_steps=2,
            seed=0,
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="none",
            working_dir=tmp_path,
            info_interval=999,
            temperature_lag_interval=None,
        )
        key = jax.random.key(2)
        positions = jax.random.uniform(
            key, (12, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        # Harmonic ignores the cell, but the P V term reads det(cell).
        cells = jnp.broadcast_to(jnp.eye(3) * 6.0, (12, 3, 3))
        result = run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
            initial_cells=cells,
            ensemble_params={"pressure": 0.01},
        )
        assert jnp.isfinite(result["log_evidence"])
        harmonic = create_harmonic()
        u = np.array(
            [
                harmonic(p, types, c, 0).energy
                for p, c in zip(result["positions"], result["cells"])
            ]
        )
        pv = 0.01 * np.linalg.det(np.asarray(result["cells"]))
        np.testing.assert_allclose(result["energies"], u + pv, rtol=1e-6)

    @pytest.mark.parametrize(
        "ensemble_params",
        [{"pressure": 0.01}, {"chemical_potentials": [0.5]}],
        ids=["pressure", "mu"],
    )
    def test_initial_energies_include_ensemble_term(
        self, tmp_path, monkeypatch, ensemble_params
    ):
        """Initial energies are U + P V or U - mu N, each with its own cell.

        The walkers get cells of different sizes, so using one cell for all
        of them would give the wrong P V. ``run_ns`` is replaced to capture
        the energies before any move changes them.
        """
        captured = {}

        class _Stop(Exception):
            pass

        def _capture(**kwargs):
            captured["energies"] = kwargs["energies"]
            raise _Stop

        monkeypatch.setattr(run_mod, "run_ns", _capture)

        n_live = 6
        positions = jax.random.uniform(
            jax.random.key(0), (n_live, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        scales = jnp.linspace(5.0, 7.0, n_live)
        cells = scales[:, None, None] * jnp.eye(3)
        with pytest.raises(_Stop):
            run_from_config(
                NSConfig(n_live=n_live, max_iterations=1, seed=0),
                MoveConfig(move_type="random_walk", step_size=0.3),
                BackendConfig(backend_type="harmonic"),
                OutputConfig(
                    format="none",
                    working_dir=tmp_path,
                    temperature_lag_interval=None,
                ),
                initial_positions=positions,
                initial_types=types,
                initial_cells=cells,
                ensemble_params=ensemble_params,
            )

        harmonic = create_harmonic()
        u = jnp.array(
            [harmonic(p, types, c, 0).energy for p, c in zip(positions, cells)]
        )
        if "pressure" in ensemble_params:
            expected = u + 0.01 * scales**3
        else:
            expected = u - 0.5
        np.testing.assert_allclose(captured["energies"], expected, rtol=1e-6)

    def test_run_from_config_format_extxyz(self, tmp_path):
        """``format='extxyz'`` writes a ``.traj.extxyz`` file."""
        ns_config = NSConfig(
            n_live=12, max_iterations=10, n_mcmc_steps=2, seed=0
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="extxyz",
            working_dir=tmp_path,
            info_interval=999,
            traj_interval=1,
            snapshot_interval=999,
            temperature_lag_interval=None,
        )
        key = jax.random.key(3)
        positions = jax.random.uniform(
            key, (12, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
        )
        assert (tmp_path / "ns.traj.extxyz").exists()

    def test_run_from_config_format_h5(self, tmp_path):
        """``format='h5'`` writes a ``.traj.h5`` file.

        The run path forwards ``wrap=output.wrap_atoms`` to every writer, so
        ``H5TrajectoryWriter`` must accept ``wrap``.
        """
        ns_config = NSConfig(
            n_live=12, max_iterations=10, n_mcmc_steps=2, seed=0
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="h5",
            working_dir=tmp_path,
            info_interval=999,
            traj_interval=1,
            snapshot_interval=999,
            temperature_lag_interval=None,
        )
        key = jax.random.key(4)
        positions = jax.random.uniform(
            key, (12, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
        )
        assert (tmp_path / "ns.traj.h5").exists()

    def test_run_from_config_with_optional_loggers(self, tmp_path):
        """Move descriptors plus ``save_acc_rates`` and
        ``save_max_neighbors`` turn on all three HDF5 loggers.
        """
        ns_config = NSConfig(
            n_live=12, max_iterations=10, n_mcmc_steps=2, seed=0
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="none",
            working_dir=tmp_path,
            info_interval=999,
            save_acc_rates=True,
            save_max_neighbors=True,
            temperature_lag_interval=None,
        )
        # Real MoveKernel; adaptation logger needs ``name`` per descriptor.
        descriptors = [
            MoveKernel(
                "rw",
                _rw.build_kernel,
                step_size=0.3,
                step_size_max=5.0,
                min_rate=0.2,
                max_rate=0.7,
            )
        ]
        key = jax.random.key(5)
        positions = jax.random.uniform(
            key, (12, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
            move_descriptors=descriptors,
        )
        assert (tmp_path / "ns.adaptation.h5").exists()
        assert (tmp_path / "ns.acc_rates.h5").exists()
        assert (tmp_path / "ns.max_neighbors.h5").exists()

    def test_run_from_config_multi_move_baseline_no_full_auto(self, tmp_path):
        """Two moves without ``full_auto`` still write the iteration-0 row.

        The adaptation logger is wired whenever move descriptors are given,
        and its baseline row has shape ``(n_runs, n_moves) = (1, 2)``. The
        population's step sizes must have one column per move to match, even
        when ``full_auto`` is off. With a single move a wrong ``(1, 1)``
        shape would match by accident, hence two moves.
        """
        ns_config = NSConfig(
            n_live=12, max_iterations=8, n_mcmc_steps=2, seed=0
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="none",
            working_dir=tmp_path,
            info_interval=999,
            temperature_lag_interval=None,
        )
        # No adaptation_config, so the run is not full_auto.
        descriptors = [
            MoveKernel("rw0", _rw.build_kernel, step_size=0.3),
            MoveKernel("rw1", _rw.build_kernel, step_size=0.15),
        ]
        key = jax.random.key(7)
        positions = jax.random.uniform(
            key, (12, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
            move_descriptors=descriptors,
        )
        assert (tmp_path / "ns.adaptation.h5").exists()


def test_configure_file_logging_writes_to_file(tmp_path):
    configure_file_logging(
        working_dir=tmp_path,
        prefix="run",
        level="info",
    )
    logging.getLogger("jaxrens").info("hello")

    # Detach the handler so later tests do not log into this tmp_path.
    for h in list(logging.getLogger("jaxrens").handlers):
        if getattr(h, "_jaxrens_managed", False):
            logging.getLogger("jaxrens").removeHandler(h)
            h.close()

    assert "hello" in (tmp_path / "run.log").read_text()
