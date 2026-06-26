"""Test CLI: config parsing, run entry point, public API."""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.cli.run import run_from_config, setup_mwg
from jaxrens.state.config import (
    BackendConfig,
    MoveConfig,
    NSConfig,
    OutputConfig,
)

# ---------------------------------------------------------------------------
# Run entry point tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_setup_mwg_random_walk(self):
        from jaxrens.backends.toy import create_harmonic

        backend = create_harmonic()
        move_config = MoveConfig(move_type="random_walk")
        init_fn, step_fn, _ = setup_mwg(move_config, backend)
        assert callable(init_fn)
        assert callable(step_fn)

    def test_setup_mwg_galilean(self):
        from jaxrens.backends.toy import create_harmonic

        backend = create_harmonic()
        move_config = MoveConfig(move_type="galilean")
        init_fn, step_fn, _ = setup_mwg(move_config, backend)
        assert callable(init_fn)
        assert callable(step_fn)

    def test_setup_mwg_unknown_raises(self):
        from jaxrens.backends.toy import create_harmonic

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

    # ---------------------------------------------------------------------
    # Branch-coverage variants of ``test_run_from_config_toy``.  Each test
    # flips a single conditional in ``run_from_config`` so the corresponding
    # block of code is exercised at least once.  Toy harmonic backend keeps
    # each one under a second.
    # ---------------------------------------------------------------------

    def test_run_from_config_with_softcore_wrap(self, tmp_path):
        """``backend_config.softcore_repulsion`` triggers the
        ``SoftCoreBackend`` wrap branch in ``run_from_config``."""
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

    def test_run_from_config_with_pressure(self, tmp_path):
        """A non-empty ``ensemble_params`` triggers the ``EnsembleBackend`` wrap
        branch and adds a PV term to the energies."""
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
        # NPT needs a cell; harmonic ignores it but EnsembleBackend reads det(cell).
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
        """``save_acc_rates`` and ``save_max_neighbors`` flags + a
        move_descriptors list register the optional acc-rates / max-neighbors
        loggers (lines 410-446 in run.py)."""
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.moves import random_walk as _rw

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
        # All three optional loggers should have produced their files.
        assert (tmp_path / "ns.adaptation.h5").exists()
        assert (tmp_path / "ns.acc_rates.h5").exists()
        assert (tmp_path / "ns.max_neighbors.h5").exists()


class TestConfigureFileLogging:
    """``configure_file_logging`` is called exactly once, early, from
    ``cli._cmd_run`` so the resolver phase reaches the log file from
    second 1.  Regression: previously called a second time inside
    ``run_*_from_config`` which truncated the resolver content (see
    jaxmd_si16_2 slurm-453556 where the file started at burn-in).
    """

    def test_writes_to_file(self, tmp_path):
        import logging

        from jaxrens.cli.run import configure_file_logging

        configure_file_logging(
            working_dir=tmp_path,
            prefix="run",
            level="info",
        )
        logging.getLogger("jaxrens").info("hello")

        for h in list(logging.getLogger("jaxrens").handlers):
            if getattr(h, "_jaxrens_managed", False):
                logging.getLogger("jaxrens").removeHandler(h)
                h.close()

        assert "hello" in (tmp_path / "run.log").read_text()
