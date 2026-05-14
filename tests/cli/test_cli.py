"""Test CLI: config parsing, run entry point, public API."""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.cli.parser import parse_input_file
from jaxrens.cli.run import setup_mwg, run_from_config
from jaxrens.state.config import NSConfig, MoveConfig, BackendConfig, OutputConfig


# ---------------------------------------------------------------------------
# Parser tests (raw-read only; dataclass-construction path removed in step 7)
# ---------------------------------------------------------------------------


class TestParser:
    def test_parse_input_file(self, tmp_path):
        config = tmp_path / "ns.inp"
        config.write_text(
            "n_walkers = 100\n"
            "max_iterations = 5000\n"
            "# this is a comment\n"
            "\n"
            "backend = lj\n"
            "step_size = 0.05\n"
        )
        raw = parse_input_file(config)
        assert raw["n_walkers"] == "100"
        assert raw["max_iterations"] == "5000"
        assert raw["backend"] == "lj"
        assert raw["step_size"] == "0.05"

    def test_parse_input_file_inline_comment(self, tmp_path):
        config = tmp_path / "ns.inp"
        config.write_text("n_walkers = 50  # inline comment\n")
        raw = parse_input_file(config)
        assert raw["n_walkers"] == "50"

    def test_parse_input_file_blank_lines(self, tmp_path):
        config = tmp_path / "ns.inp"
        config.write_text("\n\nn_walkers = 10\n\n")
        raw = parse_input_file(config)
        assert raw == {"n_walkers": "10"}


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
        ns_config = NSConfig(n_live=20, max_iterations=50, n_mcmc_steps=5, seed=42)
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="none", working_dir=tmp_path, info_interval=999
        )

        key = jax.random.key(0)
        positions = jax.random.uniform(key, (20, 1, 3), minval=-3.0, maxval=3.0)
        types = jnp.zeros((1,), dtype=jnp.int32)

        result = run_from_config(
            ns_config, move_config, backend_config, output_config,
            initial_positions=positions,
            initial_types=types,
        )

        assert result["iteration"] > 0
        assert jnp.isfinite(result["log_evidence"])


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
            working_dir=tmp_path, prefix="run", level="info",
        )
        logging.getLogger("jaxrens").info("hello")

        for h in list(logging.getLogger("jaxrens").handlers):
            if getattr(h, "_jaxrens_managed", False):
                logging.getLogger("jaxrens").removeHandler(h)
                h.close()

        assert "hello" in (tmp_path / "run.log").read_text()
