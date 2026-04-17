"""Test CLI: config parsing, run entry point, public API."""

import jax
import jax.numpy as jnp
import pytest
from pathlib import Path

from jaxrens.cli.parser import (
    parse_input_file,
    raw_to_configs,
    dict_to_input_str,
    load_config,
)
from jaxrens.cli.run import setup_mwg, run_from_config
from jaxrens.state.config import NSConfig, MoveConfig, BackendConfig, OutputConfig


# ---------------------------------------------------------------------------
# Parser tests
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

    def test_raw_to_configs(self):
        raw = {
            "n_walkers": "200",
            "backend": "harmonic",
            "step_size": "0.05",
            "move_type": "random_walk",
        }
        ns, move, backend, output = raw_to_configs(raw)

        assert isinstance(ns, NSConfig)
        assert ns.n_live == 200
        assert isinstance(move, MoveConfig)
        assert move.move_type == "random_walk"
        assert move.step_size == 0.05
        assert isinstance(backend, BackendConfig)
        assert backend.backend_type == "harmonic"
        assert isinstance(output, OutputConfig)

    def test_load_config_roundtrip(self, tmp_path):
        config = tmp_path / "test.inp"
        config.write_text(
            "n_walkers = 50\n"
            "backend = lj\n"
            "move_type = galilean\n"
            "max_neighbors_list = 10 20 30\n"
        )
        ns, move, backend, output = load_config(config)
        assert ns.n_live == 50
        assert move.move_type == "galilean"
        assert backend.max_neighbors_list == [10, 20, 30]

    def test_dict_to_input_str(self):
        s = dict_to_input_str({"n_walkers": 100, "backend": "lj"})
        assert "n_walkers = 100" in s
        assert "backend = lj" in s


# ---------------------------------------------------------------------------
# Run entry point tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_setup_mwg_random_walk(self):
        from jaxrens.backends.toy import create_harmonic

        backend = create_harmonic()
        move_config = MoveConfig(move_type="random_walk")
        init_fn, step_fn = setup_mwg(move_config, backend)
        assert callable(init_fn)
        assert callable(step_fn)

    def test_setup_mwg_galilean(self):
        from jaxrens.backends.toy import create_harmonic

        backend = create_harmonic()
        move_config = MoveConfig(move_type="galilean")
        init_fn, step_fn = setup_mwg(move_config, backend)
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
        backend_config = BackendConfig(backend_type="harmonic", n_atoms=1)
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


# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_top_level_imports(self):
        from jaxrens import (
            load_backend,
            run_ns,
            init_ns,
            ns_step,
            run_from_config,
            NSConfig,
            MoveConfig,
            BackendConfig,
            OutputConfig,
        )
        assert callable(load_backend)
        assert callable(run_ns)

    def test_low_level_imports(self):
        from jaxrens.sampling.moves import random_walk, galilean
        assert hasattr(random_walk, "build_kernel")
        assert hasattr(galilean, "build_kernel")

    def test_backend_imports(self):
        from jaxrens.backends import loader
        assert hasattr(loader, "load_backend")
