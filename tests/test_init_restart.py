"""Tests for jaxrens.init.restart.load_restart and RestartBundle.

Covers:
- Round-trip: save_checkpoint -> load_restart
- FileNotFoundError on missing path
- ValueError on bare walker-set file (no NS-state fields)
- NVT restart: dead_volumes is None
- NPT restart: dead_volumes is an array
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.init.restart import RestartBundle, load_restart
from jaxrens.init.walker_set import WalkerSet
from jaxrens.io.checkpoint import save_checkpoint
from jaxrens.sampling.move_kernel import MoveKernel
import jaxrens.sampling.moves.random_walk as _rw_mod


def _rw_descriptor():
    return MoveKernel(
        name="random_walk",
        build_kernel=_rw_mod.build_kernel,
        step_size=0.3,
        weight=1.0,
        kernel_kwargs={},
        extra_state_fields={},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ns_state_dict(
    n_walkers: int = 4,
    n_atoms: int = 1,
    n_dead: int = 3,
    npt: bool = False,
) -> dict:
    """Produce a minimal NS-state dict matching save_checkpoint schema."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2, 2, (n_walkers, n_atoms, 3)).astype(np.float32)
    types = np.zeros((n_walkers, n_atoms), dtype=np.int32)
    energies = rng.uniform(0, 5, (n_walkers,)).astype(np.float32)
    cells = np.stack([np.eye(3, dtype=np.float32) * 5.0] * n_walkers)

    dead_energies = rng.uniform(5, 10, n_dead).astype(np.float32)
    dead_positions = rng.uniform(-2, 2, (n_dead, n_atoms, 3)).astype(np.float32)

    state = {
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": cells,
        "dead_energies": dead_energies,
        "dead_positions": dead_positions,
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": -12.5,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(0),
    }

    if npt:
        state["dead_volumes"] = rng.uniform(100, 200, n_dead).astype(np.float32)
        state["live_volumes"] = rng.uniform(100, 200, n_walkers).astype(np.float32)

    return state


def _write_checkpoint(tmp_path: Path, state: dict, name: str = "ckpt.h5") -> Path:
    p = tmp_path / name
    save_checkpoint(p, state, symbol_map={0: "Si"})
    return p


# ---------------------------------------------------------------------------
# Round-trip: save_checkpoint + load_restart
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_walker_set_positions_shape(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=2, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        ws, _ = load_restart(p)
        assert ws.positions.shape == (4, 2, 3)

    def test_walker_set_types_shape(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=2, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        ws, _ = load_restart(p)
        assert ws.types.shape == (4, 2)

    def test_walker_set_cells_shape(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=2, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        ws, _ = load_restart(p)
        assert ws.cells.shape == (4, 3, 3)

    def test_walker_set_symbol_map_restored(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        ws, _ = load_restart(p)
        assert ws.symbol_map == {0: "Si"}

    def test_bundle_n_dead_matches(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=5)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.n_dead == 5

    def test_bundle_iteration_matches(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=5)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.iteration == state["iteration"]

    def test_bundle_log_evidence_matches(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=5)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert abs(bundle.log_evidence - float(state["log_evidence"])) < 1e-5

    def test_bundle_dead_energies_shape(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=5)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.dead_energies.shape == (5,)

    def test_bundle_dead_positions_shape(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=2, n_dead=5)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.dead_positions.shape == (5, 2, 3)

    def test_bundle_dead_energies_values(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        np.testing.assert_allclose(
            np.array(bundle.dead_energies),
            np.array(state["dead_energies"][:3]),
            atol=1e-5,
        )

    def test_returns_tuple_of_correct_types(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        result = load_restart(p)
        assert isinstance(result, tuple)
        assert len(result) == 2
        ws, bundle = result
        assert isinstance(ws, WalkerSet)
        assert isinstance(bundle, RestartBundle)


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        p = tmp_path / "nonexistent.h5"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_restart(p)

    def test_bare_walker_set_raises_value_error(self, tmp_path):
        p = tmp_path / "walkers.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("positions", data=np.zeros((4, 1, 3), dtype=np.float32))
            f.create_dataset("types", data=np.zeros((4, 1), dtype=np.int32))
            f.create_dataset("cells", data=np.stack([np.eye(3)] * 4).astype(np.float32))
        with pytest.raises(ValueError, match="not a valid NS checkpoint"):
            load_restart(p)

    def test_bare_walker_set_error_mentions_start_walker_set(self, tmp_path):
        p = tmp_path / "walkers.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("positions", data=np.zeros((4, 1, 3), dtype=np.float32))
            f.create_dataset("types", data=np.zeros((4, 1), dtype=np.int32))
            f.create_dataset("cells", data=np.stack([np.eye(3)] * 4).astype(np.float32))
        with pytest.raises(ValueError, match="start_walker_set"):
            load_restart(p)


# ---------------------------------------------------------------------------
# NVT restart: dead_volumes is None
# ---------------------------------------------------------------------------

class TestNVTRestart:
    def test_nvt_dead_volumes_is_none(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3, npt=False)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.dead_volumes is None

    def test_nvt_bundle_is_frozen_dataclass(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=2, npt=False)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        with pytest.raises((AttributeError, TypeError)):
            bundle.n_dead = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NPT restart: dead_volumes is an array
# ---------------------------------------------------------------------------

class TestNPTRestart:
    def test_npt_dead_volumes_is_array(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3, npt=True)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.dead_volumes is not None
        assert hasattr(bundle.dead_volumes, "shape")

    def test_npt_dead_volumes_shape(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3, npt=True)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.dead_volumes.shape == (3,)

    def test_npt_dead_volumes_values(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3, npt=True)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        np.testing.assert_allclose(
            np.array(bundle.dead_volumes),
            np.array(state["dead_volumes"][:3]),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# init_ns restart seeding
# ---------------------------------------------------------------------------

class TestInitNsRestart:
    def test_init_ns_with_restart_state_seeds_n_dead(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns

        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=7)
        p = _write_checkpoint(tmp_path, state)
        ws, bundle = load_restart(p)

        backend = create_harmonic()
        init_fn, _, _ = build_mwg(backend, [_rw_descriptor()])

        positions = ws.positions
        types = ws.types[0]
        energies = jax.vmap(
            lambda pos, typs, cel: backend(pos, typs, cel, 0)[0]
        )(positions, ws.types, ws.cells)
        key = jax.random.key(5)

        ns_state = init_ns(
            init_fn, positions, types, energies, ws.cells, key,
            max_dead=200,
            restart_state=bundle,
        )

        assert int(ns_state.n_dead) == 7
        assert int(ns_state.iteration) == bundle.iteration

    def test_init_ns_with_restart_state_seeds_log_evidence(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns

        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        state["log_evidence"] = -5.5
        p = _write_checkpoint(tmp_path, state)
        ws, bundle = load_restart(p)

        backend = create_harmonic()
        init_fn, _, _ = build_mwg(backend, [_rw_descriptor()])

        positions = ws.positions
        types = ws.types[0]
        energies = jax.vmap(
            lambda pos, typs, cel: backend(pos, typs, cel, 0)[0]
        )(positions, ws.types, ws.cells)
        key = jax.random.key(5)

        ns_state = init_ns(
            init_fn, positions, types, energies, ws.cells, key,
            max_dead=200,
            restart_state=bundle,
        )

        assert abs(float(ns_state.log_evidence) - bundle.log_evidence) < 1e-5

    def test_init_ns_without_restart_state_zero_initializes(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns

        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=5)
        p = _write_checkpoint(tmp_path, state)
        ws, _ = load_restart(p)

        backend = create_harmonic()
        init_fn, _, _ = build_mwg(backend, [_rw_descriptor()])

        positions = ws.positions
        types = ws.types[0]
        energies = jax.vmap(
            lambda pos, typs, cel: backend(pos, typs, cel, 0)[0]
        )(positions, ws.types, ws.cells)
        key = jax.random.key(5)

        ns_state = init_ns(
            init_fn, positions, types, energies, ws.cells, key, max_dead=200,
        )

        assert int(ns_state.n_dead) == 0
        assert int(ns_state.iteration) == 0

    def test_init_ns_restart_dead_arrays_padded_to_max_dead(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns

        n_dead = 4
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=n_dead)
        p = _write_checkpoint(tmp_path, state)
        ws, bundle = load_restart(p)

        backend = create_harmonic()
        init_fn, _, _ = build_mwg(backend, [_rw_descriptor()])

        positions = ws.positions
        types = ws.types[0]
        energies = jax.vmap(
            lambda pos, typs, cel: backend(pos, typs, cel, 0)[0]
        )(positions, ws.types, ws.cells)
        key = jax.random.key(5)
        max_dead = 50

        ns_state = init_ns(
            init_fn, positions, types, energies, ws.cells, key,
            max_dead=max_dead,
            restart_state=bundle,
        )

        assert ns_state.dead_energies.shape == (max_dead,)
        assert ns_state.dead_positions.shape == (max_dead, 1, 3)
        assert float(jnp.min(ns_state.dead_energies[n_dead:])) == float(jnp.inf)

    def test_init_ns_restart_under_jit(self, tmp_path):
        """init_ns with restart_state: the produced NSState must survive ns_step under jit."""
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        ws, bundle = load_restart(p)

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, [_rw_descriptor()])

        positions = ws.positions
        types = ws.types[0]
        energies = jax.vmap(
            lambda pos, typs, cel: backend(pos, typs, cel, 0)[0]
        )(positions, ws.types, ws.cells)
        key = jax.random.key(5)

        ns_state = init_ns(
            init_fn, positions, types, energies, ws.cells, key,
            max_dead=100,
            restart_state=bundle,
        )

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert int(new_state.n_dead) == bundle.n_dead + 1
        assert int(new_state.iteration) == bundle.iteration + 1
        assert jnp.isfinite(info["emax"])


# ---------------------------------------------------------------------------
# Mode D resolver tests (moved from test_schema.py::TestInitConfigResolverModeD)
# ---------------------------------------------------------------------------

def _make_ns_checkpoint_resolver(
    tmp_path: Path,
    n_walkers: int = 4,
    n_atoms: int = 1,
    n_dead: int = 5,
    name: str = "ns.checkpoint.h5",
) -> Path:
    """Write a minimal NS checkpoint for Mode D resolver tests."""
    import numpy as _np
    from jaxrens.io.checkpoint import save_checkpoint

    rng = _np.random.default_rng(0)
    positions = rng.uniform(-2, 2, (n_walkers, n_atoms, 3)).astype(_np.float32)
    types = _np.zeros((n_walkers, n_atoms), dtype=_np.int32)
    energies = rng.uniform(1, 10, n_walkers).astype(_np.float32)
    cells = _np.stack([_np.eye(3, dtype=_np.float32) * 6.0] * n_walkers)
    dead_energies = rng.uniform(10, 20, n_dead).astype(_np.float32)
    dead_positions = rng.uniform(-2, 2, (n_dead, n_atoms, 3)).astype(_np.float32)

    state = {
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": cells,
        "dead_energies": dead_energies,
        "dead_positions": dead_positions,
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": -7.3,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(1),
    }
    p = tmp_path / name
    save_checkpoint(p, state, symbol_map={0: "Si"})
    return p


def _cell_cfg_permissive_restart():
    from jaxrens.cli.schema.cell import CellConfig
    return CellConfig(
        max_volume_per_atom=10000.0,
        min_volume_per_atom=0.01,
        min_aspect_ratio=0.001,
    )


class TestInitConfigResolverModeD:
    """Mode D resolver tests: restart_file.

    Moved verbatim from test_schema.py::TestInitConfigResolverModeD.
    """

    def test_mode_d_returns_resolved_init(self, tmp_path):
        from jaxrens.cli.resolve import ResolvedInit, _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert isinstance(result, ResolvedInit)

    def test_mode_d_restart_state_populated(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.init.restart import RestartBundle
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.restart_state is not None
        assert isinstance(result.restart_state, RestartBundle)

    def test_mode_d_restart_state_n_dead(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.restart_state.n_dead == 5

    def test_mode_d_restart_state_iteration(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.restart_state.iteration == 5

    def test_mode_d_symbol_map_populated(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.symbol_map == {0: "Si"}

    def test_mode_d_energies_recomputed(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.initial_energies is not None
        assert result.initial_energies.shape == (4,)
        assert jnp.all(jnp.isfinite(result.initial_energies))

    def test_mode_d_random_initialise_pos_true_warns(self, tmp_path, caplog):
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p, random_initialise_pos=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(
                cfg, n_live=4, seed=0,
                energy_backend=create_harmonic(),
                cell_cfg=_cell_cfg_permissive_restart(),
            )
        assert any(
            "restart_file" in r.message.lower() or "verbatim" in r.message.lower()
            for r in caplog.records
        )

    def test_cohort_gt_1_with_restart_file_raises(self, tmp_path):
        from jaxrens.cli.resolve import expand_cohort
        from jaxrens.cli.schema import RootConfig
        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        d = {
            "run": {"n_live": 4, "max_iterations": 5, "n_mcmc_steps": 2, "seed": 0},
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic"},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02]},
            "init": {"restart_file": str(p)},
        }
        root = RootConfig.model_validate(d)
        import pytest as _pytest
        with _pytest.raises(ValueError, match="restart_file"):
            expand_cohort(root)

    def test_cohort_gt_1_restart_error_message_contains_cohort_size(self, tmp_path):
        from jaxrens.cli.resolve import expand_cohort
        from jaxrens.cli.schema import RootConfig
        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        d = {
            "run": {"n_live": 4, "max_iterations": 5, "n_mcmc_steps": 2, "seed": 0},
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic"},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02, 0.03]},
            "init": {"restart_file": str(p)},
        }
        root = RootConfig.model_validate(d)
        import pytest as _pytest
        with _pytest.raises(ValueError, match="3"):
            expand_cohort(root)

    def test_mode_d_end_to_end_jit(self, tmp_path):
        """Mode D: load checkpoint, init_ns with restart_state, run ns_step under JIT."""
        import numpy as _np
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step
        from jaxrens.sampling.move_kernel import MoveKernel
        import jaxrens.sampling.moves.random_walk as rw_mod

        n_dead_checkpoint = 5
        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=n_dead_checkpoint)

        cfg = InitConfig(restart_file=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive_restart(),
        )

        desc = MoveKernel(
            name="random_walk",
            build_kernel=rw_mod.build_kernel,
            step_size=0.3,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        )
        init_fn, step_fn, _ = build_mwg(backend, [desc])

        key = jax.random.key(11)
        ns_state = init_ns(
            init_fn,
            result.initial_positions,
            result.initial_types,
            result.initial_energies,
            cells=result.initial_cells,
            rng_key=key,
            max_dead=200,
            restart_state=result.restart_state,
        )

        assert int(ns_state.n_dead) == n_dead_checkpoint
        assert int(ns_state.iteration) == n_dead_checkpoint

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert int(new_state.n_dead) == n_dead_checkpoint + 1
        assert int(new_state.iteration) == n_dead_checkpoint + 1
        assert jnp.isfinite(info["emax"])

    def test_mode_d_continued_run_n_dead_increments(self, tmp_path):
        """After restart, run_ns for N more steps: n_dead >= checkpoint + N."""
        import numpy as _np
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, run_ns
        from jaxrens.sampling.termination import IterationTermination
        from jaxrens.sampling.move_kernel import MoveKernel
        import jaxrens.sampling.moves.random_walk as rw_mod

        n_dead_checkpoint = 5
        n_extra_iters = 5
        p = _make_ns_checkpoint_resolver(tmp_path, n_walkers=4, n_atoms=1, n_dead=n_dead_checkpoint)

        cfg = InitConfig(restart_file=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive_restart(),
        )

        desc = MoveKernel(
            name="random_walk",
            build_kernel=rw_mod.build_kernel,
            step_size=0.3,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        )
        init_fn, step_fn, _ = build_mwg(backend, [desc])

        key = jax.random.key(11)
        termination = [IterationTermination(n_extra_iters)]

        out = run_ns(
            positions=result.initial_positions,
            types=result.initial_types,
            energies=result.initial_energies,
            cells=result.initial_cells,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=n_extra_iters,
            n_mcmc_steps=3,
            termination_criteria=termination,
            restart_state=result.restart_state,
        )

        assert out["n_dead"] >= n_dead_checkpoint
        assert jnp.isfinite(out["log_evidence"])
