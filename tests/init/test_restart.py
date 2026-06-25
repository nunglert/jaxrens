"""Tests for jaxrens.init.restart.load_restart and RestartBundle.

Covers:
- Round-trip: save_checkpoint -> load_restart
- FileNotFoundError on missing path
- ValueError on bare walker-set file (no NS-state fields)
- NVT restart: dead_volumes is None
- NPT restart: dead_volumes is an array
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxrens.sampling.moves.random_walk as _rw_mod
from jaxrens.init.restart import (
    RestartBundle,
    infer_restart_shape,
    load_restart,
)
from jaxrens.init.walker_set import WalkerSet
from jaxrens.io.checkpoint import save_checkpoint
from jaxrens.sampling.move_kernel import MoveKernel


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
    dead_positions = rng.uniform(-2, 2, (n_dead, n_atoms, 3)).astype(
        np.float32
    )

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
        "emax": float(energies.max()),
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(0),
    }

    if npt:
        state["dead_volumes"] = rng.uniform(100, 200, n_dead).astype(
            np.float32
        )
        state["live_volumes"] = rng.uniform(100, 200, n_walkers).astype(
            np.float32
        )

    return state


def _make_ns_state_dict_no_dead(
    n_walkers: int = 4,
    n_atoms: int = 1,
    n_dead: int = 3,
) -> dict:
    """NS-state dict WITHOUT dead arrays — matches production checkpoints.

    ``_ns_state_to_checkpoint_dict`` never embeds dead-point history (it is
    streamed to ``.energies`` / ``.traj``), so real checkpoints lack
    ``dead_energies`` / ``dead_positions``.
    """
    state = _make_ns_state_dict(n_walkers, n_atoms, n_dead)
    del state["dead_energies"]
    del state["dead_positions"]
    return state


def _write_checkpoint(
    tmp_path: Path, state: dict, name: str = "ckpt.h5"
) -> Path:
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

    def test_checkpoint_without_dead_arrays_loads(self, tmp_path):
        """Real checkpoints omit dead arrays (streamed to disk) — must load.

        Regression: ``load_restart`` previously required ``dead_energies`` /
        ``dead_positions`` and rejected every production checkpoint.
        """
        state = _make_ns_state_dict_no_dead(n_walkers=4, n_atoms=2, n_dead=7)
        p = _write_checkpoint(tmp_path, state)
        ws, bundle = load_restart(p)
        assert ws.positions.shape == (4, 2, 3)
        assert bundle.iteration == 7
        assert bundle.n_dead == 7
        # Dead arrays absent → None, and init_ns never reads them.
        assert bundle.dead_energies is None
        assert bundle.dead_positions is None

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
# Step sizes (single-run / scalar branch)
# ---------------------------------------------------------------------------


class TestStepSizes:
    def test_step_sizes_round_trip(self, tmp_path):
        """Scalar checkpoint round-trips adapted step sizes into the bundle."""
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        saved_ss = np.full((4, 2), 0.0123, dtype=np.float32)  # (K, n_moves)
        state["step_sizes"] = saved_ss
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.step_sizes is not None
        assert bundle.step_sizes.shape == (4, 2)
        np.testing.assert_allclose(
            np.asarray(bundle.step_sizes), saved_ss, atol=1e-6
        )

    def test_step_sizes_absent_is_none(self, tmp_path):
        """Legacy checkpoint without step_sizes leaves the field None."""
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state)  # no step_sizes
        _, bundle = load_restart(p)
        assert bundle.step_sizes is None


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
            f.create_dataset(
                "positions", data=np.zeros((4, 1, 3), dtype=np.float32)
            )
            f.create_dataset("types", data=np.zeros((4, 1), dtype=np.int32))
            f.create_dataset(
                "cells", data=np.stack([np.eye(3)] * 4).astype(np.float32)
            )
        with pytest.raises(ValueError, match="not a valid NS checkpoint"):
            load_restart(p)

    def test_bare_walker_set_error_mentions_start_walker_set(self, tmp_path):
        p = tmp_path / "walkers.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset(
                "positions", data=np.zeros((4, 1, 3), dtype=np.float32)
            )
            f.create_dataset("types", data=np.zeros((4, 1), dtype=np.int32))
            f.create_dataset(
                "cells", data=np.stack([np.eye(3)] * 4).astype(np.float32)
            )
        with pytest.raises(ValueError, match="start_walker_set"):
            load_restart(p)


# ---------------------------------------------------------------------------
# NVT restart: dead_volumes is None
# ---------------------------------------------------------------------------


class TestNVTRestart:
    def test_nvt_dead_volumes_is_none(self, tmp_path):
        state = _make_ns_state_dict(
            n_walkers=4, n_atoms=1, n_dead=3, npt=False
        )
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)
        assert bundle.dead_volumes is None

    def test_nvt_bundle_is_frozen_dataclass(self, tmp_path):
        state = _make_ns_state_dict(
            n_walkers=4, n_atoms=1, n_dead=2, npt=False
        )
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
    def test_init_ns_with_restart_state_seeds_iteration(self, tmp_path):
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
            init_fn,
            positions,
            types,
            energies,
            ws.cells,
            key,
            restart_state=bundle,
        )

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
            init_fn,
            positions,
            types,
            energies,
            ws.cells,
            key,
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
            init_fn,
            positions,
            types,
            energies,
            ws.cells,
            key,
        )

        assert int(ns_state.iteration) == 0

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
            init_fn,
            positions,
            types,
            energies,
            ws.cells,
            key,
            restart_state=bundle,
        )

        jit_ns_step = jax.jit(
            ns_step, static_argnames=("step_fn", "n_mcmc_steps")
        )
        new_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert int(new_state.iteration) == bundle.iteration + 1
        assert jnp.isfinite(info["emax"])


# ---------------------------------------------------------------------------
# Mode D resolver tests (moved from test_schema.py::TestInitSpecResolverModeD)
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
    dead_positions = rng.uniform(-2, 2, (n_dead, n_atoms, 3)).astype(
        _np.float32
    )

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
    from jaxrens.cli.schema.cell import CellSpec

    return CellSpec(
        max_volume_per_atom=10000.0,
        min_volume_per_atom=0.01,
        min_aspect_ratio=0.001,
    )


class TestInitSpecResolverModeD:
    """Mode D resolver tests: restart_file.

    Moved verbatim from test_schema.py::TestInitSpecResolverModeD.
    """

    def test_mode_d_returns_resolved_init(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import ResolvedInit, _resolve_init
        from jaxrens.cli.schema.init import InitSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        cfg = InitSpec(restart_file=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert isinstance(result, ResolvedInit)

    def test_mode_d_restart_state_populated(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitSpec
        from jaxrens.init.restart import RestartBundle

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        cfg = InitSpec(restart_file=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.restart_state is not None
        assert isinstance(result.restart_state, RestartBundle)

    def test_mode_d_restart_state_n_dead(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        cfg = InitSpec(restart_file=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.restart_state.n_dead == 5

    def test_mode_d_restart_state_iteration(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        cfg = InitSpec(restart_file=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.restart_state.iteration == 5

    def test_mode_d_symbol_map_populated(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        cfg = InitSpec(restart_file=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        assert result.symbol_map == {0: "Si"}

    def test_mode_d_energies_recomputed(self, tmp_path):
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import (
            _finalise_initial_energies_and_counts,
            _resolve_init,
        )
        from jaxrens.cli.schema.init import InitSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        cfg = InitSpec(restart_file=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        # _resolve_init now leaves energies as None; caller finalizes.
        energies, _ = _finalise_initial_energies_and_counts(
            backend,
            result.initial_positions,
            result.initial_types,
            result.initial_cells,
        )
        assert energies is not None
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_mode_d_random_initialise_pos_true_warns(self, tmp_path, caplog):
        import logging

        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        cfg = InitSpec(restart_file=p, random_initialise_pos=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(
                cfg,
                n_live=4,
                seed=0,
                energy_backend=create_harmonic(),
                cell_cfg=_cell_cfg_permissive_restart(),
            )
        assert any(
            "restart_file" in r.message.lower()
            or "verbatim" in r.message.lower()
            for r in caplog.records
        )

    def test_multi_replica_with_restart_file_raises(self, tmp_path):
        from jaxrens.cli.resolve import resolve
        from jaxrens.cli.schema import RootSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        d = {
            "run": {
                "n_live": 4,
                "max_iterations": 5,
                "n_mcmc_steps": 2,
                "seed": 0,
            },
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic"},
            "output": {
                "format": "none",
                "working_dir": ".",
                "info_interval": 999,
            },
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02]},
            "init": {"restart_file": str(p)},
        }
        root = RootSpec.model_validate(d)
        import pytest as _pytest

        with _pytest.raises(ValueError, match="restart_file"):
            resolve(root)

    def test_multi_replica_restart_error_message_contains_n_total(
        self, tmp_path
    ):
        from jaxrens.cli.resolve import resolve
        from jaxrens.cli.schema import RootSpec

        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=5
        )
        d = {
            "run": {
                "n_live": 4,
                "max_iterations": 5,
                "n_mcmc_steps": 2,
                "seed": 0,
            },
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic"},
            "output": {
                "format": "none",
                "working_dir": ".",
                "info_interval": 999,
            },
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02, 0.03]},
            "init": {"restart_file": str(p)},
        }
        root = RootSpec.model_validate(d)
        import pytest as _pytest

        with _pytest.raises(ValueError, match="3"):
            resolve(root)

    def test_mode_d_end_to_end_jit(self, tmp_path):
        """Mode D: load checkpoint, init_ns with restart_state, run ns_step under JIT."""
        import jaxrens.sampling.moves.random_walk as rw_mod
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import (
            _finalise_initial_energies_and_counts,
            _resolve_init,
        )
        from jaxrens.cli.schema.init import InitSpec
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        n_dead_checkpoint = 5
        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=n_dead_checkpoint
        )

        cfg = InitSpec(restart_file=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        # Finalize is the caller's job after the refactor.
        energies, _ = _finalise_initial_energies_and_counts(
            backend,
            result.initial_positions,
            result.initial_types,
            result.initial_cells,
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
            energies,
            cells=result.initial_cells,
            rng_key=key,
            restart_state=result.restart_state,
        )

        assert int(ns_state.iteration) == n_dead_checkpoint

        jit_ns_step = jax.jit(
            ns_step, static_argnames=("step_fn", "n_mcmc_steps")
        )
        new_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert int(new_state.iteration) == n_dead_checkpoint + 1
        assert jnp.isfinite(info["emax"])

    def test_mode_d_continued_run_n_dead_increments(self, tmp_path):
        """After restart, run_ns for N more steps: n_dead >= checkpoint + N."""
        import jaxrens.sampling.moves.random_walk as rw_mod
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import (
            _finalise_initial_energies_and_counts,
            _resolve_init,
        )
        from jaxrens.cli.schema.init import InitSpec
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns
        from jaxrens.sampling.termination import IterationTermination

        n_dead_checkpoint = 5
        n_extra_iters = 5
        p = _make_ns_checkpoint_resolver(
            tmp_path, n_walkers=4, n_atoms=1, n_dead=n_dead_checkpoint
        )

        cfg = InitSpec(restart_file=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive_restart(),
        )
        # Finalize is the caller's job after the refactor.
        energies, _ = _finalise_initial_energies_and_counts(
            backend,
            result.initial_positions,
            result.initial_types,
            result.initial_cells,
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
            energies=energies,
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


# ---------------------------------------------------------------------------
# Shape-aware dispatcher: load_restart (batched checkpoints)
# ---------------------------------------------------------------------------


def _make_batched_ns_state_dict(
    batch_shape: tuple,
    n_walkers: int = 4,
    n_atoms: int = 1,
    max_dead: int = 20,
    npt: bool = False,
) -> dict:
    """Produce a batched NS-state dict for save_checkpoint.

    ``batch_shape`` is the leading batch prefix, e.g. ``(3,)`` for 3 parallel
    runs or ``(2, 2)`` for a (G=2, P=2) multi-GPU run.
    """
    rng = np.random.default_rng(99)

    full_positions_shape = batch_shape + (n_walkers, n_atoms, 3)
    full_types_shape = batch_shape + (n_walkers, n_atoms)
    full_energies_shape = batch_shape + (n_walkers,)
    full_cells_shape = batch_shape + (n_walkers, 3, 3)
    full_dead_energies_shape = batch_shape + (max_dead,)
    full_dead_positions_shape = batch_shape + (max_dead, n_atoms, 3)

    positions = rng.uniform(-2, 2, full_positions_shape).astype(np.float32)
    types = np.zeros(full_types_shape, dtype=np.int32)
    energies = rng.uniform(0, 5, full_energies_shape).astype(np.float32)
    cells = np.broadcast_to(
        np.eye(3, dtype=np.float32) * 5.0,
        full_cells_shape,
    ).copy()
    dead_energies = rng.uniform(5, 10, full_dead_energies_shape).astype(
        np.float32
    )
    dead_positions = rng.uniform(-2, 2, full_dead_positions_shape).astype(
        np.float32
    )

    # n_dead: per-run count, stored as batched array
    n_dead_vals = rng.integers(2, max_dead - 1, size=batch_shape).astype(
        np.int32
    )
    # iteration matches n_dead for simplicity
    iteration_vals = n_dead_vals.copy()
    # log_evidence: per-run float
    log_evidence_vals = rng.uniform(-20, -5, batch_shape).astype(np.float64)

    # Per-replica emax — empirical max of each replica's energies.
    emax_vals = energies.reshape(batch_shape + (n_walkers,)).max(axis=-1)

    # Batched rng_key matches the batch prefix — production batched NSState
    # always has rng_key with the batcher's shape_prefix.
    n_batch = int(np.prod(batch_shape)) if batch_shape else 1
    batched_key = jax.random.split(jax.random.key(0), n_batch).reshape(
        batch_shape
    )

    state: dict = {
        "positions": positions,
        "types": types,
        "energies": energies,
        "cells": cells,
        "dead_energies": dead_energies,
        "dead_positions": dead_positions,
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": log_evidence_vals,
        "iteration": iteration_vals,
        "emax": emax_vals,
        "n_dead": n_dead_vals,
        "n_walkers": n_walkers,
        "rng_key": batched_key,
    }

    if npt:
        state["dead_volumes"] = rng.uniform(
            100, 200, full_dead_energies_shape
        ).astype(np.float32)
        state["live_volumes"] = rng.uniform(
            100, 200, batch_shape + (n_walkers,)
        ).astype(np.float32)

    return state


class TestLoadRestartParallel:
    """load_restart on a (n_runs,)-shaped checkpoint → list[RestartBundle]."""

    def test_returns_list(self, tmp_path):
        n_runs = 3
        state = _make_batched_ns_state_dict(batch_shape=(n_runs,))
        p = tmp_path / "par.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        assert isinstance(result, list)
        assert len(result) == n_runs

    def test_each_element_is_restart_bundle(self, tmp_path):
        n_runs = 3
        state = _make_batched_ns_state_dict(batch_shape=(n_runs,))
        p = tmp_path / "par.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for b in result:
            assert isinstance(b, RestartBundle)

    def test_n_dead_matches_per_run(self, tmp_path):
        n_runs = 3
        state = _make_batched_ns_state_dict(batch_shape=(n_runs,))
        p = tmp_path / "par.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for r in range(n_runs):
            expected = int(np.asarray(state["n_dead"])[r])
            assert result[r].n_dead == expected

    def test_dead_energies_shape_per_run(self, tmp_path):
        n_runs = 3
        state = _make_batched_ns_state_dict(batch_shape=(n_runs,))
        p = tmp_path / "par.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for r in range(n_runs):
            n_dead = result[r].n_dead
            assert result[r].dead_energies.shape == (n_dead,)

    def test_dead_positions_shape_per_run(self, tmp_path):
        n_runs = 3
        n_atoms = 2
        state = _make_batched_ns_state_dict(
            batch_shape=(n_runs,), n_atoms=n_atoms
        )
        p = tmp_path / "par.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for r in range(n_runs):
            n_dead = result[r].n_dead
            assert result[r].dead_positions.shape == (n_dead, n_atoms, 3)

    def test_log_evidence_matches_per_run(self, tmp_path):
        n_runs = 3
        state = _make_batched_ns_state_dict(batch_shape=(n_runs,))
        p = tmp_path / "par.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for r in range(n_runs):
            expected = float(np.asarray(state["log_evidence"])[r])
            assert abs(result[r].log_evidence - expected) < 1e-4

    def test_iteration_matches_per_run(self, tmp_path):
        n_runs = 3
        state = _make_batched_ns_state_dict(batch_shape=(n_runs,))
        p = tmp_path / "par.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for r in range(n_runs):
            expected = int(np.asarray(state["iteration"])[r])
            assert result[r].iteration == expected

    def test_dead_volumes_none_for_nvt(self, tmp_path):
        state = _make_batched_ns_state_dict(batch_shape=(2,), npt=False)
        p = tmp_path / "par_nvt.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for b in result:
            assert b.dead_volumes is None

    def test_dead_volumes_array_for_npt(self, tmp_path):
        state = _make_batched_ns_state_dict(batch_shape=(2,), npt=True)
        p = tmp_path / "par_npt.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for b in result:
            assert b.dead_volumes is not None
            assert b.dead_volumes.shape == (b.n_dead,)

    def test_run_ns_parallel_roundtrip(self, tmp_path):
        """load_restart on a parallel checkpoint feeds directly into run_ns_parallel."""
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns_parallel
        from jaxrens.sampling.termination import IterationTermination

        n_runs = 2
        n_walkers = 6
        n_atoms = 1
        n_dead_checkpoint = 3  # must be < max_iterations used below
        max_iterations = 8

        rng = np.random.default_rng(77)
        positions = rng.uniform(-1, 1, (n_runs, n_walkers, n_atoms, 3)).astype(
            np.float32
        )
        energies = rng.uniform(0, 2, (n_runs, n_walkers)).astype(np.float32)
        types = np.zeros(n_atoms, dtype=np.int32)

        # Build a parallel checkpoint with exactly n_dead_checkpoint dead points per run.
        dead_energies = rng.uniform(5, 10, (n_runs, n_dead_checkpoint)).astype(
            np.float32
        )
        dead_positions = rng.uniform(
            -1, 1, (n_runs, n_dead_checkpoint, n_atoms, 3)
        ).astype(np.float32)

        state: dict = {
            "positions": positions,
            "types": np.zeros((n_runs, n_walkers, n_atoms), dtype=np.int32),
            "energies": energies,
            "cells": np.zeros((n_runs, n_walkers, 3, 3), dtype=np.float32),
            "dead_energies": dead_energies,
            "dead_positions": dead_positions,
            "dead_volumes": None,
            "live_volumes": None,
            "log_evidence": np.full((n_runs,), -6.0, dtype=np.float64),
            "iteration": np.full((n_runs,), n_dead_checkpoint, dtype=np.int32),
            "n_dead": np.full((n_runs,), n_dead_checkpoint, dtype=np.int32),
            "n_walkers": n_walkers,
            "rng_key": jax.random.key(0),
        }

        p = tmp_path / "par_rtrip.checkpoint.h5"
        save_checkpoint(p, state)
        restart_bundles = load_restart(p)

        assert isinstance(restart_bundles, list)
        assert len(restart_bundles) == n_runs

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, [_rw_descriptor()])
        keys = jax.random.split(jax.random.key(42), n_runs)

        out = run_ns_parallel(
            jnp.asarray(positions),
            jnp.asarray(types),
            jnp.asarray(energies),
            None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=keys,
            max_iterations=max_iterations,
            n_mcmc_steps=2,
            termination_criteria=[IterationTermination(max_iterations)],
            restart_states=restart_bundles,
        )
        assert jnp.all(jnp.isfinite(out["log_evidence"]))
        assert out["log_evidence"].shape == (n_runs,)


class TestLoadRestartMultiGpu:
    """load_restart on a (G, P)-shaped checkpoint → list[list[RestartBundle]]."""

    def test_returns_nested_list(self, tmp_path):
        G, P = 1, 2
        state = _make_batched_ns_state_dict(batch_shape=(G, P))
        p = tmp_path / "mgpu.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        assert isinstance(result, list)
        assert len(result) == G
        for gpu_list in result:
            assert isinstance(gpu_list, list)
            assert len(gpu_list) == P

    def test_each_element_is_restart_bundle(self, tmp_path):
        G, P = 1, 2
        state = _make_batched_ns_state_dict(batch_shape=(G, P))
        p = tmp_path / "mgpu.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for g in range(G):
            for pp in range(P):
                assert isinstance(result[g][pp], RestartBundle)

    def test_n_dead_matches_per_slot(self, tmp_path):
        G, P = 1, 2
        state = _make_batched_ns_state_dict(batch_shape=(G, P))
        p = tmp_path / "mgpu.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for g in range(G):
            for pp in range(P):
                expected = int(np.asarray(state["n_dead"])[g, pp])
                assert result[g][pp].n_dead == expected

    def test_dead_energies_shape_per_slot(self, tmp_path):
        G, P = 1, 2
        state = _make_batched_ns_state_dict(batch_shape=(G, P))
        p = tmp_path / "mgpu.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for g in range(G):
            for pp in range(P):
                n_dead = result[g][pp].n_dead
                assert result[g][pp].dead_energies.shape == (n_dead,)

    def test_log_evidence_matches_per_slot(self, tmp_path):
        G, P = 1, 2
        state = _make_batched_ns_state_dict(batch_shape=(G, P))
        p = tmp_path / "mgpu.checkpoint.h5"
        save_checkpoint(p, state)
        result = load_restart(p)
        for g in range(G):
            for pp in range(P):
                expected = float(np.asarray(state["log_evidence"])[g, pp])
                assert abs(result[g][pp].log_evidence - expected) < 1e-4

    def test_topology_mismatch_raises(self, tmp_path, monkeypatch):
        """Restarting a multi-GPU checkpoint on a host with a different
        device count must raise with a clear message — silent topology
        coercion is not supported (per §D plan)."""
        # Save a (G=2, P=2) checkpoint.
        G, P = 2, 2
        state = _make_batched_ns_state_dict(batch_shape=(G, P))
        p = tmp_path / "wrong_topology.checkpoint.h5"
        save_checkpoint(p, state)

        # Force ``len(jax.local_devices())`` to disagree with G.
        import jax as _jax

        monkeypatch.setattr(
            _jax, "local_devices", lambda *_a, **_k: [object()]
        )

        with pytest.raises(ValueError, match="Cross-topology restart is not"):
            load_restart(p)

    def test_run_ns_multi_gpu_roundtrip(self, tmp_path):
        """load_restart on a (G, P) checkpoint feeds into run_ns_multi_gpu."""
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns_multi_gpu
        from jaxrens.sampling.termination import IterationTermination

        n_gpu, n_per_gpu = 1, 2
        n_total = n_gpu * n_per_gpu
        n_walkers = 8
        n_atoms = 1
        n_dead_checkpoint = 3  # must be < max_iterations used below
        max_iterations = 8

        rng = np.random.default_rng(88)
        positions = rng.uniform(
            -1, 1, (n_total, n_walkers, n_atoms, 3)
        ).astype(np.float32)
        energies = rng.uniform(0, 2, (n_total, n_walkers)).astype(np.float32)
        types = np.zeros(n_atoms, dtype=np.int32)

        # Build a (G, P) checkpoint with deterministic n_dead per slot.
        dead_energies = rng.uniform(
            5, 10, (n_gpu, n_per_gpu, n_dead_checkpoint)
        ).astype(np.float32)
        dead_positions = rng.uniform(
            -1, 1, (n_gpu, n_per_gpu, n_dead_checkpoint, n_atoms, 3)
        ).astype(np.float32)

        state: dict = {
            "positions": positions.reshape(
                n_gpu, n_per_gpu, n_walkers, n_atoms, 3
            ),
            "types": np.zeros(
                (n_gpu, n_per_gpu, n_walkers, n_atoms), dtype=np.int32
            ),
            "energies": energies.reshape(n_gpu, n_per_gpu, n_walkers),
            "cells": np.zeros(
                (n_gpu, n_per_gpu, n_walkers, 3, 3), dtype=np.float32
            ),
            "dead_energies": dead_energies,
            "dead_positions": dead_positions,
            "dead_volumes": None,
            "live_volumes": None,
            "log_evidence": np.full(
                (n_gpu, n_per_gpu), -6.0, dtype=np.float64
            ),
            "iteration": np.full(
                (n_gpu, n_per_gpu), n_dead_checkpoint, dtype=np.int32
            ),
            "n_dead": np.full(
                (n_gpu, n_per_gpu), n_dead_checkpoint, dtype=np.int32
            ),
            "n_walkers": n_walkers,
            "rng_key": jax.random.key(0),
        }

        p = tmp_path / "mgpu_rtrip.checkpoint.h5"
        save_checkpoint(p, state)
        restart_bundles = load_restart(p)

        assert isinstance(restart_bundles, list)
        assert len(restart_bundles) == n_gpu
        assert len(restart_bundles[0]) == n_per_gpu

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, [_rw_descriptor()])
        keys = jax.random.split(jax.random.key(55), n_total)

        out = run_ns_multi_gpu(
            jnp.asarray(positions),
            jnp.asarray(types),
            jnp.asarray(energies),
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=keys,
            n_gpu=n_gpu,
            n_per_gpu=n_per_gpu,
            max_iterations=max_iterations,
            n_mcmc_steps=2,
            convergence_threshold=1e6,
            restart_states=restart_bundles,
        )
        assert jnp.all(jnp.isfinite(out["log_evidence"]))
        assert out["log_evidence"].shape == (n_gpu, n_per_gpu)


class TestLoadRestartShapeError:
    """load_restart raises ValueError for ndim >= 3 checkpoints."""

    def test_ndim3_raises_value_error(self, tmp_path):
        import h5py

        # Build a synthetic file with a 3-D log_evidence array.
        p = tmp_path / "bad_ndim3.checkpoint.h5"
        rng = np.random.default_rng(0)
        n_walkers, n_atoms = 4, 1
        shape3d = (1, 2, 3)  # 3-D batch prefix

        with h5py.File(p, "w") as f:
            f.create_dataset(
                "positions",
                data=rng.uniform(
                    0, 1, shape3d + (n_walkers, n_atoms, 3)
                ).astype(np.float32),
            )
            f.create_dataset(
                "types",
                data=np.zeros(shape3d + (n_walkers, n_atoms), dtype=np.int32),
            )
            f.create_dataset(
                "energies",
                data=rng.uniform(0, 1, shape3d + (n_walkers,)).astype(
                    np.float32
                ),
            )
            f.create_dataset(
                "cells",
                data=np.zeros(shape3d + (n_walkers, 3, 3), dtype=np.float32),
            )
            f.create_dataset(
                "dead_energies",
                data=rng.uniform(0, 1, shape3d + (5,)).astype(np.float32),
            )
            f.create_dataset(
                "dead_positions",
                data=rng.uniform(0, 1, shape3d + (5, n_atoms, 3)).astype(
                    np.float32
                ),
            )
            # log_evidence, iteration, n_dead stored as 3-D datasets
            f.create_dataset(
                "log_evidence",
                data=rng.uniform(-10, -1, shape3d).astype(np.float64),
            )
            f.create_dataset(
                "iteration", data=np.ones(shape3d, dtype=np.int32) * 5
            )
            f.create_dataset(
                "n_dead", data=np.ones(shape3d, dtype=np.int32) * 3
            )
            f.attrs["n_walkers"] = n_walkers

        with pytest.raises(ValueError, match="ndim"):
            load_restart(p)

    def test_ndim3_error_message_has_shape(self, tmp_path):
        import h5py

        p = tmp_path / "bad_ndim3_msg.checkpoint.h5"
        rng = np.random.default_rng(1)
        shape3d = (2, 3, 4)

        with h5py.File(p, "w") as f:
            f.create_dataset(
                "positions",
                data=np.zeros(shape3d + (4, 1, 3), dtype=np.float32),
            )
            f.create_dataset(
                "types", data=np.zeros(shape3d + (4, 1), dtype=np.int32)
            )
            f.create_dataset(
                "energies", data=np.zeros(shape3d + (4,), dtype=np.float32)
            )
            f.create_dataset(
                "cells", data=np.zeros(shape3d + (4, 3, 3), dtype=np.float32)
            )
            f.create_dataset(
                "dead_energies",
                data=np.zeros(shape3d + (5,), dtype=np.float32),
            )
            f.create_dataset(
                "dead_positions",
                data=np.zeros(shape3d + (5, 1, 3), dtype=np.float32),
            )
            f.create_dataset(
                "log_evidence", data=rng.uniform(-10, -1, shape3d)
            )
            f.create_dataset(
                "iteration", data=np.ones(shape3d, dtype=np.int32)
            )
            f.create_dataset("n_dead", data=np.ones(shape3d, dtype=np.int32))
            f.attrs["n_walkers"] = 4

        # New format (post-§D): error mentions the shape and rank explicitly.
        with pytest.raises(ValueError, match=r"rank 3|\(2, 3, 4\)"):
            load_restart(p)


class TestSchemaDrift:
    """If a field is added to RestartBundle, load_restart must still cover it."""

    def test_all_dataclass_fields_present_in_scalar_bundle(self, tmp_path):
        """Every public field of RestartBundle appears in a loaded scalar bundle."""
        import dataclasses

        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        _, bundle = load_restart(p)

        bundle_field_names = {
            f.name for f in dataclasses.fields(RestartBundle)
        }
        for name in bundle_field_names:
            assert hasattr(bundle, name), (
                f"RestartBundle field '{name}' missing from loaded scalar bundle. "
                f"If you added a new field to RestartBundle, update load_restart "
                f"(specifically _build_bundle_from_ckpt) to populate it."
            )

    def test_all_dataclass_fields_present_in_parallel_bundle(self, tmp_path):
        """Every public field of RestartBundle appears in a loaded parallel (ndim=1) bundle."""
        import dataclasses

        n_runs = 2
        state = _make_batched_ns_state_dict(batch_shape=(n_runs,))
        p = tmp_path / "schema_par.checkpoint.h5"
        save_checkpoint(p, state)
        bundles = load_restart(p)

        bundle_field_names = {
            f.name for f in dataclasses.fields(RestartBundle)
        }
        for b in bundles:
            for name in bundle_field_names:
                assert hasattr(b, name), (
                    f"RestartBundle field '{name}' missing from loaded parallel bundle. "
                    f"Update _build_bundle_from_ckpt if you added a new field."
                )

    def test_all_dataclass_fields_present_in_multi_gpu_bundle(self, tmp_path):
        """Every public field of RestartBundle appears in a loaded multi-GPU (ndim=2) bundle."""
        import dataclasses

        G, P = 1, 2
        state = _make_batched_ns_state_dict(batch_shape=(G, P))
        p = tmp_path / "schema_mgpu.checkpoint.h5"
        save_checkpoint(p, state)
        bundles = load_restart(p)

        bundle_field_names = {
            f.name for f in dataclasses.fields(RestartBundle)
        }
        for g in range(G):
            for pp in range(P):
                b = bundles[g][pp]
                for name in bundle_field_names:
                    assert hasattr(b, name), (
                        f"RestartBundle field '{name}' missing from multi-GPU bundle "
                        f"at [{g}][{pp}]. Update _build_bundle_from_ckpt."
                    )


class TestInferRestartShape:
    """Unit tests for infer_restart_shape helper."""

    def test_single_run_tuple(self, tmp_path):
        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state)
        loaded = load_restart(p)
        assert infer_restart_shape(loaded) == "single"

    def test_parallel_list(self, tmp_path):
        state = _make_batched_ns_state_dict(batch_shape=(3,))
        p = tmp_path / "inf_par.checkpoint.h5"
        save_checkpoint(p, state)
        loaded = load_restart(p)
        assert infer_restart_shape(loaded) == "parallel"

    def test_multi_gpu_nested_list(self, tmp_path):
        state = _make_batched_ns_state_dict(batch_shape=(1, 2))
        p = tmp_path / "inf_mgpu.checkpoint.h5"
        save_checkpoint(p, state)
        loaded = load_restart(p)
        assert infer_restart_shape(loaded) == "multi_gpu"

    def test_invalid_type_raises_type_error(self):
        from jaxrens.init.restart import infer_restart_shape

        with pytest.raises(TypeError, match="unrecognised"):
            infer_restart_shape("not_a_bundle")


# ---------------------------------------------------------------------------
# load_restart_batched: multi-replica restart loader
# ---------------------------------------------------------------------------


class TestLoadRestartBatched:
    """Multi-replica loader returning flat (n_total, ...) arrays + 2-D bundles."""

    def test_vmap_1d_checkpoint_returns_single_gpu_layout(self, tmp_path):
        from jaxrens.init.restart import load_restart_batched

        n_runs = 4
        state = _make_batched_ns_state_dict(
            batch_shape=(n_runs,), n_walkers=3, n_atoms=2
        )
        p = tmp_path / "vmap.checkpoint.h5"
        save_checkpoint(p, state)

        batched = load_restart_batched(p)
        assert batched.n_gpu == 1
        assert batched.n_per_gpu == n_runs
        assert batched.n_total == n_runs
        assert batched.positions.shape == (n_runs, 3, 2, 3)
        assert batched.cells.shape == (n_runs, 3, 3, 3)
        assert batched.types.shape == (n_runs, 3, 2)
        assert len(batched.bundles_2d) == 1
        assert len(batched.bundles_2d[0]) == n_runs
        assert all(isinstance(b, RestartBundle) for b in batched.bundles_2d[0])

    def test_pmap_2d_checkpoint_preserves_g_p_split(self, tmp_path):
        from jaxrens.init.restart import load_restart_batched

        G, P = 1, 4  # n_gpu=1 to avoid the cross-topology guard on this host.
        state = _make_batched_ns_state_dict(
            batch_shape=(G, P), n_walkers=3, n_atoms=2
        )
        p = tmp_path / "pmap.checkpoint.h5"
        save_checkpoint(p, state)

        batched = load_restart_batched(p)
        assert batched.n_gpu == G
        assert batched.n_per_gpu == P
        assert batched.n_total == G * P
        assert batched.positions.shape == (G * P, 3, 2, 3)
        assert len(batched.bundles_2d) == G
        assert len(batched.bundles_2d[0]) == P

    def test_bundles_flat_round_trips_through_bundles_2d(self, tmp_path):
        from jaxrens.init.restart import load_restart_batched

        state = _make_batched_ns_state_dict(batch_shape=(1, 6), n_walkers=2)
        p = tmp_path / "rt.checkpoint.h5"
        save_checkpoint(p, state)

        batched = load_restart_batched(p)
        flat = batched.bundles_flat
        assert len(flat) == batched.n_total
        rebuilt = [
            flat[g * batched.n_per_gpu : (g + 1) * batched.n_per_gpu]
            for g in range(batched.n_gpu)
        ]
        assert rebuilt == batched.bundles_2d

    def test_single_run_checkpoint_rejected(self, tmp_path):
        from jaxrens.init.restart import load_restart_batched

        state = _make_ns_state_dict(n_walkers=4, n_atoms=1, n_dead=3)
        p = _write_checkpoint(tmp_path, state, name="scalar.checkpoint.h5")

        with pytest.raises(ValueError, match="scalar single-run"):
            load_restart_batched(p)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from jaxrens.init.restart import load_restart_batched

        with pytest.raises(FileNotFoundError):
            load_restart_batched(tmp_path / "nope.h5")

    def test_per_slot_bundles_independent(self, tmp_path):
        """Each bundle carries its own iteration / n_dead, not a single shared one."""
        from jaxrens.init.restart import load_restart_batched

        state = _make_batched_ns_state_dict(batch_shape=(1, 3), n_walkers=2)
        # Stamp distinct iteration values so we can tell slots apart.
        state["iteration"] = np.array([[10, 20, 30]], dtype=np.int32)
        state["n_dead"] = np.array([[5, 6, 7]], dtype=np.int32)
        p = tmp_path / "per_slot.checkpoint.h5"
        save_checkpoint(p, state)

        batched = load_restart_batched(p)
        flat = batched.bundles_flat
        assert flat[0].iteration == 10
        assert flat[1].iteration == 20
        assert flat[2].iteration == 30
        assert flat[0].n_dead == 5
        assert flat[1].n_dead == 6
        assert flat[2].n_dead == 7
