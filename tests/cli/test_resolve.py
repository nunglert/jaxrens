"""Tests for the resolver in jaxrens.cli.resolve.

``resolve`` turns a validated ``RootSpec`` into a ``ResolvedConfig``: the
library dataclasses, the initial walkers with their energies, and the
batcher. The file also tests the schema methods the resolver calls
(``to_descriptor``, ``to_move_config``, ``to_backend_config``,
``build_backend``, ``to_criterion``, ``resolve_for``,
``to_ensemble_params``) and ``_resolve_init`` for each init mode.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ase
import ase.io
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import jaxrens.cli.resolve as _r
from jaxrens.backends.base import BackendResult
from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.backends.lj import LJBackend
from jaxrens.backends.toy import (
    DoubleWellBackend,
    GaussianMixtureBackend,
    HarmonicBackend,
    create_harmonic,
)
from jaxrens.cli.resolve import (
    ResolvedConfig,
    _finalise_initial_energies_and_counts,
    _resolve_init,
    resolve,
)
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.adaptation import (
    _FALLBACK_ADJUST_FACTOR,
    _FALLBACK_MAX_RATE,
    _FALLBACK_MIN_RATE,
    _FALLBACK_STEP_SIZE_MAX,
    AdaptationPolicy,
    AdaptationSpec,
)
from jaxrens.cli.schema.backend import (
    DoubleWellBackendSpec,
    GaussianMixtureBackendSpec,
    HarmonicBackendSpec,
    LJBackendSpec,
    MACEBackendSpec,
    NeuralILBackendSpec,
)
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.ensemble import NPTEnsembleSpec, NVTEnsembleSpec
from jaxrens.cli.schema.init import InitSpec
from jaxrens.cli.schema.moves import (
    AlchemicalMorphMoveSpec,
    GMCMoveSpec,
    HMCMoveSpec,
    RandomWalkMoveSpec,
    SingleAtomMoveSpec,
    VolumeMoveSpec,
)
from jaxrens.cli.schema.termination import (
    EnergyTerminationSpec,
    IterationTerminationSpec,
    PriorMassTerminationSpec,
    TemperatureTerminationSpec,
)
from jaxrens.init.restart import RestartBundle
from jaxrens.init.walker_set import load_walker_set
from jaxrens.io.checkpoint import save_checkpoint
from jaxrens.sampling.batch_descriptor import (
    PmapVmapRuns,
    ShardedSingleRun,
    SingleRun,
)
from jaxrens.sampling.nested_sampling import _choose_starting_bucket
from jaxrens.sampling.termination import (
    EnergyTermination,
    IterationTermination,
    PriorMassTermination,
    TempTermination,
)
from jaxrens.state.config import (
    BackendConfig,
    MoveConfig,
    NSConfig,
    OutputConfig,
)
from jaxrens.utils.cell import get_volume

_DATA = Path(__file__).parent.parent / "_assets" / "data" / "cli"
_MINIMAL_YAML = _DATA / "minimal.yaml"
_LJ_BACKEND_YAML = _DATA / "lj_backend.yaml"
_FULL_CONFIG_YAML = _DATA / "full_config.yaml"

_GPA_TO_EVA3 = 0.006241509  # 1 GPa in eV/Å³

_PERMISSIVE_CELL = CellSpec(
    max_volume_per_atom=10000.0,
    min_volume_per_atom=0.01,
    min_aspect_ratio=0.001,
)


def _minimal_dict() -> dict:
    return {
        "run": {
            "n_live": 20,
            "max_iterations": 50,
            "n_mcmc_steps": 5,
            "seed": 0,
        },
        "moves": [
            {"move_type": "random_walk", "step_size": 0.3},
        ],
        "backend": {
            "backend_type": "harmonic",
        },
        "output": {
            "format": "none",
            "working_dir": ".",
            "info_interval": 999,
        },
    }


def _species_dict(
    n_atoms: int = 4, n_live: int = 8, mode: str = "grid"
) -> dict:
    return {
        "run": {
            "n_live": n_live,
            "max_iterations": 5,
            "n_mcmc_steps": 3,
            "seed": 0,
        },
        "moves": [{"move_type": "random_walk", "step_size": 0.3}],
        "backend": {
            "backend_type": "harmonic",
        },
        "output": {
            "format": "none",
            "working_dir": ".",
            "info_interval": 999,
        },
        "init": {
            "start_species": f"1 {n_atoms}",
            "random_initialise_pos": True,
            "random_initialise_cell": False,
            "pos_randomization_mode": mode,
            "grid_distance": 1.5,
            "init_distance_criterion": 0.5,
            "random_init_max_n_tries": 50,
            "start_energy_ceiling_per_atom": 1e6,
            "pos_autoscale_cells": False,
        },
    }


def _sharded_root(tmp_path, *, shard_n_gpu, n_live=8, pressures=None):
    """Validate a small LJ NPT config through RootSpec so resolve() is fast."""
    cfg = {
        "run": {
            "n_live": n_live,
            "max_iterations": 1,
            "n_mcmc_steps": 1,
            "seed": 0,
        },
        "backend": {
            "type": "lj",
            "epsilon": 1.0,
            "sigma": 1.0,
            "cutoff": 2.5,
            "periodic": True,
        },
        "ensemble": {
            "type": "npt",
            "pressure": pressures if pressures is not None else 0.1,
            "pressure_units": "eva3",
        },
        "moves": [
            {"type": "random_walk", "step_size": 0.1},
        ],
        "termination": [{"type": "iteration", "max_iterations": 1}],
        "init": {
            "start_species": "18 4",
            "random_initialise_pos": True,
            "pos_randomization_mode": "grid",
            "grid_distance": 1.0,
            "random_initialise_cell": False,
            "start_energy_ceiling_per_atom": 100.0,
        },
        "cell": {
            "max_volume_per_atom": 30.0,
            "min_volume_per_atom": 1.5,
            "min_aspect_ratio": 0.6,
            "flat_V_prior": False,
        },
        "output": {
            "format": "extxyz",
            "working_dir": str(tmp_path / "out"),
            "out_file_prefix": "test",
            "info_interval": 50,
            "traj_interval": 50,
            "snapshot_interval": 100000,
            "checkpoint_interval": 100000,
        },
    }
    if shard_n_gpu != 1:
        cfg["run"]["shard_n_gpu"] = shard_n_gpu
    return RootSpec.model_validate(cfg)


class TestResolve:
    """Check that resolve() builds the expected library dataclasses."""

    def test_resolve_values_match_hand_built(self):
        d = _minimal_dict()
        root = RootSpec.model_validate(d)
        resolved = resolve(root)

        expected_ns = NSConfig(
            n_live=20,
            max_iterations=50,
            n_mcmc_steps=5,
            seed=0,
            convergence_threshold=0.1,
            n_cull=1,
        )
        expected_move = MoveConfig(
            move_type="random_walk",
            step_size=0.3,
            n_steps=10,
            weight=1.0,
            adaptation_warmup=100,
            target_acceptance=0.5,
        )
        expected_backend = BackendConfig(
            backend_type="harmonic",
            periodic=False,
            cutoff=None,
            checkpoint_path=None,
            max_neighbors_list=[30, 35, 40, 45, 50],
            max_neighbors_offset=5,
        )
        expected_output = OutputConfig(
            format="none",
            working_dir=Path("."),
            info_interval=999,
            traj_interval=1,
            snapshot_interval=100,
            checkpoint_interval=100,
            out_file_prefix="ns",
        )

        assert resolved.ns == expected_ns
        assert resolved.moves == (expected_move,)
        assert resolved.backend == expected_backend
        assert resolved.output == expected_output

    def test_resolve_multi_moves(self):
        d = _minimal_dict()
        d["moves"] = [
            {"move_type": "random_walk", "step_size": 0.2, "weight": 0.7},
            {"move_type": "galilean", "step_size": 0.05, "weight": 0.3},
        ]
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        assert len(resolved.moves) == 2
        assert resolved.moves[0].move_type == "random_walk"
        assert resolved.moves[1].move_type == "gmc"


class TestToDescriptor:
    """Check the kernel kwargs, step size, weight and name that
    to_descriptor() puts on each move's MoveKernel.
    """

    def test_random_walk_descriptor(self):
        spec = RandomWalkMoveSpec(step_size=0.2, weight=2.0)
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {}
        assert desc.step_size == pytest.approx(0.2)
        assert desc.weight == pytest.approx(2.0)
        assert desc.extra_state_fields == {}

    def test_hmc_descriptor_n_leapfrog(self):
        spec = HMCMoveSpec(n_leapfrog=15)
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {"n_leapfrog": 15}
        assert desc.extra_state_fields == {}

    def test_single_atom_descriptor(self):
        spec = SingleAtomMoveSpec()
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {}

    def test_volume_descriptor(self):
        cell_cfg = CellSpec(
            max_volume_per_atom=50.0,
            min_volume_per_atom=1.0,
            min_aspect_ratio=0.5,
            flat_V_prior=False,
        )
        spec = VolumeMoveSpec()
        desc = spec.to_descriptor(n_atoms=10, cell_cfg=cell_cfg)
        assert desc.kernel_kwargs["n_atoms"] == 10
        assert desc.kernel_kwargs["max_vol_per_atom"] == pytest.approx(50.0)

    def test_volume_descriptor_no_n_atoms_raises(self):
        spec = VolumeMoveSpec()
        with pytest.raises(ValueError, match="n_atoms"):
            spec.to_descriptor()

    def test_volume_descriptor_no_cell_cfg_raises(self):
        spec = VolumeMoveSpec()
        with pytest.raises(ValueError, match="cell_cfg"):
            spec.to_descriptor(n_atoms=10)

    def test_alchemical_morph_descriptor(self):
        spec = AlchemicalMorphMoveSpec(n_species=3)
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {"n_species": 3}

    def test_name_defaults_to_type(self):
        spec = RandomWalkMoveSpec()
        desc = spec.to_descriptor()
        assert desc.name == "random_walk"

    def test_name_override(self):
        spec = GMCMoveSpec(name="gal_heavy", n_reflect=5)
        desc = spec.to_descriptor()
        assert desc.name == "gal_heavy"


class TestToMoveConfig:
    """Check that to_move_config() maps each move spec onto MoveConfig."""

    def test_random_walk_move_config(self):
        spec = RandomWalkMoveSpec(
            step_size=0.3, weight=2.0, adaptation_warmup=50
        )
        mc = spec.to_move_config()
        assert isinstance(mc, MoveConfig)
        assert mc.move_type == "random_walk"
        assert mc.step_size == pytest.approx(0.3)
        assert mc.weight == pytest.approx(2.0)
        assert mc.adaptation_warmup == 50

    def test_galilean_n_steps_maps_to_n_reflect(self):
        spec = GMCMoveSpec(n_reflect=12)
        mc = spec.to_move_config()
        assert mc.n_steps == 12

    def test_hmc_n_steps_maps_to_n_leapfrog(self):
        spec = HMCMoveSpec(n_leapfrog=20)
        mc = spec.to_move_config()
        assert mc.n_steps == 20


def test_resolve_descriptor_matches_move():
    d = _minimal_dict()
    d["moves"] = [{"type": "hmc", "n_leapfrog": 6, "step_size": 0.02}]
    root = RootSpec.model_validate(d)
    resolved = resolve(root)
    desc = resolved.move_descriptors[0]
    assert desc.kernel_kwargs["n_leapfrog"] == 6
    assert desc.step_size == pytest.approx(0.02)


class TestToBackendConfig:
    """Check the BackendConfig that to_backend_config() builds for each
    backend spec.
    """

    def test_harmonic_to_backend_config(self):
        spec = HarmonicBackendSpec(periodic=True)
        cfg = spec.to_backend_config()
        assert isinstance(cfg, BackendConfig)
        assert cfg.backend_type == "harmonic"
        assert cfg.periodic is True
        assert cfg.cutoff is None
        assert cfg.checkpoint_path is None
        assert cfg.max_neighbors_list == [30, 35, 40, 45, 50]
        assert cfg.max_neighbors_offset == 5

    def test_lj_to_backend_config_with_cutoff(self):
        spec = LJBackendSpec(cutoff=3.5)
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "lj"
        assert cfg.cutoff == pytest.approx(3.5)
        assert cfg.checkpoint_path is None

    def test_lj_to_backend_config_no_cutoff(self):
        spec = LJBackendSpec()
        cfg = spec.to_backend_config()
        assert cfg.cutoff is None

    def test_double_well_to_backend_config(self):
        spec = DoubleWellBackendSpec()
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "double_well"

    def test_gaussian_mixture_to_backend_config(self):
        spec = GaussianMixtureBackendSpec()
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "gaussian_mixture"

    def test_neuralil_to_backend_config(self):
        spec = NeuralILBackendSpec(
            checkpoint_path="/tmp/model.pkl",
            max_neighbors_list=[20, 25, 30],
            max_neighbors_offset=3,
        )
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "neuralil"
        assert cfg.checkpoint_path == "/tmp/model.pkl"
        assert cfg.max_neighbors_list == [20, 25, 30]
        assert cfg.max_neighbors_offset == 3

    def test_mace_to_backend_config(self):
        spec = MACEBackendSpec(checkpoint_path="/tmp/mace_model")
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "mace"
        assert cfg.checkpoint_path == "/tmp/mace_model"


class TestBuildBackend:
    """Check that build_backend() builds each lightweight backend with the
    spec's parameters.
    """

    def test_harmonic_build_backend(self):
        spec = HarmonicBackendSpec(k=2.0)
        backend = spec.build_backend()
        assert isinstance(backend, HarmonicBackend)
        assert backend.k == pytest.approx(2.0)

    def test_double_well_build_backend(self):
        spec = DoubleWellBackendSpec(a=0.5, b=2.0)
        backend = spec.build_backend()
        assert isinstance(backend, DoubleWellBackend)
        assert backend.a == pytest.approx(0.5)
        assert backend.b == pytest.approx(2.0)

    def test_gaussian_mixture_build_backend(self):
        spec = GaussianMixtureBackendSpec(sigma=0.3)
        backend = spec.build_backend()
        assert isinstance(backend, GaussianMixtureBackend)
        assert backend.sigma == pytest.approx(0.3)

    def test_lj_build_backend(self):
        spec = LJBackendSpec(epsilon=0.5, sigma=1.2, cutoff=4.0)
        backend = spec.build_backend()
        assert isinstance(backend, LJBackend)
        assert backend.epsilon == pytest.approx(0.5)
        assert backend.sigma == pytest.approx(1.2)
        assert backend.cutoff == pytest.approx(4.0)


class TestResolveEnergyBackend:
    """Check that ResolvedConfig.base_backend wraps the configured model."""

    def test_resolve_base_backend_is_harmonic(self):
        root = RootSpec.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert isinstance(resolved.base_backend.model, HarmonicBackend)

    def test_resolve_base_backend_is_callable(self):
        root = RootSpec.model_validate(_minimal_dict())
        resolved = resolve(root)
        positions = jnp.zeros((1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energy = resolved.base_backend(positions, types, cell, 0).energy
        assert jnp.isfinite(energy)

    def test_resolve_base_backend_lj(self):
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved.base_backend.model, LJBackend)


class TestToCriterion:
    """Check to_criterion() for each termination spec, and the criteria
    resolve() builds from the termination list and run.max_iterations.
    """

    def test_to_criterion_iteration(self):
        spec = IterationTerminationSpec(max_iterations=42)
        crit = spec.to_criterion()
        assert isinstance(crit, IterationTermination)
        assert crit.max_iterations == 42

    def test_to_criterion_prior_mass(self):
        spec = PriorMassTerminationSpec(threshold=0.05)
        crit = spec.to_criterion(n_live=30, n_cull=1)
        assert isinstance(crit, PriorMassTermination)
        assert crit.n_live == 30
        assert crit.threshold == pytest.approx(0.05)

    def test_to_criterion_prior_mass_requires_n_live(self):
        spec = PriorMassTerminationSpec(threshold=0.05)
        with pytest.raises(ValueError, match="requires n_live"):
            spec.to_criterion()

    def test_to_criterion_temperature(self):
        spec = TemperatureTerminationSpec(target_temp=500.0, threshold=8.0)
        crit = spec.to_criterion(n_live=20, n_cull=2)
        assert isinstance(crit, TempTermination)
        assert crit.target_temp == pytest.approx(500.0)
        assert crit.n_cull == 2
        assert crit.threshold == pytest.approx(8.0)

    def test_to_criterion_energy(self):
        spec = EnergyTerminationSpec(min_energy=-3.5)
        crit = spec.to_criterion()
        assert isinstance(crit, EnergyTermination)
        assert crit.min_energy == pytest.approx(-3.5)

    def test_termination_none_resolves_to_prior_mass_and_iteration(self):
        root = RootSpec.model_validate(_minimal_dict())
        assert root.termination is None
        resolved = resolve(root)
        assert len(resolved.termination) == 2
        types_found = {type(c) for c in resolved.termination}
        assert IterationTermination in types_found
        assert PriorMassTermination in types_found

    def test_termination_none_prior_mass_uses_convergence_threshold(self):
        d = _minimal_dict()
        d["run"]["convergence_threshold"] = 0.05
        d["run"]["n_live"] = 15
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        pm = next(
            c
            for c in resolved.termination
            if isinstance(c, PriorMassTermination)
        )
        assert pm.threshold == pytest.approx(0.05)
        assert pm.n_live == 15

    def test_no_termination_and_no_max_iterations_gives_prior_mass_only(
        self,
    ):
        d = _minimal_dict()
        d["run"].pop("max_iterations")
        resolved = resolve(RootSpec.model_validate(d))
        assert len(resolved.termination) == 1
        assert isinstance(resolved.termination[0], PriorMassTermination)

    def test_two_criteria_resolve_to_two_element_tuple(self):
        d = _minimal_dict()
        # Without run.max_iterations the resolver adds no third criterion.
        d["run"].pop("max_iterations", None)
        d["termination"] = [
            {"type": "iteration", "max_iterations": 50},
            {"type": "energy", "min_energy": -99.0},
        ]
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        assert len(resolved.termination) == 2

    def test_max_iterations_appended_to_explicit_list(self):
        d = _minimal_dict()
        d["termination"] = [{"type": "energy", "min_energy": -99.0}]
        resolved = resolve(RootSpec.model_validate(d))
        energy, iteration = resolved.termination
        assert isinstance(energy, EnergyTermination)
        assert isinstance(iteration, IterationTermination)
        assert iteration.max_iterations == 50

    def test_termination_iteration_fixture_yaml(self):
        fixture = _DATA / "termination_iteration.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        assert root.termination is not None
        assert len(root.termination) == 1
        assert isinstance(root.termination[0], IterationTerminationSpec)
        assert root.termination[0].max_iterations == 5


class TestAdaptationResolve:
    """Check how AdaptationSpec.resolve_for() layers per-move overrides on
    the defaults, and the per-move policies resolve() returns.
    """

    def test_resolve_for_no_override_uses_fallbacks(self):
        cfg = AdaptationSpec()
        policy = cfg.resolve_for("random_walk")
        assert policy.min_rate == pytest.approx(_FALLBACK_MIN_RATE)
        assert policy.max_rate == pytest.approx(_FALLBACK_MAX_RATE)
        assert policy.adjust_factor == pytest.approx(_FALLBACK_ADJUST_FACTOR)
        assert policy.step_size_max == pytest.approx(_FALLBACK_STEP_SIZE_MAX)

    def test_resolve_for_with_defaults_min_rate(self):
        cfg = AdaptationSpec(defaults=AdaptationPolicy(min_rate=0.3))
        policy = cfg.resolve_for("random_walk")
        assert policy.min_rate == pytest.approx(0.3)
        assert policy.max_rate == pytest.approx(_FALLBACK_MAX_RATE)

    def test_resolve_for_per_move_overrides_default(self):
        cfg = AdaptationSpec(
            defaults=AdaptationPolicy(min_rate=0.3, max_rate=0.7),
            per_move={"galilean": AdaptationPolicy(min_rate=0.5)},
        )
        policy = cfg.resolve_for("galilean")
        assert policy.min_rate == pytest.approx(0.5)
        assert policy.max_rate == pytest.approx(0.7)

    def test_resolve_for_per_move_none_falls_through_to_defaults(self):
        cfg = AdaptationSpec(
            defaults=AdaptationPolicy(adjust_factor=2.0),
            per_move={"random_walk": AdaptationPolicy(min_rate=0.4)},
        )
        policy = cfg.resolve_for("random_walk")
        assert policy.min_rate == pytest.approx(0.4)
        assert policy.adjust_factor == pytest.approx(2.0)

    def test_resolve_for_keyed_by_move_name_not_type(self):
        cfg = AdaptationSpec(
            per_move={
                "rw_slow": AdaptationPolicy(min_rate=0.1),
                "rw_fast": AdaptationPolicy(min_rate=0.6),
            },
        )
        slow = cfg.resolve_for("rw_slow")
        fast = cfg.resolve_for("rw_fast")
        assert slow.min_rate == pytest.approx(0.1)
        assert fast.min_rate == pytest.approx(0.6)

    def test_resolved_config_has_adaptation_policies_in_move_order(self):
        d = _minimal_dict()
        d["moves"] = [
            {"type": "random_walk", "name": "rw_a"},
            {"type": "galilean", "name": "gal_b"},
        ]
        d["adaptation"] = {
            "defaults": {"min_rate": 0.3},
            "per_move": {"gal_b": {"min_rate": 0.45}},
        }
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        assert len(resolved.adaptation_policies) == 2
        assert resolved.adaptation_policies[0].min_rate == pytest.approx(0.3)
        assert resolved.adaptation_policies[1].min_rate == pytest.approx(0.45)

    def test_adaptation_overlay_fixture_yaml(self):
        fixture = _DATA / "adaptation_overlay.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        resolved = resolve(root)
        assert len(resolved.adaptation_policies) == 2
        rw_policy = resolved.adaptation_policies[0]
        gal_policy = resolved.adaptation_policies[1]
        assert rw_policy.min_rate == pytest.approx(0.3)
        assert gal_policy.min_rate == pytest.approx(0.4)
        assert gal_policy.step_size_max == pytest.approx(5.0)
        assert gal_policy.max_rate == pytest.approx(0.7)

    def test_resolved_policies_use_fallback_when_no_adaptation_set(self):
        root = RootSpec.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert len(resolved.adaptation_policies) == 1
        assert resolved.adaptation_policies[0].min_rate == pytest.approx(
            _FALLBACK_MIN_RATE
        )


class TestEnsembleResolver:
    """Check the ensemble params that the NVT and NPT specs produce,
    including the GPa to eV/Å³ conversion.
    """

    def test_nvt_to_ensemble_params_empty(self):
        spec = NVTEnsembleSpec()
        params = spec.to_ensemble_params()
        assert params == {}

    def test_npt_scalar_to_ensemble_params(self):
        spec = NPTEnsembleSpec(pressure=0.05)
        params = spec.to_ensemble_params(cohort_index=0)
        assert "pressure" in params
        assert params["pressure"] == pytest.approx(0.05)

    def test_npt_list_to_ensemble_params_by_index(self):
        pressures = [0.01, 0.02, 0.03]
        spec = NPTEnsembleSpec(pressure=pressures)
        for i, p in enumerate(pressures):
            params = spec.to_ensemble_params(cohort_index=i)
            assert params["pressure"] == pytest.approx(p)

    def test_npt_gpa_converts_to_eva3(self):
        spec = NPTEnsembleSpec(pressure=1.0, pressure_units="gpa")
        params = spec.to_ensemble_params(cohort_index=0)
        assert params["pressure"] == pytest.approx(_GPA_TO_EVA3, rel=1e-5)

    def test_npt_gpa_list_conversion(self):
        spec = NPTEnsembleSpec(pressure=[1.0, 2.0], pressure_units="gpa")
        assert spec.to_ensemble_params(cohort_index=0)[
            "pressure"
        ] == pytest.approx(_GPA_TO_EVA3)
        assert spec.to_ensemble_params(cohort_index=1)[
            "pressure"
        ] == pytest.approx(2.0 * _GPA_TO_EVA3)


class TestMultiReplicaResolver:
    """Check the batcher and per-replica params resolve() picks for the
    scalar and list-valued NPT fixtures.
    """

    def test_npt_sweep_fixture_yaml_validates(self):
        fixture = _DATA / "npt_sweep.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        assert isinstance(root.ensemble, NPTEnsembleSpec)
        resolved = resolve(root)
        assert isinstance(resolved.batcher, PmapVmapRuns)
        n_total = resolved.batcher.n_gpu * resolved.batcher.n_per_gpu
        assert n_total == 2

    def test_npt_scalar_fixture_yaml_single_run(self):
        fixture = _DATA / "npt_scalar.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved.batcher, SingleRun)
        assert len(resolved.ensemble_params_per_run) == 1
        assert resolved.ensemble_params_per_run[0][
            "pressure"
        ] == pytest.approx(0.01)


class TestShardedSingleRun:
    """Check that run.shard_n_gpu selects the ShardedSingleRun batcher and
    the configs it is rejected with.
    """

    def test_resolver_routes_shard_to_sharded_batcher(self, tmp_path):
        root = _sharded_root(tmp_path, shard_n_gpu=2)
        resolved = resolve(root)
        assert isinstance(resolved.batcher, ShardedSingleRun)
        assert resolved.batcher.n_gpu == 2

    def test_resolver_default_keeps_single_run(self, tmp_path):
        root = _sharded_root(tmp_path, shard_n_gpu=1)
        resolved = resolve(root)
        assert isinstance(resolved.batcher, SingleRun)

    def test_resolver_rejects_shard_with_pressure_list(self, tmp_path):
        root = _sharded_root(tmp_path, shard_n_gpu=2, pressures=[1.0, 2.0])
        with pytest.raises(ValueError, match="shard_n_gpu"):
            resolve(root)

    def test_resolver_rejects_indivisible_n_live(self, tmp_path):
        root = _sharded_root(tmp_path, shard_n_gpu=3, n_live=8)
        with pytest.raises(ValueError, match="divisible"):
            resolve(root)


def test_cell_spec_is_passed_through():
    d = _minimal_dict()
    d["cell"] = {"max_volume_per_atom": 9999.0}
    resolved = resolve(RootSpec.model_validate(d))
    assert resolved.cell.max_volume_per_atom == pytest.approx(9999.0)


class TestExtendedOutputResolve:
    """Check that snapshot_clean and wrap_atoms reach
    ResolvedConfig.output.
    """

    def test_snapshot_clean_is_wired_through(self):
        d = _minimal_dict()
        d["output"]["snapshot_clean"] = False
        resolved = resolve(RootSpec.model_validate(d))
        assert resolved.output.snapshot_clean is False

    def test_snapshot_clean_defaults_true(self):
        d = _minimal_dict()
        resolved = resolve(RootSpec.model_validate(d))
        assert resolved.output.snapshot_clean is True

    def test_wrap_atoms_is_wired(self):
        d = _minimal_dict()
        d["output"]["wrap_atoms"] = False
        resolved = resolve(RootSpec.model_validate(d))
        assert resolved.output.wrap_atoms is False


class TestFullConfigResolver:
    """Check that the full and minimal YAML fixtures resolve."""

    def test_full_config_resolves(self):
        with open(_FULL_CONFIG_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert resolved.init.initial_positions.shape == (20, 2, 3)
        assert resolved.init.initial_types.shape == (20, 2)

    def test_minimal_yaml_resolves(self):
        with open(_MINIMAL_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert resolved.init.initial_types.shape == (20, 1)


class TestResolveInit:
    """Check the shapes, seeding and missing-file errors of _resolve_init."""

    def test_start_species_positions_shape(self):
        cfg = InitSpec(start_species="1 3")
        result = _resolve_init(cfg, n_live=10, seed=0)
        assert result.initial_positions.shape == (10, 3, 3)

    def test_start_species_types_shape_and_dtype(self):
        cfg = InitSpec(start_species="14 2, 8 1")
        result = _resolve_init(cfg, n_live=5, seed=1)
        # initial_types carries a leading n_live axis (per-walker), matching
        # positions / cells; fixed-composition runs repeat the same row.
        assert result.initial_types.shape == (5, 3)
        assert result.initial_types.dtype == jnp.int32

    def test_start_species_n_atoms_matches_counts(self):
        cfg = InitSpec(start_species="6 3, 1 2")
        result = _resolve_init(cfg, n_live=4, seed=0)
        assert result.initial_positions.shape[1] == 5

    def test_start_config_file_nonexistent_raises(self, tmp_path):
        cfg = InitSpec(start_config_file=tmp_path / "does_not_exist.xyz")
        with pytest.raises(FileNotFoundError):
            _resolve_init(cfg, n_live=4, seed=0)

    def test_start_walker_set_nonexistent_raises_file_not_found(self):
        cfg = InitSpec(
            start_walker_set=Path("/tmp/does_not_exist_walker.extxyz")
        )
        with pytest.raises(FileNotFoundError):
            _resolve_init(cfg, n_live=4, seed=0)

    def test_restart_file_nonexistent_raises_file_not_found(self):
        cfg = InitSpec(restart_file=Path("/tmp/does_not_exist_checkpoint.h5"))
        with pytest.raises(FileNotFoundError):
            _resolve_init(cfg, n_live=4, seed=0)

    def test_positions_are_finite(self):
        cfg = InitSpec(start_species="1 3")
        result = _resolve_init(cfg, n_live=8, seed=42)
        assert jnp.all(jnp.isfinite(result.initial_positions))

    def test_deterministic_with_same_seed(self):
        cfg = InitSpec(start_species="1 3")
        r1 = _resolve_init(cfg, n_live=4, seed=99)
        r2 = _resolve_init(cfg, n_live=4, seed=99)
        assert jnp.allclose(r1.initial_positions, r2.initial_positions)

    def test_different_seeds_differ(self):
        cfg = InitSpec(start_species="1 3")
        r1 = _resolve_init(cfg, n_live=4, seed=0)
        r2 = _resolve_init(cfg, n_live=4, seed=1)
        assert not jnp.allclose(r1.initial_positions, r2.initial_positions)


class TestResolveInitModes:
    """Check the cells, positions and energies built from start_species
    (mode A) and start_config_file (mode B).
    """

    def test_start_species_cells_shape(self):
        root = RootSpec.model_validate(_species_dict(n_atoms=2, n_live=6))
        resolved = resolve(root)
        assert resolved.init.initial_cells is not None
        assert resolved.init.initial_cells.shape == (6, 3, 3)

    def test_start_species_positions_shape(self):
        root = RootSpec.model_validate(_species_dict(n_atoms=2, n_live=6))
        resolved = resolve(root)
        assert resolved.init.initial_positions is not None
        assert resolved.init.initial_positions.shape == (6, 2, 3)

    def test_start_species_initial_energies_populated(self):
        root = RootSpec.model_validate(_species_dict(n_atoms=2, n_live=4))
        resolved = resolve(root)
        assert resolved.init.initial_energies is not None
        assert resolved.init.initial_energies.shape == (4,)
        assert jnp.all(jnp.isfinite(resolved.init.initial_energies))
        # The harmonic backend has no max_neighbors_for, so no counts.
        assert resolved.init.initial_max_neighbor_counts is None

    def test_npt_initial_energies_include_pv_term(self):
        """NPT initial energies are the NVT ones plus P * V per walker.

        Grid placement does not call the backend, so both runs start from
        the same positions and cells.
        """
        pressure = 0.1
        d = _species_dict(n_atoms=2, n_live=4)
        nvt = resolve(RootSpec.model_validate(d))
        d["ensemble"] = {"type": "npt", "pressure": pressure}
        npt = resolve(RootSpec.model_validate(d))

        np.testing.assert_allclose(
            npt.init.initial_positions, nvt.init.initial_positions
        )
        volumes = np.asarray(get_volume(npt.init.initial_cells))
        np.testing.assert_allclose(
            np.asarray(npt.init.initial_energies - nvt.init.initial_energies),
            pressure * volumes,
            rtol=1e-5,
        )

    def test_random_initialise_cell_true_produces_diverse_cells(self):
        d = _species_dict(n_atoms=2, n_live=8)
        d["init"]["random_initialise_cell"] = True
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        cells = resolved.init.initial_cells
        assert cells is not None
        assert cells.shape == (8, 3, 3)
        diffs = jnp.abs(cells[1:] - cells[:-1])
        assert jnp.any(diffs > 1e-5)

    def test_grid_mode_pairwise_distances(self):
        root = RootSpec.model_validate(
            _species_dict(n_atoms=2, n_live=4, mode="grid")
        )
        resolved = resolve(root)
        positions = resolved.init.initial_positions
        grid_dist = 1.5
        for wi in range(4):
            p = positions[wi]
            d = float(jnp.linalg.norm(p[0] - p[1]))
            assert d >= grid_dist - 1e-4

    def test_uniform_mode_energies_below_ceiling(self):
        n_atoms = 2
        ceiling_per_atom = 1e6
        d = _species_dict(n_atoms=n_atoms, n_live=4, mode="uniform")
        d["init"]["start_energy_ceiling_per_atom"] = ceiling_per_atom
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        energies = resolved.init.initial_energies
        assert energies is not None
        ceiling_total = ceiling_per_atom * n_atoms
        assert jnp.all(energies <= ceiling_total + 1e-3)

    def test_random_initialise_pos_false_identical_positions(self, caplog):
        d = _species_dict(n_atoms=2, n_live=4)
        d["init"]["random_initialise_pos"] = False
        root = RootSpec.model_validate(d)
        with caplog.at_level(logging.WARNING):
            resolved = resolve(root)
        assert any(
            "correlation" in rec.message.lower() for rec in caplog.records
        )
        positions = np.asarray(resolved.init.initial_positions)
        for wi in range(1, 4):
            np.testing.assert_allclose(positions[wi], positions[0])

    def test_start_species_symbol_map_populated(self):
        cfg = InitSpec(start_species="14 8")
        result = _resolve_init(cfg, n_live=4, seed=0)
        assert result.symbol_map == {0: "Si"}

    def _make_founder(self, tmp_path, symbols=("Si", "Si"), cell_size=6.0):
        """Write a minimal single-frame extxyz and return its path."""
        pos = np.zeros((len(symbols), 3), dtype=np.float32)
        for i in range(len(symbols)):
            pos[i, 0] = i * (cell_size / (len(symbols) + 1))
        cell = np.eye(3) * cell_size
        atoms = ase.Atoms(
            symbols=list(symbols), positions=pos, cell=cell, pbc=True
        )
        p = tmp_path / "founder.extxyz"
        ase.io.write(str(p), atoms)
        return p

    def test_mode_b_random_pos_true_positions_shape(self, tmp_path):
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=6.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=5, seed=0, cell_cfg=cell_cfg)
        assert result.initial_positions.shape == (5, 2, 3)

    def test_mode_b_random_pos_false_identical_positions(
        self, tmp_path, caplog
    ):
        p = self._make_founder(tmp_path, symbols=["Si", "O"], cell_size=6.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(cfg, n_live=4, seed=0, cell_cfg=cell_cfg)
        for wi in range(1, 4):
            np.testing.assert_allclose(
                np.array(result.initial_positions[wi]),
                np.array(result.initial_positions[0]),
            )
        assert any(
            "correlated" in r.message.lower()
            or "identical" in r.message.lower()
            for r in caplog.records
        )

    def test_mode_b_random_pos_false_warning_mentions_burn_in(
        self, tmp_path, caplog
    ):
        p = self._make_founder(tmp_path, symbols=["Si"], cell_size=5.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(cfg, n_live=3, seed=0, cell_cfg=cell_cfg)
        assert any("burn-in" in r.message for r in caplog.records)

    def test_mode_b_random_cell_true_cells_diverge(self, tmp_path):
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=8.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=True,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=4, seed=7, cell_cfg=cell_cfg)
        cells = np.array(result.initial_cells)
        all_same = all(np.allclose(cells[0], cells[wi]) for wi in range(1, 4))
        assert not all_same

    def test_mode_b_random_cell_false_cells_identical(self, tmp_path):
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=6.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=4, seed=0, cell_cfg=cell_cfg)
        cells = np.array(result.initial_cells)
        for wi in range(1, 4):
            np.testing.assert_allclose(cells[0], cells[wi])

    def test_mode_b_symbol_map_from_file(self, tmp_path):
        p = self._make_founder(tmp_path, symbols=["Si", "O"], cell_size=6.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=3, seed=0, cell_cfg=cell_cfg)
        assert result.symbol_map == {0: "Si", 1: "O"}

    def test_mode_b_positions_are_finite(self, tmp_path):
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=6.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=4, seed=0, cell_cfg=cell_cfg)
        assert jnp.all(jnp.isfinite(result.initial_positions))

    def test_mode_b_energies_computed_with_backend(self, tmp_path):
        p = self._make_founder(tmp_path, symbols=["Si"], cell_size=5.0)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellSpec(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        backend = create_harmonic()
        result = _resolve_init(
            cfg, n_live=3, seed=0, energy_backend=backend, cell_cfg=cell_cfg
        )
        # _resolve_init returns positions, types and cells only; the
        # energies are computed by the caller.
        energies, _ = _finalise_initial_energies_and_counts(
            backend,
            result.initial_positions,
            result.initial_types,
            result.initial_cells,
        )
        assert energies is not None
        assert energies.shape == (3,)
        assert jnp.all(jnp.isfinite(energies))


def _write_si_o_frames(path: Path, n_frames: int) -> None:
    """Write ``n_frames`` copies of a Si2O cell, Si listed first."""
    atoms = ase.Atoms(
        "Si2O",
        positions=[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 3.0, 1.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    ase.io.write(str(path), [atoms.copy() for _ in range(n_frames)])


class TestWalkerSetInit:
    """Check _resolve_init with start_walker_set (mode C).

    The walkers come from the file unchanged: random_initialise_pos and
    random_initialise_cell only log a warning, and a walker cell outside
    the CellSpec bounds raises.
    """

    def test_extxyz_shapes(self, tmp_path):
        p = tmp_path / "walkers.extxyz"
        _write_si_o_frames(p, 4)
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0, energy_backend=create_harmonic()
        )
        assert result.initial_positions.shape == (4, 3, 3)
        assert result.initial_types.shape == (4, 3)
        assert result.initial_cells.shape == (4, 3, 3)

    def test_extxyz_symbol_map_without_z_table(self, tmp_path):
        p = tmp_path / "walkers.extxyz"
        _write_si_o_frames(p, 3)
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg, n_live=3, seed=0, energy_backend=create_harmonic()
        )
        assert result.symbol_map == {0: "Si", 1: "O"}

    def test_hdf5_positions_shape(self, tmp_path):
        rng = np.random.default_rng(7)
        p = tmp_path / "walkers.h5"
        with h5py.File(p, "w") as f:
            f["positions"] = rng.uniform(0.5, 5.5, (5, 2, 3)).astype(
                np.float32
            )
            f["types"] = np.zeros((5, 2), dtype=np.int32)
            f["cells"] = np.stack([np.eye(3) * 6.0] * 5).astype(np.float32)
            f.attrs["symbol_map"] = json.dumps({"0": "Si"})
        cfg = InitSpec(start_walker_set=p)
        result = _resolve_init(
            cfg,
            n_live=5,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_PERMISSIVE_CELL,
        )
        assert result.initial_positions.shape == (5, 2, 3)

    def test_random_initialise_pos_true_keeps_file_positions(
        self, tmp_path, caplog
    ):
        p = tmp_path / "walkers.extxyz"
        _write_si_o_frames(p, 3)
        cfg = InitSpec(start_walker_set=p, random_initialise_pos=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(
                cfg, n_live=3, seed=0, energy_backend=create_harmonic()
            )
        assert any(
            "ignoring random_initialise_pos/cell" in r.message
            for r in caplog.records
        )
        ws = load_walker_set(p, n_live_expected=3)
        np.testing.assert_allclose(
            np.array(result.initial_positions),
            np.array(ws.positions),
            atol=1e-5,
        )

    def test_random_initialise_cell_true_keeps_file_cells(
        self, tmp_path, caplog
    ):
        p = tmp_path / "walkers.extxyz"
        _write_si_o_frames(p, 3)
        cfg = InitSpec(start_walker_set=p, random_initialise_cell=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(
                cfg, n_live=3, seed=0, energy_backend=create_harmonic()
            )
        assert any(
            "ignoring random_initialise_pos/cell" in r.message
            for r in caplog.records
        )
        ws = load_walker_set(p, n_live_expected=3)
        np.testing.assert_allclose(
            np.array(result.initial_cells),
            np.array(ws.cells),
            atol=1e-5,
        )

    def test_cell_config_violation_raises(self, tmp_path):
        """A 216 Å³ cell with one atom exceeds max_volume_per_atom = 1."""
        atoms = ase.Atoms(
            ["Si"],
            positions=[[3.0, 3.0, 3.0]],
            cell=np.eye(3) * 6.0,
            pbc=True,
        )
        p = tmp_path / "toosmall.extxyz"
        ase.io.write(str(p), [atoms], format="extxyz")
        strict_cfg = CellSpec(
            max_volume_per_atom=1.0,
            min_volume_per_atom=0.0001,
            min_aspect_ratio=0.001,
        )
        cfg = InitSpec(start_walker_set=p)
        with pytest.raises(RuntimeError, match="invalid cell"):
            _resolve_init(cfg, n_live=1, seed=0, cell_cfg=strict_cfg)


def _write_restart_checkpoint(
    tmp_path: Path, n_walkers: int = 4, n_atoms: int = 1, n_dead: int = 5
) -> Path:
    """Write a single-run NS checkpoint with ``n_dead`` dead points."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2, 2, (n_walkers, n_atoms, 3)).astype(np.float32)
    state = {
        "positions": positions,
        "types": np.zeros((n_walkers, n_atoms), dtype=np.int32),
        "energies": rng.uniform(1, 10, n_walkers).astype(np.float32),
        "cells": np.stack([np.eye(3, dtype=np.float32) * 6.0] * n_walkers),
        "dead_energies": rng.uniform(10, 20, n_dead).astype(np.float32),
        "dead_positions": rng.uniform(-2, 2, (n_dead, n_atoms, 3)).astype(
            np.float32
        ),
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": -7.3,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_walkers,
        "rng_key": jax.random.key(1),
    }
    p = tmp_path / "ns.checkpoint.h5"
    save_checkpoint(p, state, symbol_map={0: "Si"})
    return p


def _write_batched_checkpoint(
    tmp_path: Path, n_dead: tuple[int, ...], n_walkers: int = 4
) -> tuple[Path, np.ndarray]:
    """Write a (n_total,)-batched checkpoint with one atom per walker.

    Returns the path and the saved positions, of shape
    ``(n_total, n_walkers, 1, 3)``.
    """
    n_total = len(n_dead)
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2, 2, (n_total, n_walkers, 1, 3)).astype(
        np.float32
    )
    energies = rng.uniform(0, 5, (n_total, n_walkers)).astype(np.float32)
    state = {
        "positions": positions,
        "types": np.zeros((n_total, n_walkers, 1), dtype=np.int32),
        "energies": energies,
        "cells": np.broadcast_to(
            np.eye(3, dtype=np.float32) * 5.0, (n_total, n_walkers, 3, 3)
        ).copy(),
        "dead_volumes": None,
        "live_volumes": None,
        "log_evidence": np.full(n_total, -10.0),
        "iteration": np.array(n_dead, dtype=np.int32),
        "emax": energies.max(axis=-1),
        "n_dead": np.array(n_dead, dtype=np.int32),
        "n_walkers": n_walkers,
        "rng_key": jax.random.split(jax.random.key(0), n_total),
    }
    p = tmp_path / "batched.checkpoint.h5"
    save_checkpoint(p, state, symbol_map={0: "Si"})
    return p, positions


def _restart_root(restart_file: Path, pressures: list[float]) -> RootSpec:
    return RootSpec.model_validate(
        {
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
            "ensemble": {"type": "npt", "pressure": pressures},
            "init": {"restart_file": str(restart_file)},
            "cell": {
                "max_volume_per_atom": 10000.0,
                "min_volume_per_atom": 0.01,
                "min_aspect_ratio": 0.001,
            },
        }
    )


class TestRestartInit:
    """Check init from restart_file (mode D): the walkers and restart
    state loaded for one or several replicas, and the errors when the
    checkpoint does not match the replica count.
    """

    @pytest.fixture(autouse=True)
    def _single_device(self, monkeypatch):
        monkeypatch.setattr(_r, "_local_device_count", lambda: 1)

    def test_single_run_loads_walkers_and_restart_state(self, tmp_path):
        p = _write_restart_checkpoint(tmp_path, n_dead=5)
        cfg = InitSpec(restart_file=p)
        result = _resolve_init(
            cfg,
            n_live=4,
            seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_PERMISSIVE_CELL,
        )
        assert result.initial_positions.shape == (4, 1, 3)
        assert isinstance(result.restart_state, RestartBundle)
        assert result.restart_state.n_dead == 5
        assert result.restart_state.iteration == 5
        assert result.symbol_map == {0: "Si"}

    def test_random_initialise_pos_true_warns(self, tmp_path, caplog):
        p = _write_restart_checkpoint(tmp_path)
        cfg = InitSpec(restart_file=p, random_initialise_pos=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(
                cfg,
                n_live=4,
                seed=0,
                energy_backend=create_harmonic(),
                cell_cfg=_PERMISSIVE_CELL,
            )
        assert any("verbatim" in r.message for r in caplog.records)

    def test_multi_replica_with_single_run_checkpoint_raises(self, tmp_path):
        p = _write_restart_checkpoint(tmp_path)
        root = _restart_root(p, [0.01, 0.02, 0.03])
        with pytest.raises(
            ValueError, match="scalar single-run checkpoint"
        ) as exc_info:
            resolve(root)
        assert "n_total=3" in str(exc_info.value)

    def test_multi_replica_loads_batched_checkpoint(self, tmp_path):
        p, positions = _write_batched_checkpoint(tmp_path, n_dead=(3, 7))
        resolved = resolve(_restart_root(p, [0.01, 0.02]))
        np.testing.assert_allclose(
            np.asarray(resolved.init.initial_positions), positions
        )
        bundles = resolved.init.restart_state
        assert len(bundles) == 1
        assert [b.n_dead for b in bundles[0]] == [3, 7]

    def test_multi_replica_count_mismatch_raises(self, tmp_path):
        p, _ = _write_batched_checkpoint(tmp_path, n_dead=(3, 7))
        root = _restart_root(p, [0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="checkpoint has n_total=2"):
            resolve(root)


class _FakeZTableBackend:
    """Stand-in for MACE: exposes ``atomic_numbers`` and a dummy __call__.

    Covers the resolver's backend-aware species mapping without needing
    mace-jax installed.
    """

    r_cutoff = 5.0

    def __init__(self, atomic_numbers: list[int]):
        self.atomic_numbers = list(atomic_numbers)

    def __call__(
        self, positions, species, cell, max_neighbors, ensemble_params=None
    ):
        return BackendResult(energy=jnp.float32(0.0))


class TestBackendAwareSpeciesMapping:
    """Check that start_species maps onto ``atomic_numbers`` when the
    backend has a z-table, and onto 0-based indices otherwise.

    Model-based backends (MACE, NeuralIL) expect species indices in the
    order of the z-table they were trained on.
    """

    def test_no_atomic_numbers_keeps_zero_based_mapping(self):
        cfg = InitSpec(
            start_species="8 3, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(
            cfg,
            n_live=2,
            seed=0,
            energy_backend=create_harmonic(),
        )
        # Sorted unique Z = [8, 22, 38] → 0-based indices [0, 1, 2].
        # types_list order is sorted-by-Z: O O O Ti Sr → idx [0,0,0,1,2].
        # initial_types is (n_live, n_atoms); every walker shares the row.
        assert result.initial_types.shape == (2, 5)
        assert list(map(int, result.initial_types[0])) == [0, 0, 0, 1, 2]
        assert bool((result.initial_types == result.initial_types[0]).all())
        assert result.symbol_map == {0: "O", 1: "Ti", 2: "Sr"}

    def test_backend_z_table_overrides_mapping(self):
        # Mimic mace_mp's 89-element z-table (Z=1..89).
        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        cfg = InitSpec(
            start_species="8 3, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(
            cfg,
            n_live=2,
            seed=0,
            energy_backend=backend,
        )
        # Z=8→idx 7 (O), Z=22→idx 21 (Ti), Z=38→idx 37 (Sr).
        assert result.initial_types.shape == (2, 5)
        assert list(map(int, result.initial_types[0])) == [7, 7, 7, 21, 37]
        assert bool((result.initial_types == result.initial_types[0]).all())
        assert result.symbol_map == {7: "O", 21: "Ti", 37: "Sr"}

    def test_missing_z_in_backend_table_raises(self):
        # Backend supports only a small subset (no Sr, Z=38).
        backend = _FakeZTableBackend(atomic_numbers=[1, 8, 22])
        cfg = InitSpec(
            start_species="8 1, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        with pytest.raises(ValueError, match="atomic numbers"):
            _resolve_init(cfg, n_live=2, seed=0, energy_backend=backend)

    def test_hamiltonian_passes_model_atomic_numbers_through(self):
        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        wrapped = EnsembleBackend(backend, pressure=0.1)
        assert wrapped.model.atomic_numbers == list(range(1, 90))

        cfg = InitSpec(
            start_species="8 3, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(
            cfg,
            n_live=2,
            seed=0,
            energy_backend=wrapped,
        )
        assert result.initial_types.shape == (2, 5)
        assert list(map(int, result.initial_types[0])) == [7, 7, 7, 21, 37]
        assert bool((result.initial_types == result.initial_types[0]).all())


class TestBackendAwareSpeciesMappingFromFile:
    """Check that file-based init modes map species onto the backend
    z-table too.

    The loaders number species 0, 1, ... in order of first appearance; with
    an 89-element z-table that would turn Si into index 0 (= H).
    """

    def test_config_file_uses_backend_z_table(self, tmp_path):
        p = tmp_path / "founder.extxyz"
        _write_si_o_frames(p, 1)
        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(cfg, n_live=2, seed=0, energy_backend=backend)
        # Si (Z=14) -> idx 13, O (Z=8) -> idx 7.
        assert list(map(int, result.initial_types[0])) == [13, 13, 7]
        assert result.symbol_map == {13: "Si", 7: "O"}

    def test_config_file_without_z_table_keeps_loader_indices(self, tmp_path):
        p = tmp_path / "founder.extxyz"
        _write_si_o_frames(p, 1)
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(
            cfg, n_live=2, seed=0, energy_backend=create_harmonic()
        )
        assert list(map(int, result.initial_types[0])) == [0, 0, 1]
        assert result.symbol_map == {0: "Si", 1: "O"}

    def test_config_file_missing_z_raises(self, tmp_path):
        p = tmp_path / "founder.extxyz"
        _write_si_o_frames(p, 1)
        backend = _FakeZTableBackend(atomic_numbers=[1, 8])
        cfg = InitSpec(
            start_config_file=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        with pytest.raises(ValueError, match=r"atomic numbers \[14\]"):
            _resolve_init(cfg, n_live=2, seed=0, energy_backend=backend)

    def test_walker_set_extxyz_uses_backend_z_table(self, tmp_path):
        p = tmp_path / "walkers.extxyz"
        _write_si_o_frames(p, 3)
        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        cfg = InitSpec(
            start_walker_set=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=backend)
        assert result.initial_types.shape == (3, 3)
        assert bool((result.initial_types == jnp.array([13, 13, 7])).all())
        assert result.symbol_map == {13: "Si", 7: "O"}

    def test_walker_set_hdf5_uses_symbol_map(self, tmp_path):
        p = tmp_path / "walkers.h5"
        with h5py.File(p, "w") as f:
            f["positions"] = np.zeros((2, 3, 3), dtype=np.float32)
            f["types"] = np.array([[0, 0, 1]] * 2, dtype=np.int32)
            f["cells"] = np.broadcast_to(
                np.eye(3, dtype=np.float32) * 6.0, (2, 3, 3)
            )
            f.attrs["symbol_map"] = json.dumps({"0": "Si", "1": "O"})
        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        cfg = InitSpec(
            start_walker_set=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(cfg, n_live=2, seed=0, energy_backend=backend)
        assert bool((result.initial_types == jnp.array([13, 13, 7])).all())
        assert result.symbol_map == {13: "Si", 7: "O"}

    def test_non_element_labels_raise(self, tmp_path):
        p = tmp_path / "walkers.h5"
        with h5py.File(p, "w") as f:
            f["positions"] = np.zeros((2, 1, 3), dtype=np.float32)
            f["types"] = np.zeros((2, 1), dtype=np.int32)
            f["cells"] = np.broadcast_to(
                np.eye(3, dtype=np.float32) * 6.0, (2, 3, 3)
            )
            f.attrs["symbol_map"] = json.dumps({"0": "A"})
        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        cfg = InitSpec(start_walker_set=p)
        with pytest.raises(ValueError, match="not element symbols"):
            _resolve_init(cfg, n_live=2, seed=0, energy_backend=backend)


class _BucketRecordingBackend:
    """Stub backend with ``max_neighbors_for`` that records the
    ``max_neighbors`` bucket each energy call is compiled at.
    """

    r_cutoff = 5.0

    def __init__(self, per_walker_count: int):
        self._per_walker_count = int(per_walker_count)
        self.calls: list[int] = []

    def max_neighbors_for(self, positions, cell):
        return jnp.int32(self._per_walker_count)

    def __call__(
        self, positions, species, cell, max_neighbors, ensemble_params=None
    ):
        self.calls.append(int(max_neighbors))
        return BackendResult(
            energy=jnp.float32(0.0),
            max_neighbor_count=jnp.int32(int(max_neighbors)),
            overflow=jnp.bool_(False),
        )


class TestFinaliseInitialBucketChoice:
    """Check the neighbor bucket _finalise_initial_energies_and_counts
    compiles the energy at.

    With ``ladder`` it uses ``_choose_starting_bucket(counts, ladder,
    offset)``, the same choice cli/run.py makes; without it, it falls back
    to ``int(max(counts))``.
    """

    def _positions_types_cells(self, n_walkers: int = 3, n_atoms: int = 2):
        positions = jnp.zeros((n_walkers, n_atoms, 3), dtype=jnp.float32)
        types = jnp.zeros((n_walkers, n_atoms), dtype=jnp.int32)
        cells = jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float32) * 10.0, (n_walkers, 3, 3)
        )
        return positions, types, cells

    def test_ladder_and_offset_used_when_provided(self):
        per_walker = 27
        backend = _BucketRecordingBackend(per_walker_count=per_walker)
        positions, types, cells = self._positions_types_cells()

        ladder = (32, 64, 128)
        offset = 4
        energies, counts = _finalise_initial_energies_and_counts(
            backend,
            positions,
            types,
            cells,
            batcher=SingleRun(),
            ladder=ladder,
            offset=offset,
        )

        assert counts is not None
        expected_bucket = int(
            _choose_starting_bucket(
                counts,
                ladder,
                offset,
            )
        )
        # 27 + 4 = 31 → smallest ladder entry ≥ 31 is 32.
        assert expected_bucket == 32
        # The energy call was compiled at the ladder bucket, not at
        # int(max(counts)) = 27.
        assert backend.calls
        assert all(c == expected_bucket for c in backend.calls)

    def test_no_ladder_falls_back_to_max_counts(self):
        per_walker = 27
        backend = _BucketRecordingBackend(per_walker_count=per_walker)
        positions, types, cells = self._positions_types_cells()

        _finalise_initial_energies_and_counts(
            backend,
            positions,
            types,
            cells,
            batcher=SingleRun(),
        )

        assert backend.calls
        assert all(c == per_walker for c in backend.calls)

    def test_resolve_uses_backend_ladder(self, monkeypatch):
        """resolve() stores the counts and compiles at the ladder bucket.

        27 neighbors + offset 5 = 32; the first entry of the default
        ladder [30, 35, 40, 45, 50] that holds it is 35.
        """
        backend = _BucketRecordingBackend(per_walker_count=27)
        monkeypatch.setattr(
            HarmonicBackendSpec, "build_backend", lambda self: backend
        )
        resolved = resolve(
            RootSpec.model_validate(_species_dict(n_atoms=2, n_live=4))
        )
        assert (
            list(map(int, resolved.init.initial_max_neighbor_counts))
            == [27] * 4
        )
        assert backend.calls == [35]
