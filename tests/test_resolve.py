"""Resolver-layer tests extracted from test_schema.py.

Covers: resolve(), _resolve_init(), to_descriptor(), to_move_config(),
to_backend_config(), build_backend(), to_criterion(), to_ensemble_params(),
adapt overlay logic, cohort expansion, full-config resolver, and
TestInitConfigResolver (Part A and Part B, Mode B resolver tests).

All classes that were renamed to avoid collisions are documented below:
- TestInitConfigResolver (first occurrence, line 1705 of original test_schema.py)
  -> kept as TestInitConfigResolverPartA (resolver unit tests for Part A init)
- TestInitConfigResolver (second occurrence, line 2080 of original test_schema.py)
  -> TestInitConfigResolverPartB, minus the two E2E tests:
       test_start_species_e2e_run_ns   -> kept in test_init_positions.py
       test_mode_b_end_to_end_jit      -> kept in test_init_structure.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jaxrens.cli.schema import RootConfig
from jaxrens.cli.schema.backend import (
    DoubleWellBackendSpec,
    GaussianMixtureBackendSpec,
    HarmonicBackendSpec,
    LJBackendSpec,
    MACEBackendSpec,
    NeuralILBackendSpec,
)
from jaxrens.cli.schema.moves import (
    RandomWalkMoveSpec,
    GalileanMoveSpec,
    GmcMoveSpec,
    HMCMoveSpec,
    SingleAtomMoveSpec,
    VolumeMoveSpec,
    AlchemicalMorphMoveSpec,
    AlchemicalShiftMoveSpec,
)
from jaxrens.cli.resolve import resolve, expand_cohort, ResolvedConfig
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig

_DATA = Path(__file__).parent / "data" / "cli"
_MINIMAL_YAML = _DATA / "minimal.yaml"
_LJ_BACKEND_YAML = _DATA / "lj_backend.yaml"
_FULL_CONFIG_YAML = _DATA / "full_config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _species_dict(n_atoms: int = 4, n_live: int = 8, mode: str = "grid") -> dict:
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


# ---------------------------------------------------------------------------
# 7. resolve() produces correct library dataclasses
# ---------------------------------------------------------------------------

class TestResolve:
    def test_resolve_types(self):
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert isinstance(resolved.ns, NSConfig)
        assert isinstance(resolved.moves, tuple)
        assert all(isinstance(m, MoveConfig) for m in resolved.moves)
        assert isinstance(resolved.backend, BackendConfig)
        assert isinstance(resolved.output, OutputConfig)

    def test_resolve_values_match_hand_built(self):
        d = _minimal_dict()
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        expected_ns = NSConfig(
            n_live=20,
            max_iterations=50,
            n_mcmc_steps=5,
            seed=0,
            convergence_threshold=0.1,
            n_cull=1,
            pressure=None,
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
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert len(resolved.moves) == 2
        assert resolved.moves[0].move_type == "random_walk"
        assert resolved.moves[1].move_type == "galilean"


# ---------------------------------------------------------------------------
# 12. to_descriptor() pins kernel kwargs mapping (parametrized)
# ---------------------------------------------------------------------------

class TestToDescriptor:
    """to_descriptor() must produce the same kernel kwargs that
    _build_kernel_kwargs used to produce in cli/run.py."""

    def test_random_walk_descriptor(self):
        spec = RandomWalkMoveSpec(step_size=0.2, weight=2.0)
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {}
        assert desc.step_size == pytest.approx(0.2)
        assert desc.weight == pytest.approx(2.0)
        assert desc.extra_state_fields == {}

    def test_galilean_descriptor_n_reflect(self):
        spec = GalileanMoveSpec(n_reflect=7, step_size=0.05)
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {"n_reflect": 7}
        assert "direction" in desc.extra_state_fields

    def test_gmc_descriptor_n_reflect(self):
        spec = GmcMoveSpec(n_reflect=3)
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {"n_reflect": 3}
        assert "direction" in desc.extra_state_fields

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
        from jaxrens.cli.schema.cell import CellConfig
        cell_cfg = CellConfig(max_volume_per_atom=50.0, min_volume_per_atom=1.0,
                              min_aspect_ratio=0.5, flat_V_prior=False)
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

    def test_alchemical_shift_descriptor(self):
        spec = AlchemicalShiftMoveSpec()
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs == {}

    def test_name_defaults_to_type(self):
        spec = RandomWalkMoveSpec()
        desc = spec.to_descriptor()
        assert desc.name == "random_walk"

    def test_name_override(self):
        spec = GalileanMoveSpec(name="gal_heavy", n_reflect=5)
        desc = spec.to_descriptor()
        assert desc.name == "gal_heavy"


# ---------------------------------------------------------------------------
# 13. to_move_config() round-trips through MoveConfig
# ---------------------------------------------------------------------------

class TestToMoveConfig:
    def test_random_walk_move_config(self):
        spec = RandomWalkMoveSpec(step_size=0.3, weight=2.0, adaptation_warmup=50)
        mc = spec.to_move_config()
        assert isinstance(mc, MoveConfig)
        assert mc.move_type == "random_walk"
        assert mc.step_size == pytest.approx(0.3)
        assert mc.weight == pytest.approx(2.0)
        assert mc.adaptation_warmup == 50

    def test_galilean_n_steps_maps_to_n_reflect(self):
        spec = GalileanMoveSpec(n_reflect=12)
        mc = spec.to_move_config()
        assert mc.n_steps == 12

    def test_hmc_n_steps_maps_to_n_leapfrog(self):
        spec = HMCMoveSpec(n_leapfrog=20)
        mc = spec.to_move_config()
        assert mc.n_steps == 20


# ---------------------------------------------------------------------------
# 14. ResolvedConfig.move_descriptors present and typed
# ---------------------------------------------------------------------------

class TestResolvedDescriptors:
    def test_resolve_produces_move_descriptors(self):
        from jaxrens.sampling.move_kernel import MoveKernel
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert hasattr(resolved, "move_descriptors")
        assert isinstance(resolved.move_descriptors, tuple)
        assert all(isinstance(d, MoveKernel) for d in resolved.move_descriptors)
        assert len(resolved.move_descriptors) == len(resolved.moves)

    def test_resolve_descriptor_matches_move(self):
        d = _minimal_dict()
        d["moves"] = [{"type": "hmc", "n_leapfrog": 6, "step_size": 0.02}]
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        desc = resolved.move_descriptors[0]
        assert desc.kernel_kwargs["n_leapfrog"] == 6
        assert desc.step_size == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 18. to_backend_config() produces correct BackendConfig
# ---------------------------------------------------------------------------

class TestToBackendConfig:
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


# ---------------------------------------------------------------------------
# 19. build_backend() for lightweight backends
# ---------------------------------------------------------------------------

class TestBuildBackend:
    def test_harmonic_build_backend(self):
        from jaxrens.backends.toy import HarmonicBackend
        spec = HarmonicBackendSpec(k=2.0)
        backend = spec.build_backend()
        assert isinstance(backend, HarmonicBackend)
        assert backend.k == pytest.approx(2.0)

    def test_double_well_build_backend(self):
        from jaxrens.backends.toy import DoubleWellBackend
        spec = DoubleWellBackendSpec(a=0.5, b=2.0)
        backend = spec.build_backend()
        assert isinstance(backend, DoubleWellBackend)
        assert backend.a == pytest.approx(0.5)
        assert backend.b == pytest.approx(2.0)

    def test_gaussian_mixture_build_backend(self):
        from jaxrens.backends.toy import GaussianMixtureBackend
        spec = GaussianMixtureBackendSpec(sigma=0.3)
        backend = spec.build_backend()
        assert isinstance(backend, GaussianMixtureBackend)
        assert backend.sigma == pytest.approx(0.3)

    def test_lj_build_backend(self):
        from jaxrens.backends.lj import LJBackend
        spec = LJBackendSpec(epsilon=0.5, sigma=1.2, cutoff=4.0)
        backend = spec.build_backend()
        assert isinstance(backend, LJBackend)
        assert backend.epsilon == pytest.approx(0.5)
        assert backend.sigma == pytest.approx(1.2)
        assert backend.cutoff == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 21. resolve() energy_backend field
# ---------------------------------------------------------------------------

class TestResolveEnergyBackend:
    def test_resolve_has_base_backend(self):
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert hasattr(resolved, "base_backend")
        assert resolved.base_backend is not None

    def test_resolve_base_backend_is_harmonic(self):
        from jaxrens.backends.toy import HarmonicBackend
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert isinstance(resolved.base_backend, HarmonicBackend)

    def test_resolve_base_backend_is_callable(self):
        import jax.numpy as jnp
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        positions = jnp.zeros((1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energy, _, _ = resolved.base_backend(positions, types, cell, 0)
        assert jnp.isfinite(energy)

    def test_resolve_base_backend_lj(self):
        from jaxrens.backends.lj import LJBackend
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved.base_backend, LJBackend)


# ---------------------------------------------------------------------------
# to_criterion tests (split from TestTerminationDiscriminatedUnion)
# ---------------------------------------------------------------------------

class TestToCriterion:
    def test_to_criterion_iteration(self):
        from jaxrens.cli.schema.termination import IterationTerminationSpec
        from jaxrens.sampling.termination import IterationTermination
        spec = IterationTerminationSpec(max_iterations=42)
        crit = spec.to_criterion()
        assert isinstance(crit, IterationTermination)
        assert crit.max_iterations == 42

    def test_to_criterion_prior_mass(self):
        from jaxrens.cli.schema.termination import PriorMassTerminationSpec
        from jaxrens.sampling.termination import PriorMassTermination
        spec = PriorMassTerminationSpec(threshold=0.05)
        crit = spec.to_criterion(n_live=30, n_cull=1)
        assert isinstance(crit, PriorMassTermination)
        assert crit.n_live == 30
        assert crit.threshold == pytest.approx(0.05)

    def test_to_criterion_prior_mass_requires_n_live(self):
        from jaxrens.cli.schema.termination import PriorMassTerminationSpec
        spec = PriorMassTerminationSpec(threshold=0.05)
        with pytest.raises(ValueError, match="requires n_live"):
            spec.to_criterion()

    def test_to_criterion_temperature(self):
        from jaxrens.cli.schema.termination import TemperatureTerminationSpec
        from jaxrens.sampling.termination import TempTermination
        spec = TemperatureTerminationSpec(target_temp=500.0, threshold=8.0)
        crit = spec.to_criterion(n_live=20, n_cull=2)
        assert isinstance(crit, TempTermination)
        assert crit.target_temp == pytest.approx(500.0)
        assert crit.n_cull == 2
        assert crit.threshold == pytest.approx(8.0)

    def test_to_criterion_energy(self):
        from jaxrens.cli.schema.termination import EnergyTerminationSpec
        from jaxrens.sampling.termination import EnergyTermination
        spec = EnergyTerminationSpec(min_energy=-3.5)
        crit = spec.to_criterion()
        assert isinstance(crit, EnergyTermination)
        assert crit.min_energy == pytest.approx(-3.5)

    def test_termination_none_resolves_to_legacy_defaults(self):
        root = RootConfig.model_validate(_minimal_dict())
        assert root.termination is None
        resolved = resolve(root)
        from jaxrens.sampling.termination import IterationTermination, PriorMassTermination
        assert len(resolved.termination) == 2
        types_found = {type(c) for c in resolved.termination}
        assert IterationTermination in types_found
        assert PriorMassTermination in types_found

    def test_termination_none_prior_mass_uses_convergence_threshold(self):
        d = _minimal_dict()
        d["run"]["convergence_threshold"] = 0.05
        d["run"]["n_live"] = 15
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        from jaxrens.sampling.termination import PriorMassTermination
        pm = next(c for c in resolved.termination if isinstance(c, PriorMassTermination))
        assert pm.threshold == pytest.approx(0.05)
        assert pm.n_live == 15

    def test_two_criteria_resolve_to_two_element_tuple(self):
        d = _minimal_dict()
        # Drop run.max_iterations so the resolver doesn't auto-append a
        # third IterationTermination on top of the explicit list.
        d["run"].pop("max_iterations", None)
        d["termination"] = [
            {"type": "iteration", "max_iterations": 50},
            {"type": "energy", "min_energy": -99.0},
        ]
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert len(resolved.termination) == 2

    def test_termination_iteration_fixture_yaml(self):
        fixture = _DATA / "termination_iteration.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        assert root.termination is not None
        assert len(root.termination) == 1
        from jaxrens.cli.schema.termination import IterationTerminationSpec
        assert isinstance(root.termination[0], IterationTerminationSpec)
        assert root.termination[0].max_iterations == 5


# ---------------------------------------------------------------------------
# Adaptation resolver tests (split from TestAdaptationConfig)
# ---------------------------------------------------------------------------

class TestAdaptationResolve:
    def test_resolve_for_no_override_uses_fallbacks(self):
        from jaxrens.cli.schema.adaptation import (
            AdaptationConfig,
            _FALLBACK_MIN_RATE,
            _FALLBACK_MAX_RATE,
            _FALLBACK_ADJUST_FACTOR,
            _FALLBACK_STEP_SIZE_MAX,
        )
        cfg = AdaptationConfig()
        policy = cfg.resolve_for("random_walk")
        assert policy.min_rate == pytest.approx(_FALLBACK_MIN_RATE)
        assert policy.max_rate == pytest.approx(_FALLBACK_MAX_RATE)
        assert policy.adjust_factor == pytest.approx(_FALLBACK_ADJUST_FACTOR)
        assert policy.step_size_max == pytest.approx(_FALLBACK_STEP_SIZE_MAX)

    def test_resolve_for_with_defaults_min_rate(self):
        from jaxrens.cli.schema.adaptation import (
            AdaptationConfig,
            AdaptationPolicy,
            _FALLBACK_MAX_RATE,
        )
        cfg = AdaptationConfig(defaults=AdaptationPolicy(min_rate=0.3))
        policy = cfg.resolve_for("random_walk")
        assert policy.min_rate == pytest.approx(0.3)
        assert policy.max_rate == pytest.approx(_FALLBACK_MAX_RATE)

    def test_resolve_for_per_move_overrides_default(self):
        from jaxrens.cli.schema.adaptation import AdaptationConfig, AdaptationPolicy
        cfg = AdaptationConfig(
            defaults=AdaptationPolicy(min_rate=0.3, max_rate=0.7),
            per_move={"galilean": AdaptationPolicy(min_rate=0.5)},
        )
        policy = cfg.resolve_for("galilean")
        assert policy.min_rate == pytest.approx(0.5)
        assert policy.max_rate == pytest.approx(0.7)

    def test_resolve_for_per_move_none_falls_through_to_defaults(self):
        from jaxrens.cli.schema.adaptation import AdaptationConfig, AdaptationPolicy
        cfg = AdaptationConfig(
            defaults=AdaptationPolicy(adjust_factor=2.0),
            per_move={"random_walk": AdaptationPolicy(min_rate=0.4)},
        )
        policy = cfg.resolve_for("random_walk")
        assert policy.min_rate == pytest.approx(0.4)
        assert policy.adjust_factor == pytest.approx(2.0)

    def test_resolve_for_keyed_by_move_name_not_type(self):
        from jaxrens.cli.schema.adaptation import AdaptationConfig, AdaptationPolicy
        cfg = AdaptationConfig(
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
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert len(resolved.adaptation_policies) == 2
        assert resolved.adaptation_policies[0].min_rate == pytest.approx(0.3)
        assert resolved.adaptation_policies[1].min_rate == pytest.approx(0.45)

    def test_adaptation_overlay_fixture_yaml(self):
        fixture = _DATA / "adaptation_overlay.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert len(resolved.adaptation_policies) == 2
        rw_policy = resolved.adaptation_policies[0]
        gal_policy = resolved.adaptation_policies[1]
        assert rw_policy.min_rate == pytest.approx(0.3)
        assert gal_policy.min_rate == pytest.approx(0.4)
        assert gal_policy.step_size_max == pytest.approx(5.0)
        assert gal_policy.max_rate == pytest.approx(0.7)

    def test_resolved_policies_use_fallback_when_no_adaptation_set(self):
        from jaxrens.cli.schema.adaptation import _FALLBACK_MIN_RATE
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert len(resolved.adaptation_policies) == 1
        assert resolved.adaptation_policies[0].min_rate == pytest.approx(_FALLBACK_MIN_RATE)


# ---------------------------------------------------------------------------
# EnsembleSpec resolver tests (split from TestEnsembleSpec)
# ---------------------------------------------------------------------------

_GPA_TO_EVA3 = 0.006241509  # 1 GPa in eV/Å³


class TestEnsembleResolver:
    def test_nvt_to_ensemble_params_empty(self):
        from jaxrens.cli.schema.ensemble import NVTEnsembleSpec
        spec = NVTEnsembleSpec()
        params = spec.to_ensemble_params()
        assert params == {}

    def test_npt_scalar_to_ensemble_params(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=0.05)
        params = spec.to_ensemble_params(cohort_index=0)
        assert "pressure" in params
        assert params["pressure"] == pytest.approx(0.05)

    def test_npt_list_to_ensemble_params_by_index(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        pressures = [0.01, 0.02, 0.03]
        spec = NPTEnsembleSpec(pressure=pressures)
        for i, p in enumerate(pressures):
            params = spec.to_ensemble_params(cohort_index=i)
            assert params["pressure"] == pytest.approx(p)

    def test_npt_gpa_converts_to_eva3(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=1.0, pressure_units="gpa")
        params = spec.to_ensemble_params(cohort_index=0)
        assert params["pressure"] == pytest.approx(_GPA_TO_EVA3, rel=1e-5)

    def test_npt_eva3_passes_through_unchanged(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=0.007, pressure_units="eva3")
        params = spec.to_ensemble_params(cohort_index=0)
        assert params["pressure"] == pytest.approx(0.007)

    def test_npt_gpa_list_conversion(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=[1.0, 2.0], pressure_units="gpa")
        assert spec.to_ensemble_params(cohort_index=0)["pressure"] == pytest.approx(_GPA_TO_EVA3)
        assert spec.to_ensemble_params(cohort_index=1)["pressure"] == pytest.approx(2.0 * _GPA_TO_EVA3)

    def test_legacy_pressure_resolver_synthesizes_correct_ensemble_params(self):
        d = _minimal_dict()
        d["run"]["pressure"] = 0.03
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert resolved.ensemble_params == {"pressure": pytest.approx(0.03)}
        assert resolved.ns.pressure == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Cohort expansion resolver tests (yaml-fixture subset from TestCohortExpansion)
# ---------------------------------------------------------------------------

class TestCohortExpansionResolver:
    def test_npt_three_pressures_and_values(self):
        """Merged from test_npt_three_pressures_three_configs and
        test_npt_three_pressures_correct_values (Step D: collapsed)."""
        d = _minimal_dict()
        d["run"]["seed"] = 10
        d["ensemble"] = {"type": "npt", "pressure": [0.01, 0.02, 0.03]}
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)

        # Shape assertion (was test_npt_three_pressures_three_configs)
        assert len(cohort) == 3

        # Value assertions (was test_npt_three_pressures_correct_values)
        assert cohort[0].ensemble_params["pressure"] == pytest.approx(0.01)
        assert cohort[1].ensemble_params["pressure"] == pytest.approx(0.02)
        assert cohort[2].ensemble_params["pressure"] == pytest.approx(0.03)

    def test_npt_sweep_fixture_yaml_validates(self):
        fixture = _DATA / "npt_sweep.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        assert isinstance(root.ensemble, NPTEnsembleSpec)
        cohort = expand_cohort(root)
        assert len(cohort) == 2

    def test_npt_scalar_fixture_yaml_single_cohort(self):
        fixture = _DATA / "npt_scalar.yaml"
        with open(fixture) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        cohort = expand_cohort(root)
        assert len(cohort) == 1
        assert cohort[0].ensemble_params["pressure"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# CellConfig resolver-warning tests (split from TestCellConfig)
# ---------------------------------------------------------------------------

class TestCellResolve:
    def test_resolved_config_has_cell_field(self):
        from jaxrens.cli.schema.cell import CellConfig
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert hasattr(resolved, "cell")
        assert isinstance(resolved.cell, CellConfig)

    def test_non_default_cell_resolves_without_warning(self, caplog):
        """Cell values are now consumed by move descriptors — no deferred warning."""
        import logging
        d = _minimal_dict()
        d["cell"] = {"max_volume_per_atom": 9999.0}
        root = RootConfig.model_validate(d)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            resolved = resolve(root)
        deferred_warnings = [
            r for r in caplog.records
            if "not yet" in r.message.lower() and "cell" in r.message.lower()
        ]
        assert len(deferred_warnings) == 0
        assert resolved.cell.max_volume_per_atom == pytest.approx(9999.0)

    def test_default_cell_no_warning(self, caplog):
        import logging
        root = RootConfig.model_validate(_minimal_dict())
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            resolve(root)
        cell_warnings = [
            r for r in caplog.records
            if "cell" in r.message.lower() and "not yet" in r.message.lower()
        ]
        assert len(cell_warnings) == 0


# ---------------------------------------------------------------------------
# ExtendedOutput resolver-warning tests (split from TestExtendedOutputSchema)
# ---------------------------------------------------------------------------

class TestExtendedOutputResolve:
    def test_deferred_fields_emit_warnings(self, caplog):
        import logging
        from jaxrens.cli.resolve import _warn_unused_output_fields
        from jaxrens.cli.schema.output import OutputSchema
        schema = OutputSchema(
            format="none",
            wrap_atoms=True,
            write_traj_db=True,
        )
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _warn_unused_output_fields(schema)
        warned = [r.message for r in caplog.records]
        assert any("wrap_atoms" in m for m in warned)
        assert any("write_traj_db" in m for m in warned)

    def test_default_output_no_warnings(self, caplog):
        import logging
        from jaxrens.cli.resolve import _warn_unused_output_fields
        from jaxrens.cli.schema.output import OutputSchema
        schema = OutputSchema(format="none")
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _warn_unused_output_fields(schema)
        assert len(caplog.records) == 0

    def test_resolve_warns_for_deferred_output_fields(self, caplog):
        import logging
        d = _minimal_dict()
        d["output"]["wrap_atoms"] = True
        root = RootConfig.model_validate(d)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            resolve(root)
        assert any("wrap_atoms" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Full config fixture — resolver tests (split from TestFullConfigFixture)
# ---------------------------------------------------------------------------

class TestFullConfigResolver:
    def test_full_config_resolves(self):
        with open(_FULL_CONFIG_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert resolved.init.initial_positions.shape == (20, 2, 3)
        assert resolved.init.initial_types.shape == (2,)

    def test_minimal_yaml_still_resolves_backward_compat(self):
        """Existing minimal fixture still resolves after step-6 additions."""
        with open(_MINIMAL_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert resolved.init.initial_types.shape == (1,)


# ---------------------------------------------------------------------------
# 31. InitConfig resolver — Part A (renamed from first TestInitConfigResolver)
# ---------------------------------------------------------------------------

class TestInitConfigResolverPartA:
    """Tests for _resolve_init and ResolvedInit (Part A resolver).

    Renamed from TestInitConfigResolver (line 1705 of original test_schema.py)
    to avoid collision with the second TestInitConfigResolver class.
    """

    def test_start_species_produces_resolved_init(self):
        from jaxrens.cli.resolve import ResolvedInit, _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        result = _resolve_init(cfg, n_live=10, seed=0)
        assert isinstance(result, ResolvedInit)

    def test_start_species_positions_shape(self):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        result = _resolve_init(cfg, n_live=10, seed=0)
        assert result.initial_positions.shape == (10, 3, 3)

    def test_start_species_types_shape_and_dtype(self):
        import jax.numpy as jnp
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="14 2, 8 1")
        result = _resolve_init(cfg, n_live=5, seed=1)
        assert result.initial_types.shape == (3,)
        assert result.initial_types.dtype == jnp.int32

    def test_start_species_n_atoms_matches_counts(self):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="6 3, 1 2")
        result = _resolve_init(cfg, n_live=4, seed=0)
        assert result.initial_positions.shape[1] == 5

    def test_start_config_file_nonexistent_raises(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_config_file=tmp_path / "does_not_exist.xyz")
        with pytest.raises(FileNotFoundError):
            _resolve_init(cfg, n_live=4, seed=0)

    def test_start_walker_set_nonexistent_raises_file_not_found(self):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_walker_set=Path("/tmp/does_not_exist_walker.extxyz"))
        with pytest.raises(FileNotFoundError):
            _resolve_init(cfg, n_live=4, seed=0)

    def test_restart_file_nonexistent_raises_file_not_found(self):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(restart_file=Path("/tmp/does_not_exist_checkpoint.h5"))
        with pytest.raises(FileNotFoundError):
            _resolve_init(cfg, n_live=4, seed=0)

    def test_resolved_config_has_init_field(self):
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert hasattr(resolved, "init")
        from jaxrens.cli.resolve import ResolvedInit
        assert isinstance(resolved.init, ResolvedInit)

    def test_positions_are_finite(self):
        import jax.numpy as jnp
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        result = _resolve_init(cfg, n_live=8, seed=42)
        assert jnp.all(jnp.isfinite(result.initial_positions))

    def test_deterministic_with_same_seed(self):
        import jax.numpy as jnp
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        r1 = _resolve_init(cfg, n_live=4, seed=99)
        r2 = _resolve_init(cfg, n_live=4, seed=99)
        assert jnp.allclose(r1.initial_positions, r2.initial_positions)

    def test_different_seeds_differ(self):
        import jax.numpy as jnp
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        r1 = _resolve_init(cfg, n_live=4, seed=0)
        r2 = _resolve_init(cfg, n_live=4, seed=1)
        assert not jnp.allclose(r1.initial_positions, r2.initial_positions)


# ---------------------------------------------------------------------------
# Part B resolver tests (renamed from second TestInitConfigResolver, line 2080)
# E2E tests were moved:
#   test_start_species_e2e_run_ns  -> test_init_positions.py
#   test_mode_b_end_to_end_jit     -> test_init_structure.py
# ---------------------------------------------------------------------------

class TestInitConfigResolverPartB:
    """Resolver unit tests for Part B (species/cell/grid resolver logic).

    Renamed from second TestInitConfigResolver (line 2080) to avoid collision
    with TestInitConfigResolverPartA above.
    """

    def test_start_species_cells_shape(self):
        root = RootConfig.model_validate(_species_dict(n_atoms=2, n_live=6))
        resolved = resolve(root)
        assert resolved.init.initial_cells is not None
        assert resolved.init.initial_cells.shape == (6, 3, 3)

    def test_start_species_positions_shape(self):
        root = RootConfig.model_validate(_species_dict(n_atoms=2, n_live=6))
        resolved = resolve(root)
        assert resolved.init.initial_positions is not None
        assert resolved.init.initial_positions.shape == (6, 2, 3)

    def test_start_species_initial_energies_populated(self):
        import jax.numpy as jnp
        root = RootConfig.model_validate(_species_dict(n_atoms=2, n_live=4))
        resolved = resolve(root)
        assert resolved.init.initial_energies is not None
        assert resolved.init.initial_energies.shape == (4,)
        assert jnp.all(jnp.isfinite(resolved.init.initial_energies))

    def test_random_initialise_cell_true_produces_diverse_cells(self):
        import jax.numpy as jnp
        d = _species_dict(n_atoms=2, n_live=8)
        d["init"]["random_initialise_cell"] = True
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        cells = resolved.init.initial_cells
        assert cells is not None
        assert cells.shape == (8, 3, 3)
        diffs = jnp.abs(cells[1:] - cells[:-1])
        assert jnp.any(diffs > 1e-5)

    def test_grid_mode_pairwise_distances(self):
        import jax.numpy as jnp
        root = RootConfig.model_validate(_species_dict(n_atoms=2, n_live=4, mode="grid"))
        resolved = resolve(root)
        positions = resolved.init.initial_positions
        grid_dist = 1.5
        for wi in range(4):
            p = positions[wi]
            d = float(jnp.linalg.norm(p[0] - p[1]))
            assert d >= grid_dist - 1e-4, f"Walker {wi}: pairwise dist {d} < {grid_dist}"

    def test_uniform_mode_energies_below_ceiling(self):
        import jax.numpy as jnp
        n_atoms = 2
        ceiling_per_atom = 1e6
        d = _species_dict(n_atoms=n_atoms, n_live=4, mode="uniform")
        d["init"]["start_energy_ceiling_per_atom"] = ceiling_per_atom
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        energies = resolved.init.initial_energies
        assert energies is not None
        ceiling_total = ceiling_per_atom * n_atoms
        assert jnp.all(energies <= ceiling_total + 1e-3)

    def test_random_initialise_pos_false_emits_warning(self, caplog):
        import logging
        d = _species_dict(n_atoms=2, n_live=4)
        d["init"]["random_initialise_pos"] = False
        root = RootConfig.model_validate(d)
        with caplog.at_level(logging.WARNING):
            resolve(root)
        assert any("correlation" in rec.message.lower() for rec in caplog.records)

    def test_start_species_symbol_map_populated(self):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="14 8")
        result = _resolve_init(cfg, n_live=4, seed=0)
        assert result.symbol_map is not None
        assert isinstance(result.symbol_map, dict)

    # -----------------------------------------------------------------
    # Mode B: start_config_file
    # -----------------------------------------------------------------

    def _make_founder(self, tmp_path, symbols=("Si", "Si"), cell_size=6.0):
        """Write a minimal single-frame extxyz and return its path."""
        import ase, ase.io
        import numpy as _np
        pos = _np.zeros((len(symbols), 3), dtype=_np.float32)
        for i in range(len(symbols)):
            pos[i, 0] = i * (cell_size / (len(symbols) + 1))
        cell = _np.eye(3) * cell_size
        atoms = ase.Atoms(symbols=list(symbols), positions=pos, cell=cell, pbc=True)
        p = tmp_path / "founder.extxyz"
        ase.io.write(str(p), atoms)
        return p

    def test_mode_b_random_pos_true_positions_shape(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=6.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellConfig(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=5, seed=0, cell_cfg=cell_cfg)
        assert result.initial_positions.shape == (5, 2, 3)

    def test_mode_b_random_pos_false_identical_positions(self, tmp_path, caplog):
        import numpy as np
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si", "O"], cell_size=6.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        cell_cfg = CellConfig(
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
        assert any("correlated" in r.message.lower() or "identical" in r.message.lower()
                   for r in caplog.records)

    def test_mode_b_random_pos_false_warning_mentions_burn_in(self, tmp_path, caplog):
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si"], cell_size=5.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        cell_cfg = CellConfig(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(cfg, n_live=3, seed=0, cell_cfg=cell_cfg)
        assert any("burn-in" in r.message for r in caplog.records)

    def test_mode_b_random_cell_true_cells_diverge(self, tmp_path):
        import numpy as np
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=8.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=True,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellConfig(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=4, seed=7, cell_cfg=cell_cfg)
        cells = np.array(result.initial_cells)
        all_same = all(np.allclose(cells[0], cells[wi]) for wi in range(1, 4))
        assert not all_same

    def test_mode_b_random_cell_false_cells_identical(self, tmp_path):
        import numpy as np
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=6.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellConfig(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=4, seed=0, cell_cfg=cell_cfg)
        cells = np.array(result.initial_cells)
        for wi in range(1, 4):
            np.testing.assert_allclose(cells[0], cells[wi])

    def test_mode_b_symbol_map_from_file(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si", "O"], cell_size=6.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellConfig(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=3, seed=0, cell_cfg=cell_cfg)
        assert result.symbol_map == {0: "Si", 1: "O"}

    def test_mode_b_positions_are_finite(self, tmp_path):
        import jax.numpy as jnp
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si", "Si"], cell_size=6.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellConfig(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        result = _resolve_init(cfg, n_live=4, seed=0, cell_cfg=cell_cfg)
        assert jnp.all(jnp.isfinite(result.initial_positions))

    def test_mode_b_energies_computed_with_backend(self, tmp_path):
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        p = self._make_founder(tmp_path, symbols=["Si"], cell_size=5.0)
        cfg = InitConfig(
            start_config_file=p,
            random_initialise_pos=True,
            random_initialise_cell=False,
            pos_randomization_mode="uniform",
        )
        cell_cfg = CellConfig(
            max_volume_per_atom=1000.0,
            min_volume_per_atom=0.1,
            min_aspect_ratio=0.01,
        )
        backend = create_harmonic()
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=backend, cell_cfg=cell_cfg)
        assert result.initial_energies is not None
        assert result.initial_energies.shape == (3,)
        assert jnp.all(jnp.isfinite(result.initial_energies))


# ---------------------------------------------------------------------------
# Backend-aware species mapping
#
# Model-based backends (MACE, future NeuralIL) expose an ``atomic_numbers``
# attribute defining the z-table the model was trained on. Species indices
# must match that table; the default 0-based unique mapping gives wrong
# one-hot encodings for multi-element systems.
# ---------------------------------------------------------------------------

class _FakeZTableBackend:
    """Stand-in for MACE: exposes ``atomic_numbers`` and a dummy __call__.

    Covers the resolver's backend-aware species mapping without needing
    mace-jax installed.
    """

    r_cutoff = 5.0

    def __init__(self, atomic_numbers: list[int]):
        self.atomic_numbers = list(atomic_numbers)

    def __call__(self, positions, species, cell, max_neighbors, ensemble_params=None):
        import jax.numpy as jnp
        return jnp.float32(0.0), jnp.int32(0), jnp.bool_(False)


class TestBackendAwareSpeciesMapping:
    """Resolver respects ``energy_backend.atomic_numbers`` when present."""

    def test_no_atomic_numbers_keeps_zero_based_mapping(self):
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        cfg = InitConfig(
            start_species="8 3, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(
            cfg, n_live=2, seed=0, energy_backend=create_harmonic(),
        )
        # Sorted unique Z = [8, 22, 38] → 0-based indices [0, 1, 2].
        # types_list order is sorted-by-Z: O O O Ti Sr → idx [0,0,0,1,2].
        assert list(map(int, result.initial_types)) == [0, 0, 0, 1, 2]
        assert result.symbol_map == {0: "O", 1: "Ti", 2: "Sr"}

    def test_backend_z_table_overrides_mapping(self):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        # Mimic mace_mp's 89-element z-table (Z=1..89).
        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        cfg = InitConfig(
            start_species="8 3, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(
            cfg, n_live=2, seed=0, energy_backend=backend,
        )
        # Z=8→idx 7 (O), Z=22→idx 21 (Ti), Z=38→idx 37 (Sr).
        assert list(map(int, result.initial_types)) == [7, 7, 7, 21, 37]
        assert result.symbol_map == {7: "O", 21: "Ti", 37: "Sr"}

    def test_missing_z_in_backend_table_raises(self):
        import pytest
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        # Backend supports only a small subset (no Sr, Z=38).
        backend = _FakeZTableBackend(atomic_numbers=[1, 8, 22])
        cfg = InitConfig(
            start_species="8 1, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        with pytest.raises(ValueError, match="atomic numbers"):
            _resolve_init(cfg, n_live=2, seed=0, energy_backend=backend)

    def test_ensemble_wrapper_passes_atomic_numbers_through(self):
        from jaxrens.backends.ensemble import EnsembleBackend
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        backend = _FakeZTableBackend(atomic_numbers=list(range(1, 90)))
        wrapped = EnsembleBackend(backend, pressure=0.1)
        assert wrapped.atomic_numbers == list(range(1, 90))

        cfg = InitConfig(
            start_species="8 3, 22 1, 38 1",
            random_initialise_pos=False,
            random_initialise_cell=False,
        )
        result = _resolve_init(
            cfg, n_live=2, seed=0, energy_backend=wrapped,
        )
        assert list(map(int, result.initial_types)) == [7, 7, 7, 21, 37]
