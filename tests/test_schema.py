"""Tests for the pydantic schema layer and CLI entry point.

Resolver-layer tests have been extracted to tests/test_resolve.py.
Mode-C/D resolver tests live in tests/test_init_walker_set.py and
tests/test_init_restart.py respectively.
JIT plumbing tests live in tests/test_mwg.py and tests/test_termination.py.
Burn-in integration tests live in tests/test_init_burn_in.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from jaxrens.cli.schema import RootConfig
from jaxrens.cli.schema.backend import (
    BackendSpec,
    BaseBackendSpec,
    DoubleWellBackendSpec,
    GaussianMixtureBackendSpec,
    HarmonicBackendSpec,
    LJBackendSpec,
    MACEBackendSpec,
    NeuralILBackendSpec,
)
from jaxrens.cli.schema.moves import (
    MoveSpec,
    RandomWalkMoveSpec,
    GalileanMoveSpec,
    GmcMoveSpec,
    HMCMoveSpec,
    SingleAtomMoveSpec,
    SingleAtomSweepMoveSpec,
    SingleAtomSwapMoveSpec,
    VolumeMoveSpec,
    ShearMoveSpec,
    StretchMoveSpec,
    AlchemicalMorphMoveSpec,
    AlchemicalShiftMoveSpec,
)
from jaxrens.cli.resolve import resolve, expand_cohort, ResolvedConfig
from jaxrens.cli.cli import _apply_overrides, _parse_set_override
from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig

_DATA = Path(__file__).parent / "data" / "cli"
_MINIMAL_YAML = _DATA / "minimal.yaml"
_MIXED_MOVES_YAML = _DATA / "mixed_moves.yaml"
_LJ_BACKEND_YAML = _DATA / "lj_backend.yaml"


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


# ---------------------------------------------------------------------------
# 1. Valid minimal dict validates without error
# ---------------------------------------------------------------------------

class TestRootConfigValidate:
    def test_minimal_dict_validates(self):
        root = RootConfig.model_validate(_minimal_dict())
        assert root.run.n_live == 20
        assert len(root.moves) == 1
        assert root.moves[0].move_type == "random_walk"
        assert root.backend.backend_type == "harmonic"
        assert root.output.format == "none"

    def test_defaults_applied(self):
        root = RootConfig.model_validate(_minimal_dict())
        assert root.run.convergence_threshold == 0.1
        assert root.moves[0].weight == 1.0
        assert root.output.traj_interval == 1


# ---------------------------------------------------------------------------
# 2. extra="forbid" rejects unknown keys
# ---------------------------------------------------------------------------

class TestExtraForbid:
    def test_unknown_root_key_raises(self):
        d = _minimal_dict()
        d["not_a_key"] = 42
        with pytest.raises(ValidationError, match="not_a_key"):
            RootConfig.model_validate(d)

    def test_unknown_run_key_raises(self):
        d = _minimal_dict()
        d["run"]["typo_field"] = 1
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)

    def test_unknown_backend_key_raises(self):
        d = _minimal_dict()
        d["backend"]["mystery"] = "x"
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)


# ---------------------------------------------------------------------------
# 3. moves: normalization — dict becomes single-element list
# ---------------------------------------------------------------------------

class TestMovesNormalization:
    def test_single_dict_normalized_to_list(self):
        d = _minimal_dict()
        d["moves"] = {"move_type": "galilean", "step_size": 0.1}
        root = RootConfig.model_validate(d)
        assert isinstance(root.moves, list)
        assert len(root.moves) == 1
        assert root.moves[0].move_type == "galilean"

    def test_list_of_two_moves(self):
        d = _minimal_dict()
        d["moves"] = [
            {"move_type": "random_walk", "step_size": 0.2},
            {"move_type": "galilean", "step_size": 0.05},
        ]
        root = RootConfig.model_validate(d)
        assert len(root.moves) == 2
        assert root.moves[1].move_type == "galilean"


# ---------------------------------------------------------------------------
# 4. --set override: nested scalar
# ---------------------------------------------------------------------------

class TestSetOverride:
    def test_nested_scalar_patch(self):
        raw = _minimal_dict()
        patched = _apply_overrides(raw, ["run.n_live=123"])
        root = RootConfig.model_validate(patched)
        assert root.run.n_live == 123

    def test_multiple_overrides_last_wins(self):
        raw = _minimal_dict()
        patched = _apply_overrides(raw, ["run.n_live=10", "run.n_live=99"])
        root = RootConfig.model_validate(patched)
        assert root.run.n_live == 99

    def test_bool_override(self):
        raw = _minimal_dict()
        patched = _apply_overrides(raw, ["backend.periodic=true"])
        root = RootConfig.model_validate(patched)
        assert root.backend.periodic is True


# ---------------------------------------------------------------------------
# 5. --set bracket indexing into moves list
# ---------------------------------------------------------------------------

class TestSetBracketIndex:
    def test_bracket_index_step_size(self):
        d = _minimal_dict()
        d["moves"] = [{"move_type": "random_walk", "step_size": 0.3}]
        patched = _apply_overrides(d, ["moves[0].step_size=0.25"])
        root = RootConfig.model_validate(patched)
        assert root.moves[0].step_size == pytest.approx(0.25)

    def test_bracket_index_move_type(self):
        d = _minimal_dict()
        d["moves"] = [{"move_type": "random_walk"}]
        patched = _apply_overrides(d, ["moves[0].move_type=galilean"])
        root = RootConfig.model_validate(patched)
        assert root.moves[0].move_type == "galilean"

    def test_parse_set_override_bracket(self):
        path, value = _parse_set_override("moves[0].step_size=0.5")
        assert path == ["moves", 0, "step_size"]
        assert value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 6. Invalid move_type literal raises clearly
# ---------------------------------------------------------------------------

class TestMoveTypeLiteral:
    def test_invalid_move_type_raises(self):
        d = _minimal_dict()
        d["moves"] = [{"move_type": "not_a_real_move"}]
        with pytest.raises(ValidationError, match="not_a_real_move"):
            RootConfig.model_validate(d)

    def test_all_registry_types_accepted(self):
        from jaxrens.cli.schema.moves import MoveType
        import typing
        valid_types = list(typing.get_args(MoveType))
        assert len(valid_types) >= 10


# ---------------------------------------------------------------------------
# 8. Round-trip: YAML -> RootConfig -> model_dump() -> re-validate
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_model_dump_round_trips(self):
        root1 = RootConfig.model_validate(_minimal_dict())
        dumped = root1.model_dump()
        dumped["output"]["working_dir"] = str(dumped["output"]["working_dir"])
        root2 = RootConfig.model_validate(dumped)
        assert root1.run == root2.run
        assert root1.moves == root2.moves
        assert root1.backend == root2.backend
        assert root1.output == root2.output

    def test_yaml_round_trip(self):
        root1 = RootConfig.model_validate(_minimal_dict())
        dumped = root1.model_dump(mode="json")
        reloaded = yaml.safe_load(yaml.safe_dump(dumped))
        root2 = RootConfig.model_validate(reloaded)
        assert root1.run == root2.run
        assert root1.moves == root2.moves
        assert root1.backend == root2.backend


# ---------------------------------------------------------------------------
# 9. Fixture YAML file validates and CLI validate subcommand works
# ---------------------------------------------------------------------------

class TestFixtureYAML:
    def test_minimal_yaml_validates(self):
        with open(_MINIMAL_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        assert root.run.n_live == 20

    def test_cli_validate_subcommand(self, capsys):
        from jaxrens.cli.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_MINIMAL_YAML)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out
        assert "random_walk" in captured.out

    def test_cli_validate_with_set_override(self, capsys):
        from jaxrens.cli.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_MINIMAL_YAML), "--set", "run.n_live=77"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out
        assert "n_live=77" in captured.out

    def test_cli_dump_schema(self, capsys):
        from jaxrens.cli.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["dump-schema"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        import json
        schema = json.loads(captured.out)
        assert "RootConfig" in schema.get("title", "") or "properties" in schema


# ---------------------------------------------------------------------------
# 10. Discriminated union: per-move-type instantiation
# ---------------------------------------------------------------------------

class TestDiscriminatedUnion:
    """Each move type parses to the correct spec subclass via the union."""

    @pytest.mark.parametrize("move_dict,expected_cls", [
        ({"type": "random_walk"}, RandomWalkMoveSpec),
        ({"type": "galilean"}, GalileanMoveSpec),
        ({"type": "gmc"}, GmcMoveSpec),
        ({"type": "hmc"}, HMCMoveSpec),
        ({"type": "single_atom"}, SingleAtomMoveSpec),
        ({"type": "single_atom_sweep"}, SingleAtomSweepMoveSpec),
        ({"type": "single_atom_swap"}, SingleAtomSwapMoveSpec),
        ({"type": "volume"}, VolumeMoveSpec),
        ({"type": "shear"}, ShearMoveSpec),
        ({"type": "stretch"}, StretchMoveSpec),
        ({"type": "alchemical_morph", "n_species": 2}, AlchemicalMorphMoveSpec),
        ({"type": "alchemical_shift"}, AlchemicalShiftMoveSpec),
    ])
    def test_correct_subclass_instantiated(self, move_dict, expected_cls):
        d = _minimal_dict()
        d["moves"] = [move_dict]
        root = RootConfig.model_validate(d)
        assert isinstance(root.moves[0], expected_cls)

    def test_nonexistent_type_raises(self):
        d = _minimal_dict()
        d["moves"] = [{"type": "nonexistent_move"}]
        with pytest.raises(ValidationError, match="nonexistent_move"):
            RootConfig.model_validate(d)

    def test_move_type_property_matches_discriminator(self):
        d = _minimal_dict()
        d["moves"] = [{"type": "random_walk"}]
        root = RootConfig.model_validate(d)
        assert root.moves[0].move_type == "random_walk"


# ---------------------------------------------------------------------------
# 11. Mixed-move list: two different types
# ---------------------------------------------------------------------------

class TestMixedMoves:
    def test_mixed_move_list_parses(self):
        with open(_MIXED_MOVES_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        assert len(root.moves) == 2
        assert isinstance(root.moves[0], RandomWalkMoveSpec)
        assert isinstance(root.moves[1], GalileanMoveSpec)

    def test_mixed_moves_resolve_to_distinct_descriptors(self):
        with open(_MIXED_MOVES_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert len(resolved.move_descriptors) == 2
        d0, d1 = resolved.move_descriptors
        import jaxrens.sampling.moves.random_walk as rw_mod
        import jaxrens.sampling.moves.galilean as gal_mod
        assert d0.build_kernel is rw_mod.build_kernel
        assert d1.build_kernel is gal_mod.build_kernel

    def test_same_type_different_name_distinguishable(self):
        d = _minimal_dict()
        d["moves"] = [
            {"type": "random_walk", "step_size": 0.1, "name": "rw_slow"},
            {"type": "random_walk", "step_size": 0.5, "name": "rw_fast"},
        ]
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert resolved.move_descriptors[0].name == "rw_slow"
        assert resolved.move_descriptors[1].name == "rw_fast"
        assert resolved.move_descriptors[0].step_size != resolved.move_descriptors[1].step_size

    def test_extra_field_on_spec_raises(self):
        d = _minimal_dict()
        d["moves"] = [{"type": "random_walk", "n_reflect": 5}]
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)


# ---------------------------------------------------------------------------
# 16. BackendSpec discriminated union: per-type instantiation
# ---------------------------------------------------------------------------

class TestBackendDiscriminatedUnion:
    """Each backend type parses to the correct spec subclass via the union."""

    @pytest.mark.parametrize("backend_dict,expected_cls", [
        ({"type": "harmonic"}, HarmonicBackendSpec),
        ({"type": "double_well"}, DoubleWellBackendSpec),
        ({"type": "gaussian_mixture"}, GaussianMixtureBackendSpec),
        ({"type": "lj"}, LJBackendSpec),
        (
            {"type": "neuralil", "checkpoint_path": "/tmp/model.pkl"},
            NeuralILBackendSpec,
        ),
        (
            {"type": "mace", "checkpoint_path": "/tmp/model"},
            MACEBackendSpec,
        ),
    ])
    def test_correct_subclass_instantiated(self, backend_dict, expected_cls):
        d = _minimal_dict()
        d["backend"] = backend_dict
        root = RootConfig.model_validate(d)
        assert isinstance(root.backend, expected_cls)

    def test_unknown_type_raises(self):
        d = _minimal_dict()
        d["backend"] = {"type": "nonexistent_backend"}
        with pytest.raises(ValidationError, match="nonexistent_backend"):
            RootConfig.model_validate(d)

    def test_backend_type_property_matches_discriminator(self):
        d = _minimal_dict()
        d["backend"] = {"type": "harmonic"}
        root = RootConfig.model_validate(d)
        assert root.backend.backend_type == "harmonic"

    def test_legacy_backend_type_key_accepted(self):
        d = _minimal_dict()
        d["backend"] = {"backend_type": "harmonic"}
        root = RootConfig.model_validate(d)
        assert isinstance(root.backend, HarmonicBackendSpec)
        assert root.backend.backend_type == "harmonic"


# ---------------------------------------------------------------------------
# 17. Per-backend extra-field rejection
# ---------------------------------------------------------------------------

class TestBackendExtraForbid:
    def test_harmonic_rejects_unknown_field(self):
        d = _minimal_dict()
        d["backend"] = {"type": "harmonic", "not_a_field": 99}
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)

    def test_lj_rejects_unknown_field(self):
        d = _minimal_dict()
        d["backend"] = {"type": "lj", "mystery": "x"}
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)

    def test_lj_rejects_harmonic_only_field(self):
        d = _minimal_dict()
        d["backend"] = {"type": "lj", "k": 2.0}
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)

    def test_harmonic_rejects_lj_only_field(self):
        d = _minimal_dict()
        d["backend"] = {"type": "harmonic", "epsilon": 1.0}
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)


# ---------------------------------------------------------------------------
# 20. Fixture YAML: lj_backend.yaml validates and round-trips
# ---------------------------------------------------------------------------

class TestLJBackendFixture:
    def test_lj_backend_yaml_validates(self):
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        assert isinstance(root.backend, LJBackendSpec)
        assert root.backend.cutoff == pytest.approx(3.0)

    def test_lj_backend_yaml_to_backend_config(self):
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        cfg = root.backend.to_backend_config()
        assert cfg.backend_type == "lj"
        assert cfg.cutoff == pytest.approx(3.0)

    def test_lj_backend_yaml_builds_backend(self):
        from jaxrens.backends.lj import LJBackend
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        backend = root.backend.build_backend()
        assert isinstance(backend, LJBackend)


# ---------------------------------------------------------------------------
# 23. TerminationSpec discriminated union — schema-level tests
# (to_criterion tests moved to test_resolve.py::TestToCriterion)
# ---------------------------------------------------------------------------

class TestTerminationDiscriminatedUnion:
    """Each termination type parses to the correct spec subclass."""

    @pytest.mark.parametrize("term_dict,expected_cls_name", [
        ({"type": "iteration", "max_iterations": 100}, "IterationTerminationSpec"),
        ({"type": "prior_mass", "n_live": 20}, "PriorMassTerminationSpec"),
        (
            {"type": "temperature", "n_walkers": 20, "target_temp": 300.0},
            "TemperatureTerminationSpec",
        ),
        ({"type": "energy", "min_energy": -10.0}, "EnergyTerminationSpec"),
    ])
    def test_correct_subclass_instantiated(self, term_dict, expected_cls_name):
        from pydantic import TypeAdapter
        from jaxrens.cli.schema.termination import TerminationSpec
        ta = TypeAdapter(TerminationSpec)
        spec = ta.validate_python(term_dict)
        assert type(spec).__name__ == expected_cls_name

    def test_unknown_type_raises(self):
        from pydantic import TypeAdapter, ValidationError
        from jaxrens.cli.schema.termination import TerminationSpec
        ta = TypeAdapter(TerminationSpec)
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "nonexistent_criterion"})

    def test_extra_field_rejected_on_iteration(self):
        from pydantic import TypeAdapter, ValidationError
        from jaxrens.cli.schema.termination import TerminationSpec
        ta = TypeAdapter(TerminationSpec)
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "iteration", "max_iterations": 5, "bogus": 1})

    def test_extra_field_rejected_on_energy(self):
        from pydantic import TypeAdapter, ValidationError
        from jaxrens.cli.schema.termination import TerminationSpec
        ta = TypeAdapter(TerminationSpec)
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "energy", "min_energy": -5.0, "extra": True})

    def test_single_dict_termination_normalized_to_list(self):
        d = _minimal_dict()
        d["termination"] = {"type": "iteration", "max_iterations": 7}
        root = RootConfig.model_validate(d)
        assert len(root.termination) == 1
        assert root.termination[0].type == "iteration"


# ---------------------------------------------------------------------------
# 25. AdaptationConfig — schema-level tests
# (resolve_for tests moved to test_resolve.py::TestAdaptationResolve)
# ---------------------------------------------------------------------------

class TestAdaptationConfig:
    def test_default_adaptation_config(self):
        from jaxrens.cli.schema.adaptation import AdaptationConfig
        cfg = AdaptationConfig()
        assert cfg.full_auto is False
        assert cfg.full_auto_steps == 0
        assert cfg.per_move == {}
        assert cfg.defaults.min_rate is None

    def test_adaptation_extra_field_rejected(self):
        d = _minimal_dict()
        d["adaptation"] = {"bogus_field": True}
        with pytest.raises(ValidationError):
            RootConfig.model_validate(d)

    def test_adaptation_policy_extra_field_rejected(self):
        from jaxrens.cli.schema.adaptation import AdaptationPolicy
        with pytest.raises(ValidationError):
            AdaptationPolicy(min_rate=0.3, bogus=1)

    def test_default_adaptation_config_round_trip(self):
        d = _minimal_dict()
        root = RootConfig.model_validate(d)
        assert root.adaptation.full_auto is False
        assert root.adaptation.defaults.min_rate is None


# ---------------------------------------------------------------------------
# 26. EnsembleSpec — schema-level tests
# (resolver tests moved to test_resolve.py::TestEnsembleResolver)
# ---------------------------------------------------------------------------

_GPA_TO_EVA3 = 0.006241509  # 1 GPa in eV/Å³


class TestEnsembleSpec:
    def test_nvt_minimal_validates(self):
        from jaxrens.cli.schema.ensemble import NVTEnsembleSpec
        spec = NVTEnsembleSpec()
        assert spec.type == "nvt"

    def test_npt_scalar_pressure_validates(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=0.01)
        assert spec.pressure == pytest.approx(0.01)
        assert spec.pressure_units == "eva3"

    def test_npt_list_pressure_validates(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=[0.01, 0.02, 0.03])
        assert len(spec.pressure) == 3

    def test_ensemble_discriminator_mismatch_raises(self):
        from pydantic import TypeAdapter, ValidationError
        from jaxrens.cli.schema.ensemble import EnsembleSpec
        ta = TypeAdapter(EnsembleSpec)
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "nonexistent_ensemble"})

    def test_nvt_extra_field_rejected(self):
        from jaxrens.cli.schema.ensemble import NVTEnsembleSpec
        with pytest.raises(ValidationError):
            NVTEnsembleSpec(bogus=1)

    def test_npt_extra_field_rejected(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        with pytest.raises(ValidationError):
            NPTEnsembleSpec(pressure=0.01, unknown_key="x")

    def test_nvt_in_root_config_default(self):
        from jaxrens.cli.schema.ensemble import NVTEnsembleSpec
        root = RootConfig.model_validate(_minimal_dict())
        assert isinstance(root.ensemble, NVTEnsembleSpec)

    def test_npt_in_root_config(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": 0.01}
        root = RootConfig.model_validate(d)
        assert isinstance(root.ensemble, NPTEnsembleSpec)
        assert root.ensemble.pressure == pytest.approx(0.01)

    def test_legacy_run_pressure_synthesizes_npt(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        d = _minimal_dict()
        d["run"]["pressure"] = 0.05
        root = RootConfig.model_validate(d)
        assert isinstance(root.ensemble, NPTEnsembleSpec)
        assert root.ensemble.pressure == pytest.approx(0.05)
        assert root.ensemble.pressure_units == "eva3"

    def test_legacy_run_pressure_plus_ensemble_raises(self):
        d = _minimal_dict()
        d["run"]["pressure"] = 0.05
        d["ensemble"] = {"type": "npt", "pressure": 0.02}
        with pytest.raises(ValidationError, match="Conflicting"):
            RootConfig.model_validate(d)


# ---------------------------------------------------------------------------
# 27. Cohort expansion — schema-level tests
# (yaml-fixture resolver tests moved to test_resolve.py::TestCohortExpansionResolver)
# ---------------------------------------------------------------------------

class TestCohortExpansion:
    def test_nvt_scalar_seed_single_element(self):
        root = RootConfig.model_validate(_minimal_dict())
        cohort = expand_cohort(root)
        assert len(cohort) == 1

    def test_nvt_cohort_index_zero(self):
        root = RootConfig.model_validate(_minimal_dict())
        cohort = expand_cohort(root)
        assert cohort[0].cohort_index == 0

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

    def test_npt_scalar_seed_auto_incremented(self):
        d = _minimal_dict()
        d["run"]["seed"] = 10
        d["ensemble"] = {"type": "npt", "pressure": [0.01, 0.02, 0.03]}
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)
        assert cohort[0].ns.seed == 10
        assert cohort[1].ns.seed == 11
        assert cohort[2].ns.seed == 12

    def test_npt_cohort_indices_correct(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": [0.01, 0.02]}
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)
        assert cohort[0].cohort_index == 0
        assert cohort[1].cohort_index == 1

    def test_single_element_cohort_from_scalar_npt(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": 0.01}
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)
        assert len(cohort) == 1
        assert cohort[0].ensemble_params["pressure"] == pytest.approx(0.01)

    def test_resolve_wraps_single_element_cohort(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": 0.01}
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert resolved.ensemble_params["pressure"] == pytest.approx(0.01)

    def test_resolve_asserts_on_multi_element_cohort(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": [0.01, 0.02]}
        root = RootConfig.model_validate(d)
        with pytest.raises(AssertionError, match="expand_cohort"):
            resolve(root)


# ---------------------------------------------------------------------------
# 28. Cohort expansion: CLI validate reports cohort size
# ---------------------------------------------------------------------------

class TestValidateCohortSize:
    def test_validate_reports_cohort_size_1(self, capsys):
        from jaxrens.cli.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "minimal.yaml")])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "cohort size: 1" in captured.out

    def test_validate_reports_cohort_size_2(self, capsys):
        from jaxrens.cli.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "npt_sweep.yaml")])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "cohort size: 2" in captured.out


# ---------------------------------------------------------------------------
# 29. End-to-end cohort run via CLI (sequential, small run)
# ---------------------------------------------------------------------------

class TestCohortRunEndToEnd:
    def test_npt_sweep_two_cohorts_run_sequentially(self):
        """NPT + pressure list of length 2: both cohort elements produce valid NS runs."""
        import jax
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns

        d = {
            "run": {
                "n_live": 8,
                "max_iterations": 5,
                "n_mcmc_steps": 3,
                "seed": 42,
            },
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic"},
            "output": {
                "format": "none",
                "working_dir": ".",
                "info_interval": 999,
            },
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02]},
        }
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)
        assert len(cohort) == 2

        results = []
        for resolved in cohort:
            backend = create_harmonic()
            init_fn, step_fn, _ = build_mwg(backend, list(resolved.move_descriptors))
            key = jax.random.key(resolved.ns.seed)
            key, key_pos = jax.random.split(key)
            positions = jax.random.uniform(key_pos, (8, 1, 3), minval=-3.0, maxval=3.0)
            types = jnp.zeros((1,), dtype=jnp.int32)
            energies = jax.vmap(
                lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
            )(positions)
            result = run_ns(
                positions=positions,
                types=types,
                energies=energies,
                cells=None,
                init_fn=init_fn,
                step_fn=step_fn,
                rng_key=key,
                max_iterations=5,
                n_mcmc_steps=3,
            )
            results.append(result)
            assert result["iteration"] > 0
            assert jnp.isfinite(result["log_evidence"])

        assert results[0]["iteration"] > 0
        assert results[1]["iteration"] > 0
        assert cohort[0].ns.seed != cohort[1].ns.seed

    def test_npt_sweep_jit_path_survives_tracing(self):
        """Cohort with NPT + pressure list: each element's ns_step traces under JIT."""
        import jax
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        d = {
            "run": {"n_live": 6, "max_iterations": 3, "n_mcmc_steps": 2, "seed": 5},
            "moves": [{"type": "random_walk", "step_size": 0.2}],
            "backend": {"type": "harmonic"},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02]},
        }
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))

        for resolved in cohort:
            backend = create_harmonic()
            init_fn, step_fn, _ = build_mwg(backend, list(resolved.move_descriptors))
            key = jax.random.key(resolved.ns.seed)
            key, key_pos = jax.random.split(key)
            positions = jax.random.uniform(key_pos, (6, 1, 3), minval=-2.0, maxval=2.0)
            types = jnp.zeros((1,), dtype=jnp.int32)
            energies = jax.vmap(
                lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
            )(positions)
            ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)
            new_state, _ = jit_ns_step(ns_state, step_fn, n_mcmc_steps=2)
            assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0


# ---------------------------------------------------------------------------
# 30. InitConfig — schema validation
# ---------------------------------------------------------------------------

class TestInitConfig:
    """Tests for the InitConfig pydantic schema (Part A)."""

    def test_start_species_single_element(self):
        """New format: Z N -> {Z: N}."""
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="18 8")
        assert cfg.start_species == "18 8"
        counts = cfg.parsed_species()
        assert counts == {18: 8}

    def test_start_species_multi_element(self):
        """New format: Z1 N1, Z2 N2 -> {Z1: N1, Z2: N2}."""
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 6, 3 6")
        counts = cfg.parsed_species()
        assert counts == {1: 6, 3: 6}
        total = sum(counts.values())
        assert total == 12

    def test_multi_composition_raises(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3: 0 16")
        with pytest.raises(ValueError, match="Multi-composition"):
            cfg.parsed_species()

    def test_mass_token_raises_not_implemented(self):
        """Three-token group (Z N mass) raises NotImplementedError."""
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 6 1.008")
        with pytest.raises(NotImplementedError, match="masses"):
            cfg.parsed_species()

    def test_empty_string_raises(self):
        from jaxrens.cli.schema.init import _parse_species_string
        with pytest.raises(ValueError, match="empty"):
            _parse_species_string("   ")

    def test_non_integer_token_raises(self):
        from jaxrens.cli.schema.init import _parse_species_string
        with pytest.raises(ValueError, match="non-negative integer"):
            _parse_species_string("H 6")

    def test_odd_number_of_tokens_raises(self):
        from jaxrens.cli.schema.init import _parse_species_string
        with pytest.raises(ValueError, match="even number"):
            _parse_species_string("1 6, 3")

    def test_single_token_raises(self):
        from jaxrens.cli.schema.init import _parse_species_string
        with pytest.raises(ValueError, match="even number"):
            _parse_species_string("1")

    def test_zero_sources_raises(self):
        from jaxrens.cli.schema.init import InitConfig
        with pytest.raises(ValidationError, match="none were set"):
            InitConfig()

    def test_two_sources_raises(self):
        from jaxrens.cli.schema.init import InitConfig
        with pytest.raises(ValidationError, match="got:"):
            InitConfig(
                start_species="1 1",
                start_config_file=Path("/tmp/fake.xyz"),
            )

    def test_start_config_file_accepted(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_config_file=Path("/tmp/atoms.xyz"))
        assert cfg.start_config_file == Path("/tmp/atoms.xyz")
        assert cfg.start_species is None

    def test_restart_file_accepted(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(restart_file=Path("/tmp/checkpoint.npz"))
        assert cfg.restart_file == Path("/tmp/checkpoint.npz")

    def test_defaults_are_sane(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 1")
        assert cfg.start_energy_ceiling_per_atom == pytest.approx(1e9)
        assert cfg.random_initialise_pos is True
        assert cfg.random_initialise_cell is True
        assert cfg.pos_randomization_mode == "grid"
        assert cfg.grid_distance == pytest.approx(1.5)
        assert cfg.init_distance_criterion == pytest.approx(1.0)
        assert cfg.random_init_max_n_tries == 100
        assert cfg.pos_autoscale_cells is False

    def test_initial_walk_defaults(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 1")
        assert cfg.initial_walk.n_walks == 0
        assert cfg.initial_walk.walklength == 100

    def test_extra_field_rejected(self):
        from jaxrens.cli.schema.init import InitConfig
        with pytest.raises(ValidationError):
            InitConfig(start_species="1 1", unknown_field=42)

    def test_yaml_round_trip_init(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(
            start_species="14 8",
            pos_randomization_mode="uniform",
            grid_distance=2.0,
        )
        dumped = cfg.model_dump(mode="json")
        import yaml as _yaml
        reloaded_raw = _yaml.safe_load(_yaml.safe_dump(dumped))
        cfg2 = InitConfig.model_validate(reloaded_raw)
        assert cfg2.start_species == "14 8"
        assert cfg2.pos_randomization_mode == "uniform"
        assert cfg2.grid_distance == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 32. CellConfig — schema validation
# (resolver tests moved to test_resolve.py::TestCellResolve)
# ---------------------------------------------------------------------------

class TestCellConfig:
    """Tests for the CellConfig pydantic schema (Part B)."""

    def test_defaults(self):
        from jaxrens.cli.schema.cell import CellConfig
        cfg = CellConfig()
        assert cfg.max_volume_per_atom == pytest.approx(1e4)
        assert cfg.min_volume_per_atom == pytest.approx(1.0)
        assert cfg.min_aspect_ratio == pytest.approx(0.8)
        assert cfg.flat_V_prior is False

    def test_extra_field_rejected(self):
        from jaxrens.cli.schema.cell import CellConfig
        with pytest.raises(ValidationError):
            CellConfig(max_volume_per_atom=100.0, mystery_field=True)

    def test_custom_values_accepted(self):
        from jaxrens.cli.schema.cell import CellConfig
        cfg = CellConfig(
            max_volume_per_atom=500.0,
            min_volume_per_atom=2.0,
            min_aspect_ratio=0.75,
            flat_V_prior=True,
        )
        assert cfg.max_volume_per_atom == pytest.approx(500.0)
        assert cfg.flat_V_prior is True

    def test_yaml_round_trip_cell(self):
        from jaxrens.cli.schema.cell import CellConfig
        cfg = CellConfig(max_volume_per_atom=200.0, min_aspect_ratio=0.6)
        dumped = cfg.model_dump(mode="json")
        import yaml as _yaml
        reloaded_raw = _yaml.safe_load(_yaml.safe_dump(dumped))
        cfg2 = CellConfig.model_validate(reloaded_raw)
        assert cfg2.max_volume_per_atom == pytest.approx(200.0)
        assert cfg2.min_aspect_ratio == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 33. Extended OutputConfig — schema-level tests
# (resolver warning tests moved to test_resolve.py::TestExtendedOutputResolve)
# ---------------------------------------------------------------------------

class TestExtendedOutputSchema:
    """Tests for the extended OutputSchema with deferred fields (Part C)."""

    def test_deferred_fields_have_correct_defaults(self):
        from jaxrens.cli.schema.output import OutputSchema
        schema = OutputSchema()
        assert schema.snapshot_time is None
        assert schema.snapshot_clean is False
        assert schema.wrap_atoms is False
        assert schema.save_stepsizes is False
        assert schema.write_traj_db is False
        assert schema.write_walkers_db is False

    def test_deferred_fields_accept_non_defaults(self):
        from jaxrens.cli.schema.output import OutputSchema
        schema = OutputSchema(
            format="none",
            snapshot_time=60.0,
            snapshot_clean=True,
            wrap_atoms=True,
            save_stepsizes=True,
            write_traj_db=True,
            write_walkers_db=True,
        )
        assert schema.snapshot_time == pytest.approx(60.0)
        assert schema.snapshot_clean is True
        assert schema.wrap_atoms is True
        assert schema.save_stepsizes is True
        assert schema.write_traj_db is True
        assert schema.write_walkers_db is True

    def test_extra_field_still_rejected(self):
        from jaxrens.cli.schema.output import OutputSchema
        with pytest.raises(ValidationError):
            OutputSchema(bogus_output_field=True)

    def test_yaml_round_trip_extended_output(self):
        from jaxrens.cli.schema.output import OutputSchema
        schema = OutputSchema(
            format="none",
            wrap_atoms=True,
            snapshot_time=30.0,
        )
        dumped = schema.model_dump(mode="json")
        import yaml as _yaml
        reloaded = _yaml.safe_load(_yaml.safe_dump(dumped))
        schema2 = OutputSchema.model_validate(reloaded)
        assert schema2.wrap_atoms is True
        assert schema2.snapshot_time == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 34. Integration: full_config.yaml validates and resolves (schema level)
# (resolver and JIT tests moved to test_resolve.py::TestFullConfigResolver)
# ---------------------------------------------------------------------------

_FULL_CONFIG_YAML = _DATA / "full_config.yaml"


class TestFullConfigFixture:
    """Schema-level integration tests for full_config.yaml."""

    def test_full_config_validates(self):
        with open(_FULL_CONFIG_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        assert root.run.n_live == 20
        assert root.init.start_species == "1 2"
        assert root.cell.max_volume_per_atom == pytest.approx(500.0)
        assert root.output.traj_interval == 5

    def test_full_config_cli_validate(self, capsys):
        from jaxrens.cli.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_FULL_CONFIG_YAML)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out
