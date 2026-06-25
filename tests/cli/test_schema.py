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

from jaxrens.cli.cli import _apply_overrides, _parse_set_override
from jaxrens.cli.resolve import ResolvedConfig, resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.backend import (
    DoubleWellBackendSpec,
    GaussianMixtureBackendSpec,
    HarmonicBackendSpec,
    LJBackendSpec,
    MACEBackendSpec,
    NeuralILBackendSpec,
)
from jaxrens.cli.schema.moves import (
    AlchemicalMorphMoveSpec,
    AlchemicalShiftMoveSpec,
    GMCMoveSpec,
    HMCMoveSpec,
    RandomWalkMoveSpec,
    ShearMoveSpec,
    SingleAtomMoveSpec,
    SingleAtomSwapMoveSpec,
    SingleAtomSweepMoveSpec,
    StretchMoveSpec,
    VolumeMoveSpec,
)
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun

_DATA = Path(__file__).parent.parent / "data" / "cli"
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


class TestRootSpecValidate:
    def test_minimal_dict_validates(self):
        root = RootSpec.model_validate(_minimal_dict())
        assert root.run.n_live == 20
        assert len(root.moves) == 1
        assert root.moves[0].move_type == "random_walk"
        assert root.backend.backend_type == "harmonic"
        assert root.output.format == "none"

    def test_defaults_applied(self):
        root = RootSpec.model_validate(_minimal_dict())
        assert root.run.convergence_threshold == 0.1
        assert root.moves[0].weight == 1.0
        assert root.output.traj_interval == 1


# ---------------------------------------------------------------------------
# 2. extra="forbid" rejects unknown keys
# ---------------------------------------------------------------------------


class TestExtraForbid:
    """One parametrized test covers ``extra="forbid"`` on every spec class.

    Each row is ``(spec_callable, bogus_kwargs)``: invoking the callable
    with those kwargs MUST raise ``ValidationError`` because the model has
    ``model_config = ConfigDict(extra="forbid")``.
    """

    @pytest.mark.parametrize(
        "spec_factory,bogus",
        [
            # Root-level keys
            (
                lambda kw: RootSpec.model_validate({**_minimal_dict(), **kw}),
                {"not_a_key": 42},
            ),
            (
                lambda kw: RootSpec.model_validate(
                    {
                        **_minimal_dict(),
                        "run": {**_minimal_dict()["run"], **kw},
                    }
                ),
                {"typo_field": 1},
            ),
            (
                lambda kw: RootSpec.model_validate(
                    {
                        **_minimal_dict(),
                        "backend": {**_minimal_dict()["backend"], **kw},
                    }
                ),
                {"mystery": "x"},
            ),
            # Per-spec extras
            (
                lambda kw: RootSpec.model_validate(
                    {
                        **_minimal_dict(),
                        "moves": [{"type": "random_walk", **kw}],
                    }
                ),
                {"n_reflect": 5},
            ),
            (
                lambda kw: __import__("pydantic")
                .TypeAdapter(
                    __import__(
                        "jaxrens.cli.schema.termination",
                        fromlist=["TerminationSpec"],
                    ).TerminationSpec,
                )
                .validate_python(
                    {"type": "iteration", "max_iterations": 5, **kw}
                ),
                {"bogus": 1},
            ),
            (
                lambda kw: __import__(
                    "jaxrens.cli.schema.adaptation",
                    fromlist=["AdaptationPolicy"],
                ).AdaptationPolicy(min_rate=0.3, **kw),
                {"bogus": 1},
            ),
            (
                lambda kw: RootSpec.model_validate(
                    {**_minimal_dict(), "adaptation": kw}
                ),
                {"bogus_field": True},
            ),
            (
                lambda kw: __import__(
                    "jaxrens.cli.schema.ensemble",
                    fromlist=["NVTEnsembleSpec"],
                ).NVTEnsembleSpec(**kw),
                {"bogus": 1},
            ),
            (
                lambda kw: __import__(
                    "jaxrens.cli.schema.ensemble",
                    fromlist=["NPTEnsembleSpec"],
                ).NPTEnsembleSpec(pressure=0.01, **kw),
                {"unknown_key": "x"},
            ),
            (
                lambda kw: __import__(
                    "jaxrens.cli.schema.init",
                    fromlist=["InitSpec"],
                ).InitSpec(start_species="1 1", **kw),
                {"unknown_field": 42},
            ),
            (
                lambda kw: __import__(
                    "jaxrens.cli.schema.cell",
                    fromlist=["CellSpec"],
                ).CellSpec(max_volume_per_atom=100.0, **kw),
                {"mystery_field": True},
            ),
            (
                lambda kw: __import__(
                    "jaxrens.cli.schema.output",
                    fromlist=["OutputSpec"],
                ).OutputSpec(**kw),
                {"bogus_output_field": True},
            ),
        ],
        ids=[
            "root",
            "run",
            "backend",
            "move",
            "termination_iteration",
            "adaptation_policy",
            "adaptation",
            "nvt_ensemble",
            "npt_ensemble",
            "init",
            "cell",
            "output",
        ],
    )
    def test_extra_field_rejected(self, spec_factory, bogus):
        with pytest.raises(ValidationError):
            spec_factory(bogus)


# ---------------------------------------------------------------------------
# 3. moves: normalization — dict becomes single-element list
# ---------------------------------------------------------------------------


class TestMovesNormalization:
    def test_single_dict_normalized_to_list(self):
        d = _minimal_dict()
        d["moves"] = {"move_type": "galilean", "step_size": 0.1}
        root = RootSpec.model_validate(d)
        assert isinstance(root.moves, list)
        assert len(root.moves) == 1
        assert root.moves[0].move_type == "gmc"

    def test_list_of_two_moves(self):
        d = _minimal_dict()
        d["moves"] = [
            {"move_type": "random_walk", "step_size": 0.2},
            {"move_type": "galilean", "step_size": 0.05},
        ]
        root = RootSpec.model_validate(d)
        assert len(root.moves) == 2
        assert root.moves[1].move_type == "gmc"


# ---------------------------------------------------------------------------
# 4. --set override: nested scalar
# ---------------------------------------------------------------------------


class TestSetOverride:
    def test_nested_scalar_patch(self):
        raw = _minimal_dict()
        patched = _apply_overrides(raw, ["run.n_live=123"])
        root = RootSpec.model_validate(patched)
        assert root.run.n_live == 123

    def test_multiple_overrides_last_wins(self):
        raw = _minimal_dict()
        patched = _apply_overrides(raw, ["run.n_live=10", "run.n_live=99"])
        root = RootSpec.model_validate(patched)
        assert root.run.n_live == 99

    def test_bool_override(self):
        raw = _minimal_dict()
        patched = _apply_overrides(raw, ["backend.periodic=true"])
        root = RootSpec.model_validate(patched)
        assert root.backend.periodic is True


# ---------------------------------------------------------------------------
# 5. --set bracket indexing into moves list
# ---------------------------------------------------------------------------


class TestSetBracketIndex:
    def test_bracket_index_step_size(self):
        d = _minimal_dict()
        d["moves"] = [{"move_type": "random_walk", "step_size": 0.3}]
        patched = _apply_overrides(d, ["moves[0].step_size=0.25"])
        root = RootSpec.model_validate(patched)
        assert root.moves[0].step_size == pytest.approx(0.25)

    def test_bracket_index_move_type(self):
        d = _minimal_dict()
        d["moves"] = [{"move_type": "random_walk"}]
        patched = _apply_overrides(d, ["moves[0].move_type=galilean"])
        root = RootSpec.model_validate(patched)
        assert root.moves[0].move_type == "gmc"

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
            RootSpec.model_validate(d)

    def test_all_registry_types_accepted(self):
        import typing

        from jaxrens.cli.schema.moves import MoveType

        valid_types = list(typing.get_args(MoveType))
        assert len(valid_types) >= 10


# ---------------------------------------------------------------------------
# 8. Round-trip: YAML -> RootSpec -> model_dump() -> re-validate
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_model_dump_round_trips(self):
        root1 = RootSpec.model_validate(_minimal_dict())
        dumped = root1.model_dump()
        dumped["output"]["working_dir"] = str(dumped["output"]["working_dir"])
        root2 = RootSpec.model_validate(dumped)
        assert root1.run == root2.run
        assert root1.moves == root2.moves
        assert root1.backend == root2.backend
        assert root1.output == root2.output

    def test_yaml_round_trip(self):
        root1 = RootSpec.model_validate(_minimal_dict())
        dumped = root1.model_dump(mode="json")
        reloaded = yaml.safe_load(yaml.safe_dump(dumped))
        root2 = RootSpec.model_validate(reloaded)
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
        root = RootSpec.model_validate(raw)
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
            main(
                [
                    "validate",
                    "-c",
                    str(_MINIMAL_YAML),
                    "--set",
                    "run.n_live=77",
                ]
            )
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
        assert "RootSpec" in schema.get("title", "") or "properties" in schema


# ---------------------------------------------------------------------------
# 10. Discriminated union: per-move-type instantiation
# ---------------------------------------------------------------------------


class TestDiscriminatedUnion:
    """Each move type parses to the correct spec subclass via the union."""

    @pytest.mark.parametrize(
        "move_dict,expected_cls",
        [
            ({"type": "random_walk"}, RandomWalkMoveSpec),
            ({"type": "galilean"}, GMCMoveSpec),
            ({"type": "gmc"}, GMCMoveSpec),
            ({"type": "hmc"}, HMCMoveSpec),
            ({"type": "single_atom"}, SingleAtomMoveSpec),
            ({"type": "single_atom_sweep"}, SingleAtomSweepMoveSpec),
            ({"type": "single_atom_swap"}, SingleAtomSwapMoveSpec),
            ({"type": "volume"}, VolumeMoveSpec),
            ({"type": "shear"}, ShearMoveSpec),
            ({"type": "stretch"}, StretchMoveSpec),
            (
                {"type": "alchemical_morph", "n_species": 2},
                AlchemicalMorphMoveSpec,
            ),
            ({"type": "alchemical_shift"}, AlchemicalShiftMoveSpec),
        ],
    )
    def test_correct_subclass_instantiated(self, move_dict, expected_cls):
        d = _minimal_dict()
        d["moves"] = [move_dict]
        root = RootSpec.model_validate(d)
        assert isinstance(root.moves[0], expected_cls)

    def test_nonexistent_type_raises(self):
        d = _minimal_dict()
        d["moves"] = [{"type": "nonexistent_move"}]
        with pytest.raises(ValidationError, match="nonexistent_move"):
            RootSpec.model_validate(d)

    def test_move_type_property_matches_discriminator(self):
        d = _minimal_dict()
        d["moves"] = [{"type": "random_walk"}]
        root = RootSpec.model_validate(d)
        assert root.moves[0].move_type == "random_walk"


# ---------------------------------------------------------------------------
# 11. Mixed-move list: two different types
# ---------------------------------------------------------------------------


class TestMixedMoves:
    def test_mixed_move_list_parses(self):
        with open(_MIXED_MOVES_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        assert len(root.moves) == 2
        assert isinstance(root.moves[0], RandomWalkMoveSpec)
        assert isinstance(root.moves[1], GMCMoveSpec)

    def test_mixed_moves_resolve_to_distinct_descriptors(self):
        with open(_MIXED_MOVES_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        resolved = resolve(root)
        assert len(resolved.move_descriptors) == 2
        d0, d1 = resolved.move_descriptors
        import jaxrens.sampling.moves.galilean as gal_mod
        import jaxrens.sampling.moves.random_walk as rw_mod

        assert d0.build_kernel is rw_mod.build_kernel
        assert d1.build_kernel is gal_mod.build_kernel

    def test_same_type_different_name_distinguishable(self):
        d = _minimal_dict()
        d["moves"] = [
            {"type": "random_walk", "step_size": 0.1, "name": "rw_slow"},
            {"type": "random_walk", "step_size": 0.5, "name": "rw_fast"},
        ]
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        assert resolved.move_descriptors[0].name == "rw_slow"
        assert resolved.move_descriptors[1].name == "rw_fast"
        assert (
            resolved.move_descriptors[0].step_size
            != resolved.move_descriptors[1].step_size
        )


# ---------------------------------------------------------------------------
# 16. BackendSpec discriminated union: per-type instantiation
# ---------------------------------------------------------------------------


class TestBackendDiscriminatedUnion:
    """Each backend type parses to the correct spec subclass via the union."""

    @pytest.mark.parametrize(
        "backend_dict,expected_cls",
        [
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
        ],
    )
    def test_correct_subclass_instantiated(self, backend_dict, expected_cls):
        d = _minimal_dict()
        d["backend"] = backend_dict
        root = RootSpec.model_validate(d)
        assert isinstance(root.backend, expected_cls)

    def test_unknown_type_raises(self):
        d = _minimal_dict()
        d["backend"] = {"type": "nonexistent_backend"}
        with pytest.raises(ValidationError, match="nonexistent_backend"):
            RootSpec.model_validate(d)

    def test_backend_type_property_matches_discriminator(self):
        d = _minimal_dict()
        d["backend"] = {"type": "harmonic"}
        root = RootSpec.model_validate(d)
        assert root.backend.backend_type == "harmonic"

    def test_legacy_backend_type_key_accepted(self):
        d = _minimal_dict()
        d["backend"] = {"backend_type": "harmonic"}
        root = RootSpec.model_validate(d)
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
            RootSpec.model_validate(d)

    def test_lj_rejects_unknown_field(self):
        d = _minimal_dict()
        d["backend"] = {"type": "lj", "mystery": "x"}
        with pytest.raises(ValidationError):
            RootSpec.model_validate(d)

    def test_lj_rejects_harmonic_only_field(self):
        d = _minimal_dict()
        d["backend"] = {"type": "lj", "k": 2.0}
        with pytest.raises(ValidationError):
            RootSpec.model_validate(d)

    def test_harmonic_rejects_lj_only_field(self):
        d = _minimal_dict()
        d["backend"] = {"type": "harmonic", "epsilon": 1.0}
        with pytest.raises(ValidationError):
            RootSpec.model_validate(d)


# ---------------------------------------------------------------------------
# 20. Fixture YAML: lj_backend.yaml validates and round-trips
# ---------------------------------------------------------------------------


class TestLJBackendFixture:
    def test_lj_backend_yaml_validates(self):
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        assert isinstance(root.backend, LJBackendSpec)
        assert root.backend.cutoff == pytest.approx(3.0)

    def test_lj_backend_yaml_to_backend_config(self):
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        cfg = root.backend.to_backend_config()
        assert cfg.backend_type == "lj"
        assert cfg.cutoff == pytest.approx(3.0)

    def test_lj_backend_yaml_builds_backend(self):
        from jaxrens.backends.lj import LJBackend

        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        backend = root.backend.build_backend()
        assert isinstance(backend, LJBackend)


# ---------------------------------------------------------------------------
# 23. TerminationSpec discriminated union — schema-level tests
# (to_criterion tests moved to test_resolve.py::TestToCriterion)
# ---------------------------------------------------------------------------


class TestTerminationDiscriminatedUnion:
    """Each termination type parses to the correct spec subclass."""

    @pytest.mark.parametrize(
        "term_dict,expected_cls_name",
        [
            (
                {"type": "iteration", "max_iterations": 100},
                "IterationTerminationSpec",
            ),
            ({"type": "prior_mass"}, "PriorMassTerminationSpec"),
            (
                {"type": "temperature", "target_temp": 300.0},
                "TemperatureTerminationSpec",
            ),
            ({"type": "energy", "min_energy": -10.0}, "EnergyTerminationSpec"),
        ],
    )
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

    def test_single_dict_termination_normalized_to_list(self):
        d = _minimal_dict()
        d["termination"] = {"type": "iteration", "max_iterations": 7}
        root = RootSpec.model_validate(d)
        assert len(root.termination) == 1
        assert root.termination[0].type == "iteration"


# ---------------------------------------------------------------------------
# 25. AdaptationSpec — schema-level tests
# (resolve_for tests moved to test_resolve.py::TestAdaptationResolve)
# ---------------------------------------------------------------------------


class TestAdaptationSpec:
    def test_default_adaptation_config(self):
        from jaxrens.cli.schema.adaptation import AdaptationSpec

        cfg = AdaptationSpec()
        assert cfg.full_auto is False
        assert cfg.adjust_interval == 0
        assert cfg.per_move == {}
        assert cfg.defaults.min_rate is None

    def test_full_auto_steps_alias_emits_deprecation(self):
        """Legacy ``full_auto_steps`` key is coerced to ``adjust_interval``."""
        from jaxrens.cli.schema.adaptation import AdaptationSpec

        with pytest.warns(DeprecationWarning, match="full_auto_steps"):
            cfg = AdaptationSpec.model_validate({"full_auto_steps": 20})
        assert cfg.adjust_interval == 20

    def test_full_auto_steps_alias_in_root_yaml(self):
        """Legacy key still works when nested in a full RootSpec."""
        d = _minimal_dict()
        d["adaptation"] = {"full_auto": True, "full_auto_steps": 50}
        with pytest.warns(DeprecationWarning, match="full_auto_steps"):
            root = RootSpec.model_validate(d)
        assert root.adaptation.adjust_interval == 50
        assert root.adaptation.full_auto is True

    def test_default_adaptation_config_round_trip(self):
        d = _minimal_dict()
        root = RootSpec.model_validate(d)
        assert root.adaptation.full_auto is False
        assert root.adaptation.defaults.min_rate is None


# ---------------------------------------------------------------------------
# 25b. RootSpec.interval_units — schema-level tests
# (resolver scaling lives in tests/test_interval_units.py)
# ---------------------------------------------------------------------------


class TestIntervalUnitsField:
    def test_default_is_absolute(self):
        root = RootSpec.model_validate(_minimal_dict())
        assert root.interval_units == "absolute"

    def test_per_walker_accepted(self):
        d = _minimal_dict()
        d["interval_units"] = "per_walker"
        root = RootSpec.model_validate(d)
        assert root.interval_units == "per_walker"

    def test_invalid_value_rejected(self):
        d = _minimal_dict()
        d["interval_units"] = "sweep"
        with pytest.raises(ValidationError):
            RootSpec.model_validate(d)


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

    def test_nvt_in_root_config_default(self):
        from jaxrens.cli.schema.ensemble import NVTEnsembleSpec

        root = RootSpec.model_validate(_minimal_dict())
        assert isinstance(root.ensemble, NVTEnsembleSpec)

    def test_npt_in_root_config(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec

        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": 0.01}
        root = RootSpec.model_validate(d)
        assert isinstance(root.ensemble, NPTEnsembleSpec)
        assert root.ensemble.pressure == pytest.approx(0.01)

    def test_run_pressure_is_rejected(self):
        """``run.pressure`` was removed; it's now an unknown field (use
        ``ensemble: {type: npt, ...}`` instead)."""
        d = _minimal_dict()
        d["run"]["pressure"] = 0.05
        with pytest.raises(ValidationError):
            RootSpec.model_validate(d)


# ---------------------------------------------------------------------------
# 27. Topology resolution — schema-level tests
# (cohort path removed; resolver now always produces one ResolvedConfig
# whose ``batcher`` is SingleRun for n_total=1 and PmapVmapRuns otherwise.)
# ---------------------------------------------------------------------------


class TestTopologyResolution:
    def test_nvt_scalar_resolves_to_single_run(self):
        root = RootSpec.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert isinstance(resolved.batcher, SingleRun)
        assert len(resolved.ensemble_params_per_run) == 1

    def test_npt_scalar_resolves_to_single_run(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": 0.01}
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        assert isinstance(resolved.batcher, SingleRun)
        assert len(resolved.ensemble_params_per_run) == 1
        assert resolved.ensemble_params_per_run[0][
            "pressure"
        ] == pytest.approx(0.01)

    def test_npt_three_pressures_resolves_to_multi_replica(self):
        """A list-valued pressure routes to PmapVmapRuns with per-replica params."""
        d = _minimal_dict()
        d["run"]["seed"] = 10
        d["ensemble"] = {"type": "npt", "pressure": [0.01, 0.02, 0.03]}
        root = RootSpec.model_validate(d)
        resolved = resolve(root)

        assert isinstance(resolved.batcher, PmapVmapRuns)
        n_total = resolved.batcher.n_gpu * resolved.batcher.n_per_gpu
        assert n_total == 3
        assert len(resolved.ensemble_params_per_run) == 3

        # Per-replica pressures in the expected order.
        assert resolved.ensemble_params_per_run[0][
            "pressure"
        ] == pytest.approx(0.01)
        assert resolved.ensemble_params_per_run[1][
            "pressure"
        ] == pytest.approx(0.02)
        assert resolved.ensemble_params_per_run[2][
            "pressure"
        ] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# 28. Topology resolution: CLI validate prints the correct topology line
# ---------------------------------------------------------------------------


class TestValidateTopologyLine:
    def test_validate_reports_single_run_topology(self, capsys):
        from jaxrens.cli.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "minimal.yaml")])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "SingleRun" in captured.out

    def test_validate_reports_multi_replica_topology(self, capsys):
        from jaxrens.cli.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_DATA / "npt_sweep.yaml")])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # A 2-pressure list routes to the multi-replica dispatch.
        assert "n_gpu=" in captured.out
        assert "2 replica" in captured.out


# ---------------------------------------------------------------------------
# 30. InitSpec — schema validation
# ---------------------------------------------------------------------------


class TestInitSpec:
    """Tests for the InitSpec pydantic schema (Part A)."""

    def test_start_species_single_element(self):
        """New format: Z N -> {Z: N}."""
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(start_species="18 8")
        assert cfg.start_species == "18 8"
        counts = cfg.parsed_species()
        assert counts == {18: 8}

    def test_start_species_multi_element(self):
        """New format: Z1 N1, Z2 N2 -> {Z1: N1, Z2: N2}."""
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(start_species="1 6, 3 6")
        counts = cfg.parsed_species()
        assert counts == {1: 6, 3: 6}
        total = sum(counts.values())
        assert total == 12

    def test_multi_composition_raises(self):
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(start_species="1 3: 0 16")
        with pytest.raises(ValueError, match="Multi-composition"):
            cfg.parsed_species()

    def test_mass_token_raises_not_implemented(self):
        """Three-token group (Z N mass) raises NotImplementedError."""
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(start_species="1 6 1.008")
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
        from jaxrens.cli.schema.init import InitSpec

        with pytest.raises(ValidationError, match="none were set"):
            InitSpec()

    def test_two_sources_raises(self):
        from jaxrens.cli.schema.init import InitSpec

        with pytest.raises(ValidationError, match="got:"):
            InitSpec(
                start_species="1 1",
                start_config_file=Path("/tmp/fake.xyz"),
            )

    def test_start_config_file_accepted(self):
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(start_config_file=Path("/tmp/atoms.xyz"))
        assert cfg.start_config_file == Path("/tmp/atoms.xyz")
        assert cfg.start_species is None

    def test_restart_file_accepted(self):
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(restart_file=Path("/tmp/checkpoint.npz"))
        assert cfg.restart_file == Path("/tmp/checkpoint.npz")

    def test_defaults_are_sane(self):
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(start_species="1 1")
        assert cfg.start_energy_ceiling_per_atom == pytest.approx(1e9)
        assert cfg.random_initialise_pos is True
        assert cfg.random_initialise_cell is True
        assert cfg.pos_randomization_mode == "grid"
        assert cfg.grid_distance == pytest.approx(1.5)
        assert cfg.init_distance_criterion == pytest.approx(1.0)
        assert cfg.random_init_max_n_tries == 100
        assert cfg.pos_autoscale_cells is False

    def test_initial_walk_defaults(self):
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(start_species="1 1")
        assert cfg.initial_walk.n_walks == 0
        assert cfg.initial_walk.walklength == 100

    def test_yaml_round_trip_init(self):
        from jaxrens.cli.schema.init import InitSpec

        cfg = InitSpec(
            start_species="14 8",
            pos_randomization_mode="uniform",
            grid_distance=2.0,
        )
        dumped = cfg.model_dump(mode="json")
        import yaml as _yaml

        reloaded_raw = _yaml.safe_load(_yaml.safe_dump(dumped))
        cfg2 = InitSpec.model_validate(reloaded_raw)
        assert cfg2.start_species == "14 8"
        assert cfg2.pos_randomization_mode == "uniform"
        assert cfg2.grid_distance == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 32. CellSpec — schema validation
# (resolver tests moved to test_resolve.py::TestCellResolve)
# ---------------------------------------------------------------------------


class TestCellSpec:
    """Tests for the CellSpec pydantic schema (Part B)."""

    def test_defaults(self):
        from jaxrens.cli.schema.cell import CellSpec

        cfg = CellSpec()
        assert cfg.max_volume_per_atom == pytest.approx(1e4)
        assert cfg.min_volume_per_atom == pytest.approx(1.0)
        assert cfg.min_aspect_ratio == pytest.approx(0.8)
        assert cfg.flat_V_prior is False

    def test_custom_values_accepted(self):
        from jaxrens.cli.schema.cell import CellSpec

        cfg = CellSpec(
            max_volume_per_atom=500.0,
            min_volume_per_atom=2.0,
            min_aspect_ratio=0.75,
            flat_V_prior=True,
        )
        assert cfg.max_volume_per_atom == pytest.approx(500.0)
        assert cfg.flat_V_prior is True

    def test_yaml_round_trip_cell(self):
        from jaxrens.cli.schema.cell import CellSpec

        cfg = CellSpec(max_volume_per_atom=200.0, min_aspect_ratio=0.6)
        dumped = cfg.model_dump(mode="json")
        import yaml as _yaml

        reloaded_raw = _yaml.safe_load(_yaml.safe_dump(dumped))
        cfg2 = CellSpec.model_validate(reloaded_raw)
        assert cfg2.max_volume_per_atom == pytest.approx(200.0)
        assert cfg2.min_aspect_ratio == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 33. Extended OutputConfig — schema-level tests
# (resolver warning tests moved to test_resolve.py::TestExtendedOutputResolve)
# ---------------------------------------------------------------------------


class TestExtendedOutputSpec:
    """Tests for the extended OutputSpec optional fields (Part C)."""

    def test_optional_fields_have_correct_defaults(self):
        from jaxrens.cli.schema.output import OutputSpec

        schema = OutputSpec()
        assert schema.snapshot_clean is True
        assert schema.wrap_atoms is True
        # save_acc_rates was promoted out of deferred into a real field.
        assert schema.save_acc_rates is False
        assert schema.acc_rates_interval == 1

    def test_optional_fields_accept_non_defaults(self):
        from jaxrens.cli.schema.output import OutputSpec

        schema = OutputSpec(
            format="none",
            snapshot_clean=True,
            wrap_atoms=True,
        )
        assert schema.snapshot_clean is True
        assert schema.wrap_atoms is True

    def test_removed_fields_rejected(self):
        """``snapshot_time`` / ``write_traj_db`` / ``write_walkers_db`` were
        removed; ``extra='forbid'`` must now reject them."""
        from pydantic import ValidationError

        from jaxrens.cli.schema.output import OutputSpec

        for key in ("snapshot_time", "write_traj_db", "write_walkers_db"):
            with pytest.raises(ValidationError):
                OutputSpec(**{key: 1})

    def test_save_acc_rates_accept_non_defaults(self):
        from jaxrens.cli.schema.output import OutputSpec

        schema = OutputSpec(save_acc_rates=True, acc_rates_interval=10)
        assert schema.save_acc_rates is True
        assert schema.acc_rates_interval == 10

    def test_yaml_round_trip_extended_output(self):
        from jaxrens.cli.schema.output import OutputSpec

        schema = OutputSpec(
            format="none",
            wrap_atoms=True,
            snapshot_clean=True,
        )
        dumped = schema.model_dump(mode="json")
        import yaml as _yaml

        reloaded = _yaml.safe_load(_yaml.safe_dump(dumped))
        schema2 = OutputSpec.model_validate(reloaded)
        assert schema2.wrap_atoms is True
        assert schema2.snapshot_clean is True


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
        root = RootSpec.model_validate(raw)
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
