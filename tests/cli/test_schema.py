"""Tests for the pydantic config schema in ``jaxrens.cli.schema``.

They cover parsing and validation of each section (moves, backend,
termination, adaptation, ensemble, init, cell, output), the ``--set``
override helpers, the committed fixture YAMLs, and checks that keep the
schema and its documentation in step: every field has a description, and
the YAML examples in the docs, docstrings and tutorials parse.
"""

from __future__ import annotations

import importlib
import inspect
import textwrap
import typing
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

import jaxrens.sampling.moves.galilean as gal_mod
import jaxrens.sampling.moves.random_walk as rw_mod
from jaxrens.backends.lj import LJBackend
from jaxrens.cli.cli import _apply_overrides, _parse_set_override, main
from jaxrens.cli.resolve import resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.adaptation import AdaptationPolicy, AdaptationSpec
from jaxrens.cli.schema.backend import (
    DoubleWellBackendSpec,
    GaussianMixtureBackendSpec,
    HarmonicBackendSpec,
    LJBackendSpec,
    MACEBackendSpec,
    NeuralILBackendSpec,
)
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.ensemble import (
    EnsembleSpec,
    NPTEnsembleSpec,
    NVTEnsembleSpec,
)
from jaxrens.cli.schema.init import InitSpec, _parse_species_string
from jaxrens.cli.schema.moves import (
    AlchemicalMorphMoveSpec,
    GMCMoveSpec,
    HMCMoveSpec,
    RandomWalkMoveSpec,
    ShearMoveSpec,
    SingleAtomMoveSpec,
    SingleAtomSweepMoveSpec,
    SpeciesSwapMoveSpec,
    StretchMoveSpec,
    VolumeMoveSpec,
)
from jaxrens.cli.schema.output import OutputSpec
from jaxrens.cli.schema.run import RunSpec
from jaxrens.cli.schema.termination import TerminationSpec
from jaxrens.unvalidated import UnvalidatedFeatureWarning, reset

_DATA = Path(__file__).parent.parent / "_assets" / "data" / "cli"
_MINIMAL_YAML = _DATA / "minimal.yaml"
_MIXED_MOVES_YAML = _DATA / "mixed_moves.yaml"
_LJ_BACKEND_YAML = _DATA / "lj_backend.yaml"


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


class TestRootSpecValidate:
    """Check the minimal config and the defaults it fills in."""

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
            lambda kw: TypeAdapter(TerminationSpec).validate_python(
                {"type": "iteration", "max_iterations": 5, **kw}
            ),
            {"bogus": 1},
        ),
        (
            lambda kw: AdaptationPolicy(min_rate=0.3, **kw),
            {"bogus": 1},
        ),
        (
            lambda kw: RootSpec.model_validate(
                {**_minimal_dict(), "adaptation": kw}
            ),
            {"bogus_field": True},
        ),
        (
            lambda kw: NVTEnsembleSpec(**kw),
            {"bogus": 1},
        ),
        (
            lambda kw: NPTEnsembleSpec(pressure=0.01, **kw),
            {"unknown_key": "x"},
        ),
        (
            lambda kw: InitSpec(start_species="1 1", **kw),
            {"unknown_field": 42},
        ),
        (
            lambda kw: CellSpec(max_volume_per_atom=100.0, **kw),
            {"mystery_field": True},
        ),
        (
            lambda kw: OutputSpec(**kw),
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
def test_extra_field_rejected(spec_factory, bogus):
    """Every spec class has ``extra="forbid"``, so an unknown key raises."""
    with pytest.raises(ValidationError):
        spec_factory(bogus)


class TestMovesNormalization:
    """Check that ``moves`` accepts a single dict or a list."""

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


class TestSetOverride:
    """Check ``--set`` overrides of nested scalars."""

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


class TestSetBracketIndex:
    """Check ``--set`` overrides that index into the moves list."""

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


class TestRoundTrip:
    """Check that a dumped config validates back to the same values."""

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


class TestFixtureYAML:
    """Check the ``minimal.yaml`` fixture, also through ``validate``."""

    def test_minimal_yaml_validates(self):
        with open(_MINIMAL_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        assert root.run.n_live == 20

    def test_cli_validate_subcommand(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_MINIMAL_YAML)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out
        assert "random_walk" in captured.out

    def test_cli_validate_with_set_override(self, capsys):
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


class TestDiscriminatedUnion:
    """Check that each move type parses to its own spec class."""

    @pytest.mark.parametrize(
        "move_dict,expected_cls",
        [
            ({"type": "random_walk"}, RandomWalkMoveSpec),
            ({"type": "galilean"}, GMCMoveSpec),
            ({"type": "gmc"}, GMCMoveSpec),
            ({"type": "hmc"}, HMCMoveSpec),
            ({"type": "single_atom"}, SingleAtomMoveSpec),
            ({"type": "single_atom_sweep"}, SingleAtomSweepMoveSpec),
            ({"type": "species_swap"}, SpeciesSwapMoveSpec),
            (
                {"type": "species_swap", "species": ["Ge", "Si"]},
                SpeciesSwapMoveSpec,
            ),
            ({"type": "volume"}, VolumeMoveSpec),
            ({"type": "shear"}, ShearMoveSpec),
            ({"type": "stretch"}, StretchMoveSpec),
            (
                {"type": "alchemical_morph", "n_species": 2},
                AlchemicalMorphMoveSpec,
            ),
        ],
    )
    def test_correct_subclass_instantiated(self, move_dict, expected_cls):
        d = _minimal_dict()
        d["moves"] = [move_dict]
        root = RootSpec.model_validate(d)
        assert isinstance(root.moves[0], expected_cls)

    @pytest.mark.parametrize("key", ["type", "move_type"])
    def test_nonexistent_type_raises(self, key):
        """Both the ``type`` key and the legacy ``move_type`` key."""
        d = _minimal_dict()
        d["moves"] = [{key: "nonexistent_move"}]
        with pytest.raises(ValidationError, match="nonexistent_move"):
            RootSpec.model_validate(d)


class TestMixedMoves:
    """Check a list of two move types, parsed and resolved."""

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


class TestSpeciesSwapSpec:
    """Check that SpeciesSwapMoveSpec turns element symbols into type codes."""

    SYMBOLS = {0: "Ge", 1: "Si", 2: "Sn"}

    def test_unscoped_descriptor(self):
        desc = SpeciesSwapMoveSpec().to_descriptor(
            n_atoms=6, symbol_map=self.SYMBOLS
        )
        assert desc.name == "species_swap"
        assert desc.kernel_kwargs == {"n_species": 3}
        assert desc.mutates == frozenset({"types"})

    def test_scoped_descriptor_resolves_symbols_to_codes(self):
        desc = SpeciesSwapMoveSpec(species=("Si", "Sn")).to_descriptor(
            n_atoms=6, symbol_map=self.SYMBOLS
        )
        assert desc.name == "species_swap_Si_Sn"
        assert desc.kernel_kwargs == {"n_species": 3, "species": (1, 2)}

    def test_bare_symbol_shorthand_rejected_at_parse_time(self):
        with pytest.raises(ValidationError, match="at least two elements"):
            SpeciesSwapMoveSpec(species="Ge")

    def test_single_distinct_element_rejected_at_parse_time(self):
        with pytest.raises(ValidationError, match="distinct element"):
            SpeciesSwapMoveSpec(species=("Ge", "Ge"))

    def test_absent_element_rejected(self):
        with pytest.raises(ValueError, match="not present in the system"):
            SpeciesSwapMoveSpec(species=("Ge", "Pu")).to_descriptor(
                n_atoms=6, symbol_map=self.SYMBOLS
            )

    def test_missing_symbol_map_rejected(self):
        with pytest.raises(ValueError, match="requires symbol_map"):
            SpeciesSwapMoveSpec().to_descriptor(n_atoms=6, symbol_map=None)


class TestGMCSpeciesSchema:
    """Check how GMCMoveSpec resolves species, names and direction fields."""

    SYMBOL_MAP = {0: "Ge", 1: "Si"}

    def test_bare_string_species_is_wrapped(self):
        assert GMCMoveSpec(species="Ge").species == ("Ge",)

    def test_symbols_resolve_to_type_codes(self):
        spec = GMCMoveSpec(species=["Si"])
        kwargs = spec._kernel_kwargs(symbol_map=self.SYMBOL_MAP)
        assert kwargs["species"] == (1,)
        assert kwargs["direction_field"] == "direction_gmc_Si"
        assert kwargs["n_reflect"] == 5

    def test_auto_name_and_explicit_name(self):
        assert GMCMoveSpec(species=["Ge"])._effective_name() == "gmc_Ge"
        assert (
            GMCMoveSpec(species=["Ge", "Si"])._effective_name() == "gmc_Ge_Si"
        )
        assert (
            GMCMoveSpec(species=["Ge"], name="fast")._effective_name()
            == "fast"
        )
        # An explicit name still gets its own direction field when scoped.
        assert (
            GMCMoveSpec(species=["Ge"], name="fast")._direction_field()
            == "direction_fast"
        )

    def test_unscoped_spec_is_unchanged(self):
        spec = GMCMoveSpec()
        assert spec._effective_name() == "gmc"
        assert spec._direction_field() == "direction"
        # The kernel defaults supply everything except n_reflect.
        assert spec._kernel_kwargs(symbol_map=self.SYMBOL_MAP) == {
            "n_reflect": 5
        }
        assert "direction" in spec._extra_state_fields()

    def test_refresh_direction_reaches_kernel_kwargs(self):
        spec = GMCMoveSpec(refresh_direction=True)
        assert spec._kernel_kwargs(symbol_map=self.SYMBOL_MAP) == {
            "n_reflect": 5,
            "refresh_direction": True,
        }

    def test_unknown_species_raises(self):
        spec = GMCMoveSpec(species=["Xe"])
        with pytest.raises(ValueError, match="not present in the system"):
            spec.to_descriptor(symbol_map=self.SYMBOL_MAP)

    def test_missing_symbol_map_raises(self):
        spec = GMCMoveSpec(species=["Ge"])
        with pytest.raises(ValueError, match="requires symbol_map"):
            spec.to_descriptor()

    def test_unscoped_spec_needs_no_symbol_map(self):
        """An unscoped spec resolves without a symbol_map."""
        desc = GMCMoveSpec(n_reflect=4).to_descriptor()
        assert desc.name == "gmc"
        assert desc.kernel_kwargs == {"n_reflect": 4}
        assert "direction" in desc.extra_state_fields


class TestBackendDiscriminatedUnion:
    """Check that each backend type parses to its own spec class."""

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

    def test_legacy_backend_type_key_accepted(self):
        d = _minimal_dict()
        d["backend"] = {"backend_type": "harmonic"}
        root = RootSpec.model_validate(d)
        assert isinstance(root.backend, HarmonicBackendSpec)
        assert root.backend.backend_type == "harmonic"


class TestBackendExtraForbid:
    """Check that each backend rejects fields of other backends."""

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


class TestLJBackendFixture:
    """Check that ``lj_backend.yaml`` parses and builds an LJ backend."""

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
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        backend = root.backend.build_backend()
        assert isinstance(backend, LJBackend)


class TestTerminationDiscriminatedUnion:
    """Check that each termination type parses to its own spec class."""

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
        ta = TypeAdapter(TerminationSpec)
        spec = ta.validate_python(term_dict)
        assert type(spec).__name__ == expected_cls_name

    def test_unknown_type_raises(self):
        ta = TypeAdapter(TerminationSpec)
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "nonexistent_criterion"})

    def test_single_dict_termination_normalized_to_list(self):
        d = _minimal_dict()
        d["termination"] = {"type": "iteration", "max_iterations": 7}
        root = RootSpec.model_validate(d)
        assert len(root.termination) == 1
        assert root.termination[0].type == "iteration"


class TestAdaptationSpec:
    """Check the adaptation defaults and the legacy ``full_auto_steps`` key."""

    def test_default_adaptation_config(self):
        cfg = AdaptationSpec()
        assert cfg.full_auto is True
        assert cfg.adjust_interval == 100
        assert cfg.per_move == {}
        assert cfg.defaults.min_rate is None

    def test_full_auto_steps_alias_emits_deprecation(self):
        """Legacy ``full_auto_steps`` key is coerced to ``adjust_interval``."""
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
        assert root.adaptation.full_auto is True
        assert root.adaptation.defaults.min_rate is None


class TestIntervalUnitsField:
    """Check the values ``interval_units`` accepts."""

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


class TestNCullWarningAndValidation:
    """Check that ``run.n_cull > 1`` raises the unvalidated-feature warning
    and that cull widths the population cannot support are rejected.

    The warning mechanism itself is tested in tests/test_unvalidated.py.
    """

    @pytest.fixture(autouse=True)
    def _fresh_markers(self):
        """Warnings fire once per process; reset them around each test."""
        reset()
        yield
        reset()

    def test_default_n_cull_is_silent(self, recwarn):
        RootSpec.model_validate(_minimal_dict())
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_n_cull_above_one_warns(self):
        d = _minimal_dict()
        d["run"]["n_cull"] = 2
        with pytest.warns(UnvalidatedFeatureWarning, match=r"run\.n_cull > 1"):
            RootSpec.model_validate(d)

    @pytest.mark.parametrize("n_cull", [0, -1])
    def test_n_cull_below_one_rejected(self, n_cull):
        d = _minimal_dict()
        d["run"]["n_cull"] = n_cull
        with pytest.raises(ValidationError):
            RootSpec.model_validate(d)

    def test_n_cull_too_wide_for_population_rejected(self):
        d = _minimal_dict()
        n_live = RootSpec.model_validate(_minimal_dict()).run.n_live
        d["run"]["n_cull"] = n_live // 2 + 1
        with pytest.raises(ValidationError, match="2 \\* n_cull"):
            RootSpec.model_validate(d)


class TestEnsembleSpec:
    """Check the NVT and NPT ensemble specs and the default ensemble."""

    def test_nvt_minimal_validates(self):
        spec = NVTEnsembleSpec()
        assert spec.type == "nvt"

    def test_npt_scalar_pressure_validates(self):
        spec = NPTEnsembleSpec(pressure=0.01)
        assert spec.pressure == pytest.approx(0.01)
        assert spec.pressure_units == "eva3"

    def test_npt_list_pressure_validates(self):
        spec = NPTEnsembleSpec(pressure=[0.01, 0.02, 0.03])
        assert len(spec.pressure) == 3

    def test_ensemble_discriminator_mismatch_raises(self):
        ta = TypeAdapter(EnsembleSpec)
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "nonexistent_ensemble"})

    def test_nvt_in_root_config_default(self):
        root = RootSpec.model_validate(_minimal_dict())
        assert isinstance(root.ensemble, NVTEnsembleSpec)

    def test_npt_in_root_config(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": 0.01}
        root = RootSpec.model_validate(d)
        assert isinstance(root.ensemble, NPTEnsembleSpec)
        assert root.ensemble.pressure == pytest.approx(0.01)

    def test_run_pressure_is_rejected(self):
        """Pressure belongs in ``ensemble``; ``run.pressure`` is unknown."""
        d = _minimal_dict()
        d["run"]["pressure"] = 0.05
        with pytest.raises(ValidationError):
            RootSpec.model_validate(d)


class TestInitSpec:
    """Check ``start_species`` parsing, the one-source rule and the init
    defaults.
    """

    def test_start_species_single_element(self):
        """``"Z N"`` -> ``{Z: N}``."""
        cfg = InitSpec(start_species="18 8")
        assert cfg.start_species == "18 8"
        counts = cfg.parsed_species()
        assert counts == {18: 8}

    def test_start_species_multi_element(self):
        """``"Z1 N1, Z2 N2"`` -> ``{Z1: N1, Z2: N2}``."""
        cfg = InitSpec(start_species="1 6, 3 6")
        counts = cfg.parsed_species()
        assert counts == {1: 6, 3: 6}
        total = sum(counts.values())
        assert total == 12

    def test_multi_composition_raises(self):
        cfg = InitSpec(start_species="1 3: 0 16")
        with pytest.raises(ValueError, match="Multi-composition"):
            cfg.parsed_species()

    def test_mass_token_raises_not_implemented(self):
        """Three-token group (Z N mass) raises NotImplementedError."""
        cfg = InitSpec(start_species="1 6 1.008")
        with pytest.raises(NotImplementedError, match="masses"):
            cfg.parsed_species()

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_species_string("   ")

    def test_non_integer_token_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            _parse_species_string("H 6")

    def test_odd_number_of_tokens_raises(self):
        with pytest.raises(ValueError, match="even number"):
            _parse_species_string("1 6, 3")

    def test_single_token_raises(self):
        with pytest.raises(ValueError, match="even number"):
            _parse_species_string("1")

    def test_zero_sources_raises(self):
        with pytest.raises(ValidationError, match="none were set"):
            InitSpec()

    def test_two_sources_raises(self):
        with pytest.raises(ValidationError, match="got:"):
            InitSpec(
                start_species="1 1",
                start_config_file=Path("/tmp/fake.xyz"),
            )

    def test_start_config_file_accepted(self):
        cfg = InitSpec(start_config_file=Path("/tmp/atoms.xyz"))
        assert cfg.start_config_file == Path("/tmp/atoms.xyz")
        assert cfg.start_species is None

    def test_restart_file_accepted(self):
        cfg = InitSpec(restart_file=Path("/tmp/checkpoint.npz"))
        assert cfg.restart_file == Path("/tmp/checkpoint.npz")

    def test_defaults_are_sane(self):
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
        cfg = InitSpec(start_species="1 1")
        assert cfg.initial_walk.n_walks == 0
        assert cfg.initial_walk.walklength == 100

    def test_yaml_round_trip_init(self):
        cfg = InitSpec(
            start_species="14 8",
            pos_randomization_mode="uniform",
            grid_distance=2.0,
        )
        dumped = cfg.model_dump(mode="json")
        reloaded_raw = yaml.safe_load(yaml.safe_dump(dumped))
        cfg2 = InitSpec.model_validate(reloaded_raw)
        assert cfg2.start_species == "14 8"
        assert cfg2.pos_randomization_mode == "uniform"
        assert cfg2.grid_distance == pytest.approx(2.0)


class TestCellSpec:
    """Check the cell defaults and the initial minimum-volume floor."""

    def test_defaults(self):
        cfg = CellSpec()
        assert cfg.max_volume_per_atom == pytest.approx(1e4)
        assert cfg.min_volume_per_atom == pytest.approx(1.0)
        assert cfg.min_aspect_ratio == pytest.approx(0.8)
        assert cfg.flat_V_prior is False
        # Unset initial floor falls back to the sampling constraint.
        assert cfg.initial_min_volume_per_atom is None
        assert cfg.effective_initial_min_volume_per_atom == pytest.approx(1.0)

    def test_initial_min_volume_decoupled_from_constraint(self):
        cfg = CellSpec(
            min_volume_per_atom=1.0, initial_min_volume_per_atom=20.0
        )
        # The init floor is independent of the (smaller) sampling constraint.
        assert cfg.min_volume_per_atom == pytest.approx(1.0)
        assert cfg.effective_initial_min_volume_per_atom == pytest.approx(20.0)

    def test_initial_min_volume_rejects_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            CellSpec(initial_min_volume_per_atom=-1.0)

    def test_initial_min_volume_rejects_above_max(self):
        with pytest.raises(ValueError, match="exceeds"):
            CellSpec(
                max_volume_per_atom=50.0, initial_min_volume_per_atom=60.0
            )

    def test_yaml_round_trip_cell(self):
        cfg = CellSpec(max_volume_per_atom=200.0, min_aspect_ratio=0.6)
        dumped = cfg.model_dump(mode="json")
        reloaded_raw = yaml.safe_load(yaml.safe_dump(dumped))
        cfg2 = CellSpec.model_validate(reloaded_raw)
        assert cfg2.max_volume_per_atom == pytest.approx(200.0)
        assert cfg2.min_aspect_ratio == pytest.approx(0.6)


class TestExtendedOutputSpec:
    """Check the optional output fields and that removed ones are rejected."""

    def test_optional_fields_have_correct_defaults(self):
        schema = OutputSpec()
        assert schema.snapshot_clean is True
        assert schema.wrap_atoms is True
        assert schema.save_acc_rates is False
        assert schema.acc_rates_interval == 1

    def test_optional_fields_accept_non_defaults(self):
        schema = OutputSpec(
            format="none",
            snapshot_clean=False,
            wrap_atoms=False,
        )
        assert schema.snapshot_clean is False
        assert schema.wrap_atoms is False

    def test_removed_fields_rejected(self):
        """``snapshot_time``, ``write_traj_db`` and ``write_walkers_db`` are
        no longer fields, so ``extra='forbid'`` rejects them.
        """
        for key in ("snapshot_time", "write_traj_db", "write_walkers_db"):
            with pytest.raises(ValidationError):
                OutputSpec(**{key: 1})

    def test_yaml_round_trip_extended_output(self):
        schema = OutputSpec(
            format="none",
            wrap_atoms=False,
            snapshot_clean=False,
        )
        dumped = schema.model_dump(mode="json")
        reloaded = yaml.safe_load(yaml.safe_dump(dumped))
        schema2 = OutputSpec.model_validate(reloaded)
        assert schema2.wrap_atoms is False
        assert schema2.snapshot_clean is False


_FULL_CONFIG_YAML = _DATA / "full_config.yaml"


class TestFullConfigFixture:
    """Check that ``full_config.yaml`` parses and passes ``validate``."""

    def test_full_config_validates(self):
        with open(_FULL_CONFIG_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootSpec.model_validate(raw)
        assert root.run.n_live == 20
        assert root.init.start_species == "1 2"
        assert root.cell.max_volume_per_atom == pytest.approx(500.0)
        assert root.output.traj_interval == 5

    def test_full_config_cli_validate(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_FULL_CONFIG_YAML)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out


_SCHEMA_MODULES = (
    "adaptation",
    "backend",
    "cell",
    "constraints",
    "ensemble",
    "init",
    "inter_re",
    "moves",
    "output",
    "root",
    "run",
    "termination",
)


def _all_spec_models():
    """Every pydantic model declared in ``jaxrens.cli.schema.*``."""
    for mod_name in _SCHEMA_MODULES:
        module = importlib.import_module(f"jaxrens.cli.schema.{mod_name}")
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj.__module__ == module.__name__
            ):
                yield module.__name__, name, obj


class TestFieldDescriptions:
    """Check that every config field has a ``Field(description=...)``.

    The description feeds the generated tables in ``docs/reference/config``,
    the ``jaxrens dump-schema`` JSON and pydantic's validation errors. A
    field without one shows up as a blank cell in the config reference.
    """

    def test_every_field_has_a_description(self):
        missing = [
            f"{module}.{cls}.{field}"
            for module, cls, model in _all_spec_models()
            for field, info in model.model_fields.items()
            if not (info.description or "").strip()
        ]
        assert not missing, (
            "config schema fields without Field(description=...): "
            + ", ".join(sorted(missing))
        )

    def test_descriptions_reach_the_json_schema(self):
        """``dump-schema`` carries the same descriptions the docs render.

        The check compares with the model field, not a fixed phrase, so
        editing a description does not break it.
        """
        schema = RootSpec.model_json_schema()
        assert schema["properties"]["run"]["description"] == (
            RootSpec.model_fields["run"].description
        )

        run_props = schema["$defs"]["RunSpec"]["properties"]
        for name, info in RunSpec.model_fields.items():
            assert run_props[name]["description"] == info.description


class TestDocumentedExampleConfig:
    """Check that the "A complete config" block in the docs parses.

    ``docs/reference/config.rst`` opens with a YAML skeleton covering every
    top-level section. It is written by hand while the tables around it are
    generated, so a key renamed in the schema would otherwise go unnoticed.
    """

    _DOC = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "reference"
        / "config.rst"
    )

    def _example_yaml(self) -> str:
        text = self._DOC.read_text()
        start = text.index(
            ".. code-block:: yaml", text.index("A complete config")
        )
        body, started = [], False
        for line in text[start:].splitlines()[1:]:
            if not line.strip():
                body.append("")
            elif line.startswith("   "):
                started = True
                body.append(line[3:])
            elif started:
                break
        return "\n".join(body)

    @pytest.mark.skipif(
        not _DOC.exists(), reason="docs tree not present in this checkout"
    )
    def test_example_validates_without_warnings(self, recwarn):
        root = RootSpec.model_validate(yaml.safe_load(self._example_yaml()))
        assert [m.type for m in root.moves] == [
            "gmc",
            "volume",
            "shear",
            "stretch",
        ]
        assert root.backend.type == "lj"
        assert root.ensemble.type == "npt"
        assert [t.type for t in root.termination] == ["prior_mass"]
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    @pytest.mark.skipif(
        not _DOC.exists(), reason="docs tree not present in this checkout"
    )
    def test_example_covers_every_top_level_key(self):
        documented = set(yaml.safe_load(self._example_yaml()))
        declared = set(RootSpec.model_fields) - {"interval_units"}
        assert declared - documented == set(), (
            "config sections missing from the documented example: "
            f"{sorted(declared - documented)}"
        )


# Discriminated-union sections whose variants are rendered as tabs in the
# configuration reference: (YAML section, module, union alias).
_VARIANT_UNIONS = [
    ("moves", "jaxrens.cli.schema.moves", "MoveSpec"),
    ("backend", "jaxrens.cli.schema.backend", "BackendSpec"),
    ("ensemble", "jaxrens.cli.schema.ensemble", "EnsembleSpec"),
    ("termination", "jaxrens.cli.schema.termination", "TerminationSpec"),
    ("constraints", "jaxrens.cli.schema.constraints", "ConstraintSpec"),
]


def _union_members(module_path: str, attr: str):
    """Concrete variant classes behind a discriminated-union alias."""
    union = getattr(importlib.import_module(module_path), attr)
    args = typing.get_args(union)
    members = [m for m in typing.get_args(args[0]) if inspect.isclass(m)]
    return members or [args[0]]


def _variant_cases():
    """One parametrize case per (section, concrete variant) pair.

    The cases come from the unions themselves, so a new variant fails
    until it has an example instead of being skipped.
    """
    for section, module_path, attr in _VARIANT_UNIONS:
        for spec in _union_members(module_path, attr):
            yield pytest.param(section, spec, id=f"{section}-{spec.__name__}")


def _docstring_examples(model, section: str) -> list[str]:
    """Pull indented ``<section>:`` literal blocks out of a docstring."""
    doc = inspect.getdoc(model) or ""
    blocks, current = [], None
    for line in doc.splitlines():
        if current is not None:
            if not line.strip() or line.startswith((" ", "\t")):
                current.append(line)
                continue
            blocks.append(current)
            current = None
        # reST opens a literal block either with a bare ``::`` line or
        # with ``::`` appended to the preceding sentence; the move
        # docstrings use both spellings.
        if line.strip().endswith("::"):
            current = []
    if current is not None:
        blocks.append(current)
    return [
        textwrap.dedent("\n".join(b))
        for b in blocks
        if f"{section}:" in "\n".join(b)
    ]


@pytest.mark.parametrize("section,spec", list(_variant_cases()))
def test_docstring_example_present_and_valid(section, spec):
    """Every variant spec has a YAML example in its docstring, and it parses.

    The config reference renders each variant's docstring as its tab, so
    these snippets are what a reader copies. They are written by hand
    inside generated pages, so a renamed field would otherwise go
    unnoticed.
    """
    examples = _docstring_examples(spec, section)
    assert examples, (
        f"{spec.__name__} has no ``{section}:`` example in its "
        "docstring; the config reference renders that docstring as the "
        "variant's tab, so the tab would show no usage snippet."
    )
    declared = spec.model_fields["type"].default
    for text in examples:
        fragment = yaml.safe_load(text)
        base = _minimal_dict()
        base.update(fragment)
        root = RootSpec.model_validate(base)
        entries = getattr(root, section)
        if not isinstance(entries, list):
            entries = [entries]
        assert any(e.type == declared for e in entries), (
            f"{spec.__name__} example declares no `type: {declared}` "
            f"entry; got {[e.type for e in entries]}"
        )


class TestTutorialConfigs:
    """Check that every tutorial config parses against the current schema.

    The docs pages include these files, so a renamed field would otherwise
    ship a broken example.
    """

    _TUTORIALS = sorted(
        (Path(__file__).resolve().parents[2] / "examples" / "tutorials").glob(
            "*/config.yaml"
        )
    )

    def test_tutorials_are_discovered(self):
        assert self._TUTORIALS, "no examples/tutorials/*/config.yaml found"

    @pytest.mark.parametrize("path", _TUTORIALS, ids=lambda p: p.parent.name)
    def test_tutorial_config_validates(self, path):
        RootSpec.model_validate(yaml.safe_load(path.read_text()))
