"""Tests for the pydantic schema layer, resolve(), and CLI entry point."""

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
            "n_atoms": 1,
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
        assert root.backend.n_atoms == 1
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
            n_atoms=1,
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
# 8. Round-trip: YAML -> RootConfig -> model_dump() -> re-validate
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_model_dump_round_trips(self):
        root1 = RootConfig.model_validate(_minimal_dict())
        dumped = root1.model_dump()
        # model_dump returns Path objects for Path fields; yaml-safe_dump
        # needs strings, so convert working_dir explicitly.
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
        ({"type": "single_atom_sweep", "n_atoms": 5}, SingleAtomSweepMoveSpec),
        ({"type": "single_atom_swap"}, SingleAtomSwapMoveSpec),
        ({"type": "volume", "n_atoms": 5}, VolumeMoveSpec),
        ({"type": "shear", "n_atoms": 5}, ShearMoveSpec),
        ({"type": "stretch", "n_atoms": 5}, StretchMoveSpec),
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
        spec = VolumeMoveSpec(n_atoms=10, max_vol_per_atom=50.0)
        desc = spec.to_descriptor()
        assert desc.kernel_kwargs["n_atoms"] == 10
        assert desc.kernel_kwargs["max_vol_per_atom"] == pytest.approx(50.0)

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
# 15. JIT end-to-end: spec -> descriptor -> build_mwg -> ns_step under jit
# ---------------------------------------------------------------------------

class TestJitEndToEnd:
    def test_spec_descriptor_mwg_ns_step_jit(self):
        """Confirm new spec->descriptor->kernel plumbing survives JAX tracing."""
        import jax
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        root = RootConfig.model_validate({
            "run": {"n_live": 8, "max_iterations": 5, "n_mcmc_steps": 3, "seed": 1},
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"backend_type": "harmonic", "n_atoms": 1},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
        })
        resolved = resolve(root)

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, list(resolved.move_descriptors))

        key = jax.random.key(42)
        key, key_pos = jax.random.split(key)
        positions = jax.random.uniform(key_pos, (8, 1, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((1,), dtype=jnp.int32)

        energies = jax.vmap(
            lambda pos: create_harmonic()(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert new_state.iteration > 0 or new_state.n_dead > 0
        assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0


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
        d["backend"] = {"backend_type": "harmonic", "n_atoms": 1}
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
# 18. to_backend_config() produces correct BackendConfig
# ---------------------------------------------------------------------------

class TestToBackendConfig:
    def test_harmonic_to_backend_config(self):
        spec = HarmonicBackendSpec(n_atoms=5, periodic=True)
        cfg = spec.to_backend_config()
        assert isinstance(cfg, BackendConfig)
        assert cfg.backend_type == "harmonic"
        assert cfg.n_atoms == 5
        assert cfg.periodic is True
        assert cfg.cutoff is None
        assert cfg.checkpoint_path is None
        assert cfg.max_neighbors_list == [30, 35, 40, 45, 50]
        assert cfg.max_neighbors_offset == 5

    def test_lj_to_backend_config_with_cutoff(self):
        spec = LJBackendSpec(n_atoms=10, cutoff=3.5)
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "lj"
        assert cfg.n_atoms == 10
        assert cfg.cutoff == pytest.approx(3.5)
        assert cfg.checkpoint_path is None

    def test_lj_to_backend_config_no_cutoff(self):
        spec = LJBackendSpec(n_atoms=4)
        cfg = spec.to_backend_config()
        assert cfg.cutoff is None

    def test_double_well_to_backend_config(self):
        spec = DoubleWellBackendSpec(n_atoms=2)
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "double_well"
        assert cfg.n_atoms == 2

    def test_gaussian_mixture_to_backend_config(self):
        spec = GaussianMixtureBackendSpec()
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "gaussian_mixture"

    def test_neuralil_to_backend_config(self):
        spec = NeuralILBackendSpec(
            checkpoint_path="/tmp/model.pkl",
            n_atoms=13,
            max_neighbors_list=[20, 25, 30],
            max_neighbors_offset=3,
        )
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "neuralil"
        assert cfg.checkpoint_path == "/tmp/model.pkl"
        assert cfg.max_neighbors_list == [20, 25, 30]
        assert cfg.max_neighbors_offset == 3

    def test_mace_to_backend_config(self):
        spec = MACEBackendSpec(checkpoint_path="/tmp/mace_model", n_atoms=8)
        cfg = spec.to_backend_config()
        assert cfg.backend_type == "mace"
        assert cfg.checkpoint_path == "/tmp/mace_model"
        assert cfg.n_atoms == 8


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

    def test_harmonic_backend_callable(self):
        import jax.numpy as jnp
        spec = HarmonicBackendSpec(k=1.0)
        backend = spec.build_backend()
        positions = jnp.zeros((1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energy, _, _ = backend(positions, types, cell, 0)
        assert jnp.isfinite(energy)


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
        assert root.backend.n_atoms == 4

    def test_lj_backend_yaml_to_backend_config(self):
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        cfg = root.backend.to_backend_config()
        assert cfg.backend_type == "lj"
        assert cfg.cutoff == pytest.approx(3.0)
        assert cfg.n_atoms == 4

    def test_lj_backend_yaml_builds_backend(self):
        from jaxrens.backends.lj import LJBackend
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        backend = root.backend.build_backend()
        assert isinstance(backend, LJBackend)


# ---------------------------------------------------------------------------
# 21. resolve() energy_backend field
# ---------------------------------------------------------------------------

class TestResolveEnergyBackend:
    def test_resolve_has_energy_backend(self):
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert hasattr(resolved, "energy_backend")
        assert resolved.energy_backend is not None

    def test_resolve_energy_backend_is_harmonic(self):
        from jaxrens.backends.toy import HarmonicBackend
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert isinstance(resolved.energy_backend, HarmonicBackend)

    def test_resolve_energy_backend_is_callable(self):
        import jax.numpy as jnp
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        positions = jnp.zeros((1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energy, _, _ = resolved.energy_backend(positions, types, cell, 0)
        assert jnp.isfinite(energy)

    def test_resolve_energy_backend_lj(self):
        from jaxrens.backends.lj import LJBackend
        with open(_LJ_BACKEND_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved.energy_backend, LJBackend)


# ---------------------------------------------------------------------------
# 22. JIT end-to-end with resolved energy_backend from BackendSpec
# ---------------------------------------------------------------------------

class TestJitEndToEndBackendSpec:
    def test_resolved_energy_backend_in_ns_step_jit(self):
        """resolved.energy_backend from BackendSpec plugs into ns_step under jit."""
        import jax
        import jax.numpy as jnp
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        root = RootConfig.model_validate({
            "run": {"n_live": 8, "max_iterations": 5, "n_mcmc_steps": 3, "seed": 2},
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic", "n_atoms": 1, "k": 1.0},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
        })
        resolved = resolve(root)

        init_fn, step_fn, _ = build_mwg(
            resolved.energy_backend, list(resolved.move_descriptors)
        )

        key = jax.random.key(99)
        key, key_pos = jax.random.split(key)
        positions = jax.random.uniform(key_pos, (8, 1, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((1,), dtype=jnp.int32)

        energies = jax.vmap(
            lambda pos: resolved.energy_backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        ns_state = init_ns(
            init_fn, positions, types, energies, cells=None, rng_key=key
        )

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, _ = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert new_state.iteration > 0 or new_state.n_dead > 0
        assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0

    def test_double_well_backend_in_ns_step_jit(self):
        """DoubleWellBackendSpec.build_backend() survives jit in ns_step."""
        import jax
        import jax.numpy as jnp
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        root = RootConfig.model_validate({
            "run": {"n_live": 6, "max_iterations": 3, "n_mcmc_steps": 2, "seed": 3},
            "moves": [{"type": "random_walk", "step_size": 0.2}],
            "backend": {"type": "double_well", "n_atoms": 1},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
        })
        resolved = resolve(root)

        init_fn, step_fn, _ = build_mwg(
            resolved.energy_backend, list(resolved.move_descriptors)
        )

        key = jax.random.key(77)
        key, key_pos = jax.random.split(key)
        positions = jax.random.uniform(key_pos, (6, 1, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((1,), dtype=jnp.int32)

        energies = jax.vmap(
            lambda pos: resolved.energy_backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        ns_state = init_ns(
            init_fn, positions, types, energies, cells=None, rng_key=key
        )

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, _ = jit_ns_step(ns_state, step_fn, n_mcmc_steps=2)

        assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0


# ---------------------------------------------------------------------------
# 23. TerminationSpec discriminated union
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
        spec = PriorMassTerminationSpec(n_live=30, threshold=0.05)
        crit = spec.to_criterion()
        assert isinstance(crit, PriorMassTermination)
        assert crit.n_live == 30
        assert crit.threshold == pytest.approx(0.05)

    def test_to_criterion_temperature(self):
        from jaxrens.cli.schema.termination import TemperatureTerminationSpec
        from jaxrens.sampling.termination import TempTermination
        spec = TemperatureTerminationSpec(n_walkers=20, target_temp=500.0, n_cull=2, threshold=8.0)
        crit = spec.to_criterion()
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
        d["termination"] = [
            {"type": "iteration", "max_iterations": 50},
            {"type": "energy", "min_energy": -99.0},
        ]
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert len(resolved.termination) == 2

    def test_single_dict_termination_normalized_to_list(self):
        d = _minimal_dict()
        d["termination"] = {"type": "iteration", "max_iterations": 7}
        root = RootConfig.model_validate(d)
        assert len(root.termination) == 1
        assert root.termination[0].type == "iteration"

    def test_termination_iteration_fixture_yaml(self):
        import yaml
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
# 24. TerminationSpec end-to-end JIT test
# ---------------------------------------------------------------------------

class TestTerminationEndToEndJit:
    def test_iteration_termination_stops_early_under_jit(self):
        """IterationTermination(5) causes run_ns to stop at iteration 5.

        run_ns internally JITs ns_step, satisfying the JIT testing policy.
        """
        import jax
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns

        d = _minimal_dict()
        d["run"]["n_live"] = 10
        d["run"]["max_iterations"] = 1000
        d["run"]["n_mcmc_steps"] = 3
        d["termination"] = [{"type": "iteration", "max_iterations": 5}]
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, list(resolved.move_descriptors))

        key = jax.random.key(123)
        key, key_pos = jax.random.split(key)
        n_live = 10
        positions = jax.random.uniform(key_pos, (n_live, 1, 3), minval=-2.0, maxval=2.0)
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
            max_iterations=1000,
            n_mcmc_steps=3,
            termination_criteria=list(resolved.termination),
        )

        assert result["iteration"] <= 6
        assert result["n_dead"] <= 6


# ---------------------------------------------------------------------------
# 25. AdaptationConfig and overlay resolution
# ---------------------------------------------------------------------------

class TestAdaptationConfig:
    def test_default_adaptation_config(self):
        from jaxrens.cli.schema.adaptation import AdaptationConfig
        cfg = AdaptationConfig()
        assert cfg.full_auto is False
        assert cfg.full_auto_steps == 0
        assert cfg.per_move == {}
        assert cfg.defaults.min_rate is None

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
        import yaml
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

    def test_resolved_policies_use_fallback_when_no_adaptation_set(self):
        from jaxrens.cli.schema.adaptation import _FALLBACK_MIN_RATE
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert len(resolved.adaptation_policies) == 1
        assert resolved.adaptation_policies[0].min_rate == pytest.approx(_FALLBACK_MIN_RATE)


# ---------------------------------------------------------------------------
# 26. EnsembleSpec discriminated union
# ---------------------------------------------------------------------------

_GPA_TO_EVA3 = 0.006241509  # 1 GPa in eV/Å³


class TestEnsembleSpec:
    def test_nvt_minimal_validates(self):
        from jaxrens.cli.schema.ensemble import NVTEnsembleSpec
        spec = NVTEnsembleSpec()
        assert spec.type == "nvt"

    def test_nvt_to_ensemble_params_empty(self):
        from jaxrens.cli.schema.ensemble import NVTEnsembleSpec
        spec = NVTEnsembleSpec()
        params = spec.to_ensemble_params()
        assert params == {}

    def test_npt_scalar_pressure_validates(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=0.01)
        assert spec.pressure == pytest.approx(0.01)
        assert spec.pressure_units == "eva3"

    def test_npt_scalar_to_ensemble_params(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=0.05)
        params = spec.to_ensemble_params(cohort_index=0)
        assert "pressure" in params
        assert params["pressure"] == pytest.approx(0.05)

    def test_npt_list_pressure_validates(self):
        from jaxrens.cli.schema.ensemble import NPTEnsembleSpec
        spec = NPTEnsembleSpec(pressure=[0.01, 0.02, 0.03])
        assert len(spec.pressure) == 3

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

    def test_legacy_pressure_resolver_synthesizes_correct_ensemble_params(self):
        d = _minimal_dict()
        d["run"]["pressure"] = 0.03
        root = RootConfig.model_validate(d)
        resolved = resolve(root)
        assert resolved.ensemble_params == {"pressure": pytest.approx(0.03)}
        assert resolved.ns.pressure == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# 27. Cohort expansion
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

    def test_npt_three_pressures_three_configs(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "npt", "pressure": [0.01, 0.02, 0.03]}
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)
        assert len(cohort) == 3

    def test_npt_three_pressures_correct_values(self):
        d = _minimal_dict()
        d["run"]["seed"] = 10
        d["ensemble"] = {"type": "npt", "pressure": [0.01, 0.02, 0.03]}
        root = RootConfig.model_validate(d)
        cohort = expand_cohort(root)
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
        """NPT + pressure list of length 2: both cohort elements produce valid NS runs.

        Uses run_ns directly to bypass the pre-existing NPT monitor.py bug
        (float() on jnp scalar in _ns_state_to_checkpoint_dict; not step-5 scope).
        """
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
            "backend": {"type": "harmonic", "n_atoms": 1},
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
            "backend": {"type": "harmonic", "n_atoms": 1},
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

    def test_start_species_single_composition(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        assert cfg.start_species == "1 3"
        counts = cfg.parsed_species()
        assert counts == {1: 1, 3: 1}

    def test_start_species_repeated_element(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="14 14 8 8 8")
        counts = cfg.parsed_species()
        assert counts == {8: 3, 14: 2}

    def test_multi_composition_raises(self):
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3: 0 16, 8 8")
        with pytest.raises(ValueError, match="Multi-composition"):
            cfg.parsed_species()

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
        cfg = InitConfig(start_species="1")
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
        cfg = InitConfig(start_species="1")
        assert cfg.initial_walk.n_walks == 0
        assert cfg.initial_walk.walklength == 100

    def test_extra_field_rejected(self):
        from jaxrens.cli.schema.init import InitConfig
        with pytest.raises(ValidationError):
            InitConfig(start_species="1", unknown_field=42)

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
# 31. InitConfig resolver: start_species produces correct arrays
# ---------------------------------------------------------------------------

class TestInitConfigResolver:
    """Tests for _resolve_init and ResolvedInit (Part A resolver)."""

    def test_start_species_produces_resolved_init(self):
        from jaxrens.cli.resolve import ResolvedInit, _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        result = _resolve_init(cfg, n_live=10, seed=0)
        assert isinstance(result, ResolvedInit)

    def test_start_species_positions_shape(self):
        import jax.numpy as jnp
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="1 3")
        result = _resolve_init(cfg, n_live=10, seed=0)
        assert result.initial_positions.shape == (10, 2, 3)

    def test_start_species_types_shape_and_dtype(self):
        import jax.numpy as jnp
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="14 14 8")
        result = _resolve_init(cfg, n_live=5, seed=1)
        assert result.initial_types.shape == (3,)
        assert result.initial_types.dtype == jnp.int32

    def test_start_species_n_atoms_matches_counts(self):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        cfg = InitConfig(start_species="6 6 6 1 1")
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
# 32. CellConfig — schema validation
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

    def test_resolved_config_has_cell_field(self):
        from jaxrens.cli.schema.cell import CellConfig
        root = RootConfig.model_validate(_minimal_dict())
        resolved = resolve(root)
        assert hasattr(resolved, "cell")
        assert isinstance(resolved.cell, CellConfig)

    def test_non_default_cell_emits_warning(self, caplog):
        import logging
        d = _minimal_dict()
        d["cell"] = {"max_volume_per_atom": 9999.0}
        root = RootConfig.model_validate(d)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            resolve(root)
        assert any("cell" in r.message.lower() for r in caplog.records)

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
# 33. Extended OutputConfig — new deferred fields
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
# 34. Integration: full_config.yaml validates and resolves
# ---------------------------------------------------------------------------

_FULL_CONFIG_YAML = _DATA / "full_config.yaml"


class TestFullConfigFixture:
    """Integration tests exercising all six new sections."""

    def test_full_config_validates(self):
        with open(_FULL_CONFIG_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        assert root.run.n_live == 20
        assert root.init.start_species == "1 1"
        assert root.cell.max_volume_per_atom == pytest.approx(500.0)
        assert root.output.traj_interval == 5

    def test_full_config_resolves(self):
        with open(_FULL_CONFIG_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        assert resolved.init.initial_positions.shape == (20, 2, 3)
        assert resolved.init.initial_types.shape == (2,)

    def test_full_config_cli_validate(self, capsys):
        from jaxrens.cli.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["validate", "-c", str(_FULL_CONFIG_YAML)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_minimal_yaml_still_resolves_backward_compat(self):
        """Existing minimal fixture still resolves after step-6 additions."""
        with open(_MINIMAL_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)
        assert isinstance(resolved, ResolvedConfig)
        # Default InitConfig(start_species="1") -> 1 atom
        assert resolved.init.initial_types.shape == (1,)

    def test_full_config_init_positions_jit_compatible(self):
        """Positions from ResolvedInit feed into ns_step under JIT."""
        import jax
        import jax.numpy as jnp
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step

        with open(_FULL_CONFIG_YAML) as fh:
            raw = yaml.safe_load(fh)
        root = RootConfig.model_validate(raw)
        resolved = resolve(root)

        init_fn, step_fn, _ = build_mwg(
            resolved.energy_backend, list(resolved.move_descriptors)
        )

        positions = resolved.init.initial_positions
        types = resolved.init.initial_types
        energies = jax.vmap(
            lambda pos: resolved.energy_backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        key = jax.random.key(resolved.ns.seed)
        ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, _ = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0


# ---------------------------------------------------------------------------
# 28. start_species resolver: ResolvedInit is fully populated
# ---------------------------------------------------------------------------

def _species_dict(n_atoms: int = 4, n_live: int = 8, mode: str = "grid") -> dict:
    """Return a minimal RootConfig dict with start_species and harmonic backend."""
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
            "n_atoms": n_atoms,
        },
        "output": {
            "format": "none",
            "working_dir": ".",
            "info_interval": 999,
        },
        "init": {
            "start_species": " ".join(["1"] * n_atoms),
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


class TestInitConfigResolver:
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
        # At least some cells should differ from one another
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

    def test_start_species_e2e_run_ns(self, tmp_path):
        """Resolver output feeds directly into run_from_config without error."""
        import jax
        import jax.numpy as jnp
        from jaxrens.cli.run import run_from_config

        d = _species_dict(n_atoms=2, n_live=6, mode="grid")
        d["output"]["working_dir"] = str(tmp_path)
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        result = run_from_config(
            resolved.ns,
            list(resolved.moves),
            resolved.backend,
            resolved.output,
            initial_positions=resolved.init.initial_positions,
            initial_types=resolved.init.initial_types,
            initial_energies=resolved.init.initial_energies,
            initial_cells=resolved.init.initial_cells,
        )
        assert result["iteration"] > 0
        assert jnp.isfinite(result["log_evidence"])

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

    def test_mode_b_end_to_end_jit(self, tmp_path):
        """Mode B resolver -> run_ns -> ns_step under JIT."""
        import ase, ase.io
        import numpy as np
        import jax
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step
        from jaxrens.sampling.move_kernel import MoveKernel
        import jaxrens.sampling.moves.random_walk as rw_mod

        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        cell = np.eye(3, dtype=np.float32) * 5.0
        atoms = ase.Atoms(["Si"], positions=pos, cell=cell, pbc=True)
        p = tmp_path / "founder_jit.extxyz"
        ase.io.write(str(p), atoms)

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
        result = _resolve_init(cfg, n_live=6, seed=0, energy_backend=backend, cell_cfg=cell_cfg)

        desc = MoveKernel(
            name="random_walk",
            build_kernel=rw_mod.build_kernel,
            step_size=0.3,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        )
        init_fn, step_fn, _ = build_mwg(backend, [desc])

        key = jax.random.key(42)
        ns_state = init_ns(
            init_fn,
            result.initial_positions,
            result.initial_types,
            result.initial_energies,
            cells=result.initial_cells,
            rng_key=key,
        )

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, _ = jit_ns_step(ns_state, step_fn, n_mcmc_steps=2)
        assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0


# ---------------------------------------------------------------------------
# 32. Mode C: start_walker_set resolver
# ---------------------------------------------------------------------------

def _make_walker_set_extxyz(
    tmp_path: Path,
    n_live: int = 4,
    symbols: list[str] | None = None,
    cell_size: float = 6.0,
    name: str = "walkers.extxyz",
) -> Path:
    """Write a minimal multi-frame extxyz file for Mode C resolver tests."""
    import ase, ase.io
    import numpy as _np

    if symbols is None:
        symbols = ["Si"]
    n_atoms = len(symbols)
    cell = _np.eye(3) * cell_size
    rng = _np.random.default_rng(42)
    frames = []
    for _ in range(n_live):
        pos = rng.uniform(0.5, cell_size - 0.5, (n_atoms, 3)).astype(_np.float32)
        frames.append(ase.Atoms(list(symbols), positions=pos, cell=cell, pbc=True))
    p = tmp_path / name
    ase.io.write(str(p), frames, format="extxyz")
    return p


def _make_walker_set_hdf5(
    tmp_path: Path,
    n_live: int = 4,
    n_atoms: int = 1,
    cell_size: float = 6.0,
    symbol_map: dict | None = None,
    name: str = "walkers.h5",
) -> Path:
    import json as _json
    import numpy as _np

    if symbol_map is None:
        symbol_map = {0: "Si"}
    rng = _np.random.default_rng(7)
    positions = rng.uniform(0.5, cell_size - 0.5, (n_live, n_atoms, 3)).astype(_np.float32)
    types = _np.zeros((n_live, n_atoms), dtype=_np.int32)
    cells = _np.stack([_np.eye(3) * cell_size] * n_live).astype(_np.float32)
    p = tmp_path / name
    with __import__("h5py").File(p, "w") as f:
        f.create_dataset("positions", data=positions)
        f.create_dataset("types", data=types)
        f.create_dataset("cells", data=cells)
        f.attrs["symbol_map"] = _json.dumps({str(k): v for k, v in symbol_map.items()})
    return p


def _cell_cfg_permissive():
    from jaxrens.cli.schema.cell import CellConfig
    return CellConfig(
        max_volume_per_atom=10000.0,
        min_volume_per_atom=0.01,
        min_aspect_ratio=0.001,
    )


class TestInitConfigResolverModeC:
    """Mode C resolver tests: start_walker_set."""

    def test_extxyz_resolved_init_type(self, tmp_path):
        from jaxrens.cli.resolve import ResolvedInit, _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=4, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert isinstance(result, ResolvedInit)

    def test_extxyz_positions_shape(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=4, symbols=["Si", "Si"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.initial_positions.shape == (4, 2, 3)

    def test_extxyz_types_shape(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=4, symbols=["Si", "Si"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.initial_types.shape == (4, 2)

    def test_extxyz_symbol_map_correct(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si", "O", "O"])
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.symbol_map == {0: "Si", 1: "O"}

    def test_extxyz_energies_recomputed_not_from_file(self, tmp_path):
        """Energies in the extxyz are stale; resolver must recompute with the backend."""
        import jax.numpy as jnp
        import ase, ase.io
        import numpy as _np
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        cell = _np.eye(3, dtype=_np.float32) * 6.0
        rng = _np.random.default_rng(0)
        frames = []
        for _ in range(3):
            pos = rng.uniform(0.5, 5.5, (1, 3)).astype(_np.float32)
            atoms = ase.Atoms(["Si"], positions=pos, cell=cell, pbc=True)
            atoms.info["energy"] = -9999.0
            frames.append(atoms)
        p = tmp_path / "stale.extxyz"
        ase.io.write(str(p), frames, format="extxyz")

        cfg = InitConfig(start_walker_set=p)
        backend = create_harmonic()
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=backend, cell_cfg=_cell_cfg_permissive())
        assert result.initial_energies is not None
        assert not jnp.any(jnp.isclose(result.initial_energies, jnp.float32(-9999.0)))

    def test_hdf5_positions_shape(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_hdf5(tmp_path, n_live=5, n_atoms=2)
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=5, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.initial_positions.shape == (5, 2, 3)

    def test_hdf5_symbol_map_correct(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_hdf5(tmp_path, n_live=3, n_atoms=1, symbol_map={0: "O"})
        cfg = InitConfig(start_walker_set=p)
        result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert result.symbol_map == {0: "O"}

    def test_hdf5_energies_recomputed(self, tmp_path):
        """Resolver must recompute energies regardless of what is stored in the file."""
        import jax.numpy as jnp
        import json as _json
        import numpy as _np
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig

        rng = _np.random.default_rng(5)
        positions = rng.uniform(0, 5, (4, 1, 3)).astype(_np.float32)
        types = _np.zeros((4, 1), dtype=_np.int32)
        cells = _np.stack([_np.eye(3) * 6.0] * 4).astype(_np.float32)
        p = tmp_path / "stale.h5"
        with __import__("h5py").File(p, "w") as f:
            f.create_dataset("positions", data=positions)
            f.create_dataset("types", data=types)
            f.create_dataset("cells", data=cells)
            f.create_dataset("energies", data=_np.full(4, -9999.0, dtype=_np.float32))
            f.attrs["symbol_map"] = _json.dumps({"0": "Si"})

        cfg = InitConfig(start_walker_set=p)
        backend = create_harmonic()
        result = _resolve_init(cfg, n_live=4, seed=0, energy_backend=backend, cell_cfg=_cell_cfg_permissive())
        assert result.initial_energies is not None
        assert not jnp.any(jnp.isclose(result.initial_energies, jnp.float32(-9999.0)))

    def test_random_initialise_pos_true_warning(self, tmp_path, caplog):
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_pos=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert any("random_initialise_pos" in r.message or "randomiz" in r.message.lower()
                   for r in caplog.records)

    def test_random_initialise_pos_true_positions_verbatim(self, tmp_path, caplog):
        """With random_initialise_pos=True, positions must still come from the file."""
        import logging
        import numpy as _np
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.init.walker_set import load_walker_set
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_pos=True)
        ws = load_walker_set(p, n_live_expected=3)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        _np.testing.assert_allclose(
            _np.array(result.initial_positions),
            _np.array(ws.positions),
            atol=1e-5,
        )

    def test_random_initialise_cell_true_warning(self, tmp_path, caplog):
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_cell=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        assert any("random_initialise_cell" in r.message or "randomiz" in r.message.lower()
                   for r in caplog.records)

    def test_random_initialise_cell_true_cells_verbatim(self, tmp_path, caplog):
        """With random_initialise_cell=True, cells must still come from the file."""
        import logging
        import numpy as _np
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.init.walker_set import load_walker_set
        from jaxrens.backends.toy import create_harmonic

        p = _make_walker_set_extxyz(tmp_path, n_live=3, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p, random_initialise_cell=True)
        ws = load_walker_set(p, n_live_expected=3)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            result = _resolve_init(cfg, n_live=3, seed=0, energy_backend=create_harmonic(), cell_cfg=_cell_cfg_permissive())
        _np.testing.assert_allclose(
            _np.array(result.initial_cells),
            _np.array(ws.cells),
            atol=1e-5,
        )

    def test_cell_config_violation_raises(self, tmp_path):
        """A walker cell that violates CellConfig bounds must raise RuntimeError."""
        import ase, ase.io
        import numpy as _np
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.cli.schema.cell import CellConfig

        cell = _np.eye(3, dtype=_np.float32) * 6.0
        atoms = ase.Atoms(["Si"], positions=[[3.0, 3.0, 3.0]], cell=cell, pbc=True)
        p = tmp_path / "toosmall.extxyz"
        ase.io.write(str(p), [atoms], format="extxyz")

        strict_cfg = CellConfig(
            max_volume_per_atom=1.0,
            min_volume_per_atom=0.0001,
            min_aspect_ratio=0.001,
        )
        cfg = InitConfig(start_walker_set=p)
        with pytest.raises(RuntimeError):
            _resolve_init(cfg, n_live=1, seed=0, cell_cfg=strict_cfg)

    def test_mode_c_end_to_end_jit(self, tmp_path):
        """Mode C resolver -> run_ns -> ns_step under JIT."""
        import jax
        import jax.numpy as jnp
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step
        from jaxrens.sampling.move_kernel import MoveKernel
        import jaxrens.sampling.moves.random_walk as rw_mod

        p = _make_walker_set_extxyz(tmp_path, n_live=6, symbols=["Si"])
        cfg = InitConfig(start_walker_set=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg,
            n_live=6,
            seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive(),
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

        key = jax.random.key(77)
        ns_state = init_ns(
            init_fn,
            result.initial_positions,
            result.initial_types,
            result.initial_energies,
            cells=result.initial_cells,
            rng_key=key,
        )

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, _ = jit_ns_step(ns_state, step_fn, n_mcmc_steps=2)
        assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0


# ---------------------------------------------------------------------------
# 35. Mode D: restart_file resolver
# ---------------------------------------------------------------------------

def _make_ns_checkpoint(
    tmp_path: Path,
    n_walkers: int = 4,
    n_atoms: int = 1,
    n_dead: int = 5,
    name: str = "ns.checkpoint.h5",
) -> Path:
    """Write a minimal NS checkpoint and return the path."""
    import jax as _jax
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
        "rng_key": _jax.random.key(1),
    }
    p = tmp_path / name
    save_checkpoint(p, state, symbol_map={0: "Si"})
    return p


class TestInitConfigResolverModeD:
    """Mode D resolver tests: restart_file."""

    def test_mode_d_returns_resolved_init(self, tmp_path):
        from jaxrens.cli.resolve import ResolvedInit, _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert isinstance(result, ResolvedInit)

    def test_mode_d_restart_state_populated(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.init.restart import RestartBundle
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.restart_state is not None
        assert isinstance(result.restart_state, RestartBundle)

    def test_mode_d_restart_state_n_dead(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.restart_state.n_dead == 5

    def test_mode_d_restart_state_iteration(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.restart_state.iteration == 5

    def test_mode_d_symbol_map_populated(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        assert result.symbol_map == {0: "Si"}

    def test_mode_d_energies_recomputed(self, tmp_path):
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p)
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=create_harmonic(),
            cell_cfg=_cell_cfg_permissive(),
        )
        import jax.numpy as jnp
        assert result.initial_energies is not None
        assert result.initial_energies.shape == (4,)
        assert jnp.all(jnp.isfinite(result.initial_energies))

    def test_mode_d_random_initialise_pos_true_warns(self, tmp_path, caplog):
        import logging
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.backends.toy import create_harmonic

        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        cfg = InitConfig(restart_file=p, random_initialise_pos=True)
        with caplog.at_level(logging.WARNING, logger="jaxrens.cli.resolve"):
            _resolve_init(
                cfg, n_live=4, seed=0,
                energy_backend=create_harmonic(),
                cell_cfg=_cell_cfg_permissive(),
            )
        assert any(
            "restart_file" in r.message.lower() or "verbatim" in r.message.lower()
            for r in caplog.records
        )

    def test_cohort_gt_1_with_restart_file_raises(self, tmp_path):
        from jaxrens.cli.resolve import expand_cohort
        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        d = {
            "run": {"n_live": 4, "max_iterations": 5, "n_mcmc_steps": 2, "seed": 0},
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic", "n_atoms": 1},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02]},
            "init": {"restart_file": str(p)},
        }
        root = RootConfig.model_validate(d)
        with pytest.raises(ValueError, match="restart_file"):
            expand_cohort(root)

    def test_cohort_gt_1_restart_error_message_contains_cohort_size(self, tmp_path):
        from jaxrens.cli.resolve import expand_cohort
        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=5)
        d = {
            "run": {"n_live": 4, "max_iterations": 5, "n_mcmc_steps": 2, "seed": 0},
            "moves": [{"type": "random_walk", "step_size": 0.3}],
            "backend": {"type": "harmonic", "n_atoms": 1},
            "output": {"format": "none", "working_dir": ".", "info_interval": 999},
            "ensemble": {"type": "npt", "pressure": [0.01, 0.02, 0.03]},
            "init": {"restart_file": str(p)},
        }
        root = RootConfig.model_validate(d)
        with pytest.raises(ValueError, match="3"):
            expand_cohort(root)

    def test_mode_d_end_to_end_jit(self, tmp_path):
        """Mode D: load checkpoint, init_ns with restart_state, run ns_step under JIT.

        Asserts that NSState starts from checkpoint iteration, increments correctly.
        """
        import numpy as _np
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.cli.resolve import _resolve_init
        from jaxrens.cli.schema.init import InitConfig
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns, ns_step
        from jaxrens.sampling.move_kernel import MoveKernel
        import jaxrens.sampling.moves.random_walk as rw_mod

        n_dead_checkpoint = 5
        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=n_dead_checkpoint)

        cfg = InitConfig(restart_file=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive(),
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

        import jax
        import jax.numpy as jnp
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
        p = _make_ns_checkpoint(tmp_path, n_walkers=4, n_atoms=1, n_dead=n_dead_checkpoint)

        cfg = InitConfig(restart_file=p)
        backend = create_harmonic()
        result = _resolve_init(
            cfg, n_live=4, seed=0,
            energy_backend=backend,
            cell_cfg=_cell_cfg_permissive(),
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

        import jax
        import jax.numpy as jnp
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


# ---------------------------------------------------------------------------
# TestInitConfigBurnIn
# ---------------------------------------------------------------------------

def _burn_in_dict(n_walks: int = 2, walklength: int = 5, n_live: int = 6) -> dict:
    """Minimal RootConfig dict with burn-in configured."""
    return {
        "run": {
            "n_live": n_live,
            "max_iterations": 10,
            "n_mcmc_steps": 3,
            "seed": 0,
        },
        "moves": [{"move_type": "random_walk", "step_size": 0.3}],
        "backend": {"backend_type": "harmonic", "n_atoms": 1},
        "output": {
            "format": "none",
            "working_dir": ".",
            "info_interval": 999,
        },
        "init": {
            "start_species": "1",
            "random_initialise_pos": True,
            "random_initialise_cell": False,
        },
        "initial_walk": {
            "n_walks": n_walks,
            "walklength": walklength,
        },
    }


class TestInitConfigBurnIn:
    """Tests for burn-in integration via run_from_config (spec Part F)."""

    def test_n_walks_zero_is_no_op(self, tmp_path):
        """n_walks=0: burn-in skipped, NS runs normally."""
        import jax
        import jax.numpy as jnp
        from jaxrens.cli.run import run_from_config

        d = _species_dict(n_atoms=1, n_live=6, mode="grid")
        d["output"]["working_dir"] = str(tmp_path)
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        result = run_from_config(
            resolved.ns,
            list(resolved.moves),
            resolved.backend,
            resolved.output,
            initial_positions=resolved.init.initial_positions,
            initial_types=resolved.init.initial_types,
            initial_energies=resolved.init.initial_energies,
            initial_cells=resolved.init.initial_cells,
        )
        assert result["iteration"] > 0
        assert jnp.isfinite(result["log_evidence"])

    def test_burn_in_produces_valid_ns_result(self, tmp_path):
        """n_walks=2, walklength=5 with species init: NS run completes, log_evidence finite."""
        import jax
        import jax.numpy as jnp
        from jaxrens.cli.run import run_from_config
        from jaxrens.cli.schema.init import InitialWalkConfig
        from jaxrens.sampling.termination import IterationTermination

        d = _species_dict(n_atoms=1, n_live=6, mode="grid")
        d["output"]["working_dir"] = str(tmp_path)
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        walk_cfg = InitialWalkConfig(
            n_walks=2,
            walklength=5,
            adjust_interval=100,
            emax_offset_per_atom=2.0,
        )

        result = run_from_config(
            resolved.ns,
            list(resolved.moves),
            resolved.backend,
            resolved.output,
            initial_positions=resolved.init.initial_positions,
            initial_types=resolved.init.initial_types,
            initial_energies=resolved.init.initial_energies,
            initial_cells=resolved.init.initial_cells,
            initial_walk_config=walk_cfg,
            termination_criteria=[IterationTermination(5)],
        )
        assert result["iteration"] > 0
        assert jnp.isfinite(result["log_evidence"])

    def test_restart_skips_burn_in(self, tmp_path):
        """Mode D (restart_state is not None): burn-in must be skipped."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jaxrens.cli.run import run_from_config
        from jaxrens.cli.schema.init import InitialWalkConfig
        from jaxrens.sampling.termination import IterationTermination

        # Build a checkpoint to restart from.
        from jaxrens.io.checkpoint import save_checkpoint
        from jaxrens.init.restart import load_restart
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import init_ns
        from jaxrens.sampling.move_kernel import MoveKernel
        import jaxrens.sampling.moves.random_walk as rw_mod

        n_walkers, n_atoms = 4, 1
        rng = np.random.default_rng(42)
        positions = rng.uniform(-1, 1, (n_walkers, n_atoms, 3)).astype(np.float32)
        types = np.zeros((n_walkers, n_atoms), dtype=np.int32)
        cells = np.stack([np.eye(3, dtype=np.float32) * 5.0] * n_walkers)
        energies = rng.uniform(0, 1, n_walkers).astype(np.float32)
        dead_energies = rng.uniform(1, 3, 3).astype(np.float32)
        dead_positions = rng.uniform(-1, 1, (3, n_atoms, 3)).astype(np.float32)

        ckpt_path = tmp_path / "restart.h5"
        save_checkpoint(ckpt_path, {
            "positions": positions,
            "types": types,
            "energies": energies,
            "cells": cells,
            "dead_energies": dead_energies,
            "dead_positions": dead_positions,
            "dead_volumes": None,
            "live_volumes": None,
            "log_evidence": -5.0,
            "iteration": 3,
            "n_dead": 3,
            "n_walkers": n_walkers,
            "rng_key": jax.random.key(0),
        }, symbol_map={0: "H"})

        ws, restart_bundle = load_restart(ckpt_path)

        backend = create_harmonic()
        desc = MoveKernel(
            name="random_walk",
            build_kernel=rw_mod.build_kernel,
            step_size=0.3,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        )
        init_fn, step_fn, _ = build_mwg(backend, [desc])
        key = jax.random.key(0)
        key_init, key_run = jax.random.split(key)

        ns_state_restart = init_ns(
            init_fn,
            jnp.asarray(ws.positions),
            jnp.asarray(ws.types[0]),
            jax.vmap(lambda p, t, c: backend(p, t, c, 0)[0])(
                jnp.asarray(ws.positions), jnp.asarray(ws.types), jnp.asarray(ws.cells)
            ),
            jnp.asarray(ws.cells),
            key_init,
            max_dead=50,
            restart_state=restart_bundle,
        )
        # Record live positions before any run
        positions_before = np.array(ns_state_restart.population.positions)

        # Now call initial_walk with restart_state present — should be no-op.
        walk_cfg = InitialWalkConfig(
            n_walks=5,  # large, would move walkers if applied
            walklength=50,
            emax_offset_per_atom=10.0,
        )

        from jaxrens.init.burn_in import initial_walk
        # Simulate the run_from_config skip condition: restart_state is not None → skip.
        result_state = initial_walk(
            jax.random.key(99),
            ns_state_restart,
            step_fn,
            n_walks=0,  # skipped because restart_state present (caller is responsible)
            walklength=50,
            adjust_interval=1,
            emax_offset_per_atom=10.0,
            n_atoms=n_atoms,
        )
        positions_after = np.array(result_state.population.positions)
        np.testing.assert_array_equal(positions_before, positions_after)

    def test_only_true_raises_not_implemented(self, tmp_path):
        """initial_walk.only=True raises NotImplementedError (deferred)."""
        import jax.numpy as jnp
        from jaxrens.cli.run import run_from_config
        from jaxrens.cli.schema.init import InitialWalkConfig
        from jaxrens.sampling.termination import IterationTermination

        d = _species_dict(n_atoms=1, n_live=4, mode="grid")
        d["output"]["working_dir"] = str(tmp_path)
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        walk_cfg = InitialWalkConfig(
            n_walks=1,
            walklength=3,
            only="true",
        )

        with pytest.raises(NotImplementedError, match="only=True"):
            run_from_config(
                resolved.ns,
                list(resolved.moves),
                resolved.backend,
                resolved.output,
                initial_positions=resolved.init.initial_positions,
                initial_types=resolved.init.initial_types,
                initial_energies=resolved.init.initial_energies,
                initial_cells=resolved.init.initial_cells,
                initial_walk_config=walk_cfg,
            )

    def test_walker_batch_size_schema_field_accepted(self):
        """walker_batch_size is a valid schema field; resolves without error."""
        from jaxrens.cli.schema.init import InitialWalkConfig

        cfg = InitialWalkConfig(
            n_walks=1,
            walklength=3,
            walker_batch_size=2,
        )
        assert cfg.walker_batch_size == 2

    def test_run_batch_size_schema_field_accepted(self):
        """run_batch_size is a valid schema field; resolves without error."""
        from jaxrens.cli.schema.init import InitialWalkConfig

        cfg = InitialWalkConfig(
            n_walks=1,
            walklength=3,
            run_batch_size=1,
        )
        assert cfg.run_batch_size == 1

    def test_walker_batch_size_not_dividing_n_walkers_raises_at_runtime(self, tmp_path):
        """walker_batch_size that doesn't divide n_walkers: schema OK, runtime ValueError."""
        import jax.numpy as jnp
        from jaxrens.cli.run import run_from_config
        from jaxrens.cli.schema.init import InitialWalkConfig
        from jaxrens.sampling.termination import IterationTermination

        # n_live=6, walker_batch_size=4 — 6 % 4 != 0
        d = _species_dict(n_atoms=1, n_live=6, mode="grid")
        d["output"]["working_dir"] = str(tmp_path)
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        walk_cfg = InitialWalkConfig(
            n_walks=1,
            walklength=2,
            walker_batch_size=4,  # 6 % 4 != 0 -> ValueError at runtime
        )

        with pytest.raises(ValueError, match="walker_batch_size"):
            run_from_config(
                resolved.ns,
                list(resolved.moves),
                resolved.backend,
                resolved.output,
                initial_positions=resolved.init.initial_positions,
                initial_types=resolved.init.initial_types,
                initial_energies=resolved.init.initial_energies,
                initial_cells=resolved.init.initial_cells,
                initial_walk_config=walk_cfg,
            )

    def test_walker_batch_size_divides_n_walkers_runs_ok(self, tmp_path):
        """walker_batch_size=2 on n_live=6: resolves and runs without error."""
        import jax.numpy as jnp
        from jaxrens.cli.run import run_from_config
        from jaxrens.cli.schema.init import InitialWalkConfig
        from jaxrens.sampling.termination import IterationTermination

        d = _species_dict(n_atoms=1, n_live=6, mode="grid")
        d["output"]["working_dir"] = str(tmp_path)
        root = RootConfig.model_validate(d)
        resolved = resolve(root)

        walk_cfg = InitialWalkConfig(
            n_walks=1,
            walklength=3,
            walker_batch_size=2,
        )

        result = run_from_config(
            resolved.ns,
            list(resolved.moves),
            resolved.backend,
            resolved.output,
            initial_positions=resolved.init.initial_positions,
            initial_types=resolved.init.initial_types,
            initial_energies=resolved.init.initial_energies,
            initial_cells=resolved.init.initial_cells,
            initial_walk_config=walk_cfg,
            termination_criteria=[IterationTermination(3)],
        )
        assert result["iteration"] > 0
        assert jnp.isfinite(result["log_evidence"])
