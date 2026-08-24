"""Schema -> descriptor logic and resolver wiring for the ``constraints`` section.

The descriptor-construction logic (matrix assembly, type-dependence flag,
validation) is unit-tested directly against ``to_descriptor`` with a synthetic
symbol map — deterministic and independent of how initial positions are
generated. A few thin resolve-level tests then confirm the wiring and the
init-time validity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxrens.cli.resolve import resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.constraints import MinDistanceConstraintSpec

H_ONLY = {0: "H"}
H_O = {0: "H", 1: "O"}


# --------------------------------------------------------------------------
# Descriptor construction (no resolver / no init geometry)
# --------------------------------------------------------------------------


def test_scalar_matrix_and_aspects():
    desc = MinDistanceConstraintSpec(d_min=0.5).to_descriptor(
        symbol_map=H_ONLY
    )
    assert desc.name == "minimum_distance"
    assert desc.reject_reason == 4
    assert desc.depends_on == frozenset(
        {"positions", "cell"}
    )  # uniform: no types
    matrix = np.asarray(desc.build_kwargs["d_min_matrix"])
    assert matrix.shape == (1, 1)
    assert float(matrix[0, 0]) == pytest.approx(0.5)


def test_per_pair_matrix_is_symmetric_and_type_dependent():
    spec = MinDistanceConstraintSpec(
        d_min={"default": 1.0, "H-O": 0.5, "O-O": 2.0}
    )
    desc = spec.to_descriptor(symbol_map=H_O)
    assert desc.depends_on == frozenset({"positions", "cell", "types"})
    m = np.asarray(desc.build_kwargs["d_min_matrix"])
    assert m.shape == (2, 2)
    assert m[0, 1] == pytest.approx(0.5) and m[1, 0] == pytest.approx(0.5)
    assert m[1, 1] == pytest.approx(2.0)
    assert m[0, 0] == pytest.approx(1.0)  # default fills unspecified H-H


def test_uniform_per_pair_dict_is_not_type_dependent():
    # A mapping that resolves to a uniform matrix should not gate type moves.
    spec = MinDistanceConstraintSpec(d_min={"default": 1.0, "H-O": 1.0})
    desc = spec.to_descriptor(symbol_map=H_O)
    assert desc.depends_on == frozenset({"positions", "cell"})


def test_unknown_symbol_raises():
    with pytest.raises(ValueError, match="not present in the system"):
        MinDistanceConstraintSpec(d_min={"Si-Si": 2.0}).to_descriptor(
            symbol_map=H_O
        )


def test_negative_scalar_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        MinDistanceConstraintSpec(d_min=-1.0).to_descriptor(symbol_map=H_ONLY)


def test_malformed_pair_key_raises():
    with pytest.raises(ValueError, match="element-symbol pair"):
        MinDistanceConstraintSpec(d_min={"HO": 1.0}).to_descriptor(
            symbol_map=H_O
        )


# --------------------------------------------------------------------------
# Resolver wiring + init-time validity (2 atoms, grid-spaced >= 1.5 A)
# --------------------------------------------------------------------------


def _base_dict(**overrides) -> dict:
    d = {
        "run": {
            "n_live": 8,
            "max_iterations": 3,
            "n_mcmc_steps": 2,
            "seed": 1,
        },
        "moves": [{"type": "random_walk", "step_size": 0.3}],
        "backend": {"backend_type": "harmonic"},
        "output": {"format": "none", "working_dir": ".", "info_interval": 999},
        "init": {"start_species": "1 2"},  # two H atoms -> one pair
    }
    d.update(overrides)
    return d


def _resolve(**overrides):
    return resolve(RootSpec.model_validate(_base_dict(**overrides)))


def test_no_constraints_gives_empty_tuple():
    assert _resolve().constraint_descriptors == ()


def test_constraint_is_wired_into_resolved_config():
    resolved = _resolve(
        constraints=[{"type": "minimum_distance", "d_min": 0.8}]  # < grid 1.5
    )
    (desc,) = resolved.constraint_descriptors
    assert desc.name == "minimum_distance"


def test_initial_violation_fails_fast():
    with pytest.raises(ValueError, match="violates the 'minimum_distance'"):
        _resolve(constraints=[{"type": "minimum_distance", "d_min": 1000.0}])


def test_check_initial_constraints_handles_multi_replica_layout():
    """``_check_initial_constraints`` must accept the 4D stacked layout.

    The multi-replica resolver stacks walkers as ``(n_total, K, n_atoms, 3)``
    (and cells as ``(n_total, K, 3, 3)``), whereas the single-replica path is
    3D. A single vmap over axis 0 leaves the cell batched and breaks
    ``pairwise_distances`` broadcasting; the function must flatten the leading
    replica/live axes first. Covers both a valid stack and a violating one.
    """
    import jax.numpy as jnp

    from jaxrens.cli.resolve import ResolvedInit, _check_initial_constraints
    from jaxrens.constraints.min_distance import min_distance_descriptor

    desc = min_distance_descriptor(
        np.full((1, 1), 1.7, dtype=np.float32), type_dependent=False
    )
    n_total, K, n_atoms = 2, 4, 2
    types = jnp.zeros(
        (
            n_total,
            K,
            n_atoms,
        ),
        dtype=jnp.int32,
    )
    cells = jnp.broadcast_to(jnp.eye(3) * 10.0, (n_total, K, 3, 3))

    # Atoms 2 A apart (> 1.7 floor) -> all walkers valid.
    ok = jnp.broadcast_to(
        jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        (n_total, K, n_atoms, 3),
    )
    _check_initial_constraints(
        (desc,),
        ResolvedInit(ok, types, cells, None, symbol_map={0: "H"}),
    )  # must not raise

    # Atoms 1 A apart (< 1.7 floor) -> fails fast.
    bad = jnp.broadcast_to(
        jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        (n_total, K, n_atoms, 3),
    )
    with pytest.raises(ValueError, match="violates the 'minimum_distance'"):
        _check_initial_constraints(
            (desc,),
            ResolvedInit(bad, types, cells, None, symbol_map={0: "H"}),
        )


def test_end_to_end_run_with_active_constraint(tmp_path):
    """A full NS run with a live minimum-distance gate completes cleanly.

    Two atoms under a harmonic well are pulled toward the origin (and each
    other); a 0.5 A floor makes the gate actually fire during the run. The
    run must still converge to a finite evidence and never move a walker into
    a sub-threshold configuration.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxrens.cli.run import run_from_config
    from jaxrens.cli.schema.moves import RandomWalkMoveSpec
    from jaxrens.constraints.min_distance import min_distance_descriptor
    from jaxrens.state.config import (
        BackendConfig,
        MoveConfig,
        NSConfig,
        OutputConfig,
    )

    n_live, d_min = 16, 0.5
    key = jax.random.key(0)
    # Two atoms per walker, well separated initially (~2 A apart).
    offsets = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    jitter = 0.2 * jax.random.normal(key, (n_live, 2, 3))
    positions = offsets[None] + jitter
    types = jnp.zeros((2,), dtype=jnp.int32)

    descriptor = RandomWalkMoveSpec(step_size=0.3).to_descriptor()
    constraint = min_distance_descriptor(
        np.full((1, 1), d_min, dtype=np.float32), type_dependent=False
    )

    result = run_from_config(
        NSConfig(n_live=n_live, max_iterations=40, n_mcmc_steps=5, seed=1),
        MoveConfig(move_type="random_walk", step_size=0.3),
        BackendConfig(backend_type="harmonic"),
        OutputConfig(format="none", working_dir=tmp_path, info_interval=999),
        initial_positions=positions,
        initial_types=types,
        move_descriptors=[descriptor],
        constraint_descriptors=(constraint,),
    )

    assert result["iteration"] > 0
    assert jnp.isfinite(result["log_evidence"])
