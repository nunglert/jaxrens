"""Test that all public modules import cleanly.

Part of PR 1: test infrastructure safety net.
Catches broken imports, circular dependencies, and missing __init__.py files.
"""


def test_import_jaxrens():
    import jaxrens
    assert hasattr(jaxrens, "__version__")


def test_import_types():
    from jaxrens.types import Positions, Types, Box, PRNGKey, Params
    assert Positions is not None


def test_import_base():
    from jaxrens.base import (
        MoveInfo,
        StepFn,
        EnergyFn,
        TrajectoryWriter,
        NSCallback,
    )
    assert MoveInfo is not None


def test_import_state():
    from jaxrens.state import (
        WalkerState,
        NSState,
        NSConfig,
        MoveConfig,
        BackendConfig,
        OutputConfig,
    )
    assert WalkerState is not None


def test_import_state_walker():
    from jaxrens.state.walker import WalkerState, static_field
    assert WalkerState is not None


def test_import_state_ns():
    from jaxrens.state.ns import NSState
    assert NSState is not None


def test_import_state_config():
    from jaxrens.state.config import NSConfig, MoveConfig, BackendConfig, OutputConfig
    assert NSConfig is not None


def test_import_subpackages():
    """All sub-packages should be importable."""
    import jaxrens.sampling
    import jaxrens.sampling.moves
    import jaxrens.sampling.adaptation
    import jaxrens.backends
    import jaxrens.io
    import jaxrens.cli
    import jaxrens.utils
