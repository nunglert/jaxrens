"""Unit tests for ``jaxrens._jax_init._configure_xla_flags``.

These poke the ``XLA_FLAGS`` env var directly and call the function; they do
*not* start a JAX backend, so they run anywhere (no GPU needed) and stay fast.
"""

import pytest

from jaxrens._jax_init import _configure_xla_flags


@pytest.fixture(autouse=True)
def _clean_xla_env(monkeypatch):
    """Start each test from a known-empty XLA/knob env."""
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("JAXRENS_XLA_AUTOTUNE", raising=False)


def test_default_is_xla_stock_autotune(monkeypatch):
    """Unset knob -> XLA's own level 4, i.e. fully autotuned kernels."""
    _configure_xla_flags()
    import os

    assert os.environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=4"


@pytest.mark.parametrize("level", ["0", "1", "2", "3", "4"])
def test_explicit_level_honoured(monkeypatch, level):
    monkeypatch.setenv("JAXRENS_XLA_AUTOTUNE", level)
    _configure_xla_flags()
    import os

    assert os.environ["XLA_FLAGS"] == f"--xla_gpu_autotune_level={level}"


def test_appends_to_existing_flags(monkeypatch):
    """We merge, not clobber -- a user's other XLA_FLAGS survive."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/some/dir")
    _configure_xla_flags()
    import os

    flags = os.environ["XLA_FLAGS"]
    assert "--xla_dump_to=/some/dir" in flags
    assert "--xla_gpu_autotune_level=4" in flags


def test_user_autotune_level_left_untouched(monkeypatch):
    """If the user already picked a level in XLA_FLAGS, don't second-guess."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    monkeypatch.setenv("JAXRENS_XLA_AUTOTUNE", "4")  # ignored -- user wins
    _configure_xla_flags()
    import os

    assert os.environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=0"


@pytest.mark.parametrize("bad", ["9", "-1", "high", ""])
def test_invalid_level_warns_and_falls_back(monkeypatch, bad):
    monkeypatch.setenv("JAXRENS_XLA_AUTOTUNE", bad)
    with pytest.warns(RuntimeWarning, match="JAXRENS_XLA_AUTOTUNE"):
        _configure_xla_flags()
    import os

    assert os.environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=4"
