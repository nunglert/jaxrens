"""Verify that JAX is not imported by lightweight jaxrens paths, and that
the ``jax_enable_x64=False`` pin still fires before any JAX-using path.

Each check runs in a fresh subprocess so ``sys.modules`` starts clean.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _python_check(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_jaxrens_does_not_import_jax():
    r = _python_check(
        "import jaxrens, sys; "
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


def test_postprocess_does_not_import_jax():
    r = _python_check(
        "from jaxrens.postprocess import Monitor, MonitorCollection; "
        "import sys; "
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "module",
    ["adaptation_log", "energy_log", "re_stats_log", "acc_rates_log"],
)
def test_io_logging_modules_do_not_import_jax(module):
    r = _python_check(
        f"from jaxrens.io import {module}; "
        "import sys; "
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


def test_cli_plot_does_not_import_jax():
    r = _python_check(
        "from jaxrens.cli.plot import plot_file; "
        "import sys; "
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


def test_cli_dispatcher_module_does_not_import_jax():
    # ``jaxrens.cli.cli`` defers JAX-using sibling imports
    # (``cli.resolve`` / ``cli.run`` / ``cli.schema``) inside their handler
    # functions so the ``plot`` and ``dump-schema`` subcommands stay
    # JAX-free.  Importing the CLI module itself must not load JAX.
    r = _python_check(
        "from jaxrens.cli.cli import main; "
        "import sys; "
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


# The curated top-level API (``from jaxrens import expectation`` /
# ``from jaxrens import run_ns``) is currently DISABLED in
# ``src/jaxrens/__init__.py`` (lazy PEP 562 ``__getattr__`` commented out), so
# these two tests are commented out too.  Reinstate them together with that
# block.
#
# def test_jax_free_root_reexport_does_not_import_jax():
#     # Postprocess re-exports (``expectation``, ``heat_capacity``, ...) are
#     # flagged ``needs_pin=False`` and resolve into ``postprocess.thermodynamics``,
#     # whose parent ``postprocess/__init__.py`` is JAX-free.  The full
#     # ``from jaxrens import ...`` path must stay JAX-free.
#     #
#     # (Pydantic config re-exports like ``NSConfig`` are also ``needs_pin=False``
#     # but their parent ``state/__init__.py`` already loads JAX-using siblings,
#     # so they pay the JAX cost transitively — not a leak of the lazy refactor.)
#     r = _python_check(
#         "from jaxrens import expectation; "
#         "import sys; "
#         "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)"
#     )
#     assert r.returncode == 0, r.stderr
#
#
# def test_lazy_root_export_applies_x64_pin():
#     r = _python_check(
#         "from jaxrens import run_ns; "
#         "import jax; "
#         "assert jax.config.read('jax_enable_x64') is False"
#     )
#     assert r.returncode == 0, r.stderr


def test_direct_submodule_import_applies_x64_pin():
    r = _python_check(
        "from jaxrens.sampling.nested_sampling import ns_step; "
        "import jax; "
        "assert jax.config.read('jax_enable_x64') is False"
    )
    assert r.returncode == 0, r.stderr


def test_direct_move_info_import_applies_x64_pin():
    # ``MoveInfo`` is imported directly from tests/user code, bypassing any
    # module that would otherwise have pinned x64 — the pin must fire from
    # ``jaxrens.sampling``'s own __init__ and from the module itself.
    r = _python_check(
        "from jaxrens.sampling.base import MoveInfo; "
        "import jax; "
        "assert jax.config.read('jax_enable_x64') is False"
    )
    assert r.returncode == 0, r.stderr
