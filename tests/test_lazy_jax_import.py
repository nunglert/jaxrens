"""Check that lightweight jaxrens paths don't import JAX, and that the
``jax_enable_x64=False`` pin still fires before any JAX-using path.

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
        "assert 'jax' not in sys.modules, "
        "sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


def test_postprocess_does_not_import_jax():
    r = _python_check(
        "from jaxrens.postprocess import Monitor, MonitorCollection; "
        "import sys; "
        "assert 'jax' not in sys.modules, "
        "sorted(m for m in sys.modules if 'jax' in m)"
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
        "assert 'jax' not in sys.modules, "
        "sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


def test_cli_plot_does_not_import_jax():
    r = _python_check(
        "from jaxrens.cli.plot import plot_file; "
        "import sys; "
        "assert 'jax' not in sys.modules, "
        "sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


def test_cli_dispatcher_module_does_not_import_jax():
    # cli.cli defers its JAX-using imports (cli.resolve, cli.run, cli.schema)
    # into the subcommand handlers, so ``plot`` and ``dump-schema`` stay
    # JAX-free.
    r = _python_check(
        "from jaxrens.cli.cli import main; "
        "import sys; "
        "assert 'jax' not in sys.modules, "
        "sorted(m for m in sys.modules if 'jax' in m)"
    )
    assert r.returncode == 0, r.stderr


def test_direct_submodule_import_applies_x64_pin():
    r = _python_check(
        "from jaxrens.sampling.nested_sampling import ns_step; "
        "import jax; "
        "assert jax.config.read('jax_enable_x64') is False"
    )
    assert r.returncode == 0, r.stderr


def test_direct_move_info_import_applies_x64_pin():
    # Importing MoveInfo alone bypasses every module that would pin x64 first,
    # so the pin must fire from jaxrens.sampling and from the module itself.
    r = _python_check(
        "from jaxrens.sampling.base import MoveInfo; "
        "import jax; "
        "assert jax.config.read('jax_enable_x64') is False"
    )
    assert r.returncode == 0, r.stderr
