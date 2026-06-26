"""Each backend's missing-dependency ImportError names its pip extra.

When an optional backend dependency is absent, the ``_require_*`` guard
should tell the user exactly how to install it (``pip install '.[<extra>]'``)
rather than just reporting the bare import failure.  These tests force the
``_AVAILABLE`` flag off so they run regardless of what is actually installed.
"""

from __future__ import annotations

import importlib

import pytest

# (module, availability-flag attr, require-fn attr, expected extra token)
_CASES = [
    (
        "jaxrens.backends.mace",
        "_MACE_JAX_AVAILABLE",
        "_require_mace",
        ".[mace]",
    ),
    (
        "jaxrens.backends.nequix",
        "_NEQUIX_AVAILABLE",
        "_require_nequix",
        ".[nequix]",
    ),
    (
        "jaxrens.backends.jaxmd",
        "_JAXMD_AVAILABLE",
        "_require_jaxmd",
        ".[jaxmd]",
    ),
    (
        "jaxrens.backends.neuralil",
        "_NEURALIL_AVAILABLE",
        "_require_neuralil",
        ".[neuralil]",
    ),
]


@pytest.mark.parametrize("mod_name,flag,require_fn,extra", _CASES)
def test_require_suggests_pip_extra(
    monkeypatch, mod_name, flag, require_fn, extra
):
    mod = importlib.import_module(mod_name)
    monkeypatch.setattr(mod, flag, False)
    with pytest.raises(ImportError) as exc:
        getattr(mod, require_fn)()
    msg = str(exc.value)
    assert "pip install" in msg
    assert extra in msg
