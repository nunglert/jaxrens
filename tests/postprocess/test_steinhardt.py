"""Validate the JAX Steinhardt-Nelson Q/W parameters against QUIP ``get_qw``.

The reference values in ``tests/data/postprocess/steinhardt`` were produced by
QUIP's ``get_qw`` binary (``steinhardt_nelson_qw_module::calc_qw``) on a spread
of structures -- fcc/bcc/hcp/diamond crystals, a rattled crystal, rocksalt
NaCl, an isolated benzene molecule, and a 3-frame trajectory -- for several
``(l, r_cut, r_cut_min)`` settings.  See
``experiments/steinhardt/gen_reference.py`` for the generator.

For each atom we compare per-atom ``q_l`` to machine precision.  ``w_l`` is
``Re(sum c^3 wigner3j) / (sum|c|^2)^1.5``, a 0/0 form for atoms whose
bond-order vector vanishes by symmetry (``q_l ~ 0``, e.g. ``q2`` on a perfectly
cubic site); QUIP and this port then emit uncorrelated round-off there, so
those atoms are excluded from the ``w`` comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from jaxrens.postprocess import calc_qw

_DATA = Path(__file__).parent.parent / "data" / "postprocess" / "steinhardt"
_MANIFEST = json.loads((_DATA / "get_qw_reference.json").read_text())
_TOL = _MANIFEST["tolerance"]

# Flatten (case, frame) into one parametrisation: each frame is its own test.
_PARAMS = []
for _case in _MANIFEST["cases"]:
    for _fi in range(len(_case["frames"])):
        _label = (
            f"{_case['structure']}-f{_fi}"
            f"-l{_case['l']}-rmin{_case['r_cut_min']}"
        )
        _PARAMS.append(pytest.param(_case, _fi, id=_label))


@pytest.mark.parametrize("case, frame_index", _PARAMS)
def test_calc_qw_matches_get_qw(case, frame_index):
    frames = read(_DATA / case["structure"], index=":")
    atoms = frames[frame_index]

    q, w = calc_qw(atoms, case["l"], case["r_cut"], case["r_cut_min"])

    ref = case["frames"][frame_index]
    q_ref = np.asarray(ref["q"])
    w_ref = np.asarray(ref["w"])

    assert q.shape == q_ref.shape
    np.testing.assert_allclose(q, q_ref, atol=_TOL, rtol=0.0)

    # exclude atoms whose bond-order vector vanishes (w = 0/0 is undefined)
    w_defined = q_ref > 1e-6
    if np.any(w_defined):
        np.testing.assert_allclose(
            w[w_defined], w_ref[w_defined], atol=_TOL, rtol=0.0
        )
