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
from ase.io import read, write

from jaxrens.postprocess.steinhardt import (
    annotate_trajectory_steinhardt,
    calc_qw,
)

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


# ---------------------------------------------------------------------------
# Post-hoc trajectory annotation
# ---------------------------------------------------------------------------


def _write_multiframe_extxyz(path):
    """A 3-frame fcc-Cu trajectory (read from shipped reference inputs)."""
    frames = read(_DATA / "multiframe.xyz", index=":")
    write(str(path), frames)
    return frames


class TestAnnotateTrajectory:
    def test_writes_qw_arrays_and_means(self, tmp_path):
        path = tmp_path / "run.traj.extxyz"
        frames = _write_multiframe_extxyz(path)

        out = annotate_trajectory_steinhardt(path, [4, 6], r_cut=3.2)
        assert out.name == "run.traj.annotated.extxyz"

        annotated = read(str(out), index=":")
        assert len(annotated) == len(frames)
        for src, atoms in zip(frames, annotated):
            for l in (4, 6):
                q_ref, w_ref = calc_qw(src, l, 3.2)
                # per-atom columns round-trip through extxyz
                np.testing.assert_allclose(
                    atoms.get_array(f"q{l}"), q_ref, atol=1e-6, rtol=0.0
                )
                np.testing.assert_allclose(
                    atoms.get_array(f"w{l}"), w_ref, atol=1e-6, rtol=0.0
                )
                # per-frame summaries land in info
                assert atoms.info[f"q{l}_mean"] == pytest.approx(
                    q_ref.mean(), abs=1e-6
                )

    def test_in_place_overwrites_original(self, tmp_path):
        path = tmp_path / "run.traj.extxyz"
        _write_multiframe_extxyz(path)

        out = annotate_trajectory_steinhardt(
            path, [6], r_cut=3.2, in_place=True
        )
        assert out == path
        for atoms in read(str(path), index=":"):
            assert "q6" in atoms.arrays

    def test_rejects_non_extxyz(self, tmp_path):
        path = tmp_path / "run.traj.h5"
        path.write_bytes(b"not really h5")
        with pytest.raises(ValueError, match="extxyz"):
            annotate_trajectory_steinhardt(path, [6], r_cut=3.2)
