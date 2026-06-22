"""Tests for post-hoc committee-uncertainty trajectory annotation.

Builds a tiny plain ensemble NeuralIL backend, writes small extxyz / h5
trajectories, annotates them, and checks the per-frame uncertainty fields.
Skipped if NeuralIL is not installed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxrens.backends.neuralil import _NEURALIL_IMPORT_ERROR, is_available

neuralil_required = pytest.mark.skipif(
    not is_available(),
    reason=f"NeuralIL not installed: {_NEURALIL_IMPORT_ERROR}",
)

_SYMBOLS = ["H", "Si", "H", "Si"]
_TYPES = np.array([0, 1, 0, 1], dtype=np.int32)
_BASE_POS = np.array(
    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    dtype=np.float64,
)
_CELL = 10.0 * np.eye(3)


def _build_ensemble_backend(n_ensemble=3, embed_d=4, r_cutoff=4.0, n_max=3):
    """A tiny plain (non-softcore) ensemble NeuralILBackend for fast tests."""
    import jax
    import jax.numpy as jnp
    from neuralil.bessel_descriptors import PowerSpectrumGenerator
    from neuralil.model import NeuralIL, ResNetCore
    from neuralil.plain_ensembles.model import PlainEnsemble

    from jaxrens.backends.neuralil import NeuralILBackend

    n_types = 2
    dg = PowerSpectrumGenerator(n_max, r_cutoff, n_types, (1, 1, 1))
    individual = NeuralIL(
        n_types,
        embed_d,
        r_cutoff,
        dg,
        dg.process_some_data,
        ResNetCore([8, 4]),
    )
    ensemble = PlainEnsemble(individual, n_ensemble)
    params = ensemble.init(
        jax.random.key(0),
        jnp.asarray(_BASE_POS, dtype=jnp.float32),
        jnp.asarray(_TYPES),
        jnp.asarray(_CELL, dtype=jnp.float32),
        max_neighbors=8,
        method=ensemble.calc_potential_energy,
    )
    return NeuralILBackend(
        model_params=params,
        r_cutoff=r_cutoff,
        sorted_elements=["H", "Si"],
        supercell_trafo=(1, 1, 1),
        n_max=n_max,
        embed_d=embed_d,
        core_widths=[8, 4],
        is_ensemble=True,
        has_morse=False,
        n_ensemble=n_ensemble,
        softcore=False,
    )


def _write_extxyz(path, n_frames=5):
    from ase import Atoms
    from ase.io import write as ase_write

    frames = []
    for k in range(n_frames):
        atoms = Atoms(
            symbols=_SYMBOLS,
            positions=_BASE_POS + 0.05 * k,
            cell=_CELL,
            pbc=True,
        )
        atoms.info["ns_energy"] = 0.0
        frames.append(atoms)
    ase_write(str(path), frames)


def _write_h5(path, n_frames=3):
    import h5py

    with h5py.File(path, "w") as f:
        for k in range(n_frames):
            grp = f.create_group(str(k))
            grp.create_dataset("positions", data=_BASE_POS + 0.05 * k)
            grp.create_dataset("types", data=_TYPES)
            grp.create_dataset("box", data=_CELL)
            grp.attrs["iteration"] = k


@neuralil_required
class TestAnnotateUncertainty:
    def test_extxyz_energy_and_force(self, tmp_path):
        from ase.io import read as ase_read

        from jaxrens.postprocess.uncertainty import (
            annotate_trajectory_uncertainty,
        )

        backend = _build_ensemble_backend()
        path = tmp_path / "ns.traj.extxyz"
        _write_extxyz(path, n_frames=5)

        # chunk_size=2 over 5 frames → chunks (2, 2, 1): exercises stitching.
        out = annotate_trajectory_uncertainty(
            path, backend, with_forces=True, chunk_size=2
        )
        assert out != path  # sibling, not in place
        assert out.name == "ns.traj.annotated.extxyz"

        frames = ase_read(str(out), index=":")
        assert len(frames) == 5
        for atoms in frames:
            assert atoms.info["ns_energy_std"] > 0.0
            fstd = atoms.get_array("ns_force_std")
            assert fstd.shape == (4,)
            assert np.all(fstd >= 0.0)
            assert atoms.info["ns_force_std_max"] == pytest.approx(
                float(fstd.max()), abs=1e-6
            )

    def test_extxyz_energy_only(self, tmp_path):
        from ase.io import read as ase_read

        from jaxrens.postprocess.uncertainty import (
            annotate_trajectory_uncertainty,
        )

        backend = _build_ensemble_backend()
        path = tmp_path / "ns.traj.extxyz"
        _write_extxyz(path, n_frames=3)

        out = annotate_trajectory_uncertainty(
            path, backend, with_forces=False, chunk_size=2
        )
        for atoms in ase_read(str(out), index=":"):
            assert atoms.info["ns_energy_std"] > 0.0
            assert "ns_force_std" not in atoms.arrays

    def test_in_place_overwrites_original(self, tmp_path):
        from ase.io import read as ase_read

        from jaxrens.postprocess.uncertainty import (
            annotate_trajectory_uncertainty,
        )

        backend = _build_ensemble_backend()
        path = tmp_path / "ns.traj.extxyz"
        _write_extxyz(path, n_frames=2)

        out = annotate_trajectory_uncertainty(
            path, backend, with_forces=True, in_place=True, chunk_size=8
        )
        assert out == path
        for atoms in ase_read(str(path), index=":"):
            assert atoms.info["ns_energy_std"] > 0.0

    def test_h5_energy_and_force(self, tmp_path):
        import h5py

        from jaxrens.postprocess.uncertainty import (
            annotate_trajectory_uncertainty,
        )

        backend = _build_ensemble_backend()
        path = tmp_path / "ns.traj.h5"
        _write_h5(path, n_frames=3)

        out = annotate_trajectory_uncertainty(
            path, backend, with_forces=True, chunk_size=2
        )
        assert out.name == "ns.traj.annotated.h5"

        with h5py.File(out, "r") as f:
            for k in range(3):
                grp = f[str(k)]
                assert grp.attrs["energy_std"] > 0.0
                assert grp["force_std"].shape == (4,)
                assert np.all(grp["force_std"][:] >= 0.0)


def test_annotation_chunk_size():
    """Chunk size mirrors the NS per-device walk batch
    ``(1 + n_extra) * runs_per_device``."""
    from jaxrens.cli.run import _annotation_chunk_size

    # No batcher / single run → 1 + n_extra.
    assert _annotation_chunk_size(0) == 1
    assert _annotation_chunk_size(3) == 4
    # vmap-only: shape_prefix=(n_runs,) → runs_per_device = n_runs.
    assert _annotation_chunk_size(2, SimpleNamespace(shape_prefix=(5,))) == 15
    # pmap+vmap: shape_prefix=(n_gpu, n_per_gpu) → runs_per_device = n_per_gpu.
    assert _annotation_chunk_size(1, SimpleNamespace(shape_prefix=(2, 4))) == 8
    # Empty prefix (SingleRun) → runs_per_device = 1.
    assert _annotation_chunk_size(0, SimpleNamespace(shape_prefix=())) == 1


@neuralil_required
class TestPostRunStep:
    """The ``_maybe_annotate_uncertainty`` driver step (run.py)."""

    def _cfg(self, **overrides):
        base = dict(
            write_uncertainty=True,
            format="extxyz",
            write_force_uncertainty=True,
            uncertainty_in_place=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_annotates_for_committee(self, tmp_path):
        from ase.io import read as ase_read

        from jaxrens.cli.run import _maybe_annotate_uncertainty

        backend = _build_ensemble_backend()
        path = tmp_path / "ns.traj.extxyz"
        _write_extxyz(path, n_frames=3)

        _maybe_annotate_uncertainty(self._cfg(), backend, [path])

        out = tmp_path / "ns.traj.annotated.extxyz"
        assert out.exists()
        frames = ase_read(str(out), index=":")
        assert all("ns_energy_std" in a.info for a in frames)
        assert all("ns_force_std" in a.arrays for a in frames)

    def test_noop_when_disabled(self, tmp_path):
        from jaxrens.cli.run import _maybe_annotate_uncertainty

        path = tmp_path / "ns.traj.extxyz"
        _write_extxyz(path, n_frames=2)
        # write_uncertainty=False → no work, backend never touched.
        _maybe_annotate_uncertainty(
            self._cfg(write_uncertainty=False), None, [path]
        )
        assert not (tmp_path / "ns.traj.annotated.extxyz").exists()

    def test_skips_non_committee(self, tmp_path):
        from jaxrens.cli.run import _maybe_annotate_uncertainty

        path = tmp_path / "ns.traj.extxyz"
        _write_extxyz(path, n_frames=2)
        # A backend with no ``is_ensemble`` / ``members`` → committee is None.
        _maybe_annotate_uncertainty(self._cfg(), object(), [path])
        assert not (tmp_path / "ns.traj.annotated.extxyz").exists()


@neuralil_required
class TestCliAnnotate:
    """End-to-end CLI handler against the real (2-member ensemble) fixture."""

    def test_cmd_annotate_uncertainty(self, tmp_path):
        import argparse

        from ase import Atoms
        from ase.io import read as ase_read
        from ase.io import write as ase_write

        from jaxrens.backends.neuralil import create_neuralil
        from jaxrens.cli.cli import _cmd_annotate_uncertainty

        fixture = Path(__file__).parent.parent / "fixtures" / "neuralil_tiny"
        model_pkl = fixture / "model.pkl"
        backend = create_neuralil(
            pickle_file=str(model_pkl), supercell_trafo=(1, 1, 1)
        )
        ref = np.load(fixture / "reference.npz", allow_pickle=True)
        positions = np.asarray(ref["positions"])
        types = np.asarray(ref["types"])
        cell = np.asarray(ref["cell"])
        symbols = [backend.sorted_elements[int(t)] for t in types]

        traj = tmp_path / "t.extxyz"
        frames = [
            Atoms(
                symbols=symbols,
                positions=positions + 0.02 * k,
                cell=cell,
                pbc=True,
            )
            for k in range(2)
        ]
        ase_write(str(traj), frames)

        args = argparse.Namespace(
            traj=str(traj),
            model=str(model_pkl),
            supercell=[1, 1, 1],
            forces=True,
            in_place=False,
            chunk_size=8,
        )
        assert _cmd_annotate_uncertainty(args) == 0

        out = tmp_path / "t.annotated.extxyz"
        assert out.exists()
        for atoms in ase_read(str(out), index=":"):
            assert atoms.info["ns_energy_std"] >= 0.0
            assert "ns_force_std" in atoms.arrays
            assert atoms.get_array("ns_force_std").shape == (len(symbols),)
