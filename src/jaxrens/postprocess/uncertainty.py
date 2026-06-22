"""Post-hoc committee-uncertainty annotation of NS trajectories.

Reads a written trajectory (extxyz or HDF5), batch-evaluates committee energy
(and optionally per-atom force) uncertainty over all frames via a committee
backend's ``members`` / ``energy_members`` methods, and writes the result back
as per-frame ``ns_energy_std`` (+ ``ns_force_std`` per-atom column /
``ns_force_std_max`` scalar).

Pure post-processing: no dependency on the sampling hot path or the run driver.
The committee evaluation is batched over frames in fixed-size chunks (the
per-member force jacobian ``(M, N, 3)`` over all frames at once would OOM), and
``max_neighbors`` is computed exactly from the trajectory geometry (no streaming
overflow truncation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op

logger = logging.getLogger(__name__)

__all__ = ["annotate_trajectory_uncertainty"]


# ---------------------------------------------------------------------------
# Path / format helpers
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".extxyz", ".xyz"):
        return "extxyz"
    if suffix in (".h5", ".hdf5"):
        return "h5"
    raise ValueError(
        f"Cannot infer trajectory format from {path.name!r}; expected one of "
        ".extxyz/.xyz/.h5/.hdf5."
    )


def _annotated_path(path: Path) -> Path:
    return path.parent / (path.stem + ".annotated" + path.suffix)


# ---------------------------------------------------------------------------
# Batched committee evaluation
# ---------------------------------------------------------------------------


def _chunked_map(fn: Any, arrays: tuple, chunk_size: int) -> Any:
    """``jax.vmap`` ``fn`` over the leading axis of each array, in chunks.

    Bounds peak memory (a full vmap of the per-member force jacobian over all
    frames would OOM). The trailing remainder chunk has a different leading
    size, so it triggers one extra compile — acceptable.
    """
    n = arrays[0].shape[0]
    vfn = jax.jit(jax.vmap(fn))
    outs = [
        vfn(*[a[i : i + chunk_size] for a in arrays])
        for i in range(0, n, chunk_size)
    ]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)


def _max_neighbors_over_frames(
    backend: Any,
    positions: jnp.ndarray,
    cells: jnp.ndarray,
    chunk_size: int,
) -> int:
    """Exact static neighbor-buffer size = max geometry count over all frames."""
    mnf = jax.jit(jax.vmap(backend.max_neighbors_for))
    n = positions.shape[0]
    peak = 0
    for i in range(0, n, chunk_size):
        counts = mnf(positions[i : i + chunk_size], cells[i : i + chunk_size])
        peak = max(peak, int(counts.max()))
    return peak


def _eval_uncertainty(
    backend: Any,
    positions: np.ndarray,
    types: np.ndarray,
    cells: np.ndarray,
    with_forces: bool,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(energy_std (F,), force_std (F, N) | None)`` over all frames."""
    from jaxrens.backends.base import committee_uncertainty

    positions = jnp.asarray(positions, dtype=jnp.float32)
    types = jnp.asarray(types, dtype=jnp.int32)
    cells = jnp.asarray(cells, dtype=jnp.float32)

    max_neighbors = _max_neighbors_over_frames(
        backend, positions, cells, chunk_size
    )

    if with_forces:

        def per_frame(pos, spc, cell):
            res = backend.members(pos, spc, cell, max_neighbors)
            return committee_uncertainty(res)

        energy_std, force_std = _chunked_map(
            per_frame, (positions, types, cells), chunk_size
        )
        return np.asarray(energy_std), np.asarray(force_std)

    def per_frame_energy(pos, spc, cell):
        return backend.energy_members(pos, spc, cell, max_neighbors).std()

    energy_std = _chunked_map(
        per_frame_energy, (positions, types, cells), chunk_size
    )
    return np.asarray(energy_std), None


# ---------------------------------------------------------------------------
# extxyz
# ---------------------------------------------------------------------------


def _annotate_extxyz(
    path: Path,
    backend: Any,
    with_forces: bool,
    in_place: bool,
    chunk_size: int,
) -> Path:
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    frames = ase_read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        logger.warning(
            "annotate_uncertainty: %s has no frames; skipping.", path
        )
        return path

    n_atoms = len(frames[0])
    if any(len(f) != n_atoms for f in frames):
        raise ValueError(
            "annotate_uncertainty expects a constant atom count across the "
            "trajectory; got mixed N. Split the file by composition first."
        )

    sym_to_code = {s: i for i, s in enumerate(backend.sorted_elements)}
    positions = np.stack([f.get_positions() for f in frames])
    types = np.stack(
        [[sym_to_code[s] for s in f.get_chemical_symbols()] for f in frames]
    ).astype(np.int32)
    cells = np.stack([np.asarray(f.cell[:]) for f in frames])

    energy_std, force_std = _eval_uncertainty(
        backend, positions, types, cells, with_forces, chunk_size
    )

    for i, atoms in enumerate(frames):
        atoms.info["ns_energy_std"] = float(energy_std[i])
        if force_std is not None:
            fstd = np.asarray(force_std[i])
            atoms.info["ns_force_std_max"] = float(fstd.max())
            atoms.info["ns_force_std_mean"] = float(fstd.mean())
            atoms.new_array("ns_force_std", fstd)

    out_path = path if in_place else _annotated_path(path)
    ase_write(str(out_path), frames)
    return out_path


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------


def _is_dead_point_key(key: str) -> bool:
    # Dead points are groups named by the (non-negative int) iteration index;
    # snapshots are "snapshot_<i>" and are skipped.
    return key.isdigit()


def _annotate_h5(
    path: Path,
    backend: Any,
    with_forces: bool,
    in_place: bool,
    chunk_size: int,
) -> Path:
    import shutil

    import h5py

    out_path = path if in_place else _annotated_path(path)
    if not in_place:
        shutil.copyfile(path, out_path)

    with h5py.File(out_path, "a") as f:
        keys = sorted((k for k in f.keys() if _is_dead_point_key(k)), key=int)
        if not keys:
            logger.warning(
                "annotate_uncertainty: %s has no dead-point groups; skipping.",
                path,
            )
            return out_path

        positions = np.stack([f[k]["positions"][:] for k in keys])
        types = np.stack([f[k]["types"][:] for k in keys]).astype(np.int32)
        cells = np.stack(
            [
                f[k]["box"][:] if "box" in f[k] else np.zeros((3, 3))
                for k in keys
            ]
        )

        energy_std, force_std = _eval_uncertainty(
            backend, positions, types, cells, with_forces, chunk_size
        )

        for i, k in enumerate(keys):
            grp = f[k]
            grp.attrs["energy_std"] = float(energy_std[i])
            if force_std is not None:
                fstd = np.asarray(force_std[i])
                if "force_std" in grp:
                    del grp["force_std"]
                grp.create_dataset("force_std", data=fstd)
                grp.attrs["force_std_max"] = float(fstd.max())
                grp.attrs["force_std_mean"] = float(fstd.mean())

    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_trajectory_uncertainty(
    traj_path: str | Path,
    committee_backend: Any,
    *,
    with_forces: bool = True,
    in_place: bool = False,
    chunk_size: int = 64,
) -> Path:
    """Annotate a written NS trajectory with committee energy/force uncertainty.

    Args:
        traj_path: Path to an extxyz (``.extxyz``/``.xyz``) or HDF5
            (``.h5``/``.hdf5``) trajectory written by the NS run.
        committee_backend: A committee (ensemble) backend exposing ``members``,
            ``energy_members``, ``max_neighbors_for`` and ``sorted_elements``
            (e.g. an ensemble ``NeuralILBackend``). For a non-committee backend
            the committee spread is zero (all-zero σ).
        with_forces: If True, also compute and write per-atom force uncertainty
            (the per-member force jacobian). If False, only energy σ is computed
            (cheaper — no jacobian).
        in_place: If True, edit the original file; else write a sibling
            ``*.annotated.<ext>`` (default).
        chunk_size: Number of frames evaluated per batched ``vmap`` call. Bounds
            peak memory (the force jacobian over all frames at once would OOM).

    Returns:
        Path to the annotated file (the original when ``in_place``, else the
        ``*.annotated.<ext>`` sibling).
    """
    traj_path = Path(traj_path)
    fmt = _detect_format(traj_path)
    if not getattr(committee_backend, "is_ensemble", False):
        logger.warning(
            "annotate_uncertainty: backend is not an ensemble committee; "
            "uncertainty will be zero for every frame."
        )
    if fmt == "extxyz":
        return _annotate_extxyz(
            traj_path, committee_backend, with_forces, in_place, chunk_size
        )
    return _annotate_h5(
        traj_path, committee_backend, with_forces, in_place, chunk_size
    )
