"""Truncate-on-restart helpers for the streamed output artifacts.

On restart the live NS state is restored from a checkpoint written at global
iteration ``M`` (``= checkpoint['iteration'] = n_dead``), but the append-only
output streams (``.energies``, ``.traj``, the per-iteration HDF5 logs) may
contain records *past* ``M`` — whatever the previous process flushed before it
died between two checkpoints.  Resuming and appending would duplicate the
``[M, last_flushed]`` window, corrupting the dead-point sequence (log_Z is
reconstructed from dead-point *row order*, see
``postprocess/thermodynamics.py``) and the trajectory.

These helpers rewind each artifact to the checkpoint by dropping every record
whose global iteration label is ``>= restart_iteration``.  They rely on the
loop emitting a *monotonic* ``record_iteration`` across restart (see
``sampling/run_loop.py``), so the cut point is unambiguous and stable across
repeated restarts.

All helpers are no-ops when ``restart_iteration <= 0`` (fresh run) or the file
does not exist yet.
"""

from __future__ import annotations

from pathlib import Path


def truncate_energies(path: Path | str, restart_iteration: int) -> None:
    """Drop ``.energies`` data rows with iteration label >= restart_iteration.

    The first line is the header (``n_walkers n_cull n_dof 0.0 n_atoms``);
    every subsequent line is ``<iteration> <energy> <volume>``.
    """
    path = Path(path)
    if restart_iteration <= 0 or not path.exists():
        return
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return
    header, rows = lines[0], lines[1:]
    kept = [
        r for r in rows
        if r.strip() and int(r.split()[0]) < restart_iteration
    ]
    with open(path, "w") as f:
        f.write(header)
        f.writelines(kept)


def truncate_extxyz(path: Path | str, restart_iteration: int) -> None:
    """Keep only extxyz frames whose ``info['iter']`` < restart_iteration."""
    path = Path(path)
    if restart_iteration <= 0 or not path.exists():
        return
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    frames = ase_read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    kept = [
        a for a in frames
        if int(a.info.get("iter", -1)) < restart_iteration
    ]
    if kept:
        ase_write(str(path), kept)  # overwrites in place
    else:
        path.unlink()


def truncate_h5_iterations(path: Path | str, restart_iteration: int) -> None:
    """Resize every per-iteration dataset to drop rows >= restart_iteration.

    Works for any of the per-iteration HDF5 logs (``.adaptation.h5``,
    ``.acc_rates.h5``, ``.max_neighbors.h5``, ``.re_stats.h5``): each carries a
    top-level ``iterations`` dataset and stores all other per-entry datasets
    with a matching leading axis (extensible, ``maxshape=(None, ...)``).  Every
    dataset whose leading axis equals the current entry count is resized to the
    kept length, recursing into subgroups (e.g. ``adjustment_stats/``).
    """
    path = Path(path)
    if restart_iteration <= 0 or not path.exists():
        return
    import h5py
    import numpy as np

    with h5py.File(path, "r+") as f:
        if "iterations" not in f:
            return
        iters = np.asarray(f["iterations"][:])
        n_old = int(iters.shape[0])
        n_keep = int(np.sum(iters < restart_iteration))
        if n_keep >= n_old:
            return

        def _resize_group(grp: "h5py.Group") -> None:
            for key in list(grp.keys()):
                item = grp[key]
                if isinstance(item, h5py.Group):
                    _resize_group(item)
                elif item.shape and item.shape[0] == n_old:
                    item.resize(n_keep, axis=0)

        _resize_group(f)


def truncate_h5_traj(path: Path | str, restart_iteration: int) -> None:
    """Delete per-iteration groups (keyed by ``str(iteration)``) >= cut.

    The HDF5 trajectory writer stores each dead point as a group named after
    its iteration.  Snapshot groups (``snapshot_*``) are left untouched — they
    are keyed separately and overwritten by cadence, not appended.
    """
    path = Path(path)
    if restart_iteration <= 0 or not path.exists():
        return
    import h5py

    with h5py.File(path, "r+") as f:
        for key in list(f.keys()):
            if key.startswith("snapshot_"):
                continue
            try:
                it = int(key)
            except ValueError:
                continue
            if it >= restart_iteration:
                del f[key]
