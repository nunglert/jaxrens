"""Trajectory writers: pluggable output format implementations.

TrajectoryWriter protocol + ExtxyzTrajectoryWriter, H5TrajectoryWriter,
NullTrajectoryWriter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class TrajectoryWriter(Protocol):
    """Protocol for trajectory output backends.

    The three writers below implement it structurally (no inheritance);
    ``TrajectoryCallback`` in ``jaxrens.cli.monitor`` only ever calls these
    three methods.
    """

    def write_dead_point(
        self, iteration: int, walker: Any, energy: float
    ) -> None:
        ...

    def write_walker_snapshot(self, iteration: int, walkers: Any) -> None:
        ...

    def close(self) -> None:
        ...


def _wrap_positions_np(positions: Any, cell: Any) -> np.ndarray:
    """Wrap positions into the home cell (numpy; writer-side, jax-free).

    Mirrors :func:`jaxrens.utils.cell.wrap_positions` (cell rows are lattice
    vectors, ``positions = frac @ cell``).  No-op for a missing or degenerate
    (non-periodic) cell, so callers can apply it unconditionally.
    """
    pos = np.asarray(positions, dtype=float)
    if cell is None:
        return pos
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
        return pos
    frac = pos @ np.linalg.inv(cell)
    frac -= np.floor(frac)
    return frac @ cell


def _wrapped_walker(walker: Any, wrap: bool) -> Any:
    """Return a ``WalkerState`` with positions wrapped into its own cell.

    Normalizes *walker* to a ``WalkerState`` first, so both typed walkers and
    loose dicts are handled on one path.  A no-op (aside from normalization)
    when wrapping is off or the walker is non-periodic/degenerate.  Returns a
    new ``WalkerState``; the caller's walker is not mutated.
    """
    import jax.numpy as jnp

    from jaxrens.io.formats import ensure_walker_state

    walker = ensure_walker_state(walker)
    if not wrap or walker.cell is None:
        return walker
    wrapped_pos = _wrap_positions_np(walker.positions, walker.cell)
    return walker.set(positions=jnp.asarray(wrapped_pos))


class ExtxyzTrajectoryWriter:
    """Write dead points in extended XYZ format via ASE."""

    def __init__(
        self,
        path: Path | str,
        symbol_map: dict[int, str],
        wrap: bool = True,
        mode: str = "w",
        restart_iteration: int = 0,
        clean_snapshots: bool = False,
    ):
        self.path = Path(path)
        self.symbol_map = symbol_map
        self.wrap = wrap
        self._mode = mode
        self.clean_snapshots = clean_snapshots
        # Path of the most recently written walker snapshot.  When
        # ``clean_snapshots`` is set we delete it as soon as the *next*
        # snapshot is safely on disk (so the directory keeps at most one
        # walker snapshot, but a complete one always exists).
        self._prev_snapshot_path: Path | None = None
        # First write of a run with mode="w" must overwrite any leftover
        # file; subsequent writes within the same run must append so frames
        # accumulate instead of replacing each other.
        self._first_write = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Restart: rewind frames flushed past the checkpoint before appending.
        if mode == "a" and restart_iteration > 0:
            from jaxrens.io.restart_truncate import truncate_extxyz

            truncate_extxyz(self.path, restart_iteration)

    def write_dead_point(
        self, iteration: int, walker: Any, energy: float
    ) -> None:
        from ase.io import write as ase_write

        from jaxrens.io.formats import walker_to_ase_atoms

        atoms = walker_to_ase_atoms(walker, self.symbol_map)
        atoms.info["iter"] = iteration
        atoms.info["ns_energy"] = energy
        if self.wrap and any(atoms.get_pbc()):
            atoms.wrap()
        append = (self._mode == "a") or not self._first_write
        ase_write(str(self.path), atoms, append=append)
        self._first_write = False

    def write_walker_snapshot(self, iteration: int, walkers: Any) -> None:
        # Write all walkers as a snapshot file
        snapshot_path = self.path.with_suffix(f".snap.{iteration}.extxyz")
        from ase.io import write as ase_write

        from jaxrens.io.formats import iter_walker_states, walker_to_ase_atoms

        atoms_list = []
        for i, w in enumerate(iter_walker_states(walkers)):
            atoms = walker_to_ase_atoms(w, self.symbol_map)
            if self.wrap and any(atoms.get_pbc()):
                atoms.wrap()
            atoms.info["iter"] = iteration
            atoms.info["walker_idx"] = i
            atoms_list.append(atoms)

        ase_write(str(snapshot_path), atoms_list)

        # snapshot_clean: now that the new snapshot is fully on disk, drop
        # the previous one (the "second last") so the output directory
        # doesn't accumulate one walker dump per interval.  Deleting only
        # after the new write guarantees at least one complete snapshot
        # always exists, even if the run crashes mid-write.
        if (
            self.clean_snapshots
            and self._prev_snapshot_path is not None
            and self._prev_snapshot_path != snapshot_path
        ):
            try:
                self._prev_snapshot_path.unlink()
            except OSError as exc:
                logger.warning(
                    "snapshot_clean: could not delete previous snapshot %s: %s",
                    self._prev_snapshot_path,
                    exc,
                )
        self._prev_snapshot_path = snapshot_path

    def close(self) -> None:
        pass


class H5TrajectoryWriter:
    """Write dead points in HDF5 format."""

    def __init__(
        self,
        path: Path | str,
        symbol_map: dict[int, str],
        wrap: bool = True,
        mode: str = "w",
        restart_iteration: int = 0,
        clean_snapshots: bool = False,
    ):
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.symbol_map = symbol_map
        self.wrap = wrap
        self._mode = mode
        self.clean_snapshots = clean_snapshots
        # Name of the most recently written snapshot group.  When
        # ``clean_snapshots`` is set we delete it once the next snapshot
        # group is written.  HDF5 does not return freed bytes to the OS, but
        # its free-space manager reuses them for subsequent same-size writes,
        # so the file stays bounded instead of growing one group per snapshot.
        self._prev_snapshot_name: str | None = None
        # Restart: drop per-iteration groups flushed past the checkpoint.
        # Done before opening so the append handle sees the rewound file.
        if mode == "a" and restart_iteration > 0:
            from jaxrens.io.restart_truncate import truncate_h5_traj

            truncate_h5_traj(self.path, restart_iteration)
        self._file = h5py.File(self.path, self._mode)
        self._file.attrs["symbol_map"] = str(symbol_map)
        _stamp_unvalidated(self._file)

    def write_dead_point(
        self, iteration: int, walker: Any, energy: float
    ) -> None:
        from jaxrens.io.formats import walker_to_h5_group

        grp = self._file.create_group(str(iteration))
        walker_to_h5_group(grp, _wrapped_walker(walker, self.wrap))
        grp.attrs["iteration"] = iteration
        grp.attrs["energy"] = energy

    def write_walker_snapshot(self, iteration: int, walkers: Any) -> None:
        from jaxrens.io.formats import iter_walker_states, walker_to_h5_group

        grp_name = f"snapshot_{iteration}"
        grp = self._file.create_group(grp_name)
        grp.attrs["iteration"] = iteration

        # Store the full per-walker state (positions, types, energy, box) in
        # one subgroup per walker, mirroring the extxyz writer's per-frame
        # dump.  ``walker_to_h5_group`` is the same helper used for dead
        # points, so snapshots round-trip via ``h5_group_to_walker``.
        walkers_list = list(iter_walker_states(walkers))
        grp.attrs["n_walkers"] = len(walkers_list)

        for i, w in enumerate(walkers_list):
            walker_to_h5_group(
                grp.create_group(f"walker_{i}"), _wrapped_walker(w, self.wrap)
            )

        # snapshot_clean: drop the previous snapshot group now that the new
        # one is fully written, keeping only the latest in the file.
        if (
            self.clean_snapshots
            and self._prev_snapshot_name is not None
            and self._prev_snapshot_name != grp_name
            and self._prev_snapshot_name in self._file
        ):
            del self._file[self._prev_snapshot_name]
        self._prev_snapshot_name = grp_name

    def close(self) -> None:
        # Re-stamp: markers that fired after the writer was built (a move
        # kernel rebuilt mid-run, say) still belong on the output file.
        _stamp_unvalidated(self._file)
        self._file.close()


class NullTrajectoryWriter:
    """No-op writer for benchmarking or when output is disabled."""

    def write_dead_point(self, *args: Any, **kwargs: Any) -> None:
        pass

    def write_walker_snapshot(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _stamp_unvalidated(h5file: Any) -> None:
    """Record any unvalidated code paths this run touched in the file's attrs.

    A stderr warning is invisible in a batch job and gone by the time anyone
    reads the output; an attribute on the trajectory travels with the data.
    Written at open *and* at close: the first covers a run that crashes, the
    second picks up markers that fired after the writer was constructed.
    """
    from jaxrens.unvalidated import triggered

    records = triggered()
    if not records:
        return
    h5file.attrs["unvalidated_features"] = [
        f"{r.feature} (since {r.since}): {r.concern}" for r in records
    ]


def create_trajectory_writer(
    format: str,
    path: Path | str,
    symbol_map: dict[int, str],
    **kwargs: Any,
) -> TrajectoryWriter:
    """Factory for trajectory writers."""
    match format:
        case "extxyz":
            return ExtxyzTrajectoryWriter(path, symbol_map, **kwargs)
        case "h5":
            return H5TrajectoryWriter(path, symbol_map, **kwargs)
        case "none":
            return (
                NullTrajectoryWriter()
            )  # ignores mode/wrap/restart_iteration kwargs
        case _:
            raise ValueError(
                f"Unknown trajectory format {format!r}. Supported formats "
                f"are 'extxyz', 'h5', and 'none' (writes nothing)."
            )
