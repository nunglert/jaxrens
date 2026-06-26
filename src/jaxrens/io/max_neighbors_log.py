"""Per-iteration neighbor-bucket diagnostic log: HDF5 writer and reader.

Schema v1 (HDF5 root):
  - Dataset ``iterations``         shape (N,)                  int64
  - Dataset ``max_neighbor_count`` shape (N, n_runs, n_walkers) int32
  - Dataset ``bucket_size``        shape (N, n_runs)           int32
  - Dataset ``overflow``           shape (N, n_runs)           bool
  - Attribute ``n_runs``           int
  - Attribute ``n_walkers``        int
  - Attribute ``max_neighbors_log_schema_version`` = 1

The full per-walker distribution is stored so postprocessing can plot the
spread (min / max / quantiles, or a 2-D heatmap of count-vs-iteration).
Bucket-size and overflow are kept for diagnostic clarity — together they
let a reader see exactly how the outer loop's bucket-ladder management
chased the observed peak, including the brief overflow blips that triggered
each upgrade.

A typical NS run with ``n_live=500`` and ``n_runs=1`` over 10 000 iterations
produces ~20 MB int32 — well within "always cheap" territory.  The file is
opt-in via ``OutputConfig.save_max_neighbors`` so default runs see no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jaxrens.io._buffered_h5 import BufferedH5Logger

logger = logging.getLogger(__name__)


@dataclass
class MaxNeighborsLog:
    """Parsed neighbor-bucket trace loaded from a ``.max_neighbors.h5`` file.

    Attributes:
        iterations:         shape (n_entries,) int64 — NS iteration indices
        max_neighbor_count: shape (n_entries, n_runs, n_walkers) int32
        bucket_size:        shape (n_entries, n_runs) int32
        overflow:           shape (n_entries, n_runs) bool
        n_runs:             number of parallel runs (1 for single-run)
        n_walkers:          number of walkers per run
    """

    iterations: np.ndarray
    max_neighbor_count: np.ndarray
    bucket_size: np.ndarray
    overflow: np.ndarray
    n_runs: int
    n_walkers: int

    @property
    def peak_per_iter(self) -> np.ndarray:
        """Convenience: ``(n_entries, n_runs)`` max-over-walkers reduction."""
        return self.max_neighbor_count.max(axis=-1)

    @property
    def headroom(self) -> np.ndarray:
        """``bucket_size - peak_per_iter`` — slack remaining at each iter."""
        return self.bucket_size.astype(np.int64) - self.peak_per_iter.astype(
            np.int64
        )


class MaxNeighborsLogger(BufferedH5Logger):
    """Append-only HDF5 writer for per-iteration neighbor-bucket diagnostics.

    Buffers writes in memory and flushes once the NS iteration index has
    advanced by ``flush_interval`` since the last flush (and on ``close()``).
    The threshold is in absolute NS iterations; per-walker YAML units are
    handled upstream by ``_apply_interval_units``.  The file is only created
    on the first flush, so runs without the callback never produce an empty
    artefact.

    Args:
        path:           Destination file path (``.max_neighbors.h5``).
        n_runs:         Number of parallel NS runs (use 1 for SingleRun).
        n_walkers:      Number of walkers per run; sets the per-walker
                        dataset shape only.
        flush_interval: Flush after the NS iteration index has advanced by
                        this many absolute iterations since the previous
                        flush.  Default 1000.
    """

    def __init__(
        self,
        path: Path | str,
        n_runs: int,
        n_walkers: int,
        flush_interval: int = 1000,
        mode: str = "w",
        restart_iteration: int = 0,
    ) -> None:
        super().__init__(path, flush_interval, mode, restart_iteration)
        self.n_runs = int(n_runs)
        self.n_walkers = int(n_walkers)

        self._buf_counts: list[np.ndarray] = []
        self._buf_buckets: list[np.ndarray] = []
        self._buf_overflow: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_entry(
        self,
        iteration: int,
        max_neighbor_count: np.ndarray,
        bucket_size: np.ndarray,
        overflow: np.ndarray,
    ) -> None:
        """Buffer one entry.  Auto-flushes once the iteration index has
        advanced by ``flush_interval * n_walkers`` since the last flush.

        Args:
            iteration: NS iteration index (Python int).
            max_neighbor_count: shape ``(n_runs, n_walkers)`` or
                ``(n_walkers,)`` for single-run.
            bucket_size: shape ``(n_runs,)`` or scalar for single-run.
            overflow: shape ``(n_runs,)`` or scalar for single-run.
        """
        if self._closed:
            raise RuntimeError("MaxNeighborsLogger has been closed.")

        counts = self._coerce_counts(max_neighbor_count)
        buckets = self._coerce_per_run(bucket_size, dtype=np.int32)
        of = self._coerce_per_run(overflow, dtype=np.bool_)

        iter_int = int(iteration)
        self._buf_iters.append(iter_int)
        self._buf_counts.append(counts)
        self._buf_buckets.append(buckets)
        self._buf_overflow.append(of)

        self._maybe_flush(iter_int)

    @staticmethod
    def read(path: Path | str) -> MaxNeighborsLog:
        """Load a ``.max_neighbors.h5`` file into a :class:`MaxNeighborsLog`."""
        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            iterations = np.array(f["iterations"][:], dtype=np.int64)
            max_count = np.array(f["max_neighbor_count"][:], dtype=np.int32)
            bucket_size = np.array(f["bucket_size"][:], dtype=np.int32)
            overflow = np.array(f["overflow"][:], dtype=np.bool_)
            n_runs = int(f.attrs["n_runs"])
            n_walkers = int(f.attrs["n_walkers"])

        return MaxNeighborsLog(
            iterations=iterations,
            max_neighbor_count=max_count,
            bucket_size=bucket_size,
            overflow=overflow,
            n_runs=n_runs,
            n_walkers=n_walkers,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_counts(self, arr) -> np.ndarray:
        """Ensure ``max_neighbor_count`` is shape ``(n_runs, n_walkers)`` int32."""
        arr = np.asarray(arr, dtype=np.int32)
        if arr.ndim == 1:
            if arr.shape[0] == self.n_walkers and self.n_runs == 1:
                arr = arr[None, :]
            else:
                raise ValueError(
                    f"Expected counts of shape ({self.n_walkers},) for n_runs=1 "
                    f"or ({self.n_runs}, {self.n_walkers}), got {arr.shape}."
                )
        elif arr.ndim > 2:
            # Flatten any leading prefix (G, P, K) → (G*P, K).
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape != (self.n_runs, self.n_walkers):
            raise ValueError(
                f"Expected counts of shape ({self.n_runs}, {self.n_walkers}), "
                f"got {arr.shape}."
            )
        return arr

    def _coerce_per_run(self, arr, *, dtype) -> np.ndarray:
        """Ensure a per-run scalar (``bucket_size`` or ``overflow``) is
        shape ``(n_runs,)`` with the requested dtype."""
        arr = np.asarray(arr, dtype=dtype)
        if arr.ndim == 0:
            arr = np.broadcast_to(arr, (self.n_runs,)).copy()
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.shape != (self.n_runs,):
            raise ValueError(
                f"Expected per-run array of shape ({self.n_runs},), got {arr.shape}."
            )
        return arr

    def _flush(self) -> None:
        """Write buffered entries to HDF5, extending existing datasets."""
        if not self._buf_iters:
            return

        iters_new = np.array(self._buf_iters, dtype=np.int64)
        counts_new = np.stack(self._buf_counts, axis=0).astype(np.int32)
        buckets_new = np.stack(self._buf_buckets, axis=0).astype(np.int32)
        of_new = np.stack(self._buf_overflow, axis=0).astype(np.bool_)
        n_new = len(iters_new)

        with self._open_flush_file() as f:
            if "iterations" not in f:
                f.create_dataset(
                    "iterations",
                    data=iters_new,
                    maxshape=(None,),
                    chunks=(256,),
                )
                f.create_dataset(
                    "max_neighbor_count",
                    data=counts_new,
                    maxshape=(None, self.n_runs, self.n_walkers),
                    chunks=(min(256, n_new), self.n_runs, self.n_walkers),
                )
                f.create_dataset(
                    "bucket_size",
                    data=buckets_new,
                    maxshape=(None, self.n_runs),
                    chunks=(min(256, n_new), self.n_runs),
                )
                f.create_dataset(
                    "overflow",
                    data=of_new,
                    maxshape=(None, self.n_runs),
                    chunks=(min(256, n_new), self.n_runs),
                )
                f.attrs["n_runs"] = self.n_runs
                f.attrs["n_walkers"] = self.n_walkers
                f.attrs["max_neighbors_log_schema_version"] = 1
            else:
                n_old = f["iterations"].shape[0]
                f["iterations"].resize(n_old + n_new, axis=0)
                f["iterations"][n_old:] = iters_new

                f["max_neighbor_count"].resize(n_old + n_new, axis=0)
                f["max_neighbor_count"][n_old:] = counts_new

                f["bucket_size"].resize(n_old + n_new, axis=0)
                f["bucket_size"][n_old:] = buckets_new

                f["overflow"].resize(n_old + n_new, axis=0)
                f["overflow"][n_old:] = of_new

        self._buf_iters.clear()
        self._buf_counts.clear()
        self._buf_buckets.clear()
        self._buf_overflow.clear()
