"""Per-fire inter-RE swap log: HDF5 writer and reader.

Schema v1 (HDF5 root):
  - Dataset ``iterations``             shape (N,)            int64
  - Dataset ``n_accepted_per_pair``    shape (N, n_pairs)    int32
  - Dataset ``n_attempted_per_pair``   shape (N, n_pairs)    int32
  - Attribute ``n_pairs``              int   (= n_runs - 1)
  - Attribute ``flavor``               str   ('pressure' | 'xrens' | 'semi_grand')
  - Attribute ``re_stats_log_schema_version`` = 1

This file complements ``<prefix>.acc_rates.h5`` (chain-phase per-move
acceptance) and ``<prefix>.adaptation.h5`` (step-size adapt events).
One row per swap *fire* — natural cadence is dictated by upstream
``inter_re.re_interval``, so the iteration index in the file is the
authoritative record of when the manager actually did work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RELog:
    """Parsed per-fire RE swap trace loaded from an .re_stats.h5 file.

    Attributes:
        iterations:           shape (n_entries,) int64 — NS iteration indices
                              on which the manager fired.
        n_accepted_per_pair:  shape (n_entries, n_pairs) int32
        n_attempted_per_pair: shape (n_entries, n_pairs) int32
        n_pairs:              number of adjacent replica pairs (= n_runs - 1).
        flavor:               'pressure' | 'xrens' | 'semi_grand'.
    """

    iterations: np.ndarray
    n_accepted_per_pair: np.ndarray
    n_attempted_per_pair: np.ndarray
    n_pairs: int
    flavor: str

    @property
    def acceptance_rates(self) -> np.ndarray:
        """Per-fire per-pair acceptance rate (float32, zero where 0 attempted)."""
        denom = np.maximum(self.n_attempted_per_pair, 1)
        return (self.n_accepted_per_pair / denom).astype(np.float32)


class RELogger:
    """Append-only HDF5 writer for per-fire inter-RE swap counts.

    Buffers writes in memory and flushes once the NS iteration index
    has advanced by ``flush_interval`` since the last flush (and on
    ``close()``).  The threshold is in absolute NS iterations; per-walker
    YAML units are handled upstream by ``_apply_interval_units``.  The
    file is only created on the first flush, so runs without any swap
    fires never produce an empty artefact.

    Args:
        path:           Destination file path (.re_stats.h5).
        n_pairs:        Number of adjacent replica pairs (= n_runs - 1).
                        May be 0 (for n_runs < 2) — the writer accepts
                        but a callback should not push such entries.
        flavor:         RE flavor label, one of 'pressure', 'xrens',
                        'semi_grand'.  Stored as an HDF5 attr.
        flush_interval: Flush after the NS iteration index has advanced
                        by this many absolute iterations since the
                        previous flush.  Default 1000.
    """

    def __init__(
        self,
        path: Path | str,
        n_pairs: int,
        flavor: str,
        flush_interval: int = 1000,
        mode: str = "w",
    ) -> None:
        self.path = Path(path)
        self.n_pairs = int(n_pairs)
        self.flavor = str(flavor)
        self.flush_interval = max(1, int(flush_interval))
        self._mode = mode
        # First flush honors ``mode``; subsequent flushes always append.
        self._first_flush = True

        self._buf_iters: list[int] = []
        self._buf_n_acc: list[np.ndarray] = []
        self._buf_n_att: list[np.ndarray] = []
        self._last_flush_iter: int | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_entry(
        self,
        iteration: int,
        n_accepted_per_pair: np.ndarray,
        n_attempted_per_pair: np.ndarray,
    ) -> None:
        """Buffer one entry.  Auto-flushes once the iteration index has
        advanced by ``flush_interval`` absolute iters since the last flush.

        Args:
            iteration:            NS iteration index (Python int).
            n_accepted_per_pair:  shape (n_pairs,) int.
            n_attempted_per_pair: shape (n_pairs,) int.
        """
        if self._closed:
            raise RuntimeError("RELogger has been closed.")

        n_acc = self._coerce(n_accepted_per_pair)
        n_att = self._coerce(n_attempted_per_pair)

        iter_int = int(iteration)
        self._buf_iters.append(iter_int)
        self._buf_n_acc.append(n_acc)
        self._buf_n_att.append(n_att)

        if self._last_flush_iter is None:
            self._last_flush_iter = iter_int
        elif iter_int - self._last_flush_iter >= self.flush_interval:
            self._flush()
            self._last_flush_iter = iter_int

    def close(self) -> None:
        """Flush remaining buffer and close the logger.

        If no entries were ever written, no file is created.
        """
        if not self._closed:
            if self._buf_iters:
                self._flush()
            self._closed = True

    @staticmethod
    def read(path: Path | str) -> RELog:
        """Load an .re_stats.h5 file into an RELog dataclass."""
        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            iterations = np.array(f["iterations"][:], dtype=np.int64)
            n_acc = np.array(f["n_accepted_per_pair"][:], dtype=np.int32)
            n_att = np.array(f["n_attempted_per_pair"][:], dtype=np.int32)
            n_pairs = int(f.attrs["n_pairs"])
            flavor = str(f.attrs["flavor"])

        return RELog(
            iterations=iterations,
            n_accepted_per_pair=n_acc,
            n_attempted_per_pair=n_att,
            n_pairs=n_pairs,
            flavor=flavor,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is shape (n_pairs,) int32."""
        arr = np.asarray(arr, dtype=np.int32)
        if arr.shape != (self.n_pairs,):
            raise ValueError(
                f"RELogger expected per-pair array of shape ({self.n_pairs},), "
                f"got shape {arr.shape}"
            )
        return arr

    def _flush(self) -> None:
        """Append buffered entries to the HDF5 file (creating it if needed)."""
        import h5py

        if not self._buf_iters:
            return

        n_new = len(self._buf_iters)
        new_iters = np.array(self._buf_iters, dtype=np.int64)
        new_n_acc = np.stack(self._buf_n_acc, axis=0)        # (n_new, n_pairs)
        new_n_att = np.stack(self._buf_n_att, axis=0)

        self.path.parent.mkdir(parents=True, exist_ok=True)

        mode = self._mode if self._first_flush else "a"
        self._first_flush = False
        with h5py.File(self.path, mode) as f:
            if "iterations" not in f:
                f.attrs["n_pairs"] = self.n_pairs
                f.attrs["flavor"] = self.flavor
                f.attrs["re_stats_log_schema_version"] = 1
                f.create_dataset(
                    "iterations",
                    data=new_iters,
                    maxshape=(None,),
                    chunks=True,
                )
                f.create_dataset(
                    "n_accepted_per_pair",
                    data=new_n_acc,
                    maxshape=(None, self.n_pairs),
                    chunks=True,
                )
                f.create_dataset(
                    "n_attempted_per_pair",
                    data=new_n_att,
                    maxshape=(None, self.n_pairs),
                    chunks=True,
                )
            else:
                old_n = f["iterations"].shape[0]
                f["iterations"].resize((old_n + n_new,))
                f["iterations"][old_n:] = new_iters
                f["n_accepted_per_pair"].resize((old_n + n_new, self.n_pairs))
                f["n_accepted_per_pair"][old_n:] = new_n_acc
                f["n_attempted_per_pair"].resize((old_n + n_new, self.n_pairs))
                f["n_attempted_per_pair"][old_n:] = new_n_att

        self._buf_iters.clear()
        self._buf_n_acc.clear()
        self._buf_n_att.clear()
