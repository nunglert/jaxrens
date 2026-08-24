"""Per-iteration chain-phase acceptance log: HDF5 writer and reader.

Schema v1 (HDF5 root):
  - Dataset ``iterations``    shape (N,)                 int64
  - Dataset ``n_accepted``    shape (N, n_runs, n_moves) int64
  - Dataset ``n_proposed``    shape (N, n_runs, n_moves) int64
  - Attribute ``move_names``  JSON-encoded list of strings
  - Attribute ``n_runs``      int
  - Attribute ``n_moves``     int
  - Attribute ``acc_rates_log_schema_version`` = 1

This file complements ``<prefix>.adaptation.h5``: while the adaptation log
captures step-size *events* (initial baseline + each bisection), this log
captures the chain-phase per-move acceptance signal *every iteration*
(subject to the user-configured interval).  Raw counts are stored so
downstream code can re-aggregate over arbitrary windows; the per-iter
rate is ``n_accepted / max(n_proposed, 1)``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jaxrens.io._buffered_h5 import BufferedH5Logger

logger = logging.getLogger(__name__)


@dataclass
class AccRatesLog:
    """Parsed chain-phase acc-rate trace loaded from an .acc_rates.h5 file.

    Attributes:
        iterations:   shape (n_entries,) int64 — NS iteration indices
        n_accepted:   shape (n_entries, n_runs, n_moves) int64
        n_proposed:   shape (n_entries, n_runs, n_moves) int64
        move_names:   list of move name strings, length n_moves
        n_runs:       number of parallel runs (1 for single-run)
        n_moves:      number of move types
    """

    iterations: np.ndarray
    n_accepted: np.ndarray
    n_proposed: np.ndarray
    move_names: list[str]
    n_runs: int
    n_moves: int

    @property
    def acceptance_rates(self) -> np.ndarray:
        """Convenience: per-iter per-move chain acceptance rate.

        Shape ``(n_entries, n_runs, n_moves)``.  Defined as
        ``n_accepted / max(n_proposed, 1)``; zero where no moves were
        proposed (shouldn't happen on real runs, but the guard keeps
        plotting code from dividing by zero).
        """
        denom = np.maximum(self.n_proposed, 1)
        return (self.n_accepted / denom).astype(np.float32)


class AccRatesLogger(BufferedH5Logger):
    """Append-only HDF5 writer for per-iteration chain acc counts.

    Buffers writes in memory and flushes once the NS iteration index has
    advanced by ``flush_interval`` since the last flush (and on
    ``close()``).  The threshold is in absolute NS iterations; per-walker
    YAML units are handled upstream by ``_apply_interval_units`` exactly
    like every other ``*_interval`` field.  The file is only created on
    the first flush, so runs without the callback never produce an empty
    artefact.

    Args:
        path:           Destination file path (.acc_rates.h5).
        move_names:     Ordered list of move names.
        n_runs:         Number of parallel NS runs (use 1 for single-run).
        flush_interval: Flush after the NS iteration index has advanced by
                        this many absolute iterations since the previous
                        flush.  Default 1000.
    """

    def __init__(
        self,
        path: Path | str,
        move_names: list[str],
        n_runs: int,
        flush_interval: int = 1000,
        mode: str = "w",
        restart_iteration: int = 0,
    ) -> None:
        super().__init__(path, flush_interval, mode, restart_iteration)
        self.move_names = list(move_names)
        self.n_runs = int(n_runs)
        self.n_moves = len(move_names)

        self._buf_n_acc: list[np.ndarray] = []
        self._buf_n_prop: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_entry(
        self,
        iteration: int,
        n_accepted: np.ndarray,
        n_proposed: np.ndarray,
    ) -> None:
        """Buffer one entry.  Auto-flushes once the iteration index has
        advanced by ``flush_interval`` absolute iters since the last flush.

        Args:
            iteration:   NS iteration index (Python int).
            n_accepted:  shape (n_runs, n_moves) or (n_moves,) for n_runs=1.
            n_proposed:  same shape rules as n_accepted.
        """
        if self._closed:
            raise RuntimeError(
                "AccRatesLogger has already been closed; it cannot record "
                "further per-move acceptance rates. A logger is closed once at end of "
                "run, so this means the NS loop wrote after teardown "
                "(or the same logger was closed twice)."
            )

        n_acc = self._coerce(n_accepted)
        n_prop = self._coerce(n_proposed)

        iter_int = int(iteration)
        self._buf_iters.append(iter_int)
        self._buf_n_acc.append(n_acc)
        self._buf_n_prop.append(n_prop)

        self._maybe_flush(iter_int)

    @staticmethod
    def read(path: Path | str) -> AccRatesLog:
        """Load an .acc_rates.h5 file into an AccRatesLog dataclass."""
        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            iterations = np.array(f["iterations"][:], dtype=np.int64)
            n_accepted = np.array(f["n_accepted"][:], dtype=np.int64)
            n_proposed = np.array(f["n_proposed"][:], dtype=np.int64)
            move_names = json.loads(f.attrs["move_names"])
            n_runs = int(f.attrs["n_runs"])
            n_moves = int(f.attrs["n_moves"])

        return AccRatesLog(
            iterations=iterations,
            n_accepted=n_accepted,
            n_proposed=n_proposed,
            move_names=move_names,
            n_runs=n_runs,
            n_moves=n_moves,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is shape (n_runs, n_moves) int64."""
        arr = np.asarray(arr, dtype=np.int64)
        if arr.ndim == 1:
            if arr.shape[0] == self.n_moves:
                arr = arr[None, :]
            else:
                raise ValueError(
                    f"Expected array of shape ({self.n_moves},) or "
                    f"({self.n_runs}, {self.n_moves}), got {arr.shape}"
                )
        if arr.shape != (self.n_runs, self.n_moves):
            raise ValueError(
                f"Expected shape ({self.n_runs}, {self.n_moves}), got {arr.shape}"
            )
        return arr

    def _flush(self) -> None:
        """Write buffered entries to HDF5, extending existing datasets."""
        if not self._buf_iters:
            return

        iters_new = np.array(self._buf_iters, dtype=np.int64)
        acc_new = np.stack(self._buf_n_acc, axis=0).astype(np.int64)
        prop_new = np.stack(self._buf_n_prop, axis=0).astype(np.int64)
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
                    "n_accepted",
                    data=acc_new,
                    maxshape=(None, self.n_runs, self.n_moves),
                    chunks=(min(256, n_new), self.n_runs, self.n_moves),
                )
                f.create_dataset(
                    "n_proposed",
                    data=prop_new,
                    maxshape=(None, self.n_runs, self.n_moves),
                    chunks=(min(256, n_new), self.n_runs, self.n_moves),
                )
                f.attrs["move_names"] = json.dumps(self.move_names)
                f.attrs["n_runs"] = self.n_runs
                f.attrs["n_moves"] = self.n_moves
                f.attrs["acc_rates_log_schema_version"] = 1
            else:
                n_old = f["iterations"].shape[0]
                f["iterations"].resize(n_old + n_new, axis=0)
                f["iterations"][n_old:] = iters_new

                f["n_accepted"].resize(n_old + n_new, axis=0)
                f["n_accepted"][n_old:] = acc_new

                f["n_proposed"].resize(n_old + n_new, axis=0)
                f["n_proposed"][n_old:] = prop_new

        self._buf_iters.clear()
        self._buf_n_acc.clear()
        self._buf_n_prop.clear()
