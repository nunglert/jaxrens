"""Adaptation trace logger: HDF5 writer and reader for per-move step sizes and rates.

Schema v1 (HDF5 root):
  - Dataset ``iterations``         shape (N,)               int64
  - Dataset ``step_sizes``         shape (N, n_runs, n_moves) float32
  - Dataset ``acceptance_rates``   shape (N, n_runs, n_moves) float32
  - Attribute ``move_names``       JSON-encoded list of strings
  - Attribute ``n_runs``           int
  - Attribute ``n_moves``          int

Schema v2 (all of v1 plus):
  - Attribute ``adaptation_log_schema_version`` = 2
  - Group ``adjustment_stats/`` containing datasets (N, n_runs, n_moves) unless noted:
    - ``n_rounds``            int32  — bisection rounds executed per adjust call
    - ``converged``           bool   — did the loop converge?
    - ``cap_hits``            int32  — rounds where proposed ss hit max_step_size
    - ``floor_hits``          int32  — rounds where proposed ss hit 1e-20 floor
    - ``bracket_detected``    bool   — both too-high and too-low observed?
    - ``reject_reason_counts``  shape (N, n_runs, n_moves, 4) int32

Schema v3 (all of v2 plus):
  - Attribute ``adaptation_log_schema_version`` = 3
  - Dataset ``n_evaluations``      shape (N, n_runs, n_moves) int64
      — total backend calls per iter per move (energy-only + with-gradient)
  - Dataset ``n_grad_evaluations`` shape (N, n_runs, n_moves) int64
      — subset of n_evaluations that used value_and_grad (i.e. galilean, hmc)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jaxrens.io._buffered_h5 import BufferedH5Logger

logger = logging.getLogger(__name__)

_FLUSH_INTERVAL = (
    1  # flush buffer to HDF5 every N entries — crash-durable by default
)


@dataclass
class AdaptationLog:
    """Parsed adaptation trace loaded from an .adaptation.h5 file.

    Attributes:
        iterations:          shape (n_entries,)               — NS iteration indices
        step_sizes:          shape (n_entries, n_runs, n_moves) — per-move step sizes
        acceptance_rates:    shape (n_entries, n_runs, n_moves) — per-move acc rates
        move_names:          list of move name strings, length n_moves
        n_runs:              number of parallel runs (1 for single-run)
        n_moves:             number of move types
        adjustment_stats:    dict of per-adjust-call diagnostics, or None for v1/v2 files.
            Keys (all arrays shaped (n_entries, n_runs, n_moves) unless noted)::

              - "n_rounds"            int32
              - "converged"           bool
              - "cap_hits"            int32
              - "floor_hits"          int32
              - "bracket_detected"    bool
              - "reject_reason_counts" shape (n_entries, n_runs, n_moves, 4) int32
        n_evaluations:       shape (n_entries, n_runs, n_moves) int64 — total backend
            calls per iter per move. None for v1/v2 files.
        n_grad_evaluations:  shape (n_entries, n_runs, n_moves) int64 — value_and_grad
            subset. None for v1/v2 files.
    """

    iterations: np.ndarray  # (n_entries,)
    step_sizes: np.ndarray  # (n_entries, n_runs, n_moves)
    acceptance_rates: np.ndarray  # (n_entries, n_runs, n_moves)
    move_names: list[str]
    n_runs: int
    n_moves: int
    adjustment_stats: "dict[str, np.ndarray] | None" = None
    n_evaluations: "np.ndarray | None" = (
        None  # (n_entries, n_runs, n_moves) int64
    )
    n_grad_evaluations: "np.ndarray | None" = (
        None  # (n_entries, n_runs, n_moves) int64
    )


class AdaptationLogger(BufferedH5Logger):
    """Append-only HDF5 writer for per-iteration per-move step sizes and rates.

    Buffers writes in memory and flushes to HDF5 every ``_FLUSH_INTERVAL``
    entries (and on ``close()``).  The file is only created when the first
    ``write_entry`` call arrives, so runs without adaptation never produce
    an empty artefact.

    Args:
        path:        Destination file path (.adaptation.h5).
        move_names:  Ordered list of move names.
        n_runs:      Number of parallel NS runs (use 1 for single-run).
    """

    def __init__(
        self,
        path: Path | str,
        move_names: list[str],
        n_runs: int,
        mode: str = "w",
        restart_iteration: int = 0,
    ) -> None:
        # AdaptationLogger keeps its own buffer-length flush trigger
        # (``_FLUSH_INTERVAL`` entries), so the base's iteration-interval
        # cadence is unused here — pass it through for the shared scaffolding.
        super().__init__(path, _FLUSH_INTERVAL, mode, restart_iteration)
        self.move_names = list(move_names)
        self.n_runs = int(n_runs)
        self.n_moves = len(move_names)

        # In-memory buffers
        self._buf_ss: list[np.ndarray] = []  # each (n_runs, n_moves)
        self._buf_acc: list[np.ndarray] = []  # each (n_runs, n_moves)

        # Adjustment stats buffers — keyed by stat name; each entry (n_runs, n_moves) or (n_runs, n_moves, 4)
        self._buf_adj: dict[str, list[np.ndarray]] = {}

        # v3: evaluation counter buffers
        self._buf_n_evals: list[
            np.ndarray
        ] = []  # each (n_runs, n_moves) int64
        self._buf_n_grad_evals: list[
            np.ndarray
        ] = []  # each (n_runs, n_moves) int64

        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_entry(
        self,
        iteration: int,
        step_sizes: np.ndarray,
        acceptance_rates: np.ndarray,
        adjustment_stats: "dict[str, np.ndarray] | None" = None,
        n_evaluations: "np.ndarray | None" = None,
        n_grad_evaluations: "np.ndarray | None" = None,
    ) -> None:
        """Buffer one entry.  Auto-flushes every ``_FLUSH_INTERVAL`` entries.

        Args:
            iteration:        NS iteration index (Python int).
            step_sizes:       shape (n_runs, n_moves) or (n_moves,) for n_runs=1.
            acceptance_rates: same shape rules as step_sizes.
            adjustment_stats: optional dict of per-adjust-call diagnostics.
                Keys: "n_rounds", "converged", "cap_hits", "floor_hits",
                "bracket_detected" (all shape (n_runs, n_moves) or (n_moves,)),
                "reject_reason_counts" (shape (n_runs, n_moves, 4) or (n_moves, 4)).
                If None, no adjustment_stats group is written (v1 behaviour).
            n_evaluations:    optional shape (n_runs, n_moves) or (n_moves,) int64.
                Total backend calls per iter per move. When provided (together with
                n_grad_evaluations), schema v3 datasets are written.
            n_grad_evaluations: optional, same shape as n_evaluations. Subset that
                used value_and_grad.
        """
        if self._closed:
            raise RuntimeError(
                "AdaptationLogger has already been closed; it cannot record "
                "further step-size adaptation events. A logger is closed once at end of "
                "run, so this means the NS loop wrote after teardown "
                "(or the same logger was closed twice)."
            )

        ss = self._coerce(step_sizes)
        acc = self._coerce(acceptance_rates)

        self._buf_iters.append(int(iteration))
        self._buf_ss.append(ss)
        self._buf_acc.append(acc)

        if adjustment_stats is not None:
            for key, val in adjustment_stats.items():
                arr = np.asarray(val)
                # Coerce to (n_runs, n_moves[, 4]) — handle 1D / 2D inputs
                if key == "reject_reason_counts":
                    if arr.ndim == 2:
                        # (n_moves, 4) -> (1, n_moves, 4)
                        arr = arr[None, :]
                    # else assume (n_runs, n_moves, 4)
                    arr = arr.astype(np.int32)
                elif key in ("n_rounds", "cap_hits", "floor_hits"):
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    arr = arr.astype(np.int32)
                else:  # converged, bracket_detected
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    arr = arr.astype(bool)
                if key not in self._buf_adj:
                    self._buf_adj[key] = []
                self._buf_adj[key].append(arr)

        # v3: buffer evaluation counts if provided
        if n_evaluations is not None:
            arr = np.asarray(n_evaluations, dtype=np.int64)
            if arr.ndim == 1:
                arr = arr[None, :]  # (n_moves,) -> (1, n_moves)
            self._buf_n_evals.append(arr)
        if n_grad_evaluations is not None:
            arr = np.asarray(n_grad_evaluations, dtype=np.int64)
            if arr.ndim == 1:
                arr = arr[None, :]
            self._buf_n_grad_evals.append(arr)

        if len(self._buf_iters) >= _FLUSH_INTERVAL:
            self._flush()

    def close(self) -> None:
        """Flush remaining buffer and close the logger.

        If no entries were ever written, no file is created.
        """
        if not self._closed:
            if self._buf_iters:
                self._flush()
            self._closed = True
            # Ensure v3 buffers are cleared even if _flush was not called
            # (e.g., zero entries — no file created path)
            self._buf_n_evals.clear()
            self._buf_n_grad_evals.clear()

    @staticmethod
    def read(path: Path | str) -> AdaptationLog:
        """Load an .adaptation.h5 file into an AdaptationLog dataclass.

        Supports schema v1 (no adjustment_stats), v2 (with adjustment_stats/
        group), and v3 (v2 plus n_evaluations / n_grad_evaluations datasets).
        Legacy v1/v2 files produce ``n_evaluations=None`` and
        ``n_grad_evaluations=None``.

        Args:
            path: Path to the HDF5 file.

        Returns:
            Populated AdaptationLog.
        """
        import h5py

        _ADJ_STAT_DTYPES = {
            "n_rounds": np.int32,
            "converged": bool,
            "cap_hits": np.int32,
            "floor_hits": np.int32,
            "bracket_detected": bool,
            "reject_reason_counts": np.int32,
        }

        path = Path(path)
        with h5py.File(path, "r") as f:
            iterations = np.array(f["iterations"][:], dtype=np.int64)
            step_sizes = np.array(f["step_sizes"][:], dtype=np.float32)
            acceptance_rates = np.array(
                f["acceptance_rates"][:], dtype=np.float32
            )
            move_names = json.loads(f.attrs["move_names"])
            n_runs = int(f.attrs["n_runs"])
            n_moves = int(f.attrs["n_moves"])

            # v2: load adjustment_stats group if present
            adjustment_stats: "dict[str, np.ndarray] | None" = None
            if "adjustment_stats" in f:
                grp = f["adjustment_stats"]
                adjustment_stats = {}
                for key, dtype in _ADJ_STAT_DTYPES.items():
                    if key in grp:
                        adjustment_stats[key] = np.array(
                            grp[key][:], dtype=dtype
                        )

            # v3: load evaluation count datasets if present
            n_evaluations: "np.ndarray | None" = None
            n_grad_evaluations: "np.ndarray | None" = None
            if "n_evaluations" in f:
                n_evaluations = np.array(f["n_evaluations"][:], dtype=np.int64)
            if "n_grad_evaluations" in f:
                n_grad_evaluations = np.array(
                    f["n_grad_evaluations"][:], dtype=np.int64
                )

        return AdaptationLog(
            iterations=iterations,
            step_sizes=step_sizes,
            acceptance_rates=acceptance_rates,
            move_names=move_names,
            n_runs=n_runs,
            n_moves=n_moves,
            adjustment_stats=adjustment_stats,
            n_evaluations=n_evaluations,
            n_grad_evaluations=n_grad_evaluations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is shape (n_runs, n_moves) float32."""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            # (n_moves,) -> (1, n_moves)  or caller passes single-run 1D
            if arr.shape[0] == self.n_moves:
                arr = arr[None, :]  # (1, n_moves)
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
        ss_new = np.stack(self._buf_ss, axis=0).astype(np.float32)
        acc_new = np.stack(self._buf_acc, axis=0).astype(np.float32)
        n_new = len(iters_new)

        # Prepare adjustment stats arrays (may be empty if no stats were provided)
        _ADJ_STAT_MAXSHAPES: dict[str, tuple] = {
            "n_rounds": (None, self.n_runs, self.n_moves),
            "converged": (None, self.n_runs, self.n_moves),
            "cap_hits": (None, self.n_runs, self.n_moves),
            "floor_hits": (None, self.n_runs, self.n_moves),
            "bracket_detected": (None, self.n_runs, self.n_moves),
            "reject_reason_counts": (None, self.n_runs, self.n_moves, 4),
        }
        has_adj = bool(self._buf_adj)
        adj_new: dict[str, np.ndarray] = {}
        if has_adj:
            for key, buf_list in self._buf_adj.items():
                adj_new[key] = np.stack(buf_list, axis=0)

        # v3: prepare evaluation count arrays
        has_evals = bool(self._buf_n_evals) and bool(self._buf_n_grad_evals)
        evals_new: "np.ndarray | None" = None
        grad_evals_new: "np.ndarray | None" = None
        if has_evals:
            evals_new = np.stack(self._buf_n_evals, axis=0).astype(np.int64)
            grad_evals_new = np.stack(self._buf_n_grad_evals, axis=0).astype(
                np.int64
            )

        with self._open_flush_file() as f:
            if "iterations" not in f:
                # First write: create extensible datasets and write attrs
                f.create_dataset(
                    "iterations",
                    data=iters_new,
                    maxshape=(None,),
                    chunks=(256,),
                )
                f.create_dataset(
                    "step_sizes",
                    data=ss_new,
                    maxshape=(None, self.n_runs, self.n_moves),
                    chunks=(min(256, n_new), self.n_runs, self.n_moves),
                )
                f.create_dataset(
                    "acceptance_rates",
                    data=acc_new,
                    maxshape=(None, self.n_runs, self.n_moves),
                    chunks=(min(256, n_new), self.n_runs, self.n_moves),
                )
                f.attrs["move_names"] = json.dumps(self.move_names)
                f.attrs["n_runs"] = self.n_runs
                f.attrs["n_moves"] = self.n_moves

                # Determine schema version based on what data is present
                if has_evals:
                    f.attrs["adaptation_log_schema_version"] = 3
                elif has_adj:
                    f.attrs["adaptation_log_schema_version"] = 2

                if has_adj:
                    grp = f.create_group("adjustment_stats")
                    for key, arr in adj_new.items():
                        maxshape = _ADJ_STAT_MAXSHAPES.get(
                            key, (None,) * arr.ndim
                        )
                        chunk_shape = (min(256, n_new),) + arr.shape[1:]
                        grp.create_dataset(
                            key,
                            data=arr,
                            maxshape=maxshape,
                            chunks=chunk_shape,
                        )

                if has_evals:
                    f.create_dataset(
                        "n_evaluations",
                        data=evals_new,
                        maxshape=(None, self.n_runs, self.n_moves),
                        chunks=(min(256, n_new), self.n_runs, self.n_moves),
                    )
                    f.create_dataset(
                        "n_grad_evaluations",
                        data=grad_evals_new,
                        maxshape=(None, self.n_runs, self.n_moves),
                        chunks=(min(256, n_new), self.n_runs, self.n_moves),
                    )
            else:
                # Subsequent write: extend and fill
                n_old = f["iterations"].shape[0]
                f["iterations"].resize(n_old + n_new, axis=0)
                f["iterations"][n_old:] = iters_new

                f["step_sizes"].resize(n_old + n_new, axis=0)
                f["step_sizes"][n_old:] = ss_new

                f["acceptance_rates"].resize(n_old + n_new, axis=0)
                f["acceptance_rates"][n_old:] = acc_new

                if has_adj and "adjustment_stats" in f:
                    grp = f["adjustment_stats"]
                    for key, arr in adj_new.items():
                        if key in grp:
                            grp[key].resize(n_old + n_new, axis=0)
                            grp[key][n_old:] = arr

                if has_evals and "n_evaluations" in f:
                    f["n_evaluations"].resize(n_old + n_new, axis=0)
                    f["n_evaluations"][n_old:] = evals_new
                if has_evals and "n_grad_evaluations" in f:
                    f["n_grad_evaluations"].resize(n_old + n_new, axis=0)
                    f["n_grad_evaluations"][n_old:] = grad_evals_new

        # Clear buffers
        self._buf_iters.clear()
        self._buf_ss.clear()
        self._buf_acc.clear()
        for buf_list in self._buf_adj.values():
            buf_list.clear()
        self._buf_n_evals.clear()
        self._buf_n_grad_evals.clear()
