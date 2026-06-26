"""Shared lifecycle for the append-only buffered HDF5 log writers.

The per-iteration log writers (acceptance rates, RE swap stats, max-neighbor
counts, step-size adaptation) all buffer entries in memory and periodically
flush them to an append-only HDF5 file, creating the file lazily on the first
flush and honoring restart truncation.  That common buffer/flush/close
lifecycle lives here; each subclass supplies its own payload buffers,
``_coerce`` validation, and ``_flush`` dataset layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class BufferedH5Logger:
    """Buffer/flush/close scaffolding shared by the buffered HDF5 loggers.

    Subclasses must:

      * call ``super().__init__(...)`` and then set up their own payload
        buffers and config attributes;
      * implement :meth:`_flush` (using :meth:`_open_flush_file` to obtain the
        file handle); and
      * in ``write_entry``, append to ``self._buf_iters`` and their payload
        buffers, then trigger a flush — either via :meth:`_maybe_flush` for
        iteration-interval cadence, or their own condition.

    The constructor seeds ``self._buf_iters`` (the shared iteration buffer),
    the flush watermark, and the ``_closed`` flag, and applies restart
    truncation when reopening in append mode.
    """

    def __init__(
        self,
        path: Path | str,
        flush_interval: int,
        mode: str = "w",
        restart_iteration: int = 0,
    ) -> None:
        self.path = Path(path)
        self.flush_interval = max(1, int(flush_interval))
        self._mode = mode
        # First flush honors ``mode`` (so mode="w" truncates a stale file);
        # subsequent flushes always append.
        self._first_flush = True
        # Restart: rewind entries flushed past the checkpoint before appending.
        if mode == "a" and restart_iteration > 0:
            from jaxrens.io.restart_truncate import truncate_h5_iterations

            truncate_h5_iterations(self.path, restart_iteration)

        self._buf_iters: list[int] = []
        self._last_flush_iter: int | None = None
        self._closed = False

    def _maybe_flush(self, iter_int: int) -> None:
        """Flush once the iteration index has advanced by ``flush_interval``.

        Tracks an absolute-iteration watermark; the first recorded iteration
        seeds it without flushing.
        """
        if self._last_flush_iter is None:
            self._last_flush_iter = iter_int
        elif iter_int - self._last_flush_iter >= self.flush_interval:
            self._flush()
            self._last_flush_iter = iter_int

    def _open_flush_file(self) -> Any:
        """Open the HDF5 file for a flush, creating parent dirs.

        The first flush uses the constructor ``mode`` (``"w"`` truncates any
        stale file); every later flush appends.  Returns an open
        ``h5py.File`` — use it as a context manager.
        """
        import h5py

        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = self._mode if self._first_flush else "a"
        self._first_flush = False
        return h5py.File(self.path, mode)

    def close(self) -> None:
        """Flush any buffered entries and mark the logger closed.

        If no entries were ever written, no file is created.
        """
        if not self._closed:
            if self._buf_iters:
                self._flush()
            self._closed = True

    def _flush(self) -> None:  # pragma: no cover - subclass responsibility
        raise NotImplementedError
