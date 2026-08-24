"""Tiny, dependency-free ANSI styling helpers for the CLI.

Colour is emitted only when the stream is an interactive terminal, so
piped output, log files and pytest ``capsys`` captures stay plain ASCII
(the subcommand tests assert on raw substrings).  Honours the de-facto
``NO_COLOR`` / ``FORCE_COLOR`` conventions and ``TERM=dumb``.

This module imports nothing from JAX (and nothing heavy at all) so the
``--version`` / ``--help`` / ``dump-schema`` paths stay fast on a
GPU-less box.
"""

from __future__ import annotations

import os
import sys
from typing import IO

_CODES = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "grey": "90",
}


def supports_color(stream: IO[str] | None = None) -> bool:
    """Return whether *stream* (default: stdout) should receive ANSI codes."""
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def style(text: str, *names: str, stream: IO[str] | None = None) -> str:
    """Wrap *text* in the ANSI codes named in *names* (e.g. ``"bold"``).

    A no-op returning *text* unchanged when the target stream is not a
    colour-capable terminal.
    """
    if not names or not supports_color(stream):
        return text
    prefix = "".join(f"\x1b[{_CODES[n]}m" for n in names if n in _CODES)
    return f"{prefix}{text}\x1b[{_CODES['reset']}m"
