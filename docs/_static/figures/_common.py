"""Shared setup for the docs figure-generation scripts.

Not runnable on its own — imported by ``generate_concepts.py``,
``generate_treemap.py``, and ``generate_tutorials.py`` for the output
directory, the shared matplotlib style, and the LoC counter the treemap
walk needs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parent

# Chunk size for jax.lax.map(..., batch_size=...) in the grid-evaluation
# figures: bounds peak memory to one chunk's worth of intermediates instead
# of materializing all n*n grid points' worth at once (which OOMs for the
# larger backends), while still batching each chunk through a single vmap'd
# call rather than dispatching point by point.
GRID_BATCH = 4096

plt.rcParams.update(
    {
        "figure.figsize": (5.0, 3.2),
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)


def count_loc(path: Path) -> int:
    """Non-blank, non-comment LoC in a Python file."""
    n = 0
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n
