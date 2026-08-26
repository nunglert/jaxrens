"""Regenerate all static figures used by the docs.

Run from the repo root:
    python docs/_static/figures/generate.py

This just calls each group's own ``main()`` in turn. Each group is also a
standalone script and can be run (or imported) on its own:

    python docs/_static/figures/generate_concepts.py    # NS-loop / MWG /
                                                          # ensembles / batch
                                                          # shapes
    python docs/_static/figures/generate_treemap.py     # src/jaxrens package
                                                          # treemap
    python docs/_static/figures/generate_tutorials.py   # tutorial energy
                                                          # surfaces
"""

from __future__ import annotations

import generate_concepts
import generate_treemap
import generate_tutorials
from _common import FIGDIR

if __name__ == "__main__":
    generate_concepts.main()
    generate_treemap.main()
    generate_tutorials.main()
    print(f"wrote figures to {FIGDIR}")
