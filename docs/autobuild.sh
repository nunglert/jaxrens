# sphinx-autobuild docs docs/_build/html \
# --open-browser \
# --watch src/jaxrens \          # rebuild when you edit docstrings, not just docs/
# --ignore "**/_build/**" \      # don't loop on its own output (usually auto-ignored)
# --ignore "**/.ipynb_checkpoints/**" \
#
# No -j: sphinx's parallel build forks worker processes, and by the time it
# forks, autodoc has already imported jaxrens (-> JAX) into the main process.
# JAX is multithreaded, so forking after that point is unsafe (deadlock risk,
# and prints "os.fork() was called ... this will likely lead to a deadlock").
#
# --ignore the treemap outputs: docs/conf.py's builder-inited hook
# regenerates pkg_treemap.{html,svg} under docs/_static/figures/ on every
# build. Without excluding them, the watcher sees its own output change and
# rebuilds again -- and again -- looping forever even when nothing else has
# changed.
# Run from the repo root (as documented) so these relative paths resolve
# correctly -- sphinx-autobuild matches --ignore patterns against the
# invoking CWD, not against `docs/`.
sphinx-autobuild docs docs/_build/html \
  --ignore "docs/_static/figures/pkg_treemap.html" \
  --ignore "docs/_static/figures/pkg_treemap.svg"
