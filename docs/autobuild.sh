# sphinx-autobuild docs docs/_build/html \
# --open-browser \
# --watch src/jaxrens \          # rebuild when you edit docstrings, not just docs/
# --ignore "**/_build/**" \      # don't loop on its own output (usually auto-ignored)
# --ignore "**/.ipynb_checkpoints/**" \
# -j auto                        # parallel build; nbsphinx notebook execution is slow
sphinx-autobuild docs docs/_build/html -j auto  