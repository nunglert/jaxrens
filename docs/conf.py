"""Sphinx configuration for jaxrens."""

from __future__ import annotations

import importlib.metadata
import os
import sys

import sphinx.util.logging

sys.path.insert(0, os.path.abspath("../src"))
# Local Sphinx extensions (docs/_ext/): yaml_schema renders the config
# specs as YAML-keyed tables for the configuration guide.
sys.path.insert(0, os.path.abspath("_ext"))

# nbsphinx calls `pandoc` via subprocess; the pypandoc_binary wheel
# installs the binary inside site-packages and doesn't add it to PATH,
# so point there explicitly before the build starts.
try:
    import pypandoc as _pypandoc  # type: ignore[import-not-found]

    _pandoc_path = _pypandoc.get_pandoc_path()
    _pandoc_dir = os.path.dirname(_pandoc_path)
    os.environ["PATH"] = os.pathsep.join(
        [_pandoc_dir, os.environ.get("PATH", "")]
    )
except Exception:
    pass


project = "jaxrens"
copyright = "2026, Nico Unglert"  # noqa: A001
author = "Nico Unglert"

version = importlib.metadata.version("jaxrens")
release = version


extensions = (
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "yaml_schema",
)

mermaid_output_format = "raw"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'default'});"
# Inline pan/zoom is provided by docs/_static/mermaid-zoom.{js,css}
# (visible toolbar + wheel/drag handlers). Fullscreen modal stays on
# (sphinxcontrib-mermaid v1.0+ default, "⛶" button per diagram).

# Register the ELK layout plugin so per-diagram `layout: elk` frontmatter
# actually engages the ELK engine (mermaid.live bundles it by default; the
# bare mermaid CDN doesn't). Without this flag, `layout: elk` is silently
# ignored and dagre is used, which is why the docs build differs from the
# live editor.
mermaid_include_elk = True
mermaid_elk_version = "0.2.0"

templates_path = ["_templates"]
# TODO.md is a working scratch list that lives beside the docs but is not
# one of them -- excluded so it stops raising "not included in any
# toctree" on every build.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "_build",
    "test*.py",
    "TODO.md",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_heading_anchors = 2
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "html_admonition",
    "html_image",
]


autodoc_mock_imports = [
    "mace_jax",
    "neuralil",
    "flax",
    "cuequivariance",
    "cuequivariance_torch",
    "cuequivariance_jax",
    "cuequivariance_ops_cu12",
    "cuequivariance_ops_jax_cu12",
    "cuequivariance_ops_torch_cu12",
    "torch",
]

autodoc_typehints = "description"
autodoc_preserve_defaults = True
autoclass_content = "class"
autodoc_member_order = "bysource"

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_preprocess_types = True


autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_show_config = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_show_validator_summary = False
autodoc_pydantic_settings_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False
# Render each field's ``Field(description=...)`` as its body text. Every
# schema field carries one (enforced by tests/cli/test_schema.py), and the
# same string is what ``jaxrens dump-schema`` emits and what pydantic quotes
# in validation errors — so the docs, the JSON schema, and the error messages
# all read from one source. The previous "docstring" policy rendered nothing,
# because the specs use ``Field(description=...)`` rather than attribute
# docstrings, which is why this page used to be a bare type dump.
autodoc_pydantic_field_doc_policy = "description"


# Notebook execution toggle. "auto" runs every tutorial at build time
# (gold standard — outputs can't drift); "never" renders the .ipynb as-is.
# Default to "never" so a broken in-flight refactor doesn't gate the docs
# build — flip via the JAXRENS_DOCS_EXECUTE env var when you want fresh
# tutorial outputs (e.g. JAXRENS_DOCS_EXECUTE=auto sphinx-build …).
nbsphinx_execute = os.environ.get("JAXRENS_DOCS_EXECUTE", "never")
nbsphinx_allow_errors = False
# Pygments lexer for code cells.  nbsphinx normally reads the language from the
# notebook's ``language_info`` metadata, but the tutorials are generated from
# ``examples/tutorials/*.py`` via jupytext, which strips notebook metadata — so
# without this the cells fall back to the ``'none'`` lexer (no highlighting).
# ``ipython3`` highlights Python plus IPython magics.
nbsphinx_codecell_lexer = "ipython3"


python_use_unqualified_type_names = True

language = "en"
html_static_path = ["_static"]
html_css_files: list[str] = ["mermaid-zoom.css", "yaml-schema.css"]
html_js_files: list[str] = ["mermaid-zoom.js"]

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/nunglert/jaxrens",
    "source_branch": "main",
    "source_directory": "docs/",
    # Vivid wordmark reads on both backgrounds; same file for both variants.
    "light_logo": "jaxrens_logo.svg",
    "dark_logo": "jaxrens_logo.svg",
    # Logo is a wordmark, so drop the redundant "jaxrens" text under it.
    "sidebar_hide_name": True,
}
html_favicon = "_static/favicon.png"
html_show_sphinx = False
html_show_sourcelink = False
html_title = "jaxrens"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    # ASE intentionally omitted: the docs moved to ase-lib.org and no
    # longer publish a reachable ``objects.inv`` (the old
    # wiki.fysik.dtu.dk URL 404s). Nothing in these docs uses ``:ase:``
    # cross-refs, so the mapping was dead weight that only produced a
    # "failed to reach inventory" warning on every build. Re-add with a
    # working inventory URL if ASE cross-refs are ever needed.
}


# ---------------------------------------------------------------------------
# autodoc cleanups (see setup())
# ---------------------------------------------------------------------------

_BUILTIN_DICT_DOC_MARKER = "dict() -> new empty dictionary"


def _clean_autodoc_docstring(app, what, name, obj, options, lines):
    """Fix two autodoc rendering artefacts that emit invalid reST.

    1. A module-level constant re-exported into another module (e.g.
       ``DEFAULT_SOFTCORE_KWARGS`` imported into ``backends.neuralil``) has no
       docstring in the importing module, so ``.. autodata::`` falls back to
       the builtin ``dict.__doc__`` — which is not valid reST and triggers
       "Unexpected indentation" / "Inline strong" warnings.  Replace it.

    2. jaxtyping shape annotations (``Float[Array, "*B N 3"]``) reach the
       napoleon-generated ``:type:`` / ``:vartype:`` / ``:rtype:`` fields with
       a bare ``*`` batch prefix, which docutils misreads as an unterminated
       emphasis marker.  Escape the asterisks in those type fields.

    Connected at priority > 500 so it runs *after* napoleon has converted the
    Google-style sections into field lists.
    """
    if (
        what == "data"
        and lines
        and _BUILTIN_DICT_DOC_MARKER in "\n".join(lines)
    ):
        lines[:] = [
            f"Re-exported constant; see :data:`{name}`.",
            "",
        ]
        return

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            stripped.startswith(":type ")
            or stripped.startswith(":vartype ")
            or stripped.startswith(":rtype:")
        ):
            lines[i] = line.replace("*", "\\*")


# ---------------------------------------------------------------------------
# Package treemap regeneration (see setup())
# ---------------------------------------------------------------------------


def _regen_treemap(app):
    """Rebuild ``_static/figures/pkg_treemap.{html,svg}`` from ``src/jaxrens``.

    The treemap is derived from the live package layout (module count, LoC,
    classes, functions), so a committed copy goes stale the moment a module is
    added or removed.  Regenerating it on every build keeps the API-reference
    landing figure honest; the walk is pure stdlib plus plotly/squarify and
    takes under a second, so it costs nothing.

    The two artefacts are gitignored — this hook is the only thing that
    produces them.  A failure is therefore a real problem (the ``<iframe>`` in
    ``reference/index.rst`` and ``user/introduction.md`` would 404), so it is
    reported as a Sphinx warning rather than swallowed.  The other figures
    (``generate_concepts.py``, ``generate_tutorials.py``) are synthetic /
    tutorial-derived and stay committed.
    """
    figdir = os.path.join(os.path.dirname(__file__), "_static", "figures")
    sys.path.insert(0, figdir)
    try:
        import generate_treemap  # type: ignore[import-not-found]

        generate_treemap.fig_pkg_treemap()
    except Exception as exc:  # pragma: no cover - build-time diagnostic
        logger = sphinx.util.logging.getLogger(__name__)
        logger.warning(
            "could not regenerate the package treemap (%s); "
            "reference/index and user/introduction will show a broken frame. "
            "Install the docs extra: pip install -e '.[docs]'",
            exc,
        )
    finally:
        sys.path.remove(figdir)


def setup(app):
    app.connect(
        "autodoc-process-docstring", _clean_autodoc_docstring, priority=600
    )
    app.connect("builder-inited", _regen_treemap)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
