"""Sphinx configuration for jaxrens."""

from __future__ import annotations

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# nbsphinx calls `pandoc` via subprocess; the pypandoc_binary wheel
# installs the binary inside site-packages and doesn't add it to PATH,
# so point there explicitly before the build starts.
try:
    import pypandoc as _pypandoc  # type: ignore[import-not-found]
    _pandoc_path = _pypandoc.get_pandoc_path()
    _pandoc_dir = os.path.dirname(_pandoc_path)
    os.environ["PATH"] = os.pathsep.join([_pandoc_dir, os.environ.get("PATH", "")])
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
exclude_patterns = ["Thumbs.db", ".DS_Store", "_build", "test*.py"]

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
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_show_config = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False


# Notebook execution toggle. "auto" runs every tutorial at build time
# (gold standard — outputs can't drift); "never" renders the .ipynb as-is.
# Default to "never" so a broken in-flight refactor doesn't gate the docs
# build — flip via the JAXRENS_DOCS_EXECUTE env var when you want fresh
# tutorial outputs (e.g. JAXRENS_DOCS_EXECUTE=auto sphinx-build …).
nbsphinx_execute = os.environ.get("JAXRENS_DOCS_EXECUTE", "never")
nbsphinx_allow_errors = False


python_use_unqualified_type_names = True

language = "en"
html_static_path = ["_static"]
html_css_files: list[str] = ["mermaid-zoom.css"]
html_js_files: list[str] = ["mermaid-zoom.js"]

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/nunglert/jaxrens",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_show_sphinx = False
html_show_sourcelink = False
html_title = "jaxrens"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
