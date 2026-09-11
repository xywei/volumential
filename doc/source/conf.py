"""Sphinx build configuration for the Volumential documentation.

Build with::

    sphinx-build -W --keep-going -b html doc/source doc/build/html

``volumential`` must be importable, so build inside an environment created as
described in ``DEVELOPMENT.md``.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path


# Make an uninstalled source checkout importable, so that the build works from
# a plain clone as well as from an installed environment.  This has to run
# before ``volumential`` is imported, hence the import below the statement.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from volumential.version import VERSION, VERSION_TEXT


# -- Project information --------------------------------------------------

project = "Volumential"
author = "Xiaoyu Wei"
copyright = f"{datetime.now(tz=UTC).year}, {author}"

# The short X.Y version, and the full version including alpha/beta/rc tags.
version = str(VERSION[0])
release = VERSION_TEXT


# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
templates_path = ["_templates"]
todo_include_todos = True

# The build is run with ``-W --keep-going``, so every cross-reference below
# has to resolve.  ``nitpick_ignore`` carries the exceptions.
nitpicky = True

nitpick_ignore = [
    # boxtree no longer documents ``DeviceDataRecord``, so the base class of
    # the interpolation lookups is absent from its inventory.
    ("py:class", "boxtree.tools.DeviceDataRecord"),
    # mpmath and pyfmmlib publish no objects.inv at all.
    ("py:mod", "mpmath"),
    ("py:mod", "pyfmmlib"),
]

nitpick_ignore_regex = [
    # numpy publishes ``ArrayLike`` as a ``py:data`` type alias, so the
    # annotation has no class target under any of the names autodoc renders
    # it with: the bare alias under Sphinx 8, a private ``numpy._typing``
    # path under Sphinx 9, at a depth that moves with the numpy version.
    ("py:class", r"([\w.]+\.)?ArrayLike"),
]


# -- MyST (Markdown) ------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3


# -- autodoc / autosummary ------------------------------------------------

# Generate a stub page per module from ``_templates/autosummary``.
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # Document every object on the page of the module that defines it: a
    # re-export shim such as ``volumential.expansion_wrangler_fpnd`` lists
    # its names in ``__all__``, and without this they would be documented a
    # second time there.
    "ignore-module-all": True,
    # Module-level ``logger`` objects are an implementation detail.
    "exclude-members": "logger",
}
autodoc_member_order = "bysource"

# Optional backends are not installed in every doc environment; autodoc must
# not fail on them.
autodoc_mock_imports = [
    "gmsh",
    "gmsh_interop",
    "pyfmmlib",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# -- intersphinx ----------------------------------------------------------

intersphinx_mapping = {
    "arraycontext": ("https://documen.tician.de/arraycontext", None),
    "boxtree": ("https://documen.tician.de/boxtree", None),
    "loopy": ("https://documen.tician.de/loopy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "meshmode": ("https://documen.tician.de/meshmode", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "pytential": ("https://documen.tician.de/pytential", None),
    "python": ("https://docs.python.org/3", None),
    "pytools": ("https://documen.tician.de/pytools", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sumpy": ("https://documen.tician.de/sumpy", None),
}


# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Volumential"
html_static_path = ["_static"]
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "github_url": "https://github.com/xywei/volumential",
    "navbar_align": "left",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "show_toc_level": 2,
    # Only for the pages that have a source file in the repository; see
    # ``_disable_edit_button_on_generated_pages`` below.
    "use_edit_page_button": True,
    "switcher": {
        # Served from the site itself: ``_static/switcher.json`` carries a
        # single "latest" entry until tagged versions are published.
        "json_url": "_static/switcher.json",
        "version_match": "latest",
    },
}

html_context = {
    "github_user": "xywei",
    "github_repo": "volumential",
    "github_version": "main",
    "doc_path": "doc/source",
    # Follow the reader's system preference; the navbar toggle overrides it.
    "default_mode": "auto",
}


_GENERATED_API_PREFIX = "api/generated/"


def _disable_edit_button_on_generated_pages(
    app, pagename, templatename, context, doctree
):
    """Hide "Edit this page" where there is no source file to edit.

    ``sphinx.ext.autosummary`` writes the API pages at build time and they are
    not committed, so an edit link into the repository would be a dead link.
    """
    if pagename.startswith(_GENERATED_API_PREFIX):
        context["theme_use_edit_page_button"] = False


def setup(app):
    app.connect("html-page-context", _disable_edit_button_on_generated_pages)


# -- Options for the link checker -----------------------------------------

linkcheck_timeout = 30
linkcheck_retries = 2
linkcheck_ignore = [
    # Not published yet; the GitHub Pages deployment lands in a later phase.
    r"https://xywei\.github\.io/volumential/?.*",
]


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

latex_documents = [
    (
        root_doc,
        "Volumential.tex",
        "Volumential Documentation",
        author,
        "manual",
    ),
]
