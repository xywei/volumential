"""Sphinx build configuration for the Volumential documentation.

Build with::

    sphinx-build -W --keep-going -b html doc/source doc/build/html

``volumential`` must be importable, so build inside an environment created as
described in ``DEVELOPMENT.md``.
"""

import html
import json
import posixpath
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


# -- Coverage of the API by the site --------------------------------------

# ``sphinx-build -b coverage`` answers one question: is every object of every
# imported module reachable from a page of this site?  The ``Documentation``
# job of ``.github/workflows/ci-full.yml`` runs it and uploads
# ``doc/build/coverage/``, so a module that never reaches a page becomes
# visible instead of staying silent.
#
# It is deliberately *not* a docstring check.  ``autodoc_default_options`` sets
# ``undoc-members``, so an object with no docstring still gets an entry and
# still counts as covered here.  ``doc/tools/docstring_gaps.py`` is the
# docstring half of the same question, and CI runs it beside this builder.
coverage_ignore_modules = [
    # A 2019 finite-element experiment that nothing in the tree imports; the
    # autosummary template leaves it out of the API reference too.
    r"volumential\.qbfem",
]
coverage_show_missing_items = True
coverage_write_headline = True
coverage_statistics_to_report = True
coverage_statistics_to_stdout = True


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


# -- Redirects for the pre-Diataxis flat layout ---------------------------

# Until the 2026-09 restructure every page lived directly under the site root,
# and the published build is replaced in place, so a bookmark or an external
# link to one of those flat URLs would 404 against the new tree.  Emit a
# meta-refresh stub at each old path instead.  These are cheap and removable:
# delete an entry once its inbound links have aged out.
_LEGACY_REDIRECTS = {
    # Moved under a section.
    "derivative_support.html": "user-guide/derivative_support.html",
    "helmholtz_split.html": "user-guide/helmholtz_split.html",
    "m1_kernels.html": "user-guide/m1_kernels.html",
    "nearfield_symmetry.html": "user-guide/nearfield_symmetry.html",
    "validation_matrix.html": "user-guide/validation_matrix.html",
    # Superseded; sent to the page that took over the content.
    "api/modules.html": "api/index.html",
    "development.html": "development/index.html",
    "indices_tables.html": "api/index.html",
    "install.html": "getting-started/installation.html",
    "intro.html": "index.html",
    "sphinx.html": "development/index.html",
}

# The script carries ``location.hash`` across, so a deep link into one of
# the long moved pages (``helmholtz_split.html#automatic-regime-planner``)
# lands on its section rather than at the top; the meta refresh is the
# no-JavaScript fallback and loses the fragment.
_REDIRECT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="1; url={target}">
    <link rel="canonical" href="{target}">
    <title>Page moved</title>
    <script>
      window.location.replace({target_js} + window.location.hash);
    </script>
  </head>
  <body>
    <p>This page has moved to <a href="{target}">{target}</a>.</p>
  </body>
</html>
"""


def _write_legacy_redirects(app, exception):
    """Write a meta-refresh stub at every pre-restructure URL."""
    if exception is not None or app.builder.name != "html":
        return

    out_dir = Path(app.outdir)
    # Ownership comes from the *current* document set, never from what is on
    # disk: an incremental build into an output directory that still holds a
    # removed page's HTML would otherwise skip that page's redirect and leave
    # the stale content served.
    current_pages = {f"{docname}.html" for docname in app.env.found_docs}
    for source, target in _LEGACY_REDIRECTS.items():
        if source in current_pages:
            # A real page owns this path now; never shadow it.
            continue
        stub = out_dir / source
        stub.parent.mkdir(parents=True, exist_ok=True)
        href = posixpath.relpath(target, posixpath.dirname(source) or ".")
        stub.write_text(
            _REDIRECT_TEMPLATE.format(
                target=html.escape(href, quote=True),
                target_js=json.dumps(href),
            ),
            encoding="utf-8",
        )


def setup(app):
    app.connect("html-page-context", _disable_edit_button_on_generated_pages)
    app.connect("build-finished", _write_legacy_redirects)


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
