"""Sphinx build configuration for the Volumential documentation.

Build with ``make -C doc html`` (or ``sphinx-build -W -b html doc/source
<outdir>`` to treat warnings as errors).  ``volumential`` must be importable,
so build inside an environment created as described in ``DEVELOPMENT.md``.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

from volumential.version import VERSION, VERSION_TEXT


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    "boxtree": ("https://documen.tician.de/boxtree", None),
    "loopy": ("https://documen.tician.de/loopy", None),
    "meshmode": ("https://documen.tician.de/meshmode", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "python": ("https://docs.python.org/3", None),
    "pytools": ("https://documen.tician.de/pytools", None),
    "sumpy": ("https://documen.tician.de/sumpy", None),
}

# Optional backends are not installed in every doc environment; autodoc must
# not fail on them.
autodoc_mock_imports = [
    "gmsh",
    "gmsh_interop",
    "pyfmmlib",
]

source_suffix = ".rst"
master_doc = "index"

project = "Volumential"
copyright = f"{datetime.now(tz=UTC).year}, Xiaoyu Wei"
author = "Xiaoyu Wei"

# The short X.Y version, and the full version including alpha/beta/rc tags.
version = str(VERSION[0])
release = VERSION_TEXT

language = "en"
exclude_patterns = []
suppress_warnings = ["docutils"]
pygments_style = "sphinx"
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

latex_documents = [
    (
        master_doc,
        "Volumential.tex",
        "Volumential Documentation",
        author,
        "manual",
    ),
]
