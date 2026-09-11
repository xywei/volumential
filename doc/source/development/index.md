# Development

```{toctree}
:maxdepth: 1

contributing
testing
ci
releases
ai-assisted-development
```

`DEVELOPMENT.md` at the repository root is the canonical environment guide and
stays that way: it is what a maintainer reads in a terminal, before the
documentation is buildable. This section is everything *around* it — how to
contribute, what the test tiers are, what CI and the review bots do, and how
versioning works.

## Environment

Set up with {doc}`../getting-started/installation`, then read `DEVELOPMENT.md`
for the parts that only matter once you are producing evidence:

- the dependency-provisioning rules (inducer stack from Git sources,
  `pyfmmlib` with OpenMP and the batched P2M wrappers, the post-provisioning
  traversal sanity check, and the thread caps to record in run metadata);
- remote setup for heavier numerical experiments;
- lint, type-check and test commands.

## Lint and types

`ruff.toml` at the repository root is the single lint configuration — 85
columns, Python 3.11 target — and it replaced the former
`[flake8]`/`[isort]`/`[pycodestyle]` sections of `setup.cfg`.

```bash
uvx ruff@0.13.0 check
uvx ruff@0.13.0 check --fix          # fixable rules only; re-read every hunk
uvx basedpyright -p pyproject.toml --level error
```

Two conventions are worth knowing before the first pull request.

**The `[lint.per-file-ignores]` block is a baseline, not a preference.** It is
a per-file record of the rules a file still violates, measured rather than
hand-written: exactly the set each file trips with the block emptied. New files
start clean. When you clean a file, delete its baseline entry in the same
commit. Entries appear and disappear as the tree changes, so never edit the
block by hand — re-measure it.

**`ruff format` is not a gate.** Adopting it would reformat 92 of 117 files
(about 5.7k changed lines), and enabling `flake8-quotes` (`Q`) would rewrite
448 single-quoted literals. Either is a tree-wide rewrite that collides with
every open branch, so the `[format]` section of `ruff.toml` records the
intended style for a future coordinated reformat and nothing enforces it today.

## Documentation

The site is built with Sphinx and `pydata-sphinx-theme`; the `doc` extra
carries everything it needs. The build imports `volumential`, so it needs an
environment with the OpenCL stack (`pyopencl`, `loopy`) installed.

```bash
# Name every extra you want: uv sync is exact, so --extra doc alone would
# uninstall pytest and the rest of the test extra.
uv sync --active --extra test --extra doc

# The build CI Full runs: -W makes every warning an error, and --keep-going
# reports all of them instead of stopping at the first.  No warning class is
# suppressed in conf.py, so a malformed docstring fails the build like a bad
# cross-reference does.
sphinx-build -W --keep-going -b html doc/source doc/build/html

# External links.
sphinx-build -b linkcheck doc/source doc/build/linkcheck

# Live preview at http://127.0.0.1:8000, rebuilding on save.
sphinx-autobuild doc/source doc/build/html
```

### Writing pages

New pages are MyST Markdown (`.md`). The reStructuredText pages that remain are
substantial existing documents kept where they are rather than rewritten;
either format is read by the same build, and `myst_enable_extensions` turns on
`amsmath`, `colon_fence`, `deflist` and `dollarmath`, so `$...$` and `$$...$$`
math works in Markdown.

The three constructs worth knowing, since they are the ones that differ from
plain Markdown:

````markdown
```{note}
A directive. Any Sphinx directive works with this fence.
```

{doc}`../user-guide/index` and {mod}`volumential.volume_fmm` are roles.

```{toctree}
:maxdepth: 1

some-page
```
````

A page reaches the sidebar by being in a `toctree`, and a page that is in none
is a build warning — which, under `-W`, is a build failure.

### Where the API pages come from

`doc/source/api/` is generated. `sphinx.ext.autosummary` writes one page per
module from the templates in `doc/source/_templates/autosummary/`, so a new
module needs no edit there; the generated tree is not committed.

The build is `nitpicky`, which means an unresolvable cross-reference in a
docstring fails it. When the name belongs to a dependency, add an intersphinx
target in `doc/source/conf.py`. Add a `nitpick_ignore` entry — with a comment
saying why — only when no usable inventory target exists: either the project
publishes no `objects.inv` at all (`mpmath`, `pyfmmlib`), or it publishes one
that does not document the referenced object (`boxtree` no longer documents
`boxtree.tools.DeviceDataRecord`, though its inventory is otherwise fine).

### Redirect stubs for the old flat URLs

Before the 2026-09 restructure every page lived directly under the site root,
and the published build is replaced in place, so an external link to
`/nearfield_symmetry.html` would 404 against the new tree. `conf.py` writes a
meta-refresh stub at each of those old paths on `build-finished`; the mapping
is `_LEGACY_REDIRECTS`, it never shadows a real page, and an entry can be
deleted once its inbound links have aged out. Moving a page again means adding
an entry there in the same commit.

## Documentation layout

The site follows a Diátaxis-style split, and a new page belongs in exactly one
of these:

| Section | For | Example |
| --- | --- | --- |
| Getting started | A reader who has not run the code yet | installing, a first potential |
| User guide | Understanding a mechanism you are using | the Helmholtz split |
| Design notes | Why a mechanism has its shape; no derivations | windowed channels |
| Benchmarks | Producing and promoting evidence | metadata sidecars |
| API reference | Generated; edit the docstring instead | — |
| Development | The project, not the library | this page |

When you add a kernel, a mode or a derivative path, update
{doc}`../user-guide/validation_matrix` in the same pull request that adds the
tests.
