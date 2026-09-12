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
uv sync --extra test --extra doc

# The build CI Full runs: -W makes every warning an error, and --keep-going
# reports all of them instead of stopping at the first.  No warning class is
# suppressed in conf.py, so a malformed docstring fails the build like a bad
# cross-reference does.
sphinx-build -W --keep-going -b html doc/source doc/build/html

# External links.
sphinx-build -b linkcheck doc/source doc/build/linkcheck

# Live preview at http://127.0.0.1:8000, rebuilding on save.  --watch is what
# picks up an edit to a notebook: they live outside doc/source, and the
# staging copy in conf.py only runs when a build starts.
sphinx-autobuild --watch examples doc/source doc/build/html
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

`doc/source/api/generated/` is generated — and only that subdirectory.
`sphinx.ext.autosummary` writes one page per module there from the templates in
`doc/source/_templates/autosummary/`, so a new module needs no edit and the
tree is not committed. `doc/source/api/index.rst` above it *is* committed and
hand-written: it holds the autosummary seed, the module map and the note about
the unsupported `volumential.qbfem` package, so a change to how the API is
introduced goes there.

The build is `nitpicky`, which means an unresolvable cross-reference in a
docstring fails it. When the name belongs to a dependency, add an intersphinx
target in `doc/source/conf.py`. Add a `nitpick_ignore` entry — with a comment
saying why — only when no usable inventory target exists: either the project
publishes no `objects.inv` at all (`mpmath`, `pyfmmlib`), or it publishes one
that does not document the referenced object (`boxtree` no longer documents
`boxtree.tools.DeviceDataRecord`, though its inventory is otherwise fine).

### Example notebooks

The notebooks are maintained in `examples/`, beside the scripts they
demonstrate. Sphinx reads only what is under `doc/source`, so `conf.py` copies
`examples/*.ipynb` into `doc/source/examples/notebooks/` on `builder-inited`;
that directory is generated and git-ignored, like `api/generated/`. Add a
notebook to `examples/` and it gets a page, because
{doc}`../examples/index` globs the staged directory — but add the paragraph
that says what it costs to run, since a reader cannot tell from the rendering.

Nothing is executed: `nb_execution_mode = "off"`. A notebook needs an OpenCL
device and, at its committed settings, more time than a documentation build
has, so a page shows the prose, the code and whatever outputs the notebook
carries in the repository — today, none. Commit them stripped. Above 2 MB the
staged copy drops the outputs anyway rather than shipping them into the page;
the file in `examples/` is never modified.

### Docstring and API coverage

Two different questions, and CI answers both in the `Documentation` job of
`CI Full`, uploading the answers as a `docs-coverage-*` artifact.

`sphinx-build -q -W --keep-going -b coverage` asks whether every module,
function, class and method of the package reaches a page of this site. `-q` is
load-bearing rather than tidiness: `sphinx.ext.coverage` logs an undocumented
*object* at info level unless the app is quiet, in which case it logs a
warning — and only a warning is something `-W` fails on.

It does not see properties: the builder inspects a class attribute only when
it is a method or a function, so a `@property` that fell off a page would go
unreported. Nothing here documents a property anywhere but on its class's
page, so the gap has no reach today; know about it before relying on the
report for a new kind of page. It is at 100% and should
stay there. `coverage_modules` in `conf.py` is what makes it a real check:
without it the builder looks only at the modules it already saw on a page, so a
module that fell out of the autosummary tree would not be examined at all and
the total would stay at 100%. With it, a module in the package but not on a
page — and a module on a page but not in the package — is a warning, which
under `-W` fails the job.

It is *not* a docstring check. `undoc-members` is what puts the whole public
surface on the API pages, and an object with no docstring still gets an entry
there and still counts as covered.

`doc/tools/docstring_gaps.py` asks whether every public object *has* a
docstring. It parses the tree with `ast`, so it needs no OpenCL stack, no
import and no Sphinx, and runs anywhere:

```bash
python doc/tools/docstring_gaps.py --max-gaps 5
```

`--max-gaps` is the ratchet, and that is the number CI passes, so the command
above gives the same verdict CI does; drop it to read the report without a
verdict. Lower the recorded number in the same commit that lowers the count;
raise it only deliberately, and say why. (CI also passes `--output`, which
only names the file inside the artifact.)

### Sitemap and social metadata

`html_baseurl` is the GitHub Pages URL the site is heading for, and three
things read it: Sphinx writes a `canonical` link per page, `sphinx-sitemap`
writes `sitemap.xml`, and `sphinxext.opengraph` writes `og:url`. The sitemap
uses the flat `{link}` scheme, since the site publishes `latest` only, and
leaves out `genindex`, `py-modindex` and `search`.

There is deliberately no `robots.txt`. A crawler reads the robots policy from
the origin root only — `https://xywei.github.io/robots.txt` — which belongs to
the user site, not to this project's build output, so a file shipped at
`/volumential/robots.txt` would never be read. The same subpath applies to the
sitemap, which is why it has to be submitted by URL rather than advertised:
it is served at `https://xywei.github.io/volumential/sitemap.xml`, not at the
origin root.

Social-card images are off (`ogp_social_cards`): generating one per page needs
matplotlib and a bundled font. The landing page sets its own description in
front matter, because the extension derives one by walking the doctree and that
page opens with display math.

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
| Examples | What each program under `examples/` does and costs | the gallery |
| User guide | Understanding a mechanism you are using | the Helmholtz split |
| Design notes | Why a mechanism has its shape; no derivations | windowed channels |
| Benchmarks | Producing and promoting evidence | metadata sidecars |
| API reference | `api/generated/` is generated — edit the docstring; `api/index.rst` is the hand-written overview | the module map |
| Development | The project, not the library | this page |

When you add a kernel, a mode or a derivative path, update
{doc}`../user-guide/validation_matrix` in the same pull request that adds the
tests.
