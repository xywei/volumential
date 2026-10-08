# Release and versioning

## Where the version lives

`volumential/version.py` owns two things:

- `VERSION` / `VERSION_TEXT`, the package version. `pyproject.toml` currently
  carries it as a literal (`2017.1a0`), and the documentation build reads the
  module, so the two have to be changed together.
- `KERNEL_VERSION`, the revision token mixed into the cache keys of generated
  kernels so that cached {mod}`loopy` binaries are not reused across source
  changes. It is `(VERSION, <git revision>, 0)` — derived, not hand-maintained,
  with a deterministic fingerprint as the fallback for artifacts installed
  outside a Git checkout. It is not the package version and does not track
  releases.
- `LOOPY_LANG_VERSION`, the loopy language version the generated kernels
  declare.

Related invalidation knobs that are *not* the version: a table's build
configuration is hashed into the table-cache fingerprint, so changing a
`DuffyBuildConfig` field invalidates cached tables by itself — which is exactly
why the build-routing strictness switch of
{doc}`../user-guide/table-build-routing` is an environment variable instead.
A fix that changes the values a DuffyRadial builder produces bumps
`DUFFY_BUILDER_REVISION` in `volumential/nearfield_potential_table.py`, not
the table-cache schema version, which describes the layout of the cache file.
Every table records the revision it was built at, and the table manager's
loader reads the cached tables the fix affects, at an older revision or none,
as cache misses, so those are rebuilt and the rest of the cache stays valid.

## Status

The project is pre-release (`Development Status :: 3 - Alpha`) and has no Git
tags yet. Practically, "the current version" means `main`, and the change log
is by merged pull request: {doc}`../changelog`.

The documentation is configured `latest`-only.
`doc/source/_static/switcher.json` carries a single `latest` entry and the
theme's version switcher is wired to it, so publishing tagged versions later is
a matter of adding entries rather than of changing the build. That entry points
at the GitHub Pages site, <https://xywei.github.io/volumential/>, which
`.github/workflows/docs-pages.yml` deploys from `main` (see {doc}`ci`). Until a
first tag is published the switcher lists that one entry and nothing else.

## Releases are Git tags

A release is a tag `v<version>`, and it is installed from GitHub:

```bash
pip install "volumential @ git+https://github.com/xywei/volumential@v<version>"
```

It is not on PyPI. PyPI rejects any distribution whose metadata has a direct
reference (`name @ git+https://...`), pinned or not, in an extra or not, and
Volumential's dependencies on the `inducer` stack have to be such references:
`pytential` has no installable release on PyPI, and `sumpy`, `boxtree`,
`meshmode` and `arraycontext` only years-old ones
([#211](https://github.com/xywei/volumential/issues/211)). `pyproject.toml`
pins each of them to a commit, so a tag fixes the whole dependency set that
matters, and an install of the tag gets the commits CI tested at that tag
({doc}`../getting-started/installation`).

Pushing the tag runs `.github/workflows/publish.yml`. Its build job checks that
the tag, `pyproject.toml` and `volumential.version` agree, builds the sdist
and the wheel, and checks them with `twine check --strict`; its release job,
the only one with write access to the repository, creates the GitHub Release
of the tag with both files attached. The PyPI project keeps a Trusted
Publisher for that workflow, unused until the `inducer` packages are released
on PyPI and the dependencies can be plain names again. `DEVELOPMENT.md` has
the steps.

## When tagging starts

The pieces that have to move together:

1. `volumential/version.py` and the `version` field of `pyproject.toml`. The
   suggestion for the version itself is a calendar version, `<year>.<n>`, as
   the `inducer` packages use.
2. A Git tag on the release commit.
3. A new entry in `doc/source/_static/switcher.json`, with the previous
   `latest` gaining a version-specific URL.
4. A section in {doc}`../changelog`.

Until then, anything that needs to identify a build should identify a commit —
which is what {doc}`../benchmarks/index` requires of a measurement anyway, and
why promoted evidence pins the generating commit and the pinned dependency
commits rather than a version string.
