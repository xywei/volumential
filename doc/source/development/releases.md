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

## Status

The project is pre-release (`Development Status :: 3 - Alpha`) and has no Git
tags yet. Practically, "the current version" means `main`, and the change log
is by merged pull request: {doc}`../changelog`.

The documentation is configured `latest`-only.
`doc/source/_static/switcher.json` carries a single `latest` entry and the
theme's version switcher is wired to it, so publishing tagged versions later is
a matter of adding entries rather than of changing the build. That entry points
at the GitHub Pages URL, which does not serve the site yet — the deployment is
the last phase of
[#145](https://github.com/xywei/volumential/issues/145), and until it lands the
published build is the one at <https://xiaoyu-wei.com/docs/volumential/>. The
switcher is forward configuration, not a live index.

## When tagging starts

The pieces that have to move together:

1. `volumential/version.py` and the `version` field of `pyproject.toml`.
2. A Git tag on the release commit.
3. A new entry in `doc/source/_static/switcher.json`, with the previous
   `latest` gaining a version-specific URL.
4. A section in {doc}`../changelog`.

Until then, anything that needs to identify a build should identify a commit —
which is what the run metadata of {doc}`../benchmarks/index` already does, and
why promoted evidence pins the generating commit and the locked dependency
commits rather than a version string.
