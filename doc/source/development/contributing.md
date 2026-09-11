# Contributing

Volumential is a research code with a small maintainer group, and most changes
land as pull requests against `main`.

## Before you start

- Set up an environment per {doc}`../getting-started/installation` and run the
  traversal sanity check. A contribution measured in an environment that fails
  that check cannot be evaluated.
- If the change touches numerics, decide up front which test tier it belongs in
  ({doc}`testing`) and whether it needs a benchmark row
  ({doc}`../benchmarks/index`).

## The loop

```bash
uvx ruff@0.13.0 check
uvx basedpyright -p pyproject.toml --level error
uv run pytest -q
sphinx-build -W --keep-going -b html doc/source doc/build/html   # if docs changed
```

Two of those are pull-request gates and two are not, and the difference matters
before you assume CI will catch something ({doc}`ci` has the full picture):

- `basedpyright` and the default pytest suite run on every pull request, as
  written above.
- `ruff` runs on every pull request, but only as `ruff check --select
  E9,F63,F7,F82` with an unpinned version — the error-level smoke subset, not
  the `ruff.toml` rule set. The pinned full check above is a local check; it can
  fail on baseline diagnostics that pull-request CI never looks at.
- The documentation build runs in `CI Full`, which has **no** `pull_request`
  trigger. A documentation change that breaks the `-W` build is not caught
  until it is on `main`. Run it locally.

## What a change should carry

- **Tests.** A behaviour change without a test that fails before it is not
  reviewable. For a new kernel, mode or derivative path, also update
  {doc}`../user-guide/validation_matrix` in the same pull request — a feature
  is not "covered" until there is at least one CI-friendly regression test, and
  for high-order numerical claims either a `full_accuracy` test or a documented
  benchmark and provenance path.
- **Documentation.** If the change alters something a user would otherwise
  discover by reading the source, it belongs on a page in this site. The layout
  table in {doc}`index` says which section.
- **Lint baseline.** If you cleaned a file, delete its entry from the
  `[lint.per-file-ignores]` block of `ruff.toml` in the same commit.
- **No vendored patches.** A patch that exists only inside one environment's
  `site-packages` is an incident to remediate, not a fix. Upstream it, put it
  on a tracked fork branch, or vendor and commit it.

## Commit and pull-request style

Commits use conventional prefixes — `feat:`, `fix:`, `perf:`, `refactor:`,
`docs:`, `ci:`, `chore:`, `deps:`, `bench:` — with an optional scope, as in
`fix(tables): make the batched-to-scalar Duffy fallback loud and recorded`.
Write the subject as a statement of what the change does, not of what it is
about.

Keep a pull request to one reviewable idea. The automated reviewers
({doc}`ci`) comment inline, and a branch that mixes four unrelated changes gets
four unrelated review threads on the same diff.

## Things that are easy to get wrong

- Passing the same array as both `src_weights` and `src_func` to
  `drive_volume_fmm`. The first is the density times the quadrature weights;
  the second is the bare density.
- Building a traversal with separate-but-identical source and target arrays.
  Use `targets=None`; `VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE=1` makes the
  mistake fail loudly.
- Quoting a timing from `--backend auto`, or from a cold process, without
  saying so. See {doc}`../benchmarks/index`.
- Assuming a cached table is the table you think it is. Check
  `build_routing`; see {doc}`../user-guide/table-build-routing`.
