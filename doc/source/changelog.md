# Changelog

Volumential has no Git tags yet and is pre-release
(`Development Status :: 3 - Alpha`), so there is nothing to organize this list
by except merged pull requests. Each entry links to the pull request, which
carries the review discussion and the full diff. See {doc}`development/releases`
for what will have to change when tagging starts.

## Unreleased

### September 2026

Documentation
: [#146](https://github.com/xywei/volumential/pull/146) — rebuild the
  documentation toolchain: `pydata-sphinx-theme`, MyST Markdown,
  `sphinx-copybutton`, `sphinx-design`, a recursive `autosummary` API tree, and
  a `nitpicky` build with warnings as errors and a link check.
: [#144](https://github.com/xywei/volumential/pull/144) — acknowledge
  AI-assisted development and automated review
  ({doc}`development/ai-assisted-development`).

Evidence and benchmarks
: [#143](https://github.com/xywei/volumential/pull/143) — Paper 1 evidence
  library, drivers, table-builder fixes and modernization.
: [#134](https://github.com/xywei/volumential/pull/134) — per-phase operation
  and time shares of end-to-end solves.
: [#133](https://github.com/xywei/volumential/pull/133) — Helmholtz and
  windowed-assembled rows for the 3D composition benchmark.
: [#132](https://github.com/xywei/volumential/pull/132) — 3D split-parameter
  sweep with a resolved FMM-order rule.

Tables and numerics
: [#139](https://github.com/xywei/volumential/pull/139) — emit `cos`/`sin`
  instead of `cdouble_exp` in the fused Duffy program, about a tenfold speedup
  on complex kernels ({doc}`user-guide/table-build-routing`).
: [#137](https://github.com/xywei/volumential/pull/137) — make the
  batched-to-scalar Duffy fallback loud and recorded, with a strict mode that
  also refuses cached tables of unverifiable provenance.
: [#131](https://github.com/xywei/volumential/pull/131) — windowed RKE table
  assembly: certified assembly of fixed-parameter tables from a
  parameter-independent windowed channel family
  ({doc}`design-notes/windowed-channels`).

Infrastructure
: [#141](https://github.com/xywei/volumential/pull/141) — restore CI on `main`
  (Python 3.12 test environments, `loopy` bounds check).
: [#140](https://github.com/xywei/volumential/pull/140) — modernize the
  codebase: six reviewed packages, the `volumential.wranglers` split, and
  `ruff.toml` as the single lint configuration.
: [#135](https://github.com/xywei/volumential/pull/135) — track upstream
  `boxtree` and `pyfmmlib` `main`, so the `fmmlib` extra resolves through
  `uv.lock` rather than to a release that lacks OpenMP and the batched
  wrappers ({doc}`getting-started/installation`).

## Earlier

Before September 2026 the project's history lives in the pull requests and
commits themselves; `git log` is the reference. The pieces of it that still
shape the current design are described in {doc}`design-notes/index` and in the
user guide rather than reconstructed here.
