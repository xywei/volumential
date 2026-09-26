# Changelog

Volumential has no Git tags yet and is pre-release
(`Development Status :: 3 - Alpha`), so there is nothing to organize this list
by except merged pull requests. Each entry links to the pull request, which
carries the review discussion and the full diff. See {doc}`development/releases`
for what will have to change when tagging starts.

## Unreleased

### September 2026

Documentation
: [#152](https://github.com/xywei/volumential/pull/152) — build, link-check
  and preview the site on every pull request against `main` through a
  `Documentation` CI job,
  add the GitHub Pages deployment workflow, and retire the GitLab documentation
  job ({doc}`development/ci`).
: [#149](https://github.com/xywei/volumential/pull/149) — content hygiene: a
  gallery page for the example scripts, the notebooks rendered without
  execution, a sitemap and OpenGraph metadata, docstrings on six modules, API-page
  coverage reported by `sphinx.ext.coverage`, and docstring coverage
  ratcheted by `interrogate`.
: [#148](https://github.com/xywei/volumential/pull/148) — information
  architecture: the Diátaxis layout of this site, redirects for the old page
  URLs, and `README.md` reduced to a landing page (reviewed as
  [#147](https://github.com/xywei/volumential/pull/147), re-landed on `main`
  as #148).
: [#146](https://github.com/xywei/volumential/pull/146) — rebuild the
  documentation toolchain: `pydata-sphinx-theme`, MyST Markdown,
  `sphinx-copybutton`, `sphinx-design`, a recursive `autosummary` API tree, and
  a `nitpicky` build with warnings as errors and a link check.
: [#144](https://github.com/xywei/volumential/pull/144) — acknowledge
  AI-assisted development and automated review
  ({doc}`development/ai-assisted-development`).

Evidence and benchmarks
: [#163](https://github.com/xywei/volumential/pull/163) — move the benchmark
  drivers, the two split-order convergence sweeps and their driver-only tests
  out to the manuscript repository, which pins the Volumential revision it
  measured; `7c75ed1` is the last revision that carries them in this tree. The
  `Examples (Smoke)` CI job no longer runs their smoke modes, and
  {doc}`benchmarks/index` is now contributor guidance on what a measurement
  must record rather than a per-driver reference.
: [#143](https://github.com/xywei/volumential/pull/143) — Paper 1 evidence
  library, drivers, table-builder fixes and modernization.
: [#134](https://github.com/xywei/volumential/pull/134) — per-phase operation
  and time shares of end-to-end solves.
: [#133](https://github.com/xywei/volumential/pull/133) — Helmholtz and
  windowed-assembled rows for the 3D composition benchmark.
: [#132](https://github.com/xywei/volumential/pull/132) — 3D split-parameter
  sweep with a resolved FMM-order rule.

Tables and numerics
: [#185](https://github.com/xywei/volumential/pull/185) — the far-field
  direct-sum test from #179 now checks List 1 against the geometry; before, a
  far box filed in List 1 instead of List 3 or 4 dropped out of both the FMM
  and the direct sum, and the test passed.
: [#179](https://github.com/xywei/volumential/pull/179) — pin what #175
  relies on. One test compares the far field with a direct sum on a tree whose
  non-leaf boxes have leaf colleagues; another splits leaves of two levels in
  one `BoxTree.refine_and_coarsen` call, which a `boxtree` older than upstream
  13c9db9 gets wrong without an error; and `test_volume_fmm_laplace` tightens
  from 5e-2 to 1e-3. The user guide now states what the 2:1 balancer allows in
  List 4 and what it costs in accuracy ({ref}`graded-tree-list4`,
  [#178](https://github.com/xywei/volumential/issues/178)).
: [#175](https://github.com/xywei/volumential/pull/175) — keep box-tree
  refinement local. `BoxTree.refine_and_coarsen`, and with it
  `update_mesh`, used to refine every leaf of a level whenever it refined one,
  so a refined mesh was always uniform; now it refines the marked leaves and
  keeps adjacent leaves within one level of each other, so refined meshes are
  adaptive and node counts and errors change for existing callers. A regression
  test checks the volume FMM against the exact solution on a graded tree, and
  `examples/laplace2d_adaptive.py` with its gallery card shows an adaptive
  volume calculation next to the uniform one ({doc}`examples/gallery`). The
  gallery manifest now also records the commit of each recorded package
  installed from Git ({doc}`development/gallery-assets`).
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
: [#189](https://github.com/xywei/volumential/pull/189) — run the
  full-accuracy tier. Its 32 tests built their context on the first fp64 GPU
  whatever `PYOPENCL_CTX` said, so on the GPU-less runner of `CI Full` all of
  them skipped and the job stayed green. The tests that build their own
  context, and the shared near-field table builds, now run on the device
  `PYOPENCL_CTX` selects, CPU or GPU; without it the fp64 ones take an fp64
  GPU and fall back to an fp64 CPU. The 42 fp64 volume FMM regressions that
  the same helper gated join the tier, because the pull-request suite cannot
  afford them on a CPU. `CI Full` runs all 74 on the PoCL CPU, fails if any
  of them skips, and runs the two 3D split-versus-nonsplit tests at the
  reduced size that `VOLUMENTIAL_FULL_ACCURACY_REDUCED=1` selects
  ({doc}`development/testing`). What the first runs turned up is in
  [#190](https://github.com/xywei/volumential/issues/190).
: [#181](https://github.com/xywei/volumential/pull/181) — lift the macOS
  skips. Six test modules skipped themselves on macOS unless
  `VOLUMENTIAL_RUN_UNSTABLE_DARWIN_TESTS=1` was set. The aborts they avoided
  were the PoCL 3.1 kernel-link failure that
  [#177](https://github.com/xywei/volumential/pull/177) fixed, so the skips and
  the variable are gone, and the macOS job in `CI Full` runs the same tests as
  `Testing (Linux)`. The pytest configuration now lives only in
  `pyproject.toml`, because pytest ignored it there while `pytest.ini` existed
  ({doc}`development/testing`).
: [#157](https://github.com/xywei/volumential/pull/157) — three CI and
  backend-portability fixes ([#151](https://github.com/xywei/volumential/issues/151)):
  `set -euo pipefail` in the `Examples (Smoke)` job, which could only fail on
  its last command and so reported a broken `helmholtz2d.py` green for weeks;
  a branch-free `r**power * log(r + 1e-300)` in the Helmholtz split kernels, so
  that no relational reaches sumpy's CSE and the split works on the symengine
  backend as well as on sympy; and a source build of `pyfmmlib` in
  `Testing (Linux)`, which puts the eight batched-P2M tests of
  `test_fmmlib_batched_stages.py` under CI for the first time
  ({doc}`development/ci`).
: [#150](https://github.com/xywei/volumential/pull/150) — follow upstream
  `sumpy` `main`: its sympy-to-pymbolic mapper became sympy-only and its FFT-app
  helper keyword-only, so the batched Duffy builders now fold symbol-free
  scaling constants to literals and inherit sumpy's cached FFT-plan hook;
  `sumpy` and `loopy` relocked.
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
