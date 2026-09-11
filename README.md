# Volumential

[![CI](https://github.com/xywei/volumential/actions/workflows/ci.yml/badge.svg)](https://github.com/xywei/volumential/actions/workflows/ci.yml)

Volumential (VOLUME poteNTIAL) provides a toolset to solve volume potential
integral equations efficiently via the Fast Multipole Method.

## Quick Start

Install an OpenCL runtime first (for example, in conda:
`conda install pyopencl pocl`), then use `uv` for dependency resolution:

```bash
uv sync --active --extra test
uv run python examples/laplace2d.py
uv run python examples/helmholtz2d.py
uv run python examples/helmholtz3d.py
uv run python examples/poisson3d.py
```

`uv.lock` pins the inducer-stack dependencies (`boxtree`, `sumpy`, `loopy`,
`pyopencl`, `pytools`, `modepy`, `arraycontext`, `meshmode`, `pytential`) to
main-branch Git sources rather than to released wheels, because those projects
publish releases rarely and the wheels have shipped defects that silently
corrupt adaptive-tree results.

`pyfmmlib` is pinned the same way, and for the same reason: the PyPI release
has neither the OpenMP option nor the batched wrappers the FMMLib backend
needs, and `FPNDFMMLibExpansionWrangler` then falls back to its serial path in
silence. Since #135 it has a `tool.uv.sources` entry, so the extra resolves to
the Git source at the commit `uv.lock` pins, and the recipe is simply:

```bash
uv sync --active --extra fmmlib
```

The source build needs a host with `gfortran` and `ninja`.

See `DEVELOPMENT.md` for the full provisioning recipe, including the
verification that the batched wrappers are present and the post-install
traversal sanity check that every experiment environment must pass.

## Repository Layout

- `volumential/`: library source (see the architecture map below)
- `test/`: pytest suite
- `examples/`: maintained end-to-end examples (`laplace2d.py`, `laplace3d.py`,
  `helmholtz2d.py`, `helmholtz3d.py`, `poisson3d.py`,
  `branched_flow_helmholtz2d.py`, the two `*_split_p_convergence.py` drivers,
  and the `*.ipynb` notebooks)
- `benchmarks/`: reproducible benchmark drivers that emit CSV/JSON evidence;
  each has a lightweight `--mode smoke` for CI and a `--mode full` for
  controlled runs (see `benchmarks/README.md`)
- `doc/`: Sphinx documentation
- `ruff.toml`: the single lint configuration (line length 85, py311)
- `DEVELOPMENT.md`: environment provisioning and day-to-day workflow

## Architecture Map

Geometry and box meshes

- `meshgen.py`: box mesh generation over a boxtree tree-of-boxes
  (`MeshGen1D/2D/3D`, `make_uniform_cubic_grid`, `build_geometry_info`).
- `tree_interactive_build.py`: interactive/adaptive refinement and coarsening
  with 2:1 level restriction.
- `geometry.py`: `BoundingBoxFactory`, `BoxFMMGeometryFactory` and the
  `BoxFMMGeometryData` handed to the wranglers.

Near-field tables

- `nearfield_potential_table.py`: `NearFieldInteractionTable`, the near-field
  quadrature tables and their symmetry-reduced payloads.
- `table_manager.py`: `NearFieldInteractionTableManager`, the SQLite-backed
  table cache (schema `2.1.0`).
- `list1_gallery.py`: enumeration of the List 1 interaction cases.
- `list1_symmetry.py`: symmetry operations acting on case vectors.
- `orbit_arithmetic.py`: orbit canonicalization and sign bookkeeping.
- `rke_table_assembly.py`: certified RKE assembly of fixed-parameter tables.
- `singular_integral_2d.py`: 2D Duffy-transform singular quadrature
  (`box_quad`).
- `list1.py`: the on-device List 1 near-field evaluators
  (`KernelScalingPolicy`, `NearFieldEvalBase`, `NearFieldFromCSR`) that read
  table data through the CSR interaction lists.

Volume FMM

- `volume_fmm.py`: `drive_volume_fmm` plus interpolation of the box-mesh
  potential onto user targets.
- `expansion_wrangler_interface.py`: the wrangler interface and the
  tree-independent data it needs.
- `expansion_wrangler_fpnd.py`: the FPND wranglers (sumpy and FMMLib backends)
  and the near-field Helmholtz split. This module is being reorganized into a
  `volumential/wranglers/` subpackage; the current module path stays importable
  through compatibility re-exports.

Fields, sources and post-processing

- `interpolation.py`: transfer between meshmode discretizations and the box
  mesh.
- `lagrange.py`: barycentric Lagrange basis utilities.
- `gaussian.py`: Gaussian source fixtures for free-space demos.
- `function_extension.py`: layer-potential source extension with regularity
  constraints.
- `symbolic.py`: symbolic expression evaluation helpers for source densities.

Support

- `tools.py`: the loopy kernel cache wrapper and scalar-field evaluation.
- `opcounters.py`: explicit operation counters for benchmark drivers.
- `__init__.py`, `version.py`: public surface (`__all__`), version and the
  kernel revision used in cache keys.
- `qbfem/`: a 2019 finite-element experiment that nothing else in the tree
  imports; it is kept only for reference and is not part of the supported API.

Debug tip: when validating source-node evaluations, set
`VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE=1` to fail fast if a traversal is built
with separate-but-identical source/target arrays (use `targets=None` to build a
true coincident tree).

## Near-Field Table Build Routing

Near-field DuffyRadial tables are built by a batched OpenCL kernel.  If that
build raises, the table falls back to the scalar per-entry builder, which is
orders of magnitude slower and converges differently at the same requested
quadrature orders.  The fallback is therefore never silent:

- it logs a `WARNING` plus a `[duffy:builder] mode=scalar-fallback` line and
  emits a `RuntimeWarning` carrying the kernel class, dimension, exception
  type and reason;
- it records `table.build_routing` (`batched`, `scalar`, `scalar-adaptive` or
  `scalar-fallback`) and `table.build_fallback_reason` on the table, and both
  are persisted with the cached payload, so a warm, cache-loaded table still
  reports how it was originally built
  (`volumential.opcounters.direct_build_routing`);
- the Paper 1 benchmark drivers emit it as a `direct_build_routing` CSV column.

Set `VOLUMENTIAL_DUFFY_NO_FALLBACK=1` to turn the fallback into a
`RuntimeError` instead — campaign runs use this so that a table which quietly
dropped to the scalar builder cannot be recorded as a batched build.  Any
value other than unset, `0`, `false`, `no` or `off` enables the strict mode.
Strict mode also applies on the *load* path, where the builder never runs: a
cached table whose recorded routing is `scalar-fallback`, or which records no
routing at all (a payload written before routing was recorded, so its
provenance cannot be verified), is refused with an
`UnverifiedBuildRoutingError` naming the table and the remedy — rebuild it
with `force_recompute=True`, or unset the switch to accept the cached data.
Externally assembled tables (`build_method = ExternalAssembly`, e.g. a
registered windowed RKE assembly) are exempt: they were never a DuffyRadial
build, which is why the assemblers clear the routing, and their provenance is
carried by the build method, the provenance kind and the payload checksum the
load path verifies.
Without that, a strict campaign whose cache had already been warmed would load
and use exactly the data the switch exists to refuse.
It is an environment switch rather than a `DuffyBuildConfig` field because the
build config is hashed into the table-cache fingerprint, and an operational
strictness policy should not invalidate cached numerical data.

### Complex Exponentials in the Generated Quadrature Kernel

The fused Duffy quadrature kernel rewrites every `exp(re + i*im)` into
`exp(re) * (cos(im) + i*sin(im))` before code generation, so complex-valued
kernels reach the device as real `exp`/`cos`/`sin` calls and never as
`cdouble_exp`. pyopencl implements `cdouble_exp` with the OpenCL
`sincos(x, &cosx)` out-parameter builtin, which on the PoCL 7.0 / LLVM 19.1.7
CPU driver costs about 200 ns per call against about 1.6 ns for a separate
`sin`/`cos` pair; since the quadrature evaluates the kernel at every Duffy
node, that one builtin made the 3D Helmholtz direct table build roughly ten
times slower than the otherwise identical Yukawa build. The rewrite is
`exp(a+b) = exp(a)exp(b)` with Euler's formula over an exact structural split
of the exponent, so it is valid for genuinely complex exponents (the damped
`exp((-a + i b) r)` form included) and leaves real exponents untouched.

The rewrite is exact in value but not in conditioning once the *phase* `im`
can itself be complex: for `z = x + i y`, `cos z` and `sin z` both grow like
`exp(|y|)/2` while `exp(i z)` decays like `exp(-y)`, so Euler's formula turns
a decaying exponential into a cancelling difference of two large terms. It
also replaces one `cdouble_exp`, which promotes its whole argument to double,
with bare real calls whose precision loopy infers from the expression.

So the rewrite happens only for a phase, and a magnitude, that are *provably*
real doubles, checked node by node. Every leaf must be a real-typed constant
at least as wide as a double, a variable the kernel has not left unproven, an
arithmetic combination of those, or a call to a function that is real for real
arguments; anything else -- an unrecognised node type, a `hankel1` call, a
`complex128(0j)` that promotes the operation around it, a post-CSE
`CommonSubexpression` wrapping any of those -- keeps its `cdouble_exp`.

An expression made only of constants is refused whatever their Python types,
because nothing in it fixes the emitted precision -- loopy writes the constant
real half of `exp(-200 + 1j*k)` as `exp((float) (-200.0f))`, which underflows
where `cdouble_exp` kept the finite `exp(-200)`.

A kernel argument counts as proven only when its declared dtype is a real
floating type at least as wide as a double. Arguments a caller supplies
through `extra_kernel_kwarg_types` are checked by the same rule, since they
are not in `integral_knl.get_args()`. Complex (the wave number of
`HelmholtzKernel(dim, allow_evanescent=True)`), narrow, integer, and
undeclared dtypes are all unproven. That is deliberately blunt: loopy's
constant-dtype inference cannot be reproduced from the expression tree — an
integer argument alone narrows the result of a floating builtin, and even a
plain `3.0` beside an integer is emitted as `3.0f` — so the guard does not try
to model it. No sumpy kernel this table builds has such an argument
(Helmholtz's `k` and Yukawa's `lam` are both `float64`), so the rule costs
nothing in practice and gives a guarantee instead of an approximation.

The global scaling constant is rewritten under the same guard, since it is
evaluated inside both quadrature loops, and the split walks the `/1` wrapper
`SympyToPymbolicMapper` leaves around it.

Measured effect at 3D, `q = 3`, source box level 2: Helmholtz per
(entry x node) cost drops from ~74 ns to ~8 ns, matching the real-valued
Yukawa kernel, with table entries agreeing to 3e-16 relative.

## Near-Field Symmetry and Cache Format

- Near-field table storage uses orbit canonicalization over
  `(source_mode, target_mode, interaction_case)` and stores only canonical
  entries.
- Derivative kernels are supported with sign-aware orbit metadata:
  runtime lookup applies a per-entry sign factor when reconstructing from
  canonical entries.
- SQLite cache schema `2.1.0` stores table content in the `payload` blob only;
  legacy dense blob columns were removed.
- Symmetry-reduced payloads persist only finite canonical data arrays
  (`reduced_entry_ids` + `reduced_data`) and do not store NaN sentinels.

## Documentation

[Browse the documentation online.](http://xiaoyu-wei.com/docs/volumential/)

Build it locally with `uv sync --active --extra doc` and `make -C doc html`.

## License

Volumential is developed and released under the terms of the MIT license,
though it also makes use of third-party packages under their own licensing
terms. See the [LICENSE](./LICENSE.md) file for details.

## Acknowledgements

We would like to thank people and organizations that contributed to or supported the
Volumential project, without whom the project would not have been possible.

The research that started the Volumential project was supported by the
[National Science Foundation][nsf] under grant DMS-1654756,
and by the [Department of Computer Science][uiuc-cs] at the
[University of Illinois at Urbana-Champaign][uiuc].
Part of the work was performed while the authors were participating in
the [HKUST][hkust]-[ICERM][icerm] workshop "Integral Equation Methods, Fast
Algorithms and Their Applications to Fluid Dynamics and Materials
Science" held in 2017.

Thanks very much to the [Department of Mathematics][hkust-math] at
[Hong Kong University of Science and Technology][hkust]
for funding Xiaoyu Wei to work on the project
as a PhD student under the Postgraduate Studentship and
the Overseas Research Award.

The project's name `volumential` [courtesy of Andreas Klöckner][volumential-name].

[nsf]: https://www.nsf.gov/
[hkust-math]: https://www.math.ust.hk/
[hkust]: https://www.ust.hk/home
[icerm]: https://icerm.brown.edu/
[uiuc-cs]: https://cs.illinois.edu/
[uiuc]: https://illinois.edu/
[volumential-name]: https://gitlab.tiker.net/xywei/volumential/issues/2
