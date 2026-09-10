# Volumential

![pipeline](https://gitlab.tiker.net/xywei/volumential/badges/master/pipeline.svg)
![coverage](https://gitlab.tiker.net/xywei/volumential/badges/master/coverage.svg)

Volumential (VOLUME poteNTIAL) provides toolset to solve volume potential integral equations
efficiently via Fast Multipole Method.

## Repository Layout

- `volumential/`: library source
- `test/`: pytest suite
- `examples/`: maintained end-to-end examples (`laplace2d.py`, `laplace3d.py`,
  `helmholtz2d.py`, `helmholtz3d.py`, `poisson3d.py`,
  `poisson3d_volumential.ipynb`, `poisson2d_pytential_volumential.ipynb`,
  `helmholtz3d_volumential.ipynb`)
- `doc/`: Sphinx documentation
- `DEVELOPMENT.md`: local/remote development environment guide

Legacy `contrib/`, `benchmarks/`, `experiments/`, and `docker/` trees were
removed as part of repository cleanup.

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

For a fully reproducible local + remote setup, see `DEVELOPMENT.md`.

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
a decaying exponential into a cancelling difference of two large terms.
A phase that is not *provably* real therefore keeps its `cdouble_exp`, which
evaluates the decaying result directly. The test is positive rather than a
search for complex arguments, because the split hands any node it does not
walk into — a post-CSE `CommonSubexpression`, say — to the real part
wholesale, so `exp(1j*CSE((3+40j)*r))` has a complex phase without naming a
complex argument anywhere. A phase passes only if every node in it is a real
constant, a variable the kernel does not leave unproven, an arithmetic
combination of those, or a call to a function that is real for real
arguments. A kernel argument counts as real only if its *declared* dtype is
real: `HelmholtzKernel(dim, allow_evanescent=True)` declares a `complex128`
wave number, and an argument declared with no dtype at all
(`KernelArgument(lp.ValueArg("k"))`, which loopy leaves as `<auto/runtime>`
and which accepts a complex value at run time) is likewise not proven real. Ordinary
Helmholtz and Yukawa pass and are rewritten as above.

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
