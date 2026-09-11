# Near-field table build routing

Near-field DuffyRadial tables are built by a batched OpenCL kernel. If that
build raises, the table falls back to the scalar per-entry builder, which is
orders of magnitude slower and converges differently at the same requested
quadrature orders. A table that quietly dropped to the scalar path is therefore
both a performance surprise and an accuracy surprise, which is why the fallback
is never silent.

## The fallback is loud and recorded

- It logs a `WARNING` plus a `[duffy:builder] mode=scalar-fallback` line and
  emits a `RuntimeWarning` carrying the kernel class, dimension, exception type
  and reason.
- It records `table.build_routing` (`batched`, `scalar`, `scalar-adaptive` or
  `scalar-fallback`) and `table.build_fallback_reason` on the table, and both
  are persisted with the cached payload, so a warm, cache-loaded table still
  reports how it was originally built
  (`volumential.opcounters.direct_build_routing`).
- The Paper 1 benchmark drivers emit it as a `direct_build_routing` CSV column.

## Strict mode

```bash
export VOLUMENTIAL_DUFFY_NO_FALLBACK=1
```

turns the fallback into a `RuntimeError` instead. Campaign runs use this so
that a table which quietly dropped to the scalar builder cannot be recorded as
a batched build. Any value other than unset, `0`, `false`, `no` or `off`
enables strict mode.

Strict mode also applies on the **load** path, where the builder never runs: a
cached table whose recorded routing is `scalar-fallback`, or which records no
routing at all (a payload written before routing was recorded, so its
provenance cannot be verified), is refused with an
`UnverifiedBuildRoutingError` naming the table and the remedy — rebuild it with
`force_recompute=True`, or unset the switch to accept the cached data. Without
that, a strict campaign whose cache had already been warmed would load and use
exactly the data the switch exists to refuse.

Externally assembled tables (`build_method = ExternalAssembly`, for example a
registered windowed RKE assembly; see
{doc}`../design-notes/windowed-channels`) are exempt: they were never a
DuffyRadial build, which is why the assemblers clear the routing, and their
provenance is carried by the build method, the provenance kind and the payload
checksum the load path verifies.

Strictness is an environment switch rather than a `DuffyBuildConfig` field
because the build config is hashed into the table-cache fingerprint, and an
operational strictness policy should not invalidate cached numerical data.

## Complex exponentials in the generated quadrature kernel

The fused Duffy quadrature kernel rewrites every `exp(re + i*im)` into
`exp(re) * (cos(im) + i*sin(im))` before code generation, so complex-valued
kernels reach the device as real `exp`/`cos`/`sin` calls and never as
`cdouble_exp`. `pyopencl` implements `cdouble_exp` with the OpenCL
`sincos(x, &cosx)` out-parameter builtin, which on the PoCL 7.0 / LLVM 19.1.7
CPU driver costs about 200 ns per call against about 1.6 ns for a separate
`sin`/`cos` pair; since the quadrature evaluates the kernel at every Duffy
node, that one builtin made the 3D Helmholtz direct table build roughly ten
times slower than the otherwise identical Yukawa build. The rewrite is
`exp(a+b) = exp(a)exp(b)` with Euler's formula over an exact structural split
of the exponent, so it is valid for genuinely complex exponents (the damped
`exp((-a + i b) r)` form included) and leaves real exponents untouched.

### Why it is guarded

The rewrite is exact in value but not in conditioning once the *phase* `im` can
itself be complex: for `z = x + i y`, `cos z` and `sin z` both grow like
`exp(|y|)/2` while `exp(i z)` decays like `exp(-y)`, so Euler's formula turns a
decaying exponential into a cancelling difference of two large terms. It also
replaces one `cdouble_exp`, which promotes its whole argument to double, with
bare real calls whose precision `loopy` infers from the expression.

So the rewrite happens only for a phase, and a magnitude, that are *provably*
real doubles, checked node by node. Every leaf must be a real-typed constant at
least as wide as a double, a variable the kernel has not left unproven, an
arithmetic combination of those, or a call to a function that is real for real
arguments; anything else — an unrecognised node type, a `hankel1` call, a
`complex128(0j)` that promotes the operation around it, a post-CSE
`CommonSubexpression` wrapping any of those — keeps its `cdouble_exp`.

An expression made only of constants is refused whatever their Python types,
because nothing in it fixes the emitted precision: `loopy` writes the constant
real half of `exp(-200 + 1j*k)` as `exp((float) (-200.0f))`, which underflows
where `cdouble_exp` kept the finite `exp(-200)`.

A kernel argument counts as proven only when its declared dtype is a real
floating type at least as wide as a double. Arguments a caller supplies through
`extra_kernel_kwarg_types` are checked by the same rule, since they are not in
`integral_knl.get_args()`. Complex (the wave number of
`HelmholtzKernel(dim, allow_evanescent=True)`), narrow, integer and undeclared
dtypes are all unproven. That is deliberately blunt: `loopy`'s constant-dtype
inference cannot be reproduced from the expression tree — an integer argument
alone narrows the result of a floating builtin, and even a plain `3.0` beside
an integer is emitted as `3.0f` — so the guard does not try to model it. No
sumpy kernel this table builds has such an argument (Helmholtz's `k` and
Yukawa's `lam` are both `float64`), so the rule costs nothing in practice and
gives a guarantee instead of an approximation.

The global scaling constant is rewritten under the same guard, since it is
evaluated inside both quadrature loops, and the split walks the `/1` wrapper
`SympyToPymbolicMapper` leaves around it.

### Measured effect

At 3D, `q = 3`, source box level 2: the Helmholtz per-(entry × node) cost drops
from about 74 ns to about 8 ns, matching the real-valued Yukawa kernel, with
table entries agreeing to `3e-16` relative.
