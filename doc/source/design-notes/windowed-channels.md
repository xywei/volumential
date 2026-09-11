# Windowed singular channels and certified assembly

*Implemented in {mod}`volumential.rke_table_assembly`. Exercised by
`benchmarks/windowed_rke_sweep.py`, `benchmarks/complex_channel_closure.py`
and the `--include-windowed` paths of the composition drivers.*

## The problem

A fixed-parameter near-field table for Helmholtz or Yukawa depends on the wave
number. Sweeping a parameter — a wave-number ladder, a continuation in a
time-step-derived screening length — therefore costs one full Duffy table build
per parameter value, and each build is the expensive part of the whole
pipeline. Worse, those builds need special-function quadrature (Bessel and
Hankel evaluations at every Duffy node), which is exactly the part the batched
device kernel is worst at.

## Channel assembly

The reduced-kernel-expansion (RKE) idea is to assemble the fixed-parameter
table as a linear combination of *parameter-independent* channel tables:

- 2D: $T(k) = T[\text{Laplace}] + c_0(k)\,T[\text{const}]
  + \sum_n \bigl( a_n(k)\, T[r^{2n}\log r] + b_n(k)\, T[r^{2n}] \bigr)$
- 3D: $T(k) = T[\text{Laplace}] + \sum_n c_n(k)\, T[r^{n-1}]$

with the exact small-argument series coefficients — the same series that
defines the private `_HelmholtzSplitSeriesRemainderKernel` of
{mod}`volumential.wranglers`. Yukawa uses the principal-branch substitution
$k = i\lambda$, under which the assembled table is real.

Every channel integrand is elementary (powers and `log`), so all channel tables
build through the batched device Duffy path; no special-function quadrature is
needed anywhere. The channel family is parameter independent, so once it is
built and cached, a further parameter costs only coefficient evaluation and a
linear combination over the symmetry-reduced entries.

The truncation order is chosen from the requested tolerance by an explicit
majorant of the omitted series tail on the near-field separation region. The
assembler returns `(table, certificate)`, where the **certificate** is a
provenance dictionary recording that bound together with the numerically
computed $L^1$ norms of the source basis functions — which is what converts a
kernel-space bound into a table-entry bound.

There is no status field on it: an assembly that cannot meet the bound does not
return a refused certificate, it *raises* — `RKETruncationError` or
`RKEConditioningError` for the classical assembler, `RKEWindowCoverageError` or
`RKEWindowConditioningError` for the windowed one. A caller branches on the
exception, not on a field.

The `ok` / `refused` / `failed` taxonomy that appears in the benchmark CSVs is
a *driver-level* classification layered on top of that: the wrappers call the
assembler, record `refused` when it raises one of the certified refusals above,
and `failed` when it raises anything else. A refusal is a measurement, not an
error, which is exactly why the drivers keep the row.

## Why the classical channels are not enough

The classical channels $r^{2m}\log r$ and $r^{m-1}$ grow polynomially, so their
coefficients and their table entries scale in opposite directions as the
retained order rises. Past a modest $\theta = |{\rm parameter}| \cdot b$ (with
$b$ the source-box extent) the linear combination is a cancelling sum of large
terms and the certificate refuses before the accuracy target is met.

## The windowed channels

The windowed assembler (`assemble_windowed_parameterized_table`) replaces the
growing channels by Gaussian-windowed ones:

- 2D: $\chi_m(r) = \tfrac12 (r^2/4)^m\, \Gamma(-m, x)$
- 3D: $\chi_m(r) = \tfrac{1}{2\sqrt{\pi}} (r^2/4)^{m-1/2}\,
  \Gamma(\tfrac12 - m, x)$

with $x = r^2/(4 t_w)$ and window scale $t_w = (b/\Theta)^2$ for source-box
extent $b$ and the single design declaration $\Theta$ (`window_theta`). Each
channel carries the kernel's full singular germ but dies off beyond
$r \sim b/\Theta$.

Two properties make this work:

1. **No overflow in the pairing.** Tables store the *normalized* channels
   $\psi_m = \chi_m / t_w^m$ and pair them with coefficients
   $(-\zeta t_w)^m / m!$. The coefficient magnitude is bounded by
   $|\zeta t_w|^m/m! = (\theta/\Theta)^{2m}/m!$ for every covered
   squared-frequency parameter $\zeta$ ($\zeta = \lambda^2$ for Yukawa,
   $\zeta = -k^2$ for Helmholtz, complex $\zeta$ for damped waves), so neither
   factor carries the opposing $t_w^m$ scaling that would overflow or underflow
   before their product did.
2. **No truncation error at all.** The online part is the smooth remainder
   $R = G - \sum_{m < p_\star} \bigl((-\zeta t_w)^m/m!\bigr)\psi_m$, evaluated
   pointwise from the kernel and the closed channel forms and integrated
   against the source modes by a tensor-product Gauss-Legendre rule. Because
   $R$ is defined as the *exact* difference, the certificate's
   `truncation_tail_bound` is structurally `0.0`, and the only error terms left
   are the two the scheme already owns: singular channel quadrature and smooth
   remainder quadrature.

It is also a smaller family. The windowed assembler builds exactly one channel
table per retained order, $\psi_0 \ldots \psi_{p_\star - 1}$, so `p_star`
tables in either dimension; the classical family needs $1 + n$ tables in 3D and
$2 + 2n$ in 2D for $n$ retained terms, because each 2D term contributes both an
$r^{2n}$ and an $r^{2n}\log r$ channel. The remainder is evaluated online
rather than tabulated, so it adds none.

The channel family depends on the declaration $\Theta$ only, never on the swept
parameter, so one cached family serves every $\theta = |{\rm parameter}|\cdot b$
up to $\Theta$. The lower half plane is used for the branch of the square root,
so the assembler's selected root is the outgoing branch at every sampled phase
of a damped complex frequency, continuous with the $-ik$ endpoint.

## How an assembled table reaches the evaluator

An assembled table is not a DuffyRadial build, and it does not pretend to be
one. It is registered through the ordinary table manager
(`register_external_table`) with `build_method = ExternalAssembly`, reloaded
through the ordinary cache path, and applied by the identical evaluator that
serves a directly built table. That is deliberate: the comparison the
benchmarks make is between two *provisioning strategies* for the same
evaluator, not between two evaluators.

Because an external assembly never ran the Duffy builder, it is exempt from the
build-routing strictness of {doc}`../user-guide/table-build-routing`; its
provenance is carried instead by the build method, the provenance kind and the
payload checksum the load path verifies.

## Where to look

- Mechanism and certificates: {mod}`volumential.rke_table_assembly`.
- Online split that motivates the channel structure:
  {doc}`../user-guide/helmholtz_split`.
- Evidence, including certificate status, condition numbers, per-parameter
  assembly/registration/load costs and break-even models:
  {doc}`../benchmarks/index`.
