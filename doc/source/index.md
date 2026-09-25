---
# Two consumers, two keys. Keep the landing-page summary explicit rather than
# deriving it from whichever visual/prose block happens to lead the doctree.
# ``html_meta`` is the ordinary search-engine description; the top-level key
# supplies the OpenGraph description used by link unfurls.
"og:description": >-
  Volumential evaluates volume potentials over box-shaped domains with the
  Fast Multipole Method, for the Laplace, Helmholtz and Yukawa kernels in
  two and three dimensions.
myst:
  html_meta:
    description: >-
      Volumential evaluates volume potentials over box-shaped domains with the
      Fast Multipole Method, for the Laplace, Helmholtz and Yukawa kernels in
      two and three dimensions.
---

# Volumential

Volumential evaluates volume potentials, integrals of a kernel against a
source density over a box, with the Fast Multipole Method. This is what the
first example computes.

```{figure} _static/gallery/laplace2d/laplace2d_overview.svg
:alt: Four panels over the square from -0.5 to 0.5. Top left, the source density: a positive peak at the origin inside a faint negative ring. Top right, the computed volume potential: a Gaussian bump. Bottom left, the Gaussian reference, which looks identical. Bottom right, the pointwise error on a logarithmic scale, at most about 8e-11, arranged in nested squares that follow the FMM boxes.
:width: 100%

Computed by `examples/laplace2d.py` at full settings (quadrature order 9,
6 mesh levels, multipole order 20, 82944 quadrature nodes). The source
$f = -\Delta u$ is chosen so that $u = e^{-160 \lVert \boldsymbol{x} \rVert^2}$
solves $-\Delta u = f$ in the whole plane; the example integrates $f$ over the
box only, which leaves out source mass at rounding level. The example prints
the largest difference between its computed potential $u_h$ and $u$ over the
nodes; for this run, `Error = 8.410442587858608e-11`. Regenerate with
`python doc/tools/render_gallery.py laplace2d --pyopencl-ctx portable:0 --full`.
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Run the first example
:link: getting-started/first-volume-potential
:link-type: doc

Reproduce the figure above, then read the program behind it in six short
steps: source, quadrature nodes, tree, near-field table, FMM, error.
:::

:::{grid-item-card} {octicon}`image` Browse the gallery
:link: examples/gallery
:link-type: doc

Computed figures from the maintained examples, in two and three dimensions,
each with the settings and the command that produced it.
:::

::::

The name is short for VOLUME poteNTIAL. For a kernel $G$ and a source density
$f$ on a box-shaped domain $\Omega$, Volumential evaluates

$$
u(\boldsymbol{x}) = \int_{\Omega} G(\boldsymbol{x}, \boldsymbol{y})\,
f(\boldsymbol{y}) \, \mathrm{d}\boldsymbol{y}.
$$

The far field is an ordinary particle FMM over the volume quadrature nodes;
the near field is read from precomputed, symmetry-reduced interaction tables.
That split — *far field by particle approximation, near field direct* — is
what the code calls the `fpnd` strategy, and it is the thing most of this
documentation is about.

Supported kernels are Laplace, Helmholtz and Yukawa (modified Helmholtz) in
two and three dimensions, with potential and target-gradient outputs, on
uniform and adaptively refined 2:1-balanced trees.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting-started/index
:link-type: doc

Install the stack, evaluate a first volume potential, and pick the OpenCL
device you meant to use.
:::

:::{grid-item-card} {octicon}`beaker` Examples
:link: examples/index
:link-type: doc

A visual map of the maintained programs and notebooks, followed by the
device, cache and cost details needed to reproduce them.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: user-guide/index
:link-type: doc

The volume-FMM workflow end to end: meshes and trees, near-field tables and
their symmetry reduction, the Helmholtz split, derivatives, and what is
validated.
:::

:::{grid-item-card} {octicon}`light-bulb` Design notes
:link: design-notes/index
:link-type: doc

Short accounts of the two mechanisms that are easiest to misread from the
source alone: windowed singular channels with certified assembly, and ORBIT
canonicalization.
:::

:::{grid-item-card} {octicon}`graph` Benchmarks and reproducibility
:link: benchmarks/index
:link-type: doc

What a measurement of this library has to record to be worth quoting: the
resolved device, the revision, the parameters, first call versus warm.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

One generated page per module, with a map from the pieces of the volume FMM
to the module that owns them.
:::

:::{grid-item-card} {octicon}`tools` Development
:link: development/index
:link-type: doc

Contributing, the test tiers and their markers, CI and the review bots, and
release and versioning.
:::

::::

## Where to start

- Never run Volumential before: {doc}`getting-started/installation`, then
  {doc}`getting-started/first-volume-potential`.
- Looking for a program close to your problem: {doc}`examples/gallery`, then
  {doc}`examples/index` for run/cost/cache details.
- Want to understand the machinery: {doc}`user-guide/volume-fmm-workflow`.
- Chasing a slow or wrong table: {doc}`user-guide/table-build-routing` and
  {doc}`user-guide/nearfield_symmetry`.
- Measuring, or quoting a measurement: {doc}`benchmarks/index`.

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
examples/index
user-guide/index
design-notes/index
Benchmarks <benchmarks/index>
api/index
development/index
changelog
```
