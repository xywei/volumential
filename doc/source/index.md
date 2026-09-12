# Volumential

**Volumential** (VOLUME poteNTIAL) evaluates volume potentials

$$
u(\boldsymbol{x}) = \int_{\Omega} G(\boldsymbol{x}, \boldsymbol{y})\,
f(\boldsymbol{y}) \, \mathrm{d}\boldsymbol{y}
$$

over box-shaped domains with the Fast Multipole Method. The far field is an
ordinary particle FMM over the volume quadrature nodes; the near field is read
from precomputed, symmetry-reduced interaction tables. That split — *far field
by particle approximation, near field direct* — is what the code calls the
`fpnd` strategy, and it is the thing most of this documentation is about.

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

The eight programs under `examples/`, what each one costs to run and which
device it lands on, plus the notebooks rendered as pages.
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

The drivers under `benchmarks/`, their metadata sidecars, what a promoted
measurement has to record, and how results are promoted.
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
- Looking for a program close to your problem: {doc}`examples/index`.
- Want to understand the machinery: {doc}`user-guide/volume-fmm-workflow`.
- Chasing a slow or wrong table: {doc}`user-guide/table-build-routing` and
  {doc}`user-guide/nearfield_symmetry`.
- Reproducing a number from a paper: {doc}`benchmarks/index`.

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
