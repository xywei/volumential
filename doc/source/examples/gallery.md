# Visual gallery

If you are deciding whether Volumential fits a problem, start here rather than
with the API reference. The examples below are maintained programs or notebooks,
not separate documentation demos.

```{figure} ../_static/gallery/examples-showcase.svg
:alt: Illustrated overview of the Poisson 2-D, Poisson 3-D and Helmholtz
      branched-flow examples.
:width: 100%

A map of the example families. The panels are navigation illustrations, not
benchmark output; the linked programs produce the numerical diagnostics.
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` First volume potential
:link: ../getting-started/first-volume-potential
:link-type: doc

**Laplace · 2-D · start here**

Run a manufactured Gaussian problem, see the source/reference fields, then walk
through mesh → tree → near-field table → far-field FMM → error check.
:::

:::{grid-item-card} {octicon}`image` Poisson on a starfish domain

**Poisson · 2-D · geometry + AMR**

The notebook `poisson2d_pytential_volumential.ipynb` couples Volumential to
Pytential, enforces Dirichlet data with a harmonic boundary correction, and
compares uniform refinement with boundary-focused AMR.

{doc}`notebooks/poisson2d_pytential_volumential`
:::

:::{grid-item-card} {octicon}`stack` Poisson in three dimensions

**Poisson · 3-D · slices + error localization**

`poisson3d.py` writes orthogonal exact/FMM/error slices and a 3-D error cloud;
the notebook adds an interactive isosurface view when Plotly is available.

{doc}`notebooks/poisson3d_volumential`
:::

:::{grid-item-card} {octicon}`pulse` Helmholtz branched flow

**Helmholtz · 2-D · full application**

A free-space Lippmann--Schwinger solve through a smooth random medium. The
example writes the refractive-index perturbation, normalized intensity and
real part of the total field.

See {ref}`branched-flow-example` for the run command and cost warning.
:::

:::{grid-item-card} {octicon}`split` Near field vs. far field
:link: ../user-guide/near-field-anatomy
:link-type: doc

**Mechanism · read before the deep implementation pages**

See why the singular neighborhood leaves the ordinary particle FMM path, what
is tabulated, and what can be reused across source densities.
:::

:::{grid-item-card} {octicon}`graph` Convergence studies

**Accuracy · q-order / refinement / residuals**

The Helmholtz examples and Poisson notebooks include parameter sweeps. Use them
for numerical behavior; use {doc}`../benchmarks/index` before quoting timings.
:::

::::

## Generate the figures yourself

The documentation build deliberately does not execute scientific workloads.
The maintained examples write the richer numerical figures when you run them:

```bash
# The first tutorial: writes source/computed/exact/error + tree SVGs.
VOLUMENTIAL_EXAMPLE_SMOKE=1 \
VOLUMENTIAL_GALLERY_OUTPUT_DIR=build/gallery/laplace2d \
uv run --with matplotlib python examples/laplace2d.py

# 3-D slices and point-cloud error.
VOLUMENTIAL_EXAMPLE_SMOKE=1 \
VOLUMENTIAL_POISSON3D_OUTPUT_DIR=build/gallery/poisson3d \
uv run --with matplotlib python examples/poisson3d.py

# Refractive-index perturbation, intensity and Re(u).
uv run --with matplotlib python examples/branched_flow_helmholtz2d.py \
    --smoke --output-dir build/gallery/branched-flow
```

Maintainers can regenerate the curated asset tree with
`doc/tools/render_gallery.py`; {doc}`../development/gallery-assets` records
the provenance rules.

## Need the operational details?

The main {doc}`index` page remains the source of truth for cache filenames,
device-selection behavior, smoke modes and the cost class of every example.
