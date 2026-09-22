# Visual gallery

Start here if you want to understand Volumential by looking at problems and
outputs before reading the implementation.

The diagrams on this page are **schematics**, not measured numerical output.
Computed plots from the maintained examples are regenerated explicitly with
`doc/tools/render_gallery.py`; normal Sphinx builds never execute OpenCL
workloads. The renderer records the Git revision and commands beside the images
so a gallery update remains reviewable evidence rather than decoration.

## The core idea

```{figure} ../images/near-far.svg
:alt: A target box and its neighboring boxes use a precomputed near-field table, while well-separated boxes are handled by the particle FMM.
:align: center

The volume-specific part of Volumential is local. Singular and near-singular
List 1 interactions come from precomputed tables; the rest is an ordinary
particle FMM.
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Laplace 2-D — start here
:link: ../getting-started/first-volume-potential
:link-type: doc

**See:** a smooth manufactured source, its volume potential, and the error
against the exact Gaussian solution.

**Run:** `VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run python examples/laplace2d.py`

This is the smallest maintained end-to-end route through mesh → tree → table →
wrangler → volume FMM.
:::

:::{grid-item-card} Poisson 3-D — see where the error lives
:link: index
:link-type: doc

**See:** orthogonal exact/computed/error slices, a 3-D point-cloud error view,
and optional interactive isosurfaces.

**Run:** `VOLUMENTIAL_EXAMPLE_SMOKE=1 uv run --with matplotlib python examples/poisson3d.py`

Use this example when a scalar error norm is not enough and you want spatial
diagnostics.
:::

:::{grid-item-card} Helmholtz branched flow — application scale
:link: index
:link-type: doc

**See:** refractive-index perturbation, normalized intensity, and the real part
of the total field for a free-space Lippmann-Schwinger solve.

**Run:** `uv run python examples/branched_flow_helmholtz2d.py --smoke`

The default configuration is publication-scale; the smoke configuration is
the entry point for exploration.
:::

:::{grid-item-card} Near-field machinery — understand the differentiator
:link: ../user-guide/volume-fmm-workflow
:link-type: doc

**See:** why the self box and List 1 neighbors cannot be treated like ordinary
far-field particles, how table lookup replaces point quadrature there, and how
symmetry reduces storage.

Continue from the workflow page to {doc}`../user-guide/nearfield_symmetry`
when you want the storage and reconstruction details.
:::

::::

## Which example should I copy?

- **I am new to the library:** {doc}`../getting-started/first-volume-potential`.
- **I want a 3-D manufactured solve and diagnostic plots:** `poisson3d.py`.
- **I want to couple volume and boundary integral machinery on a curved
  domain:** the 2-D Poisson notebook.
- **I want Helmholtz with the Laplace-table smooth correction:** `helmholtz2d.py`
  or `helmholtz3d.py`.
- **I want a substantial scattering application:** `branched_flow_helmholtz2d.py`.

The full operational notes — first-run table cost, cache names, smoke modes,
device selection, and notebook setup — remain on {doc}`index`.


## Regenerating computed figures

On a configured OpenCL host:

```bash
export PYOPENCL_CTX=portable:0
uv run --with matplotlib python doc/tools/render_gallery.py
```

The renderer delegates to the maintained examples rather than reimplementing
their numerical problems:

- `laplace2d.py` writes a four-panel source / FMM / exact / log-error overview
  plus the actual volume tree when
  `VOLUMENTIAL_LAPLACE2D_OUTPUT_DIR` is set;
- `poisson3d.py` supplies its existing orthogonal slices and 3-D error cloud;
- `branched_flow_helmholtz2d.py --smoke` supplies its existing
  refractive-index / intensity / field figure.

The outputs and `manifest.json` land under
`doc/source/images/generated/`. This is an explicit maintainer operation:
table builds and OpenCL execution never become a prerequisite for reading or
building the documentation.
