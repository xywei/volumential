# Visual gallery

If you are deciding whether Volumential fits a problem, start here rather than
with the API reference. The first three cards show the committed output of a
maintained example, rendered at the example's full settings by
`doc/tools/render_gallery.py`, with the settings, the numbers the run printed,
and the command that regenerates the figure. The near/far card is a
schematic and says so; the Poisson 2-D card has no figure and says why. A
thumbnail is a crop: select it to open the full figure. The documentation
build runs none of these examples.

:::::{card} Laplace 2-D: a first volume potential

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
:columns: 12 12 5 5

```{image} ../gallery/laplace2d/laplace2d_overview.svg
:alt: Four panels over the square: the source, the computed potential, the Gaussian reference, and the pointwise error on a logarithmic scale, at most about 8e-11.
:class: gallery-thumb
:width: 100%
```

:::

:::{grid-item}
:columns: 12 12 7 7

**Start here.** The source $f = -\Delta u$, the computed potential $u_h$, the
whole-space Gaussian reference $u = e^{-160 \lVert \boldsymbol{x} \rVert^2}$
and the pointwise error $|u_h - u|$ on $[-1/2, 1/2]^2$.

`examples/laplace2d.py`, full settings: quadrature order 9, 6 mesh levels,
multipole order 20, 82944 quadrature nodes. The run printed
`Error = 8.410442587858608e-11`, the maximum of $|u_h - u|$ over the nodes.

:::

::::

```bash
python doc/tools/render_gallery.py laplace2d \
    --pyopencl-ctx portable:0 --full
```

+++
{doc}`../getting-started/first-volume-potential` builds this run step by
step, with the tree the FMM used.
:::::

:::::{card} Poisson 3-D: where the error lives

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
:columns: 12 12 5 5

```{image} ../gallery/poisson3d/poisson3d_slices.png
:alt: A three by three grid of slices through the cube on the planes z = 0, y = 0 and x = 0. The exact and FMM columns look identical; the log10 error is largest, about 1e-5, around the two Gaussians and falls to 1e-12 and below away from them, in blocks that follow the boxes of the tree.
:class: gallery-thumb
:width: 100%
```

:::

:::{grid-item}
:columns: 12 12 7 7

Two Gaussians of opposite sign, each $e^{-240 r^2}$, on $[-1/2, 1/2]^3$, cut
by the planes $z = 0$, $y = 0$ and $x = 0$. The columns are the manufactured
solution (titled "Exact"), the FMM potential interpolated to 180 × 180 points
per plane, and $\log_{10}$ of their difference. "Exact" is the whole-space
solution of $-\Delta u = f$, while the example integrates over the cube only;
both Gaussians are below $e^{-40}$ on the cube's boundary, so the source left
out is far below the errors shown. The error is largest around the two
Gaussians and falls by several orders of magnitude away from them, in blocks
that follow the boxes of the tree.

`examples/poisson3d.py`, full settings: quadrature order 7, 5 mesh levels
(16³ leaf cells), multipole order 16, near-field table quadrature orders 16
(regular) and 80 (radial). At the quadrature nodes the run printed a maximum
absolute error of `5.768837e-07` and a relative L2 error of `1.190132e-07`.
The slices reach about $10^{-5}$ because they add interpolation from the nodes
to the points between them.

:::

::::

```bash
python doc/tools/render_gallery.py poisson3d \
    --pyopencl-ctx portable:0 --full
```

+++
The example: {ref}`poisson3d-example`. Its notebook,
{doc}`notebooks/poisson3d_volumential`, adds an interactive isosurface view
when Plotly is installed.
:::::

:::::{card} Helmholtz 2-D: branched flow

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
:columns: 12 12 5 5

```{image} ../gallery/branched-flow/branched_flow.png
:alt: The normalized intensity of a plane wave entering from the left, which breaks into bright branching filaments separated by dark regions. The full figure also shows the refractive-index perturbation and the real part of the field.
:class: gallery-thumb gallery-thumb-middle-panel
:width: 100%
```

:::

:::{grid-item}
:columns: 12 12 7 7

A plane wave with $k = 12$, incident along $+x$, through a smooth random
medium, solved as a free-space Lippmann–Schwinger equation on $[-8, 8]^2$
with no artificial boundary or PML. The figure stacks the refractive-index
perturbation $n - 1$, the intensity $|u|^2$ normalized by the mean incident
intensity, and $\operatorname{Re} u$; each color scale is clipped at the 99.5th
percentile of the plotted magnitude. The thumbnail shows the intensity, where the wave
breaks into branching filaments that keep spreading past the medium.

`examples/branched_flow_helmholtz2d.py`, default (full) configuration:
quadrature order 4, 9 levels, 1048576 points, the `fmmlib` far field, GMRES
to a tolerance of $10^{-8}$ without a preconditioner. The medium is a sum of
2048 Gaussian bumps of width 0.55 (the example's `correlation_length`
parameter; refractive-index RMS 0.05, seed 17) on $[-6.4, 3] \times [-5, 5]$,
faded out over a layer of width 0.5. The run printed 34 GMRES iterations and a
relative true residual of `9.567252072103188e-09`. It needs `pyfmmlib`.

:::

::::

```bash
python doc/tools/render_gallery.py branched-flow \
    --pyopencl-ctx portable:0 --full
```

+++
Run command, smoke mode and cost warning: {ref}`branched-flow-example`.
:::::

:::::{card} Near field and far field (schematic)

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
:columns: 12 12 5 5

```{image} ../gallery/near-far-anatomy.svg
:alt: Schematic. A five by five block of boxes. The target box and its eight neighbors are table lookups, the target box's own interaction included; the outer ring is handled by the FMM with multipole and local expansions.
:class: gallery-thumb gallery-thumb-whole
:width: 100%
```

:::

:::{grid-item}
:columns: 12 12 7 7

**Schematic, not computed output.** Why the singular neighborhood of a target
box leaves the ordinary particle FMM path, what is tabulated, and what can be
reused when the source density changes. Read it before the detailed
implementation pages.

The diagram is drawn by hand and edited directly in
`doc/source/gallery/near-far-anatomy.svg`.

:::

::::

+++
{doc}`../user-guide/near-field-anatomy`
:::::

:::::{card} Poisson 2-D: a starfish domain with AMR

**No figure.** The notebook `poisson2d_pytential_volumential.ipynb` couples
Volumential to Pytential, enforces Dirichlet data with a harmonic boundary
correction, and compares uniform refinement with boundary-focused adaptive
refinement. Its figures exist only when the notebook runs: it is committed
without outputs, the documentation does not execute notebooks, and no
maintained script produces them, so there is no computed figure to show here.
Run the notebook to see them. A gallery figure of an adaptive calculation is
tracked in [#171](https://github.com/xywei/volumential/issues/171).

+++
{doc}`notebooks/poisson2d_pytential_volumential`
:::::

## Which example should I copy?

- **New to the library:** {doc}`../getting-started/first-volume-potential`.
- **A 3-D manufactured solve with spatial error diagnostics:** `poisson3d.py`.
- **Volume and boundary integral machinery together, on a curved domain:** the
  2-D Poisson notebook.
- **Helmholtz with the Laplace-table smooth correction:** `helmholtz2d.py` or
  `helmholtz3d.py`, which are convergence studies in the quadrature order.
- **A large scattering application:** `branched_flow_helmholtz2d.py`.

Use the convergence studies for numerical behavior, and read
{doc}`../benchmarks/index` before quoting a timing.

## Regenerate the figures

The documentation build deliberately runs no scientific workload; it only
shows the files committed under `doc/source/gallery/`. All of the computed
figures above come from one command, run in the Volumential environment with
matplotlib (and `pyfmmlib` for branched flow) installed:

```bash
python doc/tools/render_gallery.py all --full --pyopencl-ctx portable:0
```

The command on each card regenerates that figure alone. Without `--full` the
renderer uses the examples' smoke settings, which run in seconds and show that
the figure path works but do not resolve the problems, so their figures are
not results. `doc/source/gallery/manifest.json` records the revision, command,
environment, device type and package versions behind each computed figure,
and {doc}`../development/gallery-assets` describes the renderer and the rules
for changing a figure.

## Need the operational details?

The main {doc}`index` page remains the source of truth for cache filenames,
device-selection behavior, smoke modes and the cost class of every example.
