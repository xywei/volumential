# Visual gallery

If you are deciding whether Volumential fits a problem, start here rather than
with the API reference. Each card below shows the committed output of a
maintained example, rendered at the example's full settings by
`doc/tools/render_gallery.py`, and names the settings and the command that
produced it. The near/far diagram is the one schematic, and it is labeled as
such. A thumbnail is a crop; select it to open the full figure. The
documentation build runs none of these examples.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Laplace 2-D: a first volume potential

```{image} ../_static/gallery/laplace2d/laplace2d_overview.svg
:alt: Four panels over the square from -0.5 to 0.5: the source density, a positive peak inside a faint negative ring; the computed potential and the Gaussian reference, which look identical; and the pointwise error on a logarithmic scale, at most about 8e-11, in nested squares that follow the FMM boxes.
:class: gallery-thumb
:width: 100%
```

**Start here.** The source $f = -\Delta u$, the computed potential $u_h$, the
whole-space Gaussian reference $u = e^{-160 \lVert \boldsymbol{x} \rVert^2}$
and the pointwise error $|u_h - u|$ on $[-1/2, 1/2]^2$.

`examples/laplace2d.py`, full settings: quadrature order 9, 6 mesh levels,
multipole order 20, 82944 quadrature nodes. The run printed
`Error = 8.410442587858608e-11`, the maximum of $|u_h - u|$ over the nodes.

```bash
python doc/tools/render_gallery.py laplace2d \
    --pyopencl-ctx portable:0 --full
```

+++
{doc}`../getting-started/first-volume-potential` builds this run step by
step, with the tree the FMM used.
:::

:::{grid-item-card} Poisson 3-D: where the error lives

```{image} ../_static/gallery/poisson3d/poisson3d_slices.png
:alt: A three by three grid of slices through the cube from -0.5 to 0.5, on the planes z = 0, y = 0 and x = 0. The manufactured solution, a positive and a negative Gaussian, and the FMM result look identical; the log10 error panels show errors up to about 1e-6 in a box-shaped region around the positive Gaussian and about 1e-7 elsewhere.
:class: gallery-thumb
:width: 100%
```

Two Gaussians, one positive and one negative, on $[-1/2, 1/2]^3$. Rows are
the planes $z = 0$, $y = 0$ and $x = 0$. Columns are the manufactured solution
(titled "Exact"), the FMM potential interpolated to 180 × 180 points per
plane, and $\log_{10}$ of their difference. The error is largest, about
$10^{-6}$, in a box-shaped region around the positive Gaussian, with stripes
that trace the leaf cells. "Exact" is the whole-space solution of
$-\Delta u = f$; the example integrates over the cube only, and its Gaussians
are effectively zero at the cube's boundary.

`examples/poisson3d.py`, full settings: quadrature order 7, 5 mesh levels
(16³ leaf cells), multipole order 16, near-field table quadrature orders 16
(regular) and 80 (radial). At the quadrature nodes the run printed a maximum
absolute error of `5.498458e-07` and a relative L2 error of `2.340261e-06`;
the slice errors include interpolation to points between the nodes.

```bash
python doc/tools/render_gallery.py poisson3d \
    --pyopencl-ctx portable:0 --full
```

+++
The example and its notebook: {ref}`poisson3d-example` and
{doc}`notebooks/poisson3d_volumential`, which adds an interactive isosurface
view when Plotly is installed.
:::

:::{grid-item-card} Helmholtz 2-D: branched flow

```{image} ../_static/gallery/branched-flow/branched_flow.png
:alt: Three stacked panels over the square from -8 to 8. Top, the refractive-index perturbation, smooth random blobs confined to a rectangle. Middle, the normalized intensity of a plane wave entering from the left, which breaks into bright branching filaments separated by dark regions. Bottom, the real part of the total field, straight wavefronts on the left that bend and break up where the intensity branches.
:class: gallery-thumb
:width: 100%
```

A plane wave with $k = 12$, incident along $+x$, through a smooth random
medium. The full figure stacks the refractive-index perturbation $n - 1$, the
intensity $|u|^2$ normalized by the mean incident intensity, and
$\operatorname{Re} u$. The thumbnail is centered on the intensity, where the
wave breaks into branching filaments that keep spreading past the medium. The
solve is a free-space Lippmann–Schwinger equation on $[-8, 8]^2$, with no
artificial boundary or PML.

`examples/branched_flow_helmholtz2d.py`, default (full) configuration:
quadrature order 4, 9 levels, 1048576 points, the `fmmlib` far field; 2048
Gaussian bumps (refractive-index RMS 0.05, correlation length 0.55, seed 17) on
the core $[-6.4, 3] \times [-5, 5]$ with a transition layer of width 0.5; GMRES
to a tolerance of $10^{-8}$ without a preconditioner. The run printed 34 GMRES
iterations and a relative true residual of `9.567252072103188e-09`. It needs
`pyfmmlib`.

```bash
python doc/tools/render_gallery.py branched-flow \
    --pyopencl-ctx portable:0 --full
```

+++
Run command, smoke mode and cost warning: {ref}`branched-flow-example`.
:::

:::{grid-item-card} Near field and far field (schematic)

```{image} ../_static/gallery/near-far-anatomy.svg
:alt: Schematic. A five by five block of boxes with the target box in the middle. Its eight neighbors are marked as table lookups; the outer ring of boxes is marked as handled by the FMM with multipole and local expansions.
:class: gallery-thumb gallery-thumb-whole
:width: 100%
```

**Schematic, not computed output.** Why the singular neighborhood of a target
box leaves the ordinary particle FMM path, what is tabulated, and what can be
reused when the source density changes. Read it before the detailed
implementation pages.

The diagram is drawn by hand; its source is
`doc/gallery-src/near-far-anatomy.svg`.

+++
{doc}`../user-guide/near-field-anatomy`
:::

:::{grid-item-card} Poisson 2-D: a starfish domain with AMR

**No figure.** The notebook `poisson2d_pytential_volumential.ipynb`
couples Volumential to Pytential, enforces Dirichlet data with a harmonic
boundary correction, and compares uniform refinement with boundary-focused
adaptive refinement. Its figures exist only when the notebook runs: it is
committed without outputs, the documentation does not execute notebooks, and
no maintained script produces them, so there is no computed figure to show
here. Run the notebook to see them.

+++
{doc}`notebooks/poisson2d_pytential_volumential`
:::

::::

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
shows the files committed under `doc/source/_static/gallery/`. All of the
computed figures above come from one command, run in the Volumential
environment with matplotlib (and `pyfmmlib` for branched flow) installed:

```bash
python doc/tools/render_gallery.py all --full --pyopencl-ctx portable:0
```

The command on each card regenerates that figure alone. Without `--full` the
renderer uses the examples' smoke settings, which run in seconds and show that
the figure path works but do not resolve the problems, so their figures are
not results. `doc/source/_static/gallery/manifest.json` records the revision,
command, environment and package versions behind each committed figure, and
{doc}`../development/gallery-assets` describes the renderer and the rules for
changing a figure.

## Need the operational details?

The main {doc}`index` page remains the source of truth for cache filenames,
device-selection behavior, smoke modes and the cost class of every example.
