# Gallery assets

The documentation treats figures as reproducible evidence rather than as
decorative screenshots. Numerical gallery images come from maintained examples
through `doc/tools/render_gallery.py`; Sphinx never executes those examples and
never calls the renderer. A documentation build only consumes the files
committed under `doc/source/gallery/`.

That directory holds two kinds of file. The computed figures and their
`manifest.json` are written by the renderer, one subdirectory per target. The
schematics (`near-far-anatomy.svg`, `volume-fmm-workflow.svg`) are drawn by
hand and edited in place; nothing generates them, and every page that shows
one says it is a schematic.

## Regenerating

The renderer needs the normal Volumential/OpenCL environment plus matplotlib,
and an explicit PyOpenCL context selector, given either as `PYOPENCL_CTX` or as
`--pyopencl-ctx`:

```bash
export PYOPENCL_CTX=portable:0
uv run --with matplotlib python doc/tools/render_gallery.py laplace2d
uv run --with matplotlib python doc/tools/render_gallery.py poisson3d
uv run --with matplotlib python doc/tools/render_gallery.py branched-flow
```

Without a selector the renderer stops instead of letting PyOpenCL pick a device.
It also removes `PYOPENCL_TEST` from the examples' environment, because
`pyopencl.create_some_context` prefers that variable over `PYOPENCL_CTX`, and it
runs the examples with standard input closed, so PyOpenCL cannot prompt for a
choice either. Every inherited `VOLUMENTIAL_*` variable is removed as well:
some of them change a computation (`poisson3d.py` reads its resolution from
`VOLUMENTIAL_POISSON3D_*`, the library reads cache and build switches), and a
render is defined by the renderer's own settings alone.

Smoke settings are the default. They run in seconds and show that the figure
path works, but they do not resolve the problems: the smoke runs of `laplace2d`
and `poisson3d` have errors of order one, and the `branched-flow` smoke domain is
little more than a wavelength across, too short for branches to form. `--full`
switches every target to its example's full settings, and `all` runs every
target. Committed figures are full-settings renders from one command:

```bash
uv run --with matplotlib python doc/tools/render_gallery.py all --full --pyopencl-ctx portable:0
```

At full settings `branched-flow` is the example's publication-pilot
configuration, about a million unknowns with the `fmmlib` far-field backend, so
it needs `pyfmmlib` as {ref}`branched-flow-example` describes, and it dominates
the run. `poisson3d` builds a 3-D near-field table the first time it runs.

| Target | Producer | Figures copied into the gallery |
| --- | --- | --- |
| `laplace2d` | `examples/laplace2d.py` via `VOLUMENTIAL_GALLERY_OUTPUT_DIR` | `laplace2d_overview.svg`, `laplace2d_tree.svg` |
| `poisson3d` | `examples/poisson3d.py` via `VOLUMENTIAL_POISSON3D_OUTPUT_DIR` | `poisson3d_slices.png` |
| `branched-flow` | `examples/branched_flow_helmholtz2d.py --output-dir ...` | `branched_flow.png` |

Before any example runs, the renderer imports the packages the examples import,
in the examples' interpreter, to record their versions, and it stops if
`volumential` would come from anywhere but this checkout. Then, for each target,
it

1. deletes the files that the previous run recorded for the target in the
   manifest, together with the figures it is about to write;
2. runs the example with `PYOPENCL_CTX`, `PYTHONHASHSEED=0`, `MPLBACKEND=Agg`
   and the smoke setting in its environment, and with
   `build/gallery-work/<mode>/<target>/` as its working and output directory.
   Git ignores that directory. Everything the example writes lands there: data
   files, interactive HTML, figures the gallery does not use, and the
   near-field tables it caches, so each mode and target keeps its own tables
   and none of them is read from or written to the repository root;
3. fails if any expected figure is missing, and otherwise copies only the
   figures into `doc/source/gallery/<target>/`;
4. records the run in `doc/source/gallery/manifest.json`.

A table cached under `build/gallery-work/` is reused by later renders in the
same mode. Delete the directory to rebuild the tables; a fresh clone, which is
where committed figures come from, always builds them.

`--output-dir` and `--work-dir` move the gallery root and the scratch
directory, for example to render a preview outside the documentation tree.
`*.png` is ignored repository-wide; `doc/source/gallery/` is exempt, so the
PNG figures of `poisson3d` and `branched-flow` can be committed there.

The gallery sits outside `html_static_path` on purpose. Sphinx copies every
image a page shows into `_images/`, so a gallery under `_static/` would be
published twice. `manifest.json` is therefore not part of the built site; read
it in the repository.

## The manifest

`manifest.json` has one entry per target under `targets`, so regenerating one
target leaves the records of the others in place. An entry holds

| Field | Meaning |
| --- | --- |
| `revision` | the commit checked out when the renderer ran |
| `dirty` | whether tracked files differed from that commit; untracked files and the gallery directory itself are not counted |
| `mode` | `smoke` or `full` |
| `pyopencl_ctx` | the context selector string passed to the example |
| `device_type` | the type of device that selector resolved to in the examples' environment (`CPU` or `GPU`), not its name |
| `regenerate` | the renderer invocation that reproduces the entry |
| `command` | the example invocation, with the script path relative to the repository root |
| `working_directory` | where the example ran |
| `environment` | the variables the renderer set for the example; `null` means it removed the variable |
| `outputs` | the copied figures, relative to the manifest |
| `versions` | as imported by the examples' interpreter: `volumential`, `pyopencl`, `numpy`, `matplotlib` and Python, and the distributions that decide the computed digits, `boxtree`, `sumpy`, `loopy`, `pymbolic`, `modepy`, `pytential` and `pyfmmlib` (`null` when not installed) |

The manifest, not `uv.lock`, describes the environment a figure was rendered
in; the two need not agree.

Paths are repository-relative. A directory outside the repository appears as
`<output-dir>/...` or `<work-dir>/...`, never as an absolute path. The manifest
deliberately carries nothing that identifies a machine: no host or user name, no
absolute path, and no device or CPU name. The examples' own console output does
name the device, so logs of a gallery run do not belong in the repository.

## Determinism

Rendering is kept free of incidental variation so that regenerating a figure in
the same environment does not churn its bytes:

- the Laplace figures have a fixed size, font size and DPI of 150 for their
  rasterized shaded layers, a fixed SVG hash salt, and no date or creator
  metadata;
- the PNG figures of `poisson3d` and `branched-flow` are saved at a fixed 150 DPI
  without the Matplotlib version tag;
- `branched-flow` draws its random medium from a fixed seed, and the renderer
  fixes `PYTHONHASHSEED`.

A different device, driver or library version can still change the computed
numbers and therefore the pixels; the manifest's selector, device type and
versions say which environment produced a committed figure.

## CI preview

The `Examples (Smoke)` job of {doc}`ci` runs

```bash
python doc/tools/render_gallery.py laplace2d --output-dir build/gallery --pyopencl-ctx "$PYOPENCL_CTX"
```

and uploads `build/gallery/` (the two Laplace figures and their manifest) as the
`gallery-laplace2d-*` artifact, kept for 14 days. The upload is attempted on
every outcome, so the artifact exists whenever the renderer wrote files. On a
pull request the recorded revision is the merge commit GitHub builds for the
branch, not the branch head. The smoke settings are chosen to run quickly, not
to resolve the problem, so the preview shows that the figure path works, not how
accurate the method is.

## Policy

- Do not call `render_gallery.py` from `conf.py`, a Sphinx extension, or a
  documentation build hook. A docs build must remain possible without a compute
  device.
- Prefer a maintained example over a second docs-only implementation of the same
  numerical problem.
- Commit the curated static image that a page uses together with the updated
  `manifest.json`. Regenerate committed assets from a clean checkout of a pushed
  commit, so that `dirty` is `false` and the recorded revision can be checked
  out.
- A figure caption names the example, whether smoke or full settings produced
  it, and the `regenerate` command. Anything not computed by an example is
  labeled a schematic.
- A computed figure compared against a reference says what the reference is. A
  whole-space solution is a reference for a box integral only where the source
  outside the box is negligible at the error level shown.
- Smoke output is appropriate for explaining a mechanism. Do not quote its
  accuracy or timing as a result; {doc}`../benchmarks/index` defines the
  evidence needed for performance claims.
- Interactive HTML (for example the Poisson 3-D Plotly isosurface) stays in the
  work directory; it is useful for local inspection, but a page needs a static
  figure.

`examples/laplace2d.py` is the reference pattern: setting
`VOLUMENTIAL_GALLERY_OUTPUT_DIR` does not change the computation. After the
solve, the example writes two SVG files from the data and tree it has already
produced. `laplace2d_overview.svg` has four panels: the source
$f = -\Delta u$, the computed volume potential $u_h$, the reference
$u = e^{-\alpha |x|^2}$, and the pointwise error $|u_h - u|$. Each shows the
values at the quadrature nodes, shaded by linear interpolation over a Delaunay
triangulation of the nodes. $u_h$ and $u$ share one color scale; the error uses
a logarithmic scale whose floor is $\epsilon \max|u|$, with $\epsilon$ the
double-precision machine epsilon, so differences below the rounding of $u$ are
drawn at the floor. The reference is
the whole-space solution of $-\Delta u = f$, while the example integrates $f$
over the box $[-0.5, 0.5]^2$ only. Outside the box the Gaussian factor is at most
$e^{-40}$ for $\alpha = 160$, so the difference between the two is at rounding
level, but the reference is not the exact value of the box integral.
`laplace2d_tree.svg` shows the tree the FMM traversed, with the quadrature nodes
as dots and heavier outlines for coarser levels; the example builds it from the
mesh, so its leaves are the mesh cells. The figure titles carry the settings
(smoke or full, quadrature order, mesh levels, multipole order, node count) and
the maximum error of the run.
