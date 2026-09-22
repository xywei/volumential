# Gallery assets

The documentation treats figures as reproducible evidence rather than as
decorative screenshots. Numerical gallery images come from maintained examples;
Sphinx never executes those examples.

The committed, device-independent concept diagrams can be regenerated anywhere:

```bash
python doc/tools/render_gallery.py concepts
```

Numerical assets need the normal Volumential/OpenCL environment plus matplotlib.
By default the tool requests smoke/reduced settings:

```bash
uv run --with matplotlib python doc/tools/render_gallery.py laplace2d
uv run --with matplotlib python doc/tools/render_gallery.py poisson3d
uv run --with matplotlib python doc/tools/render_gallery.py branched-flow
```

Use `all` to run every group. Pass `--full` only when an intentionally
full-resolution result is being curated; it may turn the branched-flow example
into a publication-scale run and can require the `fmmlib` extra.

The output root is `doc/source/_static/gallery/`. The numerical subdirectories
are owned by the corresponding maintained examples:

| Directory | Producer |
| --- | --- |
| `laplace2d/` | `examples/laplace2d.py` via `VOLUMENTIAL_GALLERY_OUTPUT_DIR` |
| `poisson3d/` | `examples/poisson3d.py` via `VOLUMENTIAL_POISSON3D_OUTPUT_DIR` |
| `branched-flow/` | `examples/branched_flow_helmholtz2d.py --output-dir ...` |

## Policy

- Do not call `render_gallery.py` from `conf.py`, a Sphinx extension, or a
  documentation build hook. A docs build must remain possible without a compute
  device.
- Prefer a maintained example over a second docs-only implementation of the same
  numerical problem.
- Commit the curated static image that a page uses, together with the source or
  command that regenerates it.
- Smoke output is appropriate for explaining a mechanism. Do not quote its
  timing as a benchmark; {doc}`../benchmarks/index` defines the evidence needed
  for performance claims.
- Interactive HTML (for example the Poisson 3-D Plotly isosurface) is useful for
  local inspection, but the documentation should also carry a static fallback.

`examples/laplace2d.py` is the reference pattern: setting
`VOLUMENTIAL_GALLERY_OUTPUT_DIR` does not change the computation. It simply
writes a four-panel source/computed/exact/error SVG and a tree SVG from the data
and tree that the maintained example already produced.
