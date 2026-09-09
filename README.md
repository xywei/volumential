# Volumential

[![CI](https://github.com/xywei/volumential/actions/workflows/ci.yml/badge.svg)](https://github.com/xywei/volumential/actions/workflows/ci.yml)

Volumential (VOLUME poteNTIAL) provides a toolset to solve volume potential
integral equations efficiently via the Fast Multipole Method.

## Quick Start

Install an OpenCL runtime first (for example, in conda:
`conda install pyopencl pocl`), then use `uv` for dependency resolution:

```bash
uv sync --active --extra test
uv run python examples/laplace2d.py
uv run python examples/helmholtz2d.py
uv run python examples/helmholtz3d.py
uv run python examples/poisson3d.py
```

`uv.lock` pins the inducer-stack dependencies (`boxtree`, `sumpy`, `loopy`,
`pyopencl`, `pytools`, `modepy`, `arraycontext`, `meshmode`, `pytential`) to
main-branch Git sources rather than to released wheels, because those projects
publish releases rarely and the wheels have shipped defects that silently
corrupt adaptive-tree results. Add `--extra fmmlib` for the FMMLib backend; see
`DEVELOPMENT.md` for the full provisioning recipe, including the post-install
traversal sanity check that every experiment environment must pass.

## Repository Layout

- `volumential/`: library source (see the architecture map below)
- `test/`: pytest suite
- `examples/`: maintained end-to-end examples (`laplace2d.py`, `laplace3d.py`,
  `helmholtz2d.py`, `helmholtz3d.py`, `poisson3d.py`,
  `branched_flow_helmholtz2d.py`, the two `*_split_p_convergence.py` drivers,
  and the `*.ipynb` notebooks)
- `benchmarks/`: reproducible benchmark drivers that emit CSV/JSON evidence;
  each has a lightweight `--mode smoke` for CI and a `--mode full` for
  controlled runs (see `benchmarks/README.md`)
- `doc/`: Sphinx documentation
- `ruff.toml`: the single lint configuration (line length 85, py311)
- `DEVELOPMENT.md`: environment provisioning and day-to-day workflow

## Architecture Map

Geometry and box meshes

- `meshgen.py`: box mesh generation over a boxtree tree-of-boxes
  (`MeshGen1D/2D/3D`, `make_uniform_cubic_grid`, `build_geometry_info`).
- `tree_interactive_build.py`: interactive/adaptive refinement and coarsening
  with 2:1 level restriction.
- `geometry.py`: `BoundingBoxFactory`, `BoxFMMGeometryFactory` and the
  `BoxFMMGeometryData` handed to the wranglers.

Near-field tables

- `nearfield_potential_table.py`: `NearFieldInteractionTable`, the near-field
  quadrature tables and their symmetry-reduced payloads.
- `table_manager.py`: `NearFieldInteractionTableManager`, the SQLite-backed
  table cache (schema `2.1.0`).
- `list1_gallery.py`: enumeration of the List 1 interaction cases.
- `list1_symmetry.py`: symmetry operations acting on case vectors.
- `orbit_arithmetic.py`: orbit canonicalization and sign bookkeeping.
- `rke_table_assembly.py`: certified RKE assembly of fixed-parameter tables.
- `singular_integral_2d.py`: 2D Duffy-transform singular quadrature
  (`box_quad`).
- `list1.py`: the on-device List 1 near-field evaluators
  (`KernelScalingPolicy`, `NearFieldEvalBase`, `NearFieldFromCSR`) that read
  table data through the CSR interaction lists.

Volume FMM

- `volume_fmm.py`: `drive_volume_fmm` plus interpolation of the box-mesh
  potential onto user targets.
- `expansion_wrangler_interface.py`: the wrangler interface and the
  tree-independent data it needs.
- `expansion_wrangler_fpnd.py`: the FPND wranglers (sumpy and FMMLib backends)
  and the near-field Helmholtz split. This module is being reorganized into a
  `volumential/wranglers/` subpackage; the current module path stays importable
  through compatibility re-exports.

Fields, sources and post-processing

- `interpolation.py`: transfer between meshmode discretizations and the box
  mesh.
- `lagrange.py`: barycentric Lagrange basis utilities.
- `gaussian.py`: Gaussian source fixtures for free-space demos.
- `function_extension.py`: layer-potential source extension with regularity
  constraints.
- `symbolic.py`: symbolic expression evaluation helpers for source densities.

Support

- `tools.py`: the loopy kernel cache wrapper and scalar-field evaluation.
- `opcounters.py`: explicit operation counters for benchmark drivers.
- `__init__.py`, `version.py`: public surface (`__all__`), version and the
  kernel revision used in cache keys.
- `qbfem/`: a 2019 finite-element experiment that nothing else in the tree
  imports; it is kept only for reference and is not part of the supported API.

Debug tip: when validating source-node evaluations, set
`VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE=1` to fail fast if a traversal is built
with separate-but-identical source/target arrays (use `targets=None` to build a
true coincident tree).

## Near-Field Symmetry and Cache Format

- Near-field table storage uses orbit canonicalization over
  `(source_mode, target_mode, interaction_case)` and stores only canonical
  entries.
- Derivative kernels are supported with sign-aware orbit metadata:
  runtime lookup applies a per-entry sign factor when reconstructing from
  canonical entries.
- SQLite cache schema `2.1.0` stores table content in the `payload` blob only;
  legacy dense blob columns were removed.
- Symmetry-reduced payloads persist only finite canonical data arrays
  (`reduced_entry_ids` + `reduced_data`) and do not store NaN sentinels.

## Documentation

[Browse the documentation online.](http://xiaoyu-wei.com/docs/volumential/)

Build it locally with `uv sync --active --extra doc` and `make -C doc html`.

## License

Volumential is developed and released under the terms of the MIT license,
though it also makes use of third-party packages under their own licensing
terms. See the [LICENSE](./LICENSE.md) file for details.

## Acknowledgements

We would like to thank people and organizations that contributed to or supported the
Volumential project, without whom the project would not have been possible.

The research that started the Volumential project was supported by the
[National Science Foundation][nsf] under grant DMS-1654756,
and by the [Department of Computer Science][uiuc-cs] at the
[University of Illinois at Urbana-Champaign][uiuc].
Part of the work was performed while the authors were participating in
the [HKUST][hkust]-[ICERM][icerm] workshop "Integral Equation Methods, Fast
Algorithms and Their Applications to Fluid Dynamics and Materials
Science" held in 2017.

Thanks very much to the [Department of Mathematics][hkust-math] at
[Hong Kong University of Science and Technology][hkust]
for funding Xiaoyu Wei to work on the project
as a PhD student under the Postgraduate Studentship and
the Overseas Research Award.

The project's name `volumential` [courtesy of Andreas Klöckner][volumential-name].

[nsf]: https://www.nsf.gov/
[hkust-math]: https://www.math.ust.hk/
[hkust]: https://www.ust.hk/home
[icerm]: https://icerm.brown.edu/
[uiuc-cs]: https://cs.illinois.edu/
[uiuc]: https://illinois.edu/
[volumential-name]: https://gitlab.tiker.net/xywei/volumential/issues/2
