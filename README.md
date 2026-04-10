# Volumential

![pipeline](https://gitlab.tiker.net/xywei/volumential/badges/master/pipeline.svg)
![coverage](https://gitlab.tiker.net/xywei/volumential/badges/master/coverage.svg)

Volumential (VOLUME poteNTIAL) provides toolset to solve volume potential integral equations
efficiently via Fast Multipole Method.

## Repository Layout

- `volumential/`: library source
- `test/`: pytest suite
- `examples/`: maintained end-to-end examples (`laplace2d.py`, `laplace3d.py`,
  `helmholtz2d.py`, `helmholtz3d.py`, `poisson3d.py`,
  `poisson3d_volumential.ipynb`, `poisson2d_pytential_volumential.ipynb`,
  `helmholtz3d_volumential.ipynb`)
- `doc/`: Sphinx documentation
- `DEVELOPMENT.md`: local/remote development environment guide

Legacy `contrib/`, `benchmarks/`, `experiments/`, and `docker/` trees were
removed as part of repository cleanup.

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

For a fully reproducible local + remote setup, see `DEVELOPMENT.md`.

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
