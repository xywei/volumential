# Volumential

[![CI](https://github.com/xywei/volumential/actions/workflows/ci.yml/badge.svg)](https://github.com/xywei/volumential/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Volumential (VOLUME poteNTIAL) evaluates volume potentials over box-shaped
domains with the Fast Multipole Method: the far field is an ordinary particle
FMM over the volume quadrature nodes, while the near field — where the
integrand is singular and no point rule converges — is read from precomputed,
symmetry-reduced interaction tables. Laplace, Helmholtz and Yukawa are
supported in 2D and 3D, for potential and target gradient, on uniform and
adaptively refined 2:1-balanced trees.

## Install

```bash
micromamba create -n volumential-dev -c conda-forge -c nodefaults \
  python=3.12 pyopencl pocl scipy numpy && micromamba activate volumential-dev
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
git clone https://github.com/xywei/volumential.git && cd volumential
uv sync --extra test
```

(`micromamba activate` is a shell function: in a shell that has not been
`micromamba shell init`-ed, run `eval "$(micromamba shell hook -s bash)"`
first. `UV_PROJECT_ENVIRONMENT` is what points `uv` at the conda environment —
without it `uv` builds a `.venv` with no OpenCL runtime. See
[Installation][docs-install].)

Released wheels of the `inducer` stack have shipped defects that corrupt
adaptive-tree results *silently*, so `uv.lock` pins those dependencies (and
`pyfmmlib`) to Git sources. Run the traversal sanity check before trusting a
fresh environment — see [Installation][docs-install].

## Run

```bash
export PYOPENCL_CTX=portable:0        # otherwise the device is picked for you
uv run python examples/laplace2d.py
```

That evaluates a 2D Laplace volume potential against a manufactured Gaussian
solution and prints the error. `VOLUMENTIAL_EXAMPLE_SMOKE=1` runs the same
thing in seconds at reduced order. The
[annotated twenty-line version][docs-first] explains what each stage does.

## Documentation

The site is built from [`doc/source`](./doc/source). The currently published
build is at <https://xiaoyu-wei.com/docs/volumential/>; a GitHub Pages
deployment from `main` is tracked by
[#145](https://github.com/xywei/volumential/issues/145).

- [Getting started][docs-getting-started] — install, a first volume potential,
  device selection
- [User guide][docs-user-guide] — the volume-FMM workflow, near-field tables
  and their symmetry reduction, table build routing, derivatives, the Helmholtz
  split, what is validated
- [Design notes][docs-design-notes] — windowed singular channels and certified
  assembly, ORBIT canonicalization
- [Benchmarks and reproducibility][docs-benchmarks] — the drivers, metadata
  sidecars, what a promoted measurement must record
- [API reference][docs-api] — one page per module
- [Development][docs-development] — contributing, tests and markers, CI and the
  review bots, release and versioning
- [Changelog][docs-changelog]

Build it locally with `uv sync --extra test --extra doc` and
`sphinx-build -W --keep-going -b html doc/source doc/build/html`.

## Repository layout

- `volumential/` — library source
- `test/` — pytest suite ([tiers and markers][docs-testing])
- `examples/` — maintained end-to-end examples
- `benchmarks/` — reproducible benchmark drivers
  ([`benchmarks/README.md`](./benchmarks/README.md))
- `doc/` — this documentation
- `DEVELOPMENT.md` — environment provisioning and day-to-day workflow
- `ruff.toml` — the single lint configuration

## AI-assisted development

Parts of this codebase, its tests, benchmark drivers and documentation were
written or revised with AI assistance, and pull requests are reviewed by the
maintainer and by automated code-review services. The full statement is
[AI-assisted development][docs-ai] in the documentation.

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

[docs-install]: ./doc/source/getting-started/installation.md
[docs-first]: ./doc/source/getting-started/first-volume-potential.md
[docs-getting-started]: ./doc/source/getting-started/index.md
[docs-user-guide]: ./doc/source/user-guide/index.md
[docs-design-notes]: ./doc/source/design-notes/index.md
[docs-benchmarks]: ./doc/source/benchmarks/index.md
[docs-api]: ./doc/source/api/index.rst
[docs-development]: ./doc/source/development/index.md
[docs-testing]: ./doc/source/development/testing.md
[docs-changelog]: ./doc/source/changelog.md
[docs-ai]: ./doc/source/development/ai-assisted-development.md
[nsf]: https://www.nsf.gov/
[hkust-math]: https://www.math.ust.hk/
[hkust]: https://www.ust.hk/home
[icerm]: https://icerm.brown.edu/
[uiuc-cs]: https://cs.illinois.edu/
[uiuc]: https://illinois.edu/
[volumential-name]: https://gitlab.tiker.net/xywei/volumential/issues/2
