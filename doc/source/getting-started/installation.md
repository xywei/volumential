# Installation

`pyproject.toml` plus [uv](https://docs.astral.sh/uv/) are the source of truth
for dependency resolution, and `uv.lock` records the exact resolution an
environment was built from — a commit for each Git-sourced dependency, a
version and artifact hashes for each one that comes from PyPI. `DEVELOPMENT.md` at the repository root carries the
same recipe in the form a maintainer runs it; this page is the version a new
user needs.

## Prerequisites

- Python **3.12** — the version CI tests, and the only one exercised.
  `requires-python` is `>=3.11` and nothing in Volumential itself needs 3.12.
  `loopy` used to import `override` from the standard-library `typing` module,
  which gained it only in 3.12, so `import loopy` failed outright under 3.11;
  at the revision `uv.lock` pins that import now comes from
  `typing_extensions`. Treat 3.11 as untested rather than as known-broken: no
  job runs it.
- An **OpenCL runtime**. [PoCL](https://portablecl.org/) is the default tested
  backend; a vendor ICD (CUDA, ROCm) works too.
- **`uv`**.
- **`gfortran` and `ninja`**, only for the optional `fmmlib` extra.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The OpenCL runtime is easiest to obtain from conda-forge, so create the base
environment there and let `uv` fill in the rest:

```bash
micromamba create -n volumential-dev -c conda-forge -c nodefaults \
  python=3.12 pyopencl pocl scipy numpy

eval "$(micromamba shell hook -s bash)"   # once per shell, if not shell-init'd
micromamba activate volumential-dev
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"

git clone https://github.com/xywei/volumential.git
cd volumential
uv sync --extra test --extra doc
```

Two lines there are load-bearing and both fail quietly if you skip them.

`micromamba activate` is a shell function, not a binary, so a fresh shell has
to evaluate the hook before it exists. `micromamba shell init -s bash` makes it
permanent; the `eval` line above is the per-shell form.

`UV_PROJECT_ENVIRONMENT` is what makes `uv` use the conda environment as the
project environment. Activating conda sets `CONDA_PREFIX`, not `VIRTUAL_ENV`,
and `uv`'s `--active` flag keys on `VIRTUAL_ENV` — so `uv sync --active` inside
an activated conda environment does **not** target it: it creates `.venv` in
the checkout and installs there, and every later `uv run` uses that same
`.venv`. The result builds and imports, and has none of the conda-provided
OpenCL runtime this recipe exists to supply. Export the variable once after
activating and both `uv sync` and `uv run` do the right thing.

## Why the dependencies come from Git

`[tool.uv.sources]` points most of the `inducer` stack at its main branches,
and `uv.lock` records the resolved commits: `arraycontext`, `boxtree`, `cgen`,
`genpy`, `gmsh_interop`, `loopy`, `meshmode`, `modepy`, `pyfmmlib`,
`pymbolic`, `pytential`, `pytools`, `pyvisfile`, `sumpy`. **`pyopencl` is not
among them** — it resolves from PyPI, and `uv.lock` pins a release
(`2026.1.2`) rather than a commit, so that is the version an audit of an
evidence environment should expect to find. These projects release rarely, and released wheels have shipped
defects that corrupt results *silently*, which is a different and worse failure
than a crash.

The specific one that matters here: `boxtree` must be at or after the upstream
commit that fixed `refine_and_coarsen_tree_of_boxes` (parent/child id remapping
after the level reorder, the tile-versus-repeat parent assignment, and an
`np.inf` sentinel on an integer array). Anything older — the `2024.10` wheel
included — silently corrupts List 1 neighbour lists on reordered adaptive
trees. Do not substitute PyPI wheels in an environment that produces evidence,
and capture the locked commits in the metadata of any promoted result.

A patch that exists only inside one environment's `site-packages` is an
incident to remediate, never a fix: patches belong upstream, on a tracked fork
branch, or vendored and committed.

## The FMMLib backend

```bash
uv sync --extra test --extra doc --extra fmmlib
```

`uv sync` is an *exact* sync: it uninstalls whatever the requested set does not
include. Naming only `--extra fmmlib` would therefore remove the `test` and
`doc` extras installed above, `pytest` included, so list every extra you want
in the environment on each sync.

This builds `pyfmmlib` from upstream `main`, which carries both the restored
OpenMP feature option ([inducer/pyfmmlib#93](https://github.com/inducer/pyfmmlib/pull/93))
and the batched `{l,h}{2,3}dformmp_imany` wrappers
([inducer/pyfmmlib#94](https://github.com/inducer/pyfmmlib/pull/94)). The PyPI
`2024.1.1` release has neither. Since
[#135](https://github.com/xywei/volumential/pull/135) `pyfmmlib` has a
`[tool.uv.sources]` entry like the inducer packages, so the extra resolves to
the commit `uv.lock` pins; installing it by hand with
`uv pip install "pyfmmlib @ git+..."` still works but bypasses the lock, and
two hosts provisioned on different days then end up on different revisions.

The `openmp` feature option defaults to `auto`, so a host with a usable OpenMP
toolchain needs no extra build flag. Verify the build before trusting any
FMMLib timing — `FPNDFMMLibExpansionWrangler` falls back to the serial per-box
path *without complaining* when the batched entry points are missing, so a
mis-provisioned environment is correct but slow:

```bash
# Batched wrappers present.  The backend picks {l,h}{2,3}dformmp_imany from the
# equation and the dimension, so check all four: a successful 3D Laplace import
# does not rule out a 2D or Helmholtz fallback.
python -c "from pyfmmlib import \
    h2dformmp_imany, h3dformmp_imany, l2dformmp_imany, l3dformmp_imany"
python - <<'PY'
import pathlib
import subprocess

import pyfmmlib

so = next(pathlib.Path(pyfmmlib.__file__).parent.glob("_internal*.so"))
print(so)
subprocess.run(["ldd", str(so)], check=True)   # expect a libgomp line
PY
```

Batched P2M is bit-identical to the per-box path and GEMM L2P agrees at
roundoff (`test/test_fmmlib_batched_stages.py`), so adopting them needs no
accuracy re-measurement — but any FMMLib-stage *timing* narrative must be
re-measured after adoption.

## Verify the environment

### Imports and quick tests

These need `UV_PROJECT_ENVIRONMENT` exported, as above — `uv run` otherwise
creates and uses `.venv` rather than the conda environment.

```bash
uv run pytest -q test/test_import.py
uv run pytest -q test/test_public_surface.py
uv run pytest -q test/test_duffy_tanh_sinh.py
```

### The traversal check

Run this once after creating an environment, and again after any inducer-stack
update, *before* the environment is used for evidence. It builds the graded 3D
case `(q_order, initial levels, adapt steps) = (3, 4, 3)` and reports its
List 1 diagnostics:

```bash
python benchmarks/adaptive_timing_3d.py --mode full \
  --out build/benchmarks/adaptive-timing-3d.csv
```

In the `laplace3d-q3-l4-a3` row, `cross_level_list1_fraction` must equal
`0.16588653810147913`, that is `n_cross_level_list1_interactions` = `3544` out
of `n_list1_interactions` = `21364`. A different value means the tree-of-boxes
refinement is the broken one: stop and re-provision. The driver also fails
loudly if the adaptive tree comes out uniform, unbalanced, or free of
cross-level List 1 work.

This check is the reason the Git sources above are not optional. It is cheap,
it is decisive, and skipping it is how a corrupted neighbour list reaches a
plot.

## Next

- {doc}`first-volume-potential` — a volume potential in about twenty lines.
- {doc}`device-selection` — make sure the run lands on the device you meant.
- `DEVELOPMENT.md` and {doc}`../development/index` — the maintainer-facing
  version of all of the above, plus lint, types and the test tiers.
