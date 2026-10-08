# Installation

Volumential is installed from GitHub. A release is a Git tag, and
`pyproject.toml` pins the `inducer` packages it depends on to Git commits.
For a development environment, `pyproject.toml` plus
[uv](https://docs.astral.sh/uv/) are the source of truth for dependency
resolution, and `uv.lock` records the exact resolution an environment was
built from — those same commits, and a version and artifact hashes for each
dependency that comes from PyPI. `DEVELOPMENT.md` at the repository root
carries the same recipe in the form a maintainer runs it; this page is the
version a new user needs.

## Install a release

```bash
pip install "volumential @ git+https://github.com/xywei/volumential@v<version>"
uv pip install "volumential @ git+https://github.com/xywei/volumential@v<version>"
```

No version is tagged yet; until one is, `@main` or `@<commit>` installs the
same way, and extras go before the `@`, as in
`"volumential[fmmlib] @ git+https://github.com/xywei/volumential@main"`.
pip or uv builds Volumential and installs the `inducer` packages at the
commits its `pyproject.toml` pins — `arraycontext`, `boxtree`, `loopy`,
`meshmode`, `modepy`, `pymbolic`, `pytential`, `pytools`, `sumpy`, and
`gmsh_interop` with the `test` extra — which are the commits CI tests, and
everything else from PyPI. That needs:

- **`git`**, to clone each pinned dependency;
- **a C compiler**, for the extension `pytential` builds;
- to run anything, an **OpenCL runtime**: the conda-forge environment of
  [Install](#install) below provides PoCL, and a vendor ICD works too.

Releases are not on PyPI. PyPI rejects any distribution whose metadata names
a dependency by a direct reference (`name @ git+https://...`), pinned or not,
in an extra or not. The pins have to be such references, because `pytential`
has no installable release on PyPI and `sumpy`, `boxtree`, `meshmode` and
`arraycontext` only years-old ones
([#211](https://github.com/xywei/volumential/issues/211)). Each release is
also a GitHub Release, with its sdist and wheel attached; they carry the same
pins.

The rest of this page builds a development environment from a clone, which
is also the recipe for any environment that produces evidence.

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
- **`micromamba`** (or `conda`/`mamba`), which is how this recipe provides the
  OpenCL runtime. Skip it only if the host already has a working ICD and a
  3.12 interpreter, in which case create the environment however you normally
  would and pick the recipe up at the `git clone`.
- **`git` and a C compiler**: the pinned dependencies are cloned, and
  `pytential` builds a C extension.
- **`gfortran` and `ninja`**, only for the optional `fmmlib` extra, and only
  where `pyfmmlib` has no wheel: PyPI has wheels for Linux x86_64.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)      # if micromamba is absent
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

Two lines there are load-bearing, and they fail differently if you skip them:
without the `eval`, `micromamba activate` fails at once because the shell
function does not exist yet; without `UV_PROJECT_ENVIRONMENT`, `uv` fails
quietly and installs into a checkout-local `.venv` instead.

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

`pyproject.toml` pins most of the `inducer` stack to commits of its main
branches, as direct references (`name @ git+https://...@<commit>`), and
`uv.lock` records the same commits: `arraycontext`, `boxtree`, `loopy`,
`meshmode`, `modepy`, `pymbolic`, `pytential`, `pytools`, `sumpy`, and
`gmsh_interop` in the `test` and `gmsh_support` extras. **`pyopencl` and
`pyfmmlib` are not among them** — they resolve from PyPI, and `uv.lock` pins a
release (`pyopencl` `2026.1.2`, `pyfmmlib` `2026.1`) rather than a commit, so
that is the version an audit of an evidence environment should expect to
find; so do `cgen`, `genpy` and `islpy`, which Volumential only needs through
the packages above. These projects release rarely, and released wheels have
shipped defects that corrupt results *silently*, which is a different and
worse failure than a crash.

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

The extra asks for `pyfmmlib` `2026.1` or later, the first release with both
the restored OpenMP feature option
([inducer/pyfmmlib#93](https://github.com/inducer/pyfmmlib/pull/93)) and the
batched `{l,h}{2,3}dformmp_imany` wrappers
([inducer/pyfmmlib#94](https://github.com/inducer/pyfmmlib/pull/94)); its
Linux x86_64 wheels on PyPI carry the wrappers and link `libgomp`. Earlier
releases, `2024.1.1` included, have neither, which is why the extra took
`pyfmmlib` from a locked commit of upstream `main` from
[#135](https://github.com/xywei/volumential/pull/135) until `2026.1` came out.

Where there is no wheel, the sdist is built, and its `openmp` feature option
defaults to `auto`, so a host with a usable OpenMP toolchain needs no extra
build flag. Verify the installation before trusting any FMMLib timing —
`FPNDFMMLibExpansionWrangler` falls back to the serial per-box path *without
complaining* when the batched entry points are missing, so a mis-provisioned
environment is correct but slow:

```bash
# Batched wrappers present.  The backend picks {l,h}{2,3}dformmp_imany from the
# equation and the dimension, so check all four: a successful 3D Laplace import
# does not rule out a 2D or Helmholtz fallback.  This covers charge sources
# only.
python -c "from pyfmmlib import \
    h2dformmp_imany, h3dformmp_imany, l2dformmp_imany, l3dformmp_imany"
# Dipole sources (a DirectionalSourceDerivative kernel, i.e. a dipole_vec)
# take a different P2M wrapper, {l,h}{2,3}dformmp_dp_imany, and fall back to
# the per-box routine just as silently when it is absent.  pyfmmlib 2026.1
# generates both families, so an import failure here means
# the dipole P2M will run per box, not that the charge path is broken.
python -c "from pyfmmlib import \
    h2dformmp_dp_imany, h3dformmp_dp_imany, \
    l2dformmp_dp_imany, l3dformmp_dp_imany"
python - <<'PY'
import pathlib
import platform
import subprocess

import pyfmmlib

so = next(pathlib.Path(pyfmmlib.__file__).parent.glob("_internal*.so"))
print(so)
if platform.system() == "Darwin":
    # macOS has no ldd, and the OpenMP runtime is libomp rather than libgomp.
    subprocess.run(["otool", "-L", str(so)], check=True)
else:
    subprocess.run(["ldd", str(so)], check=True)
PY
```

Expect a line for *an* OpenMP runtime: `libgomp` for a GCC build, `libomp`
for Clang (the usual case on macOS, and a possible one on Linux), `libiomp`
for Intel. The name follows the compiler, not the operating system, so do not
read a missing `libgomp` on Linux as a missing OpenMP. A missing runtime and
an unavailable `ldd` look the same otherwise, which is why the check branches
instead of assuming Linux.

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
update, *before* the environment is used for evidence. The driver that builds
the graded 3D case `(q_order, initial levels, adapt steps) = (3, 4, 3)` and
reports its List 1 diagnostics left the tree with the rest of the benchmark
drivers ({doc}`../benchmarks/index`), but it is one `git archive` away:
`7c75ed1` is the last revision of `main` that carries it, and it imports only
its sibling `adaptive_timing.py` besides the installed library.

```bash
mkdir -p build/traversal-check
git archive 7c75ed1 benchmarks | tar -x -C build/traversal-check
python build/traversal-check/benchmarks/adaptive_timing_3d.py --mode full \
  --out build/traversal-check/adaptive-timing-3d.csv
```

In the `laplace3d-q3-l4-a3` row of that CSV, `cross_level_list1_fraction` must
equal `0.16588653810147913`, that is `n_cross_level_list1_interactions` =
`3544` out of `n_list1_interactions` = `21364`. A different value means the
tree-of-boxes refinement is the broken one: stop and re-provision. The driver
also fails loudly if the adaptive tree comes out uniform, unbalanced, or free
of cross-level List 1 work; that is the same verdict.

This check is the reason the Git pins above are not optional. It is cheap,
it is decisive, and skipping it is how a corrupted neighbour list reaches a
plot.

## Next

- {doc}`first-volume-potential` — a volume potential in about twenty lines.
- {doc}`device-selection` — make sure the run lands on the device you meant.
- `DEVELOPMENT.md` and {doc}`../development/index` — the maintainer-facing
  version of all of the above, plus lint, types and the test tiers.
