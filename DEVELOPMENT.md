# Development Environment

`pyproject.toml` + `uv` are the source of truth for dependency resolution, and
`uv.lock` records the exact dependency commits an environment was built from.

## Supported Setup

- Python: `3.12` (the version CI tests; see the note below)
- OpenCL runtime: required (`pocl` is the default tested backend)
- Package manager: `uv`
- Fortran toolchain (`gfortran`, `ninja`): required only for the `fmmlib` extra

`requires-python` in `pyproject.toml` is still `>=3.11`, and nothing in
Volumential itself needs 3.12. CI pins 3.12 because that is the environment
the suite is exercised in. The original reason was harder than that: `loopy`
imported `override` from the standard-library `typing` module, which gained it
only in 3.12, so `import loopy` failed outright under 3.11. At the `loopy`
revision `uv.lock` currently pins, that import comes from `typing_extensions`,
so 3.11 is untested rather than known-broken -- but nothing runs it, so do not
provision an evidence environment on it without re-measuring.

## Local Setup

1. Install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate an environment that includes OpenCL runtime support.
   Using micromamba is recommended:

   ```bash
   micromamba create -n volumential-dev -c conda-forge -c nodefaults \
     python=3.12 pyopencl pocl scipy numpy
   eval "$(micromamba shell hook -s bash)"   # unless the shell is init'd
   micromamba activate volumential-dev
   ```

   `micromamba activate` is a shell function, so a shell that has not been
   `micromamba shell init`-ed needs the hook first; otherwise activation fails
   and `uv sync --active` installs into the wrong interpreter.

3. Sync project dependencies:

   ```bash
   uv sync --active --extra test --extra doc
   ```

4. Run targeted checks. `--active` keeps `uv run` in the environment step 3
   filled, rather than the project's own `.venv`:

   ```bash
   uv run --active pytest -q test/test_import.py
   uv run --active pytest -q test/test_public_surface.py
   uv run --active pytest -q test/test_duffy_tanh_sinh.py
   ```

## Dependency Provisioning Rules

These rules exist because released wheels of the scientific stack have shipped
defects that corrupt results *silently*. They apply to every environment that
produces evidence, local or remote.

### Inducer stack from Git sources

The `inducer` projects (`boxtree`, `sumpy`, `loopy`, `pyopencl`, `pytools`,
`modepy`, `arraycontext`, `meshmode`, `pytential`) release rarely, so
`[tool.uv.sources]` installs them from main-branch Git sources and `uv.lock`
records the resolved commits. Do not swap them for PyPI wheels in an
experiment environment, and capture the locked commits in the run metadata of
any promoted result.

`boxtree` in particular must be at or after the upstream commit that fixed
`refine_and_coarsen_tree_of_boxes` (parent/child id remapping after the level
reorder, the tile-vs-repeat parent assignment, and an `np.inf` sentinel on an
integer array). Anything older -- including the `2024.10` wheel -- silently
corrupts List 1 neighbor lists on reordered adaptive trees. A patch that only
exists inside one environment's `site-packages` is an incident to remediate,
never a fix: patches belong upstream, on a tracked fork branch, or vendored and
committed.

### pyfmmlib with OpenMP and the batched P2M wrappers

The FMMLib backend needs `pyfmmlib` built from upstream `main`, which now
carries both the restored OpenMP feature option (inducer/pyfmmlib#93) and the
batched `{l,h}{2,3}dformmp_imany` wrappers (inducer/pyfmmlib#94). Released
wheels have neither. The interim locally patched branch is retired; the recipe
is a source build on a host with `gfortran` and `ninja`:

```bash
uv sync --active --extra test --extra doc --extra fmmlib
```

`uv sync` is an exact sync, so naming only `--extra fmmlib` uninstalls the
`test` and `doc` extras: list every extra the environment needs on each sync.

Since #135, `pyfmmlib` has a `[tool.uv.sources]` entry pointing at upstream
`main`, so the `fmmlib` extra resolves to the Git source at the commit
`uv.lock` pins rather than to the PyPI `2024.1.1` release — the release that
has neither the OpenMP option nor the batched wrappers. Installing it by hand
with `uv pip install "pyfmmlib @ git+..."` still works but bypasses the lock,
so two experiment hosts provisioned on different days can end up on different
revisions; prefer the extra. The `openmp` feature option defaults to `auto`,
so a host with a usable OpenMP toolchain needs no extra build flag — the `ldd`
check below is what confirms it took.

Verify the build before trusting FMMLib timings -- `FPNDFMMLibExpansionWrangler`
falls back to the serial per-box path without complaining when the batched
entry points are missing, so a mis-provisioned environment is correct but slow:

```bash
# Batched wrappers present.  The backend selects {l,h}{2,3}dformmp_imany from
# the equation and dimension, so check all four.
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
accuracy re-measurement -- but any FMMLib-stage *timing* narrative must be
re-measured after adoption.

### Traversal sanity check after provisioning

Run this once after creating an environment and after any inducer-stack update,
before the environment is used for evidence. It builds the graded 3D case
`(q_order, initial levels, adapt steps) = (3, 4, 3)` and reports its List 1
diagnostics:

```bash
python benchmarks/adaptive_timing_3d.py --mode full \
  --out build/benchmarks/adaptive-timing-3d.csv
```

In the `laplace3d-q3-l4-a3` row, `cross_level_list1_fraction` must equal
`0.16588653810147913`, i.e. `n_cross_level_list1_interactions` = `3544` out of
`n_list1_interactions` = `21364`. A different value means the tree-of-boxes
refinement is the broken one; stop and re-provision.
The driver also fails loudly if the adaptive tree comes out uniform, unbalanced,
or free of cross-level List 1 work.

### Thread caps and run metadata

Set the thread counts explicitly rather than inheriting a host default, and
record the values you used alongside any promoted timing:

```bash
export OMP_NUM_THREADS=1          # FMMLib/OpenMP stages
export POCL_MAX_PTHREAD_COUNT=4   # pocl worker threads
```

Run metadata for a promoted result should carry, at minimum: the locked
dependency commits, the `pyfmmlib` source revision, `OMP_NUM_THREADS`,
the pocl thread cap, the selected OpenCL platform, and the benchmark
parameters. Keep host-identifying details out of anything published.

### OpenCL ICD discovery

Select the pocl platform explicitly instead of taking whichever platform
enumerates first:

```bash
export PYOPENCL_CTX=portable:0
export PYOPENCL_TEST=portable:0
```

On NixOS, point ICD discovery at a single vendor directory as well; without it
`pyopencl` fails with `PLATFORM_NOT_FOUND_KHR` even when drivers are installed:

```bash
export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors
export OPENCL_VENDOR_PATH=/run/opengl-driver/etc/OpenCL/vendors
```

## Remote Setup

Heavier numerical experiments belong on a suitable, currently idle remote
machine. Select the host through the private remote-compute tracker; keep host
names, user names, remote paths and load figures out of the repository and out
of any published artifact.

```bash
ssh <remote-host>
git clone <repo-url>
cd volumential
micromamba create -n volumential-dev -c conda-forge -c nodefaults \
  python=3.12 pyopencl pocl scipy numpy
micromamba activate volumential-dev
uv sync --active --extra test --extra doc
```

Then run the provisioning checks above on that host before using it for
evidence, and run long jobs under `tmux` with `nice`, logging to a file.

## Lint, Types and Tests

```bash
# Lint: ruff.toml is the single configuration (85 columns, py311).
uvx ruff@0.13.0 check
uvx ruff@0.13.0 check --fix          # fixable rules only; re-read every hunk

# Types: basedpyright, configured in pyproject.toml.
uvx basedpyright -p pyproject.toml --level error

# Tests.
uv run --active pytest -q                     # default suite
uv run --active pytest --longrun              # include long-running checks
uv run --active pytest --full-accuracy        # include the high-cost accuracy markers
```

`ruff.toml` carries a `[lint.per-file-ignores]` baseline of pre-existing
violations so that `ruff check` passes on the tree as it stands. When you clean
a file, delete its baseline entry in the same commit. `ruff format` is not a
gate: adopting it (or the `Q` quote rules) would rewrite most of the tree at
once, so the `[format]` section only records the intended style.

## Documentation

The site is built with Sphinx and `pydata-sphinx-theme`; the `doc` extra
carries everything it needs. The build imports `volumential`, so it needs an
environment with the OpenCL stack (`pyopencl`, `loopy`) installed.

```bash
uv sync --active --extra doc

# The build CI Full runs: -W makes every warning an error, --keep-going
# reports all of them.  conf.py suppresses no warning class.
sphinx-build -W --keep-going -b html doc/source doc/build/html

# External links.
sphinx-build -b linkcheck doc/source doc/build/linkcheck

# Live preview at http://127.0.0.1:8000, rebuilding on save.
sphinx-autobuild doc/source doc/build/html
```

New pages are MyST Markdown (`.md`); the remaining reStructuredText pages are
substantial existing documents kept as they are. `doc/source/development/`
documents the section layout and which section a new page belongs in.

`doc/source/api/` is generated: `sphinx.ext.autosummary` writes one page per
module from the templates in `doc/source/_templates/autosummary/`, so a new
module needs no edit there. The build is `nitpicky`, which means an
unresolvable cross-reference in a docstring fails it; add an intersphinx
target when the name belongs to a dependency, and a commented
`nitpick_ignore` entry in `doc/source/conf.py` only when a third-party project
publishes no inventory for it.

## Notes

- Keep local and remote environments on the same Python minor version.
- `pytest.ini` currently wins over `[tool.pytest.ini_options]` in
  `pyproject.toml`; edit `pytest.ini` until the duplication is resolved.
