# Development Environment

`pyproject.toml` + `uv` are the source of truth for dependency resolution, and
`uv.lock` records the exact dependency commits an environment was built from.

## Supported Setup

- Python: `3.11` (recommended for local and CI parity)
- OpenCL runtime: required (`pocl` is the default tested backend)
- Package manager: `uv`
- Fortran toolchain (`gfortran`, `ninja`): required only for the `fmmlib` extra

## Local Setup

1. Install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate an environment that includes OpenCL runtime support.
   Using micromamba is recommended:

   ```bash
   micromamba create -n volumential-dev -c conda-forge -c nodefaults \
     python=3.11 pyopencl pocl scipy numpy
   micromamba activate volumential-dev
   ```

3. Sync project dependencies:

   ```bash
   uv sync --active --extra test --extra doc
   ```

4. Run targeted checks:

   ```bash
   uv run pytest -q test/test_import.py
   uv run pytest -q test/test_public_surface.py
   uv run pytest -q test/test_duffy_tanh_sinh.py
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
uv sync --active --extra fmmlib
```

Verify the build before trusting FMMLib timings -- `FPNDFMMLibExpansionWrangler`
falls back to the serial per-box path without complaining when the batched
entry points are missing, so a mis-provisioned environment is correct but slow:

```bash
python -c "from pyfmmlib import l3dformmp_imany"   # batched wrappers present
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
`0.16588653810147913` over `21364` cross-level interactions. A different value
means the tree-of-boxes refinement is the broken one; stop and re-provision.
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
  python=3.11 pyopencl pocl scipy numpy
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
uv run pytest -q                     # default suite
uv run pytest --longrun              # include long-running checks
uv run pytest --full-accuracy        # include the high-cost accuracy markers
```

`ruff.toml` carries a `[lint.per-file-ignores]` baseline of pre-existing
violations so that `ruff check` passes on the tree as it stands. When you clean
a file, delete its baseline entry in the same commit. `ruff format` is not a
gate: adopting it (or the `Q` quote rules) would rewrite most of the tree at
once, so the `[format]` section only records the intended style.

## Documentation

```bash
uv sync --active --extra doc
make -C doc html                     # output in doc/build/html
sphinx-build -W -b html doc/source doc/build/html   # warnings as errors
```

## Notes

- Keep local and remote environments on the same Python minor version.
- `pytest.ini` currently wins over `[tool.pytest.ini_options]` in
  `pyproject.toml`; edit `pytest.ini` until the duplication is resolved.
