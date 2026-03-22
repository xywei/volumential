# Development Environment

This project now uses `pyproject.toml` + `uv` as the source of truth for
dependency resolution.

## Supported Setup

- Python: `3.11` (recommended for local and CI parity)
- OpenCL runtime: required (`pocl` is the default tested backend)
- Package manager: `uv`

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

3. Sync project dependencies from `pyproject.toml`:

   ```bash
   uv sync --active --extra test --extra doc
   ```

4. Run targeted checks:

   ```bash
   uv run pytest -q test/test_import.py
   uv run pytest -q test/test_duffy_tanh_sinh.py
   ```

## Remote Setup

Use the same environment recipe on remote machines used for heavier numerical
experiments (for example `ipa`).

```bash
ssh ipa
git clone <repo-url>
cd volumential
micromamba create -n volumential-dev -c conda-forge -c nodefaults \
  python=3.11 pyopencl pocl scipy numpy
micromamba activate volumential-dev
uv sync --active --extra test --extra doc
uv run pytest -q test/test_volume_fmm.py::test_volume_fmm_laplace
```

## Notes

- `tool.uv.sources` in `pyproject.toml` pins the inducer-stack packages to the
  upstream Git sources used for development.
- Keep local and remote environments on the same Python minor version for
  reproducibility.
- On NixOS, make sure OpenCL ICD discovery points at a single vendor
  directory before running tests. For example:

  ```bash
  export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors
  export OPENCL_VENDOR_PATH=/run/opengl-driver/etc/OpenCL/vendors
  ```

  Without this, `pyopencl` may fail with `PLATFORM_NOT_FOUND_KHR` even when
  OpenCL drivers are installed.
