"""OpenCL device selection for the tests that need double precision.

The volume FMM regressions and the full-accuracy sweeps build their own
context instead of taking the ``ctx_factory`` fixture, because they need a
device with fp64 whatever platform the fixture was pinned to.
:func:`create_fp64_context_or_skip` is the one place that decides which device
that is.

* With ``PYOPENCL_CTX`` set, the context is exactly the device it selects, CPU
  or GPU. The test is skipped only if that device lacks fp64; a selector that
  matches nothing is an error, not a skip.
* Without it, the first fp64 GPU is preferred and the first fp64 CPU is the
  fallback, so a CPU-only host runs these tests rather than skipping them.
  This default never picks the ``Intel(R) OpenCL`` platform, which
  ``conftest.py`` already marks as crashing on these paths.
"""

import os

import pytest

import pyopencl as cl


#: The Intel CPU runtime; see ``XFAIL_OPENCL_PLATFORMS`` in ``conftest.py``.
INTEL_OPENCL_PLATFORM_NAME = "Intel(R) OpenCL"


def device_supports_fp64(device) -> bool:
    """Return whether `device` advertises double-precision support."""
    extensions = getattr(device, "extensions", "").split()
    has_khr_fp64 = "cl_khr_fp64" in extensions
    has_double_config = bool(getattr(device, "double_fp_config", 0))
    return has_khr_fp64 or has_double_config


def _default_fp64_device():
    """Return the first fp64 GPU, else the first fp64 CPU, else None."""
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    candidates = [
        device
        for platform in platforms
        if platform.name != INTEL_OPENCL_PLATFORM_NAME
        for device in platform.get_devices()
        if device_supports_fp64(device)
    ]
    for device_type in (cl.device_type.GPU, cl.device_type.CPU):
        for device in candidates:
            if device.type & device_type:
                return device
    return None


def create_fp64_context_or_skip() -> cl.Context:
    """Return a context on an fp64 device, honoring ``PYOPENCL_CTX``.

    See the module docstring for how the device is chosen.
    """
    ctx_spec = os.environ.get("PYOPENCL_CTX")
    if ctx_spec:
        # Pass the selector as answers: with the environment alone,
        # create_some_context would prefer PYOPENCL_TEST when it is also set.
        ctx = cl.create_some_context(interactive=False, answers=ctx_spec.split(":"))
        lacking = [
            device.name for device in ctx.devices if not device_supports_fp64(device)
        ]
        if lacking:
            pytest.skip(
                f"PYOPENCL_CTX={ctx_spec!r} selects a device without fp64 "
                f"support: {', '.join(lacking)}"
            )
        return ctx

    device = _default_fp64_device()
    if device is None:
        pytest.skip(
            "No OpenCL GPU or CPU device with fp64 support available "
            "(set PYOPENCL_CTX to choose one explicitly)"
        )
    return cl.Context([device])
