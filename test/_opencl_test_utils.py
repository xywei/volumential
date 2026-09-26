"""OpenCL device selection for the tests that build their own context.

Some tests do not take the ``ctx_factory`` fixture. The volume FMM regressions
and the full-accuracy sweeps need a device with fp64 whatever platform the
fixture was pinned to, and the near-field tables built by the session fixture
in ``conftest.py`` and by ``test_nearfield_potential_table.py`` want one queue,
not one per platform. The functions here are the one place that decides which
device those are, and both honor ``PYOPENCL_CTX``: when it is set, the context
is exactly the device it selects, CPU or GPU, and a selector that matches
nothing, or more than one device, is an error, not a skip.

Without ``PYOPENCL_CTX``, :func:`create_fp64_context_or_skip` prefers the
first fp64 GPU and falls back to the first fp64 CPU, so a CPU-only host runs
these tests rather than skipping them. That default never picks the
``Intel(R) OpenCL`` platform, which ``conftest.py`` already marks as crashing
on these paths. :func:`create_table_build_queue_or_skip` keeps the rule the
table builds always had: the first device of the first other platform, and
the Intel runtime only when there is nothing else.
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


def create_pyopencl_ctx_context() -> cl.Context | None:
    """Return a context on exactly the device ``PYOPENCL_CTX`` selects.

    Returns *None* when the variable is unset or empty. A selector that
    matches no platform or device raises, as it does in
    :func:`pyopencl.create_some_context`. So does one that selects several
    devices, such as ``portable:0,1``: the callers build their queues without
    naming a device, and would silently run on the first of them.
    """
    ctx_spec = os.environ.get("PYOPENCL_CTX")
    if not ctx_spec:
        return None
    # Pass the selector as answers: with the environment alone,
    # create_some_context would prefer PYOPENCL_TEST when it is also set.
    ctx = cl.create_some_context(interactive=False, answers=ctx_spec.split(":"))
    if len(ctx.devices) != 1:
        raise RuntimeError(
            f"PYOPENCL_CTX={ctx_spec!r} selects {len(ctx.devices)} devices: "
            f"{', '.join(device.name for device in ctx.devices)}; the tests "
            "that build their own context need exactly one"
        )
    return ctx


def create_fp64_context_or_skip() -> cl.Context:
    """Return a context on an fp64 device, honoring ``PYOPENCL_CTX``.

    Skips if the device ``PYOPENCL_CTX`` selects lacks fp64, rather than
    substituting another one. See the module docstring for the default.
    """
    ctx = create_pyopencl_ctx_context()
    if ctx is not None:
        lacking = [
            device.name for device in ctx.devices if not device_supports_fp64(device)
        ]
        if lacking:
            pytest.skip(
                f"PYOPENCL_CTX={os.environ['PYOPENCL_CTX']!r} selects a device "
                f"without fp64 support: {', '.join(lacking)}"
            )
        return ctx

    device = _default_fp64_device()
    if device is None:
        pytest.skip(
            "No OpenCL GPU or CPU device with fp64 support outside the "
            f"{INTEL_OPENCL_PLATFORM_NAME} platform (set PYOPENCL_CTX to choose "
            "one explicitly)"
        )
    return cl.Context([device])


def create_table_build_queue_or_skip() -> cl.CommandQueue:
    """Return a queue for building near-field tables, honoring ``PYOPENCL_CTX``.

    See the module docstring for the device it takes without the variable.
    """
    ctx = create_pyopencl_ctx_context()
    if ctx is not None:
        return cl.CommandQueue(ctx)

    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    preferred = [p for p in platforms if p.name != INTEL_OPENCL_PLATFORM_NAME]
    intel = [p for p in platforms if p.name == INTEL_OPENCL_PLATFORM_NAME]
    for platform in preferred + intel:
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))

    pytest.skip("No OpenCL devices available for table builds")
