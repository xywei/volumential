"""Shared run-provenance capture for the benchmark drivers.

A column of seconds is not evidence on its own.  Three facts decide what a
benchmark number means, and none of them can be recovered from a driver's
command line after the fact:

* **Which device actually ran.**  ``--backend pocl-cpu`` and
  ``cl.create_some_context(interactive=False)`` are *requests*; what the ICD
  loader resolved is a different fact.  The resolved platform, device and
  driver version belong in the metadata, not the requested backend token.
* **Which host CPU, and with what worker-thread cap.**  The same PoCL build
  of the same table differs by well over an order of magnitude between a CPU
  without hardware FMA and a current one, with no code change at all (#138),
  and ``OMP_NUM_THREADS`` / ``POCL_MAX_PTHREAD_COUNT`` decide how much of the
  host a run was allowed to use.
* **Whether the FFT-accelerated multipole-to-local had a real FFT.**  sumpy
  uses VkFFT when it can and falls back to a loopy FFT -- several times the
  arithmetic -- when it cannot.  That is a cost class, not a detail, and
  "``pyvkfft`` is installed" does not settle it: sumpy also honours
  ``SUMPY_FFT_BACKEND``, refuses VkFFT on an out-of-order queue, and refuses
  it on PoCL 7 and later, which miscompiles it.

Separately, a driver that reports one total for a timed solve or table build
hides the largest single cost in a one-shot run: the first solve of a
*process* pays sumpy's code generation, measured at 884 s for a 3D Helmholtz
order-23 path whose warm solve takes 1.5 s (#136).  :func:`time_repeats` and
:func:`first_call_and_warm` keep the two apart.

Nothing here touches an OpenCL platform at import time, so unit tests and
``--list-cases``-style paths can use it on a machine with no device.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


__all__ = (
    "OPENCL_PROVENANCE_KEYS",
    "PROVENANCE_KEYS",
    "collect_run_provenance",
    "first_call_and_warm",
    "opencl_provenance",
    "public_argv",
    "public_path",
    "resolved_device_line",
    "time_repeats",
)


#: Top-level keys :func:`collect_run_provenance` always returns, in order.
#: A consumer may key on these; the dict is additive, so it may grow.
PROVENANCE_KEYS = (
    "opencl",
    "cpu_model",
    "omp_num_threads",
    "pocl_max_pthread_count",
    "pyvkfft_importable",
    "sumpy_fft_backend",
)

#: Keys of the ``opencl`` sub-dict, in order.
OPENCL_PROVENANCE_KEYS = (
    "platform",
    "platform_version",
    "device",
    "device_type",
    "driver_version",
    "vendor",
    "max_compute_units",
)

#: ``cl_device_type`` is a bit field, so a device can legitimately report
#: more than one of these (a CPU device that is also the platform default).
_DEVICE_TYPE_BITS = (
    (1 << 0, "DEFAULT"),
    (1 << 1, "CPU"),
    (1 << 2, "GPU"),
    (1 << 3, "ACCELERATOR"),
    (1 << 4, "CUSTOM"),
)

#: Worker-thread caps worth recording, mapped to the provenance key that
#: carries them.  Both decide how much of the host a run was allowed to use.
_THREAD_CAP_VARS = (
    ("omp_num_threads", "OMP_NUM_THREADS"),
    ("pocl_max_pthread_count", "POCL_MAX_PTHREAD_COUNT"),
)


# {{{ host facts


def _cpu_model() -> str | None:
    """The host CPU's model string, or ``None`` when it cannot be read.

    ``/proc/cpuinfo`` is the only place that carries the marketing model
    name on Linux; :func:`platform.processor` returns the machine type
    there.  The fallbacks keep the field populated, if less specific, on
    other platforms rather than reporting nothing at all.
    """
    if sys.platform.startswith("linux"):
        try:
            with open(
                "/proc/cpuinfo", encoding="utf-8", errors="replace"
            ) as infile:
                for line in infile:
                    name, separator, value = line.partition(":")
                    if separator and name.strip().lower() == "model name":
                        model = value.strip()
                        if model:
                            return model
        except OSError:
            pass

    return platform.processor() or platform.machine() or None


def _thread_cap(variable: str) -> int | str | None:
    """The value of one worker-thread cap environment variable.

    Returns the integer when the variable parses as one, the raw string when
    it does not (``OMP_NUM_THREADS`` accepts a comma-separated list of
    per-nesting-level counts), and ``None`` when it is unset.  An unset cap
    and a cap of one are very different runs, so they must not collapse onto
    the same recorded value.
    """
    raw = os.environ.get(variable)
    if raw is None:
        return None
    stripped = raw.strip()
    try:
        return int(stripped)
    except ValueError:
        return stripped


def _pyvkfft_importable() -> bool:
    """Whether ``import pyvkfft`` succeeds in this process.

    Checked by import rather than by distribution metadata: an installed but
    broken ``pyvkfft`` is as good as an absent one.  This is a *necessary*
    condition for a real FFT, not a sufficient one -- what sumpy actually
    selected is :func:`_sumpy_fft_backend`.
    """
    if "pyvkfft" in sys.modules:
        return True
    try:
        if importlib.util.find_spec("pyvkfft") is None:
            return False
        importlib.import_module("pyvkfft")
    except Exception:
        # Any failure to import it -- missing, broken, or unusable on this
        # device -- has the same consequence for the FFT that runs.
        return False
    return True


def _looks_like_queue(source: Any) -> bool:
    """Whether ``source`` is a command queue rather than a context/device."""
    return hasattr(source, "device") and hasattr(source, "finish")


def _sumpy_fft_backend(source: Any) -> str | None:
    """Which FFT backend sumpy selects for ``source``, if it can be asked.

    ``pyvkfft_importable`` is necessary but not sufficient: sumpy honours
    ``SUMPY_FFT_BACKEND``, refuses VkFFT on an out-of-order queue, and
    refuses it on PoCL 7 and later (which miscompiles it), so an importable
    ``pyvkfft`` routinely still runs the loopy fallback.  This asks sumpy
    itself, and returns ``None`` -- "not determined" -- when there is no
    queue to ask about or sumpy does not expose the selector.
    """
    if not _looks_like_queue(source):
        return None
    try:
        from sumpy.tools import _get_fft_backend

        return str(_get_fft_backend(source).name)
    except Exception:
        return None


# }}}


# {{{ device facts


def _device_type_name(value: Any) -> str:
    """Render a ``cl_device_type`` bit field as a stable string."""
    try:
        bits = int(value)
    except (TypeError, ValueError):
        return str(value)

    names = [name for bit, name in _DEVICE_TYPE_BITS if bits & bit]
    if not names:
        return f"UNKNOWN(0x{bits:x})"
    return "|".join(names)


def _resolve_device(source: Any) -> Any:
    """The OpenCL device behind a context, a command queue or a device.

    Drivers hold whichever of the three is convenient where they build their
    metadata, and none of them should have to know what the others expose.
    """
    devices = getattr(source, "devices", None)
    if devices is not None:
        devices = list(devices)
        if not devices:
            raise ValueError("the OpenCL context has no devices")
        # Every driver here builds a single-device context; a multi-device
        # one is described by the first device rather than refused.
        return devices[0]

    device = getattr(source, "device", None)
    if device is not None:
        return device

    if hasattr(source, "platform") and hasattr(source, "name"):
        return source

    raise TypeError(
        "expected a pyopencl Context, CommandQueue or Device, got "
        f"{type(source).__name__}"
    )


def _attribute(obj: Any, name: str) -> Any:
    """``getattr`` that survives a driver refusing to answer a query.

    Some ICDs raise instead of returning a value for an optional device
    query, and provenance capture must never be the thing that fails an
    otherwise completed benchmark run.
    """
    try:
        return getattr(obj, name)
    except Exception:
        return None


def opencl_provenance(source: Any) -> dict[str, Any]:
    """Describe the *resolved* OpenCL device behind ``source``.

    :arg source: a live :class:`pyopencl.Context`,
        :class:`pyopencl.CommandQueue` or :class:`pyopencl.Device`.
    :returns: a dict with the keys of :data:`OPENCL_PROVENANCE_KEYS`.
    """
    device = _resolve_device(source)
    device_platform = _attribute(device, "platform")

    max_compute_units = _attribute(device, "max_compute_units")
    if max_compute_units is not None:
        max_compute_units = int(max_compute_units)

    return {
        "platform": _attribute(device_platform, "name"),
        "platform_version": _attribute(device_platform, "version"),
        "device": _attribute(device, "name"),
        "device_type": _device_type_name(_attribute(device, "type")),
        "driver_version": _attribute(device, "driver_version"),
        "vendor": _attribute(device, "vendor"),
        "max_compute_units": max_compute_units,
    }


def collect_run_provenance(source: Any) -> dict[str, Any]:
    """Everything a promoted benchmark number needs about its machine.

    :arg source: a live :class:`pyopencl.Context`,
        :class:`pyopencl.CommandQueue` or :class:`pyopencl.Device`.  Pass the
        **queue** where there is one: ``sumpy_fft_backend`` can only be
        determined from a queue, and is ``None`` otherwise.
    :returns: a dict with the keys of :data:`PROVENANCE_KEYS`.  It is JSON
        serializable and carries no host name, user name or path, so it is
        safe to commit next to a CSV.
    """
    provenance: dict[str, Any] = {"opencl": opencl_provenance(source)}
    provenance["cpu_model"] = _cpu_model()
    for key, variable in _THREAD_CAP_VARS:
        provenance[key] = _thread_cap(variable)
    provenance["pyvkfft_importable"] = _pyvkfft_importable()
    provenance["sumpy_fft_backend"] = _sumpy_fft_backend(source)
    return provenance


def resolved_device_line(provenance: dict[str, Any]) -> str:
    """A one-line ``RESOLVED-DEVICE`` record for a driver's stdout.

    A log that says only which ``--backend`` was asked for cannot be used to
    attribute a cost; this line says what answered.  Accepts either the full
    dict from :func:`collect_run_provenance` or its ``opencl`` sub-dict.
    """
    opencl = provenance.get("opencl", provenance)
    return (
        "RESOLVED-DEVICE "
        f"platform={opencl.get('platform')!r} "
        f"platform_version={opencl.get('platform_version')!r} "
        f"device={opencl.get('device')!r} "
        f"device_type={opencl.get('device_type')} "
        f"driver_version={opencl.get('driver_version')!r}"
    )


# }}}


# {{{ keeping infrastructure out of a promotable sidecar


def public_path(path) -> str:
    """``path`` relative to the working directory, or just its basename.

    A sidecar is promoted next to its CSV, so an absolute path in it
    publishes a user name and a mount layout.  Anything outside the run's
    own directory is reduced to a basename, which is all a reader needs.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return path.name


def public_argv(argv: list[str]) -> list[str]:
    """``argv`` with absolute and escaping paths reduced by :func:`public_path`.

    Handles both ``--out /abs/path`` and ``--out=/abs/path``; a token that is
    not a path outside the working directory is left exactly as typed, so the
    recorded command still reproduces the run.
    """
    result = []
    for token in argv:
        option, separator, value = token.partition("=")
        candidate = value if separator else token
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            candidate = public_path(candidate_path)
        result.append(option + separator + candidate if separator else candidate)
    return result


# }}}


# {{{ first call versus warm repeats


def time_repeats(
    func: Callable[[], Any],
    *,
    repeats: int = 1,
    sync: Callable[[], Any] | None = None,
) -> tuple[Any, list[float]]:
    """Call ``func`` ``repeats`` times; return ``(last_result, samples_s)``.

    :arg sync: called once before the first sample and again after each
        call, inside the timed region.  Pass ``queue.finish`` for OpenCL
        work: a device call only *enqueues*, so an unsynchronized clock
        measures enqueue time and charges the real work to whatever blocks
        next -- the mis-diagnosis #136 had to be corrected for.

    The samples come back in call order, so ``samples_s[0]`` is this
    process's first call of this code path and pays whatever code generation
    and kernel compilation it triggers.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    if sync is not None:
        sync()

    result = None
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        if sync is not None:
            sync()
        samples.append(time.perf_counter() - start)

    return result, samples


def first_call_and_warm(
    samples_s: Iterable[float],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Split a repeat series into its first call and its warm median.

    :arg samples_s: the wall time of each repeat, in call order.
    :arg prefix: prepended to every returned key, so a driver that times
        more than one thing can keep them apart (``prefix="fmm_"`` gives
        ``fmm_first_call_s`` and its siblings).

    ``first_call_s`` is reported on its own because the first solve or table
    build of a process pays sumpy's code generation, which can be two orders
    of magnitude larger than the work being measured.  ``warm_s`` is the
    median of the remaining repeats, and is ``None`` when the driver made
    only one call -- an honest "not measured", never a warm number silently
    contaminated by code generation.
    """
    samples = [float(value) for value in samples_s]
    if not samples:
        raise ValueError("samples_s must contain at least one measurement")

    warm = samples[1:]
    return {
        f"{prefix}first_call_s": samples[0],
        f"{prefix}warm_s": statistics.median(warm) if warm else None,
        f"{prefix}warm_repeat_count": len(warm),
        f"{prefix}samples_s": samples,
    }


# }}}

# vim: foldmethod=marker
