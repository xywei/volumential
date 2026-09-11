"""Shared pytest configuration for the Volumential test suite.

This module owns everything that the whole suite relies on:

* the ``ctx_factory`` fixture (re-exported from :mod:`pyopencl.tools`), which
  parametrizes OpenCL-using tests over the available platforms;
* the ``--longrun`` and ``--full-accuracy`` command line options and the
  corresponding ``longrun`` fixture / ``full_accuracy`` marker, which keep a
  default ``pytest`` run reasonably short;
* the ``slow`` marker, which labels (but does not skip) the handful of tests
  that dominate the wall clock, so that ``-m 'not slow'`` gives a quick run;
* the xfail policy for OpenCL platforms that are known to crash;
* the session-scoped ``table_2d_order1`` near-field table, which is expensive
  enough that every test that needs it shares one build; and
* the cleanup of stray table caches at the end of a session.
"""

__copyright__ = "Copyright (C) 2018 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import contextlib
from pathlib import Path

import pytest
from filelock import FileLock

import pyopencl as cl

# setup the ctx_factory fixture
from pyopencl.tools import (  # noqa: F401
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

from volumential.table_manager import NearFieldInteractionTableManager as NFTManager


#: Markers this suite defines, registered in :func:`pytest_configure` so that
#: they work no matter which ini file pytest picks up.
SUITE_MARKERS = (
    "full_accuracy: high-cost derivative accuracy tests, skipped unless "
    "--full-accuracy",
    "slow: tests whose aggregate runtime exceeds roughly 30 s on a CPU OpenCL "
    "backend; they still run by default, deselect them with -m 'not slow'",
)

XFAIL_OPENCL_PLATFORMS = {
    # This Intel OpenCL CPU backend has been observed to core-dump on
    # volumential nearfield/FMM test paths; keep this visible as an xfail
    # until upstream/backend stability is confirmed.
    "Intel(R) OpenCL": (
        "known Intel OpenCL backend crash (core dump) on volumential "
        "nearfield path"
    ),
}

INTEL_OPENCL_PLATFORM_NAME = "Intel(R) OpenCL"

_CTX_FACTORY_XFAIL_REASON_CACHE = {}


def _get_xfail_reason_for_ctx_factory(ctx_factory) -> str | None:
    """Return the xfail reason for `ctx_factory`, or None if it is usable."""
    if ctx_factory in _CTX_FACTORY_XFAIL_REASON_CACHE:
        return _CTX_FACTORY_XFAIL_REASON_CACHE[ctx_factory]

    ctx = ctx_factory()
    platform_names = {dev.platform.name for dev in ctx.devices}
    for name in platform_names:
        if name in XFAIL_OPENCL_PLATFORMS:
            reason = XFAIL_OPENCL_PLATFORMS[name]
            _CTX_FACTORY_XFAIL_REASON_CACHE[ctx_factory] = reason
            return reason

    _CTX_FACTORY_XFAIL_REASON_CACHE[ctx_factory] = None
    return None


def pytest_addoption(parser) -> None:
    """Add extra command line options.

    --longrun  Skip expensive tests unless told otherwise.
    --full-accuracy  Enable very expensive high-accuracy regression tests.

    """
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="enable longrundecorated tests",
    )
    parser.addoption(
        "--full-accuracy",
        action="store_true",
        dest="full_accuracy",
        default=False,
        help="enable full_accuracy marked tests",
    )


def pytest_configure(config) -> None:
    """Register this suite's markers.

    ``pytest.ini`` declares them as well; registering here keeps ``--strict-
    markers`` runs working if that file is ever folded into ``pyproject.toml``.
    """
    for marker in SUITE_MARKERS:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items) -> None:
    """Apply the ``full_accuracy`` opt-in and the OpenCL platform xfails."""
    run_full_accuracy = bool(config.getoption("full_accuracy"))

    for item in items:
        if "full_accuracy" in item.keywords and not run_full_accuracy:
            item.add_marker(
                pytest.mark.skip(reason="needs --full-accuracy option to run")
            )

        callspec = getattr(item, "callspec", None)
        if callspec is None or "ctx_factory" not in callspec.params:
            continue

        ctx_factory = callspec.params["ctx_factory"]
        xfail_reason = _get_xfail_reason_for_ctx_factory(ctx_factory)
        if xfail_reason:
            item.add_marker(pytest.mark.xfail(reason=xfail_reason, run=False))


def _create_table_build_queue() -> cl.CommandQueue:
    """Return a queue for building near-field tables, preferring non-Intel."""
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    for platform in platforms:
        if platform.name == INTEL_OPENCL_PLATFORM_NAME:
            continue
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))

    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))

    pytest.skip("No OpenCL devices available for table build")


def _remove_quietly(path: Path) -> None:
    """Delete `path` if it exists, ignoring every filesystem error."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def longrun(request):
    """Skip the requesting test unless ``--longrun`` was passed."""
    if not request.config.option.longrun:
        pytest.skip("needs --longrun option to run")


@pytest.fixture(scope="session")
def requires_pypvfmm(request):
    """Skip the requesting test unless :mod:`pypvfmm` is importable."""
    try:
        import pypvfmm  # noqa: F401
    except ImportError:
        pytest.skip("needs pypvfmm to run")


@pytest.fixture(scope="session")
def table_2d_order1(tmp_path_factory, request):
    """A shared 2D order-1 Laplace near-field table.

    Building this table dominates the runtime of the table-manager tests, so
    it is built once per session (once per ``pytest-xdist`` run, guarded by a
    file lock, when workers are in play).
    """
    worker_id = getattr(request.config, "workerinput", {}).get("workerid")

    if not worker_id:
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        queue = _create_table_build_queue()
        with NFTManager("nft.hdf5", progress_bar=True) as table_manager:
            table, _ = table_manager.get_table(2, "Laplace", q_order=1, queue=queue)
        _remove_quietly(Path("nft.hdf5"))
        return table

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "nft.hdf5"
    with FileLock(str(fn) + ".lock"):
        queue = _create_table_build_queue()
        with NFTManager(str(fn), progress_bar=True) as table_manager:
            table, _ = table_manager.get_table(2, "Laplace", q_order=1, queue=queue)
        return table


def pytest_sessionfinish(session, exitstatus) -> None:
    """Remove the table caches a run may have left in the working directory."""
    for table_file in Path.cwd().glob("*.hdf5"):
        _remove_quietly(table_file)
