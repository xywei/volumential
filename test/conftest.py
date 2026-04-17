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

import glob
import subprocess

import pyopencl as cl
import pytest
from filelock import FileLock

# setup the ctx_factory fixture
from pyopencl.tools import (  # noqa: F401
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

from volumential.table_manager import NearFieldInteractionTableManager as NFTManager


XFAIL_OPENCL_PLATFORMS = {
    # Intel OpenCL CPU backend on ipa has been observed to core-dump on
    # volumential nearfield/FMM test paths; keep this visible as an xfail
    # until upstream/backend stability is confirmed.
    "Intel(R) OpenCL": "known Intel OpenCL backend crash (core dump) on volumential nearfield path",
}


_CTX_FACTORY_XFAIL_REASON_CACHE = {}


def _get_xfail_reason_for_ctx_factory(ctx_factory):
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


def pytest_addoption(parser):
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


def pytest_collection_modifyitems(config, items):
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


def _create_table_build_queue():
    try:
        platforms = cl.get_platforms()
    except cl.LogicError as exc:
        pytest.skip(f"OpenCL platforms unavailable: {exc}")

    for platform in platforms:
        if platform.name == "Intel(R) OpenCL":
            continue
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))

    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            return cl.CommandQueue(cl.Context([devices[0]]))

    pytest.skip("No OpenCL devices available for table build")


@pytest.fixture(scope="session")
def longrun(request):
    if not request.config.option.longrun:
        pytest.skip("needs --longrun option to run")


@pytest.fixture(scope="session")
def requires_pypvfmm(request):
    try:
        import pypvfmm  # noqa: F401
    except ImportError:
        pytest.skip("needs pypvfmm to run")


@pytest.fixture(scope="session")
def table_2d_order1(tmp_path_factory, request):
    worker_id = getattr(request.config, "workerinput", {}).get("workerid")

    if not worker_id:
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        queue = _create_table_build_queue()
        with NFTManager("nft.hdf5", progress_bar=True) as table_manager:
            table, _ = table_manager.get_table(2, "Laplace", q_order=1, queue=queue)
        subprocess.check_call(["rm", "-f", "nft.hdf5"])
        return table

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "nft.hdf5"
    with FileLock(str(fn) + ".lock"):
        queue = _create_table_build_queue()
        with NFTManager(str(fn), progress_bar=True) as table_manager:
            table, _ = table_manager.get_table(2, "Laplace", q_order=1, queue=queue)
        return table
    return table


def pytest_sessionfinish(session, exitstatus):
    # remove table caches
    for table_file in glob.glob("*.hdf5"):
        subprocess.call(["rm", "-f", table_file])
