
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

import pytest
from filelock import FileLock

# setup the ctx_factory fixture
from pyopencl.tools import (  # NOQA
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from volumential.table_manager import NearFieldInteractionTableManager as NFTManager


def pytest_addoption(parser):
    """Add extra command line options.

    --longrun  Skip expensive tests unless told otherwise.

    """
    parser.addoption("--longrun", action="store_true", dest="longrun",
                     default=False, help="enable longrundecorated tests")


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
def table_2d_order1(tmp_path_factory, worker_id):
    if not worker_id:
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        with NFTManager("nft.hdf5", progress_bar=True) as table_manager:
            table, _ = table_manager.get_table(2, "Laplace", q_order=1)
        subprocess.check_call(["rm", "-f", "nft.hdf5"])
        return table

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "nft.hdf5"
    with FileLock(str(fn) + ".lock"):
        with NFTManager(str(fn), progress_bar=True) as table_manager:
            table, _ = table_manager.get_table(2, "Laplace", q_order=1)
        return table
    return table


def pytest_sessionfinish(session, exitstatus):
    # remove table caches
    for table_file in glob.glob("*.hdf5"):
        subprocess.call(["rm", "-f", table_file])
