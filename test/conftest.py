from __future__ import absolute_import, division, print_function

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

import pytest  # noqa: F401

# setup the ctx_factory fixture
from pyopencl.tools import (  # NOQA
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)


def pytest_addoption(parser):
    """Add extra command line options.

    --longrun  Skip expensive tests unless told otherwise.

    """
    parser.addoption('--longrun', action='store_true', dest="longrun",
                     default=False, help="enable longrundecorated tests")


@pytest.fixture(scope='session')
def longrun(request):
    if not request.config.option.longrun:
        pytest.skip("needs --longrun option to run")


@pytest.fixture(scope='session')
def requires_pypvfmm(request):
    try:
        import pypvfmm  # noqa: F401
    except ImportError:
        pytest.skip("needs pypvfmm to run")


def pytest_sessionstart(session):
    from volumential.table_manager import NearFieldInteractionTableManager

    # clean the table file, in case there was an aborted previous test run
    import subprocess
    subprocess.call(['rm', '-f', '/tmp/volumential-tests.hdf5'])

    # pre-compute a basic table that is re-used in many tests.
    # can be easily turned off to run individual tests that do not require
    # the table cache.
    if 0:
        with NearFieldInteractionTableManager(
                "/tmp/volumential-tests.hdf5") as tm:
            table, _ = tm.get_table(
                    2, "Laplace",
                    q_order=1, force_recompute=False, queue=None)


def pytest_sessionfinish(session, exitstatus):
    # remove table caches
    import subprocess
    subprocess.call(['rm', '-f', '/tmp/volumential-tests.hdf5'])
