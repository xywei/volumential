__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

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
import os
from pytools.persistent_dict import WriteOncePersistentDict

from volumential.singular_integral_2d import box_quad
from volumential.version import VERSION_TEXT
from volumential.table_manager import NearFieldInteractionTableManager  # noqa: F401
from volumential.nearfield_potential_table import (  # noqa: F401
    NearFieldInteractionTable,
)

volumential_version = VERSION_TEXT

__all__ = ["volumential_version", "box_quad", "nearfield_potential_table"]

__doc__ = """
:mod:`volumential` can compute 2/3D volume potentials using FMM.
"""

code_cache = WriteOncePersistentDict("volumential-code-cache-v0-"+VERSION_TEXT)

# {{{ optimization control

OPT_ENABLED = True

OPT_ENABLED = "VOLUMENTIAL_NO_OPT" not in os.environ


def set_optimization_enabled(flag):
    """Set whether the :mod:`loopy` kernels should be optimized."""
    global OPT_ENABLED
    OPT_ENABLED = flag

# }}}

# {{{ cache control


CACHING_ENABLED = True

CACHING_ENABLED = (
    "VOLUMENTIAL_NO_CACHE" not in os.environ
    and "CG_NO_CACHE" not in os.environ)


def set_caching_enabled(flag):
    """Set whether :mod:`loopy` is allowed to use disk caching for its various
    code generation stages.
    """
    global CACHING_ENABLED
    CACHING_ENABLED = flag


class CacheMode:
    """A context manager for setting whether :mod:`volumential` is allowed to use
    disk caches.
    """

    def __init__(self, new_flag):
        self.new_flag = new_flag

    def __enter__(self):
        global CACHING_ENABLED
        self.previous_mode = CACHING_ENABLED
        CACHING_ENABLED = self.new_flag

    def __exit__(self, exc_type, exc_val, exc_tb):
        global CACHING_ENABLED
        CACHING_ENABLED = self.previous_mode
        del self.previous_mode

# }}}

# vim: filetype=pyopencl:fdm=marker
