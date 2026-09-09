"""Volumential computes 2D and 3D volume potentials using the fast multipole
method.

This module is the package entry point.  It owns

- the re-exported top-level names listed in :data:`__all__`
  (:class:`~volumential.nearfield_potential_table.NearFieldInteractionTable`,
  :class:`~volumential.table_manager.NearFieldInteractionTableManager` and
  :func:`~volumential.singular_integral_2d.box_quad`),
- the package version string :data:`volumential_version`,
- the persistent :data:`code_cache` used by generated :mod:`loopy` kernels, and
- the process-wide optimization and caching switches
  (:data:`OPT_ENABLED`, :data:`CACHING_ENABLED`, :func:`set_optimization_enabled`,
  :func:`set_caching_enabled` and :class:`CacheMode`).

Everything else lives in the submodules; see the documentation's module map.
"""

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
from types import TracebackType

from pytools.persistent_dict import WriteOncePersistentDict

from volumential.nearfield_potential_table import NearFieldInteractionTable
from volumential.singular_integral_2d import box_quad
from volumential.table_manager import NearFieldInteractionTableManager
from volumential.version import VERSION_TEXT


volumential_version = VERSION_TEXT

__all__ = [
    "CACHING_ENABLED",
    "OPT_ENABLED",
    "CacheMode",
    "NearFieldInteractionTable",
    "NearFieldInteractionTableManager",
    "box_quad",
    "code_cache",
    "nearfield_potential_table",
    "set_caching_enabled",
    "set_optimization_enabled",
    "volumential_version",
]

code_cache = WriteOncePersistentDict(
    "volumential-code-cache-v0-" + VERSION_TEXT,
    safe_sync=False,
)

# {{{ optimization control

OPT_ENABLED = True

OPT_ENABLED = "VOLUMENTIAL_NO_OPT" not in os.environ


def set_optimization_enabled(flag: bool) -> None:
    """Set whether the :mod:`loopy` kernels should be optimized."""
    global OPT_ENABLED
    OPT_ENABLED = flag


# }}}

# {{{ cache control


CACHING_ENABLED = True

CACHING_ENABLED = (
    "VOLUMENTIAL_NO_CACHE" not in os.environ and "CG_NO_CACHE" not in os.environ
)


def set_caching_enabled(flag: bool) -> None:
    """Set whether :mod:`loopy` is allowed to use disk caching for its various
    code generation stages.
    """
    global CACHING_ENABLED
    CACHING_ENABLED = flag


class CacheMode:
    """A context manager for setting whether :mod:`volumential` is allowed to use
    disk caches.
    """

    def __init__(self, new_flag: bool) -> None:
        self.new_flag = new_flag

    def __enter__(self) -> None:
        global CACHING_ENABLED
        self.previous_mode = CACHING_ENABLED
        CACHING_ENABLED = self.new_flag

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        global CACHING_ENABLED
        CACHING_ENABLED = self.previous_mode
        del self.previous_mode


# }}}

# vim: filetype=pyopencl:fdm=marker
