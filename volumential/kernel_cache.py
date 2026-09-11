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

__doc__ = """Disk-cached :mod:`loopy` kernel generation.

This module owns :class:`KernelCacheWrapper`, the mix-in that every
:mod:`volumential` operator with a generated :mod:`loopy` kernel inherits from.
It is re-exported by :mod:`volumential.tools` for backwards compatibility.
"""

import logging
from typing import Any

from pytools import memoize_method


logger = logging.getLogger(__name__)


class KernelCacheWrapper:
    """Mix-in providing a disk-cached :mod:`loopy` kernel.

    Subclasses must implement :meth:`get_cache_key`, :meth:`get_kernel` and
    :meth:`get_optimized_kernel`, and set a ``name`` attribute used in log
    messages.
    """

    # FIXME: largely code duplication with sumpy.

    def __init__(self) -> None:
        self.name = "KernelCacheWrapper"
        raise RuntimeError("KernelCacheWrapper objects should not be constructed")

    def get_cache_key(self) -> tuple[Any, ...]:
        """Return a hashable key identifying the generated kernel."""
        raise NotImplementedError("Unimplemented cache key")

    def get_kernel(self):
        """Return the unoptimized :mod:`loopy` kernel."""
        raise NotImplementedError()

    def get_optimized_kernel(self):
        """Return the transformed (optimized) :mod:`loopy` kernel."""
        raise NotImplementedError()

    @memoize_method
    def get_cached_optimized_kernel(self, **kwargs):
        """Return the kernel for *kwargs*, using the on-disk code cache."""
        from sumpy import CACHING_ENABLED, OPT_ENABLED, code_cache

        cache_key = None
        if CACHING_ENABLED:
            import loopy.version
            from sumpy.version import KERNEL_VERSION as SUMPY_KERNEL_VERSION

            from volumential.version import KERNEL_VERSION

            cache_key = (
                self.get_cache_key()
                + tuple(sorted(kwargs.items()))
                + (loopy.version.DATA_MODEL_VERSION,)
                + (SUMPY_KERNEL_VERSION,)
                + (KERNEL_VERSION,)
                + (OPT_ENABLED,)
            )

            try:
                result = code_cache[cache_key]
                logger.debug("%s: kernel cache hit [key=%s]", self.name, cache_key)
                return result
            except KeyError:
                pass

        logger.info("%s: kernel cache miss [key=%s]", self.name, cache_key)

        from pytools import MinRecursionLimit

        with MinRecursionLimit(3000):
            if OPT_ENABLED:
                knl = self.get_optimized_kernel(**kwargs)
            else:
                knl = self.get_kernel()

        if CACHING_ENABLED:
            code_cache.store_if_not_present(cache_key, knl)

        return knl


# vim: filetype=pyopencl.python:fdm=marker
