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

__doc__ = """Miscellaneous host-side utilities, plus the historical grab-bag
import surface.

This module owns two small host-side helpers, :func:`clean_file` and
:func:`import_code`.  Everything else it used to define now lives in a focused
module and is re-exported here unchanged, so that ``from volumential.tools
import ...`` keeps working:

* :class:`~volumential.kernel_cache.KernelCacheWrapper` --
  :mod:`volumential.kernel_cache`
* :class:`~volumential.expression_eval.ScalarFieldExpressionEvaluation` --
  :mod:`volumential.expression_eval`
* :class:`~volumential.box_operators.BoxSpecificMap`,
  :class:`~volumential.box_operators.DiscreteLegendreTransform`,
  :class:`~volumential.box_operators.InverseDiscreteLegendreTransform`,
  :class:`~volumential.box_operators.BoxSpecificReduction`,
  :class:`~volumential.box_operators.BoxSum` and
  :func:`~volumential.box_operators.generate_leading_order_filtering` --
  :mod:`volumential.box_operators`
"""

import logging
from pathlib import Path
from types import ModuleType

from volumential.box_operators import (
    BoxSpecificMap,
    BoxSpecificReduction,
    BoxSum,
    DiscreteLegendreTransform,
    InverseDiscreteLegendreTransform,
    generate_leading_order_filtering,
)
from volumential.expression_eval import ScalarFieldExpressionEvaluation
from volumential.kernel_cache import KernelCacheWrapper


logger = logging.getLogger(__name__)


# {{{ clean files


def clean_file(filename, new_name=None) -> None:
    """Remove/rename file if exists.
    Fails silently when the file does not exist.
    Useful for, for example, writing output files that
    are meant to overwrite existing ones.
    """
    path = Path(filename)

    try:
        if new_name is None:
            path.unlink()
        else:
            path.rename(new_name)
    except OSError:
        logger.debug("clean_file: could not remove/rename %s", path)


# }}} End clean files

# {{{ import code


def import_code(code, name, add_to_sys_modules=True) -> ModuleType:
    """Dynamically generates a module.

    :arg code: can be any object containing code -- string, file object, or
    compiled code object. Returns a new module object initialized
    by dynamically importing the given code and optionally adds it
    to sys.modules under the given name.
    """
    module = ModuleType(name)

    if add_to_sys_modules:
        import sys

        sys.modules[name] = module

    exec(code, module.__dict__)

    return module


# }}} End import code


__all__ = [
    "BoxSpecificMap",
    "BoxSpecificReduction",
    "BoxSum",
    "DiscreteLegendreTransform",
    "InverseDiscreteLegendreTransform",
    "KernelCacheWrapper",
    "ScalarFieldExpressionEvaluation",
    "clean_file",
    "generate_leading_order_filtering",
    "import_code",
]

# vim: filetype=pyopencl.python:fdm=marker
