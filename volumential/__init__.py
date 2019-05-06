from __future__ import division

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

from volumential.singular_integral_2d import box_quad
from volumential.version import VERSION_TEXT
from volumential.table_manager import NearFieldInteractionTableManager  # noqa: F401
from volumential.nearfield_potential_table import (  # noqa: F401
    NearFieldInteractionTable,
)

volumential_version = VERSION_TEXT

__all__ = ["volumential_version", "box_quad", "nearfield_potential_table"]
"""
try:
    import volumential.meshgen as meshgen  # noqa: F401
    __all__.append("meshgen")
except ImportError:
    raise ImportError("Could not import volumential.meshgen. "
            "You need to either run 'build.sh' in "
            "volumential/contrib/meshgen11_dealii or install a "
            "version of boxtree that provides interactive tree "
            "building.")
"""

__doc__ = """
:mod:`volumential` can compute 2/3D volume potentials using FMM.
"""

# vim: filetype=pyopencl:fdm=marker
