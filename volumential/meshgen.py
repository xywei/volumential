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

__doc__ = """
Mesh generation.

.. autoclass:: MeshGen1D
   :members:
.. autoclass:: MeshGen2D
   :members:
.. autoclass:: MeshGen3D
   :members:

.. autofunction:: greet
.. autofunction:: make_uniform_cubic_grid
.. autofunction:: build_geometry_info
"""

import logging

import numpy as np

import pyopencl as cl
from pytools.obj_array import make_obj_array


try:
    from volumential.meshgen_dealii import MeshGen1D, MeshGen2D, MeshGen3D, greet
    PROVIDER = "meshgen_dealii"

    def make_uniform_cubic_grid(degree, level, dim, queue=None):
        from volumential.meshgen_dealii import make_uniform_cubic_grid as _mucg
        return _mucg(degree, level, dim)
except ImportError:
    try:
        from volumential.meshgen_boxtree import (
            MeshGen1D, MeshGen2D, MeshGen3D, greet)
        PROVIDER = "meshgen_boxtree"
    except ImportError:
        raise RuntimeError(
            "Cannot find a usable mesh generation implementation. "
            "Install the deal.II implementation or the boxtree "
            "implementation to use mesh generation.")


# FIXME: lowercase variable should be deprecated and removed
provider = PROVIDER

__all__ = (
    "provider", "PROVIDER",
    "MeshGen1D", "MeshGen2D", "MeshGen3D",
    "make_uniform_cubic_grid",
    "greet",

    "build_geometry_info",
)

logger = logging.getLogger(__name__)


# {{{ mesh utils

def build_geometry_info(ctx, queue, dim, q_order, mesh,
                        bbox=None, a=None, b=None):
    """Build tree, traversal and other geometric information for FMM computation.

    The bouding box can be specified in one of two ways:
    1. via scalars a, b, dim-homogeneous ([a, b]^dim)
    2. via bbox (e.g. np.array([[a1, b1], [a2, b2], [a3, b3]]))
    """

    if dim != mesh.dim:
        raise ValueError(
            f"Got a mesh of dimension {mesh.dim}, but expected dimension {dim}")

    if not 1 <= dim <= 3:
        raise ValueError(f"Unsupported dimension {dim}: 1 <= dim <= 3")

    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    q_points_org = q_points  # noqa: F841
    q_points = np.ascontiguousarray(np.transpose(q_points))

    q_points = make_obj_array(
            [cl.array.to_device(queue, q_points[i])
                for i in range(dim)])
    q_weights = cl.array.to_device(queue, q_weights)

    if bbox is None:
        assert np.isscalar(a) and np.isscalar(b)
        bbox = np.array([[a, b], ] * dim)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(
        queue,
        particles=q_points,
        targets=q_points,
        bbox=bbox,
        max_particles_in_box=q_order**dim * (2**dim) - 1,
        kind="adaptive-level-restricted",
    )

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tree)

    return q_points, q_weights, tree, trav

# }}}
