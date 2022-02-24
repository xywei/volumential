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

__doc__ = """
Mesh generation

.. autoclass:: MeshGenBase
   :members:
.. autoclass:: MeshGen2D
   :members:
.. autoclass:: MeshGen3D
   :members:
"""

import logging
import numpy as np
import pyopencl as cl
import modepy as mp
from pytools.obj_array import make_obj_array

logger = logging.getLogger(__name__)

provider = None

# {{{ meshgen Python provider


class MeshGenBoxtree(object):
    """Base class for Meshgen via BoxTree.
    The interface is similar to the Meshgen via Deal.II, except that
    the arguments a and b can also be of higher dimensions to allow
    added flexibility on choosing the bounding box.

    The interface of this class should be kept in sync with the dealii backend.
    """
    def __init__(self, degree, nlevels=1, a=-1, b=1, dim=None, queue=None):
        """
        Specifying bounding box:
          1. Assign scalar values to a,b for [a, b]^dim
          2. Assign dim-vectors for [a0, b0] X [a1, b1] ...

        Note: due to legacy reasons, the degree argument here is the number of
        quadrature nodes in 1D (not the polynomial degree).
        """
        assert degree > 0
        assert nlevels > 0
        self.dim = dim
        self.degree = degree
        self.quadrature_formula = mp.LegendreGaussQuadrature(degree - 1)
        self.nlevels = nlevels
        self.n_box_nodes = self.degree ** self.dim
        self.queue = queue

        x = self.quadrature_formula.nodes  # nodes in [-1, 1]
        w = self.quadrature_formula.weights
        assert len(x) == self.degree
        self.box_nodes = np.array(np.meshgrid(*((x,) * self.dim), indexing="ij")
                                  ).reshape([self.dim, -1])

        if self.dim == 1:
            self.box_weights = w
        elif self.dim == 2:
            self.box_weights = w[:, None] @ w[None, :]
        elif self.dim == 3:
            self.box_weights = (w[:, None] @ w[None, :]
                                ).reshape(-1)[:, None] @ w[None, :]
        else:
            raise ValueError
        self.box_weights = self.box_weights.reshape(-1)

        # make bounding box
        if np.array(a).size == 1:
            a = np.repeat(a, dim)
        if np.array(b).size == 1:
            b = np.repeat(b, dim)

        assert len(a) == dim
        assert len(b) == dim
        assert all(ai < bi for ai, bi in zip(a, b))
        bbox = [a, b]

        self.tob = make_tob_root(self.dim, bbox)
        for i in range(self.nlevels - 1):
            self.tob = uniformly_refined(self.tob)

    def get_q_points(self):
        lfboxes = self.tob.leaf_boxes()
        nodes = np.tile(self.box_nodes, (1, len(lfboxes)))
        box_shifts = np.repeat(
            self.tob.box_centers[:, lfboxes], self.n_box_nodes, axis=1)
        box_scales = np.repeat(
            self.tob.get_box_size(lfboxes) / 2, self.n_box_nodes)
        nodes = nodes * box_scales[None, :] + box_shifts
        return nodes.T

    def get_q_weights(self):
        lfboxes = self.tob.leaf_boxes()
        weights = np.tile(self.box_weights, len(lfboxes))
        box_scales = np.repeat(
            self.tob.get_box_size(lfboxes) / 2, self.n_box_nodes)
        weights = weights * box_scales**self.tob.dim
        return weights

    def get_cell_measures(self):
        lfboxes = self.tob.leaf_boxes()
        box_scales = np.repeat(
            self.tob.get_box_size(lfboxes) / 2, self.n_box_nodes)
        return box_scales**self.tob.dim

    def get_cell_centers(self):
        return self.bob.box_centers

    def n_cells(self):
        """Note that this value can be larger than the actual number
        of cells used in the boxtree. It mainly serves as the bound for
        iterators on cells.
        """
        return self.tob.nboxes

    def n_active_cells(self):
        return len(self.tob.leaf_boxes())

    def update_mesh(
        self, criteria, top_fraction_of_cells, bottom_fraction_of_cells
    ):
        # TODO
        raise NotImplementedError

    def print_info(self, logging_func=logger.info):
        logging_func("Number of cells: " + str(self.n_cells()))
        logging_func("Number of active cells: " + str(self.n_active_cells()))
        logging_func("Number of quad points per cell: "
                + str(self.n_box_nodes))

    def generate_gmsh(self, filename):
        """
        # TODO
        Write the active boxes as a gmsh file.
        The file format specifications can be found at:
        http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format
        """
        raise NotImplementedError


# }}} End meshgen Python provider

try:
    logger.info("Trying to find a mesh generator..")
    import volumential.meshgen_dealii  # noqa: F401

    provider = "meshgen_dealii"

except ImportError as e:
    logger.debug(repr(e))
    logger.warning("Meshgen via Deal.II is not present or unusable.")

    try:
        logger.info("Trying out BoxTree backend.")
        from boxtree.tree_build import make_tob_root, uniformly_refined

        provider = "meshgen_boxtree"

    except ImportError as ee:
        logger.debug(repr(ee))
        logger.warning("Meshgen via BoxTree is not present or unusable.")
        raise RuntimeError("Cannot find a usable Meshgen implementation.")

    else:
        # {{{ Meshgen via BoxTree
        logger.info("Using Meshgen via BoxTree interface.")

        def greet():
            return "Hello from Meshgen via BoxTree!"

        def make_uniform_cubic_grid(degree, nlevels=1, dim=2, actx=None, **kwargs):
            """Uniform cubic grid in [-1,1]^dim.
            This function provides backward compatibility with meshgen_dealii.
            """
            # For meshgen_dealii compatibility
            if "level" in kwargs:
                nlevels = kwargs["level"]

            mgen = MeshGenBoxtree(degree + 1, nlevels, -1, 1, dim)
            x = mgen.get_q_points()
            q = mgen.get_q_weights()
            radii = None
            return (x, q, radii)

        class MeshGen1D(MeshGenBoxtree):
            """Meshgen in 1D
            """
            def __init__(self, degree, nlevels=1, a=-1, b=1, queue=None):
                super().__init__(degree, nlevels, a, b, 1, queue)

        class MeshGen2D(MeshGenBoxtree):
            """Meshgen in 2D
            """
            def __init__(self, degree, nlevels=1, a=-1, b=1, queue=None):
                super().__init__(degree, nlevels, a, b, 2, queue)

        class MeshGen3D(MeshGenBoxtree):
            """Meshgen in 3D
            """
            def __init__(self, degree, nlevels=1, a=-1, b=1, queue=None):
                super().__init__(degree, nlevels, a, b, 3, queue)

        # }}} End Meshgen via BoxTree

else:
    # noexcept on importing meshgen_dealii
    logger.info("Using Meshgen via Deal.II interface.")
    from volumential.meshgen_dealii import (  # noqa: F401
        greet,
        MeshGen2D,
        MeshGen3D,
    )

    def make_uniform_cubic_grid(degree, level, dim, queue=None):
        from volumential.meshgen_dealii import make_uniform_cubic_grid as _mucg
        return _mucg(degree, level, dim)


# {{{ mesh utils

def build_geometry_info(ctx, queue, dim, q_order, mesh,
                             bbox=None, a=None, b=None):
    """Build tree, traversal and other geo info for FMM computation,
    given the box mesh over/encompassing the domain.

    The bouding box can be specified in one of two ways:
    1. via scalars a, b, dim-homogeneous ([a, b]^dim)
    2. via bbox (e.g. np.array([[a1, b1], [a2, b2], [a3, b3]]))
    """

    if dim == 1:
        if not isinstance(mesh, MeshGen1D):
            raise ValueError()

    if dim == 2:
        if not isinstance(mesh, MeshGen2D):
            raise ValueError()

    elif dim == 3:
        if not isinstance(mesh, MeshGen3D):
            raise ValueError()

    else:
        raise ValueError("only supports 1 <= dim <= 3")

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

# }}} End mesh utils

# vim: ft=pyopencl
