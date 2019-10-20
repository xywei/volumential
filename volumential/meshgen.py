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
from pytools.obj_array import make_obj_array

logger = logging.getLogger(__name__)

provider = None

# {{{ meshgen Python provider


class MeshGenBase(object):
    """Base class for Meshgen via BoxTree.
    The interface is similar to the Meshgen via Deal.II, except that
    the arguments a and b can also be of higher dimensions to allow
    added flexibility on choosing the bounding box.

    This base class cannot be used directly since the boxtree is not
    built in the constructor until dimension_specific_setup() is
    provided.

    In addition to the common capabilities of the deal.II implementation,
    this class also implements native CL getters like get_q_points_dev()
    that play well with the other libraries like boxtree.
    """

    def __init__(self, degree, nlevels, a=-1, b=1, queue=None):
        assert degree > 0
        assert nlevels > 0
        self.degree = degree
        self.quadrature_formula = GaussLegendreQuadrature(degree - 1)
        self.nlevels = nlevels

        self.bound_a = np.array([a]).flatten()
        self.bound_b = np.array([b]).flatten()
        assert len(self.bound_a) == len(self.bound_b)
        assert np.all(self.bound_a < self.bound_b)

        self.dim = len(self.bound_a)
        self.root_vertex = self.bound_a
        self.root_extent = np.max(self.bound_b - self.bound_a)

        if queue is None:
            ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(ctx)
        else:
            self.queue = queue

        # plug in dimension-specific details
        self.dimension_specific_setup()
        self.n_q_points = self.degree ** self.dim

        self.boxtree = BoxTree()
        self.boxtree.generate_uniform_boxtree(
            self.queue,
            nlevels=self.nlevels,
            root_extent=self.root_extent,
            root_vertex=self.root_vertex,
        )
        self.quadrature = QuadratureOnBoxTree(
            self.boxtree, self.quadrature_formula
        )

    def dimension_specific_setup(self):
        pass

    def get_q_points_dev(self):
        return self.quadrature.get_q_points(self.queue)

    def get_q_points(self):
        q_points_dev = self.get_q_points_dev()
        n_all_q_points = len(q_points_dev[0])
        q_points = np.zeros((n_all_q_points, self.dim))
        for d in range(self.dim):
            q_points[:, d] = q_points_dev[d].get(self.queue)
        return q_points

    def get_q_weights_dev(self):
        return self.quadrature.get_q_weights(self.queue)

    def get_q_weights(self):
        q_weights_dev = self.get_q_weights_dev()
        return q_weights_dev.get(self.queue)

    def get_cell_measures_dev(self):
        return self.quadrature.get_cell_measures(self.queue)

    def get_cell_measures(self):
        cell_measures_dev = self.get_cell_measures_dev()
        return cell_measures_dev.get(self.queue)

    def get_cell_centers_dev(self):
        return self.quadrature.get_cell_centers(self.queue)

    def get_cell_centers(self):
        cell_centers_dev = self.get_cell_centers_dev()
        n_active_cells = self.n_active_cells()
        cell_centers = np.zeros((n_active_cells, self.dim))
        for d in range(self.dim):
            cell_centers[:, d] = cell_centers_dev[d].get(self.queue)
        return cell_centers

    def n_cells(self):
        """Note that this value can be larger than the actual number
        of cells used in the boxtree. It mainly serves as the bound for
        iterators on cells.
        """
        return self.boxtree.nboxes

    def n_active_cells(self):
        return self.boxtree.n_active_boxes

    def update_mesh(
        self, criteria, top_fraction_of_cells, bottom_fraction_of_cells
    ):
        # TODO
        raise NotImplementedError

    def print_info(self, logging_func=print):
        logging_func("Number of cells: " + str(self.n_cells()))
        logging_func("Number of active cells: " + str(self.n_active_cells()))
        logging_func("Number of quad points per cell: "
                + str(self.n_q_points))

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
    logger.warn("Meshgen via Deal.II is not present or unusable.")

    try:
        logger.info("Trying out BoxTree.TreeInteractiveBuild interface.")
        import boxtree.tree_interactive_build  # noqa: F401
        from modepy import GaussLegendreQuadrature

        provider = "meshgen_boxtree"

    except ImportError as ee:
        logger.debug(repr(ee))
        logger.warn("Meshgen via BoxTree is not present or unusable.")
        raise RuntimeError("Cannot find a usable Meshgen implementation.")

    else:
        # {{{ Meshgen via BoxTree
        logger.info("Using Meshgen via BoxTree interface.")
        from boxtree.tree_interactive_build import BoxTree, QuadratureOnBoxTree

        def greet():
            return "Hello from Meshgen via BoxTree!"

        def make_uniform_cubic_grid(degree, nlevels=1, dim=2, queue=None, **kwargs):
            """Uniform cubic grid in [-1,1]^dim.
            This function provides backward compatibility with meshgen_dealii.
            """
            if queue is None:
                ctx = cl.create_some_context()
                queue = cl.CommandQueue(ctx)

            # For meshgen_dealii compatibility
            if "level" in kwargs:
                nlevels = kwargs["level"]

            tree = BoxTree()
            tree.generate_uniform_boxtree(
                queue, nlevels=nlevels, root_extent=2, root_vertex=np.zeros(dim) - 1
            )
            quad_rule = GaussLegendreQuadrature(degree - 1)
            quad = QuadratureOnBoxTree(tree, quad_rule)
            q_weights = quad.get_q_weights(queue).get(queue)
            q_points_dev = quad.get_q_points(queue)
            n_all_q_points = len(q_weights)
            q_points = np.zeros((n_all_q_points, dim))
            for d in range(dim):
                q_points[:, d] = q_points_dev[d].get(queue)

            # Adding a placeholder for deprecated point radii
            return (q_points, q_weights, None)

        class MeshGen1D(MeshGenBase):
            """Meshgen in 1D
            """

            def dimension_specific_setup(self):
                assert self.dim == 1

        class MeshGen2D(MeshGenBase):
            """Meshgen in 2D
            """

            def dimension_specific_setup(self):
                if self.dim == 1:
                    # allow passing scalar values of a and b to the constructor
                    self.dim = 2
                    self.root_vertex = np.zeros(self.dim) + self.bound_a
                else:
                    assert self.dim == 2

        class MeshGen3D(MeshGenBase):
            """Meshgen in 3D
            """

            def dimension_specific_setup(self):
                if self.dim == 1:
                    # allow passing scalar values of a and b to the constructor
                    self.dim = 3
                    self.root_vertex = np.zeros(self.dim) + self.bound_a
                else:
                    assert self.dim == 3

        # }}} End Meshgen via BoxTree

else:
    # noexcept on importing meshgen_dealii
    logger.info("Using Meshgen via Deal.II interface.")
    from volumential.meshgen_dealii import (  # noqa: F401
        greet,
        MeshGen2D,
        MeshGen3D,
        make_uniform_cubic_grid,
    )


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
