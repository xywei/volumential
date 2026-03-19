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

from volumential.tree_interactive_build import BoxTree, QuadratureOnBoxTree


logger = logging.getLogger(__name__)

provider = None

from boxtree import (  # noqa: E402
    make_tree_of_boxes_root,
    refine_and_coarsen_tree_of_boxes,
    uniformly_refine_tree_of_boxes,
)
from modepy import LegendreGaussQuadrature  # noqa: E402

# {{{ meshgen Python provider


class MeshGenBase:
    """Base class for Meshgen via BoxTree.
    The arguments a and b can be scalars or vectors to define
    dim-dependent bounding boxes.

    This base class cannot be used directly since the boxtree is not
    built in the constructor until dimension_specific_setup() is
    provided.

    This class also implements native CL getters like get_q_points_dev()
    that play well with boxtree-based workflows.
    """

    def __init__(self, degree, nlevels, a=-1, b=1, queue=None):
        assert degree > 0
        assert nlevels > 0
        self.degree = degree
        self.quadrature_formula = LegendreGaussQuadrature(degree - 1)
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
        self.n_q_points = self.degree**self.dim

        self.boxtree = BoxTree()
        self.boxtree.generate_uniform_boxtree(
            self.queue,
            nlevels=self.nlevels,
            root_extent=self.root_extent,
            root_vertex=self.root_vertex,
        )
        self.quadrature = QuadratureOnBoxTree(self.boxtree, self.quadrature_formula)

    def _leaf_boxes(self):
        return self.boxtree.active_boxes.get()

    def _leaf_levels(self):
        return self.boxtree.box_levels.get()[self._leaf_boxes()]

    def _leaf_centers(self):
        return self.boxtree.box_centers.get()[:, self._leaf_boxes()].T

    def _leaf_side_lengths(self):
        return self.boxtree.root_extent / (2 ** self._leaf_levels())

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
        return self.get_q_weights_dev().get(self.queue)

    def get_cell_measures_dev(self):
        return self.quadrature.get_cell_measures(self.queue)

    def get_cell_measures(self):
        return self.get_cell_measures_dev().get(self.queue)

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

    def update_mesh(self, criteria, top_fraction_of_cells, bottom_fraction_of_cells):
        criteria = np.asarray(criteria)
        leaf_boxes = self._leaf_boxes()

        if len(criteria) != len(leaf_boxes):
            raise ValueError("criteria must match the number of active cells")

        nleaves = len(leaf_boxes)
        if nleaves == 0:
            return

        refine_count = min(nleaves, int(np.ceil(top_fraction_of_cells * nleaves)))
        coarsen_count = min(nleaves, int(np.floor(bottom_fraction_of_cells * nleaves)))

        refine_flags = np.zeros(self.boxtree.nboxes, dtype=bool)
        coarsen_flags = np.zeros(self.boxtree.nboxes, dtype=bool)

        order = np.argsort(criteria)

        if refine_count:
            refine_leaf_boxes = leaf_boxes[order[-refine_count:]]
            refine_flags[refine_leaf_boxes] = True

        if coarsen_count:
            coarsen_leaf_boxes = leaf_boxes[order[:coarsen_count]]
            coarsen_flags[coarsen_leaf_boxes] = True

        coarsen_flags[refine_flags] = False

        self.boxtree.refine_and_coarsen(
            refine_flags=refine_flags,
            coarsen_flags=coarsen_flags,
            error_on_ignored_flags=False,
        )

    def print_info(self, logging_func=logger.info):
        logging_func("Number of cells: " + str(self.n_cells()))
        logging_func("Number of active cells: " + str(self.n_active_cells()))
        logging_func("Number of quad points per cell: " + str(self.n_q_points))

    def generate_gmsh(self, filename):
        """
        # TODO
        Write the active boxes as a gmsh file.
        The file format specifications can be found at:
        http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format
        """
        raise NotImplementedError


# }}} End meshgen Python provider

provider = "meshgen_boxtree"
logger.info("Using meshgen via current boxtree tree-of-boxes interface.")


def greet():
    return "Hello from Meshgen via BoxTree!"


def make_uniform_cubic_grid(degree, nlevels=1, dim=2, queue=None, **kwargs):
    """Uniform cubic grid in [-1,1]^dim."""
    if queue is None:
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

    if "level" in kwargs:
        nlevels = kwargs["level"]

    mesh_cls = {1: MeshGen1D, 2: MeshGen2D, 3: MeshGen3D}[dim]
    mesh = mesh_cls(degree, nlevels, a=-1, b=1, queue=queue)
    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()

    return (q_points, q_weights, None)


class MeshGen1D(MeshGenBase):
    """Meshgen in 1D"""

    def dimension_specific_setup(self):
        assert self.dim == 1


class MeshGen2D(MeshGenBase):
    """Meshgen in 2D"""

    def dimension_specific_setup(self):
        if self.dim == 1:
            self.dim = 2
            self.root_vertex = np.zeros(self.dim) + self.bound_a
        else:
            assert self.dim == 2


class MeshGen3D(MeshGenBase):
    """Meshgen in 3D"""

    def dimension_specific_setup(self):
        if self.dim == 1:
            self.dim = 3
            self.root_vertex = np.zeros(self.dim) + self.bound_a
        else:
            assert self.dim == 3


# {{{ mesh utils


def build_geometry_info(ctx, queue, dim, q_order, mesh, bbox=None, a=None, b=None):
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
        [cl.array.to_device(queue, q_points[i]) for i in range(dim)]
    )
    q_weights = cl.array.to_device(queue, q_weights)

    if bbox is None:
        assert np.isscalar(a) and np.isscalar(b)
        bbox = np.array([[a, b]] * dim)

    from boxtree import TreeBuilder
    from boxtree.array_context import PyOpenCLArrayContext

    actx = PyOpenCLArrayContext(queue)

    tb = TreeBuilder(actx)

    tree, _ = tb(
        actx,
        particles=q_points,
        targets=q_points,
        max_particles_in_box=q_order**dim * (2**dim) - 1,
        kind="adaptive-level-restricted",
    )

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(actx)
    trav, _ = tg(actx, tree)

    return q_points, q_weights, tree, trav


# }}} End mesh utils

# vim: ft=pyopencl
