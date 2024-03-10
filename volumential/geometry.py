__copyright__ = "Copyright (C) 2019 Xiaoyu Wei"

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

import numpy as np

import pyopencl as cl
from boxtree.pyfmmlib_integration import FMMLibRotationData
from pytools.obj_array import make_obj_array


# {{{ bounding box factory


class BoundingBoxFactory:
    def __init__(self, dim,
            center=None, radius=None,
            dtype=float):
        self.dim = dim
        self.dtype = np.dtype(dtype)

        self.center = center
        self.radius = radius

        self.box_center = None
        self.box_radius = None
        self.lbounds = None
        self.ubounds = None

    @property
    def materialized(self):
        """Returns true if the bounding box is readily determined,
        i.e., the __call__() method has been called.
        """
        return (self.box_center is not None)

    def __call__(self, ctx,
            expand_to_hold_mesh=None, mesh_padding_factor=0.05):
        """If expand the box to be large enough to enclose a
        mesh object.

        When center is given during initialization, the box center is
        preserved during expansion.

        When radius is given during initialization, the radius will not
        shrink and will be at least the specified value.

        :arg ctx: OpenCL context.

        :arg expand_to_hold_mesh: A :class:`meshmode.mesh.Mesh` object.

        :arg mesh_padding_factor: Controls how far out the box boundary is
            relative to the mesh.
        """
        if not expand_to_hold_mesh:
            if (self.center is None) or (self.radius is None):
                raise ValueError(
                        "A bounding box without presets "
                        "must be adapted to an mesh.")
            box_center = self.center
            box_radius = self.radius
        else:
            a = np.min(expand_to_hold_mesh.vertices, axis=1)
            b = np.max(expand_to_hold_mesh.vertices, axis=1)
            c = (a + b) * 0.5

            if self.center is None:
                box_center = c
            else:
                box_center = self.center

            if self.radius is None:
                box_radius = np.max(b - c)
            else:
                box_radius = max(self.radius, np.max(b - c))

        self.box_center = box_center
        self.box_radius = box_radius
        self.lbounds = box_center - box_radius
        self.ubounds = box_center + box_radius

        from boxtree.bounding_box import make_bounding_box_dtype
        from boxtree.tools import AXIS_NAMES

        axis_names = AXIS_NAMES[:self.dim]
        bbox_type, _ = make_bounding_box_dtype(
                ctx.devices[0], self.dim, self.dtype)

        bbox = np.empty(1, bbox_type)
        for ax, iaxis in zip(axis_names, range(self.dim)):
            bbox["min_" + ax] = self.lbounds[iaxis]
            bbox["max_" + ax] = self.ubounds[iaxis]

        return bbox

# }}} End bounding box factory

# {{{ box mesh data factory


class BoxFMMGeometryFactory:
    """Builds the geometry information needed for performing volume FMM.
    The objects of this class is designed to be "interactive", i.e., be
    manipulated repeatedly and on demand, produce a series of desired
    :class:`BoxFMMGeometryData` objects.

    .. attribute:: cl_context
        A :class:`pyopencl.Context` object.

    .. attribute:: quadrature_formula
        The quadrature formula, a :class:`modepy.quadrature.Quadrature` object.

    .. attribute:: order
        The quadrature order.

    .. attribute:: nlevels
        The number of (initial) levels.

    .. attribute:: bbox_getter
        A :class:`volumential.geometry.BoundingBoxFactory` object.

    .. attribute:: engine
        Box mesh generator that implements the
        :class:`volumential.meshgen.MeshGenBase` interface.
    """
    def __init__(
            self, cl_ctx, dim, order, nlevels, bbox_getter,
            quadrature_formula=None,
            expand_to_hold_mesh=None, mesh_padding_factor=0.05):
        """
        :arg bbox_getter: A :class:`BoundingBoxFactory` object.

        :arg expand_to_hold_mesh: (Optional) A :class:`meshmode.mesh.Mesh`
                                  object.

        :arg mesh_padding_factor: (Optional) Controls how far out the box
                                  boundary is relative to the mesh.
        """
        if dim == 1:
            raise NotImplementedError()
        elif dim == 2:
            from volumential.meshgen import MeshGen2D
            self._engine_class = MeshGen2D
        elif dim == 3:
            from volumential.meshgen import MeshGen3D
            self._engine_class = MeshGen3D

        self.dim = dim
        if quadrature_formula is None:
            from modepy import LegendreGaussQuadrature

            # order = degree + 1
            self.quadrature_formula = LegendreGaussQuadrature(order - 1)
        else:
            raise NotImplementedError()

        self.bbox_getter = bbox_getter

        self.cl_context = cl_ctx
        self._bbox = bbox_getter(self.cl_context,
                expand_to_hold_mesh, mesh_padding_factor)

        self.order = order
        self.nlevels = nlevels

        # the engine generates meshes centered at the origin
        a = -1 * self.bbox_getter.box_radius
        b = self.bbox_getter.box_radius
        self.engine = self._engine_class(self.order, self.nlevels, a, b)

    def reinit(
            self, order=None, nlevels=None,
            quadrature_formula=None,
            expand_to_hold_mesh=None, mesh_padding_factor=None):
        """Resets the engine to its initial state, and optionally also
        changes some parameters.
        """
        if order is not None:
            self.order = order

        if nlevels is not None:
            self.nlevels = nlevels

        if expand_to_hold_mesh is not None:
            self._bbox = self.bbox_getter(self.cl_context,
                    expand_to_hold_mesh, mesh_padding_factor)

        # the engine generates meshes centered at the origin
        a = -1 * self.bbox_getter.radius
        b = self.bbox_getter.radius
        self.engine = self._engine_class(order, nlevels, a, b)

    def _get_q_points(self, queue=None):
        q_points_pre = self.engine.get_q_points()
        q_points = np.ascontiguousarray(np.transpose(q_points_pre))

        # re-center the box
        q_points = q_points + self.bbox_getter.box_center[:, None]

        if queue is None:
            return q_points
        else:
            return make_obj_array(
                    [cl.array.to_device(queue, q_points[i])
                     for i in range(self.dim)])

    def _get_q_weights(self, queue=None):
        q_weights = self.engine.get_q_weights()
        if queue is None:
            return q_weights
        else:
            return cl.array.to_device(queue, q_weights)

    def _get_active_cell_centers(self, queue=None):
        cell_centers_pre = self.engine.get_cell_centers()
        cell_centers = np.ascontiguousarray(np.transpose(cell_centers_pre))

        # re-center the box
        cell_centers = cell_centers + self.bbox_getter.box_center[:, None]

        if queue is None:
            return cell_centers
        else:
            return make_obj_array(
                    [cl.array.to_device(queue, cell_centers[i])
                     for i in range(self.dim)])

    def _get_active_cell_measures(self, queue=None):
        cell_measures = self.engine.get_cell_measures()
        if queue is None:
            return cell_measures
        else:
            return cl.array.to_device(queue, cell_measures)

    def _get_active_cell_extents(self, queue=None):
        return self._get_active_cell_measures(queue) ** (1 / self.dim)

    @property
    def n_cells(self):
        return self.engine.n_cells()

    @property
    def n_active_cells(self):
        return self.engine.n_active_cells()

    @property
    def n_q_points_per_cell(self):
        return len(self.quadrature_formula.weights)

    def __call__(self, queue=None):
        if queue is None:
            queue = cl.CommandQueue(self.cl_context)

        from boxtree import TreeBuilder
        tb = TreeBuilder(self.cl_context)

        q_points = self._get_q_points(queue)

        tree, _ = tb(queue, particles=q_points, targets=q_points,
                     bbox=self._bbox, max_particles_in_box=(
                         (self.n_q_points_per_cell**self.dim) * (2**self.dim)
                         - 1),
                     kind="adaptive-level-restricted")

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(self.cl_context)
        trav, _ = tg(queue, tree)

        return BoxFMMGeometryData(
                self.cl_context,
                q_points, self._get_q_weights(queue),
                tree, trav)

# }}} End box mesh data factory


class BoxFMMGeometryData(FMMLibRotationData):
    """A container class for geometric data associated with volume
    FMM evaluations. The objects of this class are treated as immutable.

    .. attribute:: nodes
    .. attribute:: weights
    .. attribute:: tree
    .. attribute:: traversal
    """
    def __init__(self, cl_context, q_points, q_weights, tree, trav):
        self.cl_context = cl_context
        self.queue = cl.CommandQueue(cl_context)

        self.nodes = q_points
        self.weights = q_weights
        self.tree = tree
        self.traversal = trav

        super().__init__(self.queue, self.traversal)
