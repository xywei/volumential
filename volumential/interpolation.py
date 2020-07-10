__copyright__ = """
Copyright (C) 2020 Xiaoyu Wei
"""

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
import loopy as lp
from pyopencl.algorithm import KeyValueSorter
from pytools import memoize_method, ProcessLogger
from pytools.obj_array import make_obj_array
from boxtree.area_query import AreaQueryBuilder
from boxtree.tools import DeviceDataRecord

import logging
logger = logging.getLogger(__name__)


__doc__ = r"""
.. currentmodule:: volumential

From :mod:`meshmode`
-------------------------

Interpolation from functions given by DoF vectors of :mod:`meshmode`.
The underlying mesh on the :mod:`meshmode` side must be discretizing the
same bounding box.

The intersection testing assumes a boundedness property for the supported
element types: each element is bounded inside the smallest :math:`l^\infty`
ball that is centered at the element's center and covers all its vertices.
This property might be broken, for example, by high order elements that
warp the element boundary too much.

.. autoclass:: ElementsToSourcesLookup

.. autoclass:: LeavesToNodesLookup

To :mod:`meshmode`
---------------------------

"""


# {{{ output

class ElementsToSourcesLookup(DeviceDataRecord):
    """
    .. attribute:: tree

        The :class:`boxtree.Tree` instance representing the box mesh.

    .. attribute:: discr

        The :class:`meshmode.discretization.Discretization` instance
        representing the external mesh and DoF distribution.

    .. attribute:: sources_in_element_starts

        Indices into :attr:`sources_in_element_lists`.

        .. code-block:: python

            sources_in_element_lists[
                sources_in_element_starts[global_iel]
                :sources_in_element_starts[global_iel] + 1
                ]

        contains the list of source nodes residing in the given element.

        .. note:: ``global_iel`` is the global element id in `meshmode`.
            ``global_iel = mesh.groups[igrp].element_nr_base + iel``.

    .. attribute:: sources_in_element_lists

        Indices into :attr:`tree.sources`.

    .. automethod:: get
    """


class LeavesToNodesLookup(DeviceDataRecord):
    """
    .. attribute:: tree

        The :class:`boxtree.Tree` instance representing the box mesh.

    .. attribute:: discr

        The :class:`meshmode.discretization.Discretization` instance
        representing the external mesh and DoF distribution.

    .. attribute:: nodes_in_leaf_starts

        Indices into :attr:`nodes_in_leaf_lists`.

        .. code-block:: python

            nodes_in_leaf_lists[
                nodes_in_leaf_starts[box_id]:nodes_in_leaf_starts[box_id] + 1]

        contains the list of discretization nodes residing in the given leaf box.

        .. note:: Only leaf boxes have non-empty entries in this table.
            Nonetheless, this list is indexed by the global box index.

    .. attribute:: box_nodes_in_element_lists

        Indices into :attr:`tree.sources`.

    .. automethod:: get
    """

# }}} End output


# {{{ elements-to-sources lookup builder

class ElementsToSourcesLookupBuilder:
    """Given a :mod:`meshmod` mesh and a :mod:`boxtree.Tree`, both discretizing
    the same bounding box, this class helps to build a look-up table from
    element to source nodes that are positioned inside the element.
    """

    def __init__(self, context, tree, discr):
        """
        :arg tree: a :class:`boxtree.Tree`
        :arg discr: a :class: `meshmode.discretization.Discretization`

        Boxes and elements can be non-aligned as long as the domains
        (bounding boxes) are the same.
        """
        assert tree.dimensions == discr.dim
        self.dim = discr.dim
        self.context = context
        self.tree = tree
        self.discr = discr

        self.key_value_sorter = KeyValueSorter(context)
        self.area_query_builder = AreaQueryBuilder(self.context)

    # {{{ kernel generation

    @memoize_method
    def get_simplex_lookup_kernel(self):
        """Returns a loopy kernel that computes a potential vector
        representing the (q_point --> element_id) relationship.
        When a source q_point lies on the element boundary, it will be
        assigned an element depending on code scheduling. This ensures
        that the resulting lookup lists are disjoint.

        The kernel assumes that the mesh uses one single group of simplex elements.
        """
        logger.debug("start building elements-to-sources lookup kernel")

        if self.dim != 2:
            raise NotImplementedError()

        loopy_knl = lp.make_kernel(
            ["{ [ iel ]: 0 <= iel < nelements }",
             "{ [ ineighbor ]: nearby_leaves_beg <= ineighbor < nearby_leaves_end }",
             "{ [ isrc ]: 0 <= isrc < n_box_sources }"
             ],
            ["""
            for iel
                <> nearby_leaves_beg = leaves_near_ball_starts[iel]
                <> nearby_leaves_end = leaves_near_ball_starts[iel + 1]

                <> Ax = mesh_vertices_0[mesh_vertex_indices[iel, 0]]
                <> Ay = mesh_vertices_1[mesh_vertex_indices[iel, 0]]
                <> Bx = mesh_vertices_0[mesh_vertex_indices[iel, 1]]
                <> By = mesh_vertices_1[mesh_vertex_indices[iel, 1]]
                <> Cx = mesh_vertices_0[mesh_vertex_indices[iel, 2]]
                <> Cy = mesh_vertices_1[mesh_vertex_indices[iel, 2]]

                for ineighbor
                    <> ileaf = leaves_near_ball_lists[ineighbor]
                    <> box_source_beg = box_source_starts[ileaf]
                    <> n_box_sources = box_source_counts_cumul[ileaf]

                    for isrc
                        <> source_id = box_source_beg + isrc
                        <> Px = source_points_0[source_id]
                        <> Py = source_points_1[source_id]

                        <> s0 = (Px - Ax) * (By - Ay) - (Py - Ay) * (Bx - Ax)
                        <> s1 = (Px - Bx) * (Cy - By) - (Py - By) * (Cx - Bx)
                        <> s2 = (Px - Cx) * (Ay - Cy) - (Py - Cy) * (Ax - Cx)

                        result[source_id] = if(
                            s0 * s1 >= 0 and s1 * s2 >= 0,
                            iel,
                            result[source_id])
                    end
                end
            end
            """],
            [lp.ValueArg("nelements, dim, nboxes, nsources", np.int32),
             lp.GlobalArg("mesh_vertex_indices", np.int32, "nelements, dim+1"),
             lp.GlobalArg("box_source_starts", np.int32, "nboxes"),
             lp.GlobalArg("box_source_counts_cumul", np.int32, "nboxes"),
             lp.GlobalArg("leaves_near_ball_lists", np.int32, None),
             lp.GlobalArg("result", np.int32, "nsources"),
             "..."],
            name="build_sources_in_simplex_lookup",
            lang_version=(2018, 2),
        )

        logger.debug("done building elements-to-sources lookup kernel")
        return loopy_knl

    # }}} End kernel generation

    def compute_short_lists(self, queue, wait_for=None):
        """balls --> overlapping leaves
        """
        mesh = self.discr.mesh
        if len(mesh.groups) > 1:
            raise NotImplementedError("Mixed elements not supported")
        melgrp = mesh.groups[0]
        ball_centers_host = (np.max(melgrp.nodes, axis=2)
                        + np.min(melgrp.nodes, axis=2)) / 2
        ball_radii_host = np.max(
                np.max(melgrp.nodes, axis=2) - np.min(melgrp.nodes, axis=2),
                axis=0) / 2

        ball_centers = make_obj_array([
            cl.array.to_device(queue, center_coord_comp)
            for center_coord_comp in ball_centers_host])
        ball_radii = cl.array.to_device(queue, ball_radii_host)

        area_query_result, evt = self.area_query_builder(
            queue, self.tree, ball_centers, ball_radii,
            peer_lists=None, wait_for=wait_for)
        return area_query_result, evt

    def __call__(self, queue, balls_to_leaves_lookup=None, wait_for=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`
        """
        slk_plog = ProcessLogger(logger, "element-to-source lookup: run area query")

        if balls_to_leaves_lookup is None:
            balls_to_leaves_lookup, evt = \
                self.compute_short_lists(queue, wait_for=wait_for)
            wait_for = [evt]

        # -----------------------------------------------------------------
        # Refine the area query using point-in-simplex test

        logger.debug("element-to-source lookup: refine starts")

        element_lookup_kernel = self.get_simplex_lookup_kernel()

        vertices_dev = make_obj_array([
            cl.array.to_device(queue, verts)
            for verts in self.discr.mesh.vertices])

        evt, res = element_lookup_kernel(
            queue, dim=self.dim, nboxes=self.tree.nboxes,
            nelements=self.discr.mesh.nelements, nsources=self.tree.nsources,
            mesh_vertex_indices=self.discr.mesh.groups[0].vertex_indices,
            mesh_vertices_0=vertices_dev[0],
            mesh_vertices_1=vertices_dev[1],
            source_points_0=self.tree.sources[0],
            source_points_1=self.tree.sources[1],
            box_source_starts=self.tree.box_source_starts,
            box_source_counts_cumul=self.tree.box_source_counts_cumul,
            leaves_near_ball_starts=balls_to_leaves_lookup.leaves_near_ball_starts,
            leaves_near_ball_lists=balls_to_leaves_lookup.leaves_near_ball_lists,
            wait_for=wait_for)

        source_to_element_lookup, = res
        wait_for = [evt]

        # -----------------------------------------------------------------
        # Invert the source-to-element lookup by a key-value sort

        logger.debug("element-to-source lookup: key-value sort")

        sources_in_element_starts, sources_in_element_lists, evt = \
            self.key_value_sorter(
                queue,
                keys=source_to_element_lookup,
                values=cl.array.arange(
                    queue, self.tree.nsources, dtype=self.tree.box_id_dtype),
                nkeys=self.discr.mesh.nelements,
                starts_dtype=self.tree.box_id_dtype,
                wait_for=wait_for)

        slk_plog.done()

        return ElementsToSourcesLookup(
            tree=self.tree, discr=self.discr,
            sources_in_element_starts=sources_in_element_starts,
            sources_in_element_lists=sources_in_element_lists), evt

# }}} End elements-to-sources lookup builder


def interpolate_from_meshmode(queue, dof_vec, elements_to_sources_lookup):
    """
    :arg dof_vec: a DoF vector representing a field in :mod:`meshmode`
    :arg elements_to_sources_lookup: a :class:`ElementsToSourcesLookup`
    """
    # TODO
