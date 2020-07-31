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

import itertools
import numpy as np
import pyopencl as cl
import loopy as lp
from pytools import memoize_method, ProcessLogger
from pytools.obj_array import make_obj_array
from boxtree.tools import DeviceDataRecord
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import unflatten, flatten, thaw
from volumential.volume_fmm import interpolate_volume_potential

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

.. autofunction:: interpolate_from_meshmode

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
    .. attribute:: trav

        The :class:`boxtree.FMMTraversalInfo` instance representing the
        box mesh with metadata needed for interpolation. It contains a
        reference to the underlying tree as `trav.tree`.

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

    .. attribute:: nodes_in_leaf_lists

        Indices into :attr:`discr.nodes()`.

        .. note:: Unlike :class:`ElementsToSourcesLookup`, lists are not disjoint
            in the leaves-to-nodes lookup. :mod:`volumential` automatically computes
            the average contribution from overlapping boxes.

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

        from pyopencl.algorithm import KeyValueSorter
        self.key_value_sorter = KeyValueSorter(context)

        from boxtree.area_query import AreaQueryBuilder
        self.area_query_builder = AreaQueryBuilder(self.context)

    # {{{ kernel generation

    @memoize_method
    def codegen_get_dimension_specific_snippets(self):
        """Dimension-dependent code loopy instructions.
        """
        import sympy as sp
        axis_names = ["x", "y", "z"]
        axis_names = axis_names[:self.dim]

        # tolerance
        tol = -1e-12

        def make_sympy_vec(comp_names):
            comps = []
            for cn in comp_names:
                comps.append(sp.var(cn))
            return sp.Matrix(comps)

        def get_simplex_measure(vtx_names):
            mat0 = sp.ones(self.dim + 1)
            for iv, v in enumerate(vtx_names):
                vtx = sp.Matrix([sp.var(f"{v}{comp}") for comp in axis_names])
                mat0[iv, :-1] = vtx.T
            return str(mat0.det())

        if self.dim == 2:

            # {{{ 2d

            code_get_simplex = \
                """
                <> Ax = mesh_vertices_0[mesh_vertex_indices[iel, 0]]
                <> Ay = mesh_vertices_1[mesh_vertex_indices[iel, 0]]
                <> Bx = mesh_vertices_0[mesh_vertex_indices[iel, 1]]
                <> By = mesh_vertices_1[mesh_vertex_indices[iel, 1]]
                <> Cx = mesh_vertices_0[mesh_vertex_indices[iel, 2]]
                <> Cy = mesh_vertices_1[mesh_vertex_indices[iel, 2]]
                """
            code_get_point = \
                """
                <> Px = source_points_0[source_id]
                <> Py = source_points_1[source_id]
                """
            # simplex measures
            code_s0 = get_simplex_measure(["P", "B", "C"])
            code_s1 = get_simplex_measure(["A", "P", "C"])
            code_s2 = get_simplex_measure(["A", "B", "P"])
            code_compute_simplex_measures = \
                f"""
                <> s0 = {code_s0}
                <> s1 = {code_s1}
                <> s2 = {code_s2}
                """
            code_measures_have_common_sign = " and ".join([
                f"s{c1} * s{c2} >= {tol}"
                for c1, c2 in itertools.combinations(["0", "1"], 2)])

            # }}} End 2d

        elif self.dim == 3:

            # {{{ 3d

            code_get_simplex = \
                """
                <> Ax = mesh_vertices_0[mesh_vertex_indices[iel, 0]]
                <> Ay = mesh_vertices_1[mesh_vertex_indices[iel, 0]]
                <> Az = mesh_vertices_2[mesh_vertex_indices[iel, 0]]
                <> Bx = mesh_vertices_0[mesh_vertex_indices[iel, 1]]
                <> By = mesh_vertices_1[mesh_vertex_indices[iel, 1]]
                <> Bz = mesh_vertices_2[mesh_vertex_indices[iel, 1]]
                <> Cx = mesh_vertices_0[mesh_vertex_indices[iel, 2]]
                <> Cy = mesh_vertices_1[mesh_vertex_indices[iel, 2]]
                <> Cz = mesh_vertices_2[mesh_vertex_indices[iel, 2]]
                <> Dx = mesh_vertices_0[mesh_vertex_indices[iel, 3]]
                <> Dy = mesh_vertices_1[mesh_vertex_indices[iel, 3]]
                <> Dz = mesh_vertices_2[mesh_vertex_indices[iel, 3]]
                """
            code_get_point = \
                """
                <> Px = source_points_0[source_id]
                <> Py = source_points_1[source_id]
                <> Pz = source_points_2[source_id]
                """
            # simplex measures
            code_s0 = get_simplex_measure(["P", "B", "C", "D"])
            code_s1 = get_simplex_measure(["A", "P", "C", "D"])
            code_s2 = get_simplex_measure(["A", "B", "P", "D"])
            code_s3 = get_simplex_measure(["A", "B", "C", "P"])
            code_compute_simplex_measures = \
                f"""
                <> s0 = {code_s0}
                <> s1 = {code_s1}
                <> s2 = {code_s2}
                <> s3 = {code_s3}
                """
            code_measures_have_common_sign = " and ".join([
                f"s{c1} * s{c2} >= {tol}"
                for c1, c2 in itertools.combinations(["0", "1", "2"], 2)])

            # }}} End 3d

        else:
            raise NotImplementedError()

        return {"code_get_simplex": code_get_simplex,
                "code_get_point": code_get_point,
                "code_compute_simplex_measures": code_compute_simplex_measures,
                "code_measures_have_common_sign": code_measures_have_common_sign,
                }

    @memoize_method
    def get_simplex_lookup_kernel(self):
        """Returns a loopy kernel that computes a potential vector
        representing the (q_point --> element_id) relationship.
        When a source q_point lies on the element boundary, it will be
        assigned an element depending on code scheduling. This ensures
        that the resulting lookup lists are disjoint.

        The kernel assumes that the mesh uses one single group of simplex elements.
        Also, the test only works for affine elements.
        """
        logger.debug("start building elements-to-sources lookup kernel")

        snippets = self.codegen_get_dimension_specific_snippets()
        loopy_knl = lp.make_kernel(
            ["{ [ iel ]: 0 <= iel < nelements }",
             "{ [ ineighbor ]: nearby_leaves_beg <= ineighbor < nearby_leaves_end }",
             "{ [ isrc ]: 0 <= isrc < n_box_sources }"
             ],
            ["""
            for iel
                <> nearby_leaves_beg = leaves_near_ball_starts[iel]
                <> nearby_leaves_end = leaves_near_ball_starts[iel + 1]

                {code_get_simplex}

                for ineighbor
                    <> ileaf = leaves_near_ball_lists[ineighbor]
                    <> box_source_beg = box_source_starts[ileaf]
                    <> n_box_sources = box_source_counts_cumul[ileaf]

                    for isrc
                        <> source_id = box_source_beg + isrc

                        {code_get_point}
                        {code_compute_simplex_measures}

                        result[source_id] = if(
                            {code_measures_have_common_sign},
                            iel,
                            result[source_id])  {{atomic}}
                    end
                end
            end
            """.format(**snippets)],
            [lp.ValueArg("nelements, dim, nboxes, nsources", np.int32),
             lp.GlobalArg("mesh_vertex_indices", np.int32, "nelements, dim+1"),
             lp.GlobalArg("box_source_starts", np.int32, "nboxes"),
             lp.GlobalArg("box_source_counts_cumul", np.int32, "nboxes"),
             lp.GlobalArg("leaves_near_ball_lists", np.int32, None),
             lp.GlobalArg("result", np.int32, "nsources", for_atomic=True),
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

        mesh_vertices_kwargs = {
            f"mesh_vertices_{iaxis}": vertices_dev[iaxis]
            for iaxis in range(self.dim)}

        source_points_kwargs = {
            f"source_points_{iaxis}": self.tree.sources[iaxis]
            for iaxis in range(self.dim)}

        evt, res = element_lookup_kernel(
            queue, dim=self.dim, nboxes=self.tree.nboxes,
            nelements=self.discr.mesh.nelements, nsources=self.tree.nsources,
            result=cl.array.zeros(queue, self.tree.nsources, dtype=np.int32) - 1,
            mesh_vertex_indices=self.discr.mesh.groups[0].vertex_indices,
            box_source_starts=self.tree.box_source_starts,
            box_source_counts_cumul=self.tree.box_source_counts_cumul,
            leaves_near_ball_starts=balls_to_leaves_lookup.leaves_near_ball_starts,
            leaves_near_ball_lists=balls_to_leaves_lookup.leaves_near_ball_lists,
            wait_for=wait_for, **mesh_vertices_kwargs, **source_points_kwargs)

        source_to_element_lookup, = res

        wait_for = [evt]

        # elements = source_to_element_lookup.get()
        # for idx in [362,  365,  874,  877, 1386, 1389, 1898, 1901])

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


# {{{ leaves-to-nodes lookup builder

class LeavesToNodesLookupBuilder:
    """Given a :mod:`meshmod` mesh and a :mod:`boxtree.Tree`, both discretizing
    the same bounding box, this class helps to build a look-up table from
    leaf boxes to mesh nodes that are positioned inside the box.
    """

    def __init__(self, context, trav, discr):
        """
        :arg trav: a :class:`boxtree.FMMTraversalInfo`
        :arg discr: a :class: `meshmode.discretization.Discretization`

        Boxes and elements can be non-aligned as long as the domains
        (bounding boxes) are the same.
        """
        assert trav.tree.dimensions == discr.dim
        self.dim = discr.dim
        self.context = context
        self.trav = trav
        self.discr = discr

        from boxtree.area_query import LeavesToBallsLookupBuilder
        self.leaves_to_balls_lookup_builder = \
            LeavesToBallsLookupBuilder(self.context)

    def __call__(self, queue, tol=1e-12, wait_for=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`
        :tol: nodes close enough to the boundary will be treated as
            lying on the boundary, whose interpolated values are averaged.
        """
        arr_ctx = PyOpenCLArrayContext(queue)
        nodes = flatten(thaw(arr_ctx, self.discr.nodes()))
        radii = cl.array.zeros_like(nodes[0]) + tol

        lbl_lookup, evt = self.leaves_to_balls_lookup_builder(
            queue, self.trav.tree, nodes, radii, wait_for=wait_for)

        return LeavesToNodesLookup(
            trav=self.trav, discr=self.discr,
            nodes_in_leaf_starts=lbl_lookup.balls_near_box_starts,
            nodes_in_leaf_lists=lbl_lookup.balls_near_box_lists), evt

# }}} End leaves-to-nodes lookup builder


# {{{ transform helper

def compute_affine_transform(source_simplex, target_simplex):
    """Computes A and b for the affine transform :math:`y = A x + b`
    that maps ``source_simplex`` to ``target_simplex``.

    :param source_simplex: a dim-by-(dim+1) :mod:`numpy` array
    :param target_simplex: a dim-by-(dim+1) :mod:`numpy: array
    """
    assert source_simplex.shape == target_simplex.shape
    dim = source_simplex.shape[0]

    if dim == 2:
        assert source_simplex.shape == (2, 3)
        mat = np.zeros([6, 6])
        mat[:3, :2] = source_simplex.T
        mat[-3:, 2:4] = source_simplex.T
        mat[:3, -2] = 1
        mat[-3:, -1] = 1
        rhs = target_simplex.reshape(-1)
        solu = np.linalg.solve(mat, rhs)
        return solu[:4].reshape(2, 2), solu[-2:]

    elif dim == 3:
        assert source_simplex.shape == (3, 4)
        mat = np.zeros([12, 12])
        mat[:4, :3] = source_simplex.T
        mat[4:8, 3:6] = source_simplex.T
        mat[-4:, 6:9] = source_simplex.T
        mat[:4, -3] = 1
        mat[4:8, -2] = 1
        mat[-4:, -1] = 1
        rhs = target_simplex.reshape(-1)
        solu = np.linalg.solve(mat, rhs)
        return solu[:9].reshape(3, 3), solu[-3:]

    else:
        raise NotImplementedError()


def invert_affine_transform(mat_a, disp_b):
    """Inverts an affine transform given by :math:`y = A x + b`.

    :param mat_A: a dim*(dim+1)-by-dim*(dim+1) :mod:`numpy` array
    :param disp_b: a dim*(dim+1) :mod:`numpy` array
    """
    iva = np.linalg.inv(mat_a)
    ivb = - iva @ disp_b
    return iva, ivb

# }}} End transform helper


# {{{ from meshmode interpolation

def interpolate_from_meshmode(queue, dof_vec, elements_to_sources_lookup,
                              order="tree"):
    """Interpolate a DoF vector from :mod:`meshmode`.

    :arg dof_vec: a DoF vector representing a field in :mod:`meshmode`
        of shape ``(..., nnodes)``.
    :arg elements_to_sources_lookup: a :class:`ElementsToSourcesLookup`.
    :arg order: order of the output potential, either "tree" or "user".

    .. note:: This function currently supports meshes with just one element
        group. Also, the element group must be simplex-based.

    .. note:: This function does some heavy-lifting computation in Python,
        which we intend to optimize in the future. In particular, we plan
        to shift the batched linear solves and basis evaluations to
        :mod:`loopy`.

    TODO: make linear solvers available as :mod:`loopy` callables.
    TODO: make :mod:`modepy` emit :mod:`loopy` callables for basis evaluation.
    """
    if not isinstance(dof_vec, cl.array.Array):
        raise TypeError("non-array passed to interpolator")

    assert len(elements_to_sources_lookup.discr.groups) == 1
    assert len(elements_to_sources_lookup.discr.mesh.groups) == 1
    degroup = elements_to_sources_lookup.discr.groups[0]
    megroup = elements_to_sources_lookup.discr.mesh.groups[0]

    if not degroup.is_affine:
        raise ValueError(
            "interpolation requires global-to-local map, "
            "which is only available for affinely mapped elements")

    mesh = elements_to_sources_lookup.discr.mesh
    dim = elements_to_sources_lookup.discr.dim
    template_simplex = mesh.groups[0].vertex_unit_coordinates().T

    # -------------------------------------------------------
    # Inversely map source points with a global-to-local map.
    #
    # 1. For each element, solve for the affine map.
    #
    # 2. Apply the map to corresponding source points.
    #
    # This step computes `unit_sources`, the list of inversely
    # mapped source points.

    sources_in_element_starts = \
        elements_to_sources_lookup.sources_in_element_starts.get(queue)
    sources_in_element_lists = \
        elements_to_sources_lookup.sources_in_element_lists.get(queue)
    tree = elements_to_sources_lookup.tree.get(queue)

    unit_sources_host = make_obj_array(
            [np.zeros_like(srccrd) for srccrd in tree.sources])

    for iel in range(degroup.nelements):
        vertex_ids = megroup.vertex_indices[iel]
        vertices = mesh.vertices[:, vertex_ids]
        afa, afb = compute_affine_transform(vertices, template_simplex)

        beg = sources_in_element_starts[iel]
        end = sources_in_element_starts[iel + 1]
        source_ids_in_el = sources_in_element_lists[beg:end]
        sources_in_el = np.vstack(
            [tree.sources[iaxis][source_ids_in_el] for iaxis in range(dim)])

        ivmapped_el_sources = afa @ sources_in_el + afb.reshape([dim, 1])
        for iaxis in range(dim):
            unit_sources_host[iaxis][source_ids_in_el] = \
                ivmapped_el_sources[iaxis, :]

    unit_sources = make_obj_array(
        [cl.array.to_device(queue, usc) for usc in unit_sources_host])

    # -----------------------------------------------------
    # Carry out evaluations in the local (template) frames.
    #
    # 1. Assemble a resampling matrix for each element, with
    #    the basis functions and the local source points.
    #
    # 2. For each element, perform matvec on the resampling
    #    matrix and the local DoF coefficients.
    #
    # This step assumes `unit_sources` computed on device, so
    # that the previous step can be swapped with a kernel without
    # interrupting the followed computation.

    mapped_sources = np.vstack(
        [usc.get(queue) for usc in unit_sources])

    basis_funcs = degroup.basis()

    arr_ctx = PyOpenCLArrayContext(queue)
    dof_vec_view = unflatten(
            arr_ctx, elements_to_sources_lookup.discr, dof_vec)[0]
    dof_vec_view = dof_vec_view.get()

    sym_shape = dof_vec.shape[:-1]
    source_vec = np.zeros(sym_shape + (tree.nsources, ))

    for iel in range(degroup.nelements):
        beg = sources_in_element_starts[iel]
        end = sources_in_element_starts[iel + 1]
        source_ids_in_el = sources_in_element_lists[beg:end]
        mapped_sources_in_el = mapped_sources[:, source_ids_in_el]
        local_dof_vec = dof_vec_view[..., iel, :]

        # resampling matrix built from Vandermonde matrices
        import modepy as mp
        rsplm = mp.resampling_matrix(
                basis=basis_funcs,
                new_nodes=mapped_sources_in_el,
                old_nodes=degroup.unit_nodes)

        if len(sym_shape) == 0:
            local_coeffs = local_dof_vec
            source_vec[source_ids_in_el] = rsplm @ local_coeffs
        else:
            from pytools import indices_in_shape
            for sym_id in indices_in_shape(sym_shape):
                source_vec[sym_id + (source_ids_in_el, )] = \
                    rsplm @ local_dof_vec[sym_id]

    source_vec = cl.array.to_device(queue, source_vec)

    if order == "tree":
        pass  # no need to do anything
    elif order == "user":
        source_vec = source_vec[tree.sorted_target_ids]  # into user order
    else:
        raise ValueError(f"order must be 'tree' or 'user' (got {order}).")

    return source_vec

# }}} End from meshmode interpolation


# {{{ to meshmode interpolation

def interpolate_to_meshmode(queue, potential, leaves_to_nodes_lookup,
                            order="tree"):
    """
    :arg potential: a DoF vector representing a field in :mod:`volumential`,
        in tree order.
    :arg leaves_to_nodes_lookup: a :class:`LeavesToNodesLookup`.
    :arg order: order of the input potential, either "tree" or "user".

    :returns: a :class:`pyopencl.Array` of shape (nnodes, 1) containing the
        interpolated data.
    """
    if order == "tree":
        potential_in_tree_order = True
    elif order == "user":
        potential_in_tree_order = False
    else:
        raise ValueError(f"order must be 'tree' or 'user' (got {order}).")

    arr_ctx = PyOpenCLArrayContext(queue)
    target_points = flatten(thaw(
        arr_ctx, leaves_to_nodes_lookup.discr.nodes()))

    traversal = leaves_to_nodes_lookup.trav
    tree = leaves_to_nodes_lookup.trav.tree

    dim = tree.dimensions

    # infer q_order from tree
    pts_per_box = tree.ntargets // traversal.ntarget_boxes
    assert pts_per_box * traversal.ntarget_boxes == tree.ntargets

    # allow for +/- 0.25 floating point error
    q_order = int(pts_per_box**(1 / dim) + 0.25)
    assert q_order**dim == pts_per_box

    interp_p = interpolate_volume_potential(
            target_points=target_points, traversal=traversal,
            wrangler=None, potential=potential,
            potential_in_tree_order=potential_in_tree_order,
            dim=dim, tree=tree, queue=queue, q_order=q_order,
            dtype=potential.dtype, lbl_lookup=None,
            balls_near_box_starts=leaves_to_nodes_lookup.nodes_in_leaf_starts,
            balls_near_box_lists=leaves_to_nodes_lookup.nodes_in_leaf_lists)

    return interp_p

# }}} End to meshmode interpolation
