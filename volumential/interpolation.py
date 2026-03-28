__copyright__ = "Copyright (C) 2020 Xiaoyu Wei"

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
import logging
import math
from functools import lru_cache

import numpy as np

import loopy as lp
import pyopencl as cl
from arraycontext import flatten
from boxtree.array_context import PyOpenCLArrayContext as BoxtreePyOpenCLArrayContext
from boxtree.tools import DeviceDataRecord
from meshmode.array_context import PyOpenCLArrayContext as MeshmodePyOpenCLArrayContext
from meshmode.dof_array import DOFArray
from pytools import ProcessLogger, memoize_method
from pytools.obj_array import new_1d as obj_array_1d

logger = logging.getLogger(__name__)


def _get_boxtree_actx(context):
    if isinstance(context, cl.CommandQueue):
        return BoxtreePyOpenCLArrayContext(context)

    if isinstance(context, cl.Context):
        return BoxtreePyOpenCLArrayContext(cl.CommandQueue(context))

    if isinstance(context, BoxtreePyOpenCLArrayContext):
        return context

    if hasattr(context, "queue"):
        return BoxtreePyOpenCLArrayContext(context.queue)

    raise TypeError(f"unsupported context type: {type(context).__name__}")


def _compute_leaves_to_nodes_lookup_tol(tree, tol):
    coord_dtype = np.dtype(tree.coord_dtype)
    root_scale = max(1.0, float(abs(tree.root_extent)))
    relative_tol = 64.0 * np.finfo(coord_dtype).eps * root_scale

    nlevels = max(1, int(getattr(tree, "nlevels", 1)))
    finest_level_extent = float(abs(tree.root_extent)) / (1 << (nlevels - 1))
    leaf_cap = 0.25 * finest_level_extent
    if leaf_cap > 0:
        relative_tol = min(relative_tol, leaf_cap)

    return max(float(tol), relative_tol)


def _count_missing_nodes_from_leaf_starts(leaf_starts, queue):
    if hasattr(leaf_starts, "with_queue"):
        leaf_starts = leaf_starts.with_queue(queue)

    if len(leaf_starts) <= 1:
        return 0

    hit_counts = leaf_starts[1:] - leaf_starts[:-1]
    missing_flags = (hit_counts == 0).astype(np.int32)
    return int(cl.array.sum(missing_flags).get(queue))


def _make_constant_array(queue, template, value):
    ary = cl.array.empty(queue, template.shape, dtype=template.dtype)
    ary.fill(np.asarray(value, dtype=template.dtype))
    return ary


@lru_cache(maxsize=16)
def _build_from_meshmode_resampling_kernel(dtype_descr):
    dtype = np.dtype(dtype_descr)

    loopy_knl = lp.make_kernel(
        [
            "{ [ isym, islot, idof ] : "
            "0 <= isym < nvecs and "
            "0 <= islot < n_source_slots and "
            "0 <= idof < nunit_dofs }",
        ],
        """
            for isym, islot
                <> source_id = sources_in_element_lists[islot]
                <> iel = source_element_indices[islot]
                source_vec[isym, source_id] = sum(idof,
                    rsplm[islot, idof] * dof_by_el[isym, iel, idof])
            end
        """,
        [
            lp.GlobalArg("rsplm", dtype, "n_source_slots, nunit_dofs"),
            lp.GlobalArg("dof_by_el", dtype, "nvecs, nelements, nunit_dofs"),
            lp.GlobalArg("source_element_indices", np.int32, "n_source_slots"),
            lp.GlobalArg("sources_in_element_lists", np.int32, "n_source_slots"),
            lp.GlobalArg("source_vec", dtype, "nvecs, nsources"),
            lp.ValueArg("nvecs", np.int32),
            lp.ValueArg("nelements", np.int32),
            lp.ValueArg("n_source_slots", np.int32),
            lp.ValueArg("nunit_dofs", np.int32),
            lp.ValueArg("nsources", np.int32),
        ],
        name="apply_from_meshmode_resampling",
        lang_version=(2018, 2),
    )

    loopy_knl = lp.set_options(loopy_knl, return_dict=True)
    return loopy_knl


@lru_cache(maxsize=16)
def _build_from_meshmode_map_to_template_kernel(dim, coord_dtype_descr):
    coord_dtype = np.dtype(coord_dtype_descr)

    if dim == 2:
        instructions = """
            for islot
                <> source_id = sources_in_element_lists[islot]
                <> iel = source_element_indices[islot]

                <> Ax = mesh_vertices_0[mesh_vertex_indices[iel, 0]]
                <> Ay = mesh_vertices_1[mesh_vertex_indices[iel, 0]]
                <> Bx = mesh_vertices_0[mesh_vertex_indices[iel, 1]]
                <> By = mesh_vertices_1[mesh_vertex_indices[iel, 1]]
                <> Cx = mesh_vertices_0[mesh_vertex_indices[iel, 2]]
                <> Cy = mesh_vertices_1[mesh_vertex_indices[iel, 2]]

                <> Px = source_points_0[source_id]
                <> Py = source_points_1[source_id]

                <> detM = (Bx - Ax) * (Cy - Ay) - (Cx - Ax) * (By - Ay)
                <> lam1 = ((Px - Ax) * (Cy - Ay) - (Cx - Ax) * (Py - Ay)) / detM
                <> lam2 = ((Bx - Ax) * (Py - Ay) - (Px - Ax) * (By - Ay)) / detM
                <> lam0 = 1 - lam1 - lam2

                barycentric_0[islot] = lam0
                barycentric_1[islot] = lam1
                barycentric_2[islot] = lam2
            end
        """

        args = [
            lp.GlobalArg("source_element_indices", np.int32, "n_source_slots"),
            lp.GlobalArg("sources_in_element_lists", np.int32, "n_source_slots"),
            lp.GlobalArg("mesh_vertex_indices", np.int32, "nelements, dim+1"),
            lp.GlobalArg("mesh_vertices_0", coord_dtype, "n_mesh_vertices"),
            lp.GlobalArg("mesh_vertices_1", coord_dtype, "n_mesh_vertices"),
            lp.GlobalArg("source_points_0", coord_dtype, "nsources"),
            lp.GlobalArg("source_points_1", coord_dtype, "nsources"),
            lp.GlobalArg("barycentric_0", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_1", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_2", coord_dtype, "n_source_slots"),
            lp.ValueArg("n_source_slots", np.int32),
            lp.ValueArg("nelements", np.int32),
            lp.ValueArg("nsources", np.int32),
            lp.ValueArg("n_mesh_vertices", np.int32),
            lp.ValueArg("dim", np.int32),
        ]

    elif dim == 3:
        instructions = """
            for islot
                <> source_id = sources_in_element_lists[islot]
                <> iel = source_element_indices[islot]

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

                <> Px = source_points_0[source_id]
                <> Py = source_points_1[source_id]
                <> Pz = source_points_2[source_id]

                <> v0x = Bx - Ax
                <> v0y = By - Ay
                <> v0z = Bz - Az
                <> v1x = Cx - Ax
                <> v1y = Cy - Ay
                <> v1z = Cz - Az
                <> v2x = Dx - Ax
                <> v2y = Dy - Ay
                <> v2z = Dz - Az

                <> vpx = Px - Ax
                <> vpy = Py - Ay
                <> vpz = Pz - Az

                <> detM = (
                    v0x * (v1y * v2z - v1z * v2y)
                    - v1x * (v0y * v2z - v0z * v2y)
                    + v2x * (v0y * v1z - v0z * v1y)
                )

                <> lam1 = (
                    vpx * (v1y * v2z - v1z * v2y)
                    - v1x * (vpy * v2z - vpz * v2y)
                    + v2x * (vpy * v1z - vpz * v1y)
                ) / detM

                <> lam2 = (
                    v0x * (vpy * v2z - vpz * v2y)
                    - vpx * (v0y * v2z - v0z * v2y)
                    + v2x * (v0y * vpz - v0z * vpy)
                ) / detM

                <> lam3 = (
                    v0x * (v1y * vpz - v1z * vpy)
                    - v1x * (v0y * vpz - v0z * vpy)
                    + vpx * (v0y * v1z - v0z * v1y)
                ) / detM

                <> lam0 = 1 - lam1 - lam2 - lam3

                barycentric_0[islot] = lam0
                barycentric_1[islot] = lam1
                barycentric_2[islot] = lam2
                barycentric_3[islot] = lam3
            end
        """

        args = [
            lp.GlobalArg("source_element_indices", np.int32, "n_source_slots"),
            lp.GlobalArg("sources_in_element_lists", np.int32, "n_source_slots"),
            lp.GlobalArg("mesh_vertex_indices", np.int32, "nelements, dim+1"),
            lp.GlobalArg("mesh_vertices_0", coord_dtype, "n_mesh_vertices"),
            lp.GlobalArg("mesh_vertices_1", coord_dtype, "n_mesh_vertices"),
            lp.GlobalArg("mesh_vertices_2", coord_dtype, "n_mesh_vertices"),
            lp.GlobalArg("source_points_0", coord_dtype, "nsources"),
            lp.GlobalArg("source_points_1", coord_dtype, "nsources"),
            lp.GlobalArg("source_points_2", coord_dtype, "nsources"),
            lp.GlobalArg("barycentric_0", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_1", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_2", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_3", coord_dtype, "n_source_slots"),
            lp.ValueArg("n_source_slots", np.int32),
            lp.ValueArg("nelements", np.int32),
            lp.ValueArg("nsources", np.int32),
            lp.ValueArg("n_mesh_vertices", np.int32),
            lp.ValueArg("dim", np.int32),
        ]

    else:
        raise NotImplementedError

    loopy_knl = lp.make_kernel(
        ["{ [ islot ] : 0 <= islot < n_source_slots }"],
        instructions,
        args,
        name="map_sources_to_template_simplex",
        lang_version=(2018, 2),
    )
    loopy_knl = lp.set_options(loopy_knl, return_dict=True)
    loopy_knl = lp.fix_parameters(loopy_knl, dim=int(dim))
    return loopy_knl


@lru_cache(maxsize=16)
def _build_from_meshmode_basis_tabulation_kernel(
    dim, coord_dtype_descr, value_dtype_descr
):
    coord_dtype = np.dtype(coord_dtype_descr)
    value_dtype = np.dtype(value_dtype_descr)

    if dim == 2:
        basis_val_expr = (
            "bernstein_coeffs[ibasis]"
            " * (barycentric_0[islot] ** bernstein_alpha_0[ibasis])"
            " * (barycentric_1[islot] ** bernstein_alpha_1[ibasis])"
            " * (barycentric_2[islot] ** bernstein_alpha_2[ibasis])"
        )
        args = [
            lp.GlobalArg("barycentric_0", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_1", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_2", coord_dtype, "n_source_slots"),
            lp.GlobalArg("bernstein_alpha_0", np.int32, "n_basis"),
            lp.GlobalArg("bernstein_alpha_1", np.int32, "n_basis"),
            lp.GlobalArg("bernstein_alpha_2", np.int32, "n_basis"),
            lp.GlobalArg("bernstein_coeffs", value_dtype, "n_basis"),
            lp.GlobalArg("basis_old_inverse", value_dtype, "n_basis, nunit_dofs"),
            lp.GlobalArg("rsplm", value_dtype, "n_source_slots, nunit_dofs"),
            lp.ValueArg("n_source_slots", np.int32),
            lp.ValueArg("n_basis", np.int32),
            lp.ValueArg("nunit_dofs", np.int32),
        ]
    elif dim == 3:
        basis_val_expr = (
            "bernstein_coeffs[ibasis]"
            " * (barycentric_0[islot] ** bernstein_alpha_0[ibasis])"
            " * (barycentric_1[islot] ** bernstein_alpha_1[ibasis])"
            " * (barycentric_2[islot] ** bernstein_alpha_2[ibasis])"
            " * (barycentric_3[islot] ** bernstein_alpha_3[ibasis])"
        )
        args = [
            lp.GlobalArg("barycentric_0", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_1", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_2", coord_dtype, "n_source_slots"),
            lp.GlobalArg("barycentric_3", coord_dtype, "n_source_slots"),
            lp.GlobalArg("bernstein_alpha_0", np.int32, "n_basis"),
            lp.GlobalArg("bernstein_alpha_1", np.int32, "n_basis"),
            lp.GlobalArg("bernstein_alpha_2", np.int32, "n_basis"),
            lp.GlobalArg("bernstein_alpha_3", np.int32, "n_basis"),
            lp.GlobalArg("bernstein_coeffs", value_dtype, "n_basis"),
            lp.GlobalArg("basis_old_inverse", value_dtype, "n_basis, nunit_dofs"),
            lp.GlobalArg("rsplm", value_dtype, "n_source_slots, nunit_dofs"),
            lp.ValueArg("n_source_slots", np.int32),
            lp.ValueArg("n_basis", np.int32),
            lp.ValueArg("nunit_dofs", np.int32),
        ]
    else:
        raise NotImplementedError

    loopy_knl = lp.make_kernel(
        [
            "{ [ islot, idof, ibasis ] : "
            "0 <= islot < n_source_slots and "
            "0 <= idof < nunit_dofs and "
            "0 <= ibasis < n_basis }",
        ],
        f"""
            for islot, idof
                rsplm[islot, idof] = sum(ibasis,
                    basis_old_inverse[ibasis, idof] * ({basis_val_expr}))
            end
        """,
        args,
        name="tabulate_from_meshmode_resampling",
        lang_version=(2018, 2),
    )

    loopy_knl = lp.set_options(loopy_knl, return_dict=True)
    return loopy_knl


@lru_cache(maxsize=16)
def _get_from_meshmode_simplex_bernstein_indices(dim, degree):
    dim = int(dim)
    degree = int(degree)

    if dim == 2:
        indices = [
            (a0, a1, degree - a0 - a1)
            for a0 in range(degree + 1)
            for a1 in range(degree + 1 - a0)
        ]
    elif dim == 3:
        indices = [
            (a0, a1, a2, degree - a0 - a1 - a2)
            for a0 in range(degree + 1)
            for a1 in range(degree + 1 - a0)
            for a2 in range(degree + 1 - a0 - a1)
        ]
    else:
        raise NotImplementedError

    return np.asarray(indices, dtype=np.int32)


def _get_from_meshmode_simplex_bernstein_coefficients(bernstein_indices, dtype):
    if len(bernstein_indices) == 0:
        return np.empty(0, dtype=dtype)

    degree = int(np.sum(bernstein_indices[0]))
    degree_factorial = math.factorial(degree)

    coeffs = []
    for alpha in bernstein_indices:
        denom = 1
        for exponent in alpha:
            denom *= math.factorial(int(exponent))
        coeffs.append(degree_factorial / denom)

    return np.asarray(coeffs, dtype=dtype)


def _map_to_simplex_barycentric(points, simplex_vertices):
    points = np.asarray(points)
    simplex_vertices = np.asarray(simplex_vertices)

    dim = simplex_vertices.shape[0]
    v0 = simplex_vertices[:, 0]
    edge_mat = simplex_vertices[:, 1:] - v0.reshape(dim, 1)
    rhs = points - v0.reshape(dim, 1)

    bary_rest = np.linalg.solve(edge_mat, rhs)
    bary0 = 1 - np.sum(bary_rest, axis=0, keepdims=True)
    return np.ascontiguousarray(np.vstack([bary0, bary_rest]))


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
        self.boxtree_actx = _get_boxtree_actx(context)
        self.tree = tree
        self.discr = discr

        from pyopencl.algorithm import KeyValueSorter

        self.key_value_sorter = KeyValueSorter(context)

        from boxtree.area_query import AreaQueryBuilder

        self.area_query_builder = AreaQueryBuilder(self.boxtree_actx)

    # {{{ kernel generation

    @memoize_method
    def codegen_get_dimension_specific_snippets(self):
        """Dimension-dependent code loopy instructions."""
        import sympy as sp

        axis_names = ["x", "y", "z"]
        axis_names = axis_names[: self.dim]

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

            code_get_simplex = """
                <> Ax = mesh_vertices_0[mesh_vertex_indices[iel, 0]]
                <> Ay = mesh_vertices_1[mesh_vertex_indices[iel, 0]]
                <> Bx = mesh_vertices_0[mesh_vertex_indices[iel, 1]]
                <> By = mesh_vertices_1[mesh_vertex_indices[iel, 1]]
                <> Cx = mesh_vertices_0[mesh_vertex_indices[iel, 2]]
                <> Cy = mesh_vertices_1[mesh_vertex_indices[iel, 2]]
                """
            code_get_point = """
                <> Px = source_points_0[source_id]
                <> Py = source_points_1[source_id]
                """
            # simplex measures
            code_s0 = get_simplex_measure(["P", "B", "C"])
            code_s1 = get_simplex_measure(["A", "P", "C"])
            code_s2 = get_simplex_measure(["A", "B", "P"])
            code_compute_simplex_measures = f"""
                <> s0 = {code_s0}
                <> s1 = {code_s1}
                <> s2 = {code_s2}
                """
            code_measures_have_common_sign = " and ".join(
                [
                    f"s{c1} * s{c2} >= {tol}"
                    for c1, c2 in itertools.combinations(["0", "1"], 2)
                ]
            )

            # }}} End 2d

        elif self.dim == 3:
            # {{{ 3d

            code_get_simplex = """
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
            code_get_point = """
                <> Px = source_points_0[source_id]
                <> Py = source_points_1[source_id]
                <> Pz = source_points_2[source_id]
                """
            # simplex measures
            code_s0 = get_simplex_measure(["P", "B", "C", "D"])
            code_s1 = get_simplex_measure(["A", "P", "C", "D"])
            code_s2 = get_simplex_measure(["A", "B", "P", "D"])
            code_s3 = get_simplex_measure(["A", "B", "C", "P"])
            code_compute_simplex_measures = f"""
                <> s0 = {code_s0}
                <> s1 = {code_s1}
                <> s2 = {code_s2}
                <> s3 = {code_s3}
                """
            code_measures_have_common_sign = " and ".join(
                [
                    f"s{c1} * s{c2} >= {tol}"
                    for c1, c2 in itertools.combinations(["0", "1", "2"], 2)
                ]
            )

            # }}} End 3d

        else:
            raise NotImplementedError()

        return {
            "code_get_simplex": code_get_simplex,
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
            [
                "{ [ iel ]: 0 <= iel < nelements }",
                "{ [ ineighbor ]: nearby_leaves_beg <= ineighbor < nearby_leaves_end }",
                "{ [ isrc ]: 0 <= isrc < n_box_sources }",
            ],
            [
                """
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

                        result[source_id] = (
                            iel if {code_measures_have_common_sign} else result[source_id]
                        )  {{atomic}}
                    end
                end
            end
            """.format(**snippets)
            ],
            [
                lp.ValueArg("nelements, dim, nboxes, nsources", np.int32),
                lp.GlobalArg("mesh_vertex_indices", np.int32, "nelements, dim+1"),
                lp.GlobalArg("box_source_starts", np.int32, "nboxes"),
                lp.GlobalArg("box_source_counts_cumul", np.int32, "nboxes"),
                lp.GlobalArg("leaves_near_ball_lists", np.int32, None),
                lp.GlobalArg("result", np.int32, "nsources", for_atomic=True),
                "...",
            ],
            name="build_sources_in_simplex_lookup",
            lang_version=(2018, 2),
        )

        logger.debug("done building elements-to-sources lookup kernel")
        return loopy_knl

    # }}} End kernel generation

    def compute_short_lists(self, actx, wait_for=None):
        """balls --> overlapping leaves"""
        if not isinstance(actx, MeshmodePyOpenCLArrayContext):
            if isinstance(actx, cl.CommandQueue):
                from warnings import warn

                warn(
                    "Command queue passed to the interpolator. "
                    "Supply an array context to enable proper caching."
                )
                actx = MeshmodePyOpenCLArrayContext(actx)
            else:
                raise ValueError

        mesh = self.discr.mesh
        if len(mesh.groups) > 1:
            raise NotImplementedError("Mixed elements not supported")
        melgrp = mesh.groups[0]
        ball_centers_host = (
            np.max(melgrp.nodes, axis=2) + np.min(melgrp.nodes, axis=2)
        ) / 2
        ball_radii_host = (
            np.max(np.max(melgrp.nodes, axis=2) - np.min(melgrp.nodes, axis=2), axis=0)
            / 2
        )

        ball_centers = obj_array_1d(
            [
                cl.array.to_device(actx.queue, center_coord_comp)
                for center_coord_comp in ball_centers_host
            ]
        )
        ball_radii = cl.array.to_device(actx.queue, ball_radii_host)

        area_query_result, evt = self.area_query_builder(
            self.boxtree_actx,
            self.tree,
            ball_centers,
            ball_radii,
            peer_lists=None,
            wait_for=wait_for,
        )
        return area_query_result, evt

    def __call__(self, actx, balls_to_leaves_lookup=None, wait_for=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`
        """
        if not isinstance(actx, MeshmodePyOpenCLArrayContext):
            if isinstance(actx, cl.CommandQueue):
                from warnings import warn

                warn(
                    "Command queue passed to the interpolator. "
                    "Supply an array context to enable proper caching."
                )
                actx = MeshmodePyOpenCLArrayContext(actx)
            else:
                raise ValueError

        slk_plog = ProcessLogger(logger, "element-to-source lookup: run area query")

        if balls_to_leaves_lookup is None:
            balls_to_leaves_lookup, evt = self.compute_short_lists(
                actx, wait_for=wait_for
            )
            wait_for = [evt]

        # -----------------------------------------------------------------
        # Refine the area query using point-in-simplex test

        logger.debug("element-to-source lookup: refine starts")

        element_lookup_kernel = self.get_simplex_lookup_kernel()
        element_lookup_executor = element_lookup_kernel.executor(actx.queue.context)

        vertices_dev = obj_array_1d(
            [
                cl.array.to_device(actx.queue, verts)
                for verts in self.discr.mesh.vertices
            ]
        )

        mesh_vertices_kwargs = {
            f"mesh_vertices_{iaxis}": vertices_dev[iaxis] for iaxis in range(self.dim)
        }

        source_points_kwargs = {
            f"source_points_{iaxis}": self.tree.sources[iaxis]
            for iaxis in range(self.dim)
        }

        evt, res = element_lookup_executor(
            actx.queue,
            dim=self.dim,
            nboxes=self.tree.nboxes,
            nelements=self.discr.mesh.nelements,
            nsources=self.tree.nsources,
            result=cl.array.zeros(actx.queue, self.tree.nsources, dtype=np.int32) - 1,
            mesh_vertex_indices=self.discr.mesh.groups[0].vertex_indices,
            box_source_starts=self.tree.box_source_starts,
            box_source_counts_cumul=self.tree.box_source_counts_cumul,
            leaves_near_ball_starts=balls_to_leaves_lookup.leaves_near_ball_starts,
            leaves_near_ball_lists=balls_to_leaves_lookup.leaves_near_ball_lists,
            wait_for=wait_for,
            **mesh_vertices_kwargs,
            **source_points_kwargs,
        )

        (source_to_element_lookup,) = res

        wait_for = [evt]

        # elements = source_to_element_lookup.get()
        # for idx in [362,  365,  874,  877, 1386, 1389, 1898, 1901])

        # -----------------------------------------------------------------
        # Invert the source-to-element lookup by a key-value sort

        logger.debug("element-to-source lookup: key-value sort")

        sources_in_element_starts, sources_in_element_lists, evt = (
            self.key_value_sorter(
                actx.queue,
                keys=source_to_element_lookup,
                values=cl.array.arange(
                    actx.queue, self.tree.nsources, dtype=self.tree.box_id_dtype
                ),
                nkeys=self.discr.mesh.nelements,
                starts_dtype=self.tree.box_id_dtype,
                wait_for=wait_for,
            )
        )

        slk_plog.done()

        return ElementsToSourcesLookup(
            tree=self.tree,
            discr=self.discr,
            sources_in_element_starts=sources_in_element_starts,
            sources_in_element_lists=sources_in_element_lists,
        ), evt


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
        self.boxtree_actx = _get_boxtree_actx(context)
        self.trav = trav
        self.discr = discr

        from boxtree.area_query import LeavesToBallsLookupBuilder

        self.leaves_to_balls_lookup_builder = LeavesToBallsLookupBuilder(
            self.boxtree_actx
        )

    def __call__(self, actx, tol=1e-12, wait_for=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`
        :tol: nodes close enough to the boundary will be treated as
            lying on the boundary, whose interpolated values are averaged.
        """
        if not isinstance(actx, MeshmodePyOpenCLArrayContext):
            if isinstance(actx, cl.CommandQueue):
                from warnings import warn

                warn(
                    "Command queue passed to the interpolator. "
                    "Supply an array context to enable proper caching."
                )
                actx = MeshmodePyOpenCLArrayContext(actx)
            else:
                raise ValueError

        nodes = flatten(actx.thaw(self.discr.nodes()), actx, leaf_class=DOFArray)
        nodes = obj_array_1d([coord.with_queue(actx.queue) for coord in nodes])
        lookup_tol = _compute_leaves_to_nodes_lookup_tol(self.trav.tree, tol)
        radii = _make_constant_array(actx.queue, nodes[0], lookup_tol)

        area_query, _ = self.leaves_to_balls_lookup_builder.area_query_builder(
            self.boxtree_actx,
            self.trav.tree,
            nodes,
            radii,
            wait_for=wait_for,
        )

        missing_count = _count_missing_nodes_from_leaf_starts(
            area_query.leaves_near_ball_starts,
            actx.queue,
        )
        if missing_count:
            raise RuntimeError(
                "leaves-to-nodes lookup failed to cover all nodes; "
                f"missing {missing_count} nodes. "
                "Ensure the tree bounding box encloses all discretization nodes "
                f"(lookup_tol={lookup_tol:.3e})."
            )

        nkeys = self.trav.tree.nboxes
        nballs_p_1 = len(area_query.leaves_near_ball_starts)

        starts_expander_knl = (
            self.leaves_to_balls_lookup_builder.get_starts_expander_kernel(
                self.trav.tree.box_id_dtype
            )
        )
        expanded_starts = self.boxtree_actx.np.zeros(
            len(area_query.leaves_near_ball_lists),
            self.trav.tree.box_id_dtype,
        )
        evt = starts_expander_knl(
            expanded_starts,
            area_query.leaves_near_ball_starts,
            nballs_p_1,
        )

        nodes_in_leaf_starts, nodes_in_leaf_lists, evt = (
            self.leaves_to_balls_lookup_builder.key_value_sorter(
                self.boxtree_actx.queue,
                area_query.leaves_near_ball_lists,
                expanded_starts,
                nkeys,
                starts_dtype=self.trav.tree.box_id_dtype,
                wait_for=[evt],
            )
        )

        return LeavesToNodesLookup(
            trav=self.trav,
            discr=self.discr,
            nodes_in_leaf_starts=nodes_in_leaf_starts,
            nodes_in_leaf_lists=nodes_in_leaf_lists,
        ), evt


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
    ivb = -iva @ disp_b
    return iva, ivb


# }}} End transform helper


# {{{ from meshmode interpolation


def interpolate_from_meshmode(actx, dof_vec, elements_to_sources_lookup, order="tree"):
    """Interpolate a DoF vector from :mod:`meshmode`.

    :arg dof_vec: a DoF vector representing a field in :mod:`meshmode`
        of shape ``(..., nnodes)``.
    :arg elements_to_sources_lookup: a :class:`ElementsToSourcesLookup`.
    :arg order: order of the output potential, either "tree" or "user".

    .. note:: This function currently supports meshes with just one element
        group. Also, the element group must be simplex-based.

    .. note:: Affine inverse mapping, basis tabulation, and local matrix-vector
        application are carried out by :mod:`loopy` kernels.
    """
    if not isinstance(dof_vec, cl.array.Array):
        raise TypeError("non-array passed to interpolator")

    if not isinstance(actx, MeshmodePyOpenCLArrayContext):
        if isinstance(actx, cl.CommandQueue):
            from warnings import warn

            warn(
                "Command queue passed to the interpolator. "
                "Supply an array context to enable proper caching."
            )
            actx = MeshmodePyOpenCLArrayContext(actx)
        else:
            raise ValueError

    assert len(elements_to_sources_lookup.discr.groups) == 1
    assert len(elements_to_sources_lookup.discr.mesh.groups) == 1
    degroup = elements_to_sources_lookup.discr.groups[0]
    megroup = elements_to_sources_lookup.discr.mesh.groups[0]

    if not degroup.is_affine:
        raise ValueError(
            "interpolation requires global-to-local map, "
            "which is only available for affinely mapped elements"
        )

    mesh = elements_to_sources_lookup.discr.mesh
    dim = elements_to_sources_lookup.discr.dim
    template_simplex = np.ascontiguousarray(mesh.groups[0].vertex_unit_coordinates().T)

    sources_in_element_starts = (
        elements_to_sources_lookup.sources_in_element_starts.get(actx.queue)
    )
    sources_in_element_lists = np.ascontiguousarray(
        elements_to_sources_lookup.sources_in_element_lists.get(actx.queue),
        dtype=np.int32,
    )

    tree = elements_to_sources_lookup.tree
    nsources = int(tree.nsources)

    nunit_dofs = int(degroup.unit_nodes.shape[1])
    value_dtype = np.dtype(np.result_type(dof_vec.dtype, np.float64))
    coord_dtype = np.dtype(tree.coord_dtype)

    source_element_indices = np.repeat(
        np.arange(degroup.nelements, dtype=np.int32),
        np.diff(sources_in_element_starts),
    )
    assert len(source_element_indices) == len(sources_in_element_lists)

    source_element_indices_dev = cl.array.to_device(actx.queue, source_element_indices)
    sources_in_element_lists_dev = cl.array.to_device(
        actx.queue, sources_in_element_lists
    )

    mesh_vertices_dev = obj_array_1d(
        [
            cl.array.to_device(
                actx.queue,
                np.ascontiguousarray(mesh.vertices[iaxis], dtype=coord_dtype),
            )
            for iaxis in range(dim)
        ]
    )
    n_source_slots = len(sources_in_element_lists)
    barycentric_dev = obj_array_1d(
        [
            cl.array.empty(actx.queue, n_source_slots, dtype=coord_dtype)
            for _ in range(dim + 1)
        ]
    )

    if n_source_slots > 0:
        map_knl = _build_from_meshmode_map_to_template_kernel(dim, coord_dtype.str)
        map_executor = map_knl.executor(actx.queue.context)

        map_kwargs = {
            "source_element_indices": source_element_indices_dev,
            "sources_in_element_lists": sources_in_element_lists_dev,
            "mesh_vertex_indices": np.ascontiguousarray(
                megroup.vertex_indices, dtype=np.int32
            ),
            "n_source_slots": np.int32(n_source_slots),
            "nelements": np.int32(degroup.nelements),
            "nsources": np.int32(nsources),
            "n_mesh_vertices": np.int32(mesh.vertices.shape[1]),
        }

        for iaxis in range(dim):
            map_kwargs[f"mesh_vertices_{iaxis}"] = mesh_vertices_dev[iaxis]
            map_kwargs[f"source_points_{iaxis}"] = tree.sources[iaxis]

        for iaxis in range(dim + 1):
            map_kwargs[f"barycentric_{iaxis}"] = barycentric_dev[iaxis]

        evt, map_res = map_executor(actx.queue, **map_kwargs)
        barycentric_dev = obj_array_1d(
            [map_res[f"barycentric_{iaxis}"] for iaxis in range(dim + 1)]
        )
        for iaxis in range(dim + 1):
            barycentric_dev[iaxis].add_event(evt)

    degree = int(degroup.order)
    rsplm = cl.array.empty(actx.queue, (n_source_slots, nunit_dofs), dtype=value_dtype)
    if n_source_slots > 0:
        bernstein_indices = _get_from_meshmode_simplex_bernstein_indices(dim, degree)
        if len(bernstein_indices) != nunit_dofs:
            raise ValueError(
                "unexpected simplex polynomial space size: "
                f"got {len(bernstein_indices)}, expected {nunit_dofs}"
            )

        unit_nodes = np.ascontiguousarray(degroup.unit_nodes, dtype=coord_dtype)
        unit_barycentric = _map_to_simplex_barycentric(unit_nodes, template_simplex)

        bernstein_coeffs_real = _get_from_meshmode_simplex_bernstein_coefficients(
            bernstein_indices, np.float64
        )
        bernstein_basis_old = np.ones(
            (len(bernstein_indices), nunit_dofs), dtype=np.float64
        )
        for iaxis in range(dim + 1):
            bernstein_basis_old *= (
                unit_barycentric[iaxis][None, :] ** bernstein_indices[:, iaxis][:, None]
            )
        bernstein_basis_old *= bernstein_coeffs_real[:, None]

        basis_old_inverse = np.ascontiguousarray(
            np.linalg.inv(bernstein_basis_old.T), dtype=value_dtype
        )
        bernstein_coeffs = np.ascontiguousarray(
            bernstein_coeffs_real, dtype=value_dtype
        )

        bernstein_alpha_dev = obj_array_1d(
            [
                cl.array.to_device(
                    actx.queue,
                    np.ascontiguousarray(bernstein_indices[:, iaxis], dtype=np.int32),
                )
                for iaxis in range(dim + 1)
            ]
        )
        bernstein_coeffs_dev = cl.array.to_device(actx.queue, bernstein_coeffs)
        basis_old_inverse_dev = cl.array.to_device(actx.queue, basis_old_inverse)

        basis_knl = _build_from_meshmode_basis_tabulation_kernel(
            dim, coord_dtype.str, value_dtype.str
        )
        basis_executor = basis_knl.executor(actx.queue.context)

        basis_kwargs = {
            "bernstein_coeffs": bernstein_coeffs_dev,
            "basis_old_inverse": basis_old_inverse_dev,
            "rsplm": rsplm,
            "n_source_slots": np.int32(n_source_slots),
            "n_basis": np.int32(len(bernstein_indices)),
            "nunit_dofs": np.int32(nunit_dofs),
        }

        for iaxis in range(dim + 1):
            basis_kwargs[f"barycentric_{iaxis}"] = barycentric_dev[iaxis]
            basis_kwargs[f"bernstein_alpha_{iaxis}"] = bernstein_alpha_dev[iaxis]

        evt, basis_res = basis_executor(actx.queue, **basis_kwargs)
        rsplm = basis_res["rsplm"]
        rsplm.add_event(evt)

    from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array

    dof_vec_work = dof_vec
    if np.dtype(dof_vec.dtype) != value_dtype:
        dof_vec_work = dof_vec.astype(value_dtype)

    dof_vec_flat = to_tagged_cl_array(dof_vec_work)
    sym_shape = dof_vec_flat.shape[:-1]
    nvecs = 1 if len(sym_shape) == 0 else int(np.prod(sym_shape, dtype=np.int64))
    dof_by_el = dof_vec_flat.reshape((nvecs, degroup.nelements, nunit_dofs))

    source_vec = cl.array.zeros(actx.queue, (nvecs, nsources), dtype=value_dtype)

    if n_source_slots > 0:
        apply_knl = _build_from_meshmode_resampling_kernel(value_dtype.str)
        apply_executor = apply_knl.executor(actx.queue.context)

        evt, apply_res = apply_executor(
            actx.queue,
            rsplm=rsplm,
            dof_by_el=dof_by_el,
            source_element_indices=source_element_indices_dev,
            sources_in_element_lists=sources_in_element_lists_dev,
            source_vec=source_vec,
            nvecs=np.int32(nvecs),
            nelements=np.int32(degroup.nelements),
            n_source_slots=np.int32(n_source_slots),
            nunit_dofs=np.int32(nunit_dofs),
            nsources=np.int32(nsources),
        )
        source_vec = apply_res["source_vec"]
        source_vec.add_event(evt)

    if len(sym_shape) == 0:
        source_vec = source_vec[0]
    else:
        source_vec = source_vec.reshape(sym_shape + (nsources,))

    if order == "tree":
        pass  # no need to do anything
    elif order == "user":
        if len(sym_shape) == 0:
            source_vec = source_vec[tree.sorted_target_ids]
        else:
            source_vec_flat = source_vec.reshape((nvecs, nsources))
            source_vec_user = cl.array.empty(
                actx.queue, (nvecs, nsources), dtype=value_dtype
            )
            for ivec in range(nvecs):
                source_vec_user[ivec] = source_vec_flat[ivec][tree.sorted_target_ids]
            source_vec = source_vec_user.reshape(sym_shape + (nsources,))
    else:
        raise ValueError(f"order must be 'tree' or 'user' (got {order}).")

    return source_vec


# }}} End from meshmode interpolation


# {{{ to meshmode interpolation


def interpolate_to_meshmode(actx, potential, leaves_to_nodes_lookup, order="tree"):
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

    if not isinstance(actx, MeshmodePyOpenCLArrayContext):
        if isinstance(actx, cl.CommandQueue):
            from warnings import warn

            warn(
                "Command queue passed to the interpolator. "
                "Supply an array context to enable proper caching."
            )
            actx = MeshmodePyOpenCLArrayContext(actx)
        else:
            raise ValueError

    target_points = flatten(
        actx.thaw(leaves_to_nodes_lookup.discr.nodes()),
        actx,
        leaf_class=DOFArray,
    )

    traversal = leaves_to_nodes_lookup.trav
    tree = leaves_to_nodes_lookup.trav.tree

    dim = tree.dimensions

    from volumential.volume_fmm import interpolate_volume_potential

    # infer q_order from tree
    pts_per_box = tree.ntargets // traversal.ntarget_boxes
    assert pts_per_box * traversal.ntarget_boxes == tree.ntargets

    # allow for +/- 0.25 floating point error
    q_order = int(pts_per_box ** (1 / dim) + 0.25)
    assert q_order**dim == pts_per_box

    interp_p = interpolate_volume_potential(
        target_points=target_points,
        traversal=traversal,
        wrangler=None,
        potential=potential,
        potential_in_tree_order=potential_in_tree_order,
        dim=dim,
        tree=tree,
        queue=actx.queue,
        q_order=q_order,
        dtype=potential.dtype,
        lbl_lookup=None,
        balls_near_box_starts=leaves_to_nodes_lookup.nodes_in_leaf_starts,
        balls_near_box_lists=leaves_to_nodes_lookup.nodes_in_leaf_lists,
    )

    return interp_p


# }}} End to meshmode interpolation
