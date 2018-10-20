__copyright__ = "Copyright (C) 2017 - 2018 Andreas Klockner, Xiaoyu Wei"

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

dim = 2

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
)

from pytential import bind, sym, norm  # noqa
from pytential.target import PointsTarget

from pytools.obj_array import make_obj_array
import pytential.symbolic.primitives as p

from sumpy.kernel import FactorizedBiharmonicKernel, YukawaKernel
from sumpy.kernel import AxisTargetDerivative, LaplacianTargetDerivative
from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion

import pymbolic
import pymbolic as pmbl

from volumential.expansion_wrangler_interface import ExpansionWranglerCodeContainer
from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler

import loopy as lp
from functools import partial

from tabulate import tabulate

import warnings
from loopy import LoopyWarning

warnings.filterwarnings("ignore", category=LoopyWarning)

from enum import Enum


class Geometry(Enum):
    RegularRectangle = 1
    Circle = 2


import volumential.meshgen as mg
from volumential.table_manager import NearFieldInteractionTableManager
from volumential.volume_fmm import drive_volume_fmm

cl_ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(cl_ctx)
if "Portable Computing Language" not in queue.device.__str__():
    from warnings import warn

    warn("Not running on POCL.")

# clean up the mess
def clean_file(filename):
    import os

    try:
        os.remove(filename)
    except OSError:
        pass


verbose = False
n_iters = 2
output_interval = 1

# {{{ all kinds of orders
h = 0.003
mesh_order = 2  # for gmsh

bdry_quad_order = 4
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 2 * bdry_quad_order

fmm_order = 5
fmm_order_ykw = 5

visual_order = 2

adaptive_mesh = False
n_levels = 8
refined_n_cells = 10000
n_refinement_loops = 100
rratio_top = 1
rratio_bot = 0

vol_quad_order = 4
# }}}

# {{{ mesh generation
nelements = 20

from enum import Enum


class Geometry(Enum):
    RegularRectangle = 1
    Circle = 2


shape = Geometry.Circle
# shape = Geometry.RegularRectangle
# }}}

# {{{ physical parameters
# For energy stability,
# 1. s >= (3M^2-1)/2, where M=max(|phi|)
#    e.g., when M=1, need S>=1
# 2. alpha_tild >= N/2, where N=|sqrt(2)/3 * (pi/2)^2 * cos(theta_y)|
#    e.g., when theta_y=0, N=1.163144 is the maximum possible value of N
# alpha_tild = 0 in the notes.
s = 1.5
alpha_tild = 0

epsilon = 0.01
delta_t = 0.05
final_t = delta_t * 1
theta_y = 60.0 / 180.0 * np.pi

b = s / (epsilon ** 2)
c = 1.0 / (epsilon * delta_t)

# boundary convex splitting
# bs = alpha_tild / epsilon
bs = 0
# }}}

# {{{ bounding box and pre-computed table
box_a = -0.25
box_b = 0.25
root_table_source_extent = 0.5

table_filename = "nft_ch.hdf5"
# }}} End box tree and pre-computed table

# {{{ PDE's rhs formulae

phi = pmbl.var("phi")
phi_x = pmbl.var("phi_x")
phi_y = pmbl.var("phi_y")
lap_phi = pmbl.var("lap_phi")
lap_phi3 = 6 * phi * (phi_x ** 2 + phi_y ** 2) + 3 * phi ** 2 * lap_phi


# specific surface energy
def gamma(z):
    import math
    from pymbolic.functions import sin

    if 0:
        # NOTE: a different sign (as in [Gao, Wang, 2014]), correspons to
        # a different theta_y ( vs pi - theta_y )
        gamma_coef = math.sqrt(2) / 3 * math.cos(theta_y)

    else:
        # NOTE: as in the notes
        gamma_coef = -math.sqrt(2) / 3 * math.cos(theta_y)

    pi_val = math.pi
    return gamma_coef * sin(pi_val / 2 * z)


gamma_expr = gamma(phi)
b1_expr = (c - bs) * phi + 1 / epsilon * pmbl.differentiate(gamma(phi), "phi")

f0_expr = c * phi
f1_expr = c * phi - (1 + s) / epsilon ** 2 * lap_phi + 1 / epsilon ** 2 * lap_phi3
f2_expr = (phi * phi * phi - (1 + s) * phi) / epsilon ** 2
f2_x_expr = pmbl.differentiate(f2_expr, "phi") * phi_x
f2_y_expr = pmbl.differentiate(f2_expr, "phi") * phi_y

# f2_x_expr = (3 * phi**2 * phi_x - (1 + s) * phi_x) / epsilon**2
# f2_y_expr = (3 * phi**2 * phi_y - (1 + s) * phi_y) / epsilon**2

# }}} End PDE rhs formulae

# {{{ initials

x = pmbl.var("x")
y = pmbl.var("y")
r = (x ** 2 + y ** 2) ** (1 / 2)

# equilibrium (asymptotically) phi = tanh(x / sqrt (2 * epsilon))
def tanh(z):
    # from pymbolic.functions import exp
    from pymbolic.functions import exp

    return (exp(z) - exp(-z)) / (exp(z) + exp(-z))


def ellipse(x, y, axa, axb, thickness=0.1):
    import math

    rad = x ** 2 / axa ** 2 + y ** 2 / axb ** 2 - 1
    return tanh(rad / math.sqrt(2 * thickness))


import math

if 0:
    print("Using interface (with CL) initials")
    phi_init = tanh(10 * x / math.sqrt(2 * epsilon))
    init_filename = "ch_init_contact.npz"
elif 1:
    print("Using interface (without CL) initials")
    phi_init = tanh(10 * (r - 0.1) / math.sqrt(2 * epsilon))
    init_filename = "ch_init_circle.npz"
else:
    # constant
    print("Using constant initials")
    # phi_init = x * 0 + 1
    phi_init = x * 0 - 1
    init_filename = "ch_init_pure.npz"


phi_x_init = pmbl.differentiate(phi_init, "x")
phi_y_init = pmbl.differentiate(phi_init, "y")
laphi_init = pmbl.differentiate(
    pmbl.differentiate(phi_init, "x"), "x"
) + pmbl.differentiate(pmbl.differentiate(phi_init, "y"), "y")


def expr_to_init(expr):
    return pmbl.substitute(
        expr,
        {
            "phi": phi_init,
            "phi_x": phi_x_init,
            "phi_y": phi_y_init,
            "lap_phi": laphi_init,
        },
    )


def math_func_mangler(target, name, arg_dtypes):
    if len(arg_dtypes) == 1 and isinstance(name, pymbolic.primitives.Lookup):
        arg_dtype, = arg_dtypes

        fname = name.name
        if not (
            isinstance(name.aggregate, pymbolic.primitives.Variable)
            and name.aggregate.name == "math"
        ):
            raise RuntimeError("unexpected aggregate '%s'" % str(name.aggregate))

        if arg_dtype.is_complex():
            if arg_dtype.numpy_dtype == np.complex64:
                tpname = "cfloat"
            elif arg_dtype.numpy_dtype == np.complex128:
                tpname = "cdouble"
            else:
                raise RuntimeError("unexpected complex type '%s'" % arg_dtype)

            return lp.CallMangleInfo(
                target_name="%s_%s" % (tpname, fname),
                result_dtypes=(arg_dtype,),
                arg_dtypes=(arg_dtype,),
            )

        else:
            return lp.CallMangleInfo(
                target_name="%s" % fname,
                result_dtypes=(arg_dtype,),
                arg_dtypes=(arg_dtype,),
            )

    return None


# Initial values for f1, f2, b1 etc.
from volumential.tools import ScalarFieldExpressionEvaluation


def get_init_func(expr):
    import math

    init_expr = expr_to_init(expr)
    return ScalarFieldExpressionEvaluation(
        dim=2,
        expression=init_expr,
        variables=[x, y],
        function_manglers=[math_func_mangler],
    )


initial_phi = get_init_func(phi)
initial_phi_x = get_init_func(phi_x)
initial_phi_y = get_init_func(phi_y)

f1_func = get_init_func(f1_expr)
# f1_x_func = get_init_func(f1_x_expr)
# f1_y_func = get_init_func(f1_y_expr)

f2_func = get_init_func(f2_expr)
f2_x_func = get_init_func(f2_x_expr)
f2_y_func = get_init_func(f2_y_expr)

b1_func = get_init_func(b1_expr)

# }}} End initials


def full_coverage(x):
    return 1


def circular_coverage(r, x):
    rad = np.linalg.norm(x)
    if rad > r:
        return 0
    else:
        return 1


def rectangular_coverage(a, b, c, d, x):
    if x[0] > a and x[0] < b and x[1] > c and x[1] < d:
        return 1
    else:
        return 0


geometry_mask = full_coverage


def main():
    from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_1
    import logging

    if verbose:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("volume mesh generation")

    # {{{ volume mesh generation

    # Cache the mesh for better speed & reproducibility
    volume_mesh_filename = "volume_mesh_" + str(shape) + ".meshmode"

    import os

    if os.path.isfile(volume_mesh_filename) and 0:
        print(
            "Loading cached volume mesh in",
            volume_mesh_filename,
            "(make sure it is the desired one)",
        )

        import pickle

        with open(volume_mesh_filename, "rb") as handle:
            unserialized_mesh_data = pickle.load(handle)

        from meshmode.mesh import Mesh, SimplexElementGroup
        from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL

        groups = [
            SimplexElementGroup(
                order=g["order"],
                vertex_indices=np.array(g["vertex_indices"], dtype=np.int32),
                nodes=np.array(g["nodes"]),
                # element_nr_base=g["element_nr_base"],
                node_nr_base=g["node_nr_base"],
                unit_nodes=np.array(g["unit_nodes"]),
                dim=g["dim"],
            )
            for g in unserialized_mesh_data["groups"]
        ]

        mesh = Mesh(
            vertices=np.array(unserialized_mesh_data["vertices"]),
            groups=groups,
            skip_tests=False,
            node_vertex_consistency_tolerance=None,
            nodal_adjacency=None,
            facial_adjacency_groups=None,
            boundary_tags=[BTAG_ALL, BTAG_REALLY_ALL],
            vertex_id_dtype=np.int32,
            element_id_dtype=np.int32,
        )

    else:

        if shape == Geometry.RegularRectangle:
            from meshmode.mesh.generation import generate_regular_rect_mesh

            ext = 1.0
            mesh = generate_regular_rect_mesh(
                a=(-ext / 2.0, -ext / 2.0),
                b=(ext / 2.0, ext / 2.0),
                n=(int(ext / h), int(ext / h)),
            )

        elif shape == Geometry.Circle:
            print("running on a circle")
            from meshmode.mesh.io import generate_gmsh, FileSource

            mesh = generate_gmsh(
                FileSource("circle.step"),
                2,
                order=mesh_order,
                force_ambient_dim=2,
                other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h],
            )

        else:
            RuntimeError("unsupported geometry")

        print("Saving cached volume mesh in", volume_mesh_filename)
        import pickle
        from meshmode.mesh.io import to_json

        with open(volume_mesh_filename, "wb") as handle:
            pickle.dump(to_json(mesh), handle, protocol=pickle.HIGHEST_PROTOCOL)

    if shape == Geometry.RegularRectangle:
        geometry_mask = partial(
            rectangular_coverage, -ext / 2, ext / 2, -ext / 2, ext / 2
        )
    elif shape == Geometry.Circle:
        # mask can be larger than actual domain
        geometry_mask = partial(circular_coverage, 0.3)
    else:
        RuntimeError("unsupported geometry")

    logger.info("%d elements" % mesh.nelements)

    # }}}

    print("discretization and connections")

    # {{{ discretization and connections

    vol_discr = Discretization(
        cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(vol_quad_order)
    )

    from meshmode.mesh import BTAG_ALL
    from meshmode.discretization.connection import make_face_restriction

    pre_density_connection = make_face_restriction(
        vol_discr, InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order), BTAG_ALL
    )
    pre_density_discr = pre_density_connection.to_discr

    from pytential.qbx import (
        QBXLayerPotentialSource,
        QBXTargetAssociationFailedException,
    )

    qbx, _ = QBXLayerPotentialSource(
        pre_density_discr,
        fine_order=bdry_ovsmp_quad_order,
        qbx_order=qbx_order,
        fmm_order=fmm_order,
    ).with_refinement()

    density_discr = qbx.density_discr

    # composition of connetions
    # vol_discr --> pre_density_discr --> density_discr
    # via ChainedDiscretizationConnection

    #    from meshmode.mesh.generation import ellipse, make_curve_mesh
    #    from functools import partial
    #
    #    mesh = make_curve_mesh(
    #                partial(ellipse, 2),
    #                np.linspace(0, 1, nelements+1),
    #                mesh_order)
    #
    #    pre_density_discr = Discretization(
    #            cl_ctx, mesh,
    #            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))
    #
    #    from pytential.qbx import (
    #            QBXLayerPotentialSource, QBXTargetAssociationFailedException)
    #    qbx, _ = QBXLayerPotentialSource(
    #            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
    #            fmm_order=fmm_order,
    #            expansion_disks_in_tree_have_extent=True,
    #            ).with_refinement()
    #    density_discr = qbx.density_discr

    # boundary nodes
    bdry_nodes_x = density_discr.nodes()[0].with_queue(queue).get()
    bdry_nodes_y = density_discr.nodes()[1].with_queue(queue).get()
    bdry_nodes = make_obj_array(  # get() first for CL compatibility issues
        [
            cl.array.to_device(queue, bdry_nodes_x),
            cl.array.to_device(queue, bdry_nodes_y),
        ]
    )

    print("density discr has", len(bdry_nodes_y), "nodes.")

    # boundary (unit) normals
    bdry_normals = bind(density_discr, sym.normal(dim))(queue).as_vector(dtype=object)
    bdry_normal_lens = np.sqrt(sum(vec_comp.get() ** 2 for vec_comp in bdry_normals))
    assert np.linalg.norm(bdry_normal_lens - 1) < 1e-12

    # gradient to normal derivative
    bdnormal_x = bdry_normals[0]
    bdnormal_y = bdry_normals[1]

    def grad_to_normal(der_x, der_y):
        return der_x * bdnormal_x + der_y * bdnormal_y

    # }}}

    print("quad points for box tree")

    # {{{ quad points for box tree

    if not adaptive_mesh:
        box_mesh = mg.MeshGen2D(vol_quad_order, n_levels, a=box_a, b=box_b)
        box_mesh.print_info()
        q_points = box_mesh.get_q_points()
        q_weights = box_mesh.get_q_weights()
        q_radii = None

    else:
        box_mesh = mg.MeshGen2D(vol_quad_order, n_levels, a=box_a, b=box_b)
        iloop = 0
        while box_mesh.n_active_cells() < refined_n_cells:
            iloop += 1
            crtr = np.array(
                [
                    np.abs(geometry_mask(c) * m)
                    for (c, m) in zip(
                        box_mesh.get_cell_centers(), box_mesh.get_cell_measures()
                    )
                ]
            )
            assert (np.max(np.abs(crtr))) > 0
            box_mesh.update_mesh(crtr, rratio_top, rratio_bot)
            if iloop > n_refinement_loops:
                print("Max number of refinement loops reached.")
                break

        box_mesh.print_info()
        q_points = box_mesh.get_q_points()
        q_weights = box_mesh.get_q_weights()
        q_radii = None

    try:
        box_mesh.generate_gmsh("box_grid.msh")
    except:
        pass

    legacy_msh_file = True
    if legacy_msh_file:
        import os

        os.system("gmsh box_grid.msh convert_grid -")

    assert len(q_points) == len(q_weights)
    assert q_points.shape[1] == dim

    q_points_org = q_points
    q_points_org2 = np.ascontiguousarray(np.transpose(q_points))

    q_points = make_obj_array(
        [cl.array.to_device(queue, q_points_org2[i]) for i in range(dim)]
    )

    # 1 inside, 0 outside
    mask_tmp = np.array([geometry_mask(qp) for qp in q_points_org])
    q_point_masks = cl.array.to_device(queue, mask_tmp)

    int_q_point_indicies = cl.array.to_device(queue, np.nonzero(mask_tmp)[0])
    ext_q_point_indicies = cl.array.to_device(queue, np.nonzero(1 - mask_tmp)[0])
    assert len(int_q_point_indicies) + len(ext_q_point_indicies) == len(q_points_org)

    ext_q_points = make_obj_array(
        [cl.array.to_device(queue, q_points_org2[i][mask_tmp == 0]) for i in range(dim)]
    )
    for i in range(dim):
        assert len(ext_q_points[i]) == len(ext_q_point_indicies)

    q_weights = cl.array.to_device(queue, q_weights)

    # }}} End quad points for box tree

    print("box discretization")

    # {{{ box discretization

    # over the bounding box
    box_discr = PointsTarget(q_points, normals=None)

    # }}} End box discretization

    print("tree and trav")

    # {{{ tree and trav
    from boxtree.tools import AXIS_NAMES

    axis_names = AXIS_NAMES[:dim]

    from pytools import single_valued

    coord_dtype = single_valued(coord.dtype for coord in q_points)
    from boxtree.bounding_box import make_bounding_box_dtype

    bbox_type, _ = make_bounding_box_dtype(cl_ctx.devices[0], dim, coord_dtype)

    bbox = np.empty(1, bbox_type)
    for ax in axis_names:
        bbox["min_" + ax] = box_a
        bbox["max_" + ax] = box_b

    # tune max_particles_in_box to reconstruct the mesh
    from boxtree import TreeBuilder

    tb = TreeBuilder(cl_ctx)
    tree, _ = tb(
        queue,
        particles=q_points,
        targets=q_points,
        bbox=bbox,
        max_particles_in_box=vol_quad_order ** 2 * 4 - 1,
        kind="adaptive-level-restricted",
    )

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(cl_ctx)
    trav, _ = tg(queue, tree)

    if 0:
        import matplotlib.pyplot as plt

        from boxtree.visualization import TreePlotter

        plotter = TreePlotter(tree.get(queue=queue))
        plotter.draw_tree(fill=False, edgecolor="black")
        # plotter.draw_box_numbers()
        plotter.set_bounding_box()
        plt.gca().set_aspect("equal")

        plt.draw()
        plt.savefig("ch_tree.png")
        print("Tree picture saved")

    # }}} End tree and trav

    print("The tree has", tree.nlevels, "levels")

    # get an idea of the finest grid size
    delta_x = 2 * (box_b - box_a) * (0.5 ** tree.nlevels)
    print("delta_x =", delta_x)

    print("near field table")

    # {{{ near field table
    if False:
        clean_file(table_filename)
        clean_file("tmp_" + table_filename)

    tm = NearFieldInteractionTableManager(
        table_filename, root_extent=root_table_source_extent
    )

    import loopy as lp
    from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator

    chop = CahnHilliardOperator(b=b, c=c)

    n_brick_levels = 15
    brick_quad_order = 50

    extra_kernel_kwargs = {"lam1": chop.lambdas[0], "lam2": chop.lambdas[1]}
    extra_kernel_kwarg_types = (
        lp.ValueArg("lam1", np.float64),
        lp.ValueArg("lam2", np.float64),
    )

    # box size should be integral times of largest table size
    # TODO: in principle the converse should also work
    assert (
        abs(
            int((box_b - box_a) / root_table_source_extent) * root_table_source_extent
            - (box_b - box_a)
        )
        < 1e-15
    ) or (
        abs(
            int(root_table_source_extent / (box_b - box_a)) * (box_b - box_a)
            - root_table_source_extent
        )
        < 1e-15
    )

    print("getting tables")
    near_field_table = {}

    # pass a debug parameter list to enable more checks
    table_debug = {"strict_loading": True}

    # {{{ cahn-hilliard kernel
    nftable = []
    for l in range(0, tree.nlevels):
        print("Getting table at level", l)
        tb, _ = tm.get_table(
            dim,
            "Cahn-Hilliard",
            vol_quad_order,
            source_box_level=l,
            compute_method="DrosteSum",
            adaptive_level=False,
            n_levels=n_brick_levels,
            n_brick_quad_points=brick_quad_order,
            b=b,
            c=c,
            extra_kernel_kwargs=extra_kernel_kwargs,
            extra_kernel_kwarg_types=extra_kernel_kwarg_types,
            # debug=table_debug
        )
        nftable.append(tb)
    near_field_table[nftable[0].integral_knl.__repr__()] = nftable

    nftable_dx = []
    for l in range(0, tree.nlevels):
        print("Getting table dx at level", l)
        tb, _ = tm.get_table(
            dim,
            "Cahn-Hilliard-Dx",
            vol_quad_order,
            source_box_level=l,
            compute_method="DrosteSum",
            adaptive_level=False,
            n_levels=n_brick_levels,
            n_brick_quad_points=brick_quad_order,
            b=b,
            c=c,
            extra_kernel_kwargs=extra_kernel_kwargs,
            extra_kernel_kwarg_types=extra_kernel_kwarg_types,
        )
        nftable_dx.append(tb)
    near_field_table[nftable_dx[0].integral_knl.__repr__()] = nftable_dx

    nftable_dy = []
    for l in range(0, tree.nlevels):
        print("Getting table dy at level", l)
        tb, _ = tm.get_table(
            dim,
            "Cahn-Hilliard-Dy",
            vol_quad_order,
            source_box_level=l,
            compute_method="DrosteSum",
            adaptive_level=False,
            n_levels=n_brick_levels,
            n_brick_quad_points=brick_quad_order,
            b=b,
            c=c,
            extra_kernel_kwargs=extra_kernel_kwargs,
            extra_kernel_kwarg_types=extra_kernel_kwarg_types,
        )
        nftable_dy.append(tb)
    near_field_table[nftable_dy[0].integral_knl.__repr__()] = nftable_dy

    if 0:
        # laplacian and its gradients
        nftable2 = []
        for l in range(0, tree.nlevels):
            print("Getting table 2 at level", l)
            tb, _ = tm.get_table(
                dim,
                "Cahn-Hilliard-Laplacian",
                vol_quad_order,
                source_box_level=l,
                compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=b,
                c=c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types,
            )
            nftable2.append(tb)
        near_field_table[nftable2[0].integral_knl.__repr__()] = nftable2
        nftable2_dx = []

        for l in range(0, tree.nlevels):
            print("Getting table 2_dx at level", l)
            tb, _ = tm.get_table(
                dim,
                "Cahn-Hilliard-Laplacian-Dx",
                vol_quad_order,
                source_box_level=l,
                compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=b,
                c=c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types,
            )
            nftable2_dx.append(tb)
        near_field_table[nftable2_dx[0].integral_knl.__repr__()] = nftable2_dx

        nftable2_dy = []
        for l in range(0, tree.nlevels):
            print("Getting table 2_dy at level", l)
            tb, _ = tm.get_table(
                dim,
                "Cahn-Hilliard-Laplacian-Dy",
                vol_quad_order,
                source_box_level=l,
                compute_method="DrosteSum",
                adaptive_level=False,
                n_levels=n_brick_levels,
                n_brick_quad_points=brick_quad_order,
                b=b,
                c=c,
                extra_kernel_kwargs=extra_kernel_kwargs,
                extra_kernel_kwarg_types=extra_kernel_kwarg_types,
            )
            nftable2_dy.append(tb)
        near_field_table[nftable2_dy[0].integral_knl.__repr__()] = nftable2_dy

    # }}} End cahn-hilliard kernel

    # {{{ yukawa kernel G1

    yukawa_extra_kernel_kwargs = {"lam": chop.lambdas[0]}
    yukawa_extra_kernel_kwarg_types = (lp.ValueArg("lam", np.float64),)

    nftable_ykw = []
    for l in range(0, tree.nlevels):
        print("Getting table yukawa at level", l)
        tb, _ = tm.get_table(
            dim,
            "Yukawa",
            vol_quad_order,
            source_box_level=l,
            compute_method="DrosteSum",
            adaptive_level=False,
            n_levels=n_brick_levels,
            n_brick_quad_points=brick_quad_order,
            lam=yukawa_extra_kernel_kwargs["lam"],
            extra_kernel_kwargs=yukawa_extra_kernel_kwargs,
            extra_kernel_kwarg_types=yukawa_extra_kernel_kwarg_types,
        )
        nftable_ykw.append(tb)
    near_field_table[nftable_ykw[0].integral_knl.__repr__()] = nftable_ykw

    nftable_ykw_dx = []
    for l in range(0, tree.nlevels):
        print("Getting table yukawa-dx at level", l)
        tb, _ = tm.get_table(
            dim,
            "Yukawa-Dx",
            vol_quad_order,
            source_box_level=l,
            compute_method="DrosteSum",
            adaptive_level=False,
            n_levels=n_brick_levels,
            n_brick_quad_points=brick_quad_order,
            lam=yukawa_extra_kernel_kwargs["lam"],
            extra_kernel_kwargs=yukawa_extra_kernel_kwargs,
            extra_kernel_kwarg_types=yukawa_extra_kernel_kwarg_types,
        )
        nftable_ykw_dx.append(tb)
    near_field_table[nftable_ykw_dx[0].integral_knl.__repr__()] = nftable_ykw_dx

    nftable_ykw_dy = []
    for l in range(0, tree.nlevels):
        print("Getting table yukawa-dy at level", l)
        tb, _ = tm.get_table(
            dim,
            "Yukawa-Dy",
            vol_quad_order,
            source_box_level=l,
            compute_method="DrosteSum",
            adaptive_level=False,
            n_levels=n_brick_levels,
            n_brick_quad_points=brick_quad_order,
            lam=yukawa_extra_kernel_kwargs["lam"],
            extra_kernel_kwargs=yukawa_extra_kernel_kwargs,
            extra_kernel_kwarg_types=yukawa_extra_kernel_kwarg_types,
        )
        nftable_ykw_dy.append(tb)
    near_field_table[nftable_ykw_dy[0].integral_knl.__repr__()] = nftable_ykw_dy

    # }}} End  yukawa kernel

    # print("Using table list of length", len(nftable))
    # }}} End near field table

    print("sumpy kernel expansion")

    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    exclude_self = True
    dtype = np.complex128

    if exclude_self:
        target_to_source = np.arange(tree.ntargets, dtype=np.int32)
        self_extra_kwargs = {"target_to_source": target_to_source}
    else:
        self_extra_kwargs = {}

    # {{{ sumpy kernel expansion (cahn-hilliard)

    sumpy_knl = FactorizedBiharmonicKernel(dim)
    sumpy_knl_dx = AxisTargetDerivative(0, sumpy_knl)
    sumpy_knl_dy = AxisTargetDerivative(1, sumpy_knl)

    sumpy_knl2 = LaplacianTargetDerivative(sumpy_knl)
    sumpy_knl2_dx = AxisTargetDerivative(0, sumpy_knl2)
    sumpy_knl2_dy = AxisTargetDerivative(1, sumpy_knl2)

    ch_out_knls = [sumpy_knl, sumpy_knl_dx, sumpy_knl_dy]
    # sumpy_knl2, sumpy_knl2_dx, sumpy_knl2_dy]

    wcc = ExpansionWranglerCodeContainer(
        cl_ctx,
        partial(mpole_expn_class, sumpy_knl),
        partial(local_expn_class, sumpy_knl),
        ch_out_knls,
        exclude_self=exclude_self,
    )

    wrangler = FPNDExpansionWrangler(
        code_container=wcc,
        queue=queue,
        tree=tree,
        near_field_table=near_field_table,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=vol_quad_order,
        self_extra_kwargs=self_extra_kwargs,
        kernel_extra_kwargs=extra_kernel_kwargs,
    )

    # }}} End sumpy kernel expansion

    # {{{ sumpy kernel expansion (yukawa)

    sumpy_knl_ykw = YukawaKernel(dim)
    sumpy_knl_ykw_dx = AxisTargetDerivative(0, sumpy_knl_ykw)
    sumpy_knl_ykw_dy = AxisTargetDerivative(1, sumpy_knl_ykw)

    ykw_out_knls = [sumpy_knl_ykw, sumpy_knl_ykw_dx, sumpy_knl_ykw_dy]

    wcc_ykw = ExpansionWranglerCodeContainer(
        cl_ctx,
        partial(mpole_expn_class, sumpy_knl_ykw),
        partial(local_expn_class, sumpy_knl_ykw),
        ykw_out_knls,
        exclude_self=exclude_self,
    )

    wrangler_ykw = FPNDExpansionWrangler(
        code_container=wcc_ykw,
        queue=queue,
        tree=tree,
        near_field_table=near_field_table,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order_ykw,
        quad_order=vol_quad_order,
        self_extra_kwargs=self_extra_kwargs,
        kernel_extra_kwargs=yukawa_extra_kernel_kwargs,
    )

    # }}} End sumpy kernel expansion (yukawa)

    # {{{ preprocessing

    from volumential.volume_fmm import interpolate_volume_potential

    def pot_to_bdry(volume_pot):
        return interpolate_volume_potential(bdry_nodes, trav, wrangler, volume_pot).real

    bdry_normal_x = bdry_normals[0].get()
    bdry_normal_y = bdry_normals[1].get()

    line_search_interval = (delta_x * 5, -delta_x * 5)
    line_search_sample_size = 50
    n_bdry_nodes = len(bdry_nodes_x)

    target_parameters = np.linspace(
        line_search_interval[0], line_search_interval[1], line_search_sample_size
    )
    fattened_bdry_nodes_x = np.concatenate(
        [bdry_nodes_x + tgt_s * bdry_normal_x for tgt_s in target_parameters]
    )
    fattened_bdry_nodes_y = np.concatenate(
        [bdry_nodes_y + tgt_s * bdry_normal_y for tgt_s in target_parameters]
    )
    fattened_bdry_nodes = make_obj_array(
        [
            cl.array.to_device(queue, fattened_bdry_nodes_x),
            cl.array.to_device(queue, fattened_bdry_nodes_y),
        ]
    )

    def grad_pot_to_bdry_normal(volume_pot_x, volume_pot_y, search_for=None):
        # Sample along the normal direction for the max and min values nearby,
        # then return the extreme value attained closer to the interior.

        fattened_bdry_pot_x = interpolate_volume_potential(
            fattened_bdry_nodes, trav, wrangler, volume_pot_x
        ).real.get()
        fattened_bdry_pot_y = interpolate_volume_potential(
            fattened_bdry_nodes, trav, wrangler, volume_pot_y
        ).real.get()
        assert len(fattened_bdry_pot_x) == n_bdry_nodes * line_search_sample_size
        assert len(fattened_bdry_pot_y) == n_bdry_nodes * line_search_sample_size

        fattened_bdry_pot_normal = (
            np.concatenate([bdry_normal_x for lid in range(line_search_sample_size)])
            * fattened_bdry_pot_x
            + np.concatenate([bdry_normal_y for lid in range(line_search_sample_size)])
            * fattened_bdry_pot_y
        )

        max_vals = -np.ones(n_bdry_nodes) * np.inf
        max_idxs = np.ones(n_bdry_nodes) * np.nan
        min_vals = np.ones(n_bdry_nodes) * np.inf
        min_idxs = np.ones(n_bdry_nodes) * np.nan

        for lid in range(line_search_sample_size):
            vals_curr = fattened_bdry_pot_normal[
                lid * n_bdry_nodes : (lid + 1) * n_bdry_nodes
            ]

            mask_larger = vals_curr > max_vals
            max_vals[mask_larger] = vals_curr[mask_larger]
            max_idxs[mask_larger] = lid

            mask_smaller = vals_curr < min_vals
            min_vals[mask_smaller] = vals_curr[mask_smaller]
            min_idxs[mask_smaller] = lid

        assert np.all(min_vals <= max_vals)

        if search_for is None:

            mask_interior_larger = (
                (max_idxs >= min_idxs)
                & (max_idxs != line_search_sample_size - 1)
                & (max_idxs != 0)
            ) | ((min_idxs == 0) | (min_idxs == line_search_sample_size - 1))

            mask_interior_smaller = (
                (min_idxs >= max_idxs)
                & (min_idxs != line_search_sample_size - 1)
                & (min_idxs != 0)
            ) | ((max_idxs == 0) | (max_idxs == line_search_sample_size - 1))

            print(
                len(np.nonzero(np.any(mask_interior_larger & mask_interior_smaller))[0])
            )
            assert np.all(mask_interior_larger | mask_interior_smaller)

            interior_vals = np.ones(n_bdry_nodes) * np.nan
            interior_vals[mask_interior_larger] = max_vals[mask_interior_larger]
            interior_vals[mask_interior_smaller] = min_vals[mask_interior_smaller]

            # filter out noisy modes
            freq_tol = 1e-6
            ivhat = np.fft.fft(interior_vals)
            mmg = np.max(np.abs(ivhat.real))
            ivhat[np.abs(ivhat.real) < mmg * freq_tol] = 0
            interior_vals = np.fft.ifft(ivhat).real

            return cl.array.to_device(queue, interior_vals.copy())

        elif search_for == "max":

            return cl.array.to_device(queue, max_vals.copy())

        elif search_for == "min":

            return cl.array.to_device(queue, min_vals.copy())

        else:

            raise NotImplementedError

    # }}} End preprocessing

    # for internal debugging use
    force_direct_evaluation = False

    print("time marching")

    # {{{ time marching

    for i_iter in range(n_iters):

        print("**********************************************")
        print("* Time step", i_iter, "at t =", str(delta_t * i_iter))
        print("**********************************************")

        # {{{ volume potentials

        volume_potential_method = "variational"
        # volume_potential_method = "direct"

        if volume_potential_method == "direct":

            from warnings import warn

            warn("Using direct volume potential evaluation (unstable)")

            if i_iter == 0:
                f1_vals = cl.array.to_device(
                    queue, np.array([f1_func(qp) for qp in q_points_org])
                )

                f2_vals = cl.array.to_device(
                    queue, np.array([f2_func(qp) for qp in q_points_org])
                )
            else:
                import math

                f1_vals = pmbl.evaluate(
                    f1_expr,
                    {
                        "phi": new_phi,
                        "phi_x": new_phi_x,
                        "phi_y": new_phi_y,
                        "lap_phi": new_lap_phi,
                        "math": cl.clmath,
                    },
                )
                f2_vals = pmbl.evaluate(
                    f2_expr,
                    {
                        "phi": new_phi,
                        "phi_x": new_phi_x,
                        "lap_phi": new_lap_phi,
                        "math": cl.clmath,
                    },
                )

            print("Updated source density.")

            pot = drive_volume_fmm(
                trav,
                wrangler,
                f1_vals * q_weights,
                f1_vals,
                direct_evaluation=force_direct_evaluation,
            )
            # pot = np.real(pot)

            pot_ykw = drive_volume_fmm(
                trav,
                wrangler_ykw,
                f1_vals * q_weights,
                f1_vals,
                direct_evaluation=force_direct_evaluation,
            )

            u_tild_pot = pot[0].real
            u_tild_x_pot = pot[1].real
            u_tild_y_pot = pot[2].real

            # pot2 = pot[3]
            # pot2_x = pot[4]
            # pot2_y = pot[5]

            pot2 = pot_ykw[0].real + chop.lambdas[1] ** 2 * pot[0].real
            pot2_x = pot_ykw[1].real + chop.lambdas[1] ** 2 * pot[1].real
            pot2_y = pot_ykw[2].real + chop.lambdas[1] ** 2 * pot[2].real

            # i.e., f2_vals - pot_ykw[0] + chop.lambdas[0]**2 * pot[0]
            v_tild_pot = f2_vals - pot2 + b * u_tild_pot

        elif volume_potential_method == "variational":

            if i_iter == 0:
                # This can be slow, so we load the save data if exists.
                import os.path

                if 0 and os.path.isfile(init_filename):
                    print(
                        "Loading cached initials in",
                        init_filename,
                        "(make sure it is the desired one)",
                    )
                    init_cache = np.load(init_filename)
                    old_phi = cl.array.to_device(queue, init_cache["phi"])
                    old_phi_x = cl.array.to_device(queue, init_cache["phi_x"])
                    old_phi_y = cl.array.to_device(queue, init_cache["phi_y"])
                else:
                    print("Evaluating initials")
                    old_phi = cl.array.to_device(
                        queue, initial_phi(queue, q_points_org2)
                    )
                    old_phi_x = cl.array.to_device(
                        queue, initial_phi_x(queue, q_points_org2)
                    )
                    old_phi_y = cl.array.to_device(
                        queue, initial_phi_y(queue, q_points_org2)
                    )
                    np.savez(
                        init_filename,
                        phi=old_phi.get(),
                        phi_x=old_phi_x.get(),
                        phi_y=old_phi_y.get(),
                    )
                    print("Initials saved in", init_filename)
            else:
                # new_phi from the previous step
                old_phi = new_phi.real
                old_phi_x = new_phi_x.real
                old_phi_y = new_phi_y.real

            # f0 = c * phi
            f0_vals = pmbl.evaluate(f0_expr, {"phi": old_phi})

            # f1 = f0 + laplace(f2)
            # not explicitly computed

            # f2 = (1/epsilon**2) * (phi**3 - (1+s)*phi)
            f2_vals = pmbl.evaluate(f2_expr, {"phi": old_phi})

            # used for post-processing volume potentials
            f2_x_vals = pmbl.evaluate(
                f2_x_expr, {"phi": old_phi, "phi_x": old_phi_x, "phi_y": old_phi_y}
            )
            f2_y_vals = pmbl.evaluate(
                f2_y_expr, {"phi": old_phi, "phi_x": old_phi_x, "phi_y": old_phi_y}
            )

            print("Updated source density.")

            print("Volume potentials 1/4")
            pot_v0f0 = drive_volume_fmm(
                trav,
                wrangler,
                f0_vals * q_weights,
                f0_vals,
                direct_evaluation=force_direct_evaluation,
            )

            print("Volume potentials 2/4")
            pot_v0f2 = drive_volume_fmm(
                trav,
                wrangler,
                f2_vals * q_weights,
                f2_vals,
                direct_evaluation=force_direct_evaluation,
            )

            print("Volume potentials 3/4")
            pot_v1f0 = drive_volume_fmm(
                trav,
                wrangler_ykw,
                f0_vals * q_weights,
                f0_vals,
                direct_evaluation=force_direct_evaluation,
            )

            print("Volume potentials 4/4")
            pot_v1f2 = drive_volume_fmm(
                trav,
                wrangler_ykw,
                f2_vals * q_weights,
                f2_vals,
                direct_evaluation=force_direct_evaluation,
            )

            print("Processing volume potentials")
            # u_tild = V0[f1], which is replaced with
            #          V0[f0] + V1[f2] + lam2**2 * V0[f2]
            u_tild_pot, u_tild_x_pot, u_tild_y_pot = (
                pot_v0f0[i].real
                + pot_v1f2[i].real
                + chop.lambdas[1] ** 2 * pot_v0f2[i].real
                for i in range(3)
            )

            # v_tild = -V1[f1] + lam1**2 * V0[f1]
            # V1[f1] is replaced with V1[f0] + lam1**2 * V1[f2] + f2
            v_tild_pot = (
                -(pot_v1f0[0].real + chop.lambdas[0] ** 2 * pot_v1f2[0].real + f2_vals)
                + chop.lambdas[0] ** 2 * u_tild_pot
            )
            v_tild_x_pot = (
                -(
                    pot_v1f0[1].real
                    + chop.lambdas[0] ** 2 * pot_v1f2[1].real
                    + f2_x_vals
                )
                + chop.lambdas[0] ** 2 * u_tild_x_pot
            )
            v_tild_y_pot = (
                -(
                    pot_v1f0[2].real
                    + chop.lambdas[0] ** 2 * pot_v1f2[2].real
                    + f2_y_vals
                )
                + chop.lambdas[0] ** 2 * u_tild_y_pot
            )

        else:

            raise NotImplementedError

        # }}} End volume potentials (u and Lap_u)

        # {{{ output initials

        if i_iter == 0 and 1:
            # output on the unstructured mesh
            from meshmode.discretization.visualization import make_visualizer

            vis = make_visualizer(queue, vol_discr, visual_order)

            nodes_x = vol_discr.nodes()[0].with_queue(queue).get()
            nodes_y = vol_discr.nodes()[1].with_queue(queue).get()
            nodes = make_obj_array(
                [cl.array.to_device(queue, nodes_x), cl.array.to_device(queue, nodes_y)]
            )

            phi = interpolate_volume_potential(nodes, trav, wrangler, old_phi).real
            phi_x = interpolate_volume_potential(nodes, trav, wrangler, old_phi_x).real
            phi_y = interpolate_volume_potential(nodes, trav, wrangler, old_phi_y).real

            if volume_potential_method == "direct":
                f1 = interpolate_volume_potential(nodes, trav, wrangler, f1_vals).real

            f2 = interpolate_volume_potential(nodes, trav, wrangler, f2_vals).real

            vtu_filename = "ch-solution-" + "{0:03}".format(i_iter) + ".vtu"
            clean_file(vtu_filename)

            vis.write_vtk_file(
                vtu_filename,
                [
                    ("x", vol_discr.nodes()[0]),
                    ("y", vol_discr.nodes()[1]),
                    # ("f1", f1),
                    ("f2", f2),
                    ("phi", phi),
                    ("phi_x", phi_x),
                    ("phi_y", phi_y),
                ],
            )

            print("Written file " + vtu_filename)

        if i_iter == 0 and 1:
            # structured immersed mesh
            from sumpy.visualization import FieldPlotter

            fplot = FieldPlotter(np.zeros(2), extent=box_b - box_a, npoints=500)
            targets = cl.array.to_device(queue, fplot.points)
            immersed_nodes = make_obj_array(
                [cl.array.to_device(queue, fplot.points[i]) for i in range(dim)]
            )

            imm_phi = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, old_phi
            ).real.get()
            imm_phi_x = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, old_phi_x
            ).real.get()
            imm_phi_y = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, old_phi_y
            ).real.get()

            vtu_filename = "ch-solution-immersed-" + "{0:03}".format(i_iter) + ".vts"
            clean_file(vtu_filename)

            fplot.write_vtk_file(
                vtu_filename,
                [("phi", imm_phi), ("phi_x", imm_phi_x), ("phi_y", imm_phi_y)],
            )
            print("Written file " + vtu_filename)

        # }}} End output initials

        # {{{ volume potential check

        def check_pde_volume():

            from sumpy.point_calculus import CalculusPatch

            vec_h = [0.25 * 2 ** (-i) for i in range(7)]
            # vec_h = [1e-1]

            vec_ru = []
            vec_rux = []
            vec_ruy = []
            vec_rlap_u = []
            vec_rlap_ux = []
            vec_rlap_uy = []

            vec_rykw = []
            vec_rykw_x = []
            vec_rykw_y = []

            for dx in vec_h:
                cp = CalculusPatch(np.zeros(2), order=4, h=dx)
                cp_targets = make_obj_array(
                    [
                        cl.array.to_device(queue, cp.points[0]),
                        cl.array.to_device(queue, cp.points[1]),
                    ]
                )

                u = interpolate_volume_potential(
                    cp_targets, trav, wrangler, u_tild_pot
                ).real
                ux = interpolate_volume_potential(
                    cp_targets, trav, wrangler, u_tild_x_pot
                ).real
                uy = interpolate_volume_potential(
                    cp_targets, trav, wrangler, u_tild_y_pot
                ).real

                lap_u = interpolate_volume_potential(
                    cp_targets, trav, wrangler, pot2
                ).real
                lap_ux = interpolate_volume_potential(
                    cp_targets, trav, wrangler, pot2_x
                ).real
                lap_uy = interpolate_volume_potential(
                    cp_targets, trav, wrangler, pot2_y
                ).real

                ykw = interpolate_volume_potential(
                    cp_targets, trav, wrangler_ykw, pot_ykw[0]
                ).real.get()
                ykw_x = interpolate_volume_potential(
                    cp_targets, trav, wrangler_ykw, pot_ykw[1]
                ).real.get()
                ykw_y = interpolate_volume_potential(
                    cp_targets, trav, wrangler_ykw, pot_ykw[2]
                ).real.get()

                u = u.get()
                ux = ux.get()
                uy = uy.get()
                lap_u = lap_u.get()
                lap_ux = lap_ux.get()
                lap_uy = lap_uy.get()

                rhs_f1 = interpolate_volume_potential(
                    cp_targets, trav, wrangler, f1_vals
                ).real.get()
                rhs_f2 = interpolate_volume_potential(
                    cp_targets, trav, wrangler, f2_vals
                ).real.get()

                vec_ru.append(
                    la.norm(
                        cp.laplace(lap_u) - chop.b * cp.laplace(u) + chop.c * u - rhs_f1
                    )
                    / la.norm(rhs_f1)
                )

                v = -cp.laplace(u) + chop.b * u + rhs_f2

                vec_rux.append(la.norm(cp.dx(u) - ux) / la.norm(ux))
                vec_ruy.append(la.norm(cp.dy(u) - uy) / la.norm(uy))

                vec_rlap_u.append(la.norm(cp.laplace(u) - lap_u) / la.norm(lap_u))
                vec_rlap_ux.append(
                    la.norm(cp.dx(cp.laplace(u)) - lap_ux) / la.norm(lap_ux)
                )
                vec_rlap_uy.append(
                    la.norm(cp.dy(cp.laplace(u)) - lap_uy) / la.norm(lap_uy)
                )

                vec_rykw.append(
                    la.norm(cp.laplace(ykw) - chop.lambdas[0] ** 2 * ykw - rhs_f1)
                    / la.norm(rhs_f1)
                )

                vec_rykw_x.append(la.norm(cp.dx(ykw) - ykw_x) / la.norm(ykw_x))
                vec_rykw_y.append(la.norm(cp.dy(ykw) - ykw_y) / la.norm(ykw_y))

            from tabulate import tabulate

            # overwrite if file exists
            with open("check_pde_volume.dat", "w") as f:
                print(
                    "Residuals of PDE and numerical vs. symbolic differentiation:",
                    file=f,
                )
                print(
                    tabulate(
                        [
                            ["h"] + vec_h,
                            ["residual_u"] + vec_ru,
                            ["residual_ux"] + vec_rux,
                            ["residual_uy"] + vec_ruy,
                            ["residual_lap_u"] + vec_rlap_u,
                            ["residual_lap_ux"] + vec_rlap_ux,
                            ["residual_lap_uy"] + vec_rlap_uy,
                            ["residual_ykw"] + vec_rykw,
                            ["residual_ykw_x"] + vec_rykw_x,
                            ["residual_ykw_y"] + vec_rykw_y,
                        ]
                    ),
                    file=f,
                )

        # }}} End volume potential

        # {{{ prepare rhs bor BIE

        print("Preparing BC")

        # FIXME: cache the area queries for pot_to_bdry.

        bdry_u_tild = pot_to_bdry(u_tild_pot)

        # FIXME: higher order extrapolation by expansion?
        # bdry_u_tild_x = pot_to_bdry(u_tild_x_pot)
        # bdry_u_tild_y = pot_to_bdry(u_tild_y_pot)
        bdry_u_tild_x = pot_to_bdry(u_tild_x_pot)
        bdry_u_tild_y = pot_to_bdry(u_tild_y_pot)
        # bdry_u_tild_normal = grad_to_normal(bdry_u_tild_x, bdry_u_tild_y)
        bdry_u_tild_normal = grad_pot_to_bdry_normal(u_tild_x_pot, u_tild_y_pot)

        bdry_v_tild = pot_to_bdry(v_tild_pot)

        if i_iter == 0:
            bdry_b1_vals = cl.array.to_device(
                queue, b1_func(queue, np.array([bdry_nodes_x, bdry_nodes_y]))
            )
        else:
            bdry_b1_vals = pmbl.evaluate(
                b1_expr, {"phi": bdry_new_phi, "math": cl.clmath}
            )

        if volume_potential_method == "direct":
            bdry_pot2 = pot_to_bdry(pot2)
            # pot2 is short for laplace(u_tild)
            bdry_pot2_x = pot_to_bdry(pot2_x)
            bdry_pot2_y = pot_to_bdry(pot2_y)
            if i_iter == 0:
                bdry_f2_x_vals = cl.array.to_device(
                    queue,
                    np.array(
                        [
                            f2_x_func([x0, x1])
                            for (x0, x1) in zip(bdry_nodes_x, bdry_nodes_y)
                        ]
                    ),
                )
                bdry_f2_y_vals = cl.array.to_device(
                    queue,
                    np.array(
                        [
                            f2_y_func([x0, x1])
                            for (x0, x1) in zip(bdry_nodes_x, bdry_nodes_y)
                        ]
                    ),
                )
            else:
                bdry_f2_x_vals = pmbl.evaluate(
                    f2_x_expr,
                    {
                        "phi": bdry_new_phi,
                        "phi_x": bdry_new_phi_x,
                        "phi_y": bdry_new_phi_y,
                        "lap_phi": bdry_new_lap_phi,
                        "math": cl.clmath,
                    },
                )
                bdry_f2_y_vals = pmbl.evaluate(
                    f2_y_expr,
                    {
                        "phi": bdry_new_phi,
                        "phi_x": bdry_new_phi_x,
                        "phi_y": bdry_new_phi_y,
                        "lap_phi": bdry_new_lap_phi,
                        "math": cl.clmath,
                    },
                )
            bdry_f2_normal = grad_to_normal(bdry_f2_x_vals, bdry_f2_y_vals)
            bdry_pot2_normal = grad_to_normal(bdry_pot2_x, bdry_pot2_y)
            bdry_v_tild_normal = (
                bdry_f2_normal - bdry_pot2_normal + b * bdry_u_tild_normal
            )

        elif volume_potential_method == "variational":
            # FIXME: higher order extrapolation?
            # bdry_v_tild_x = pot_to_bdry(v_tild_x_pot)
            # bdry_v_tild_y = pot_to_bdry(v_tild_y_pot)
            bdry_v_tild_x = pot_to_bdry(v_tild_x_pot)
            bdry_v_tild_y = pot_to_bdry(v_tild_y_pot)
            # bdry_v_tild_normal = grad_to_normal(bdry_v_tild_x, bdry_v_tild_y)

            bdry_v_tild_normal = grad_pot_to_bdry_normal(
                v_tild_x_pot, v_tild_y_pot, "min"
            )
            np.savez(
                "bdry_data" + "{0:03}".format(i_iter + 1) + ".npz",
                v_tild_normal=bdry_v_tild_normal.get(),
                u_tild_normal=bdry_u_tild_normal.get(),
            )

        else:
            raise NotImplementedError

        # FIXME: u_tild_normal has a jump at the boundary, as a result, the interpolation
        # does not pick up the interior side very well. (int < ext)
        # This can result in inaccurate contact angle.

        # sample in the normal direction for 2*sqrt(2)*delta_x inwards for find the extrema
        # (1st order)

        # QBX-like expansion from interior?

        """
        # Use this to show quantitative changes in contact angles
        min_bdry_u_tild_normal = np.min(bdry_u_tild_normal.get())
        const_bdry_u_tild_normal = cl.array.to_device(queue,
                min_bdry_u_tild_normal * np.ones(len(bdry_nodes_x)))

        bdry_g1_vals = bdry_b1_vals - const_bdry_u_tild_normal - (c - bs) * bdry_u_tild
        """

        bdry_g1_vals = bdry_b1_vals - bdry_u_tild_normal - (c - bs) * bdry_u_tild

        # FIXME: v_tild_normal has a jump at the boundary, as a result, the interpolation
        # does not pick up the interior side very well. This causes stability/conservation
        # issues. (int > ext)

        """
        # Use this boosted g2 to show a reversed volume change
        max_bdry_v_tild_normal = np.max(bdry_v_tild_normal.get())
        const_bdry_v_tild_normal = cl.array.to_device(queue,
                max_bdry_v_tild_normal * np.ones(len(bdry_nodes_x)))
        bdry_g2_vals = - second_bie_rscale * 10.0 * const_bdry_v_tild_normal
        """

        # A factor to stabilize the BIE (even with inaccurate/oscillatory g1 and g2)
        # du/dnormal ~ d(phi)/dnormal ~ O(1/epsilon)
        # mu ~ epsilon * laplace(phi) ~ O(1/epsilon)
        # dv/dnormal ~ d(mu/epsilon)/dnormal ~ O(1/epsilon**3)
        # Rescale the second BIE with epsilon*2 to get them to the same scale
        # which affects how GEMRES understands the residual.
        second_bie_rscale = epsilon ** 2

        # FIXME: conpensate for over/under estimates to satisfy mass conservation
        # The line search is 1st order. Setting a good overshoot_factor can give
        # a better estimate
        overshoot_factor = (256 * delta_x) * 10 + 1
        bdry_g2_vals = -second_bie_rscale * overshoot_factor * bdry_v_tild_normal

        # }}} End prepare rhs bor BIE

        # {{{ BIE solve

        from pytential.symbolic.pde.cahn_hilliard import CahnHilliardOperator

        chop = CahnHilliardOperator(b=b, c=c)

        unk = chop.make_unknown("sigma")
        bound_op = bind(
            qbx,
            chop.operator(
                unk, rscale=(1, second_bie_rscale), bdry_splitting_parameter=0
            ),
        )

        bc = sym.make_obj_array([bdry_g1_vals, bdry_g2_vals])

        if 0 and i_iter == 0:
            print("reconstructing matrix")
            # construct the coef matrix
            mop = bound_op.scipy_op(queue, "sigma", dtype=np.complex128)
            mm, mn = mop.shape
            assert mm == mn
            mcoef = np.zeros(mop.shape, dtype=np.complex128)

            for i in range(mm):
                print(i, "/", mm)
                ei = np.zeros(mm)
                ei[i] = 1
                coli = mop.matvec(ei)
                mcoef[:, i] = coli

            np.save("coef_matrix.npy", mcoef)

        if 0:
            bc1, bc2 = [bcg.get() for bcg in bc]
            rhs = np.array([bc1, bc2])
            np.save("rhs-" + "{0:03}".format(i_iter) + ".npy", rhs)

        # For later time steps, use previous sigma as initial guess
        if i_iter == 0:
            guess = None
        elif 0:
            guess = cl.array.to_device(
                queue, np.concatenate([sig.get() for sig in sigma])
            )
        else:
            guess = None

        from pytential.solve import gmres

        gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.complex128),
            bc,
            x0=guess,
            tol=1e-12,
            # tol=1e-6,
            progress=True,
            stall_iterations=0,
            hard_failure=True,
        )
        assert gmres_result.success
        sigma = gmres_result.solution

        # }}} End BIE solve

        # {{{ layer potential check

        def check_pde_layer():
            from sumpy.point_calculus import CalculusPatch

            vec_h = [0.25 * 2 ** (-i) for i in range(7)]
            vec_ru = []
            vec_rv = []
            vec_rux = []
            vec_ruy = []
            vec_rlap1_u = []
            vec_rlap2_u = []
            for dx in vec_h:
                cp = CalculusPatch(np.zeros(2), order=4, h=dx)
                cp_targets = cl.array.to_device(queue, cp.points)

                u, v, ux, uy, lap_u = bind(
                    (qbx_stick_out, PointsTarget(cp_targets)), chop.representation(unk)
                )(queue, sigma=sigma)

                u = u.get().real
                v = v.get().real
                ux = ux.get().real
                uy = uy.get().real
                lap_u = lap_u.get().real

                lap_u2 = -(v - chop.b * u)

                # Check for homogeneous PDEs for u and v
                vec_ru.append(
                    la.norm(cp.laplace(lap_u) - chop.b * cp.laplace(u) + chop.c * u)
                )

                vec_rv.append(la.norm(v + cp.laplace(u) - chop.b * u))

                vec_rux.append(la.norm(cp.dx(u) - ux))
                vec_ruy.append(la.norm(cp.dy(u) - uy))

                vec_rlap1_u.append(la.norm(lap_u - lap_u2))
                vec_rlap2_u.append(la.norm(lap_u - cp.laplace(u)))

            from tabulate import tabulate

            # overwrite if file exists
            with open("check_pde_layer.dat", "w") as f:
                print(
                    "Residuals of PDE and numerical vs. symbolic differentiation:",
                    file=f,
                )
                print(
                    tabulate(
                        [
                            ["h"] + vec_h,
                            ["residual_u"] + vec_ru,
                            ["residual_v"] + vec_rv,
                            ["residual_ux"] + vec_rux,
                            ["residual_uy"] + vec_ruy,
                            ["residual_lap_u1"] + vec_rlap1_u,
                            ["residual_lap_u2"] + vec_rlap2_u,
                        ]
                    ),
                    file=f,
                )

        # }}} End layer potential

        # {{{ put pieces together (bootstrap for the next step)

        print("assembling the solution")

        # qbx_stick_out = qbx.copy(target_association_tolerance=2)
        qbx_stick_out = qbx.copy(target_association_tolerance=0.05)

        # use a double layer potential to indicate the domain
        from sumpy.kernel import LaplaceKernel

        domain_mask = bind(
            (qbx_stick_out, box_discr),
            sym.D(LaplaceKernel(dim), pmbl.var("one_density"), qbx_forced_limit=-2),
        )(queue, one_density=cl.array.to_device(np.ones(len(sigma[0]))))

        bvp_sol = bind(
            (qbx_stick_out, box_discr),
            # chop.representation(unk, qbx_forced_limit=-2))(queue, sigma=sigma)
            chop.representation(unk),
        )(queue, sigma=sigma)

        # interior side
        bdry_bvp_sol = bind(
            (qbx_stick_out, density_discr),
            chop.representation(unk, qbx_forced_limit=-1),
        )(queue, sigma=sigma)

        # representaion:
        # u, v, u_x, u_y, lap(u)

        new_phi = u_tild_pot + bvp_sol[0].real
        new_phi_x = u_tild_x_pot + bvp_sol[2].real
        new_phi_y = u_tild_y_pot + bvp_sol[3].real

        def fill_exterior(vec, val):
            vec = (1 - domain_mask) * vec + domain_mask * val
            return vec

        new_phi = fill_exterior(new_phi, 1)
        new_phi_x = fill_exterior(new_phi_x, 0)
        new_phi_y = fill_exterior(new_phi_y, 0)

        new_mu = epsilon * (v_tild_pot + bvp_sol[1].real)
        if volume_potential_method == "direct":
            new_lap_phi = pot2 + bvp_sol[4].real

        """
        # Add interior part of layer potentials
        new_phi = u_tild_pot + bvp_sol[0] * q_point_masks
        new_mu = epsilon * (v_tild_pot + bvp_sol[1] * q_point_masks)
        new_phi_x = u_tild_x_pot + bvp_sol[2] * q_point_masks
        new_phi_y = u_tild_y_pot + bvp_sol[3] * q_point_masks
        new_lap_phi = pot2 + bvp_sol[4] * q_point_masks
        """

        # For exterior part of layer potentials, do a continuous extension
        # (some layer potentials have jumps if used directly)

        bdry_new_phi = bdry_u_tild + bdry_bvp_sol[0].real
        bdry_new_mu = epsilon * (bdry_v_tild + bdry_bvp_sol[1].real)
        bdry_new_phi_x = bdry_u_tild_x + bdry_bvp_sol[2].real
        bdry_new_phi_y = bdry_u_tild_y + bdry_bvp_sol[3].real

        if volume_potential_method == "direct":
            bdry_new_lap_phi = bdry_pot2 + bdry_bvp_sol[4].real

        # }}} End put pieces together

        # {{{ check pde

        pde_consistency_check = False

        if pde_consistency_check:
            print("checking PDE consistenccy of layer potentials")
            check_pde_layer()

            print("checking PDE consistenccy of volume potentials")
            check_pde_volume()

        # }}}

        # {{{ postprocess
        print("postprocessing")

        if 1:
            # usual output on the unstructured mesh
            from meshmode.discretization.visualization import make_visualizer

            vis = make_visualizer(queue, vol_discr, visual_order)

            nodes_x = vol_discr.nodes()[0].with_queue(queue).get()
            nodes_y = vol_discr.nodes()[1].with_queue(queue).get()
            nodes = make_obj_array(
                [cl.array.to_device(queue, nodes_x), cl.array.to_device(queue, nodes_y)]
            )

            phi = interpolate_volume_potential(nodes, trav, wrangler, new_phi).real
            mu = interpolate_volume_potential(nodes, trav, wrangler, new_mu).real
            if volume_potential_method == "direct":
                f1 = interpolate_volume_potential(nodes, trav, wrangler, f1_vals).real
            f2 = interpolate_volume_potential(nodes, trav, wrangler, f2_vals).real

            vtu_filename = "ch-solution-" + "{0:03}".format(i_iter + 1) + ".vtu"
            clean_file(vtu_filename)

            vis.write_vtk_file(
                vtu_filename,
                [
                    ("x", vol_discr.nodes()[0]),
                    ("y", vol_discr.nodes()[1]),
                    # ("f1", f1),
                    ("f2", f2),
                    ("phi", phi),
                    ("mu", mu),
                ],
            )

            print("Written file " + vtu_filename)

        if 0:
            # layer potentials (to see the jump conditions)
            from meshmode.mesh.io import read_gmsh

            modemesh = read_gmsh("box_grid.msh", force_ambient_dim=None)
            from meshmode.discretization.poly_element import (
                LegendreGaussLobattoTensorProductGroupFactory,
            )

            immersed_discr = Discretization(
                cl_ctx,
                modemesh,
                LegendreGaussLobattoTensorProductGroupFactory(vol_quad_order),
            )

            one_sigma = make_obj_array(
                [cl.array.to_device(queue, np.ones(len(sig))) for sig in sigma]
            )

            print(one_sigma)

            layer_potentials = bind(
                (qbx_stick_out, immersed_discr), chop.debug_representation(unk)
            )(queue, sigma=one_sigma)
            # debug_rep_names = ["S0", "S0_n", "S1", "S1_n", "S2", "S2_n"]
            debug_rep_names = ["S0", "S1", "S2"]

            vtu_filename = "ch-layer-potential.vtu"
            clean_file(vtu_filename)

            from meshmode.discretization.visualization import make_visualizer

            vis = make_visualizer(queue, immersed_discr, visual_order)

            vis.write_vtk_file(
                vtu_filename,
                [("x", immersed_discr.nodes()[0]), ("y", immersed_discr.nodes()[1])]
                + [
                    (debug_rep_names[i], layer_potentials[i])
                    for i in range(len(layer_potentials))
                ],
            )
            print("Written file " + vtu_filename)

        if 1 and i_iter % output_interval == 0:
            # immersed visualization (whole box)
            from sumpy.visualization import FieldPlotter

            fplot = FieldPlotter(np.zeros(2), extent=box_b - box_a, npoints=500)
            targets = cl.array.to_device(queue, fplot.points)
            immersed_nodes = make_obj_array(
                [cl.array.to_device(queue, fplot.points[i]) for i in range(dim)]
            )

            imm_phi = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, new_phi
            ).real.get()
            imm_mu = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, new_mu
            ).real.get()
            imm_domain_mask = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, domain_mask
            ).real.get()

            imm_u_tild = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, u_tild_pot
            ).real.get()
            imm_v_tild = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, v_tild_pot
            ).real.get()

            imm_u_tild_x = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, u_tild_x_pot
            ).real.get()
            imm_u_tild_y = interpolate_volume_potential(
                immersed_nodes, trav, wrangler, u_tild_y_pot
            ).real.get()

            if volume_potential_method == "variational":
                imm_v_tild_x = interpolate_volume_potential(
                    immersed_nodes, trav, wrangler, v_tild_x_pot
                ).real.get()
                imm_v_tild_y = interpolate_volume_potential(
                    immersed_nodes, trav, wrangler, v_tild_y_pot
                ).real.get()
            else:
                imm_v_tild_x = 0
                imm_v_tild_y = 0

            # imm_f1 = interpolate_volume_potential(immersed_nodes, trav,
            #        wrangler, f1_vals).real
            # imm_f2 = interpolate_volume_potential(immersed_nodes, trav,
            #        wrangler, f2_vals).real

            imm_bvp_solu = bind(
                (qbx_stick_out, PointsTarget(targets)), chop.representation(unk)
            )(queue, sigma=sigma)
            imm_bvp_names = ["u", "v", "u_x", "u_y", "lap_u", "v2", "v_x", "v_y"]

            if volume_potential_method == "direct":
                volume_potentials = [pot, pot_ykw]
                volume_potentials_output_names = [
                    ["ut", "ut_x", "ut_y", "lap_ut", "(lap_ut)_x", "(lap_ut)_y"],
                    ["V1[f1]", "V1[f1]_x", "V1[f1]_y"],
                ]

            elif volume_potential_method == "variational":
                volume_potentials = [pot_v0f0, pot_v0f2, pot_v1f0, pot_v1f2]
                volume_potentials_output_names = [
                    ["V0[f0]", "V0[f0]_x", "V0[f0]_y"],
                    ["V0[f2]", "V0[f2]_x", "V0[f2]_y"],
                    ["V1[f0]", "V1[f0]_x", "V1[f0]_y"],
                    ["V1[f2]", "V1[f2]_x", "V1[f2]_y"],
                ]

            imm_pot = []
            for pp in volume_potentials:
                imm_pot_outs = []
                for pouts in pp:
                    imm_pot_outs.append(
                        interpolate_volume_potential(
                            immersed_nodes, trav, wrangler, pouts
                        ).real
                    )
                imm_pot.append(imm_pot_outs)

            vtu_filename = (
                "ch-solution-immersed-" + "{0:03}".format(i_iter + 1) + ".vts"
            )
            clean_file(vtu_filename)

            fplot.write_vtk_file(
                vtu_filename,
                [
                    # ("f1", imm_f1),
                    # ("f2", imm_f2),
                    ("domain", imm_domain_mask),
                    ("phi", imm_phi),
                    ("mu", imm_mu),
                    ("u_tild", imm_u_tild),
                    ("v_tild", imm_v_tild),
                    ("u_tild_x", imm_u_tild_x),
                    ("v_tild_x", imm_v_tild_x),
                    ("u_tild_y", imm_u_tild_y),
                    ("v_tild_y", imm_v_tild_y),
                ]
                + [
                    ("LP-" + imm_bvp_names[i], imm_bvp_solu[i].real.get())
                    for i in range(len(imm_bvp_solu))
                ]
                + [
                    ("VP-" + name, data.get())
                    for name_list, pot_list in zip(
                        volume_potentials_output_names, imm_pot
                    )
                    for (name, data) in zip(name_list, pot_list)
                ],
            )
            print("Written file " + vtu_filename)

        if 1 and i_iter % output_interval == 0:
            # boundary only
            from meshmode.discretization.visualization import make_visualizer

            bdry_vis = make_visualizer(queue, density_discr, visual_order)

            bdry_normals = bind(density_discr, sym.normal(dim))(queue).as_vector(
                dtype=object
            )

            bdry_bvp_solu = bind(
                (qbx_stick_out, density_discr),
                chop.representation(unk, qbx_forced_limit=-1),
            )(queue, sigma=sigma)
            bdry_bvp_names = ["u", "v", "u_x", "u_y", "lap_u", "v2", "v_x", "v_y"]

            vtu_filename = "ch-bdry-" + "{0:03}".format(i_iter + 1) + ".vtu"
            clean_file(vtu_filename)

            bdry_vis.write_vtk_file(
                vtu_filename,
                [
                    ("sigma[0]", sigma[0].real),
                    ("sigma[1]", sigma[1].real),
                    ("bdry_normals", bdry_normals),
                    ("b1", bdry_b1_vals),
                    ("u_tild_normal", bdry_u_tild_normal),
                    ("v_tild_normal", bdry_v_tild_normal),
                    # ("f2_normal", bdry_f2_normal),
                    # ("lap_u_tild_normal", bdry_pot2_normal),
                    ("u_tild", bdry_u_tild),
                    ("g1", bdry_g1_vals),
                    ("g2", bdry_g2_vals),
                    ("b1", bdry_b1_vals),
                    ("phi", bdry_new_phi),
                    ("phi_x", bdry_new_phi_x),
                    ("phi_y", bdry_new_phi_y),
                    # ("lap_phi", bdry_new_lap_phi),
                ]
                + [
                    ("BVPSolu-" + bdry_bvp_names[i], bdry_bvp_solu[i].real)
                    for i in range(len(bdry_bvp_solu))
                ],
            )
            print("Written file " + vtu_filename)

        # }}} End postprocess

    # }}} End time marching


if __name__ == "__main__":
    main()

# vim: fdm=marker:ft=pyopencl
